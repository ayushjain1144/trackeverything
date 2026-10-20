# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union
import logging
import ipdb

import numpy as np
import torch
from opt_einsum import contract
import torch.nn.functional as F

from ...utils.common_utils import apply_homo_transform

st = ipdb.set_trace
logger = logging.getLogger(__name__)


def apply_log_to_norm(input_data):
    """
    Normalize the input data and apply a logarithmic transformation based on the normalization factor.

    Args:
        input_data (torch.Tensor): The input tensor to be normalized and transformed.

    Returns:
        torch.Tensor: The transformed tensor after normalization and logarithmic scaling.
    """
    org_d = input_data.norm(dim=-1, keepdim=True)
    input_data = input_data / org_d.clip(min=1e-8)
    input_data = input_data * torch.log1p(org_d)
    return input_data


def safe_inverse_np(a):
    inv = a.copy()
    r_transpose = np.swapaxes(a[..., :3, :3], -1, -2)  # inverse of rotation matrix
    inv[..., :3, :3] = r_transpose
    inv[..., :3, 3:4] = -np.matmul(r_transpose, a[..., :3, 3:4])
    return inv


def batch_unproject_np(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    is_distance: bool = False,
) -> np.ndarray:
    assert depth.ndim == 3, "depth must be a 2D array"
    t, h, w = depth.shape
    assert intrinsic.shape == (
        t,
        3,
        3,
    ), f"intrinsic must be a 3D array of shape (t, 3, 3), got {intrinsic.shape}"
    assert extrinsic.shape == (
        t,
        4,
        4,
    ), f"extrinsic must be a 3D array of shape (t, 4, 4), got {extrinsic.shape}"

    # Generate 3D point cloud in world coordinates
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    uv_homogeneous = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)
    K_inv = np.linalg.inv(intrinsic).astype(np.float32)
    camera_coords = contract("nij, xyj -> nxyi", K_inv, uv_homogeneous)
    if is_distance:
        camera_coords = camera_coords / np.sqrt((camera_coords**2).sum(-1))[..., None]
    camera_coords = camera_coords * depth[..., None]
    camera_coords = np.concatenate(
        (camera_coords, np.ones_like(camera_coords[..., :1])), axis=-1
    )

    # inv_extrinsics = np.linalg.inv(extrinsic)
    inv_extrinsics = safe_inverse_np(extrinsic)
    world_coordinates = contract("nij, nxyj -> nxyi", inv_extrinsics, camera_coords)
    return world_coordinates[..., :3]


def batch_distance_to_depth_np(
    distances: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    h, w = distances.shape[-2:]
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    # Keep a single (H, W, 3) UV grid and broadcast over T (and B) — avoid
    # materializing a full (T, H, W, 3) copy via einops.repeat.
    uv_homogeneous = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)
    K_inv = np.linalg.inv(intrinsics)
    if len(distances.shape) == 3:  # t h w
        # K_inv: (T,3,3), uv: (H,W,3) → camera_coords (T,H,W,3)
        camera_coords = np.einsum("tij,hwj->thwi", K_inv, uv_homogeneous)
    elif len(distances.shape) == 4:  # b t h w
        camera_coords = np.einsum("btij,hwj->bthwi", K_inv, uv_homogeneous)
    else:
        raise ValueError(
            f"distances must be a 3D or 4D tensor, got {len(distances.shape)}D tensor"
        )

    depth = camera_coords[..., -1] * distances / np.sqrt((camera_coords**2).sum(-1))

    return depth


def quaternion_to_rotation_matrix(
    q: Tuple[float, float, float, float],
):
    q_x, q_y, q_z, q_w = q
    R = np.array(
        [
            [
                1 - 2 * (q_y**2 + q_z**2),
                2 * (q_x * q_y - q_z * q_w),
                2 * (q_x * q_z + q_y * q_w),
            ],
            [
                2 * (q_x * q_y + q_z * q_w),
                1 - 2 * (q_x**2 + q_z**2),
                2 * (q_y * q_z - q_x * q_w),
            ],
            [
                2 * (q_x * q_z - q_y * q_w),
                2 * (q_y * q_z + q_x * q_w),
                1 - 2 * (q_x**2 + q_y**2),
            ],
        ]
    )
    return R


def get_transformation_matrix(
    translation,
    quaternion,
):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quaternion_to_rotation_matrix(quaternion)
    T[:3, 3] = translation
    return T


# borrowed from Adam's code
def opencv_from_cameras_projection(focal, p0, image_size):
    # Retype the image_size correctly and flip to width, height.
    image_size_wh = np.flip(image_size, 0)

    # NDC to screen conversion.
    scale = min(image_size_wh)
    scale = np.array([scale, scale])
    c0 = image_size_wh

    principal_point = -p0 * scale + c0
    focal_length = focal * scale

    camera_matrix = np.array(
        [
            [focal_length[0], 0.0, principal_point[0], 0.0],
            [0.0, focal_length[1], principal_point[1], 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    return camera_matrix


def project(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    cam_T_world: torch.Tensor,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Project 3D points to 2D pixel coordinates
    One should not assume this transform is invertible.

    Args:
        points: (B, ....,  3),
        intrinsics: (B, ...,  3, 3)
        extrinsics: (B, ..., 4, 4)
    Returns:
        coords_2d: (B, N, T, 3), the last dimension is (x, y, d). Note that x corresponds to width.
    """

    coords_local = apply_homo_transform(points, cam_T_world)
    coords_pixel = torch.einsum("...ij,...j->...i", intrinsics, coords_local)
    coords_pixel = coords_pixel[..., :2] / torch.clamp(coords_pixel[..., 2:3], min=1e-6)

    clamped_x = torch.clamp(
        coords_pixel[..., 0],
        min=-image_size[1] * 2,
        max=image_size[1] * 2,
    )
    clamped_y = torch.clamp(
        coords_pixel[..., 1],
        min=-image_size[0] * 2,
        max=image_size[0] * 2,
    )

    result = torch.cat(
        [clamped_x[..., None], clamped_y[..., None], coords_local[..., 2:]], dim=-1
    )
    return result


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert len(arr.shape) == 3
    arr -= arr.min(1, keepdims=True)[0]
    arr_max = arr.max(1, keepdims=True)[0] + 1

    keys = torch.zeros(arr.shape[0], arr.shape[1], dtype=torch.long, device=arr.device)

    # Fortran style indexing
    for j in range(arr.shape[2] - 1):
        keys += arr[..., j]
        keys *= arr_max[..., j + 1]
    keys += arr[..., -1]
    return keys


# Per-axis knee of the signed-log grid: bins are near-uniform for |x| << a and
# logarithmic beyond it.
_SIGNEDLOG_AXIS_A = 0.25


def _signedlog_bins(xyz: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """Integer per-axis signed-log bins, calibrated so the cell at ``|x| = 1``
    still has edge ``voxel_size``: cells are finer near the origin (the first
    camera) and coarser far away."""
    a = _SIGNEDLOG_AXIS_A
    scale = (a + 1.0) / float(voxel_size)
    return torch.floor(torch.sign(xyz) * torch.log1p(xyz.abs() / a) * scale).long()


def voxelization(xyz, voxel_size, mode="uniform"):
    """
    Inputs:
        xyz: tensor [B, N, 3]
        voxel_size: float
        mode: ``uniform`` (historical ``round(x/v)``) or ``signedlog``
            (per-axis log bins whose edge is ``voxel_size`` at ``|x| = 1``).
    Outputs:
        point_to_voxel_all: tensor [B, N], is the mapping from original point cloud to voxel
    """
    B, N, _ = xyz.shape
    mode = str(mode or "uniform")
    if mode == "uniform":
        xyz = xyz / voxel_size
        xyz = torch.round(xyz).long()
    elif mode == "signedlog":
        xyz = _signedlog_bins(xyz, voxel_size)
    else:
        raise ValueError(f"voxelization supports uniform/signedlog, got {mode!r}")
    xyz = xyz - xyz.min(1, keepdim=True)[0]

    keys = ravel_hash_vec(xyz)  # [B, N] int64

    # Sync-free per-batch unique inverse via sort + cumsum + scatter.
    # torch.unique(..., return_inverse=True) forces a cudaMemcpy (output size is
    # unknown to the host), so a Python for-loop over B of it produced B syncs
    # per voxelization call and was dominating streaming_forward_step in
    # chrome traces. The labeling produced here matches torch.unique's in the
    # sense that points with identical keys get the same compact id in [0, K-1];
    # the exact integer assignment may differ but scatter_mean/merge only rely
    # on the grouping.
    sorted_keys, sort_idx = torch.sort(keys, dim=1)
    is_new = torch.ones_like(sorted_keys, dtype=torch.bool)
    if N > 1:
        is_new[:, 1:] = sorted_keys[:, 1:] != sorted_keys[:, :-1]
    voxel_id_sorted = is_new.to(torch.long).cumsum(dim=1) - 1
    point_to_voxel = torch.empty_like(keys)
    point_to_voxel.scatter_(1, sort_idx, voxel_id_sorted)
    return point_to_voxel


def voxelize_memory_p2v(
    mem_coords: torch.Tensor,
    mem_ever_dynamic: torch.Tensor,
    voxel_size: float,
    compress_to: Optional[int] = None,
    max_tries: int = 2,
    mode: str = "uniform",
):
    """Voxelize accumulated memory, coarsening ONLY never-dynamic points.

    Args:
        mem_coords: ``(M, 3)`` memory point coordinates.
        mem_ever_dynamic: ``(M,)`` bool; True if the point has ever been dynamic.
        voxel_size: base voxel size used for ever-dynamic points (and for the
            whole cloud when compression is inactive).
        compress_to: if set and the never-dynamic count exceeds this, re-voxelize
            the static subset at ``voxel_size * 2`` then ``* 4`` (up to
            ``max_tries``), stopping early once the static voxel count is
            ``<= compress_to``.
        max_tries: maximum coarsening attempts (default 2 -> 2x then 4x).

    Returns:
        ``(p2v_mem, mem_vocab, eff_vox)`` where ``p2v_mem`` is ``(M,)`` long ids
        in ``[0, mem_vocab)`` and ``eff_vox`` is the voxel size used for the
        static subset.
    """
    # Short-circuit: keep byte-for-byte identical to a plain whole-memory
    # voxelization when compression is disabled or already under budget.
    mem_static = ~mem_ever_dynamic
    n_static = int(mem_static.sum().item())
    n_dyn = int(mem_ever_dynamic.sum().item())
    M = mem_coords.shape[0]
    need_coarsen = compress_to is not None and n_static > compress_to
    if not need_coarsen:
        p2v_mem = voxelization(mem_coords[None], voxel_size, mode=mode)[0]
        mem_vocab = int(p2v_mem.max().item()) + 1 if p2v_mem.numel() > 0 else 0
        if compress_to is not None:
            logger.info(
                "[compress_memory] skip coarsen: mem=%d static=%d dyn=%d "
                "target=%d (already <= target); vocab=%d vox=%.4f",
                M, n_static, n_dyn, compress_to, mem_vocab, voxel_size,
            )
        return p2v_mem, mem_vocab, voxel_size

    # Coarsen the static subset only; dynamic points stay at ``voxel_size``.
    # Disjoint id ranges keep a coarsened static voxel from merging with a
    # dynamic point in the same xyz cell.
    eff_vox = voxel_size
    p2v_static = None
    n_static_vox = n_static
    hit_target = True
    if n_static > 0:
        hit_target = False
        for k in range(1, max_tries + 1):
            cand = voxel_size * (2 ** k)
            p2v_static = voxelization(mem_coords[mem_static][None], cand, mode=mode)[0]
            eff_vox = cand
            n_static_vox = int(p2v_static.max().item()) + 1
            logger.info(
                "[compress_memory] try %d/%d: static %d -> %d voxels @ vox=%.4f "
                "(target=%d)",
                k, max_tries, n_static, n_static_vox, cand, compress_to,
            )
            if n_static_vox <= compress_to:
                hit_target = True
                break

    # Build disjoint id ranges: static [0, Ks), dynamic [Ks, Ks+Kd).
    p2v_mem = torch.empty(M, dtype=torch.long, device=mem_coords.device)

    if n_static > 0:
        assert p2v_static is not None
        Ks = int(p2v_static.max().item()) + 1
        p2v_mem[mem_static] = p2v_static
    else:
        Ks = 0

    mem_dyn = mem_ever_dynamic
    if n_dyn > 0:
        p2v_dyn = voxelization(mem_coords[mem_dyn][None], voxel_size, mode=mode)[0]
        Kd = int(p2v_dyn.max().item()) + 1
        p2v_mem[mem_dyn] = p2v_dyn + Ks
    else:
        Kd = 0

    logger.info(
        "[compress_memory] done: mem=%d static_in=%d static_out=%d dyn=%d "
        "vocab=%d eff_vox=%.4f target=%d hit_target=%s",
        M, n_static, Ks, n_dyn, Ks + Kd, eff_vox, compress_to, hit_target,
    )
    return p2v_mem, Ks + Kd, eff_vox


def merge_per_frame_p2v_labels(
    p2v_bt_n: torch.Tensor,
    base_offset: Union[int, torch.Tensor] = 0,
) -> torch.Tensor:
    """Concatenate per-frame compact voxel ids into one global id space.

    ``p2v_bt_n`` is ``[num_frames, n_per_frame]`` from :func:`voxelization`.
    ``base_offset`` is added to the cumulative row offsets (e.g. voxel count
    after a separate memory voxelization).
    """
    device, dtype = p2v_bt_n.device, p2v_bt_n.dtype
    # Vectorized per-frame max + offsets (avoids Python for-loop over the
    # batch dim).
    max_lengths = p2v_bt_n.amax(dim=-1) + 1  # [num_frames]
    cumsum = torch.cumsum(max_lengths, dim=0)
    offsets = torch.cat(
        [torch.zeros(1, device=device, dtype=dtype), cumsum[:-1]]
    )
    if isinstance(base_offset, torch.Tensor):
        bo = base_offset.to(device=device, dtype=dtype)
    else:
        bo = torch.tensor(base_offset, device=device, dtype=dtype)
    offsets = offsets + bo
    return (p2v_bt_n + offsets[:, None]).reshape(-1)


def voxel_map_to_source(voxel_map, poin2voxel):
    """
    Input:
        voxel_map (B, N1, ....)
        point2voxel (B, N)
    Output:
        src_new (B, N, ....)
    """
    bs = voxel_map.shape[0]
    src_new = torch.stack([voxel_map[i, poin2voxel[i]] for i in range(bs)])
    return src_new


# @torch.jit.script
def unvoxelize_pc_with_p2v_lists(
    pc,
    all_p2vs: List[torch.Tensor],
    all_p2vs_len: List[int] = None,
):
    """
    Input:
        pc: (N, 3)
        all_p2vs: list of (N')
    Output:
        pc: (N2, 3)
    """
    if len(all_p2vs) == 0:
        return pc
    pc_new = []
    all_p2vs = all_p2vs[::-1]
    if all_p2vs_len is not None:
        all_p2vs_len = all_p2vs_len[::-1]
    
    for i, p2v in enumerate(all_p2vs):
        pc = voxel_map_to_source(pc[None], p2v[None]).squeeze(0)
        if i != len(all_p2vs) - 1:
            if all_p2vs_len is not None:
                old_pc_len = all_p2vs_len[i + 1]
            else:
                old_pc_len = all_p2vs[i + 1].max() + 1
            pc_new.append(pc[old_pc_len:])
            pc = pc[:old_pc_len]
    pc_new.append(pc)
    return torch.cat(pc_new[::-1])


def unvoxelize_coords_with_residuals(
    coords: torch.Tensor,
    all_p2vs: List[torch.Tensor],
    all_coord_residuals: List[torch.Tensor],
    all_p2vs_len: List[int] = None,
) -> torch.Tensor:
    """Unvoxelize coordinates while restoring each input point's 3D offset.

    Each residual is aligned with the corresponding pre-voxelization p2v and
    equals ``pre_coords - voxel_coords[p2v]``. For trajectory tensors shaped
    ``(N, T, 3)``, the same spatial residual broadcasts across target time.
    """
    if len(all_p2vs) == 0:
        return coords
    if len(all_p2vs) != len(all_coord_residuals):
        raise ValueError(
            "Expected one coordinate residual per p2v mapping, got "
            f"{len(all_coord_residuals)} residuals and {len(all_p2vs)} mappings"
        )

    coords_new = []
    p2vs = all_p2vs[::-1]
    residuals = all_coord_residuals[::-1]
    p2v_lens = all_p2vs_len[::-1] if all_p2vs_len is not None else None

    for i, (p2v, residual) in enumerate(zip(p2vs, residuals)):
        coords = voxel_map_to_source(coords[None], p2v[None]).squeeze(0)
        residual = residual.to(device=coords.device, dtype=coords.dtype)
        while residual.ndim < coords.ndim:
            residual = residual.unsqueeze(-2)
        coords = coords + residual

        if i != len(p2vs) - 1:
            if p2v_lens is not None:
                old_coord_len = p2v_lens[i + 1]
            else:
                old_coord_len = p2vs[i + 1].max() + 1
            coords_new.append(coords[old_coord_len:])
            coords = coords[:old_coord_len]

    coords_new.append(coords)
    return torch.cat(coords_new[::-1])


def dense_sample_voxel_map(
    dense_sample_pts,
    window_end,
    Hf,
    Wf,
    all_p2vs,
    all_p2vs_len,
    num_voxels,
    device,
):
    """Map each sampled dense cell (src_frame, y, x) to its current voxel id.

    Returns:
        vox_of_cell: (M,) long voxel id in the current scene voxel space, or
            -1 for cells whose source frame has not been observed yet.
        born: (M,) bool, True where src_frame < window_end.
    """
    born = dense_sample_pts[:, 0] < window_end
    fi = (
        (dense_sample_pts[:, 0] * Hf + dense_sample_pts[:, 1]) * Wf
        + dense_sample_pts[:, 2]
    ).long()
    fi = torch.where(born, fi, torch.zeros_like(fi))
    payload = torch.arange(num_voxels, device=device, dtype=torch.float32)[:, None]
    vox_of_pixel = unvoxelize_pc_with_p2v_lists(payload, all_p2vs, all_p2vs_len)  # (window_end*Hf*Wf, 1)
    vox_of_cell = vox_of_pixel[fi, 0].round().long()
    vox_of_cell = torch.where(born, vox_of_cell, torch.full_like(vox_of_cell, -1))
    return vox_of_cell, born


def compute_subsample_dense_mask(
    all_p2vs: List[torch.Tensor],
    all_p2vs_len: List[int],
    subsample_alive_list: List[torch.Tensor],
) -> torch.Tensor:
    """Compute a flat boolean mask over the original dense grid indicating which
    pixels have valid (non-garbage) predictions after voxel subsampling.

    When voxels are subsampled and the p2v is remapped (unsampled -> 0), pixels
    that mapped to removed voxels silently pick up voxel-0's value during
    unvoxelization.  This function walks the p2v chain (latest window first)
    and ANDs per-window survival masks so that a pixel is True only if its
    voxel survived subsampling in *every* window.

    Args:
        all_p2vs: remapped p2v tensors, one per window (chronological order).
        all_p2vs_len: voxel count after each window's voxelisation+subsampling.
        subsample_alive_list: per-window boolean tensors (pre-vox point level)
            indicating which points' voxels were kept, or None for windows
            where no subsampling occurred.  Must be same length as all_p2vs.

    Returns:
        Flat bool tensor of length sum(raw_pixels_per_window), in chronological
        window order – same layout as the output of unvoxelize_pc_with_p2v_lists.
    """
    if not all_p2vs:
        return torch.tensor([], dtype=torch.bool)

    num_windows = len(all_p2vs)
    device = all_p2vs[-1].device
    alive = torch.ones(all_p2vs_len[-1], dtype=torch.bool, device=device)

    dense_mask_parts: List[torch.Tensor] = []

    for w in range(num_windows - 1, -1, -1):
        expanded_alive = alive[all_p2vs[w]]

        if subsample_alive_list[w] is not None:
            expanded_alive = expanded_alive & subsample_alive_list[w]

        if w > 0:
            mem_len = all_p2vs_len[w - 1]
            dense_mask_parts.append(expanded_alive[mem_len:])
            alive = expanded_alive[:mem_len]
        else:
            dense_mask_parts.append(expanded_alive)

    return torch.cat(dense_mask_parts[::-1])


def invalid_to_zeros(arr, valid_mask, ndim=999):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = 0
        nnz = valid_mask.view(len(valid_mask), -1).sum(1)
    else:
        nnz = arr.numel() // len(arr) if len(arr) else 0  # number of point per image
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr, nnz


def normalize_scene_geometry(intrinsics, extrinsics, depths, norm_mode='avg_dis', norm_mask=None, subsample_factor=1):
    if subsample_factor > 1:
        depths = depths[:, ::subsample_factor, ::subsample_factor]
        if norm_mask is not None:
            norm_mask = norm_mask[:, ::subsample_factor, ::subsample_factor]
        intrinsics = intrinsics.copy()
        intrinsics[:, :2, :] /= subsample_factor
    valid = depths > 0
    if norm_mask is not None:
        valid = valid & norm_mask
    pts = batch_unproject_np(
        depths,
        intrinsics,
        extrinsics,
        is_distance=False,
    )
    norm_factor = normalize_pointcloud(
        torch.from_numpy(pts).flatten(0, 2)[None],
        norm_mode=norm_mode,
        valid=torch.from_numpy(valid).flatten(0, 2)[None],
    )
    return norm_factor.cpu().numpy()

def normalize_pointcloud(pts, norm_mode='avg_dis', valid=None):
    """ renorm pointmaps pts1 with norm_mode
    """
    assert pts.ndim >= 3 and pts.shape[-1] == 3
    norm_mode, dis_mode = norm_mode.split('_')

    if norm_mode == 'avg':
        # gather all points together (joint normalization)
        nan_pts, nnz = invalid_to_zeros(pts, valid, ndim=3)
        all_pts = nan_pts

        # compute distance to origin
        all_dis = all_pts.norm(dim=-1)
        if dis_mode == 'dis':
            pass  # do nothing
        elif dis_mode == 'log1p':
            all_dis = torch.log1p(all_dis)
        elif dis_mode == 'warp-log1p':
            # actually warp input points before normalizing them
            log_dis = torch.log1p(all_dis)
            warp_factor = log_dis / all_dis.clip(min=1e-8)
            H1, W1 = pts.shape[1:-1]
            pts = pts * warp_factor[:, :W1 * H1].view(-1, H1, W1, 1)
            all_dis = log_dis  # this is their true distance afterwards
        else:
            raise ValueError(f'bad {dis_mode=}')

        norm_factor = all_dis.sum(dim=1) / (nnz + 1e-8)
    else:
        # gather all points together (joint normalization)
        nan_pts, nnz = invalid_to_zeros(pts, valid, ndim=3)
        all_pts = nan_pts

        # compute distance to origin
        all_dis = all_pts.norm(dim=-1)
        # exclude invalid points from reduction (they were zeroed, so have dist 0)
        if valid is not None:
            valid_flat = valid.view(valid.shape[0], -1).expand_as(all_dis)
            all_dis = all_dis.clone().masked_fill(~valid_flat, float('nan'))

        if norm_mode == 'avg':
            norm_factor = all_dis.nanmean(dim=1)
        elif norm_mode == 'median':
            norm_factor = all_dis.nanmedian(dim=1).values.detach()
        elif norm_mode == 'sqrt':
            norm_factor = all_dis.sqrt().nanmean(dim=1)**2
        else:
            raise ValueError(f'bad {norm_mode=}')

    norm_factor = norm_factor.clip(min=1e-8)
        
    return norm_factor


def apply_homo_transform_torch(
    coords: torch.Tensor,
    transform: torch.Tensor,
    homo_input: bool = False,
):
    """
    Args:
        coords: (..., D) tensor of coordinates (typically D=3)
        transform: (..., D+1, D+1) homogeneous transform
        homo_input: if True, coords are already homogeneous (..., D+1)

    Returns:
        coords_transformed: (..., D) transformed coordinates
    """
    if homo_input:
        coords_homo = coords
    else:
        ones = torch.ones_like(coords[..., :1])
        coords_homo = torch.cat([coords, ones], dim=-1)
        
    while transform.dim() < coords_homo.dim() + 1:
        transform = transform.unsqueeze(1)

    coords_transformed_homo = torch.einsum(
        "...ij,...j->...i", transform, coords_homo
    )

    coords_transformed = (
        coords_transformed_homo[..., :-1]
        / coords_transformed_homo[..., -1:].clamp(min=1e-8)
    )

    return coords_transformed


def project_trajs_3d_to_2d(trajs_3d, extrinsics, intrinsics):
    traj_local = apply_homo_transform_torch(
            trajs_3d,
            extrinsics,
    )
    z_depth = traj_local[..., 2]
    traj_2d = apply_homo_transform_torch(
        traj_local,
        intrinsics,
        homo_input=True,
    )
    return traj_2d, z_depth


def world_to_local(world_coords, intrinsics, extrinsics):
    world_coords_homo = np.concatenate(
        [world_coords, np.ones_like(world_coords[..., :1])], axis=-1
    )
    local_coords = np.einsum("...ij,...j->...i", extrinsics, world_coords_homo)[
        ..., :3
    ]
    local_coords_2d = np.einsum("...ij,...j->...i", intrinsics, local_coords)[
        ..., :2
    ]
    local_coords_2d /= local_coords[..., 2:3]
    return np.concatenate([local_coords_2d, local_coords[..., -1:]], axis=-1)


def local_to_world(local_coords, intrinsics, extrinsics):
    inv_intrinsics = np.linalg.inv(intrinsics)
    inv_extrinsics = np.linalg.inv(extrinsics)
    local_coords_homo = np.concatenate(
        [local_coords[..., :2], np.ones_like(local_coords[..., :1])], axis=-1
    )
    camera_coords = np.einsum(
        "...ij,...j->...i", inv_intrinsics, local_coords_homo
    )[..., :3]
    camera_coords = camera_coords * local_coords[..., 2:3] / camera_coords[..., 2:3]
    camera_coords_homo = np.concatenate(
        [camera_coords, np.ones_like(camera_coords[..., :1])], axis=-1
    )
    world_coords = np.einsum(
        "...ij,...j->...i", inv_extrinsics, camera_coords_homo
    )[..., :3]
    return world_coords
    
    


def depth_edge(depth: torch.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch.Tensor = None) -> torch.BoolTensor:
    """
    Compute the edge mask of a depth map. The edge is defined as the pixels whose neighbors have a large difference in depth.
    
    Args:
        depth (torch.Tensor): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (torch.Tensor): shape (..., height, width) of dtype torch.bool
    """
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff = (F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
    else:
        diff = (F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2))

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    edge = edge.reshape(*shape)
    return edge