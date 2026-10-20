# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import NamedTuple, Optional

import ipdb

import numpy as np
import torch
from einops import rearrange, repeat

from ..datasets.utils.dynamic_labels import dynamic_extent_l2
from ..datasets.utils.geometry import (
    unvoxelize_coords_with_residuals,
    unvoxelize_pc_with_p2v_lists,
)
from ..training.datatypes import DenseTrackWindow, Prediction, TrainData


st = ipdb.set_trace
logger = logging.getLogger(__name__)


def _as_sequence_dynamic(dense_dynamic):
    if isinstance(dense_dynamic, torch.Tensor) and dense_dynamic.dim() == 5:
        return dense_dynamic.any(dim=-1)
    return dense_dynamic


def gather_dense_dynamic_logits_at_sample_points(dense_logits, points):
    """Gather dense-grid logits at dense-sample cell locations.

    Args:
        dense_logits: ``(B, T, H, W, T)`` source→target static/dynamic logits
            (or legacy ``(B, T, H, W)``).
        points: (B, M, 3) int ``[src_frame, y, x]`` at the same grid resolution.

    Returns:
        (B, T, M) logits. For 5D input, the source cell's full target-time track
        ``logits[b, src, y, x, :]``. For legacy 4D, fixed ``(y, x)`` across T.
    """
    B, T, H, W, T2 = dense_logits.shape
    src = points[..., 0].long().clamp(0, T - 1)  # (B, M)
    y = points[..., 1].long().clamp(0, H - 1)
    x = points[..., 2].long().clamp(0, W - 1)
    b_idx = torch.arange(B, device=dense_logits.device)[:, None]  # (B, 1)
    # (B, M, T_tgt) -> (B, T_tgt, M)
    return dense_logits[b_idx, src, y, x, :].permute(0, 2, 1)


def dense_dynamic_logits_diagonal(dense_logits):
    """Extract per-frame (B, T, H, W) from source→target (B, T, H, W, T) logits.

    Returns ``logits[b, t, y, x, t]`` (diagonal of source/target time). Legacy
    4D inputs are returned unchanged.
    """
    if dense_logits is None or dense_logits.dim() != 5:
        return dense_logits
    T = dense_logits.shape[1]
    return torch.stack(
        [dense_logits[:, t, :, :, t] for t in range(T)], dim=1
    )


def prepare_dense_sample_tracks_for_eval(sample, preds):
    """Build a query-style eval variant from the subsampled dense-cell tracks.

    The predicted per-cell full tracks (`preds.dense_sample_coords`) are compared
    against the grid-anchored GT tracks (`sample.dense_sample_*`). We synthesize a
    query point per cell at its source frame so the standard query tracking metric
    can be reused (reported under the `dense_sample_` prefix).
    """
    ds_trajs = sample.dense_sample_trajs_3d  # (B, T, M, 3)
    ds_vis = sample.dense_sample_visibs  # (B, T, M)
    ds_val = sample.dense_sample_valids  # (B, T, M)
    ds_pts = sample.dense_sample_points  # (B, M, 3) = [src_frame, y, x]
    ds_dyn = sample.dense_sample_dynamic  # (B, M)

    B, T, M, _ = ds_trajs.shape
    src = ds_pts[..., 0].long().clamp(0, T - 1)  # (B, M)

    # a cell is only tracked from its source (birth) frame onward; frames before
    # birth are not valid predictions.
    frame_idx = torch.arange(T, device=ds_val.device)  # (T,)
    pre_birth = frame_idx[None, :, None] < src[:, None, :]  # (B, T, M)
    ds_val = ds_val.bool() & ~pre_birth
    ds_vis = ds_vis.bool() & ~pre_birth

    xyz_at_src = torch.gather(
        ds_trajs, 1, src[:, None, :, None].expand(B, 1, M, 3)
    ).squeeze(1)  # (B, M, 3)
    gt_query_point = torch.cat(
        [src.to(ds_trajs.dtype)[..., None], xyz_at_src], dim=-1
    )  # (B, M, 4)

    sample.gt_trajs_3d = ds_trajs
    sample.visibs = ds_vis.bool()
    sample.valids = ds_val.bool()
    sample.gt_query_point = gt_query_point
    sample.is_dynamic = ds_dyn.bool() if ds_dyn is not None else None
    
    sample.est_query_point = sample.gt_query_point
    sample.est_trajs_3d = sample.gt_trajs_3d

    preds_ds = Prediction(
        coords=preds.dense_sample_coords,
        visibs=preds.dense_sample_visibs,
        is_dynamic=preds.is_dynamic,
    )
    if preds.dense_sample_dynamic_logits is not None:
        preds_ds.query_dynamic_logits = preds.dense_sample_dynamic_logits
    elif preds.dense_dynamic_logits is not None:
        preds_ds.query_dynamic_logits = gather_dense_dynamic_logits_at_sample_points(
            preds.dense_dynamic_logits, ds_pts
        )
    return sample, preds_ds


# --- query points read out of the dense tracks ("query_through_dense") -------

# Coordinates beyond this are sentinels, not predictions.
_QTD_VALID_ABS = 50.0
_QTD_INVALID = -1000.0


def _project_to_pixels(xyz, t, intrinsics, extrinsics):
    """World points (N, 3) at frames t (N,) -> pixel uv (N, 2) and camera z (N,)."""
    ones = torch.ones(xyz.shape[0], 1, device=xyz.device, dtype=xyz.dtype)
    cam_h = torch.einsum("nij,nj->ni", extrinsics[t], torch.cat([xyz, ones], dim=-1))
    cam = cam_h[:, :3] / cam_h[:, 3:].clamp(min=1e-8)
    pix_h = torch.einsum("nij,nj->ni", intrinsics[t], cam)
    return pix_h[:, :2] / pix_h[:, 2:].clamp(min=1e-8), cam[:, 2]


def offset_tracks_to_query(tracks, query_t, query_xyz):
    """Shift each (N, T, 3) track by ``query_xyz - track[query_t]``.

    Only written frames move, and a track whose source frame was never
    written is left as is.
    """
    n = tracks.shape[0]
    src = tracks[torch.arange(n, device=tracks.device), query_t]
    valid_src = src.abs().amax(-1) < _QTD_VALID_ABS
    shifted = tracks + (query_xyz - src)[:, None, :]
    shifted = torch.where((tracks.abs().amax(-1) > _QTD_VALID_ABS)[..., None], tracks, shifted)
    return torch.where(valid_src[:, None, None], shifted, tracks)


def dense_window_grid(window, pred, H, W):
    """Unvoxelized dense tracks of one window.

    Returns (ws, we, coords (we, H, W, T_win, 3), visibility logits
    (we, H, W, T_win)). The residual is restored, so a cell's track is its own
    point and not the voxel center.
    """
    ws, we = window.window_start, window.window_end
    t_win = we - ws
    device = pred.dense_extrinsics.device
    dtype = window.static_coords.dtype
    n_vox = window.sd_mask.shape[0]
    trajs = window.static_coords.to(device=device, dtype=dtype)[:, None, :]
    trajs = trajs.expand(n_vox, t_win, 3).clone()
    trajs[window.sd_mask] = window.sd_trajs.to(device=device, dtype=dtype)
    trajs = _unvoxelize_window(trajs, window, restore_coord_offsets=True)
    trajs = rearrange(trajs, "(t h w) tw c -> t h w tw c", t=we, h=H, w=W)

    surface = _window_diagonal_pointmap(window, H * W).reshape(t_win, H, W, 3)
    ext = pred.dense_extrinsics[0, ws:we].to(device=device)
    intr = pred.dense_intrinsics[0, ws:we].to(device=device)
    vis = _window_static_visibs(window, surface, ext, intr).to(device=device, dtype=dtype)
    vis[window.sd_mask] = window.sd_visibs.to(device=device, dtype=dtype)
    vis = _unvoxelize_window(vis, window)
    vis = rearrange(vis, "(t h w) tw -> t h w tw", t=we, h=H, w=W)
    return ws, we, trajs, vis


def query_tracks_through_dense(pred, query_xyz, query_t, *, radius: int = 2, k: int = 4):
    """Read query tracks out of the dense tracks instead of the query branch.

    Each query is projected into its source frame on the dense feature grid.
    Among the source-frame cells in the (2*radius+1)^2 window around it, the k
    nearest in 3D are blended by inverse distance. The blend is then shifted so
    the source frame lands exactly on the query (dense cells sit at cell
    centers, queries are subpixel). Visibility comes from the nearest cell.
    Tracks are gathered window by window, so the (T, H, W, T, 3) tensor is never
    built.

    Args:
        pred: eval Prediction with ``dense_track_3d`` (``dense_tracking_at_eval``).
        query_xyz: (N, 3) world-space query points, as fed to the model.
        query_t: (N,) long source frames.
    Returns:
        tracks (N, T, 3), visibility logits (N, T).
    """
    _, T, H, W = pred.dense_shape
    intr = pred.dense_intrinsics[0]
    extr = pred.dense_extrinsics[0]
    uv, _ = _project_to_pixels(query_xyz, query_t, intr, extr)
    # cell (x, y) is centered at pixel (x + 0.5, y + 0.5)
    y0 = (uv[:, 1] - 0.5).floor().long().clamp(0, H - 1)
    x0 = (uv[:, 0] - 0.5).floor().long().clamp(0, W - 1)
    n = query_xyz.shape[0]
    device = query_xyz.device
    side = 2 * radius + 1
    kk = side * side
    off = torch.arange(-radius, radius + 1, device=device)
    yy = (y0[:, None, None] + off[None, :, None]).expand(-1, side, side).reshape(n, kk)
    xx = (x0[:, None, None] + off[None, None, :]).expand(-1, side, side).reshape(n, kk)
    inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    yy = yy.clamp(0, H - 1)
    xx = xx.clamp(0, W - 1)

    coords = torch.full((n, kk, T, 3), _QTD_INVALID, device=device, dtype=query_xyz.dtype)
    vis = torch.full((n, kk, T), _QTD_INVALID, device=device, dtype=query_xyz.dtype)
    for window in pred.dense_track_3d:
        ws, we, grid, grid_vis = dense_window_grid(window, pred, H, W)
        alive = query_t < we
        if bool(alive.any()):
            ts = query_t[alive]
            coords[alive, :, ws:we] = grid[ts[:, None], yy[alive], xx[alive]]
            vis[alive, :, ws:we] = grid_vis[ts[:, None], yy[alive], xx[alive]]
        del grid, grid_vis

    pre = torch.arange(T, device=device).view(1, 1, T) < query_t.view(n, 1, 1)
    coords = coords.masked_fill(pre.unsqueeze(-1), _QTD_INVALID)
    vis = vis.masked_fill(pre, _QTD_INVALID)
    coords = coords.masked_fill(~inside[:, :, None, None], _QTD_INVALID)
    vis = vis.masked_fill(~inside[:, :, None], _QTD_INVALID)

    row = torch.arange(n, device=device)
    src = coords[row[:, None], torch.arange(kk, device=device)[None, :], query_t[:, None]]
    ok = (src.abs().amax(-1) < _QTD_VALID_ABS) & inside
    dist = (src - query_xyz[:, None, :]).norm(dim=-1).masked_fill(~ok, float("inf"))
    found = torch.isfinite(dist).any(-1)
    vals, idxs = torch.topk(dist.masked_fill(~found[:, None], 0.0), k=min(k, kk), largest=False)
    vals = torch.where(found[:, None], vals, torch.full_like(vals, float("inf")))

    acc = torch.zeros(n, T, 3, device=device, dtype=query_xyz.dtype)
    wsum = torch.zeros(n, 1, 1, device=device, dtype=query_xyz.dtype)
    for j in range(idxs.shape[1]):
        wj = 1.0 / vals[:, j].clamp(min=1e-4)
        wj = torch.where(torch.isfinite(vals[:, j]), wj, torch.zeros_like(wj)).to(query_xyz.dtype)
        acc = acc + wj[:, None, None] * coords[row, idxs[:, j]]
        wsum = wsum + wj[:, None, None]
    blended = acc / wsum.clamp(min=1e-8)
    # no valid neighbor: hold the query still, visible
    blended = torch.where(found[:, None, None], blended, query_xyz[:, None, :].expand(n, T, 3))
    vis_out = torch.where(found[:, None], vis[row, idxs[:, 0]], torch.full((n, T), 10.0, device=device, dtype=query_xyz.dtype))
    return offset_tracks_to_query(blended, query_t, query_xyz), vis_out


def prepare_query_through_dense_for_eval(est_sample, preds):
    """Query-style Prediction whose tracks come from the dense branch.

    Scored against the same query GT as the ``query`` eval mode, so the two are
    directly comparable. Needs ``model.dense_tracking_at_eval=true``. Returns
    None when the forward kept no dense tracks.
    """
    if preds.dense_track_3d is None:
        return None
    query_xyz = est_sample.query_point[0, :, 1:].contiguous()
    T = preds.dense_shape[1]
    query_t = est_sample.query_point[0, :, 0].round().long().clamp(0, T - 1)
    tracks, vis = query_tracks_through_dense(preds, query_xyz, query_t)
    return Prediction(
        coords=tracks.permute(1, 0, 2)[None].contiguous(),
        visibs=vis.permute(1, 0)[None].contiguous(),
        is_dynamic=preds.is_dynamic,
    )


def prepare_dense_tracks_for_eval(
    gt_sample, train_data, evaluate_with_query_points=False
):
    est_dense_trajs_3d = gt_sample.est_dense_trajs_3d
    gt_dense_trajs_3d = gt_sample.dense_trajs_3d
    gt_dense_trajs_valid = gt_sample.dense_valids
    gt_dense_visibs = gt_sample.dense_visibs
    gt_dense_dynamic = _as_sequence_dynamic(gt_sample.dense_dynamic)

    gt_sample.valids = gt_dense_trajs_valid.flatten(1, 3).permute(0, 2, 1)
    valids = gt_sample.valids.any(1)[0]

    # assert gt_dense_trajs_3d.shape[0] == 1, "only one batch supported for now"
    gt_sample.gt_trajs_3d = gt_dense_trajs_3d.flatten(1, 3).permute(0, 2, 1, 3)[
        :, :, valids
    ]  # B, T, N, 3
    if est_dense_trajs_3d[0] is not None:
        gt_sample.est_trajs_3d = est_dense_trajs_3d.flatten(1, 3).permute(0, 2, 1, 3)[
            :, :, valids
        ]  # B, T, N, 3
    # gt_sample.valids = gt_dense_trajs_valid.flatten(1, 3).permute(0, 2, 1)
    gt_sample.valids = gt_sample.valids[:, :, valids]  # B, T, N
    gt_sample.visibs = gt_dense_visibs.flatten(1, 3).permute(0, 2, 1)[
        :, :, valids
    ]  # B, T, N
    gt_sample.is_dynamic = gt_dense_dynamic.flatten(1, 3)[:, valids]  # B, T, N

    if isinstance(train_data, TrainData):
        assert len(train_data.coords) == 1, "assuming one iteration for now"
        train_data.coords = [
            train_data.coords[0].flatten(1, 3).permute(0, 2, 1, 3)[:, :, valids]
        ]  # B, T, N, 3
    else:
        train_data.coords = train_data.coords.flatten(1, 3).permute(0, 2, 1, 3)[
            :, :, valids
        ]  # B, T, N, 3
    train_data.visibs = train_data.visibs.flatten(1, 3).permute(0, 2, 1)[:, :, valids]
    if train_data.is_dynamic is not None:
        train_data.is_dynamic = train_data.is_dynamic.flatten(1, 3)[:, valids]

    gt_sample.est_dense_trajs_3d = gt_sample.est_trajs_3d
    gt_sample.dense_trajs_3d = gt_sample.gt_trajs_3d
    gt_sample.dense_valids = gt_sample.valids
    gt_sample.dense_visibs = gt_sample.visibs
    gt_sample.dense_dynamic = gt_sample.is_dynamic

    return gt_sample, train_data


def find_annotated_tracks_from_all_tracks(gt_sample, train_data):
    gt_trajs_2d = gt_sample.gt_trajs_2d  # B, T, N, 2
    visibs = gt_sample.visibs  # B, T, N
    valids = gt_sample.valids  # B, T, N

    valid_2d_traj_mask = visibs & valids  # B, T, N

    h_pred, w_pred = train_data.visibs.shape[3:5]
    h_gt, w_gt = gt_sample.rgbs.shape[3:5]
    B, T = gt_sample.rgbs.shape[:2]
    N = gt_trajs_2d.shape[2]
    gt_trajs_2d = (
        gt_trajs_2d
        * torch.tensor([w_pred / w_gt, h_pred / h_gt], device=gt_trajs_2d.device)[
            None, None, None, :
        ]
    )

    height_idx = gt_trajs_2d[:, :, :, 1].long()
    width_idx = gt_trajs_2d[:, :, :, 0].long()

    height_idx = height_idx.clamp(0, h_pred - 1)
    width_idx = width_idx.clamp(0, w_pred - 1)
    b_idx = torch.arange(B).view(B, 1, 1).expand(-1, T, N).to(height_idx)
    t1_idx = torch.arange(T).view(1, T, 1).expand(B, -1, N).to(height_idx)

    if isinstance(train_data, TrainData):
        assert len(train_data.coords) == 1, "assuming one iteration for now"
        pred_trajs3d_indexed = train_data.coords[0][
            b_idx, :, t1_idx, height_idx, width_idx
        ]
    else:
        pred_trajs3d_indexed = train_data.coords[
            b_idx, :, t1_idx, height_idx, width_idx
        ]

    pred_trajs_valid = pred_trajs3d_indexed[valid_2d_traj_mask][None]  # B, N, T, 3
    pred_trajs_valid = rearrange(pred_trajs_valid, "b n t c -> b t n c")

    pred_visibs_indexed = train_data.visibs[b_idx, :, t1_idx, height_idx, width_idx]
    pred_visibs_valid = pred_visibs_indexed[valid_2d_traj_mask][None]  # B, N, T
    pred_visibs_valid = rearrange(pred_visibs_valid, "b n t -> b t n")

    pred_is_dynamic_indexed = train_data.is_dynamic[
        b_idx, t1_idx, height_idx, width_idx
    ]
    pred_is_dynamic_valid = pred_is_dynamic_indexed[valid_2d_traj_mask][None]  # B, N

    gt_trajs_3d = gt_sample.gt_trajs_3d  # B, T, N, 3

    gt_trajs_valid = repeat(gt_trajs_3d, "b t n c -> b T n t c", T=T)[
        valid_2d_traj_mask
    ][None]  # B, N, T, 3
    gt_trajs_valid = rearrange(gt_trajs_valid, "b n t c -> b t n c")

    gt_visibs_new = repeat(visibs, "b t n -> b T n t", T=T)[valid_2d_traj_mask][
        None
    ]  # B, N, T
    gt_visibs_new = rearrange(gt_visibs_new, "b n t -> b t n")
    gt_valids_new = repeat(valids, "b t n -> b T n t", T=T)[valid_2d_traj_mask][
        None
    ]  # B, N, T
    gt_valids_new = rearrange(gt_valids_new, "b n t -> b t n")
    gt_is_dynamic = repeat(gt_sample.is_dynamic, "b n -> b T n", T=T)[
        valid_2d_traj_mask
    ][None]  # B, N

    gt_sample_new = copy.deepcopy(gt_sample)
    gt_sample_new.gt_trajs_3d = gt_trajs_valid
    gt_sample_new.visibs = gt_visibs_new
    gt_sample_new.valids = gt_valids_new
    gt_sample_new.is_dynamic = gt_is_dynamic

    train_data_new = copy.deepcopy(train_data)

    if isinstance(train_data, TrainData):
        train_data_new.coords = [pred_trajs_valid]
    else:
        train_data_new.coords = pred_trajs_valid
    train_data_new.visibs = pred_visibs_valid
    train_data_new.is_dynamic = pred_is_dynamic_valid

    return gt_sample_new, train_data_new


def compute_dense_gt_tensor(
    trajs_2d,
    trajs_3d,
    visibs,
    is_dynamic,
    H,
    W,
    gt_depths_downsampled=None,
    trajs_depths=None,
    stride=2,
    skip_tmaps=False,
):
    """
    Compute dense ground truth tensor.
    Parameters:
    trajs_2d (numpy array): 3D array of shape (S, N, D) containing 2D trajectories.
    trajs_3d (numpy array): 3D array of shape (S, N, 3)
    visibs (numpy array): 2D array of shape (S, N) containing visibility information.
    H (int): Height of the image.
    W (int): Width of the image.
    stride (int): Stride for downsampling. Defaults to 2.
    Returns:
    tmaps (numpy array): 4D array of shape (S, S * 3, H2, W2) containing trajectory maps.
    tvalids (numpy array): 4D array of shape (S, 1, H2, W2) containing validity maps.
    """
    S, N, D = trajs_2d.shape
    H2, W2 = H // stride, W // stride
    tmaps = np.zeros((S, S * 3, H2, W2))
    tvalids = np.zeros((S, 1, H2, W2))
    tvisibs = np.zeros((S, S, H2, W2))
    tis_dynamic = np.zeros((S, H2, W2))
    tbg_mask = np.zeros((S, H2, W2), dtype=bool)
    for si in range(S):
        xy = np.round(trajs_2d[si].astype(np.float32) / stride).astype(np.int64)  # N,2
        if np.sum(visibs[si]) > 0:
            # i only want the trajs which are visible in the current timestep
            # and for those, i want the retrieved traj to match the FULL traj_g (T*3)
            vis_ok = visibs[si] > 0
            xy = xy[vis_ok]  # M,2

            samp_trajs_cam0 = trajs_3d[:, vis_ok]  # S,M,3
            M = xy.shape[0]
            x, y = xy[:, 0], xy[:, 1]  # M
            samp_trajs_cam0 = np.transpose(samp_trajs_cam0, (0, 2, 1)).reshape(S * 3, M)
            tmaps[si, :, y, x] = samp_trajs_cam0.transpose(1, 0)
            tvalids[si, :, y, x] = 1
            tvisibs[si, :, y, x] = visibs[:, vis_ok].transpose(1, 0)
            tis_dynamic[si, y, x] = is_dynamic[vis_ok]
            if gt_depths_downsampled is not None:
                gt_depths_downsampled[si, y, x] = trajs_depths[si, vis_ok]

    tmaps = tmaps.reshape(S, S, 3, H2, W2)
    tmaps = rearrange(tmaps, "t1 t2 n h w -> t1 h w t2 n").astype(np.float32)
    tvisibs = rearrange(tvisibs, "t1 t2 h w -> t1 h w t2")
    tvisibs = tvisibs.astype(bool)
    tis_dynamic = tis_dynamic.astype(bool)
    tvalids = np.repeat(tvalids.squeeze(1)[..., None], S, axis=-1).astype(bool)
    tbg_mask = tbg_mask.astype(bool)
    return tmaps, tvalids, tvisibs, tis_dynamic, gt_depths_downsampled


# def compute_dense_gt_tensor(
#     trajs_2d,
#     trajs_3d,
#     visibs,
#     is_dynamic,
#     H,
#     W,
#     gt_depths_downsampled=None,
#     trajs_depths=None,
#     stride=2,
#     skip_tmaps=False,
# ):
#     """
#     Compute dense ground truth tensor.
#     Parameters:
#     trajs_2d (numpy array): 3D array of shape (S, N, D) containing 2D trajectories.
#     trajs_3d (numpy array): 3D array of shape (S, N, 3)
#     visibs (numpy array): 2D array of shape (S, N) containing visibility information.
#     H (int): Height of the image.
#     W (int): Width of the image.
#     stride (int): Stride for downsampling. Defaults to 2.
#     Returns:
#     tmaps (numpy array): 4D array of shape (S, S * 3, H2, W2) containing trajectory maps.
#     tvalids (numpy array): 4D array of shape (S, 1, H2, W2) containing validity maps.
#     """
#     # to avoid modifying the original gt_depths_downsampled
#     if gt_depths_downsampled is not None:
#         gt_depths_downsampled = gt_depths_downsampled.copy()
    
#     S, N, D = trajs_2d.shape
#     H2, W2 = H, W
#     if not skip_tmaps:
#         tmaps = np.zeros((S, S * 3, H2, W2))
#     tvalids = np.zeros((S, 1, H2, W2), dtype=bool)
#     tvisibs = np.zeros((S, S, H2, W2), dtype=bool)
#     tis_dynamic = np.zeros((S, H2, W2), dtype=bool)
#     tbg_mask = np.zeros((S, H2, W2), dtype=bool)
#     for si in range(S):
#         xy = np.floor(trajs_2d[si].astype(np.float32)).astype(np.int64)  # N,2
#         xy[:, 0] = np.clip(xy[:, 0], 0, W2 - 1)
#         xy[:, 1] = np.clip(xy[:, 1], 0, H2 - 1)
#         if np.sum(visibs[si]) > 0:
#             # i only want the trajs which are visible in the current timestep
#             # and for those, i want the retrieved traj to match the FULL traj_g (T*3)
#             vis_ok = visibs[si] > 0
#             xy = xy[vis_ok]  # M,2

#             samp_trajs_cam0 = trajs_3d[:, vis_ok]  # S,M,3
#             M = xy.shape[0]
#             x, y = xy[:, 0], xy[:, 1]  # M
#             samp_trajs_cam0 = np.transpose(samp_trajs_cam0, (0, 2, 1)).reshape(S * 3, M)
#             if not skip_tmaps:
#                 tmaps[si, :, y, x] = samp_trajs_cam0.transpose(1, 0)
#             tvalids[si, :, y, x] = 1
#             tvisibs[si, :, y, x] = visibs[:, vis_ok].transpose(1, 0)
#             tis_dynamic[si, y, x] = is_dynamic[vis_ok]
#             if gt_depths_downsampled is not None:
#                 gt_depths_downsampled[si, y, x] = trajs_depths[si, vis_ok].copy()

#     if not skip_tmaps:
#         tmaps = tmaps.reshape(S, S, 3, H2, W2)
#         tmaps = rearrange(tmaps, "t1 t2 n h w -> t1 h w t2 n").astype(np.float32)
#     else:
#         tmaps = None
#     tvisibs = rearrange(tvisibs, "t1 t2 h w -> t1 h w t2")
#     tvisibs = tvisibs.astype(bool)
#     tis_dynamic = tis_dynamic.astype(bool)
#     tvalids = np.repeat(tvalids.squeeze(1)[..., None], S, axis=-1).astype(bool)
#     tbg_mask = tbg_mask.astype(bool)
#     if tmaps is not None:
#         tmaps = tmaps[:, stride//2 - 1::stride, stride//2 - 1::stride, ...]
#     tvalids = tvalids[:, stride//2 - 1::stride, stride//2 - 1::stride, ...]
#     tvisibs = tvisibs[:, stride//2 - 1::stride, stride//2 - 1::stride, ...]
#     tis_dynamic = tis_dynamic[:, stride//2 - 1::stride, stride//2 - 1::stride, ...]
#     if gt_depths_downsampled is not None:
#         gt_depths_downsampled = gt_depths_downsampled[:, stride//2 - 1::stride, stride//2 - 1::stride, ...]
#     return tmaps, tvalids, tvisibs, tis_dynamic, gt_depths_downsampled


def compute_dense_gt_tensor_fast(
    trajs_2d,
    trajs_3d,
    visibs,
    is_dynamic,
    H,
    W,
    gt_depths_downsampled=None,
    trajs_depths=None,
    stride=2,
    skip_tmaps=False,
    dense_geom=None,
    valids=None,
):
    """
    Vectorized (loop-free) version of compute_dense_gt_tensor.
    Same inputs/outputs as the original, plus an extra src_pos_map return
    (S, H2, W2, 3) with the scattered per-pixel source positions. If dense_geom
    (S, H2, W2, 3, world frame) is given, the track whose source position is
    closest to the patch base geometry wins when several tracks share a patch.
    If `valids` (raw per-track validity) is given, source gating uses visibs & valids
    and tvalids becomes per-target (set from valids at each target frame) instead of
    source-only; tvisibs stays per-target from the raw visibs.
    """
    if gt_depths_downsampled is not None:
        gt_depths_downsampled = gt_depths_downsampled.copy()

    S, N, D = trajs_2d.shape
    H2, W2 = H // stride, W // stride

    xy_all = np.floor(trajs_2d.astype(np.float32) / stride).astype(np.int64)  # (S, N, 2)
    xy_all[..., 0] = np.clip(xy_all[..., 0], 0, W2 - 1)
    xy_all[..., 1] = np.clip(xy_all[..., 1], 0, H2 - 1)

    if valids is not None:
        vis_mask = (visibs > 0) & (valids > 0)  # (S, N)
    else:
        vis_mask = visibs > 0  # (S, N)
    si_idx, n_idx = np.where(vis_mask)  # both (K,)

    x_coords = xy_all[si_idx, n_idx, 0]  # (K,)
    y_coords = xy_all[si_idx, n_idx, 1]  # (K,)

    # Closest-to-base selection (see compute_dense_gt_tensor_fast_for_targets).
    if dense_geom is not None and len(si_idx) > 0:
        src_pos = trajs_3d[si_idx, n_idx, :]  # (K, 3)
        geom_b = dense_geom[si_idx, y_coords, x_coords, :]  # (K, 3)
        dist = np.linalg.norm(src_pos - geom_b, axis=1)  # (K,)
        pid = (si_idx * H2 + y_coords) * W2 + x_coords
        order = np.lexsort((-dist, pid))  # smallest dist last -> wins
        si_idx = si_idx[order]
        n_idx = n_idx[order]
        y_coords = y_coords[order]
        x_coords = x_coords[order]

    if not skip_tmaps:
        tmaps = np.zeros((S, S * 3, H2, W2))
    tvalids = np.zeros((S, S, H2, W2), dtype=bool)
    tvisibs = np.zeros((S, S, H2, W2), dtype=bool)
    tis_dynamic = np.zeros((S, H2, W2), dtype=bool)
    src_pos_map = np.zeros((S, 3, H2, W2), dtype=np.float32)
    # Per (source_frame, y, x) cell, the index of the sparse track that won the
    # scatter (-1 where empty). Used to build grid-anchored full GT tracks for a
    # subsampled set of dense cells (base geometry + this track's relative motion).
    src_idx_map = np.full((S, H2, W2), -1, dtype=np.int64)

    if len(si_idx) > 0:
        if not skip_tmaps:
            traj_all_targets = trajs_3d[:, n_idx, :]  # (S, P, 3)
            traj_at_source = trajs_3d[si_idx, n_idx, :]  # (P, 3)
            flow_3d = traj_all_targets - traj_at_source[None, :, :]  # (S, P, 3)
            flow_vals = flow_3d.transpose(0, 2, 1).reshape(S * 3, -1).T  # (P, S*3)
            tmaps[si_idx, :, y_coords, x_coords] = flow_vals

        if valids is not None:
            tvalids[si_idx, :, y_coords, x_coords] = valids[:, n_idx].T  # (K, S)
        else:
            tvalids[si_idx, :, y_coords, x_coords] = True
        tvisibs[si_idx, :, y_coords, x_coords] = visibs[:, n_idx].T  # (K, S)
        tis_dynamic[si_idx, y_coords, x_coords] = is_dynamic[n_idx]

        src_pos_map[si_idx, :, y_coords, x_coords] = trajs_3d[si_idx, n_idx, :]
        src_idx_map[si_idx, y_coords, x_coords] = n_idx

        if gt_depths_downsampled is not None:
            gt_depths_downsampled[si_idx, y_coords, x_coords] = trajs_depths[si_idx, n_idx]

    if not skip_tmaps:
        tmaps = tmaps.reshape(S, S, 3, H2, W2)
        tmaps = rearrange(tmaps, "t1 t2 n h w -> t1 h w t2 n").astype(np.float32)
    else:
        tmaps = None
    tvisibs = rearrange(tvisibs, "t1 t2 h w -> t1 h w t2").astype(bool)
    tis_dynamic = tis_dynamic.astype(bool)
    tvalids = rearrange(tvalids, "t1 t2 h w -> t1 h w t2").astype(bool)
    src_pos_map = rearrange(src_pos_map, "t1 c h w -> t1 h w c").astype(np.float32)
    src_idx_map = src_idx_map.astype(np.int64)

    # if tmaps is not None:
    #     tmaps = tmaps[:, stride // 2 :: stride, stride // 2 :: stride, ...]
    # tvalids = tvalids[:, stride // 2 :: stride, stride // 2 :: stride, ...]
    # tvisibs = tvisibs[:, stride // 2 :: stride, stride // 2 :: stride, ...]
    # tis_dynamic = tis_dynamic[:, stride // 2 :: stride, stride // 2 :: stride, ...]
    # if gt_depths_downsampled is not None:
    #     gt_depths_downsampled = gt_depths_downsampled[
    #         :, stride // 2 :: stride, stride // 2 :: stride, ...
    #     ]
    return tmaps, tvalids, tvisibs, tis_dynamic, gt_depths_downsampled, src_pos_map, src_idx_map


def compute_track_dynamic_for_windows(
    trajs_3d,
    target_time_indices,
    dynamic_filter_threshold,
    window_size,
    valids=None,
):
    """Compute per-track dynamic labels for fixed-size frame windows.

    A track is dynamic for a target frame's window if its 3D position moves more
    than ``dynamic_filter_threshold`` anywhere within that window. Windows are the
    fixed tiling induced by ``window_size``: ``[0, w), [w, 2w), ...``. A target
    index ``t`` may land anywhere inside its window (e.g. an interior frame like 8
    for a window of [0, 16)), not only at the window end, so the window bounds are
    derived from ``window_size`` (via ``t // w``) rather than from the gaps between
    consecutive target indices.

    ``valids`` (S, N) restricts the extent to frames where the track is valid.
    Coordinates on invalid frames are not meaningful and one junk value labels a
    stationary track dynamic. See docs/static_dynamic_labels.md.
    """
    S, N, _ = trajs_3d.shape
    target_time_indices = np.asarray(target_time_indices, dtype=np.int64).reshape(-1)
    window_size = int(window_size)
    dynamic = np.zeros((target_time_indices.size, N), dtype=bool)

    for wi, t in enumerate(target_time_indices):
        window_idx = int(t) // window_size
        start = window_idx * window_size
        stop = start + window_size
        start = int(np.clip(start, 0, S - 1))
        stop = int(np.clip(stop, start + 1, S))
        window_trajs = trajs_3d[start:stop]
        window_valids = None if valids is None else valids[start:stop]
        diff_l2 = dynamic_extent_l2(window_trajs, window_valids)
        dynamic[wi] = diff_l2 > dynamic_filter_threshold

    return dynamic


def compute_dense_gt_tensor_fast_for_targets(
    trajs_2d,
    trajs_3d,
    visibs,
    is_dynamic,
    H,
    W,
    target_time_indices,
    gt_depths_downsampled=None,
    trajs_depths=None,
    stride=2,
    skip_tmaps=False,
    dynamic_filter_threshold=None,
    dense_geom=None,
    valids=None,
    window_size=None,
):
    """
    Like compute_dense_gt_tensor_fast, but only materializes target frames listed in
    target_time_indices (global indices in [0, S-1]). Output last spatial axes are
    (H', W', K, ...) instead of (H', W', S, ...) where K = len(target_time_indices).

    `visibs` is the raw per-track visibility. If `valids` (raw per-track validity) is
    given it is used two ways: (1) source gating uses visibs & valids to pick which
    pixel a track fills, and (2) tvalids becomes per-target, set from valids at each
    target frame, so a pixel is only marked valid for a target where the track's
    position at that target is actually valid (not just valid at the source frame).
    tvisibs stays per-target from the raw visibs. When valids is None the old
    source-only behavior is kept (tvalids True for all targets at scattered pixels).

    If dense_geom (S, H2, W2, 3, world frame) is given, then when several sparse
    tracks fall into the same downsampled patch the one whose source-frame 3D
    position is closest to that patch's base geometry wins the scatter (instead of
    last-write-wins). This keeps the flow aligned with the base depth pixel the
    model is anchored to. The scattered per-pixel source positions are returned as
    src_pos_map (S, H2, W2, 3) so the caller can drop pixels where base and flow
    straddle a depth edge.
    """
    if gt_depths_downsampled is not None:
        gt_depths_downsampled = gt_depths_downsampled.copy()

    S, N, D = trajs_2d.shape
    target_time_indices = np.asarray(target_time_indices, dtype=np.int64).reshape(-1)
    K = target_time_indices.size
    if K == 0:
        raise ValueError("target_time_indices must be non-empty")
    if np.any((target_time_indices < 0) | (target_time_indices >= S)):
        raise ValueError(
            f"target_time_indices must be in [0, {S - 1}], got {target_time_indices}"
        )

    H2, W2 = H // stride, W // stride

    xy_all = np.floor(trajs_2d.astype(np.float32) / stride).astype(np.int64)
    xy_all[..., 0] = np.clip(xy_all[..., 0], 0, W2 - 1)
    xy_all[..., 1] = np.clip(xy_all[..., 1], 0, H2 - 1)

    if valids is not None:
        vis_mask = (visibs > 0) & (valids > 0)
    else:
        vis_mask = visibs > 0
    si_idx, n_idx = np.where(vis_mask)

    x_coords = xy_all[si_idx, n_idx, 0]
    y_coords = xy_all[si_idx, n_idx, 1]

    # Closest-to-base selection: reorder scattered entries so that, within each
    # (source_frame, y, x) patch, the track whose source position is nearest the
    # patch base geometry writes last and therefore wins the last-write scatter.
    if dense_geom is not None and len(si_idx) > 0:
        src_pos = trajs_3d[si_idx, n_idx, :]  # (K, 3)
        geom_b = dense_geom[si_idx, y_coords, x_coords, :]  # (K, 3)
        dist = np.linalg.norm(src_pos - geom_b, axis=1)  # (K,)
        pid = (si_idx * H2 + y_coords) * W2 + x_coords
        order = np.lexsort((-dist, pid))  # smallest dist last -> wins
        si_idx = si_idx[order]
        n_idx = n_idx[order]
        y_coords = y_coords[order]
        x_coords = x_coords[order]

    if not skip_tmaps:
        tmaps = np.zeros((S, K * 3, H2, W2))
    tvalids = np.zeros((S, K, H2, W2), dtype=bool)
    tvisibs = np.zeros((S, K, H2, W2), dtype=bool)
    tis_dynamic = np.zeros((S, H2, W2, K), dtype=bool)
    src_pos_map = np.zeros((S, 3, H2, W2), dtype=np.float32)
    # Per (source_frame, y, x) cell, the winning sparse track index (-1 if empty).
    src_idx_map = np.full((S, H2, W2), -1, dtype=np.int64)
    if dynamic_filter_threshold is None:
        window_dynamic = np.repeat(is_dynamic[None], K, axis=0)
    else:
        if window_size is None:
            raise ValueError(
                "window_size must be provided when dynamic_filter_threshold is set "
                "so per-target dynamic labels use the correct fixed-size windows"
            )
        window_dynamic = compute_track_dynamic_for_windows(
            trajs_3d,
            target_time_indices,
            dynamic_filter_threshold,
            window_size,
            valids=valids,
        )

    if len(si_idx) > 0:
        if not skip_tmaps:
            traj_at_targets = trajs_3d[
                target_time_indices[:, None], n_idx[None, :], :
            ]  # (K, P, 3)
            traj_at_source = trajs_3d[si_idx, n_idx, :]  # (P, 3)
            flow_3d = traj_at_targets - traj_at_source[None, :, :]  # (K, P, 3)
            flow_vals = flow_3d.transpose(0, 2, 1).reshape(K * 3, -1).T
            tmaps[si_idx, :, y_coords, x_coords] = flow_vals

        if valids is not None:
            val_sub = valids[target_time_indices[:, None], n_idx[None, :]]  # (K, P)
            tvalids[si_idx, :, y_coords, x_coords] = val_sub.T
        else:
            tvalids[si_idx, :, y_coords, x_coords] = True
        vis_sub = visibs[target_time_indices[:, None], n_idx[None, :]]  # (K, P)
        tvisibs[si_idx, :, y_coords, x_coords] = vis_sub.T
        tis_dynamic[si_idx, y_coords, x_coords] = window_dynamic[:, n_idx].T

        src_pos_map[si_idx, :, y_coords, x_coords] = trajs_3d[si_idx, n_idx, :]
        src_idx_map[si_idx, y_coords, x_coords] = n_idx

        if gt_depths_downsampled is not None:
            gt_depths_downsampled[si_idx, y_coords, x_coords] = trajs_depths[
                si_idx, n_idx
            ]

    if not skip_tmaps:
        tmaps = tmaps.reshape(S, K, 3, H2, W2)
        tmaps = rearrange(tmaps, "t1 k n h w -> t1 h w k n").astype(np.float32)
    else:
        tmaps = None
    tvisibs = rearrange(tvisibs, "t1 k h w -> t1 h w k").astype(bool)
    tis_dynamic = tis_dynamic.astype(bool)
    tvalids = rearrange(tvalids, "t1 k h w -> t1 h w k").astype(bool)
    src_pos_map = rearrange(src_pos_map, "t1 c h w -> t1 h w c").astype(np.float32)
    src_idx_map = src_idx_map.astype(np.int64)

    # if tmaps is not None:
    #     tmaps = tmaps[:, stride // 2 :: stride, stride // 2 :: stride, ...]
    # tvalids = tvalids[:, stride // 2 :: stride, stride // 2 :: stride, ...]
    # tvisibs = tvisibs[:, stride // 2 :: stride, stride // 2 :: stride, ...]
    # tis_dynamic = tis_dynamic[:, stride // 2 :: stride, stride // 2 :: stride, ...]
    # if gt_depths_downsampled is not None:
    #     gt_depths_downsampled = gt_depths_downsampled[
    #         :, stride // 2 :: stride, stride // 2 :: stride, ...
    #     ]
    return tmaps, tvalids, tvisibs, tis_dynamic, gt_depths_downsampled, src_pos_map, src_idx_map


# -----------------------------------------------------------------------------
# Alternative 3D upsampling methods (beyond convex combination)
# -----------------------------------------------------------------------------
# - Bilinear / bicubic: F.interpolate(flow_3d, scale_factor=st, mode='bilinear').
#   No learning; can oversmooth. No scaling of values (correct for 3D).
#
# - DPT-style: multi-scale decoder with ConvTranspose2d and optional fusion of
#   encoder features (see depth_pro/network/encoder.py, Dense Prediction Transformer).
#   Decoder refines low-res prediction to full res; output head = 3 channels (xyz).
#
# - DenseTrack3D-style: transformer-based upsampler with learned kernel weights and
#   attention over context (see third_party/delta/densetrack3d/.../upsample_transformer.py).
#
# - Joint bilateral: upsample 3D using a guide (e.g. RGB) to preserve edges;
#   classic for depth super-resolution; can be applied to xyz maps.
# -----------------------------------------------------------------------------


# def compute_dense_gt_tensor_fast_for_targets(
#     trajs_2d,
#     trajs_3d,
#     visibs,
#     is_dynamic,
#     H,
#     W,
#     target_time_indices,
#     gt_depths_downsampled=None,
#     trajs_depths=None,
#     stride=2,
#     skip_tmaps=False,
# ):
#     print("using old compute_dense_gt_tensor_fast_for_targets")
#     """
#     Like compute_dense_gt_tensor_fast, but only materializes target frames listed in
#     target_time_indices (global indices in [0, S-1]). Output last spatial axes are
#     (H', W', K, ...) instead of (H', W', S, ...) where K = len(target_time_indices).
#     """
#     if gt_depths_downsampled is not None:
#         gt_depths_downsampled = gt_depths_downsampled.copy()

#     S, N, D = trajs_2d.shape
#     target_time_indices = np.asarray(target_time_indices, dtype=np.int64).reshape(-1)
#     K = target_time_indices.size
#     if K == 0:
#         raise ValueError("target_time_indices must be non-empty")
#     if np.any((target_time_indices < 0) | (target_time_indices >= S)):
#         raise ValueError(
#             f"target_time_indices must be in [0, {S - 1}], got {target_time_indices}"
#         )

#     H2, W2 = H // stride, W // stride

#     xy_all = np.floor(trajs_2d.astype(np.float32) / stride).astype(np.int64)
#     xy_all[..., 0] = np.clip(xy_all[..., 0], 0, W2 - 1)
#     xy_all[..., 1] = np.clip(xy_all[..., 1], 0, H2 - 1)

#     vis_mask = visibs > 0
#     si_idx, n_idx = np.where(vis_mask)

#     x_coords = xy_all[si_idx, n_idx, 0]
#     y_coords = xy_all[si_idx, n_idx, 1]

#     if not skip_tmaps:
#         tmaps = np.zeros((S, K * 3, H2, W2))
#     tvalids = np.zeros((S, 1, H2, W2), dtype=bool)
#     tvisibs = np.zeros((S, K, H2, W2), dtype=bool)
#     tis_dynamic = np.zeros((S, H2, W2), dtype=bool)

#     if len(si_idx) > 0:
#         if not skip_tmaps:
#             traj_at_targets = trajs_3d[
#                 target_time_indices[:, None], n_idx[None, :], :
#             ]  # (K, P, 3)
#             traj_vals = traj_at_targets.transpose(0, 2, 1).reshape(K * 3, -1).T
#             tmaps[si_idx, :, y_coords, x_coords] = traj_vals

#         tvalids[si_idx, 0, y_coords, x_coords] = True
#         vis_sub = visibs[target_time_indices[:, None], n_idx[None, :]]  # (K, P)
#         tvisibs[si_idx, :, y_coords, x_coords] = vis_sub.T
#         tis_dynamic[si_idx, y_coords, x_coords] = is_dynamic[n_idx]

#         if gt_depths_downsampled is not None:
#             gt_depths_downsampled[si_idx, y_coords, x_coords] = trajs_depths[
#                 si_idx, n_idx
#             ]

#     if not skip_tmaps:
#         tmaps = tmaps.reshape(S, K, 3, H2, W2)
#         tmaps = rearrange(tmaps, "t1 k n h w -> t1 h w k n").astype(np.float32)
#     else:
#         tmaps = None
#     tvisibs = rearrange(tvisibs, "t1 k h w -> t1 h w k").astype(bool)
#     tis_dynamic = tis_dynamic.astype(bool)
#     tvalids = np.repeat(tvalids.squeeze(1)[..., None], K, axis=-1).astype(bool)

#     # if tmaps is not None:
#     #     tmaps = tmaps[:, stride // 2 :: stride, stride // 2 :: stride, ...]
#     # tvalids = tvalids[:, stride // 2 :: stride, stride // 2 :: stride, ...]
#     # tvisibs = tvisibs[:, stride // 2 :: stride, stride // 2 :: stride, ...]
#     # tis_dynamic = tis_dynamic[:, stride // 2 :: stride, stride // 2 :: stride, ...]
#     # if gt_depths_downsampled is not None:
#     #     gt_depths_downsampled = gt_depths_downsampled[
#     #         :, stride // 2 :: stride, stride // 2 :: stride, ...
#     #     ]
#     return tmaps, tvalids, tvisibs, tis_dynamic, gt_depths_downsampled


def _unvoxelize_window(
    voxel_tensor: torch.Tensor,
    window: DenseTrackWindow,
    *,
    restore_coord_offsets: bool = False,
) -> torch.Tensor:
    """Unvoxelize a (N_vox, ...) tensor using the window's saved p2vs.

    Returns a flat tensor of shape (B * window_end * H * W, ...). When the
    model ran without voxelization, `window.p2vs` is empty and the input is
    returned unchanged.
    """
    if len(window.p2vs) == 0:
        return voxel_tensor
    coord_residuals = getattr(window, "coord_residuals", [])
    if restore_coord_offsets and coord_residuals:
        return unvoxelize_coords_with_residuals(
            voxel_tensor,
            window.p2vs,
            coord_residuals,
            window.p2v_lens,
        )
    return unvoxelize_pc_with_p2v_lists(
        voxel_tensor,
        window.p2vs,
        window.p2v_lens,
    )


# Saturated logits matching metrics' sigmoid > 0.5 threshold, distinguishable
# from the -1000.0 "unwritten" sentinel used in dense reconstruction.
_STATIC_VISIBLE_LOGIT = 10.0
_STATIC_OCCLUDED_LOGIT = -10.0

# Coordinate written where no window ever predicted a (t_src, t_tgt) pair.
# Kept finite rather than NaN because consumers mask invalid positions by
# multiplying with a 0/1 validity mask, which NaN would poison. Far outside the
# normalized scene cube so a forgotten mask is obvious instead of plausible.
TRACK_INVALID_COORD: float = -1000.0

# Visibility / confidence / motion logit written alongside TRACK_INVALID_COORD for
# an entry no window predicted. Real logits live in a normal range, so anything
# at or below this is a sentinel rather than a prediction.
TRACK_INVALID_LOGIT: float = -1000.0

# Threshold separating a real logit from the sentinel above. Consumers testing
# "did any window write this entry" compare against this, not against 0 (which
# is the *visibility* decision boundary and a stricter, different question).
TRACK_VALID_THRESH: float = -100.0


def _stamp_prebirth_invalid(
    src_time: torch.Tensor,
    n_frames: int,
    coords: torch.Tensor,
    *logits: Optional[torch.Tensor],
) -> None:
    """Invalidate, in place, target frames earlier than a track's own frame.

    A window writes source cells ``[0, we)`` against targets ``[ws, we)``, so a
    cell born at frame 5 also receives positions for the earlier frames of its
    birth window. Those are not observations of anything:

    * A **static** voxel has no per-frame prediction there at all --
      :func:`_gather_window_tracks` broadcasts one ``static_coords`` position
      across the whole window, so the "pre-birth position" is just the birth
      position. Splatting it paints a frozen copy of the point wherever the
      object only arrives later.
    * Its **visibility** cannot veto that either: :func:`_window_static_visibs`
      overwrites only ``[:, -1]`` with the model's own logit, so on every other
      frame visibility is a geometric depth test, which a point sitting at its
      own surface position passes trivially.

    Only the dynamic subset gets a genuine per-frame position and visibility
    there, and even then the point may not have existed yet. Rather than leave
    callers to rediscover this, these entries carry the same sentinels an
    unwritten entry does.

    ``coords`` is ``(N, T, 3)``; each tensor in ``logits`` is ``(N, T)``.
    """
    pre = torch.arange(n_frames, device=coords.device).view(1, -1) < src_time.to(
        coords.device
    ).view(-1, 1)
    coords[pre] = TRACK_INVALID_COORD
    for tensor in logits:
        if tensor is not None:
            tensor[pre] = TRACK_INVALID_LOGIT


def _transform_points_w2c(points: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """Apply world-to-camera transforms. ``points`` is (F, ..., 3), ``extrinsics`` is (F, 4, 4)."""
    lead = points.shape[0]
    flat = points.reshape(lead, -1, 3)
    rot = extrinsics[:, :3, :3]
    trans = extrinsics[:, :3, 3]
    out = torch.einsum("tij,tnj->tni", rot, flat) + trans[:, None, :]
    return out.reshape(points.shape)


def _static_visibility_logits(
    points: torch.Tensor,
    surface: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    rel_thresh: float = 0.15,
    min_depth: float = 1e-3,
) -> torch.Tensor:
    """Depth-test static world points against a per-frame predicted pointmap.

    Args:
        points: ``(N, 3)`` world positions, constant over the frames tested.
        surface: ``(F, Hf, Wf, 3)`` world-space pointmap (the dense-track diagonal).
        extrinsics: ``(F, 4, 4)`` world -> camera.
        intrinsics: ``(F, 3, 3)`` at ``(Hf, Wf)`` resolution.
        rel_thresh: relative depth margin for occlusion (predicted buffers are
            noisy; 0.15 matches the value validated in ``visibility_utils``).
        min_depth: minimum camera-space depth treated as valid.

    Returns:
        ``(N, F)`` saturated visibility logits (``+/-10``).
    """
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must be (N, 3), got {tuple(points.shape)}")
    if surface.ndim != 4 or surface.shape[-1] != 3:
        raise ValueError(f"surface must be (F, Hf, Wf, 3), got {tuple(surface.shape)}")
    F, Hf, Wf, _ = surface.shape
    N = points.shape[0]
    if extrinsics.shape[0] != F or intrinsics.shape[0] != F:
        raise ValueError(
            "surface, extrinsics and intrinsics must agree on F: "
            f"{F}, {extrinsics.shape[0]}, {intrinsics.shape[0]}"
        )

    extrinsics = extrinsics.to(torch.float32)
    intrinsics = intrinsics.to(torch.float32)
    # Expand static points across frames: (F, N, 3).
    tracks = points.to(torch.float32)[None].expand(F, N, 3)
    surf_cam = _transform_points_w2c(surface.to(torch.float32), extrinsics)
    trk_cam = _transform_points_w2c(tracks, extrinsics)

    trk_z = trk_cam[..., 2]
    z_safe = trk_z.clamp_min(min_depth)
    fx = intrinsics[:, 0, 0:1]
    fy = intrinsics[:, 1, 1:2]
    cx = intrinsics[:, 0, 2:3]
    cy = intrinsics[:, 1, 2:3]
    u = fx * (trk_cam[..., 0] / z_safe) + cx
    v = fy * (trk_cam[..., 1] / z_safe) + cy

    in_front = torch.isfinite(trk_z) & (trk_z > min_depth)
    in_bounds = (
        torch.isfinite(u)
        & torch.isfinite(v)
        & (u >= 0)
        & (u <= Wf - 1)
        & (v >= 0)
        & (v <= Hf - 1)
    )

    u0 = u.nan_to_num(0.0).floor().clamp(0, Wf - 1).long()
    v0 = v.nan_to_num(0.0).floor().clamp(0, Hf - 1).long()
    u1 = (u0 + 1).clamp(0, Wf - 1)
    v1 = (v0 + 1).clamp(0, Hf - 1)

    surf_z_flat = surf_cam[..., 2].reshape(F, Hf * Wf)
    neighbours = torch.stack(
        [
            torch.gather(surf_z_flat, 1, v0 * Wf + u0),
            torch.gather(surf_z_flat, 1, v0 * Wf + u1),
            torch.gather(surf_z_flat, 1, v1 * Wf + u0),
            torch.gather(surf_z_flat, 1, v1 * Wf + u1),
        ],
        dim=0,
    )  # (4, F, N)
    valid_surface = torch.isfinite(neighbours) & (neighbours > min_depth)
    # Farthest valid neighbour is the most permissive occluder choice (TAP-Vid).
    surf_far = torch.where(valid_surface, neighbours, torch.zeros_like(neighbours))
    surf_far = surf_far.max(dim=0).values
    has_surface = valid_surface.any(dim=0)

    occluded = has_surface & (surf_far < trk_z * (1.0 - rel_thresh))
    visible = in_front & in_bounds & ~occluded

    logits = torch.where(
        visible,
        torch.full_like(trk_z, _STATIC_VISIBLE_LOGIT),
        torch.full_like(trk_z, _STATIC_OCCLUDED_LOGIT),
    )
    return logits.transpose(0, 1).contiguous()  # (N, F)


def _window_diagonal_pointmap(window: DenseTrackWindow, cells: int) -> torch.Tensor:
    """Per-frame predicted pointmap for a window's own target frames.

    For each newly-born source cell in ``[window_start, window_end)``, reads
    the model's predicted world position at that cell's own birth frame. That
    is exactly ``dense[b, t, y, x, t]`` — the diagonal of the dense track
    tensor — and is the free depth buffer for static-point occlusion tests.

    Returns:
        ``(T_win, Hf*Wf flattened as cells, 3)`` then reshaped by the caller,
        or more precisely ``(T_win * cells, 3)`` ordered ``(t, h, w)``.
    """
    ws, we = window.window_start, window.window_end
    device = window.static_coords.device
    dtype = window.static_coords.dtype
    vox_new = _window_new_cell_voxels(window, cells)
    local_t = torch.arange(we - ws, device=device).repeat_interleave(cells)
    sd_row = torch.full_like(window.sd_mask, -1, dtype=torch.long)
    sd_row[window.sd_mask] = torch.arange(int(window.sd_mask.sum()), device=device)
    row = sd_row[vox_new]
    dyn = row >= 0
    pos = window.static_coords[vox_new].to(dtype=dtype).clone()
    pos[dyn] = window.sd_trajs[row[dyn], local_t[dyn]].to(pos.dtype)
    return pos  # ((we - ws) * cells, 3)


def _window_static_visibs(
    window: DenseTrackWindow,
    surface: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    points: Optional[torch.Tensor] = None,
    static_visibs: Optional[torch.Tensor] = None,
    rel_thresh: float = 0.15,
) -> torch.Tensor:
    """Per-frame static visibility for a window's voxels.

    Runs the depth test against ``surface`` for every frame in the window, then
    overwrites the last column with the model's own visibility logits (which
    refer to the window's last frame).

    Args:
        window: dense-track snapshot.
        surface: ``(T_win, Hf, Wf, 3)`` world-space pointmap for the window.
        extrinsics: ``(T_win, 4, 4)``.
        intrinsics: ``(T_win, 3, 3)`` at ``(Hf, Wf)``.
        points: optional ``(n, 3)`` subset to test; defaults to all
            ``window.static_coords``.
        static_visibs: optional ``(n,)`` model logits matching ``points``;
            defaults to ``window.static_visibs`` when ``points`` is None.
        rel_thresh: relative depth margin for occlusion.

    Returns:
        ``(n, T_win)`` visibility logits.
    """
    pts = window.static_coords if points is None else points
    model_vis = window.static_visibs if static_visibs is None else static_visibs
    logits = _static_visibility_logits(
        pts,
        surface,
        extrinsics,
        intrinsics,
        rel_thresh=rel_thresh,
    )
    if logits.shape[1] > 0:
        logits[:, -1] = model_vis.to(device=logits.device, dtype=logits.dtype)
    return logits


def reconstruct_dense_tracks_3d(
    pred: "Prediction",
    *,
    return_visibs_confs: bool = False,
    dynamic_logits_by_target: bool = True,
    static_visibility: bool = True,
    mask_prebirth: bool = True,
):
    """Reconstruct full dense 3D tracks from per-window snapshots.

    At eval time, `streaming_forward_simple_regression_v6_windowed_overlap`
    stores one `DenseTrackWindow` per streaming window in `pred.dense_track_3d`.
    Each snapshot contains the static-dynamic trajectory predictions for the
    dynamic subset of static scene points inside that window, plus the
    window-local predicted static 3D position for every static scene point.

    Semantics of the output `dense`:
        dense[b, t_src, h, w, t_tgt, :] = predicted 3D world position at
        target frame `t_tgt` of the pixel observed at (t_src, h, w).

    A window writes source cells [0, we) against targets [ws, we), so a cell
    also gets positions for the targets *before* its own frame that fall inside
    its birth window. Targets earlier than that window are never written and
    hold `TRACK_INVALID_COORD`; the matching `dense_visibs` entries are -1000.0.
    By default (`mask_prebirth`) the in-window pre-birth entries are stamped with
    those same sentinels, so `t_tgt < t_src` is uniformly invalid and no caller
    can splat a frozen static point onto a frame it was never observed in.

    Args:
        pred: Prediction populated at eval time. Requires `pred.dense_track_3d`
            set, plus grid shape from either `pred.coords` (B, T, H, W, 3) or
            the cheap `pred.dense_shape` (B, T, H, W) fallback.
        return_visibs_confs: if True, also return dense visibility tensor
            (B, T, H, W, T), a dense confidence tensor (B, T, H, W, T), and
            a dense static/dynamic motion-logit tensor — the raw pre-threshold
            values the model uses to build `sd_mask` (threshold > 0). Entries
            not populated by a dense tracking window are set to -1000.0. Later
            overlapping windows overwrite earlier ones.
        dynamic_logits_by_target: if True, return dense logits with shape
            (B, T, H, W, T), where the last axis is target frame. This preserves
            window-local dynamic/static decisions for visualization.
        static_visibility: if True (default) and `pred.dense_extrinsics` is
            set, depth-test static points against the per-frame predicted
            pointmap instead of broadcasting `static_visibs` across the window.
            Falls back to the broadcast when cameras are missing (legacy
            snapshots).
        mask_prebirth: if True (default), invalidate every `t_tgt < t_src` entry
            (see :func:`_stamp_prebirth_invalid`). Pass False only to inspect the
            raw window output; those entries are not observations and splatting
            them produces false positives far from the object.

    Returns:
        dense: (B, T, H, W, T, 3)
        (optionally) dense_visibs: (B, T, H, W, T), dense_confs: (B, T, H, W, T),
        dense_logits: (B, T, H, W, T) by default (``dynamic_logits_by_target``
            defaults to True), or (B, T, H, W) when it is False. In the 4D
            layout ``[..., t_tgt]`` is the score from the window containing
            ``t_tgt``, so the per-window decisions are preserved. The 3D layout
            instead keeps only the *last* window's score for every cell, since
            each window overwrites the whole ``[0, we)`` source range.

    Notes:
        Windows are applied in order, so later (overlapping) windows
        overwrite earlier ones for the (t_src, t_tgt) pairs they share,
        keeping the most recent prediction per pair.
    """
    assert pred.dense_track_3d is not None and len(pred.dense_track_3d) > 0, (
        "pred.dense_track_3d is empty; run eval with dense_tracking_at_eval=True"
    )

    assert pred.dense_shape is not None, (
        "reconstruct_dense_tracks_3d needs either pred.coords (B, T, H, W, 3) "
        "or pred.dense_shape (B, T, H, W)"
    )
    B, T, H, W = pred.dense_shape
    _ref = pred.dense_track_3d[0].static_coords
    device = _ref.device
    dtype = _ref.dtype
    cells = H * W

    use_geom_vis = (
        static_visibility
        and return_visibs_confs
        and pred.dense_extrinsics is not None
        and pred.dense_intrinsics is not None
    )
    if use_geom_vis:
        assert B == 1, (
            "per-frame static visibility assumes B == 1, got "
            f"{B}; pass static_visibility=False to fall back to broadcast"
        )

    dense = torch.full(
        (B, T, H, W, T, 3), TRACK_INVALID_COORD, device=device, dtype=dtype
    )

    dense_visibs = None
    dense_confs = None
    dense_logits = None
    if return_visibs_confs:
        dense_visibs = torch.full(
            (B, T, H, W, T), -1000.0, device=device, dtype=dtype
        )
        dense_confs = torch.full(
            (B, T, H, W, T), -1000.0, device=device, dtype=dtype
        )
        logits_shape = (B, T, H, W, T) if dynamic_logits_by_target else (B, T, H, W)
        dense_logits = torch.full(logits_shape, -1000.0, device=device, dtype=dtype)

    for window in pred.dense_track_3d:
        ws, we = window.window_start, window.window_end
        T_win = we - ws
        N_vox = window.sd_mask.shape[0]

        trajs_full = window.static_coords.to(device=device, dtype=dtype)[:, None, :]
        trajs_full = trajs_full.expand(N_vox, T_win, 3).clone()
        trajs_full[window.sd_mask] = window.sd_trajs.to(device=device, dtype=dtype)
        trajs_dense = _unvoxelize_window(
            trajs_full, window, restore_coord_offsets=True
        )
        trajs_dense = rearrange(
            trajs_dense,
            "(b t h w) tw c -> b t h w tw c",
            b=B, t=we, h=H, w=W,
        )

        dense[:, :we, :, :, ws:we, :] = trajs_dense

        if return_visibs_confs:
            if use_geom_vis:
                surface = _window_diagonal_pointmap(window, cells).reshape(
                    T_win, H, W, 3
                )
                ext = pred.dense_extrinsics[0, ws:we].to(device=device)
                intr = pred.dense_intrinsics[0, ws:we].to(device=device)
                visibs_full = _window_static_visibs(window, surface, ext, intr)
                visibs_full = visibs_full.to(device=device, dtype=dtype)
            else:
                visibs_full = window.static_visibs.to(device=device, dtype=dtype)[:, None]
                visibs_full = visibs_full.expand(N_vox, T_win).clone()
            visibs_full[window.sd_mask] = window.sd_visibs.to(
                device=device, dtype=dtype
            )

            visibs_dense = _unvoxelize_window(visibs_full, window)
            visibs_dense = rearrange(
                visibs_dense,
                "(b t h w) tw -> b t h w tw",
                b=B, t=we, h=H, w=W,
            )

            confs_full = window.static_confs.to(device=device, dtype=dtype)[:, None]
            confs_full = confs_full.expand(N_vox, T_win).clone()
            confs_full[window.sd_mask] = window.sd_confs.to(
                device=device, dtype=dtype
            )
            confs_dense = _unvoxelize_window(confs_full, window)
            confs_dense = rearrange(
                confs_dense,
                "(b t h w) tw -> b t h w tw",
                b=B, t=we, h=H, w=W,
            )

            dense_visibs[:, :we, :, :, ws:we] = visibs_dense
            dense_confs[:, :we, :, :, ws:we] = confs_dense

            logits_full = window.sd_logits.to(
                device=device, dtype=dtype
            ).unsqueeze(-1)  # (N_vox, 1)
            logits_dense = _unvoxelize_window(logits_full, window)
            logits_dense = rearrange(
                logits_dense.squeeze(-1),
                "(b t h w) -> b t h w",
                b=B, t=we, h=H, w=W,
            )
            if dynamic_logits_by_target:
                dense_logits[:, :we, :, :, ws:we] = logits_dense[..., None].expand(
                    B, we, H, W, T_win
                )
            else:
                dense_logits[:, :we] = logits_dense

    if mask_prebirth:
        # (1, T_src, 1, 1, T_tgt), broadcast over the grid rather than expanded.
        pre = torch.arange(T, device=device).view(1, 1, 1, 1, T) < torch.arange(
            T, device=device
        ).view(1, T, 1, 1, 1)
        dense.masked_fill_(pre.unsqueeze(-1), TRACK_INVALID_COORD)
        for tensor in (dense_visibs, dense_confs):
            if tensor is not None:
                tensor.masked_fill_(pre, TRACK_INVALID_LOGIT)
        # The 3D logits layout has no target axis, so there is nothing to mask.
        if dense_logits is not None and dynamic_logits_by_target:
            dense_logits.masked_fill_(pre, TRACK_INVALID_LOGIT)

    if return_visibs_confs:
        return dense, dense_visibs, dense_confs, dense_logits
    return dense


class CompressedTracks(NamedTuple):
    """A flat track set read straight out of the model's per-window snapshots.

    Same content the visualizers consume after they flatten and subsample a
    reconstructed dense tensor, but obtained without ever building one.
    """

    coords: torch.Tensor  # (N, T, 3) world position per target frame
    visibs: torch.Tensor  # (N, T) raw visibility logits, -1000 where unwritten
    dyn_logits: torch.Tensor  # (N, T) raw static/dynamic logits
    src_time: torch.Tensor  # (N,) birth frame
    src_y: torch.Tensor  # (N,) birth row on the dense grid
    src_x: torch.Tensor  # (N,) birth column on the dense grid

    def to(self, device) -> "CompressedTracks":
        return CompressedTracks(*(t.to(device) for t in self))


def _window_new_cell_voxels(window, cells: int):
    """Voxel id of each source cell born in this window, ordered (t, h, w)."""
    if len(window.p2vs) == 0:
        ws, we = window.window_start, window.window_end
        return torch.arange(
            ws * cells, we * cells, device=window.static_coords.device
        )
    n_prev = window.p2v_lens[-2] if len(window.p2v_lens) >= 2 else 0
    return window.p2vs[-1][n_prev:]


def _follow_cells(windows, keys: torch.Tensor, cells: int):
    """Track a fixed set of source cells forward through the per-window p2v chain.

    ``window.p2vs[-1]`` maps ``[voxels of the previous window ; this window's new
    cells, ordered (t, h, w)]`` onto this window's voxels, so a cell's voxel is
    obtained by indexing with its previous voxel (already-born cells) or with its
    offset into the new-cell segment (cells born now). This is O(len(keys)) per
    window and independent of T.

    Following cells forward, rather than trying to keep a canonical id per voxel,
    is what makes this match the dense reconstruction: the scene is re-voxelized
    on *predicted* coordinates every window, so voxels merge often, and two cells
    that merge should simply share a trajectory from then on.

    Args:
        keys: (n,) flat source-cell ids, ``(t * Hf + y) * Wf + x``.

    Yields:
        ``(window, vox)`` where ``vox`` is (n,) with -1 for not-yet-born cells.
    """
    device = keys.device
    vox = torch.full_like(keys, -1)
    for w in windows:
        ws, we = w.window_start, w.window_end
        if len(w.p2vs) == 0:
            # Not voxelized: scene points are the cells themselves, in order.
            yield w, torch.where(keys < we * cells, keys, torch.full_like(keys, -1))
            continue

        p2v = w.p2vs[-1].to(device)
        n_prev = w.p2v_lens[-2] if len(w.p2v_lens) >= 2 else 0
        assert p2v.numel() - n_prev == (we - ws) * cells, (
            f"window [{ws},{we}) expects {(we - ws) * cells} new cells, "
            f"got {p2v.numel() - n_prev}"
        )
        born_now = (keys >= ws * cells) & (keys < we * cells)
        carried = vox >= 0

        nxt = torch.full_like(vox, -1)
        nxt[carried] = p2v[vox[carried]]
        nxt[born_now] = p2v[n_prev + (keys[born_now] - ws * cells)]
        vox = nxt
        yield w, vox


def _gather_window_tracks(
    window,
    vox: torch.Tensor,
    *,
    static_visibs_tw: Optional[torch.Tensor] = None,
):
    """Position/visibility over a window's target frames for the given voxels.

    Static voxels hold one predicted position for the whole window; dynamic ones
    are replaced by their trajectory.

    Args:
        static_visibs_tw: optional ``(n, T_win)`` precomputed static visibility
            logits (e.g. from the geometric depth test). When None, broadcasts
            ``window.static_visibs[vox]`` across the window (legacy behaviour).
    """
    t_win = window.window_end - window.window_start
    n = vox.shape[0]

    sd_row = torch.full_like(window.sd_mask, -1, dtype=torch.long)
    sd_row[window.sd_mask] = torch.arange(
        int(window.sd_mask.sum()), device=vox.device
    )
    row = sd_row[vox]
    dyn = row >= 0

    coords = window.static_coords[vox][:, None, :].expand(n, t_win, 3).clone()
    coords[dyn] = window.sd_trajs[row[dyn]].to(coords.dtype)

    if static_visibs_tw is None:
        visibs = window.static_visibs[vox][:, None].expand(n, t_win).clone()
    else:
        if static_visibs_tw.shape != (n, t_win):
            raise ValueError(
                f"static_visibs_tw must have shape {(n, t_win)}, "
                f"got {tuple(static_visibs_tw.shape)}"
            )
        visibs = static_visibs_tw.to(dtype=window.static_visibs.dtype).clone()
    visibs[dyn] = window.sd_visibs[row[dyn]].to(visibs.dtype)
    return coords, visibs


def _subsample_independent_caps(
    ever_dyn: torch.Tensor,
    *,
    max_points: int,
    max_dynamic_points: int,
    seed: int = 0,
) -> torch.Tensor:
    """Independent ever-dynamic / never-dynamic caps used by ``--long_video``.

    ``ever_dyn`` is ``(N,)`` bool. Returns sorted keep indices on the same device.
    This is the subsample inside :func:`extract_compressed_tracks` that
    ``demo.py --long_video`` runs before gathering tracks.
    """
    device = ever_dyn.device
    dyn_pool = torch.nonzero(ever_dyn, as_tuple=False).view(-1)
    stat_pool = torch.nonzero(~ever_dyn, as_tuple=False).view(-1)
    n_dyn_all = int(dyn_pool.numel())
    n_stat_all = int(stat_pool.numel())
    logger.info(
        "extract_compressed_tracks subsample: dropped %d/%d ever-dynamic "
        "(cap %d) and %d/%d never-dynamic (cap %d); kept %d",
        max(0, n_dyn_all - max_dynamic_points),
        n_dyn_all,
        max_dynamic_points,
        max(0, n_stat_all - max_points),
        n_stat_all,
        max_points,
        min(n_dyn_all, max_dynamic_points) + min(n_stat_all, max_points),
    )
    g = torch.Generator(device="cpu").manual_seed(seed)
    if dyn_pool.numel() > max_dynamic_points:
        pick = torch.randperm(dyn_pool.numel(), generator=g)[:max_dynamic_points]
        dyn_pool = dyn_pool[pick.to(device)]
    if stat_pool.numel() > max_points:
        pick = torch.randperm(stat_pool.numel(), generator=g)[:max_points]
        stat_pool = stat_pool[pick.to(device)]
    return torch.cat([dyn_pool, stat_pool]).sort().values


def extract_compressed_tracks(
    pred: "Prediction",
    *,
    max_points: int = 60000,
    max_dynamic_points: int = 20000,
    dynamic_thresh: float = 0.5,
    dense_source_mask=None,
    voxel_size: float = 0.02,
    out_device="cpu",
    seed: int = 0,
    static_visibility: bool = True,
    mask_prebirth: bool = True,
) -> CompressedTracks:
    """Flatten the model's per-window snapshots into a capped set of tracks.

    `reconstruct_dense_tracks_3d` materializes (B, T, H, W, T, 3) and the
    visualizers then immediately collapse it to one representative source cell
    per birth voxel and subsample. This produces that same track set directly,
    which is the only way long sequences are tractable: the dense tensor is
    ~330 TB at T=1500, while the output here is N_keep * T * 3 floats.

    The selection rule mirrors the dense visualizers exactly (voxelize birth
    positions at ``voxel_size``, keep the earliest source cell per voxel), so the
    result is the same track set they would have produced.

    Peak cost is O(T * Hf * Wf) for the birth-position pass plus the output.

    Args:
        pred: eval-time Prediction with `dense_track_3d` and `dense_shape` set.
        max_points: cap on never-dynamic tracks.
        max_dynamic_points: cap on ever-dynamic tracks.
        dynamic_thresh: threshold on sigmoid(sd_logits) for "ever dynamic".
        dense_source_mask: optional (T, Hf, Wf) bool over source cells; a track
            is dropped when its birth cell is masked out.
        voxel_size: birth-position voxel size for the one-track-per-voxel collapse.
        out_device: where the returned tensors live.
        seed: subsample seed.
        static_visibility: if True (default) and `pred.dense_extrinsics` is set,
            depth-test static points against the per-frame predicted pointmap
            instead of broadcasting `static_visibs` across each window.
        mask_prebirth: if True (default), invalidate every `t_tgt < t_src` entry
            (see :func:`_stamp_prebirth_invalid`).
    """
    from ..datasets.utils.geometry import voxelization

    windows = pred.dense_track_3d
    assert windows is not None and len(windows) > 0, (
        "pred.dense_track_3d is empty; run eval with dense_tracking_at_eval=True"
    )
    assert pred.dense_shape is not None, "extract_compressed_tracks needs pred.dense_shape"

    B, T, Hf, Wf = pred.dense_shape
    assert B == 1, f"extract_compressed_tracks assumes B == 1, got {B}"
    cells = Hf * Wf
    ref = windows[0].static_coords
    device, dtype = ref.device, ref.dtype

    use_geom_vis = (
        static_visibility
        and pred.dense_extrinsics is not None
        and pred.dense_intrinsics is not None
    )

    # Pass 1: birth position of every source cell, i.e. where the model puts it
    # at its own frame. This is also the free per-frame pointmap (the dense
    # track diagonal) used as the depth buffer for static visibility.
    birth_pos = torch.zeros((T * cells, 3), dtype=dtype, device=device)
    for w in windows:
        ws, we = w.window_start, w.window_end
        birth_pos[ws * cells : we * cells] = _window_diagonal_pointmap(w, cells)

    # Keep the full unmasked pointmap for geometric visibility before the
    # source-mask / voxel-collapse filters thin birth_pos down.
    surface_full = (
        birth_pos.reshape(T, Hf, Wf, 3) if use_geom_vis else None
    )

    keys = torch.arange(T * cells, device=device)
    birth_for_voxel = birth_pos
    if dense_source_mask is not None:
        sm = torch.as_tensor(dense_source_mask).detach().bool().to(device)
        if sm.shape != (T, Hf, Wf):
            raise ValueError(
                f"dense_source_mask must have shape {(T, Hf, Wf)}, got {tuple(sm.shape)}"
            )
        sel = sm.reshape(-1)
        keys = keys[sel]
        birth_for_voxel = birth_pos[sel]

    # Pass 2: one representative cell per birth voxel. Cell keys are ascending in
    # (t, y, x) order, so a stable sort by voxel leaves the earliest source cell
    # of each voxel first -- the same rule the dense visualizers use.
    if keys.numel() > 0:
        p2v = voxelization(birth_for_voxel[None].contiguous(), voxel_size).squeeze(0)
        order = torch.argsort(p2v, stable=True)
        p2v_sorted = p2v[order]
        first = torch.ones_like(p2v_sorted, dtype=torch.bool)
        first[1:] = p2v_sorted[1:] != p2v_sorted[:-1]
        keys = keys[order[first]].sort().values
        del p2v, order, p2v_sorted, first
    del birth_pos, birth_for_voxel

    # Pass 3a: which representatives are ever dynamic. Only needs the voxel id
    # per window, so it costs O(len(keys)) and no trajectory storage.
    ever_dyn = torch.zeros(keys.numel(), dtype=torch.bool, device=device)
    for w, vox in _follow_cells(windows, keys, cells):
        live = vox >= 0
        ever_dyn[live] |= (
            w.sd_logits[vox[live]].float().sigmoid() > dynamic_thresh
        )

    keys = keys[
        _subsample_independent_caps(
            ever_dyn,
            max_points=max_points,
            max_dynamic_points=max_dynamic_points,
            seed=seed,
        )
    ]
    n_keep = keys.numel()
    del ever_dyn

    coords = torch.full(
        (n_keep, T, 3), TRACK_INVALID_COORD, dtype=dtype, device=out_device
    )
    visibs = torch.full((n_keep, T), -1000.0, dtype=dtype, device=out_device)
    dyn_logits = torch.full((n_keep, T), -1000.0, dtype=dtype, device=out_device)

    # Pass 3b: fill each window's own target frames. Windows tile [0, T) exactly,
    # so every frame is written once and nothing needs to be reconciled.
    for w, vox in _follow_cells(windows, keys, cells):
        ws, we = w.window_start, w.window_end
        live = vox >= 0
        if not bool(live.any()):
            continue
        rows = torch.nonzero(live, as_tuple=False).view(-1).to(out_device)
        vox_live = vox[live]
        static_visibs_tw = None
        if use_geom_vis:
            surface = surface_full[ws:we]
            ext = pred.dense_extrinsics[0, ws:we].to(device=device)
            intr = pred.dense_intrinsics[0, ws:we].to(device=device)
            static_visibs_tw = _window_static_visibs(
                w,
                surface,
                ext,
                intr,
                points=w.static_coords[vox_live],
                static_visibs=w.static_visibs[vox_live],
            )
        w_coords, w_visibs = _gather_window_tracks(
            w, vox_live, static_visibs_tw=static_visibs_tw
        )
        coords[rows, ws:we] = w_coords.to(out_device)
        visibs[rows, ws:we] = w_visibs.to(out_device)
        dyn_logits[rows, ws:we] = (
            w.sd_logits[vox_live][:, None].expand(-1, we - ws).to(out_device, dtype)
        )
        del w_coords, w_visibs, vox_live, static_visibs_tw

    keys = keys.to(out_device)
    if mask_prebirth:
        _stamp_prebirth_invalid(
            torch.div(keys, cells, rounding_mode="floor"),
            T,
            coords,
            visibs,
            dyn_logits,
        )
    return CompressedTracks(
        coords=coords,
        visibs=visibs,
        dyn_logits=dyn_logits,
        src_time=torch.div(keys, cells, rounding_mode="floor"),
        src_y=torch.div(keys % cells, Wf, rounding_mode="floor"),
        src_x=keys % Wf,
    )


class SceneTokens(NamedTuple):
    """One track per distinct scene point, plus an exact cell -> token map.

    Where :class:`CompressedTracks` picks its track set by voxelizing birth
    positions and throwing the rest away, this one takes the grouping straight
    from the tracker: two cells share a token exactly when the tracker's own
    per-window re-voxelization merged them. Because every cell was born as some
    voxel, ``cell_to_token`` covers the whole grid by construction, with no
    projection and no possibility of a hole.
    """

    coords: torch.Tensor  # (N, T, 3) world position per target frame
    visibs: torch.Tensor  # (N, T) raw visibility logits, -1000 where unwritten
    dyn_logits: torch.Tensor  # (N, T) per-window static/dynamic logits
    src_time: torch.Tensor  # (N,) source frame of the token's earliest cell
    src_y: torch.Tensor  # (N,) source row of that cell
    src_x: torch.Tensor  # (N,) source column of that cell
    # (T, Hf, Wf) token id of every dense-grid cell, or (n,) when built for a
    # subset of cells via ``keys``.
    cell_to_token: torch.Tensor

    def to(self, device) -> "SceneTokens":
        return SceneTokens(*(t.to(device) for t in self))

    def to_dense(self, mask_prebirth: bool = True):
        """Expand to per-pixel tracks; see :func:`scene_tokens_to_dense`."""
        return scene_tokens_to_dense(self, mask_prebirth=mask_prebirth)


def merge_history_cell_to_node(
    pred: "Prediction", keys: torch.Tensor
) -> torch.Tensor:
    """Final voxel of each source cell, following ids only (no positions).

    Walks the per-window ``p2v`` chain to the last window. Two cells come out
    with the same value exactly when the tracker merged them at some window
    boundary and they stayed merged, which is the tracker's own answer to "are
    these the same physical point".

    Costs ``O(len(keys))`` per window with no position buffer at all, which is
    what keeps the lookup table linear in T rather than quadratic.
    """
    windows = pred.dense_track_3d
    assert windows is not None and len(windows) > 0, "pred.dense_track_3d is empty"
    assert pred.dense_shape is not None, "needs pred.dense_shape"
    _B, _T, Hf, Wf = pred.dense_shape
    cells = Hf * Wf

    vox = None
    for _w, vox in _follow_cells(windows, keys, cells):
        pass
    assert vox is not None
    assert bool((vox >= 0).all()), (
        "some cells are unborn after the final window; windows must tile [0, T)"
    )
    return vox


def extract_merge_history_tokens(
    pred: "Prediction",
    *,
    keys: Optional[torch.Tensor] = None,
    out_device="cpu",
    static_visibility: bool = True,
    mask_prebirth: bool = True,
) -> SceneTokens:
    """Build one token per distinct scene point from the tracker's merge chain.

    Groups the requested source cells by the voxel they end up sharing in the
    final window, then keeps the earliest-born cell of each group as that
    group's representative. The representative's trajectory necessarily spans
    every frame its group members appear in, since it was born no later than
    any of them, so ``cell_to_token`` never points at a track that has nothing
    to say at the cell's own frame.

    Args:
        pred: eval-time Prediction with `dense_track_3d` and `dense_shape` set.
        keys: optional ``(n,)`` flat dense-grid cell ids ``(t * Hf + y) * Wf + x``
            to build tokens for. Defaults to every cell, in which case
            ``cell_to_token`` is ``(T, Hf, Wf)`` and the tokens can be expanded
            back to dense tracks with :meth:`SceneTokens.to_dense`. Pass a subset
            to build tokens at a coarser grid than the tracker ran at.
        out_device: where the returned tensors live.
        static_visibility: if True (default) and `pred.dense_extrinsics` is set,
            depth-test static points against the per-frame predicted pointmap
            instead of broadcasting `static_visibs` across each window.
        mask_prebirth: if True (default), invalidate every `t_tgt < t_src` entry
            (see :func:`_stamp_prebirth_invalid`).
    """
    windows = pred.dense_track_3d
    assert windows is not None and len(windows) > 0, (
        "pred.dense_track_3d is empty; run eval with dense_tracking_at_eval=True"
    )
    assert pred.dense_shape is not None, (
        "extract_merge_history_tokens needs pred.dense_shape"
    )
    B, T, Hf, Wf = pred.dense_shape
    assert B == 1, f"extract_merge_history_tokens assumes B == 1, got {B}"
    cells = Hf * Wf
    ref = windows[0].static_coords
    device, dtype = ref.device, ref.dtype

    full_grid = keys is None
    if full_grid:
        keys = torch.arange(T * cells, device=device)
    else:
        keys = keys.to(device=device, dtype=torch.long)
        assert bool((keys[1:] > keys[:-1]).all()), "keys must be strictly ascending"

    # Group cells by the point they ended up as, and elect the earliest-born
    # cell of each group. `keys` is ascending in (t, y, x), so the first
    # occurrence within a stable sort by group is that group's earliest cell.
    vox_final = merge_history_cell_to_node(pred, keys)
    _uniq, inv = torch.unique(vox_final, return_inverse=True)
    order = torch.argsort(inv, stable=True)
    inv_sorted = inv[order]
    is_first = torch.ones_like(inv_sorted, dtype=torch.bool)
    is_first[1:] = inv_sorted[1:] != inv_sorted[:-1]
    rep_pos = order[is_first]  # positions into `keys`, one per group

    # Order tokens by their representative cell so token 0 is the earliest-born
    # point, and rewrite the cell map into that order.
    rep_pos, group_of_token = rep_pos.sort()
    token_of_group = torch.empty_like(group_of_token)
    token_of_group[inv[rep_pos]] = torch.arange(
        rep_pos.numel(), device=device
    )
    cell_to_token = token_of_group[inv]
    if full_grid:
        cell_to_token = cell_to_token.view(T, Hf, Wf)
    rep_keys = keys[rep_pos]

    # The representative is its group's earliest-born cell, so masking at the
    # representative's own frame never clips a frame a group member needs.
    tokens = _gather_tracks_for_keys(
        pred,
        rep_keys,
        out_device=out_device,
        static_visibility=static_visibility,
        dtype=dtype,
        mask_prebirth=mask_prebirth,
    )
    rep_keys_out = rep_keys.to(out_device)
    return SceneTokens(
        coords=tokens[0],
        visibs=tokens[1],
        dyn_logits=tokens[2],
        src_time=torch.div(rep_keys_out, cells, rounding_mode="floor"),
        src_y=torch.div(rep_keys_out % cells, Wf, rounding_mode="floor"),
        src_x=rep_keys_out % Wf,
        cell_to_token=cell_to_token.to(out_device),
    )


def scene_tokens_to_dense(tokens: SceneTokens, mask_prebirth: bool = True):
    """Expand per-point tracks back to one track per dense-grid cell.

    Every cell takes the track of the token the tracker merged it into, so
    ``dense[0, t, y, x]`` is ``tokens.coords[tokens.cell_to_token[t, y, x]]``.
    Needs only the tokens (e.g. a saved ``tracks.pt``), not the model's
    per-window snapshots. The result has the layout of
    :func:`reconstruct_dense_tracks_3d` and grows with T squared.

    Two cells that the tracker merged only at a later window boundary share the
    earlier-born cell's track for the frames before that merge, where the exact
    dense reconstruction would keep their separate pre-merge positions.

    Args:
        tokens: from :func:`extract_merge_history_tokens` over the full grid.
        mask_prebirth: if True (default), invalidate every ``t_tgt < t_src``
            entry like :func:`reconstruct_dense_tracks_3d` does.

    Returns:
        ``(coords, visibs, dyn_logits)`` of shape ``(1, T, Hf, Wf, T, 3)``,
        ``(1, T, Hf, Wf, T)`` and ``(1, T, Hf, Wf, T)``.
    """
    assert tokens.cell_to_token.dim() == 3, (
        "scene_tokens_to_dense needs tokens built over the full grid (keys=None)"
    )
    T, Hf, Wf = tokens.cell_to_token.shape
    idx = tokens.cell_to_token.reshape(-1).to(tokens.coords.device)
    coords = tokens.coords[idx]
    visibs = tokens.visibs[idx]
    dyn_logits = tokens.dyn_logits[idx]
    if mask_prebirth:
        src_time = torch.arange(T, device=coords.device).repeat_interleave(Hf * Wf)
        _stamp_prebirth_invalid(src_time, T, coords, visibs, dyn_logits)
    return (
        coords.view(1, T, Hf, Wf, T, 3),
        visibs.view(1, T, Hf, Wf, T),
        dyn_logits.view(1, T, Hf, Wf, T),
    )


def _gather_tracks_for_keys(
    pred: "Prediction",
    keys: torch.Tensor,
    *,
    out_device,
    static_visibility: bool,
    dtype,
    mask_prebirth: bool = True,
):
    """Positions / visibility / motion logits over all T frames for `keys`.

    Shared by the grid and merge-history extractors: walk the windows once and
    let each write its own target frames. Windows tile ``[0, T)``, so every
    frame is written exactly once and nothing has to be reconciled.

    ``mask_prebirth`` matches :func:`reconstruct_dense_tracks_3d`, so the two
    stay interchangeable (see :func:`_stamp_prebirth_invalid`).
    """
    windows = pred.dense_track_3d
    assert windows is not None
    assert pred.dense_shape is not None
    _B, T, Hf, Wf = pred.dense_shape
    cells = Hf * Wf
    device = windows[0].static_coords.device

    use_geom_vis = (
        static_visibility
        and pred.dense_extrinsics is not None
        and pred.dense_intrinsics is not None
    )
    surface_full = None
    if use_geom_vis:
        birth_pos = torch.empty((T * cells, 3), dtype=dtype, device=device)
        for w in windows:
            ws, we = w.window_start, w.window_end
            birth_pos[ws * cells : we * cells] = _window_diagonal_pointmap(w, cells)
        surface_full = birth_pos.reshape(T, Hf, Wf, 3)

    n = keys.numel()
    coords = torch.full(
        (n, T, 3), TRACK_INVALID_COORD, dtype=dtype, device=out_device
    )
    visibs = torch.full((n, T), -1000.0, dtype=dtype, device=out_device)
    dyn_logits = torch.full((n, T), -1000.0, dtype=dtype, device=out_device)

    for w, vox in _follow_cells(windows, keys, cells):
        ws, we = w.window_start, w.window_end
        live = vox >= 0
        if not bool(live.any()):
            continue
        rows = torch.nonzero(live, as_tuple=False).view(-1).to(out_device)
        vox_live = vox[live]
        static_visibs_tw = None
        if use_geom_vis:
            assert surface_full is not None
            static_visibs_tw = _window_static_visibs(
                w,
                surface_full[ws:we],
                pred.dense_extrinsics[0, ws:we].to(device=device),
                pred.dense_intrinsics[0, ws:we].to(device=device),
                points=w.static_coords[vox_live],
                static_visibs=w.static_visibs[vox_live],
            )
        w_coords, w_visibs = _gather_window_tracks(
            w, vox_live, static_visibs_tw=static_visibs_tw
        )
        coords[rows, ws:we] = w_coords.to(out_device)
        visibs[rows, ws:we] = w_visibs.to(out_device)
        dyn_logits[rows, ws:we] = (
            w.sd_logits[vox_live][:, None].expand(-1, we - ws).to(out_device, dtype)
        )
        del w_coords, w_visibs, vox_live, static_visibs_tw

    if mask_prebirth:
        _stamp_prebirth_invalid(
            torch.div(keys, cells, rounding_mode="floor"),
            T,
            coords,
            visibs,
            dyn_logits,
        )
    return coords, visibs, dyn_logits


def extract_grid_tracks(
    pred: "Prediction",
    *,
    out_device="cpu",
    static_visibility: bool = True,
    mask_prebirth: bool = True,
) -> CompressedTracks:
    """Reference extractor: one track per dense grid cell, by walking windows.

    Produces exactly what :func:`reconstruct_dense_tracks_3d` produces, with
    ``(B, T, H, W, T, ...)`` flattened to ``(T*H*W, T, ...)``, but by following
    the per-window ``p2v`` chain instead of slicing a materialized tensor.

    This exists as a **test oracle**, not as a pipeline component. It pins down
    the window-walking semantics that :func:`extract_compressed_tracks` and the
    merge-history tokens (:func:`extract_merge_history_tokens`) both rely on, by checking them against the dense
    reconstruction on a clip small enough to materialize. It is not a usable
    representation itself: it still stores ``T`` positions for each of the
    ``T*H*W`` cells, so it keeps the same ``T^2`` growth the dense tensor has.
    """
    windows = pred.dense_track_3d
    assert windows is not None and len(windows) > 0, (
        "pred.dense_track_3d is empty; run eval with dense_tracking_at_eval=True"
    )
    assert pred.dense_shape is not None, "extract_grid_tracks needs pred.dense_shape"

    B, T, Hf, Wf = pred.dense_shape
    assert B == 1, f"extract_grid_tracks assumes B == 1, got {B}"
    cells = Hf * Wf
    ref = windows[0].static_coords
    device, dtype = ref.device, ref.dtype

    keys = torch.arange(T * cells, device=device)
    coords, visibs, dyn_logits = _gather_tracks_for_keys(
        pred,
        keys,
        out_device=out_device,
        static_visibility=static_visibility,
        dtype=dtype,
        mask_prebirth=mask_prebirth,
    )
    keys = keys.to(out_device)
    return CompressedTracks(
        coords=coords,
        visibs=visibs,
        dyn_logits=dyn_logits,
        src_time=torch.div(keys, cells, rounding_mode="floor"),
        src_y=torch.div(keys % cells, Wf, rounding_mode="floor"),
        src_x=keys % Wf,
    )


def compressed_dynamic_logits_diagonal(pred: "Prediction", out_device="cpu") -> torch.Tensor:
    """Per-source-cell static/dynamic logits, i.e. the t_src == t_tgt diagonal.

    A cell (t, h, w) is a *new* cell of the window containing t, so its voxel is
    read straight off that window's p2v without any chain walking. Costs
    O(T * Hf * Wf) in total, versus the (B, T, H, W, T) tensor this replaces.
    """
    windows = pred.dense_track_3d
    assert windows is not None and len(windows) > 0, "pred.dense_track_3d is empty"
    B, T, Hf, Wf = pred.dense_shape
    cells = Hf * Wf

    ref = windows[0].sd_logits
    out = torch.full((B, T, Hf, Wf), -1000.0, dtype=ref.dtype, device=out_device)
    for w in windows:
        ws, we = w.window_start, w.window_end
        vox = _window_new_cell_voxels(w, cells)
        out[:, ws:we] = w.sd_logits[vox].reshape(1, we - ws, Hf, Wf).to(out_device)
    return out


def prepare_dense_tracks_for_eval_with_v6(
    gt_sample, train_data, evaluate_with_query_points=False
):
    (
        pred_dense_tracks,
        pred_dense_visibs,
        pred_dense_confs,
        pred_dense_dynamic,
    ) = reconstruct_dense_tracks_3d(
        train_data,
        return_visibs_confs=True,
        # 4D (B, T, H, W) source-cell logits. The default 5D (B, T, H, W, T)
        # layout breaks the rearrange below (einops expected 4 dims).
        dynamic_logits_by_target=False,
        # Benchmark scoring, not mask decoding: the dense GT can legitimately
        # cover frames before a cell's own frame (the physical point existed
        # before it became visible), and those frames are kept or dropped by
        # ``gt_sample.dense_valids`` below. Stamping TRACK_INVALID_COORD into
        # predictions the GT still scores would inflate position error, so this
        # path keeps the raw window output and lets the GT mask decide.
        mask_prebirth=False,
    )
    train_data.coords = rearrange(pred_dense_tracks, "b t1 h w t2 c -> b t2 (t1 h w) c")
    train_data.visibs = rearrange(pred_dense_visibs, "b t1 h w t2 -> b t2 (t1 h w)")
    train_data.confs = rearrange(pred_dense_confs, "b t1 h w t2 -> b t2 (t1 h w)")
    if pred_dense_dynamic.ndim == 5:
        # Unwritten targets are -1000; written window logits are duplicated
        # across t2, so max collapses to the source-cell score.
        pred_dense_dynamic = pred_dense_dynamic.max(dim=-1).values
    train_data.is_dynamic = rearrange(pred_dense_dynamic, "b t1 h w -> b (t1 h w)")

    est_dense_trajs_3d = gt_sample.est_dense_trajs_3d
    gt_dense_trajs_3d = gt_sample.dense_trajs_3d
    gt_dense_trajs_valid = gt_sample.dense_valids
    gt_dense_visibs = gt_sample.dense_visibs
    gt_dense_dynamic = _as_sequence_dynamic(gt_sample.dense_dynamic)

    gt_sample.valids = gt_dense_trajs_valid.flatten(1, 3).permute(0, 2, 1)
    valids = gt_sample.valids.any(1)[0]
    valids[:] = True

    # assert gt_dense_trajs_3d.shape[0] == 1, "only one batch supported for now"
    gt_sample.gt_trajs_3d = gt_dense_trajs_3d.flatten(1, 3).permute(0, 2, 1, 3)[
        :, :, valids
    ]  # B, T, N, 3
    if est_dense_trajs_3d[0] is not None:
        gt_sample.est_trajs_3d = est_dense_trajs_3d.flatten(1, 3).permute(0, 2, 1, 3)[
            :, :, valids
        ]  # B, T, N, 3
    # gt_sample.valids = gt_dense_trajs_valid.flatten(1, 3).permute(0, 2, 1)
    gt_sample.valids = gt_sample.valids[:, :, valids]  # B, T, N
    gt_sample.visibs = gt_dense_visibs.flatten(1, 3).permute(0, 2, 1)[
        :, :, valids
    ]  # B, T, N
    gt_sample.is_dynamic = gt_dense_dynamic.flatten(1, 3)[:, valids]  # (B, N)
    # metrics_alltrack3r does `is_dynamic & valids` with valids (B, T, N).
    if gt_sample.is_dynamic.ndim == 2 and gt_sample.valids.ndim == 3:
        gt_sample.is_dynamic = gt_sample.is_dynamic.unsqueeze(1)

    gt_sample.est_dense_trajs_3d = gt_sample.est_trajs_3d
    gt_sample.dense_trajs_3d = gt_sample.gt_trajs_3d
    gt_sample.dense_valids = gt_sample.valids
    gt_sample.dense_visibs = gt_sample.visibs
    gt_sample.dense_dynamic = gt_sample.is_dynamic

    return gt_sample, train_data