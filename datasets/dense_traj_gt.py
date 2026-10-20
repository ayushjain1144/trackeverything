# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Dense GT trajectory grids + optional depth unproject (shared by BaseDataset and tests)."""

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import ipdb
st = ipdb.set_trace

from ..utils.alltrack3r_utils import (
    compute_dense_gt_tensor_fast,
    compute_dense_gt_tensor_fast_for_targets,
)
from ..utils.common_utils import batch_unproject


# Drop a dense pixel when the scattered track's source position differs from the
# base depth geometry by more than this fraction of the pixel depth (i.e. flow and
# base straddle a depth edge). Tune during A/B; larger keeps more, smaller is stricter.
BASE_FLOW_REL_THRESH = 0.05


def _depth_discontinuity_mask(
    depth: np.ndarray, rel_thresh: float = 0.01,
) -> np.ndarray:
    """Boolean mask that is True at pixels bordering a depth discontinuity.

    A pixel is flagged if any of its 4-connected neighbours differs by more
    than *rel_thresh* × max(depth_self, depth_neighbour).

    Args:
        depth: (T, H, W) depth maps.  Zero / negative depth is treated as
               invalid and never triggers a discontinuity on its own.
        rel_thresh: relative threshold for the jump.

    Returns:
        (T, H, W) bool mask – True where depth is discontinuous.
    """
    valid = depth > 0
    diff_h = np.abs(depth[:, 1:, :] - depth[:, :-1, :])
    diff_w = np.abs(depth[:, :, 1:] - depth[:, :, :-1])

    ref_h = np.maximum(depth[:, 1:, :], depth[:, :-1, :])
    ref_w = np.maximum(depth[:, :, 1:], depth[:, :, :-1])
    

    disc_h = diff_h > rel_thresh * ref_h
    disc_w = diff_w > rel_thresh * ref_w

    mask = np.zeros_like(depth, dtype=bool)
    mask[:, :-1, :] |= disc_h
    mask[:, 1:, :]  |= disc_h
    mask[:, :, :-1] |= disc_w
    mask[:, :, 1:]  |= disc_w

    mask &= valid

    return mask


def _unproject_dense_geometry(
    gt_depths_downsampled: np.ndarray,
    gt_intrinsics_downsampled: np.ndarray,
    gt_extrinsics: np.ndarray,
) -> np.ndarray:
    """Unproject downsampled depth maps to world-space 3D points.

    Returns (T, H2, W2, 3) numpy array.
    """
    pcd = batch_unproject(
        torch.from_numpy(gt_depths_downsampled).float().unsqueeze(0),
        torch.from_numpy(gt_intrinsics_downsampled).float().unsqueeze(0),
        torch.from_numpy(gt_extrinsics).float().unsqueeze(0),
    )  # (1, T, 3, H2, W2)
    pcd = pcd.squeeze(0).numpy()  # (T, 3, H2, W2)
    return pcd.transpose(0, 2, 3, 1)  # (T, H2, W2, 3)


def compute_dense_trajs_and_unproject(
    gt_traj_2d,
    gt_traj_3d,
    visibs,
    is_dynamic,
    H,
    W,
    scale_factor,
    compute_tmaps,
    gt_depths=None,
    gt_intrinsics=None,
    gt_extrinsics=None,
    gt_depths_downsampled=None,
    gt_intrinsics_downsampled=None,
    subsamle_geometry_points=None,
    make_all_dense_static=False,
    lift_full_geometry=True,
    time_steps: Optional[Sequence[int]] = None,
    dynamic_filter_threshold: Optional[float] = None,
    valids=None,
    window_size: Optional[int] = None,
) -> Tuple[
    Optional[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    np.ndarray,
]:
    """Compute dense GT trajectories from sparse tracks + unprojected depth.

    The scatter functions output 3D *flows* (delta from source-frame position).
    When depth/intrinsics/extrinsics are available we unproject the depth at
    downsampled resolution to get pixel-aligned dense geometry, then:
        dense_trajs_3d = dense_geom + scattered_flows
    This avoids the subpixel misalignment from rounding sparse 2D coords.

    If time_steps is set (e.g. [10, 14]), only those target frame indices are
    materialized on the last axis: arrays are (T, H', W', len(time_steps), ...)
    instead of (T, H', W', T, ...).
    """
    del subsamle_geometry_points  # unused; kept for API parity with BaseDataset

    has_depth = (
        gt_extrinsics is not None
        and gt_depths_downsampled is not None
        and gt_intrinsics_downsampled is not None
    )

    # Base geometry (world frame) that every dense track is anchored to. Built
    # first so the scatter can prefer, per patch, the track whose source position
    # is closest to this base (keeps flow and base depth pixel on the same surface).
    dense_geom = None
    if has_depth:
        dense_geom = _unproject_dense_geometry(
            gt_depths_downsampled, gt_intrinsics_downsampled, gt_extrinsics,
        )  # (T, H2, W2, 3)

    if time_steps is not None:
        target_idx = np.asarray(time_steps, dtype=np.int64).reshape(-1)
        (
            dense_flows_3d,
            dense_valids,
            dense_visibs,
            dense_dynamic,
            _,
            src_pos_map,
            src_idx_map,
        ) = compute_dense_gt_tensor_fast_for_targets(
            gt_traj_2d,
            gt_traj_3d,
            visibs,
            is_dynamic,
            H,
            W,
            target_idx,
            None,
            None,
            stride=scale_factor,
            skip_tmaps=not compute_tmaps,
            dynamic_filter_threshold=dynamic_filter_threshold,
            dense_geom=dense_geom,
            valids=valids,
            window_size=window_size,
        )
    else:
        (
            dense_flows_3d,
            dense_valids,
            dense_visibs,
            dense_dynamic,
            _,
            src_pos_map,
            src_idx_map,
        ) = compute_dense_gt_tensor_fast(
            gt_traj_2d,
            gt_traj_3d,
            visibs,
            is_dynamic,
            H,
            W,
            None,
            None,
            stride=scale_factor,
            skip_tmaps=not compute_tmaps,
            dense_geom=dense_geom,
            valids=valids,
        )

    if has_depth and dense_flows_3d is not None:
        dense_trajs_3d = dense_geom[:, :, :, None, :] + dense_flows_3d

        # Per-pixel base/flow consistency: drop only pixels where the track that
        # supplied the flow sits on a different surface than the base depth pixel
        # (i.e. base + flow would be a corrupted long track). src_pos_map holds the
        # scattered track source position; base_valid marks pixels with a track.
        # dense_valids is now per-target; a pixel "has a track" if it is valid for
        # any target. any(-1) is safe because pixels with no scattered track are all
        # False anyway (so invalidating them is a no-op).
        base_valid = dense_valids.any(axis=-1)  # (T, H2, W2)
        err = np.linalg.norm(dense_geom - src_pos_map, axis=-1)  # (T, H2, W2)
        bad = base_valid & (err > BASE_FLOW_REL_THRESH * gt_depths_downsampled)
        dense_valids[bad] = False  # broadcasts over the target axis

        if False:  # temporary A/B fallback; remove after visual comparison
            disc_mask = _depth_discontinuity_mask(gt_depths_downsampled)
            dense_valids[disc_mask] = False
    else:
        dense_trajs_3d = dense_flows_3d
    # dense_geom: (T, H', W', 3) raw grid base geometry (None if no depth);
    # src_idx_map: (T, H', W') winning sparse-track index per cell (-1 if empty).
    # Both are used to build grid-anchored full GT tracks for subsampled cells.
    return dense_trajs_3d, dense_valids, dense_visibs, dense_dynamic, dense_geom, src_idx_map


# def compute_dense_trajs_and_unproject(
#     gt_traj_2d,
#     gt_traj_3d,
#     visibs_and_valids,
#     is_dynamic,
#     H,
#     W,
#     scale_factor,
#     compute_tmaps,
#     gt_depths=None,
#     gt_intrinsics=None,
#     gt_extrinsics=None,
#     subsamle_geometry_points=None,
#     make_all_dense_static=False,
#     lift_full_geometry=True,
#     time_steps: Optional[Sequence[int]] = None,
#     gt_depths_downsampled=None,
#     gt_intrinsics_downsampled=None,
# ) -> Tuple[
#     Optional[np.ndarray],
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
# ]:
#     """Compute dense GT trajectories and optionally add unprojected point cloud points.

#     If gt_depths, gt_intrinsics, and gt_extrinsics are all provided,
#     unprojected point cloud points are spliced in at each source timestep.

#     If time_steps is set (e.g. [10, 14]), only those target frame indices are
#     materialized on the last axis: arrays are (T, H', W', len(time_steps), ...)
#     instead of (T, H', W', T, ...), avoiding a full T x T dense tensor.
#     In that mode, depth/intrinsics/extrinsics are unprojected only for the
#     unique frames listed in time_steps (not all T frames).
#     """
#     print("using old compute_dense_trajs_and_unproject")
#     del subsamle_geometry_points  # unused; kept for API parity with BaseDataset
#     if time_steps is not None:
#         target_idx = np.asarray(time_steps, dtype=np.int64).reshape(-1)
#         (
#             dense_trajs_3d,
#             dense_valids,
#             dense_visibs,
#             dense_dynamic,
#             _,
#         ) = compute_dense_gt_tensor_fast_for_targets(
#             gt_traj_2d,
#             gt_traj_3d,
#             visibs_and_valids,
#             is_dynamic,
#             H,
#             W,
#             target_idx,
#             None,
#             None,
#             stride=scale_factor,
#             skip_tmaps=not compute_tmaps,
#         )
#     else:
#         (
#             dense_trajs_3d,
#             dense_valids,
#             dense_visibs,
#             dense_dynamic,
#             _,
#         ) = compute_dense_gt_tensor_fast(
#             gt_traj_2d,
#             gt_traj_3d,
#             visibs_and_valids,
#             is_dynamic,
#             H,
#             W,
#             None,
#             None,
#             stride=scale_factor,
#             skip_tmaps=not compute_tmaps,
#         )

#     # if gt_depths is not None and gt_intrinsics is not None and gt_extrinsics is not None:
#     #     T = dense_valids.shape[0]

#     #     if time_steps is not None:
#     #         target_idx = np.asarray(time_steps, dtype=np.int64).reshape(-1)
#     #         in_range = (target_idx >= 0) & (target_idx < T)
#     #         if np.any(in_range):
#     #             unique_frames = np.unique(target_idx[in_range])
#     #             d_sub = gt_depths[unique_frames]
#     #             k_sub = gt_intrinsics[unique_frames]
#     #             e_sub = gt_extrinsics[unique_frames]
#     #             unprojected_pc_points = batch_unproject(
#     #                 torch.from_numpy(d_sub).unsqueeze(0),
#     #                 torch.from_numpy(k_sub).unsqueeze(0),
#     #                 torch.from_numpy(e_sub).unsqueeze(0),
#     #             )
#     #             unprojected_pc_points = unprojected_pc_points.squeeze(0).numpy()
#     #             unprojected_pc_points = unprojected_pc_points[
#     #                 ...,
#     #                 scale_factor // 2 - 1 :: scale_factor,
#     #                 scale_factor // 2 - 1 :: scale_factor,
#     #             ].copy()
#     #             slab_rows = unprojected_pc_points.transpose(0, 2, 3, 1)
#     #             t_to_u = {int(unique_frames[i]): i for i in range(unique_frames.size)}
#     #         else:
#     #             t_to_u = {}

#     #         if dense_trajs_3d is not None:
#     #             for j, tj in enumerate(target_idx):
#     #                 if tj in t_to_u:
#     #                     dense_trajs_3d[tj, :, :, j, :] = slab_rows[
#     #                         t_to_u[tj]
#     #                     ].copy()
#     #         if dense_trajs_3d is not None or lift_full_geometry:
#     #             for j, tj in enumerate(target_idx):
#     #                 if 0 <= tj < T:
#     #                     dense_valids[tj, :, :, j] = True
#     #                     dense_visibs[tj, :, :, j] = True
#     #     else:
#     #         unprojected_pc_points = batch_unproject(
#     #             torch.from_numpy(gt_depths).unsqueeze(0),
#     #             torch.from_numpy(gt_intrinsics).unsqueeze(0),
#     #             torch.from_numpy(gt_extrinsics).unsqueeze(0),
#     #         )
#     #         unprojected_pc_points = unprojected_pc_points.squeeze(0).numpy()
#     #         unprojected_pc_points = unprojected_pc_points[
#     #             ...,
#     #             scale_factor // 2 - 1 :: scale_factor,
#     #             scale_factor // 2 - 1 :: scale_factor,
#     #         ].copy()
#     #         slab = unprojected_pc_points.transpose(0, 2, 3, 1).copy()

#     #         if dense_trajs_3d is not None:
#     #             dense_trajs_3d[np.arange(T), :, :, np.arange(T)] = slab
#     #         if dense_trajs_3d is not None or lift_full_geometry:
#     #             dense_valids[np.arange(T), :, :, np.arange(T)] = True
#     #             dense_visibs[np.arange(T), :, :, np.arange(T)] = True

#     # if make_all_dense_static:
#     #     dense_dynamic[:] = False

#     return dense_trajs_3d, dense_valids, dense_visibs, dense_dynamic