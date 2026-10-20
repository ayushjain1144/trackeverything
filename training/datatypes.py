# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class DenseTrackWindow:
    """Per-window snapshot of dense static-dynamic tracks produced at eval time.

    Saved at the last iteration of each window so that a helper can later
    reconstruct a dense (B, T, H, W, T, 3) trajectory tensor.

    All tensors live in the voxelized static-point space of the window's
    final iteration. Use `p2vs` + `p2v_lens` with
    `datasets.utils.geometry.unvoxelize_pc_with_p2v_lists` to unvoxelize
    back to the (B, window_end, H, W) grid.
    """

    window_start: int
    window_end: int
    sd_mask: torch.Tensor  # (N_vox,) bool over static points (voxel space)
    sd_trajs: torch.Tensor  # (N_sd, T_win, 3), T_win = window_end - window_start
    sd_visibs: torch.Tensor  # (N_sd, T_win)
    sd_confs: torch.Tensor  # (N_sd, T_win)
    # Raw static/dynamic motion logit per static voxel before thresholding to
    # obtain `sd_mask` (i.e. `pred_reconstruction_static_motion`).
    sd_logits: torch.Tensor  # (N_vox,)
    # Window-local predicted static coordinate per voxel. Entries marked dynamic
    # by `sd_mask` are overwritten with `sd_trajs` during reconstruction.
    static_coords: torch.Tensor  # (N_vox, 3)
    static_visibs: torch.Tensor  # (N_vox,)
    static_confs: torch.Tensor  # (N_vox,)
    p2vs: List[torch.Tensor] = field(default_factory=list)
    p2v_lens: List[int] = field(default_factory=list)
    coord_residuals: List[torch.Tensor] = field(default_factory=list)

    def to(self, device: str) -> "DenseTrackWindow":
        return DenseTrackWindow(
            window_start=self.window_start,
            window_end=self.window_end,
            sd_mask=self.sd_mask.to(device),
            sd_trajs=self.sd_trajs.to(device),
            sd_visibs=self.sd_visibs.to(device),
            sd_confs=self.sd_confs.to(device),
            sd_logits=self.sd_logits.to(device),
            static_coords=self.static_coords.to(device),
            static_visibs=self.static_visibs.to(device),
            static_confs=self.static_confs.to(device),
            p2vs=[p.to(device) for p in self.p2vs],
            p2v_lens=list(self.p2v_lens),
            coord_residuals=[r.to(device) for r in self.coord_residuals],
        )


@dataclass
class TrainData:
    iter_idx: int
    frame_range: Tuple[int, int]
    coords: Optional[List[torch.Tensor]] = None  # (B, T, N, 3)
    visibs: Optional[torch.Tensor] = None  # (B, T, N)
    is_dynamic: Optional[torch.Tensor] = None  # (B, N)
    confs: Optional[torch.Tensor] = None  # (B, T, N)
    query_coords: Optional[torch.Tensor] = None  # (B, T, N, 3)
    query_visibs: Optional[torch.Tensor] = None  # (B, T, N)
    query_confs: Optional[torch.Tensor] = None  # (B, T, N)
    noise_logits: Optional[torch.Tensor] = None  # (B, T, H, W, 1) dense noise prediction
    query_noise_logits: Optional[torch.Tensor] = None  # (B, T, N) query noise prediction
    d_along: Optional[torch.Tensor] = None  # (B, T, H, W, T_win) dense along-ray scalar prediction
    query_d_along: Optional[torch.Tensor] = None  # (B, T, N) query along-ray scalar prediction
    tgt_time_step_scene: Optional[torch.Tensor] = None  # (B, T, H, W) dense target time step prediction
    sampled_mask: Optional[torch.Tensor] = None  # (B, T, H, W) bool mask for subsampled voxels
    geom: Optional[torch.Tensor] = None  # (B, T, H, W, 3) geometric features
    flow: Optional[torch.Tensor] = None  # (B, T, H, W, 3) flow features
    query_geom: Optional[torch.Tensor] = None  # (B, T, N, 3) query geometric features
    query_flow: Optional[torch.Tensor] = None  # (B, T, N, 3) query flow features
    dense_dynamic_logits: Optional[torch.Tensor] = None  # (B, T, H, W) dense static/dynamic logits
    query_dynamic_logits: Optional[torch.Tensor] = None  # (B, N) query static/dynamic logits
    window_num: Optional[int] = None  # (B, T, H, W) window number
    prev_src_time_step: Optional[int] = None  # (B, T, H, W) previous source time step
    query_coord_at_tgt: Optional[torch.Tensor] = None  # (B, N, 3) query coordinate at target time step
    # Full-track predictions for a subsampled set of dense grid cells (M cells).
    dense_sample_coords: Optional[torch.Tensor] = None  # (B, T_win, M, 3)
    dense_sample_visibs: Optional[torch.Tensor] = None  # (B, T_win, M)
    dense_sample_valid_cells: Optional[torch.Tensor] = None  # (M,) bool: cell born & tracked this window
    dense_sample_dynamic_logits: Optional[torch.Tensor] = None  # (B, M) per-cell static/dynamic logits
    
    def __post_init__(self):
        if isinstance(self.coords, torch.Tensor):
            self.coords = [self.coords]
        if self.coords is not None:
            assert not isinstance(self.coords, torch.Tensor) and isinstance(
                self.coords, list
            ), "coords must be a list of tensors"
        # assert (
        #     self.frame_range[1] - self.frame_range[0] == self.coords[0].shape[1]
        # ), f"Frame range {self.frame_range} does not match the number of frames {self.coords[0].shape[1]}"

    def detach(self, clone: bool):
        _d = (lambda x: x.detach().clone()) if clone else (lambda x: x.detach())
        return TrainData(
            coords=[_d(c) for c in self.coords] if self.coords is not None else None,
            visibs=_d(self.visibs) if self.visibs is not None else None,
            confs=_d(self.confs) if self.confs is not None else None,
            is_dynamic=_d(self.is_dynamic) if self.is_dynamic is not None else None,
            iter_idx=self.iter_idx,
            frame_range=self.frame_range,
            query_coords=[_d(c) for c in self.query_coords]
            if self.query_coords is not None
            else None,
            query_visibs=_d(self.query_visibs) if self.query_visibs is not None else None,
            query_confs=_d(self.query_confs) if self.query_confs is not None else None,
            noise_logits=_d(self.noise_logits) if self.noise_logits is not None else None,
            query_noise_logits=_d(self.query_noise_logits) if self.query_noise_logits is not None else None,
            d_along=_d(self.d_along) if self.d_along is not None else None,
            query_d_along=_d(self.query_d_along) if self.query_d_along is not None else None,
            tgt_time_step_scene=self.tgt_time_step_scene,
            sampled_mask=_d(self.sampled_mask) if self.sampled_mask is not None else None,
            geom=_d(self.geom) if self.geom is not None else None,
            flow=_d(self.flow) if self.flow is not None else None,
            query_geom=_d(self.query_geom) if self.query_geom is not None else None,
            query_flow=_d(self.query_flow) if self.query_flow is not None else None,
            dense_dynamic_logits=_d(self.dense_dynamic_logits) if self.dense_dynamic_logits is not None else None,
            query_dynamic_logits=_d(self.query_dynamic_logits) if self.query_dynamic_logits is not None else None,
            window_num=self.window_num,
            prev_src_time_step=self.prev_src_time_step,
            query_coord_at_tgt=_d(self.query_coord_at_tgt) if self.query_coord_at_tgt is not None else None,
            dense_sample_coords=_d(self.dense_sample_coords) if self.dense_sample_coords is not None else None,
            dense_sample_visibs=_d(self.dense_sample_visibs) if self.dense_sample_visibs is not None else None,
            dense_sample_valid_cells=_d(self.dense_sample_valid_cells) if self.dense_sample_valid_cells is not None else None,
            dense_sample_dynamic_logits=_d(self.dense_sample_dynamic_logits) if self.dense_sample_dynamic_logits is not None else None,
        )

    def to(self, device: str):
        return TrainData(
            coords=[coords_.to(device) for coords_ in self.coords]
            if self.coords is not None
            else None,
            visibs=self.visibs.to(device) if self.visibs is not None else None,
            confs=self.confs.to(device) if self.confs is not None else None,
            is_dynamic=self.is_dynamic.to(device)
            if self.is_dynamic is not None
            else None,
            iter_idx=self.iter_idx,
            frame_range=self.frame_range,
            query_coords=[coords_.to(device) for coords_ in self.query_coords]
            if self.query_coords is not None
            else None,
            query_visibs=self.query_visibs.to(device)
            if self.query_visibs is not None
            else None,
            query_confs=self.query_confs.to(device)
            if self.query_confs is not None
            else None,
            noise_logits=self.noise_logits.to(device)
            if self.noise_logits is not None
            else None,
            query_noise_logits=self.query_noise_logits.to(device)
            if self.query_noise_logits is not None
            else None,
            d_along=self.d_along.to(device)
            if self.d_along is not None
            else None,
            query_d_along=self.query_d_along.to(device)
            if self.query_d_along is not None
            else None,
            tgt_time_step_scene=self.tgt_time_step_scene,
            sampled_mask=self.sampled_mask.to(device)
            if self.sampled_mask is not None
            else None,
            geom=self.geom.to(device) if self.geom is not None else None,
            flow=self.flow.to(device) if self.flow is not None else None,
            query_geom=self.query_geom.to(device) if self.query_geom is not None else None,
            query_flow=self.query_flow.to(device) if self.query_flow is not None else None,
            dense_dynamic_logits=self.dense_dynamic_logits.to(device) if self.dense_dynamic_logits is not None else None,
            query_dynamic_logits=self.query_dynamic_logits.to(device) if self.query_dynamic_logits is not None else None,
            window_num=self.window_num,
            prev_src_time_step=self.prev_src_time_step,
            query_coord_at_tgt=self.query_coord_at_tgt.to(device) if self.query_coord_at_tgt is not None else None,
            dense_sample_coords=self.dense_sample_coords.to(device) if self.dense_sample_coords is not None else None,
            dense_sample_visibs=self.dense_sample_visibs.to(device) if self.dense_sample_visibs is not None else None,
            dense_sample_valid_cells=self.dense_sample_valid_cells.to(device) if self.dense_sample_valid_cells is not None else None,
            dense_sample_dynamic_logits=self.dense_sample_dynamic_logits.to(device) if self.dense_sample_dynamic_logits is not None else None,
        )

    def time_slice(self, *, rel_start: int, rel_end: int):
        if self.coords is not None:
            assert (
                rel_start >= 0
                and rel_end <= self.coords[0].shape[1]
                and rel_start < rel_end
            ), "the range of start and end is out of bounds"
        return TrainData(
            coords=[coords_[:, rel_start:rel_end] for coords_ in self.coords]
            if self.coords is not None
            else None,
            visibs=self.visibs[:, rel_start:rel_end]
            if self.visibs is not None
            else None,
            confs=self.confs[:, rel_start:rel_end] if self.confs is not None else None,
            is_dynamic=self.is_dynamic if self.is_dynamic is not None else None,
            iter_idx=self.iter_idx,
            frame_range=(
                self.frame_range[0] + rel_start,
                self.frame_range[0] + rel_end,
            ),
            query_coords=[
                coords_[:, rel_start:rel_end] for coords_ in self.query_coords
            ]
            if self.query_coords is not None
            else None,
            query_visibs=self.query_visibs[:, rel_start:rel_end]
            if self.query_visibs is not None
            else None,
            query_confs=self.query_confs[:, rel_start:rel_end]
            if self.query_confs is not None
            else None,
            noise_logits=self.noise_logits[:, rel_start:rel_end]
            if self.noise_logits is not None
            else None,
            query_noise_logits=self.query_noise_logits[:, rel_start:rel_end]
            if self.query_noise_logits is not None
            else None,
            d_along=self.d_along[:, rel_start:rel_end]
            if self.d_along is not None
            else None,
            query_d_along=self.query_d_along[:, rel_start:rel_end]
            if self.query_d_along is not None
            else None,
            tgt_time_step_scene=self.tgt_time_step_scene,
            sampled_mask=self.sampled_mask[:, rel_start:rel_end]
            if self.sampled_mask is not None
            else None,
            geom=self.geom[:, rel_start:rel_end] if self.geom is not None else None,
            flow=self.flow[:, rel_start:rel_end] if self.flow is not None else None,
            query_geom=self.query_geom[:, rel_start:rel_end] if self.query_geom is not None else None,
            query_flow=self.query_flow[:, rel_start:rel_end] if self.query_flow is not None else None,
            dense_dynamic_logits=self.dense_dynamic_logits[:, rel_start:rel_end] if self.dense_dynamic_logits is not None else None,
            query_dynamic_logits=self.query_dynamic_logits if self.query_dynamic_logits is not None else None,
            window_num=self.window_num,
            prev_src_time_step=self.prev_src_time_step,
            query_coord_at_tgt=self.query_coord_at_tgt if self.query_coord_at_tgt is not None else None,
            dense_sample_coords=self.dense_sample_coords[:, rel_start:rel_end] if self.dense_sample_coords is not None else None,
            dense_sample_visibs=self.dense_sample_visibs[:, rel_start:rel_end] if self.dense_sample_visibs is not None else None,
            dense_sample_valid_cells=self.dense_sample_valid_cells if self.dense_sample_valid_cells is not None else None,
            dense_sample_dynamic_logits=self.dense_sample_dynamic_logits if self.dense_sample_dynamic_logits is not None else None,
        )

    def query_slice(self, s: slice):
        return TrainData(
            coords=[coords_[:, :, s] for coords_ in self.coords]
            if self.coords is not None
            else None,
            visibs=self.visibs[:, :, s] if self.visibs is not None else None,
            confs=self.confs[:, :, s] if self.confs is not None else None,
            is_dynamic=self.is_dynamic if self.is_dynamic is not None else None,
            iter_idx=self.iter_idx,
            frame_range=self.frame_range,
            query_coords=[coords_[:, :, s] for coords_ in self.query_coords]
            if self.query_coords is not None
            else None,
            query_visibs=self.query_visibs[:, :, s]
            if self.query_visibs is not None
            else None,
            query_confs=self.query_confs[:, :, s]
            if self.query_confs is not None
            else None,
            noise_logits=self.noise_logits
            if self.noise_logits is not None
            else None,
            query_noise_logits=self.query_noise_logits[:, :, s]
            if self.query_noise_logits is not None
            else None,
            d_along=self.d_along
            if self.d_along is not None
            else None,
            query_d_along=self.query_d_along[:, :, s]
            if self.query_d_along is not None
            else None,
            tgt_time_step_scene=self.tgt_time_step_scene,
            sampled_mask=self.sampled_mask,
            geom=self.geom if self.geom is not None else None,
            flow=self.flow[:, :, s] if self.flow is not None else None,
            query_geom=self.query_geom[:, :, s] if self.query_geom is not None else None,
            query_flow=self.query_flow[:, :, s] if self.query_flow is not None else None,
            dense_dynamic_logits=self.dense_dynamic_logits if self.dense_dynamic_logits is not None else None,
            query_dynamic_logits=self.query_dynamic_logits[:, s] if self.query_dynamic_logits is not None else None,
            window_num=self.window_num,
            prev_src_time_step=self.prev_src_time_step,
            query_coord_at_tgt=self.query_coord_at_tgt if self.query_coord_at_tgt is not None else None,
            dense_sample_coords=self.dense_sample_coords if self.dense_sample_coords is not None else None,
            dense_sample_visibs=self.dense_sample_visibs if self.dense_sample_visibs is not None else None,
            dense_sample_valid_cells=self.dense_sample_valid_cells if self.dense_sample_valid_cells is not None else None,
            dense_sample_dynamic_logits=self.dense_sample_dynamic_logits if self.dense_sample_dynamic_logits is not None else None,
        )


@dataclass
class Prediction:
    coords: Optional[torch.Tensor] = None  # (B, T, N, 3)
    visibs: Optional[torch.Tensor] = None  # (B, T, N)
    is_dynamic: Optional[torch.Tensor] = None  # (B, N)
    confs: Optional[torch.Tensor] = None  # (B, T, N)
    query_coords: Optional[torch.Tensor] = None  # (B, T, N, 3)
    query_visibs: Optional[torch.Tensor] = None  # (B, T, N)
    query_confs: Optional[torch.Tensor] = None  # (B, T, N)
    scene_memory: Optional[torch.Tensor] = None  # (B, N, 3)
    geom: Optional[torch.Tensor] = None  # (B, T, H, W, 3) dense reconstruction geometry
    flow: Optional[torch.Tensor] = None  # (B, T, N, 3)
    query_geom: Optional[torch.Tensor] = None  # (B, T, N, 3)
    query_flow: Optional[torch.Tensor] = None  # (B, T, N, 3)
    dense_dynamic_logits: Optional[torch.Tensor] = None  # (B, T, H, W, T) source→target SD logits
    geom_dynamic_logits: Optional[torch.Tensor] = None  # (B, T, H, W)
    query_dynamic_logits: Optional[torch.Tensor] = None  # (B, T, N) per-frame (window-broadcast)
    window_num: Optional[int] = None  # (B, T, N)
    prev_src_time_step: Optional[int] = None  # (B, T, N)
    # Per-window snapshots of static-dynamic dense tracks saved at eval time.
    # Use `utils.alltrack3r_utils.reconstruct_dense_tracks_3d` to materialize
    # the full (B, T, H, W, T, 3) tensor from these snapshots.
    dense_track_3d: Optional[List[DenseTrackWindow]] = None
    # Cheap (B, T, H, W) shape metadata for the dense feature grid. Lets dense
    # tracks be reconstructed without materializing the full pred.coords tensor
    # (e.g. under mem_efficient inference where coords is never allocated).
    dense_shape: Optional[Tuple[int, int, int, int]] = None
    # Cameras used during dense tracking, at the (Hf, Wf) feature-grid
    # resolution. Stashed at eval so reconstruct_dense_tracks_3d /
    # extract_compressed_tracks can depth-test static points without being
    # passed cameras explicitly.
    dense_extrinsics: Optional[torch.Tensor] = None  # (B, T, 4, 4) world -> camera
    dense_intrinsics: Optional[torch.Tensor] = None  # (B, T, 3, 3) at (Hf, Wf)
    # Full-track predictions for a subsampled set of dense grid cells (M cells).
    dense_sample_coords: Optional[torch.Tensor] = None  # (B, T, M, 3)
    dense_sample_visibs: Optional[torch.Tensor] = None  # (B, T, M)
    # (B, T, M) per-frame (window-broadcast), the dense_sample counterpart of
    # query_dynamic_logits. Populated even under mem_efficient, where the full
    # dense_dynamic_logits tensor is not.
    dense_sample_dynamic_logits: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.coords is not None:
            assert not self.coords.requires_grad
        if self.visibs is not None:
            assert not self.visibs.requires_grad

    def to(self, device: str):
        return Prediction(
            coords=self.coords.to(device) if self.coords is not None else None,
            visibs=self.visibs.to(device) if self.visibs is not None else None,
            confs=self.confs.to(device) if self.confs is not None else None,
            is_dynamic=self.is_dynamic.to(device)
            if self.is_dynamic is not None
            else None,
            query_coords=self.query_coords.to(device)
            if self.query_coords is not None
            else None,
            query_visibs=self.query_visibs.to(device)
            if self.query_visibs is not None
            else None,
            query_confs=self.query_confs.to(device)
            if self.query_confs is not None
            else None,
            scene_memory=self.scene_memory.to(device)
            if self.scene_memory is not None
            else None,
            geom=self.geom.to(device) if self.geom is not None else None,
            flow=self.flow.to(device) if self.flow is not None else None,
            query_geom=self.query_geom.to(device) if self.query_geom is not None else None,
            query_flow=self.query_flow.to(device) if self.query_flow is not None else None,
            dense_dynamic_logits=self.dense_dynamic_logits.to(device) if self.dense_dynamic_logits is not None else None,
            geom_dynamic_logits=self.geom_dynamic_logits.to(device) if self.geom_dynamic_logits is not None else None,
            query_dynamic_logits=self.query_dynamic_logits.to(device) if self.query_dynamic_logits is not None else None,
            window_num=self.window_num,
            prev_src_time_step=self.prev_src_time_step,
            dense_track_3d=(
                [w.to(device) for w in self.dense_track_3d]
                if self.dense_track_3d is not None
                else None
            ),
            dense_shape=self.dense_shape,
            dense_extrinsics=(
                self.dense_extrinsics.to(device)
                if self.dense_extrinsics is not None
                else None
            ),
            dense_intrinsics=(
                self.dense_intrinsics.to(device)
                if self.dense_intrinsics is not None
                else None
            ),
            dense_sample_coords=self.dense_sample_coords.to(device) if self.dense_sample_coords is not None else None,
            dense_sample_visibs=self.dense_sample_visibs.to(device) if self.dense_sample_visibs is not None else None,
            dense_sample_dynamic_logits=self.dense_sample_dynamic_logits.to(device) if self.dense_sample_dynamic_logits is not None else None,
        )

    def copy(self):
        return Prediction(
            coords=self.coords,
            visibs=self.visibs,
            is_dynamic=self.is_dynamic,
            confs=self.confs,
            query_coords=self.query_coords,
            query_visibs=self.query_visibs,
            query_confs=self.query_confs,
            scene_memory=self.scene_memory,
            geom=self.geom,
            flow=self.flow,
            query_geom=self.query_geom,
            query_flow=self.query_flow,
            dense_dynamic_logits=self.dense_dynamic_logits,
            geom_dynamic_logits=self.geom_dynamic_logits,
            query_dynamic_logits=self.query_dynamic_logits,
            window_num=self.window_num,
            prev_src_time_step=self.prev_src_time_step,
            dense_track_3d=self.dense_track_3d,
            dense_shape=self.dense_shape,
            dense_extrinsics=self.dense_extrinsics,
            dense_intrinsics=self.dense_intrinsics,
            dense_sample_coords=self.dense_sample_coords,
            dense_sample_visibs=self.dense_sample_visibs,
            dense_sample_dynamic_logits=self.dense_sample_dynamic_logits,
        )

    def time_slice(self, start: int, end: int):
        if self.coords is not None:
            assert (
                start >= 0 and end <= self.coords.shape[1] and start < end
            ), "the range of start and end is out of bounds"
        return Prediction(
            coords=self.coords[:, start:end] if self.coords is not None else None,
            visibs=self.visibs[:, start:end] if self.visibs is not None else None,
            confs=self.confs[:, start:end] if self.confs is not None else None,
            is_dynamic=self.is_dynamic if self.is_dynamic is not None else None,
            query_coords=self.query_coords[:, start:end]
            if self.query_coords is not None
            else None,
            query_visibs=self.query_visibs[:, start:end]
            if self.query_visibs is not None
            else None,
            query_confs=self.query_confs[:, start:end]
            if self.query_confs is not None
            else None,
            scene_memory=self.scene_memory,
            geom=self.geom[:, start:end] if self.geom is not None else None,
            flow=self.flow[:, start:end] if self.flow is not None else None,
            query_geom=self.query_geom[:, start:end] if self.query_geom is not None else None,
            query_flow=self.query_flow[:, start:end] if self.query_flow is not None else None,
            dense_dynamic_logits=(
                self.dense_dynamic_logits[:, start:end, :, :, start:end]
                if self.dense_dynamic_logits is not None
                else None
            ),
            geom_dynamic_logits=self.geom_dynamic_logits[:, start:end] if self.geom_dynamic_logits is not None else None,
            query_dynamic_logits=self.query_dynamic_logits[:, start:end] if self.query_dynamic_logits is not None else None,
            window_num=self.window_num,
            prev_src_time_step=self.prev_src_time_step,
            dense_track_3d=self.dense_track_3d,
            dense_sample_coords=self.dense_sample_coords[:, start:end] if self.dense_sample_coords is not None else None,
            dense_sample_visibs=self.dense_sample_visibs[:, start:end] if self.dense_sample_visibs is not None else None,
            dense_sample_dynamic_logits=self.dense_sample_dynamic_logits[:, start:end] if self.dense_sample_dynamic_logits is not None else None,
        )

    def query_slice(self, s: slice):
        return Prediction(
            coords=self.coords[:, :, s] if self.coords is not None else None,
            visibs=self.visibs[:, :, s] if self.visibs is not None else None,
            confs=self.confs[:, :, s] if self.confs is not None else None,
            is_dynamic=self.is_dynamic if self.is_dynamic is not None else None,
            query_coords=self.query_coords[:, :, s]
            if self.query_coords is not None
            else None,
            query_visibs=self.query_visibs[:, :, s]
            if self.query_visibs is not None
            else None,
            query_confs=self.query_confs[:, :, s]
            if self.query_confs is not None
            else None,
            scene_memory=self.scene_memory,
            geom=self.geom if self.geom is not None else None,
            flow=self.flow[:, :, s] if self.flow is not None else None,
            query_geom=self.query_geom[:, :, s] if self.query_geom is not None else None,
            query_flow=self.query_flow[:, :, s] if self.query_flow is not None else None,
            dense_dynamic_logits=self.dense_dynamic_logits if self.dense_dynamic_logits is not None else None,
            geom_dynamic_logits=self.geom_dynamic_logits if self.geom_dynamic_logits is not None else None,
            query_dynamic_logits=self.query_dynamic_logits[:, :, s] if self.query_dynamic_logits is not None else None,
            window_num=self.window_num,
            prev_src_time_step=self.prev_src_time_step,
            dense_track_3d=self.dense_track_3d,
            dense_sample_coords=self.dense_sample_coords if self.dense_sample_coords is not None else None,
            dense_sample_visibs=self.dense_sample_visibs if self.dense_sample_visibs is not None else None,
            dense_sample_dynamic_logits=self.dense_sample_dynamic_logits if self.dense_sample_dynamic_logits is not None else None,
        )
