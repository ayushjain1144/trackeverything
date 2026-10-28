# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-camera inference variant of TrackEverything.

``TrackEverythingMultiView`` adds a ``num_views`` argument to the streaming
forward pass and changes nothing else; the weights are the single-camera ones.
It is kept separate from ``models/track_everything.py`` so the single-camera
path stays exactly as trained.
"""

import logging
from typing import Generator, List, Optional, Tuple, Union

import ipdb

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from line_profiler import profile
from torch.amp import autocast
from torch_scatter import scatter_mean

from ..datasets.utils.geometry import (
    compute_subsample_dense_mask,
    dense_sample_voxel_map,
    merge_per_frame_p2v_labels,
    unvoxelize_coords_with_residuals,
    unvoxelize_pc_with_p2v_lists,
    voxelization,
    voxelize_memory_p2v,
)

from ..training.datatypes import DenseTrackWindow, Prediction, TrainData
from ..utils.common_utils import (
    batch_unproject,
    cast_float32,
)

from .track_everything import TrackEverything

logger = logging.getLogger(__name__)
st = ipdb.set_trace


class TrackEverythingMultiView(TrackEverything):
    """TrackEverything with synchronized multi-camera input.

    ``V`` cameras are laid out on the single time axis the single-camera model
    already uses, in time-major order: flat image index ``t = f * V + v`` for
    ``T = F * V`` images over ``F`` timesteps. Two index spaces therefore
    coexist, and conflating them is the one way this goes wrong:

    - image index (``0..T-1``) indexes ``rgb_obs``, ``depth_obs``,
      ``intrinsics``, ``extrinsics``, image features, pointmaps and every
      ``pred.*`` output buffer;
    - timestep index (``0..F-1``) indexes the source/target time embeddings,
      ``tgt_time_step_scene`` and every per-timestep trajectory tensor.

    Inference only: ``streaming_forward`` asserts ``not self.training`` when
    ``num_views > 1``.

    The camera count can be passed per call or set once on the instance as
    ``num_views``, like ``voxel_size`` and the other inference knobs.
    """

    num_views: int = 1

    @profile
    def streaming_forward(
        self,
        *,
        rgb_obs: torch.Tensor,
        depth_obs: torch.Tensor,
        num_iters: int,
        query_point: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_feats: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        disable_backprop_through_time: bool = False,
        tgt_time_step_dense: Optional[torch.Tensor] = None,
        dense_norm_mask: Optional[torch.Tensor] = None,
        dense_sample_points: Optional[torch.Tensor] = None,
        warmup_frames: int = 0,
        num_views: Optional[int] = None,
    ) -> Generator[Union[TrainData, Prediction], None, None]:
        """
        This function returns a generator which yields Prediction and TrainData objects.
        It adheres to the following protocol:
        - First several calls: yield TrainData objects with gradient
        - Last call: yield a Prediction object without gradient
        It serves as a wrapper of refine_endpoints + the trajectory refiner.

        Windows are non-overlapping; only the new frames are accumulated into the
        scene memory.

        Arguments:
            rgb_obs: (B, num_frames, 3, H, W) in [0, 1]
            depth_obs: (B, num_frames, H, W) in world units
            num_iters: Number of iterations
            query_point: (B, npts, 4).  The last dimension denotes (t, x, y, z)
            intrinsics: (B, T, 3, 3)
            extrinsics: (B, T, 4, 4)
            image_feats: (B, num_frames, F, H // 4, W // 4)
            num_views: cameras per timestep (1 = monocular); defaults to
                ``self.num_views``
        """
        pad = 0

        B, T, _, H, W = rgb_obs.shape
        logger.debug("B: %d, T: %d, H: %d, W: %d", B, T, H, W)

        V = int(self.num_views if num_views is None else num_views)
        assert V >= 1, f"num_views must be >= 1, got {V}"
        if V > 1:
            assert T % V == 0, f"T={T} must be divisible by num_views={V}"
            assert not self.training, "multiview is inference-only"
            assert warmup_frames == 0, "warmup_frames is not supported for multiview"
            assert (
                dense_sample_points is None
                or not torch.is_tensor(dense_sample_points)
                or dense_sample_points.numel() == 0
            ), "dense_sample_points is not supported for multiview"

        def _cam0(x: torch.Tensor) -> torch.Tensor:
            """Keep camera 0 of each timestep along a (B, T, ...) image axis."""
            return x if V == 1 else x[:, ::V]

        def _to_flat(x: torch.Tensor, dim: int) -> torch.Tensor:
            """Timestep axis -> image axis, replicating each timestep V times."""
            return x if V == 1 else x.repeat_interleave(V, dim=dim)

        # A window spans self.seq_len timesteps, i.e. seq_len * V images.
        seq_len = self.seq_len * V
        step = seq_len
        window_ends = list(range(seq_len, T + 1, step))
        if not window_ends or window_ends[-1] < T:
            window_ends.append(T)

        # Windows are non-overlapping (step == seq_len) and every consumer of the
        # image features reads only [window_start:window_end], so at inference we
        # encode one window at a time instead of materializing all T frames
        # (~9.4 GB at T=1500). A single-entry cache suffices: each window is
        # visited exactly once, so this costs no recompute. The grouping matches
        # the eager path, which already chunks the encoder at chunk_size=seq_len.
        lazy_feats = image_feats is None and not torch.is_grad_enabled()
        if image_feats is None and not lazy_feats:
            image_feats = self.encode_rgbs(
                rgb_obs,
                chunk_size=chunk_size,
                warmup_frames=warmup_frames,
            )

        _feat_cache = {}

        def _window_feats(start: int, end: int) -> torch.Tensor:
            if image_feats is not None:
                return image_feats[:, start:end]
            if _feat_cache.get("key") != (start, end):
                _feat_cache.clear()
                _feat_cache["key"] = (start, end)
                _feat_cache["val"] = self.encode_rgbs(
                    rgb_obs[:, start:end], chunk_size=chunk_size
                )
            return _feat_cache["val"]

        Hf, Wf = _window_feats(0, window_ends[0]).shape[-2:]

        def _window_points(start: int, end: int) -> torch.Tensor:
            with autocast("cuda", enabled=False):
                pcds = batch_unproject(
                    depth_obs[:, start:end],
                    intrinsics[:, start:end],
                    extrinsics[:, start:end],
                )  # B X t X 3 X H X W
            pts = F.interpolate(
                pcds.flatten(0, 1),
                size=(Hf, Wf),
                mode="nearest-exact",
            ).reshape(B, end - start, 3, Hf, Wf)
            del pcds
            return pts

        if dense_norm_mask is not None and dense_norm_mask[0] is not None:
            if tuple(dense_norm_mask.shape[-3:-1]) != (Hf, Wf):
                # Chunked over time: the full-T float copy is ~1.2 GB at T=1500.
                _mask_chunks = []
                for _s in range(0, T, seq_len):
                    _e = min(_s + seq_len, T)
                    _m = F.interpolate(
                        dense_norm_mask[:, _s:_e]
                        .squeeze(-1)
                        .flatten(0, 1)
                        .unsqueeze(1)
                        .float(),
                        size=(Hf, Wf),
                        mode="nearest-exact",
                    ).reshape(B, _e - _s, Hf, Wf)
                    _mask_chunks.append(_m.unsqueeze(-1).bool())
                dense_norm_mask = torch.cat(_mask_chunks, dim=1)
                del _mask_chunks

        h_scale = Hf / depth_obs.shape[-2]
        w_scale = Wf / depth_obs.shape[-1]
        logger.debug("h_scale: %s, w_scale: %s", h_scale, w_scale)

        intrinsics_scaled = intrinsics.clone()
        intrinsics_scaled[:, :, 0, 0] *= w_scale
        intrinsics_scaled[:, :, 1, 1] *= h_scale
        intrinsics_scaled[:, :, 0, 2] *= w_scale
        intrinsics_scaled[:, :, 1, 2] *= h_scale

        pred = Prediction()
        pred.dense_shape = (B, T, Hf, Wf)
        if not self.training:
            pred.dense_extrinsics = extrinsics.detach()
            pred.dense_intrinsics = intrinsics_scaled.detach()
        if (not self.use_only_query_points and not self.mem_efficient) or self.training:
            pred.coords = torch.zeros((B, T, Hf, Wf, 3), device=rgb_obs.device, dtype=rgb_obs.dtype)
            pred.visibs = torch.zeros((B, T, Hf, Wf), device=rgb_obs.device, dtype=rgb_obs.dtype)

        if self.use_query_points:
            pred.query_coords = repeat(query_point[..., 1:], "b n c -> b t n c", t=T).clone()
            pred.query_visibs = torch.zeros_like(pred.query_coords[..., 0])

        # dense-sample points: a subset of dense grid cells tracked as full
        # trajectories via the (shared) traj_decoder. Model runs B=1 internally.
        use_dense_sample = (
            dense_sample_points is not None
            and torch.is_tensor(dense_sample_points)
            and dense_sample_points.numel() > 0
            and dense_sample_points.shape[-2] > 0
        )
        dense_sample_pts = None
        num_dense_sample_pts = 0
        if use_dense_sample:
            dense_sample_pts = dense_sample_points[0].to(rgb_obs.device).long()  # (M, 3)
            num_dense_sample_pts = dense_sample_pts.shape[0]
            pred.dense_sample_coords = torch.zeros(
                (B, T, num_dense_sample_pts, 3), device=rgb_obs.device, dtype=rgb_obs.dtype
            )
            pred.dense_sample_visibs = torch.zeros(
                (B, T, num_dense_sample_pts), device=rgb_obs.device, dtype=rgb_obs.dtype
            )
            # -1000 ≈ strongly static under sigmoid > thresh, matching the dense
            # logit convention. Cheap enough (B, T, M) to keep under mem_efficient.
            pred.dense_sample_dynamic_logits = torch.full(
                (B, T, num_dense_sample_pts),
                -1000.0,
                device=rgb_obs.device,
                dtype=rgb_obs.dtype,
            )

        time = torch.arange(self.seq_len, device=rgb_obs.device)
        time = torch.cat([time, torch.ones(1, device=rgb_obs.device) * -1], dim=0)

        # scene feature cloud state (carried across windows as voxelized memory)
        scene_feats = None
        scene_coords = None  # current target-position estimate for scene points
        scene_src_coords = None  # fixed source coordinate (paper's E^xyz_src)
        scene_waft_feats = None
        scene_src_time = None
        # Eval-only: persistent per-voxel "ever been dynamic" flag used by
        # --compress_memory to coarsen only never-dynamic memory points.
        compress_active = (self.compress_memory is not None) and (not self.training)
        scene_ever_dynamic = None
        # srctgt merge key, rebuilt at the end of each window.
        mem_merge_key = None
        # query (sparse) point state
        query_feats = None
        query_coords = None
        query_src_coords = None
        query_waft_feats = None
        query_waft_coords = None
        query_src_time = None
        # trajectory-refiner recurrent state. A single merged buffer holds the
        # concatenation of [scene-track subset ; query points] since the two do
        # not attend to each other inside traj_decoder (only to scene_context),
        # so one merged pass is equivalent to two separate passes.
        track_traj_coords_seq = None
        track_traj_feats_seq = None

        # Initialize pred dynamic logit fields (will be populated per-window).
        # Dense logits are stored as a source→target track tensor (B, T, H, W, T),
        # matching dense track layout; each window broadcasts its per-voxel logit
        # onto window_start:window_end of the last axis.
        if (not self.use_only_query_points and not self.mem_efficient) or self.training and pred.dense_dynamic_logits is None:
            _H, _W = Hf, Wf
            # -1000 ≈ strongly static under sigmoid > thresh (matches reconstruct_dense_tracks_3d).
            pred.dense_dynamic_logits = torch.full(
                (B, T, _H, _W, T),
                -1000.0,
                device=rgb_obs.device,
                dtype=rgb_obs.dtype,
            )
        # (B, T, N) is small enough to keep even under mem_efficient, unlike the
        # dense (B, T, H, W, T) tensor above.
        if self.use_query_points and pred.query_dynamic_logits is None:
            pred.query_dynamic_logits = torch.full(
                (B, T, query_point.shape[1]),
                -1000.0,
                device=rgb_obs.device,
                dtype=rgb_obs.dtype,
            )

        all_p2vs = []
        all_p2vs_len = []
        all_coord_residuals = []
        subsample_alive_list = []
        query_fill_idxs = None

        num_query_points = 0

        _prev_grad_state = torch.is_grad_enabled()
        for i, window_end in enumerate(window_ends):
            window_start = 0 if i == 0 else window_ends[i - 1]
            window_size = window_end - window_start
            new_start = window_start if i == 0 else max(window_ends[i - 1], window_start)
            new_frame_count = window_end - new_start
            assert window_size % V == 0 and new_frame_count % V == 0
            # Timesteps, not images: the length of every trajectory tensor.
            traj_len = window_size // V
            new_step_count = new_frame_count // V

            is_warmup_window = warmup_frames > 0 and window_end <= warmup_frames
            torch.set_grad_enabled(not is_warmup_window and _prev_grad_state)

            image_feats_w = _window_feats(window_start, window_end)
            points_win = _window_points(window_start, window_end)
            new_frame_offset = new_start - window_start

            # only accumulate NEW (non-overlapping) frames into scene memory
            point_features_w = image_feats_w[:, new_frame_offset:]
            point_features_w = rearrange(point_features_w, "b t c h w -> b (t h w) c")

            points_w = points_win[:, new_frame_offset:]
            points_w = rearrange(points_w, "b t c h w -> b (t h w) c")

            if dense_norm_mask is not None and dense_norm_mask[0] is not None:
                # masked_fill avoids bool-index host sync (aten::nonzero on assign).
                _valid = dense_norm_mask[:, new_start:window_end].reshape(B, -1)
                points_w = points_w.masked_fill(~_valid.unsqueeze(-1), -10.0)

            # raw per-frame pointmap for the current window (used for WAFT projection)
            frame_pointmaps = rearrange(
                points_win,
                'b t c h w -> b t h w c',
            )
            # Source timestep of each new point; the V images of one timestep
            # share a source time.
            src_time_idx_w = repeat(
                torch.arange(new_step_count, device=rgb_obs.device)
                + new_frame_offset // V,
                "t -> b t v h w 1",
                b=B,
                v=V,
                h=Hf,
                w=Wf,
            )
            scene_src_time_w = src_time_idx_w.squeeze(0).flatten()

            if self.use_query_points:
                time_mask = (query_point[..., 0] >= new_start) & (
                    query_point[..., 0] < window_end
                )

                new_fill_idxs = torch.where(time_mask[0])[0]
                if query_fill_idxs is None:
                    query_fill_idxs = new_fill_idxs
                else:
                    query_fill_idxs = torch.cat([query_fill_idxs, new_fill_idxs])
                # Integer gather instead of bool-index (host sync). B=1 training.
                query_coords_w = query_point[0].index_select(0, new_fill_idxs)
                # query_point[..., 0] is an image index.
                query_src_time_w = (
                    query_point[0, :, 0].to(torch.int64).index_select(0, new_fill_idxs)
                    - window_start
                ) // V

            scene_feats_w = point_features_w.squeeze(0)
            scene_waft_feats_w = scene_feats_w

            num_mem_scene_points = 0
            if scene_feats is not None and not self.sparse_tracking:
                num_mem_scene_points = scene_coords.shape[0]
                old_time_val = -1
                scene_src_time = torch.ones_like(scene_coords[:, 0]).to(torch.int64) * old_time_val
                scene_feats = torch.cat([scene_feats, scene_feats_w], dim=0)
                scene_coords = torch.cat([scene_coords, points_w.squeeze(0)], dim=0)
                scene_waft_feats = torch.cat([
                    scene_waft_feats, scene_waft_feats_w
                ], dim=0)
                scene_src_time = torch.cat([scene_src_time, scene_src_time_w], dim=0)
                if compress_active:
                    scene_ever_dynamic = torch.cat([
                        scene_ever_dynamic,
                        torch.zeros(
                            points_w.squeeze(0).shape[0],
                            dtype=torch.bool,
                            device=scene_coords.device,
                        ),
                    ], dim=0)
            else:
                scene_feats = scene_feats_w
                scene_coords = points_w.squeeze(0)
                scene_waft_feats = scene_waft_feats_w
                scene_src_time = scene_src_time_w
                if compress_active:
                    scene_ever_dynamic = torch.zeros(
                        scene_coords.shape[0],
                        dtype=torch.bool,
                        device=scene_coords.device,
                    )

            # voxelize the scene points
            p2v_w = None
            if self.voxelize and scene_feats.shape[0] != 0:
                if num_mem_scene_points > 0:
                    if compress_active:
                        p2v_w_mem, mem_vocab, _eff_vox = voxelize_memory_p2v(
                            scene_coords[:num_mem_scene_points],
                            scene_ever_dynamic[:num_mem_scene_points],
                            self.voxel_size,
                            compress_to=self.compress_memory,
                            mode=getattr(self, "voxel_mode", "uniform"),
                        )
                    elif (
                        mem_merge_key is not None
                        and mem_merge_key.shape[0] == num_mem_scene_points
                    ):
                        # srctgt (None in "all" mode and before window 0 ends)
                        p2v_w_mem, mem_vocab = self._srctgt_memory_p2v(mem_merge_key)
                    else:
                        p2v_w_mem = voxelization(
                            scene_coords[:num_mem_scene_points][None],
                            self.voxel_size,
                            mode=getattr(self, "voxel_mode", "uniform"),
                        )[0]
                        mem_vocab = p2v_w_mem.max() + 1
                    # Grouped per timestep, so the V cameras observing one
                    # instant share a voxel grid.
                    p2v_w_frames = voxelization(
                        scene_coords[num_mem_scene_points:][None].reshape(
                            new_step_count, -1, 3
                        ),
                        self.voxel_size,
                        mode=getattr(self, "voxel_mode", "uniform"),
                    )
                    p2v_w = torch.cat(
                        [
                            p2v_w_mem,
                            merge_per_frame_p2v_labels(p2v_w_frames, base_offset=mem_vocab),
                        ]
                    )
                else:
                    p2v_w_frames = voxelization(
                        scene_coords[None].reshape(new_step_count, -1, 3),
                        self.voxel_size,
                        mode=getattr(self, "voxel_mode", "uniform"),
                    )
                    p2v_w = merge_per_frame_p2v_labels(p2v_w_frames, base_offset=0)
                # Fuse coords/feats/waft into one scatter_mean (same per-channel means)
                # and pass dim_size once so torch_scatter does not call index.max()
                # three times.
                _num_voxels = int(p2v_w.max().item()) + 1
                scene_coords_before_voxel = scene_coords
                _feat_dim = scene_feats.shape[-1]
                _waft_dim = scene_waft_feats.shape[-1]
                _packed = torch.cat(
                    [scene_coords, scene_feats, scene_waft_feats], dim=-1
                )
                _packed = scatter_mean(
                    _packed, p2v_w, dim=0, dim_size=_num_voxels
                )
                scene_coords = _packed[:, :3]
                scene_feats = _packed[:, 3 : 3 + _feat_dim]
                scene_waft_feats = _packed[:, 3 + _feat_dim : 3 + _feat_dim + _waft_dim]
                if self.residual_unvoxelize:
                    all_coord_residuals.append(
                        (
                            scene_coords_before_voxel - scene_coords[p2v_w]
                        ).detach()
                    )
                # sync-free amax replacement for torch_scatter.scatter_max, which is host-blocking
                _src_time_out = torch.full(
                    (_num_voxels,), -1,
                    dtype=scene_src_time.dtype,
                    device=scene_src_time.device,
                )
                _src_time_out.scatter_reduce_(0, p2v_w, scene_src_time, reduce="amax", include_self=False)
                scene_src_time = _src_time_out
                if compress_active:
                    _ever_out = torch.zeros(
                        _num_voxels,
                        dtype=torch.long,
                        device=scene_ever_dynamic.device,
                    )
                    _ever_out.scatter_reduce_(
                        0, p2v_w, scene_ever_dynamic.long(),
                        reduce="amax", include_self=False,
                    )
                    scene_ever_dynamic = _ever_out.bool()
                all_p2vs.append(p2v_w)
                all_p2vs_len.append(len(scene_coords))

            if self.training and self.voxelize and p2v_w is not None:
                N_max_voxels = 15000
                num_voxels = scene_coords.shape[0]
                if num_voxels > N_max_voxels:
                    perm = torch.randperm(num_voxels, device=scene_coords.device)
                    # Force-keep the voxels backing the sampled dense cells so that
                    # their full-track supervision is never silently dropped by the
                    # 15k voxel subsample.
                    forced_vox = None
                    if use_dense_sample:
                        _vox_pre, _born_pre = dense_sample_voxel_map(
                            dense_sample_pts, window_end,
                            Hf, Wf,
                            all_p2vs, all_p2vs_len, num_voxels, scene_coords.device,
                        )
                        forced_vox = torch.unique(_vox_pre[_born_pre])
                    if forced_vox is not None and forced_vox.numel() > 0:
                        forced_mask0 = torch.zeros(num_voxels, dtype=torch.bool, device=scene_coords.device)
                        forced_mask0[forced_vox] = True
                        non_forced = perm[~forced_mask0[perm]]
                        n_keep = max(N_max_voxels, int(forced_vox.numel()))
                        keep_idx = torch.cat([forced_vox, non_forced])[:n_keep]
                    else:
                        keep_idx = perm[:N_max_voxels]
                    keep_idx = keep_idx.sort().values

                    kept_mask = torch.zeros(num_voxels, dtype=torch.bool, device=scene_coords.device)
                    kept_mask[keep_idx] = True
                    original_p2v = all_p2vs[-1]
                    subsample_alive = kept_mask[original_p2v]
                    subsample_alive_list.append(subsample_alive)

                    remap = torch.zeros(num_voxels, dtype=torch.long, device=scene_coords.device)
                    remap[keep_idx] = torch.arange(len(keep_idx), device=scene_coords.device)
                    all_p2vs[-1] = remap[original_p2v]

                    scene_coords = scene_coords[keep_idx]
                    scene_feats = scene_feats[keep_idx]
                    scene_waft_feats = scene_waft_feats[keep_idx]
                    scene_src_time = scene_src_time[keep_idx]
                    all_p2vs_len[-1] = len(scene_coords)
                else:
                    subsample_alive_list.append(None)
            elif self.voxelize and p2v_w is not None:
                subsample_alive_list.append(None)
            t_size = window_size
            if self.use_query_points:
                # Image index, not timestep: the gather reads the feature from
                # the camera the query was authored in.
                time_index = query_coords_w[..., 0].to(torch.int64) - window_start
                if len(query_coords_w) > 0:
                    from ..utils.warp_utils import warp_2d
                    query_feats_w = warp_2d(
                        feats=image_feats_w,
                        coords_3d=repeat(
                            query_coords_w[..., 1:], "n c -> 1 t n c", t=t_size
                        ),
                        cam_T_world=extrinsics[:, window_start:window_end],
                        intrinsics=intrinsics_scaled[:, window_start:window_end],
                    )  # b n t c

                    query_feats_w = torch.gather(
                        query_feats_w,
                        dim=2,
                        index=repeat(
                            time_index,
                            "n -> b n 1 c",
                            b=query_feats_w.shape[0],
                            c=query_feats_w.shape[3],
                        ),
                    ).squeeze(0, 2)
                    query_waft_feats_w = query_feats_w
                    query_waft_coords_w = query_coords_w[:, 1:]

                    query_coords_w = query_coords_w[..., 1:]
                    num_query_points += len(query_coords_w)

                    if query_coords is not None:
                        old_time_val = -1
                        query_src_time_old = torch.ones_like(query_coords[:, 0]).to(torch.int64) * old_time_val
                        query_coords = torch.cat([
                            query_coords, query_coords_w
                        ])
                        query_feats = torch.cat([
                            query_feats, query_feats_w
                        ])
                        query_waft_feats = torch.cat([
                            query_waft_feats, query_waft_feats_w
                        ])
                        query_waft_coords = torch.cat([
                            query_waft_coords, query_waft_coords_w
                        ])
                        query_src_time = torch.cat([
                            query_src_time_old, query_src_time_w
                        ])
                    else:
                        query_waft_feats = query_waft_feats_w
                        query_waft_coords = query_waft_coords_w
                        query_src_time = query_src_time_w
                        query_feats = query_feats_w
                        query_coords = query_coords_w
                elif query_coords is not None:
                    logger.debug("Handling 0 query point case")
                    query_src_time_old = torch.ones_like(query_src_time) * -1
                    query_src_time = query_src_time_old
                    query_feats_w = torch.zeros(
                        (0, query_feats.shape[-1]), device=query_feats.device
                    )

            if query_coords is None:
                logger.debug("Handling 0 query point case")
                query_coords = torch.zeros((0, 3), device=scene_coords.device)
                query_feats = torch.zeros((0, scene_feats.shape[-1]), device=scene_coords.device)
                query_waft_feats = torch.zeros((0, scene_waft_feats.shape[-1]), device=scene_coords.device)
                query_waft_coords = torch.zeros((0, 3), device=scene_coords.device)
                query_src_time = torch.zeros((0,), dtype=torch.int64, device=scene_coords.device)
                query_coords_w = torch.zeros((0, 3), device=scene_coords.device)
                query_feats_w = torch.zeros((0, scene_feats_w.shape[-1]), device=scene_coords.device)
                query_waft_feats_w = torch.zeros((0, scene_waft_feats_w.shape[-1]), device=scene_coords.device)
                query_waft_coords_w = torch.zeros((0, 3), device=scene_coords.device)
                query_src_time_w = torch.zeros((0,), dtype=torch.int64, device=scene_coords.device)

            del scene_feats_w
            del (
                point_features_w,
                points_w,
            )

            # Map sampled dense cells to their current-window voxel ids.
            # Remapped if voxelize_on_endpoints collapses after iter 0.
            sample_vox_of_cell = None
            sample_born = None
            if use_dense_sample and self.voxelize and scene_coords.shape[0] != 0:
                sample_vox_of_cell, sample_born = dense_sample_voxel_map(
                    dense_sample_pts, window_end,
                    Hf, Wf,
                    all_p2vs, all_p2vs_len, scene_coords.shape[0], scene_coords.device,
                )

            pred_coords = None
            for it in range(num_iters):
                scene_coords = scene_coords.detach().clone()
                query_coords = query_coords.detach().clone()
                if scene_src_coords is not None:
                    scene_src_coords = scene_src_coords.detach().clone()
                    query_src_coords = query_src_coords.detach().clone()

                if disable_backprop_through_time:
                    scene_feats = scene_feats.detach().clone()
                    query_feats = query_feats.detach().clone()

                    scene_waft_feats = scene_waft_feats.detach().clone()
                    query_waft_coords = query_waft_coords.detach().clone()
                    query_waft_feats = query_waft_feats.detach().clone()

                    if track_traj_coords_seq is not None and track_traj_feats_seq is not None:
                        track_traj_coords_seq = track_traj_coords_seq.detach().clone()
                        track_traj_feats_seq = track_traj_feats_seq.detach().clone()
                if it == 0:
                    scene_src_coords = scene_coords.detach().clone()
                    query_src_coords = query_coords.detach().clone()

                    # reset merged trajectory-refiner recurrent state each window
                    track_traj_coords_seq = None
                    track_traj_feats_seq = None
                    scene_dyn_mask = None

                # After iter 0, collapse per-frame duplicates onto predicted
                # endpoints so later iters run on a smaller token set. Every
                # point in the trajectory-refiner subset (dynamics + sampled
                # dense cells) survives 1:1 and keeps its relative order, so
                # scene_track_mask.nonzero() is unchanged row-for-row and
                # track_traj_coords_seq / track_traj_feats_seq carry over as-is.
                if (
                    it == 1
                    and self.voxelize_on_endpoints
                    and self.voxelize
                    and scene_coords.shape[0] != 0
                    and all_p2vs
                ):
                    (
                        scene_coords,
                        scene_feats,
                        scene_waft_feats,
                        scene_src_coords,
                        scene_src_time,
                        scene_ever_dynamic,
                        scene_dyn_mask,
                        sample_vox_of_cell,
                    ) = self._collapse_scene_on_predicted_endpoints(
                        scene_coords,
                        scene_feats,
                        scene_waft_feats,
                        scene_src_coords,
                        scene_src_time,
                        scene_ever_dynamic,
                        scene_dyn_mask,
                        sample_vox_of_cell,
                        all_p2vs,
                        all_p2vs_len,
                        all_coord_residuals,
                    )

                ##### build position embeddings for scene and query points ####
                time_emb_src = self.time_enc_src(time)
                time_emb_tgt = self.time_enc_tgt(time)

                all_feats = torch.cat([scene_feats, query_feats], dim=0)
                all_coords = torch.cat([scene_coords, query_coords], dim=0)
                all_src_coords = torch.cat([scene_src_coords, query_src_coords], dim=0)
                all_src_time = torch.cat([scene_src_time, query_src_time], dim=0)
                all_waft_feats = torch.cat([scene_waft_feats, query_waft_feats], dim=0)

                with autocast("cuda", enabled=False):
                    if tgt_time_step_dense is not None and tgt_time_step_dense[0] is not None:
                        # tgt_time_step_dense is given on the image axis.
                        tgt_time_step_scene = (
                            (tgt_time_step_dense[0][i] - window_start) // V
                        ).int()
                    else:
                        tgt_time_step_scene = traj_len - 1
                    all_pos_embeds = self.pe_tgt_xyz(all_coords[None].float()).squeeze(0).to(all_coords.dtype)
                    all_src_pos_embeds = self.pe_src_xyz(all_src_coords[None].float()).squeeze(0).to(all_src_coords.dtype)
                    all_pos_embeds = all_pos_embeds + all_src_pos_embeds + time_emb_tgt[tgt_time_step_scene][None] \
                        + time_emb_src[all_src_time]

                (
                    all_feats,
                    all_pred_coords,
                    all_pred_visibs,
                    all_pred_confs,
                    all_dynamic_logits,
                ) = self.refine_endpoints(
                    all_feats,
                    all_coords,
                    all_pos_embeds,
                    _cam0(intrinsics_scaled[:, window_start:window_end]),
                    _cam0(extrinsics[:, window_start:window_end]),
                    _cam0(image_feats_w),
                    all_waft_feats,
                    _cam0(frame_pointmaps),
                    tgt_time_step_scene=tgt_time_step_scene,
                )

                # Use an explicit split index so that num_query_points == 0 yields
                # an empty query slice (Python's [-0:] aliases to [0:], which would
                # incorrectly grab the full tensor).
                _split = all_feats.shape[0] - num_query_points
                scene_feats = all_feats[:_split]
                query_feats = all_feats[_split:]
                scene_pred_coords = all_pred_coords[:_split]
                query_pred_coords = all_pred_coords[_split:]
                scene_pred_visibs = all_pred_visibs[:_split]
                query_pred_visibs = all_pred_visibs[_split:]
                scene_pred_confs = all_pred_confs[:_split]
                query_pred_confs = all_pred_confs[_split:]
                scene_dynamic_logits = all_dynamic_logits[:_split]
                query_dynamic_logits = all_dynamic_logits[_split:]

                logger.debug(
                    "[POINTS] window=(%d,%d) iter=%d scene=%d query=%d num_query_points=%d",
                    window_start, window_end, it,
                    scene_feats.shape[0], query_feats.shape[0], num_query_points,
                )

                if self.voxelize and any(s is not None for s in subsample_alive_list):
                    dense_mask = compute_subsample_dense_mask(
                        all_p2vs, all_p2vs_len, subsample_alive_list,
                    )
                    dense_mask = rearrange(
                        dense_mask, "(b t h w) -> b t h w",
                        b=B, t=window_end,
                        h=Hf, w=Wf,
                    )
                else:
                    dense_mask = None
                    
                # ---- assemble the scene "track" subset for the trajectory refiner ----
                # It is the union of (a) the dynamic scene points (eval-only dense
                # tracking) and (b) the subsampled dense cells (full-track supervision).
                # We run ONE traj_decoder pass over cat([scene_track ; query]) because
                # points only attend to the shared scene_context, not to each other.
                scene_dyn_mask_cur = None
                if scene_dyn_mask is None and (
                    (self.dense_tracking_at_eval and not self.training)
                    or self.voxelize_on_endpoints
                ):
                    scene_dyn_mask = scene_dynamic_logits.sigmoid() > self.dynamic_threshold
                    if logger.isEnabledFor(logging.DEBUG):
                        num_scene_dyn = int(scene_dyn_mask.sum().item())
                        logger.debug(
                            "[POINTS] window=(%d,%d) iter=%d num_scene_dyn=%d",
                            window_start, window_end, it, num_scene_dyn,
                        )
                if self.dense_tracking_at_eval and not self.training:
                    scene_dyn_mask_cur = scene_dyn_mask
                    if compress_active:
                        scene_ever_dynamic = scene_ever_dynamic | scene_dyn_mask

                scene_sample_mask = None
                sample_valid_cells = None
                sample_valid_idx = None
                dense_sample_cell_logits = None
                if use_dense_sample and sample_vox_of_cell is not None:
                    sample_valid_cells = sample_born & (sample_vox_of_cell >= 0)
                    # One nonzero for valid cells; integer index into voxel ids
                    # (avoids repeated bool-index host syncs).
                    sample_valid_idx = sample_valid_cells.nonzero(as_tuple=False).reshape(-1)
                    sample_vox_idx = sample_vox_of_cell.index_select(0, sample_valid_idx)
                    scene_sample_mask = torch.zeros(
                        scene_feats.shape[0], dtype=torch.bool, device=scene_feats.device
                    )
                    scene_sample_mask[sample_vox_idx] = True

                    # Per-cell SD logit, read off the same voxel the cell's track
                    # comes from. dense_sample_voxel_map walks the p2v chain that
                    # the dense unvoxelize path walks, so this equals gathering
                    # the (B, T, H, W, T) dense logits at these cells without
                    # materializing them.
                    dense_sample_cell_logits = scene_dynamic_logits.new_full(
                        (num_dense_sample_pts,), -1000.0
                    )
                    dense_sample_cell_logits.index_put_(
                        (sample_valid_idx,),
                        scene_dynamic_logits.index_select(0, sample_vox_idx),
                    )

                if scene_dyn_mask_cur is not None and scene_sample_mask is not None:
                    scene_track_mask = scene_dyn_mask_cur | scene_sample_mask
                elif scene_dyn_mask_cur is not None:
                    scene_track_mask = scene_dyn_mask_cur
                else:
                    scene_track_mask = scene_sample_mask  # may be None

                scene_dyn_trajs_preds = None
                scene_dyn_trajs_visibs = None
                scene_dyn_trajs_confs = None
                dense_sample_cell_tracks = None
                dense_sample_cell_visibs = None
                dense_sample_valid_cells = None

                run_merged = (num_query_points > 0) or (scene_track_mask is not None)
                if run_merged:
                    tgt_step = int(tgt_time_step_scene) if not torch.is_tensor(tgt_time_step_scene) else tgt_time_step_scene

                    # Convert bool track mask -> indices once; reuse for gathers.
                    scene_track_idx = None
                    if scene_track_mask is not None:
                        scene_track_idx = scene_track_mask.nonzero(as_tuple=False).reshape(-1)
                        merged_feats_state = torch.cat(
                            [scene_feats.index_select(0, scene_track_idx), query_feats], dim=0
                        )
                        merged_src_coords = torch.cat(
                            [scene_src_coords.index_select(0, scene_track_idx), query_src_coords], dim=0
                        )
                        merged_pred_coords = torch.cat(
                            [scene_pred_coords.index_select(0, scene_track_idx), query_pred_coords], dim=0
                        )
                        merged_src_time = torch.cat(
                            [scene_src_time.index_select(0, scene_track_idx), query_src_time], dim=0
                        )
                        merged_waft = torch.cat(
                            [scene_waft_feats.index_select(0, scene_track_idx), query_waft_feats], dim=0
                        )
                    else:
                        merged_feats_state = query_feats
                        merged_src_coords = query_src_coords
                        merged_pred_coords = query_pred_coords
                        merged_src_time = query_src_time
                        merged_waft = query_waft_feats

                    # recurrent pre-update on the merged buffer
                    if track_traj_coords_seq is not None:
                        track_traj_coords_seq[:, tgt_step] = merged_pred_coords
                        valid_src_m = merged_src_time >= 0
                        # Always apply where (avoids valid_src_m.any() host sync).
                        idx_m = merged_src_time.clamp_min(0)
                        N_m = track_traj_coords_seq.shape[0]
                        arange_m = torch.arange(N_m, device=track_traj_coords_seq.device)
                        existing_m = track_traj_coords_seq[arange_m, idx_m]
                        track_traj_coords_seq[arange_m, idx_m] = torch.where(
                            valid_src_m[:, None], merged_src_coords, existing_m
                        )

                    merged_out, merged_feats_out, track_traj_feats_seq = self.traj_decoder(
                        traj_feats_state=merged_feats_state,
                        src_time_step_coords=merged_src_coords,
                        tgt_time_step_coords=merged_pred_coords,
                        src_time_step=merged_src_time,
                        tgt_time_step=tgt_time_step_scene,
                        scene_context=all_feats,
                        image_feats=_cam0(image_feats_w),
                        window_len=traj_len,
                        extrinsics=_cam0(extrinsics[:, window_start:window_end]),
                        intrinsics_scaled=_cam0(
                            intrinsics_scaled[:, window_start:window_end]
                        ),
                        waft_feats=merged_waft,
                        scene_coords=_cam0(frame_pointmaps),
                        initial_traj=track_traj_coords_seq,
                        initial_traj_feats=track_traj_feats_seq,
                    )
                    track_traj_coords_seq = merged_out[..., :3]

                    # split merged output back into scene-track and query parts.
                    # explicit split index avoids the [-0:] full-tensor aliasing bug.
                    _split_t = merged_out.shape[0] - num_query_points
                    scene_track_out = merged_out[:_split_t]
                    query_traj_out = merged_out[_split_t:]
                    scene_track_feats = merged_feats_out[:_split_t]
                    query_feats = merged_feats_out[_split_t:]

                    if num_query_points > 0:
                        query_trajs_preds = query_traj_out[..., :3]
                        query_trajs_visibs = query_traj_out[..., 3]
                        query_trajs_confs = query_traj_out[..., 4]
                    else:
                        query_trajs_preds = query_coords[:, None].repeat(1, traj_len, 1)
                        query_trajs_visibs = query_pred_visibs[:, None].repeat(1, traj_len)
                        query_trajs_confs = query_pred_confs[:, None].repeat(1, traj_len)

                    # row index of each voxel within the (sorted) track subset
                    if scene_track_mask is not None:
                        sel_pos = torch.cumsum(scene_track_mask.long(), dim=0) - 1

                    # dynamic-scene subset: extract from the merged output + snapshot
                    if scene_dyn_mask_cur is not None:
                        dyn_idx_cur = scene_dyn_mask_cur.nonzero(as_tuple=False).reshape(-1)
                        dyn_rows = sel_pos.index_select(0, dyn_idx_cur)
                        scene_dyn_traj_out = scene_track_out.index_select(0, dyn_rows)
                        scene_dyn_trajs_preds = scene_dyn_traj_out[..., :3]
                        scene_dyn_trajs_visibs = scene_dyn_traj_out[..., 3]
                        scene_dyn_trajs_confs = scene_dyn_traj_out[..., 4]

                        if it == num_iters - 1 and not self.training:
                            if pred.dense_track_3d is None:
                                pred.dense_track_3d = []
                            # window_start/window_end are image indices, so the
                            # trajectories are replicated per view here.
                            pred.dense_track_3d.append(DenseTrackWindow(
                                window_start=int(window_start),
                                window_end=int(window_end),
                                sd_mask=scene_dyn_mask_cur.detach(),
                                sd_trajs=_to_flat(scene_dyn_trajs_preds.detach(), 1),
                                sd_visibs=_to_flat(scene_dyn_trajs_visibs.detach(), 1),
                                sd_confs=_to_flat(scene_dyn_trajs_confs.detach(), 1),
                                sd_logits=scene_dynamic_logits.detach(),
                                static_coords=scene_pred_coords.detach(),
                                static_visibs=scene_pred_visibs.detach(),
                                static_confs=scene_pred_confs.detach(),
                                p2vs=[p.detach() for p in all_p2vs],
                                p2v_lens=list(all_p2vs_len),
                                coord_residuals=[
                                    r.detach() for r in all_coord_residuals
                                ],
                            ))

                    # sampled dense cells: gather each cell's full predicted track
                    if use_dense_sample and scene_sample_mask is not None:
                        cell_rows = torch.zeros(
                            num_dense_sample_pts, dtype=torch.long, device=scene_track_out.device
                        )
                        # Reuse sample_valid_idx / sample_vox_idx (no extra bool-index syncs).
                        cell_rows.index_put_(
                            (sample_valid_idx,),
                            sel_pos.index_select(0, sample_vox_idx),
                        )
                        dense_sample_cell_tracks = scene_track_out.new_zeros(num_dense_sample_pts, traj_len, 3)
                        dense_sample_cell_visibs = scene_track_out.new_zeros(num_dense_sample_pts, traj_len)
                        if sample_valid_idx.numel() > 0:
                            _gathered = scene_track_out.index_select(0, cell_rows.index_select(0, sample_valid_idx))
                            dense_sample_cell_tracks.index_put_(
                                (sample_valid_idx,), _gathered[..., :3]
                            )
                            dense_sample_cell_visibs.index_put_(
                                (sample_valid_idx,), _gathered[..., 3]
                            )
                        dense_sample_valid_cells = sample_valid_cells
                else:
                    query_trajs_preds = query_coords[:, None].repeat(1, traj_len, 1)
                    query_trajs_visibs = query_pred_visibs[:, None].repeat(1, traj_len)
                    query_trajs_confs = query_pred_confs[:, None].repeat(1, traj_len)

                ######## convert this into proper train data  ############
                scene_coords = scene_pred_coords
                query_coords = query_pred_coords

                del scene_pred_coords, query_pred_coords

                _effective_we = window_end - pad if window_end > T - pad else window_end
                _effective_ws = _effective_we - window_start

                train_data = TrainData(
                    coords=None,
                    visibs=None,
                    confs=None,
                    iter_idx=it,
                    frame_range=(window_start, _effective_we),
                    tgt_time_step_scene=tgt_time_step_scene if (tgt_time_step_dense is None or tgt_time_step_dense[0] is None) else tgt_time_step_dense[0][i],
                    window_num=i,
                    prev_src_time_step=window_ends[i - 1] - 1 if i > 0 else None
                )

                if self.use_query_points:
                    N_q = pred.query_coords.shape[2]
                    _dev = query_trajs_preds.device
                    _dtype = query_trajs_preds.dtype

                    train_data.query_coords = torch.zeros(B, _effective_ws, N_q, 3, device=_dev, dtype=_dtype)
                    train_data.query_visibs = torch.zeros(B, _effective_ws, N_q, device=_dev, dtype=_dtype)
                    train_data.query_confs = torch.ones(B, _effective_ws, N_q, device=_dev, dtype=_dtype)

                    # train_data / pred are indexed on the image axis.
                    _qc = _to_flat(query_trajs_preds.permute(1, 0, 2), 0)[:_effective_ws]
                    _qv = _to_flat(query_trajs_visibs.permute(1, 0), 0)[:_effective_ws]
                    _qconf = _to_flat(query_trajs_confs.permute(1, 0), 0)[:_effective_ws]
                    train_data.query_coords[..., query_fill_idxs, :] = _qc[None]
                    train_data.query_visibs[..., query_fill_idxs] = _qv[None]
                    train_data.query_confs[..., query_fill_idxs] = _qconf[None]

                    pred.query_coords[:, window_start:window_end] = train_data.query_coords.detach()
                    pred.query_visibs[:, window_start:window_end] = train_data.query_visibs.detach()

                    if query_dynamic_logits is not None:
                        train_data.query_dynamic_logits = torch.full(
                            (B, N_q), -1000.0, device=_dev, dtype=_dtype
                        )
                        train_data.query_dynamic_logits[:, query_fill_idxs] = query_dynamic_logits[None].to(_dtype)
                        if pred.query_dynamic_logits is not None:
                            pred.query_dynamic_logits[:, window_start:window_end] = (
                                train_data.query_dynamic_logits.detach()[:, None].expand(
                                    -1, window_end - window_start, -1
                                )
                            )

                    train_data.query_coord_at_tgt = torch.zeros(B, N_q, 3, device=_dev, dtype=_dtype)
                    train_data.query_coord_at_tgt[:, query_fill_idxs, :] = query_coords[None].to(_dtype)

                if use_dense_sample and dense_sample_cell_tracks is not None:
                    _dev = dense_sample_cell_tracks.device
                    _dtype = dense_sample_cell_tracks.dtype
                    train_data.dense_sample_coords = torch.zeros(
                        B, _effective_ws, num_dense_sample_pts, 3, device=_dev, dtype=_dtype
                    )
                    train_data.dense_sample_visibs = torch.zeros(
                        B, _effective_ws, num_dense_sample_pts, device=_dev, dtype=_dtype
                    )
                    train_data.dense_sample_coords[0] = _to_flat(
                        dense_sample_cell_tracks.permute(1, 0, 2), 0
                    )[:_effective_ws]
                    train_data.dense_sample_visibs[0] = _to_flat(
                        dense_sample_cell_visibs.permute(1, 0), 0
                    )[:_effective_ws]
                    train_data.dense_sample_valid_cells = dense_sample_valid_cells
                    pred.dense_sample_coords[:, window_start:window_end] = train_data.dense_sample_coords.detach()
                    pred.dense_sample_visibs[:, window_start:window_end] = train_data.dense_sample_visibs.detach()

                # Kept outside the dense_sample_cell_tracks branch: the logits do
                # not come from the trajectory refiner, so they exist even when
                # the merged pass did not run.
                if dense_sample_cell_logits is not None and pred.dense_sample_dynamic_logits is not None:
                    train_data.dense_sample_dynamic_logits = dense_sample_cell_logits[None].to(
                        pred.dense_sample_dynamic_logits.dtype
                    )
                    pred.dense_sample_dynamic_logits[:, window_start:window_end] = (
                        train_data.dense_sample_dynamic_logits.detach()[:, None].expand(
                            B, window_end - window_start, -1
                        )
                    )

                # now that predictions are stored, update the state vectors with the latest predictions
                if num_query_points > 0:
                    tgt = int(tgt_time_step_scene) if not torch.is_tensor(tgt_time_step_scene) else tgt_time_step_scene
                    query_coords = query_trajs_preds[:, tgt]

                    valid_src = query_src_time >= 0
                    idx = query_src_time.clamp_min(0)
                    new_src = query_trajs_preds[
                        torch.arange(query_trajs_preds.shape[0], device=query_trajs_preds.device), idx
                    ]
                    query_src_coords = torch.where(valid_src[:, None], new_src, query_src_coords)

                if scene_dyn_mask is not None and scene_dyn_trajs_preds is not None:
                    tgt_sd = int(tgt_time_step_scene) if not torch.is_tensor(tgt_time_step_scene) else tgt_time_step_scene
                    dyn_idx = scene_dyn_mask.nonzero(as_tuple=False).reshape(-1)
                    scene_coords.index_copy_(0, dyn_idx, scene_dyn_trajs_preds[:, tgt_sd])

                    sd_src = scene_src_time.index_select(0, dyn_idx)
                    valid_sd = sd_src >= 0
                    idx_sd = sd_src.clamp_min(0)
                    N_sd = scene_dyn_trajs_preds.shape[0]
                    new_src_sd = scene_dyn_trajs_preds[
                        torch.arange(N_sd, device=scene_dyn_trajs_preds.device), idx_sd
                    ]
                    existing_src_sd = scene_src_coords.index_select(0, dyn_idx)
                    scene_src_coords.index_copy_(
                        0, dyn_idx, torch.where(valid_sd[:, None], new_src_sd, existing_src_sd)
                    )

                # unvoxelize the scene points and store them back
                if self.training or not self.mem_efficient:
                    if self.voxelize:
                        if self.residual_unvoxelize:
                            pred_coords = unvoxelize_coords_with_residuals(
                                scene_coords,
                                all_p2vs,
                                all_coord_residuals,
                                all_p2vs_len,
                            )
                        else:
                            pred_coords = unvoxelize_pc_with_p2v_lists(
                                scene_coords, all_p2vs, all_p2vs_len
                            )
                        pred_coords = rearrange(pred_coords, "(b t h w) c -> b t h w c", b=B, t=window_end, h=Hf, w=Wf)

                        pred_visibs = (
                            unvoxelize_pc_with_p2v_lists(
                                scene_pred_visibs, all_p2vs, all_p2vs_len
                            )
                        )
                        pred_visibs = rearrange(pred_visibs, "(b t h w) -> b t h w", b=B, t=window_end, h=Hf, w=Wf)

                        pred_confs = (
                            unvoxelize_pc_with_p2v_lists(
                                scene_pred_confs, all_p2vs, all_p2vs_len
                            )
                        )
                        pred_confs = rearrange(pred_confs, "(b t h w) -> b t h w", b=B, t=window_end, h=Hf, w=Wf)

                        if scene_dynamic_logits is not None:
                            pred_dense_dynamic_logits = (
                                unvoxelize_pc_with_p2v_lists(
                                    scene_dynamic_logits, all_p2vs, all_p2vs_len
                                )
                            )
                            pred_dense_dynamic_logits = rearrange(pred_dense_dynamic_logits, "(b t h w) -> b t h w", b=B, t=window_end, h=Hf, w=Wf)

                    else:
                        pred_coords = scene_coords
                        pred_coords = rearrange(pred_coords, "(b t h w) c -> b t h w c", b=B, t=window_end, h=Hf, w=Wf)

                        pred_visibs = scene_pred_visibs
                        pred_visibs = rearrange(pred_visibs, "(b t h w) -> b t h w", b=B, t=window_end, h=Hf, w=Wf)
                        pred_confs = scene_pred_confs
                        pred_confs = rearrange(pred_confs, "(b t h w) -> b t h w", b=B, t=window_end, h=Hf, w=Wf)

                        if scene_dynamic_logits is not None:
                            pred_dense_dynamic_logits = scene_dynamic_logits
                            pred_dense_dynamic_logits = rearrange(pred_dense_dynamic_logits, "(b t h w) -> b t h w", b=B, t=window_end, h=Hf, w=Wf)

                    train_data.coords = [pred_coords]
                    train_data.visibs = pred_visibs
                    train_data.confs = pred_confs
                    train_data.sampled_mask = dense_mask
                    _dtype = pred_coords.dtype
                    if scene_dynamic_logits is not None:
                        train_data.dense_dynamic_logits = pred_dense_dynamic_logits.to(_dtype)

                    del pred_visibs, query_pred_visibs, pred_confs, query_pred_confs, scene_pred_confs, pred_coords

                    pred.coords[:, 0:window_end] = (
                        train_data.coords[0].detach()
                    )
                    pred.visibs[:, 0:window_end] = (
                        train_data.visibs.detach()
                    )
                    if train_data.dense_dynamic_logits is not None and pred.dense_dynamic_logits is not None:
                        # cur: (B, window_end, H, W) — one logit per source cell.
                        # Broadcast onto target axis for this window (same pattern as
                        # reconstruct_dense_tracks_3d with dynamic_logits_by_target).
                        cur = train_data.dense_dynamic_logits.detach()
                        _B, _we, _H, _W = cur.shape
                        _T_win = window_end - window_start
                        pred.dense_dynamic_logits[:, 0:window_end, :, :, window_start:window_end] = (
                            cur[:, :, :, :, None].expand(_B, _we, _H, _W, _T_win)
                        )
                    if self.use_query_points:
                        pred.query_coords[:, window_start:window_end] = (
                            train_data.query_coords.detach()
                        )
                        pred.query_visibs[:, window_start:window_end] = (
                            train_data.query_visibs.detach()
                        )

                yield train_data
                del train_data

            if self.memory_merge_mode == "srctgt":
                # Source coordinate and target position per voxel, as the
                # endpoint refiner saw them in this window's last iteration.
                mem_merge_key = torch.stack(
                    [scene_src_coords.detach(), scene_coords.detach()], dim=1
                )

            del image_feats_w, points_win, frame_pointmaps

        torch.set_grad_enabled(_prev_grad_state)
        _feat_cache.clear()

        yield pred

    # Note: DDP does support using a dataclass as a return type, but other return types like Generator may cause wrong gradients to be computed.
    @cast_float32()
    @profile
    def forward(
        self,
        *,
        rgb_obs: torch.Tensor,
        depth_obs: torch.Tensor,
        num_iters: int,
        query_point: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_feats: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        disable_backprop_through_time: bool = False,
        dense_norm_mask: Optional[torch.Tensor] = None,
        tgt_time_step_dense: Optional[torch.Tensor] = None,
        dense_sample_points: Optional[torch.Tensor] = None,
        warmup_frames: int = 0,
        num_views: Optional[int] = None,
    ) -> Tuple[Prediction, List[TrainData]]:
        pred = None
        train_data = []
        for output in self.streaming_forward(
            rgb_obs=rgb_obs,
            depth_obs=depth_obs,
            num_iters=num_iters,
            query_point=query_point,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_feats=image_feats,
            chunk_size=chunk_size,
            disable_backprop_through_time=disable_backprop_through_time,
            dense_norm_mask=dense_norm_mask,
            tgt_time_step_dense=tgt_time_step_dense,
            dense_sample_points=dense_sample_points,
            warmup_frames=warmup_frames,
            num_views=num_views,
        ):
            if isinstance(output, Prediction):
                assert pred is None, "Internal error"
                pred = output
            else:
                train_data.append(output)
        assert pred is not None, "Internal error"
        return pred, train_data
