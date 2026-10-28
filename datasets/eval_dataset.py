# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import cv2
import math
import ipdb
import numpy as np
import tap
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from hydra import compose, initialize_config_dir
from line_profiler import profile
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..utils.alltrack3r_utils import compute_dense_gt_tensor

from ..utils.common_utils import setup_logger
from . import data_ops
from .base_dataset import BaseDataset
from .datatypes import RawSliceData, SliceData
from .providers.base_provider import BaseDataProvider
from .utils.dynamic_labels import dynamic_extent_l2
from .utils.crop_utils import CropArgs, get_crop_args
from .utils.geometry import world_to_local, local_to_world


st = ipdb.set_trace

_ProviderInfo = TypedDict(
    "_ProviderInfo",
    {
        "name": str,
        "weight": Union[float, int],
        "stride": int,
        "config": DictConfig,
        "override_anno": Optional[str],
    },
)

logger = logging.getLogger(__name__)


class EvalDataset(BaseDataset):
    def __init__(
        self,
        resolution: Tuple[int, int],
        provider_config: _ProviderInfo,
        query_mode: Literal["first_visible", "pass_through"],
        # first_n_seqs: Optional[int] = None,
        sample_n_seqs: Optional[int] = None,
        num_threads: int = 16,
        seed: int = 42,
        transform: Optional[ListConfig] = None,
        scale_factor: int = 4,
        epoch_len: Optional[int] = None,
        frames_per_sample: int = -1,
        scale_and_cam0_align: bool = False,
        use_gt_poses: bool = False,
        use_gt_depths: bool = False,
        compute_dense_trajs_3d: bool = False,
        compute_tmaps: bool = False,
        make_all_dense_static: bool = False,
        lift_full_geometry: bool = True,
        eval_on_key_frames: Optional[List[int]] = None,
        model_seq_len: int = 16,
        half_window_overlap: bool = False,
        to_local_alltracker: bool = False,
        traj_sampling: Optional[DictConfig] = None,
        subsample_query_trajs: bool = False,
        num_dense_sample_points: int = 0,
    ):
        self.resolution = resolution
        self.data_provider: BaseDataProvider = BaseDataProvider.from_config(
            provider_config["config"],
            name=provider_config["name"],
            override_anno=provider_config["override_anno"],
        )

        self.seq_lens = self.data_provider.seq_lens
        if self.seq_lens is None:
            self.seq_lens = self.data_provider.load_seq_lens()
        self.stride = provider_config["stride"]
        self.query_mode = query_mode
        self.sample_n_seqs = sample_n_seqs
        self.seed = seed
        self.scale_and_cam0_align = scale_and_cam0_align
        # VGGT-omega already emits unit-scale geometry in the cam0 frame, so a
        # second median-distance normalize plus cam0-align would distort voxel
        # size. Datasets with override_anno=null (GT geometry) keep the flag.
        anno_cfg = getattr(self.data_provider, "anno_config", None)
        anno_name = ""
        if anno_cfg is not None:
            anno_name = str(
                OmegaConf.select(anno_cfg, "annotator_name", default="") or ""
            )
        if "vggt_omega" in anno_name.lower():
            self.scale_and_cam0_align = False
        self.use_gt_poses = use_gt_poses
        self.use_gt_depths = use_gt_depths
        self.compute_dense_trajs_3d = compute_dense_trajs_3d
        self.compute_tmaps = compute_tmaps
        self.is_overfit = provider_config["config"].get("is_overfit", False)
        self.cache = None
        self.scale_factor = scale_factor
        self.make_all_dense_static = make_all_dense_static
        self.dynamic_filter_threshold = provider_config["config"].get(
            "dynamic_filter_threshold", 0.1
        )
        self.frames_per_sample = frames_per_sample
        self.lift_full_geometry = lift_full_geometry
        self.eval_on_key_frames = eval_on_key_frames
        self.model_seq_len = model_seq_len
        self.half_window_overlap = half_window_overlap
        self.to_local_alltracker = to_local_alltracker
        self.traj_sampling_cfg = traj_sampling
        self.subsample_query_trajs = subsample_query_trajs
        self.num_dense_sample_points = num_dense_sample_points
        self.dataset_name = provider_config["name"]
        if transform is not None:
            self.transform = data_ops.from_config(transform)
        else:
            self.transform = None

        if sample_n_seqs is not None:
            rng = np.random.default_rng(seed)
            self.seq_ids = rng.choice(len(self.seq_lens), sample_n_seqs, replace=False)
        else:
            self.seq_ids = np.arange(len(self.seq_lens))

        assert query_mode in [
            "first_visible",
            "pass_through",
        ], f"Invalid query mode: {query_mode}"

        self.num_threads = num_threads
        self.epoch_len = epoch_len
        if epoch_len is None or epoch_len == -1:
            self.epoch_len = len(self.seq_ids)

        logger.info(
            f"Initialized eval dataset with {sum(self.seq_lens[seq_id] for seq_id in self.seq_ids)} frames and {len(self.seq_lens)} sequences"
        )

    def set_epoch(self, epoch: int):
        pass

    def _postprocess_slice(
        self, sample_id: int, raw_slice_data: RawSliceData
    ) -> Optional[SliceData]:
        crop_args = CropArgs(
            crop_start=(0, 0),
            crop_end=(raw_slice_data.rgbs[0].shape[:2]),
            src_resolution=(raw_slice_data.rgbs[0].shape[:2]),
            target_resolution=self.resolution,
        )

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            gt_intrinsics = crop_args.update_intrinsics(raw_slice_data.gt_intrinsics)
            est_intrinsics = crop_args.update_intrinsics(raw_slice_data.est_intrinsics)
            rgbs = crop_args.crop_img(
                raw_slice_data.rgbs, interp_mode=cv2.INTER_LANCZOS4, executor=executor
            )
            gt_depths = crop_args.crop_img(
                raw_slice_data.gt_depths,
                interp_mode=cv2.INTER_NEAREST_EXACT,
                executor=executor,
            )
            est_depths = crop_args.crop_img(
                raw_slice_data.est_depths,
                interp_mode=cv2.INTER_NEAREST_EXACT,
                executor=executor,
            )
            if raw_slice_data.segmentation is not None:
                segmentation = crop_args.crop_img(
                    raw_slice_data.segmentation,
                    interp_mode=cv2.INTER_NEAREST_EXACT,
                    executor=executor,
                )
            else:
                segmentation = None
            assert rgbs[0].shape[:2] == self.resolution, "Internal Error"

        gt_query_point = raw_slice_data.gt_query_point
        est_query_point = raw_slice_data.est_query_point
        gt_trajs_3d = raw_slice_data.gt_trajs_3d
        est_trajs_3d = raw_slice_data.est_trajs_3d
        gt_trajs_depths = raw_slice_data.gt_trajs_depths
        visibs = raw_slice_data.visibs
        valids = raw_slice_data.valids
        gt_extrinsics = raw_slice_data.gt_extrinsics.copy()
        est_extrinsics = raw_slice_data.est_extrinsics.copy()

        _num_trajs = raw_slice_data.gt_trajs_3d.shape[1]
        gt_traj_local = self._apply_homo_transform(
            raw_slice_data.gt_trajs_3d,
            repeat(raw_slice_data.gt_extrinsics, "t i j -> t n i j", n=_num_trajs),
        )
        gt_traj_2d = self._apply_homo_transform(
            gt_traj_local,
            repeat(gt_intrinsics, "t i j -> t n i j", n=_num_trajs),
            homo_input=True,
        )
        
        # for drivetrack, we don't have any GT depths
        # we fill it here from trajs. 
        if self.dataset_name == "tapvid3d" and np.all(np.isclose(gt_depths, 0.0)):
            traj_x = np.rint(gt_traj_2d[..., 0]).astype(np.int32)
            traj_y = np.rint(gt_traj_2d[..., 1]).astype(np.int32)
            traj_t = np.broadcast_to(
                np.arange(gt_traj_2d.shape[0], dtype=np.int32)[:, None],
                gt_traj_2d.shape[:2],
            )
            in_bounds = (
                (traj_x >= 0)
                & (traj_x < gt_depths.shape[2])
                & (traj_y >= 0)
                & (traj_y < gt_depths.shape[1])
                & visibs & valids
            )
            gt_depths[
                traj_t[in_bounds],
                traj_y[in_bounds],
                traj_x[in_bounds],
            ] = gt_trajs_depths[in_bounds]

        # shift to camera 0 coordinate system
        if self.scale_and_cam0_align:
            gt_trajs_3d, gt_extrinsics, gt_query_point = self._align_to_cam0(
                gt_trajs_3d, gt_extrinsics, gt_query_point
            )
            est_trajs_3d, est_extrinsics, est_query_point = self._align_to_cam0(
                est_trajs_3d, est_extrinsics, est_query_point
            )

        num_views = int(getattr(raw_slice_data, "num_views", None) or 1)

        if self.query_mode == "first_visible":
            # assert (
            #     gt_query_point is None
            # ), "You should not override query point provided by the eval dataset"

            # With several cameras the time axis holds num_views images per
            # timestep. Restricting queries to camera 0 puts query times on
            # timestep boundaries and keeps the evaluated track set the same one
            # the single-view run sees.
            queryable = raw_slice_data.visibs & raw_slice_data.valids
            if num_views > 1:
                cam0 = np.zeros(queryable.shape[0], dtype=bool)
                cam0[::num_views] = True
                queryable = queryable & cam0[:, None]

            to_query = queryable.any(0)
            if not to_query.any():
                # POD motion/OOB filters (or depth-range cuts) can wipe every
                # track in a long window; skip rather than crash later on empty
                # query tensors. __getitem__ retries the next index (wrap-safe).
                logger.warning(
                    "Skipping eval sample %d (%s): no valid+visible tracks",
                    sample_id,
                    getattr(raw_slice_data, "seq_name", "?"),
                )
                return None

            gt_trajs_3d = gt_trajs_3d[:, to_query]  # type: ignore
            est_trajs_3d = est_trajs_3d[:, to_query]  # type: ignore
            gt_traj_2d = gt_traj_2d[:, to_query]  # type: ignore
            gt_trajs_depths = gt_trajs_depths[:, to_query]  # type: ignore
            visibs = visibs[:, to_query]  # type: ignore
            valids = valids[:, to_query]  # type: ignore

            query_frames = queryable[:, to_query].argmax(0)
            gt_query_coords = gt_trajs_3d[query_frames, np.arange(gt_trajs_3d.shape[1])]
            est_query_coords = est_trajs_3d[
                query_frames, np.arange(est_trajs_3d.shape[1])
            ]
            query_point_2d = gt_traj_2d[query_frames, np.arange(gt_traj_2d.shape[1])]
            query_times = query_frames

            gt_query_point = np.concatenate(
                [query_frames[..., None].astype(np.float32), gt_query_coords], axis=-1
            )
            est_query_point = np.concatenate(
                [query_frames[..., None].astype(np.float32), est_query_coords], axis=-1
            )
        else:
            assert (
                gt_query_point is not None
            ), "query_point should be provided by the eval dataset when query_mode is pass_through"
            assert (
                est_query_point is not None
            ), "query_point should be provided by the eval dataset when query_mode is pass_through"

        H, W = rgbs[0].shape[:2]
        # Measure the extent over VALID frames only: coordinates on invalid
        # frames are not meaningful and one junk value labels a stationary point
        # dynamic. See docs/static_dynamic_labels.md.
        diff_l2 = dynamic_extent_l2(gt_trajs_3d, valids)
        is_dynamic = (diff_l2 > self.dynamic_filter_threshold).copy()

        gt_depths_downsampled = (
            F.interpolate(
                torch.tensor(gt_depths.copy()).unsqueeze(0),
                scale_factor=1.0 / self.scale_factor,
                mode="nearest-exact",
            )
            .squeeze(0)
            .numpy()
        )

        gt_intrinsics_downsampled = gt_intrinsics.copy()
        gt_intrinsics_downsampled[:, :2] /= self.scale_factor

        est_intrinsics_downsampled = est_intrinsics.copy()
        est_intrinsics_downsampled[:, :2] /= self.scale_factor

        dense_trajs_3d, dense_valids, dense_visibs, dense_dynamic, tgt_time_step_scenes = None, None, None, None, None
        dense_geom_raw, dense_src_idx_map = None, None
        if self.compute_dense_trajs_3d and num_views > 1:
            # Dense GT is (T, T, 3, H, W) over the flat time axis, so it grows
            # by num_views^2 and its target axis no longer means "timestep".
            raise ValueError(
                f"compute_dense_trajs_3d is not supported with num_views="
                f"{num_views}; pass "
                "+eval.dataset_overrides.compute_dense_trajs_3d=false"
            )
        if self.compute_dense_trajs_3d:
            if self.eval_on_key_frames is not None:
                time_steps = self.eval_on_key_frames
                
                # handle negative time steps
                time_steps = [t if t >= 0 else t + len(gt_trajs_3d) for t in time_steps]
                tgt_time_step_scenes = np.array(time_steps)
                
                # also add the time steps before the min one
                # model_seq_len = self.model_seq_len
                # num_frames = gt_trajs_3d.shape[0]
                
                # num_windows = math.ceil(num_frames / model_seq_len)
                # # for each window, except the last one, the tgt_time_step is the last frame of the window
                # # the last window's tgt_time_step is randomly sampled to be any frame in the window
                # tgt_time_step_scenes = []
                # for i in range(num_windows - 1):
                #     tgt_time_step_scenes.append(model_seq_len * i + model_seq_len - 1)
                
                # tgt_time_step_scenes.extend(time_steps)
                # tgt_time_step_scenes = np.array(tgt_time_step_scenes)
                    
            else:
                tgt_time_step_scenes = None
            
            dense_trajs_3d, dense_valids, dense_visibs, dense_dynamic, dense_geom_raw, dense_src_idx_map = (
                self._compute_dense_trajs_and_unproject(
                    gt_traj_2d, gt_trajs_3d, visibs, is_dynamic,
                    H, W, self.scale_factor, self.compute_tmaps,
                    gt_depths=gt_depths, gt_intrinsics=gt_intrinsics,
                    gt_extrinsics=gt_extrinsics,
                    gt_depths_downsampled=gt_depths_downsampled,
                    gt_intrinsics_downsampled=gt_intrinsics_downsampled,
                    make_all_dense_static=self.make_all_dense_static,
                    lift_full_geometry=self.lift_full_geometry,
                    time_steps=tgt_time_step_scenes,
                    dynamic_filter_threshold=self.dynamic_filter_threshold,
                    valids=valids,
                    window_size=self.model_seq_len,
                )
            )
            
            # we need to add back these times to make forward pass happy
            if tgt_time_step_scenes is not None:
                model_seq_len = self.model_seq_len
                num_frames = gt_trajs_3d.shape[0]

                # Match model window scheduling used by windowed forward.
                if self.half_window_overlap:
                    half_step = max(1, model_seq_len // 2)
                    window_ends = list(range(model_seq_len, num_frames + 1, half_step))
                    if not window_ends or window_ends[-1] < num_frames:
                        window_ends.append(num_frames)
                else:
                    num_windows = math.ceil(num_frames / model_seq_len)
                    window_ends = [min(model_seq_len * (i + 1), num_frames) for i in range(num_windows)]

                # Prepend deterministic "window-end" targets for all non-final windows.
                tgt_time_step_scenes_ = []
                for window_end in window_ends[:-1]:
                    tgt_time_step_scenes_.append(window_end - 1)
                
                tgt_time_step_scenes = np.concatenate([tgt_time_step_scenes_, tgt_time_step_scenes])

        est_depths_downsampled = (
            F.interpolate(
                torch.tensor(est_depths.copy()).unsqueeze(0),
                scale_factor=1.0 / self.scale_factor,
                mode="nearest-exact",
            )
            .squeeze(0)
            .numpy()
        )

        est_dense_trajs_3d = None
        # if self.compute_dense_trajs_3d:
        #     est_dense_trajs_3d, _, _, _, _ = compute_dense_gt_tensor(
        #         gt_traj_2d,
        #         est_trajs_3d,
        #         visibs & valids,
        #         is_dynamic,
        #         H,
        #         W,
        #         None,
        #         None,
        #         stride=self.scale_factor,
        #         skip_tmaps=not self.compute_tmaps,
        #     )

        # we want a conservative norm mask to avoid outliers
        if raw_slice_data.annotated and 'kubric' in self.dataset_name:
            norm_mask = (gt_depths < 20) & (gt_depths > 0.1) & (est_depths > 0.1) & (est_depths < 5)
            gt_norm_mask = norm_mask
            est_norm_mask = norm_mask
        elif raw_slice_data.annotated:
            est_norm_mask = (est_depths > 0.1) & (est_depths < 5)
            # est_depths_t = torch.from_numpy(np.ascontiguousarray(est_depths, dtype=np.float32))
            # from .utils.geometry import depth_edge
            # _depth_edge_mask = depth_edge(est_depths_t, rtol=0.03).numpy()
            # est_norm_mask = est_norm_mask & ~_depth_edge_mask
            gt_norm_mask = (gt_depths < 20) & (gt_depths > 0.1)
        else:
            norm_mask = (gt_depths < 20) & (gt_depths > 0.1)
            
            # synthverse has large valid depths
            if 'synthverse' in self.dataset_name:
                norm_mask = gt_depths > 0.05
            gt_norm_mask = norm_mask
            est_norm_mask = norm_mask
        norm_mask_gt_only = (gt_depths < 20) & (gt_depths > 0.1)
        if 'synthverse' in self.dataset_name:
            norm_mask_gt_only = gt_depths > 0.05

        gt_norm_mask_downsampled = (
            F.interpolate(
                torch.from_numpy(gt_norm_mask.astype(np.float32)).unsqueeze(0),
                scale_factor=1.0 / self.scale_factor,
                mode="nearest-exact",
            )
            .squeeze(0)
            .bool()
            .numpy()
        )
        est_norm_mask_downsampled = (
            F.interpolate(
                torch.from_numpy(est_norm_mask.astype(np.float32)).unsqueeze(0),
                scale_factor=1.0 / self.scale_factor,
                mode="nearest-exact",
            )
            .squeeze(0)
            .bool()
            .numpy()
        )

        gt_norm_factor = 1.0
        est_norm_factor = 1.0
        if self.scale_and_cam0_align:
            gt_norm_factor, gt_depths, gt_depths_downsampled, gt_trajs_3d, gt_extrinsics, dense_trajs_3d = (
                self._normalize_scene(
                    gt_extrinsics,
                    gt_depths,
                    gt_trajs_3d,
                    gt_depths_downsampled,
                    gt_intrinsics_downsampled,
                    gt_norm_mask_downsampled,
                    norm_mode="median_dis",
                    query_point=gt_query_point,
                    dense_trajs_3d=dense_trajs_3d,
                )
            )

            # if gt_trajs_depths is not None:
            #     gt_trajs_depths /= gt_norm_factor

            est_norm_factor, est_depths, est_depths_downsampled, est_trajs_3d, est_extrinsics, est_dense_trajs_3d = (
                self._normalize_scene(
                    est_extrinsics,
                    est_depths,
                    est_trajs_3d,
                    est_depths_downsampled,
                    est_intrinsics_downsampled,
                    est_norm_mask_downsampled,
                    norm_mode="median_dis",
                    query_point=est_query_point,
                    dense_trajs_3d=est_dense_trajs_3d,
                )
            )

        # Build grid-anchored full GT tracks for a subsample of dense cells.
        # Done before query subsampling so the winning-track index space still
        # matches the full sparse arrays. Only valid for datasets with dense
        # correspondence GT (kubric / dynamic_replica / point odyssey).
        dense_sample_points = None
        dense_sample_trajs_3d = None
        dense_sample_visibs = None
        dense_sample_valids = None
        dense_sample_dynamic = None
        _ds_name = self.dataset_name.lower()
        _supports_dense_sample = (
            "kubric" in _ds_name
            or "dynamic_replica" in _ds_name
            or "pod" in _ds_name
            or "point_odyssey" in _ds_name
        )
        if (
            _supports_dense_sample
            and self.compute_dense_trajs_3d
            and self.num_dense_sample_points > 0
            and dense_src_idx_map is not None
        ):
            _ds_rng = np.random.default_rng(self.seed + sample_id + 1234)
            _dense_sample = self._build_dense_sample_gt(
                dense_src_idx_map=dense_src_idx_map,
                gt_traj_3d=gt_trajs_3d,
                visibs=visibs,
                valids=valids,
                gt_trajs_depths=gt_trajs_depths,
                is_dynamic=is_dynamic,
                gt_depths_downsampled=gt_depths_downsampled,
                gt_intrinsics_downsampled=gt_intrinsics_downsampled,
                gt_extrinsics=gt_extrinsics,
                norm_mask_downsampled=gt_norm_mask_downsampled,
                num_points=self.num_dense_sample_points,
                rng=_ds_rng,
            )
            if _dense_sample is not None:
                (
                    dense_sample_points,
                    dense_sample_trajs_3d,
                    dense_sample_visibs,
                    dense_sample_valids,
                    dense_sample_dynamic,
                ) = _dense_sample

        # assert self.scale_and_cam0_align
        # from ..utils.vis_utils import log_depth_diff
        # for t in range(len(gt_depths)):
        #     log_depth_diff(est_depths[t], gt_depths[t], mask=norm_mask[t], max_diff=1.0)
        
        if self.use_gt_poses or self.use_gt_depths:
            # we need to recompute the est_traj_3d, est_dense_traj_3d, est_query_point and est_queries
            query_frames = est_query_point[..., 0].astype(np.int32)
            est_query_point_2d = world_to_local(
                est_query_point[:, 1:],
                est_intrinsics[query_frames],
                est_extrinsics[query_frames],
            )[..., :2]

            if self.use_gt_poses:
                est_extrinsics = gt_extrinsics

            if self.use_gt_depths:
                est_depths = gt_depths
                est_depths_downsampled = gt_depths_downsampled

            est_query_point_depths = est_depths[
                query_frames,
                est_query_point_2d[:, 1].astype(np.int32),
                est_query_point_2d[:, 0].astype(np.int32),
            ]
            est_query_points_local = np.concatenate(
                [
                    est_query_point_2d,
                    est_query_point_depths[:, None],
                ],
                axis=-1,
            )
            est_query_point[:, 1:] = local_to_world(
                est_query_points_local,
                est_intrinsics[query_frames],
                est_extrinsics[query_frames],
            )
        norm_mask_ = gt_norm_mask & est_norm_mask
        gt_depths_clone = np.copy(gt_depths)
        est_depths_clone = np.copy(est_depths)
        gt_depths_clone[~norm_mask_] = np.nan
        est_depths_clone[~norm_mask_] = np.nan
        
        pd_to_gt_scale = np.nanmedian(
            (gt_depths_clone / est_depths_clone).reshape(-1), axis=-1
        )

        if np.isnan(pd_to_gt_scale):
            pd_to_gt_scale = 1.0

        if self.subsample_query_trajs:
            assert self.traj_sampling_cfg is not None, (
                "traj_sampling config is required when subsample_query_trajs is True"
            )
            rng = np.random.default_rng(self.seed + sample_id)
            select_indices = rng.permutation(len(gt_query_point))[
                : int(self.traj_sampling_cfg.num_traj)
            ]
            gt_query_point = gt_query_point[select_indices]
            est_query_point = est_query_point[select_indices]
            valids = valids[:, select_indices]
            visibs = visibs[:, select_indices]
            gt_trajs_3d = gt_trajs_3d[:, select_indices]
            gt_traj_2d = gt_traj_2d[:, select_indices]
            est_trajs_3d = est_trajs_3d[:, select_indices]
            gt_trajs_depths = gt_trajs_depths[:, select_indices]
            is_dynamic = is_dynamic[select_indices]

        traj_norm_mask = None
        norm_mask = None
        if 'kubric' in self.data_provider.name and dense_valids is not None:
            traj_norm_mask = (gt_trajs_depths < 20) & (gt_trajs_depths > 0.1)
            norm_mask = F.interpolate(
                torch.from_numpy(norm_mask_gt_only).unsqueeze(0).float(),
                scale_factor=1.0 / self.scale_factor,
                mode="nearest-exact",
            ).squeeze(0).bool().numpy()
            # norm_mask = norm_mask_gt_only[
            #     :,
            #     self.scale_factor // 2 - 1 :: self.scale_factor,
            #     self.scale_factor // 2 - 1 :: self.scale_factor,
            # ].copy()
            dense_valids = dense_valids & norm_mask[..., None]
            dense_visibs = dense_visibs & norm_mask[..., None]
            dense_dynamic = dense_dynamic & (
                norm_mask[..., None] if dense_dynamic.ndim == 4 else norm_mask
            )
            valids = valids & traj_norm_mask
            visibs = visibs & traj_norm_mask
            
        # norm_mask = (est_depths_downsampled > 0.1) & (est_depths_downsampled < 5)

        return SliceData(
            rgbs=rearrange(torch.from_numpy(rgbs), "t h w c -> t c h w").to(
                torch.float32
            )
            / 255.0,
            gt_depths=torch.from_numpy(gt_depths),
            est_depths=torch.from_numpy(est_depths),
            gt_trajs_3d=torch.from_numpy(gt_trajs_3d),
            est_trajs_3d=torch.from_numpy(est_trajs_3d),
            visibs=torch.from_numpy(visibs),
            valids=torch.from_numpy(valids),
            seq_name=raw_slice_data.seq_name,
            seq_id=torch.tensor(raw_slice_data.seq_id, dtype=torch.int32),
            gt_intrinsics=torch.from_numpy(gt_intrinsics),
            gt_extrinsics=torch.from_numpy(gt_extrinsics),
            est_intrinsics=torch.from_numpy(est_intrinsics),
            est_extrinsics=torch.from_numpy(est_extrinsics),
            gt_query_point=torch.from_numpy(gt_query_point),
            est_query_point=torch.from_numpy(est_query_point),
            orig_resolution=torch.from_numpy(raw_slice_data.orig_resolution),
            same_scale=raw_slice_data.same_scale,
            num_views=num_views,
            flags=raw_slice_data.flags.copy(),
            segmentation=torch.from_numpy(segmentation.copy())
            if segmentation is not None
            else None,
            sample_id=torch.tensor(sample_id, dtype=torch.int32),
            est_depth_roi=torch.tensor(
                raw_slice_data.est_depth_roi, dtype=torch.float32
            )
            if raw_slice_data.est_depth_roi is not None
            else None,
            gt_depth_roi=torch.tensor(raw_slice_data.gt_depth_roi, dtype=torch.float32)
            if raw_slice_data.gt_depth_roi is not None
            else None,
            gt_intrinsics_downsampled=torch.tensor(gt_intrinsics_downsampled.copy()),
            est_intrinsics_downsampled=torch.tensor(est_intrinsics_downsampled.copy()),
            gt_depths_downsampled=torch.tensor(gt_depths_downsampled),
            est_depths_downsampled=torch.tensor(est_depths_downsampled),
            is_dynamic=torch.tensor(is_dynamic) if is_dynamic is not None else None,
            dense_valids=torch.from_numpy(dense_valids) if dense_valids is not None else None,
            dense_visibs=torch.from_numpy(dense_visibs) if dense_visibs is not None else None,
            dense_dynamic=torch.from_numpy(dense_dynamic) if dense_dynamic is not None else None,
            pd_to_gt_scale=torch.tensor(pd_to_gt_scale),
            gt_norm_factor=torch.tensor(gt_norm_factor, dtype=torch.float32),
            est_norm_factor=torch.tensor(est_norm_factor, dtype=torch.float32),
            dense_trajs_3d=torch.from_numpy(dense_trajs_3d)
            if dense_trajs_3d is not None
            else None,
            est_dense_trajs_3d=torch.from_numpy(est_dense_trajs_3d)
            if est_dense_trajs_3d is not None
            else None,
            dense_norm_mask=torch.from_numpy(norm_mask[..., None]) if norm_mask is not None else None,
            traj_norm_mask=torch.from_numpy(traj_norm_mask) if traj_norm_mask is not None else None,
            tgt_time_step_scenes=torch.from_numpy(tgt_time_step_scenes).long() if tgt_time_step_scenes is not None else None,
            dense_sample_points=torch.from_numpy(dense_sample_points) if dense_sample_points is not None else None,
            dense_sample_trajs_3d=torch.from_numpy(dense_sample_trajs_3d) if dense_sample_trajs_3d is not None else None,
            dense_sample_visibs=torch.from_numpy(dense_sample_visibs) if dense_sample_visibs is not None else None,
            dense_sample_valids=torch.from_numpy(dense_sample_valids) if dense_sample_valids is not None else None,
            dense_sample_dynamic=torch.from_numpy(dense_sample_dynamic) if dense_sample_dynamic is not None else None,
            dataset_name=self.dataset_name,
            dynamic_filter_threshold=torch.tensor(
                self.dynamic_filter_threshold, dtype=torch.float32
            ),
        )

    def __len__(self) -> int:
        return self.epoch_len

    @profile
    def __getitem__(self, index: int) -> SliceData:
        # if self.is_overfit and self.cache is not None:
        #     return self.cache
        # index = 32
        # print ("DEBUG!!!!!!!!!!!!!!!!!\n" * 10)
        n = len(self.seq_ids)
        if n == 0:
            raise RuntimeError(
                f"Eval dataset '{self.dataset_name}' has no sequences"
            )
        # Skip empty / filtered samples by walking forward with wrap-around.
        # Previously this recursed as index+1 with no bound and crashed with
        # IndexError once past the last seq (e.g. POD "no valid+visible tracks").
        start = index % n
        for attempt in range(n):
            idx = (start + attempt) % n
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                slice_data = self.data_provider.load_slice(
                    seq_id=int(self.seq_ids[idx]),
                    start=0,
                    length=self.seq_lens[self.seq_ids[idx]]
                    if self.frames_per_sample < 0
                    else self.frames_per_sample,
                    stride=self.stride,
                    rng=np.random.default_rng(self.seed + idx),
                    executor=executor,
                )
                # print(slice_data.seq_name)
                if slice_data is None:
                    continue

                if self.transform is not None:
                    slice_data = self.transform(
                        slice_data,
                        rng=np.random.default_rng(self.seed + idx + 1),
                        executor=executor,
                    )
                slice_data = self._postprocess_slice(idx, slice_data)
                if slice_data is not None and self.to_local_alltracker:
                    slice_data = data_ops.to_local_alltracker(
                        slice_data,
                        executor=executor,
                        rng=np.random.default_rng(self.seed + idx),
                    )
                if slice_data is None:
                    continue
            # if self.is_overfit:
            #     self.cache = slice_data
            return slice_data

        raise RuntimeError(
            f"All {n} eval samples skipped/unavailable for dataset "
            f"'{self.dataset_name}' (started at index {index})"
        )
