# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Tuple, TypedDict, Union

import cv2
import math
import ipdb
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from line_profiler import profile
from omegaconf import DictConfig, ListConfig


from . import data_ops
from .base_dataset import BaseDataset
from .datatypes import RawSliceData, SliceData
from .providers.base_provider import BaseDataProvider
from .utils.crop_utils import CropArgs, get_crop_args
from .utils.random_utils import RandomMapping, WeightedSampler
from .utils.geometry import world_to_local, local_to_world
from ..utils.common_utils import batch_unproject

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
st = ipdb.set_trace


class TrainDataset(BaseDataset):
    def __init__(
        self,
        frames_per_sample: int,
        seed: int,
        epoch_len: int,
        resolution: Tuple[int, int],
        different_seeds_per_epoch: bool,
        traj_sampling: DictConfig,
        resize: DictConfig,
        providers_config: List[_ProviderInfo],
        pad_trajs: bool = True,
        transform: Optional[ListConfig] = None,
        num_threads: int = 16,
        scale_factor: int = 4,
        use_gt_poses: bool = False,
        use_gt_depths: bool = False,
        subsample_trajs: int = -1,
        evaluate_with_query_points: bool = False,
        subsample_query_trajs: bool = False,
        scale_and_cam0_align: bool = False,
        compute_dense_trajs_3d: bool = False,
        compute_tmaps: bool = False,
        make_all_dense_static: bool = False,
        lift_full_geometry: bool = True,
        scene_transform_augs: Optional[DictConfig] = None,
        train_on_key_frames: bool = False,
        eval_on_key_frames: bool = False,
        model_seq_len: int = 16,
        half_window_overlap: bool = False,
        to_local_alltracker: bool = False,
        num_dense_sample_points: int = 0,
        strides: Optional[Union[int, List[int], ListConfig]] = None,
        valid_depth_range: Tuple[float, float] = (0.1, 20.0),
    ):
        self.resolution = resolution
        self.frames_per_sample = frames_per_sample
        self.traj_sampling_cfg = traj_sampling
        self.resize_cfg = resize
        self.seed = seed
        self.epoch_len = epoch_len
        self.scale_factor = scale_factor
        self.use_gt_poses = use_gt_poses
        self.use_gt_depths = use_gt_depths
        self.subsample_trajs = subsample_trajs
        self.evaluate_with_query_points = evaluate_with_query_points
        self.subsample_query_trajs = subsample_query_trajs
        self.scale_and_cam0_align = scale_and_cam0_align
        self.compute_dense_trajs_3d = compute_dense_trajs_3d
        self.compute_tmaps = compute_tmaps
        self.make_all_dense_static = make_all_dense_static
        self.lift_full_geometry = lift_full_geometry
        self.scene_transform_augs_cfg = dict(scene_transform_augs) if scene_transform_augs is not None else None
        self.train_on_key_frames = train_on_key_frames
        self.eval_on_key_frames = eval_on_key_frames
        self.model_seq_len = model_seq_len
        self.half_window_overlap = half_window_overlap
        self.to_local_alltracker = to_local_alltracker
        self.num_dense_sample_points = num_dense_sample_points
        self.valid_depth_range = tuple(map(float, valid_depth_range))
        if transform is not None:
            self.transform = data_ops.from_config(transform)
        else:
            self.transform = None
        self.pad_trajs = pad_trajs
        self.num_threads = num_threads
        self.providers_config = providers_config
        self.different_seeds_per_epoch = different_seeds_per_epoch
        self.data_providers: List[BaseDataProvider] = [
            BaseDataProvider.from_config(
                provider_info["config"],
                name=provider_info["name"],
                override_anno=provider_info.get("override_anno", None),
            )
            for provider_info in providers_config
        ]
        self.is_overfit = providers_config[0]["config"]["is_overfit"]
        self.is_train = providers_config[0]["config"]["is_train"]
        self.dataset_name = providers_config[0]["name"]
        self.dynamic_filter_threshold = providers_config[0]["config"].get(
            "dynamic_filter_threshold", 0.0
        )
        if self.is_overfit:
            self.transform = None
        self.cache = None

        self.random_mapping: Optional[RandomMapping] = None
        self.provider_sampler = WeightedSampler(
            [provider_info["weight"] for provider_info in providers_config]
        )
        self.sample_groups: List[list] = [[] for _ in range(len(providers_config))]
        self.group_samplers: List[Any] = [None for _ in range(len(providers_config))]
        self.seq_lens: List[List[int]] = []

        total_frames = 0
        total_seqs = 0

        for idx, provider_info in enumerate(providers_config):
            seq_lens = self.data_providers[idx].seq_lens
            if seq_lens is None:
                # this is slow, so we cached it in the data provider
                print("loading seq_lens, this may take a while")
                seq_lens = self.data_providers[idx].load_seq_lens()
            total_seqs += len(seq_lens)
            total_frames += sum(seq_lens)

            stride_values = provider_info["stride"] if strides is None else strides
            if isinstance(stride_values, ListConfig):
                strides_for_provider = list(stride_values)
            elif isinstance(stride_values, list):
                strides_for_provider = stride_values
            else:
                strides_for_provider = [stride_values]

            weights = []
            group_weight_mode = str(
                provider_info.get("group_weight_mode", "sliding_window")
            )
            for seq_id, seq_len in enumerate(seq_lens):
                for stride in strides_for_provider:
                    num_starts = seq_len - (frames_per_sample - 1) * stride
                    if num_starts <= 0:
                        logger.warning(
                            f"Sequence {seq_id} of provider {provider_info['name']} is too short to be used for training"
                        )
                        continue
                    if group_weight_mode == "uniform_sequence":
                        # Equal P(seq); start is still uniform over valid windows.
                        weights.append(1.0)
                    elif group_weight_mode == "sliding_window":
                        weights.append(num_starts)
                    else:
                        raise ValueError(
                            f"Unknown group_weight_mode={group_weight_mode!r} for "
                            f"provider {provider_info['name']} "
                            f"(expected 'sliding_window' or 'uniform_sequence')"
                        )
                    self.sample_groups[idx].append((seq_id, stride))

            if not weights:
                raise ValueError(
                    f"Provider {provider_info['name']} has no sequence long enough "
                    f"for frames_per_sample={frames_per_sample} and strides={strides_for_provider}"
                )
            self.group_samplers[idx] = WeightedSampler(weights=weights)
            self.seq_lens.append(seq_lens)

        if epoch_len is None or epoch_len == -1:
            self.epoch_len = total_seqs

        logger.info(
            f"Initialized dataset with {total_frames} frames and {total_seqs} sequences"
        )
        logger.info(f"Epoch length is set to {epoch_len}")

    def set_epoch(self, epoch: int):
        if self.different_seeds_per_epoch:
            self.random_mapping = RandomMapping(
                f"global_seed:{self.seed}, epoch:{epoch}"
            )
        else:
            self.random_mapping = RandomMapping(f"global_seed:{self.seed}, epoch:none")

    def __len__(self) -> int:
        return self.epoch_len

    @profile
    def __getitem__(self, index: int) -> SliceData:
        assert (
            self.random_mapping is not None
        ), "set_epoch must be called before getting items"

        trial = 0
        slice_data = None

        while slice_data is None:
            provider_idx = self.provider_sampler.sample(
                self.random_mapping.random(f"select_provider:{index}:{trial}")
            )
            provider = self.data_providers[provider_idx]

            group_index = self.group_samplers[provider_idx].sample(
                self.random_mapping.random(f"select_group:{index}:{trial}")
            )
            seq_id, stride = self.sample_groups[provider_idx][group_index]
            start = self.random_mapping.randint(
                f"start:{index}:{trial}",
                0,
                int(self.seq_lens[provider_idx][seq_id])
                - (self.frames_per_sample - 1) * stride,
            )
            
            if self.is_overfit:
                start = 0
                stride = 1
                rng = np.random.default_rng(self.seed + index)
            else:
                rng = np.random.default_rng(
                    self.random_mapping.randint(f"rng_seed:{index}:{trial}", 0, 2**32)
                )


            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                try:
                    slice_data = provider.load_slice(
                        seq_id=seq_id,
                        start=start,
                        length=self.frames_per_sample,
                        stride=stride,
                        rng=rng,
                        executor=executor,
                    )
                except Exception as e:
                    slice_data = None
                    logger.warning(
                        f"An error occurred in data provider: {self.providers_config[provider_idx]['name']}/{seq_id}: {e}"
                    )
                    import traceback

                    traceback.print_exc()

                if slice_data is not None:
                    if self.transform is not None:
                        slice_data = self.transform(
                            slice_data,
                            executor=executor,
                            target_resolution=self.resolution,
                            rng=rng,
                        )
                    slice_data = self._postprocess_slice(index, slice_data, rng=rng)
                    if slice_data is not None and self.to_local_alltracker:
                        slice_data = data_ops.to_local_alltracker(slice_data, executor=executor, rng=rng)
                    if slice_data is not None and self.scene_transform_augs_cfg is not None and self.is_train:
                        slice_data = data_ops.scene_transform_augs(
                            slice_data, rng=rng, **self.scene_transform_augs_cfg
                        )
                    if slice_data is None or slice_data.gt_trajs_3d.shape[1] == 0:
                        slice_data = None
                        logger.warning(
                            f"No valid trajectories found in {self.providers_config[provider_idx]['name']}/{seq_id}"
                        )

                if slice_data is None:
                    logger.warning(
                        f"An error occurred while loading a sample from {self.providers_config[provider_idx]['name']}/{seq_id}"
                    )
                trial += 1
        
        if self.is_overfit:
            self.cache = slice_data
            
        # print(slice_data.seq_name)
        return slice_data

    def _postprocess_slice(
        self, sample_id: int, raw_slice_data: RawSliceData, rng: np.random.Generator
    ) -> SliceData:   
        if self.resize_cfg.mode == "aug_crop":
            crop_args = get_crop_args(
                target_resolution=self.resolution,
                src_resolution=(raw_slice_data.rgbs[0].shape[:2]),
                aug_crop=self.resize_cfg.aug_crop,
                rng=rng,
            )
        elif self.resize_cfg.mode == "simple_resize":
            crop_args = CropArgs(
                crop_start=(0, 0),
                crop_end=(raw_slice_data.rgbs[0].shape[:2]),
                src_resolution=(raw_slice_data.rgbs[0].shape[:2]),
                target_resolution=self.resolution,
            )
        elif self.resize_cfg.mode == "none":
            crop_args = None
            assert (
                self.resolution == raw_slice_data.rgbs[0].shape[:2]
            ), "When resize mode is none, the image passed from data provider must be of the same resolution as the dataset resolution"
        else:
            raise ValueError(f"Unknown resize mode: {self.resize_cfg.mode}")

        if crop_args is not None:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                gt_intrinsics = crop_args.update_intrinsics(
                    raw_slice_data.gt_intrinsics
                )
                est_intrinsics = crop_args.update_intrinsics(
                    raw_slice_data.est_intrinsics
                )
                # raw_slice_data.est_intrinsics = est_intrinsics
                rgbs = crop_args.crop_img(
                    raw_slice_data.rgbs,
                    interp_mode=cv2.INTER_LANCZOS4,
                    executor=executor,
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

                if raw_slice_data.bg_mask is not None:
                    bg_mask = crop_args.crop_img(
                        raw_slice_data.bg_mask.astype(np.float32),
                        interp_mode=cv2.INTER_NEAREST_EXACT,
                        executor=executor,
                    ).astype(bool)
                else:
                    bg_mask = None

                assert rgbs[0].shape[:2] == self.resolution, "Internal Error"
        else:
            gt_intrinsics = raw_slice_data.gt_intrinsics
            est_intrinsics = raw_slice_data.est_intrinsics
            rgbs = raw_slice_data.rgbs
            gt_depths = raw_slice_data.gt_depths
            est_depths = raw_slice_data.est_depths
            segmentation = raw_slice_data.segmentation
            bg_mask = raw_slice_data.bg_mask

        visibs = raw_slice_data.visibs
        valids = raw_slice_data.valids
        is_dynamic = raw_slice_data.is_dynamic

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
        gt_traj_3d = raw_slice_data.gt_trajs_3d
        est_traj_3d = raw_slice_data.est_trajs_3d
        gt_trajs_depths = raw_slice_data.gt_trajs_depths
        gt_extrinsics = raw_slice_data.gt_extrinsics.copy()
        
        visibs[gt_traj_2d[:, :, 0] > self.resolution[1] - 1] = False
        visibs[gt_traj_2d[:, :, 0] < 0] = False
        visibs[gt_traj_2d[:, :, 1] > self.resolution[0] - 1] = False
        visibs[gt_traj_2d[:, :, 1] < 0] = False
        
        # gt_traj_3d[:, ~is_dynamic] = gt_traj_3d[0:1, ~is_dynamic]
        # _num_trajs = gt_traj_3d.shape[1]
        # gt_traj_local = self._apply_homo_transform(
        #     gt_traj_3d,
        #     repeat(gt_extrinsics, "t i j -> t n i j", n=_num_trajs),
        # )
        # gt_traj_2d = self._apply_homo_transform(
        #     gt_traj_local,
        #     repeat(gt_intrinsics[:, :3, :3], "t i j -> t n i j", n=_num_trajs),
        #     homo_input=True,
        # )
        # gt_trajs_depths[:, ~is_dynamic] = gt_traj_local[:, ~is_dynamic, 2]
        
        for i in range(len(gt_traj_2d)):
            xy0 = gt_traj_2d[i].astype(np.float32)  # N,2
            _, unique_inds = np.unique(
                np.floor(xy0[visibs[i]]), axis=0, return_index=True
            )

            unique_inds = np.arange(len(xy0))[visibs[i]][unique_inds]

            # add back all inds which were not visible
            invisible_inds = np.nonzero(~visibs[i])[0]
            unique_inds = np.concatenate([unique_inds, invisible_inds])
            unique_inds = np.unique(unique_inds)

            gt_traj_2d = gt_traj_2d[:, unique_inds]
            visibs = visibs[:, unique_inds]
            gt_trajs_depths = gt_trajs_depths[:, unique_inds]
            gt_traj_3d = gt_traj_3d[:, unique_inds]
            is_dynamic = is_dynamic[unique_inds]
            est_traj_3d = est_traj_3d[:, unique_inds]
            valids = valids[:, unique_inds]


        center_coords = np.array(
            [self.resolution[1] / 2, self.resolution[0] / 2], dtype=np.float32
        )
        dists = np.linalg.norm(gt_traj_2d - center_coords, axis=-1)

        valids[dists > self.traj_sampling_cfg.outlier_threshold] = False

        visible_ratio = (visibs & valids).sum(axis=0) / visibs.shape[0]

        select_indices = np.where(
            visible_ratio >= self.traj_sampling_cfg.min_visible_ratio
        )[0]

        if (self.is_train or self.is_overfit) and self.subsample_trajs > 0:
            dynamic_unique_traj_inds = select_indices[is_dynamic[select_indices]]
            static_unique_trajs = select_indices[~is_dynamic[select_indices]]

            print(f"dynamic trajs are: {len(dynamic_unique_traj_inds)} and static trajs are: {len(static_unique_trajs)}")

            # keep half of the dynamic trajs and rest static
            if len(dynamic_unique_traj_inds) > self.subsample_trajs // 2:
                dynamic_unique_traj_inds = np.random.choice(
                    dynamic_unique_traj_inds,
                    self.subsample_trajs // 2,
                    replace=False,
                )

            if len(static_unique_trajs) > self.subsample_trajs - len(
                dynamic_unique_traj_inds
            ):
                static_unique_trajs = np.random.choice(
                    static_unique_trajs,
                    self.subsample_trajs - len(dynamic_unique_traj_inds),
                    replace=False,
                )
            print(f"dynamic trajs are: {len(dynamic_unique_traj_inds)} and static trajs are: {len(static_unique_trajs)}")
            select_indices = np.concatenate(
                [dynamic_unique_traj_inds, static_unique_trajs]
            )

        visibs = visibs[:, select_indices]
        valids = valids[:, select_indices]
        gt_traj_3d = gt_traj_3d[:, select_indices]
        est_traj_3d = est_traj_3d[:, select_indices]
        gt_trajs_depths = gt_trajs_depths[:, select_indices]
        gt_traj_2d = gt_traj_2d[:, select_indices]
        is_dynamic = is_dynamic[select_indices] if is_dynamic is not None else None

        # shift to camera 0 coordinate system
        if self.scale_and_cam0_align:
            gt_traj_3d, gt_extrinsics, _ = self._align_to_cam0(
                gt_traj_3d, gt_extrinsics
            )
            est_traj_3d, raw_slice_data.est_extrinsics, _ = self._align_to_cam0(
                est_traj_3d, raw_slice_data.est_extrinsics
            )

        # gt_traj_local = gt_traj_local[:, select_indices]
        gt_queries = (
            raw_slice_data.gt_query_point[select_indices]
            if raw_slice_data.gt_query_point is not None
            else None
        )
        est_queries = (
            raw_slice_data.est_query_point[select_indices]
            if raw_slice_data.est_query_point is not None
            else None
        )
        
        assert gt_queries is None, "you need to fix their coordinate systems if they are not none"

        visible_first_indices = visibs[0, :].nonzero()[0]
        visible_last_indices = visibs[-1, :].nonzero()[0]
        visible_middle_indices = visibs[visibs.shape[0] // 2, :].nonzero()[0]
        
        if self.traj_sampling_cfg.traj_mode == "visible_first":
            select_indices = visible_first_indices
        elif self.traj_sampling_cfg.traj_mode == "visible_last":
            select_indices = visible_last_indices
        elif self.traj_sampling_cfg.traj_mode == "visible_first+visible_last":
            select_indices = np.concatenate(
                [visible_first_indices, visible_last_indices]
            )
        elif self.traj_sampling_cfg.traj_mode == "visible_first+visible_middle":
            select_indices = np.concatenate(
                [visible_first_indices, visible_middle_indices]
            )
        elif (
            self.traj_sampling_cfg.traj_mode
            == "visible_first+visible_middle+visible_last"
        ):
            select_indices = np.concatenate(
                [visible_first_indices, visible_middle_indices, visible_last_indices]
            )
        elif self.traj_sampling_cfg.traj_mode == "random":
            select_indices = np.arange(visibs.shape[1])
        else:
            raise ValueError(
                f"Unknown trajectory selection mode: {self.traj_sampling_cfg.traj_mode}"
            )

        if not self.subsample_query_trajs:
            select_indices = select_indices[
                rng.permutation(len(select_indices))[: self.traj_sampling_cfg.num_traj]
            ]

        gt_traj_3d = gt_traj_3d[:, select_indices]
        gt_traj_2d = gt_traj_2d[:, select_indices]
        gt_traj_local = gt_traj_local[:, select_indices]
        est_traj_3d = est_traj_3d[:, select_indices]
        gt_trajs_depths = gt_trajs_depths[:, select_indices]
        visibs = visibs[:, select_indices]
        valids = valids[:, select_indices]
        gt_queries = gt_queries[select_indices] if gt_queries is not None else None
        est_queries = est_queries[select_indices] if est_queries is not None else None
        is_dynamic = is_dynamic[select_indices] if is_dynamic is not None else None
        

        if self.traj_sampling_cfg.query_mode is not None:
            assert (
                raw_slice_data.gt_query_point is None
                and raw_slice_data.est_query_point is None
            ), "Query point must not be provided if query_mode is not None"
            # if self.traj_sampling_cfg.query_mode == "all":
            #     query_times, gt_query_points, est_query_points = [], [], []
            #     valid_indices = np.where((visibs & valids))
            #     st()

            if (
                self.traj_sampling_cfg.query_mode == "random"
                # or self.traj_sampling_cfg.query_mode == "all"
            ):
                query_times, gt_query_points, est_query_points, query_point_2d = (
                    [],
                    [],
                    [],
                    [],
                )
                _vv = visibs & valids
                for i in range(gt_traj_3d.shape[1]):
                    indices = np.where(_vv[:, i])[0]
                    assert (
                        len(indices) > 0
                    ), f"No visible points found for trajectory {i}. There must be a bug in the trajectory sampler"

                    # if self.traj_sampling_cfg.query_mode == "random":
                    query_time = rng.choice(indices)
                    query_times.append(query_time)
                    gt_query_points.append(gt_traj_3d[query_time, i])
                    est_query_points.append(est_traj_3d[query_time, i])
                    query_point_2d.append(gt_traj_2d[query_time, i])
                    # else:
                    #     assert self.traj_sampling_cfg.query_mode == "all"
                    #     for query_time in indices:
                    #         query_times.append(query_time)
                    #         gt_query_points.append(gt_traj_3d[query_time, i])
                    #         est_query_points.append(est_traj_3d[query_time, i])

                if len(gt_query_points) > 0:
                    gt_query_point = np.stack(gt_query_points, axis=0)
                    est_query_point = np.stack(est_query_points, axis=0)
                    query_time = np.array(query_times, dtype=gt_query_point.dtype)[
                        :, None
                    ]
                    gt_query_point = np.concatenate(
                        [query_time, gt_query_point], axis=-1
                    )
                    est_query_point = np.concatenate(
                        [query_time, est_query_point], axis=-1
                    )
                    query_point_2d = np.stack(query_point_2d, axis=0)
                else:
                    gt_query_point = np.zeros((gt_traj_3d.shape[0], 0, 4)).astype(
                        np.float32
                    )
                    est_query_point = np.zeros((est_traj_3d.shape[0], 0, 4)).astype(
                        np.float32
                    )
                    query_point_2d = np.zeros((gt_traj_2d.shape[0], 0, 2)).astype(
                        np.float32
                    )
            elif self.traj_sampling_cfg.query_mode == "cotracker":
                N_rand = gt_traj_3d.shape[1] // 4
                query_times, gt_query_points, est_query_points, query_point_2d = (
                    [],
                    [],
                    [],
                    [],
                )
                _vv = visibs & valids
                for i in range(gt_traj_3d.shape[1]):
                    _col = _vv[:, i]
                    indices = np.where(_col)[0]
                    assert (
                        len(indices) > 0
                    ), f"No visible points found for trajectory {i}. There must be a bug in the trajectory sampler"
                    if i < N_rand:
                        query_time = rng.choice(indices)
                    else:
                        query_time = np.argmax(_col)
                    query_times.append(query_time)
                    gt_query_points.append(gt_traj_3d[query_time, i])
                    est_query_points.append(est_traj_3d[query_time, i])
                    query_point_2d.append(gt_traj_2d[query_time, i])

                if len(gt_query_points) > 0:
                    gt_query_point = np.stack(gt_query_points, axis=0)
                    est_query_point = np.stack(est_query_points, axis=0)
                    query_time = np.array(query_times, dtype=gt_query_point.dtype)[
                        :, None
                    ]
                    gt_query_point = np.concatenate(
                        [query_time, gt_query_point], axis=-1
                    )
                    est_query_point = np.concatenate(
                        [query_time, est_query_point], axis=-1
                    )
                    query_point_2d = np.stack(query_point_2d, axis=0)
                else:
                    return None
                    gt_query_point = np.zeros(
                        (gt_traj_3d.shape[0], 0, 4), dtype=gt_traj_3d.dtype
                    )
                    est_query_point = np.zeros(
                        (est_traj_3d.shape[0], 0, 4), dtype=est_traj_3d.dtype
                    )
                    query_point_2d = np.zeros(
                        (gt_traj_2d.shape[0], 0, 2), dtype=gt_traj_2d.dtype
                    )

            elif self.traj_sampling_cfg.query_mode == "first":
                assert (visibs & valids)[
                    0, :
                ].all(), "First frame must be visible since we use it as query point"
                gt_query_point = np.concatenate(
                    [np.zeros_like(gt_traj_3d[0, :, :1]), gt_traj_3d[0, :]], axis=-1
                )
                est_query_point = np.concatenate(
                    [np.zeros_like(est_traj_3d[0, :, :1]), est_traj_3d[0, :]], axis=-1
                )
                query_point_2d = gt_traj_2d[0, :]
            else:
                raise ValueError(
                    f"Unknown query mode: {self.traj_sampling_cfg.query_mode}"
                )
        else:
            query_point_2d = None
            gt_query_point = gt_queries
            est_query_point = est_queries

        if self.traj_sampling_cfg.get("only_dynamic", False):
            valids[:, is_dynamic == 0] = False

        # Make all dynamic points invalid
        # valids[:, is_dynamic == 1] = False
        # visibs[:, is_dynamic == 1] = False

        # make all static points visible
        # visibs[:, is_dynamic == 0] = True

        # compute dense gt depth
        H, W = rgbs[0].shape[:2]

        gt_depths_downsampled = (
            F.interpolate(
                torch.tensor(gt_depths.copy()).unsqueeze(0),
                scale_factor=1.0 / self.scale_factor,
                mode="nearest-exact"
            )
            .squeeze(0)
            .numpy()
        )
        # gt_depths_downsampled = gt_depths[:, self.scale_factor//2 - 1::self.scale_factor, self.scale_factor//2 - 1::self.scale_factor].copy()
        bg_mask_downsampled = None
        if bg_mask is not None:
            bg_mask_downsampled = (
                F.interpolate(
                    torch.tensor(bg_mask.copy()).float().unsqueeze(0),
                    scale_factor=1.0 / self.scale_factor,
                    mode="nearest-exact"
                )
                .bool()
                .squeeze(0)
                .numpy()
            )
            # bg_mask_downsampled = bg_mask[:, self.scale_factor//2 - 1::self.scale_factor, self.scale_factor//2 - 1::self.scale_factor].copy()

        gt_intrinsics_downsampled = gt_intrinsics.copy()
        gt_intrinsics_downsampled[..., :2, :] /= self.scale_factor

        est_intrinsics_downsampled = est_intrinsics.copy()
        est_intrinsics_downsampled[..., :2, :] /= self.scale_factor

        dense_trajs_3d, dense_valids, dense_visibs, dense_dynamic = None, None, None, None
        dense_geom_raw, dense_src_idx_map = None, None
        tgt_time_step_scenes = None
        if self.compute_dense_trajs_3d:
            if self.train_on_key_frames or self.eval_on_key_frames:
                model_seq_len = self.model_seq_len
                num_frames = gt_traj_3d.shape[0]
                
                # Match model window scheduling for dense target time selection.
                # - non-overlap: windows of length model_seq_len
                # - overlap: 50% overlap using step model_seq_len // 2
                if self.half_window_overlap:
                    half_step = max(1, model_seq_len // 2)
                    window_ends = list(range(model_seq_len, num_frames + 1, half_step))
                    if not window_ends or window_ends[-1] < num_frames:
                        window_ends.append(num_frames)
                    last_window_start = max(0, window_ends[-1] - model_seq_len)
                else:
                    num_windows = math.ceil(num_frames / model_seq_len)
                    window_ends = [min(model_seq_len * (i + 1), num_frames) for i in range(num_windows)]
                    if num_windows > 1:
                        last_window_start = window_ends[-2]
                    else:
                        last_window_start = window_ends[0]

                # For each window except the last, pick the last frame in that window.
                # For the last window, sample a random frame from that window.
                tgt_time_step_scenes = []
                for window_end in window_ends[:-1]:
                    tgt_time_step_scenes.append(window_end - 1)
                # last_window_start = max(0, window_ends[-1] - model_seq_len)
                if self.eval_on_key_frames and not self.is_train:
                    tgt_time_step_scenes.append(window_ends[-1] - 1)
                else:
                    tgt_time_step_scenes.append(
                        rng.choice(range(last_window_start, window_ends[-1]))
                    )
                tgt_time_step_scenes = np.array(tgt_time_step_scenes)
                
            else:
                tgt_time_step_scenes = None
            dense_trajs_3d, dense_valids, dense_visibs, dense_dynamic, dense_geom_raw, dense_src_idx_map = (
                self._compute_dense_trajs_and_unproject(
                    gt_traj_2d, gt_traj_3d, visibs, is_dynamic,
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
            # print(f"dynamic filter threshold is {self.dynamic_filter_threshold} for dataset {self.dataset_name}")
                            
        # Dense GT full vs time_steps: datasets/test_compute_dense_trajs_time_steps.py

        est_depths_downsampled = (
            F.interpolate(
                torch.tensor(est_depths.copy()).unsqueeze(0),
                scale_factor=1.0 / self.scale_factor,
                mode="nearest-exact"
            )
            .squeeze(0)
            .numpy()
        )
        est_dense_trajs_3d = None

        if self.traj_sampling_cfg.get("only_first_frame", False):
            dense_valids[1:, :] = False

        # we want a conservative norm mask to avoid outliers
        min_valid_depth, max_valid_depth = self.valid_depth_range
        if raw_slice_data.annotated:
            norm_mask = (
                (gt_depths < max_valid_depth)
                & (gt_depths > min_valid_depth)
                & (est_depths > 0.1)
                & (est_depths < 5)
            )
        else:
            norm_mask = (gt_depths < max_valid_depth) & (
                gt_depths > min_valid_depth
            )
            
            # synthverse has large valid depths
            if 'synthverse' in self.dataset_name:
                norm_mask = gt_depths > 0.05
    
        norm_mask_gt_only = (gt_depths < max_valid_depth) & (
            gt_depths > min_valid_depth
        )
        if 'synthverse' in self.dataset_name:
            norm_mask_gt_only = gt_depths > 0.05

        # import wandb
        # _depth_t = torch.from_numpy(np.ascontiguousarray(gt_depths, dtype=np.float32))
        # _depth_edge_mask = depth_edge(_depth_t, rtol=0.03).numpy()
        # _depth_edge_mask_est = depth_edge(torch.from_numpy(np.ascontiguousarray(est_depths, dtype=np.float32)), rtol=0.03).numpy()
        # norm_mask = norm_mask & ~_depth_edge_mask & ~_depth_edge_mask_est
        
        # remove depth edges based on gt depth from norm mask
        # norm_mask = norm_mask & ~_depth_edge_mask
        # norm_mask_gt_only = norm_mask_gt_only & ~_depth_edge_mask

        norm_mask_downsampled = (
            F.interpolate(
                torch.from_numpy(norm_mask.astype(np.float32)).unsqueeze(0),
                scale_factor=1.0 / self.scale_factor,
                mode="nearest-exact",
            )
            .squeeze(0)
            .bool()
            .numpy()
        )

        # from ..utils.vis_utils import log_norm_mask_overlay_wandb
        # log_norm_mask_overlay_wandb(rgbs, norm_mask)

        if norm_mask.mean() < 0.05 and raw_slice_data.annotated and 'kubric' in self.dataset_name:
            print("looks like a bad sample")
            return None
        # st()
        
        # visualize rgbd maps
        # from ..utils.vis_utils import visualize_rgbd
        # visualize_rgbd(
        #     rgbs, gt_depths, legend="gt_depths"
        # )
        # visualize_rgbd(
        #     rgbs, est_depths, legend="est_depths"
        # )
        # est_depths = gt_depths.copy()
        # est_depths_downsampled = gt_depths_downsampled.copy()
        gt_norm_factor = 1.0
        est_norm_factor = 1.0
        if self.scale_and_cam0_align:
            gt_norm_factor, gt_depths, gt_depths_downsampled, gt_traj_3d, gt_extrinsics, dense_trajs_3d = (
                self._normalize_scene(
                    gt_extrinsics,
                    gt_depths,
                    gt_traj_3d,
                    gt_depths_downsampled,
                    gt_intrinsics_downsampled,
                    norm_mask_downsampled,
                    norm_mode="median_dis",
                    query_point=gt_query_point,
                    dense_trajs_3d=dense_trajs_3d,
                )
            )
            if gt_queries is not None:
                gt_queries[:, 1:] /= gt_norm_factor

            est_norm_factor, est_depths, est_depths_downsampled, est_traj_3d, raw_slice_data.est_extrinsics, est_dense_trajs_3d = (
                self._normalize_scene(
                    raw_slice_data.est_extrinsics,
                    est_depths,
                    est_traj_3d,
                    est_depths_downsampled,
                    est_intrinsics_downsampled,
                    norm_mask_downsampled,
                    norm_mode="median_dis",
                    query_point=est_query_point,
                    dense_trajs_3d=est_dense_trajs_3d,
                )
            )
            if est_queries is not None:
                est_queries[:, 1:] /= est_norm_factor

        # Build grid-anchored full GT tracks for a subsample of dense cells.
        # Done here (post-normalization, pre query-subsample) so the winning track
        # index space still matches the full sparse arrays.
        dense_sample_points = None
        dense_sample_trajs_3d = None
        dense_sample_visibs = None
        dense_sample_valids = None
        dense_sample_dynamic = None
        if (
            self.compute_dense_trajs_3d
            and self.num_dense_sample_points > 0
            and dense_src_idx_map is not None
        ):
            _dense_sample = self._build_dense_sample_gt(
                dense_src_idx_map=dense_src_idx_map,
                gt_traj_3d=gt_traj_3d,
                visibs=visibs,
                valids=valids,
                gt_trajs_depths=gt_trajs_depths,
                is_dynamic=is_dynamic,
                gt_depths_downsampled=gt_depths_downsampled,
                gt_intrinsics_downsampled=gt_intrinsics_downsampled,
                gt_extrinsics=gt_extrinsics,
                norm_mask_downsampled=norm_mask_downsampled,
                num_points=self.num_dense_sample_points,
                rng=rng,
                valid_depth_range=self.valid_depth_range,
            )
            if _dense_sample is not None:
                (
                    dense_sample_points,
                    dense_sample_trajs_3d,
                    dense_sample_visibs,
                    dense_sample_valids,
                    dense_sample_dynamic,
                ) = _dense_sample

        # everything seems ok uptil here
        if self.use_gt_poses or self.use_gt_depths:
            # st()
            # we need to recompute the est_traj_3d, est_dense_traj_3d, est_query_point and est_queries
            query_frames = est_query_point[..., 0].astype(np.int32)
            est_query_point_2d = world_to_local(
                est_query_point[:, 1:],
                est_intrinsics[query_frames],
                raw_slice_data.est_extrinsics[query_frames],
            )[..., :2]
            
            gt_query_point_2d = world_to_local(
                gt_query_point[:, 1:],
                gt_intrinsics[query_frames],
                gt_extrinsics[query_frames],
            )[..., :2]
            
            if self.use_gt_poses:
                raw_slice_data.est_extrinsics = gt_extrinsics

            if self.use_gt_depths:
                est_depths = gt_depths
                est_depths_downsampled = gt_depths_downsampled
            
            if est_query_point_2d[:, 1].astype(np.int32).max() >= est_depths.shape[1] or est_query_point_2d[:, 0].astype(np.int32).max() >= est_depths.shape[2] or est_query_point_2d[:, 1].astype(np.int32).min() < 0 or est_query_point_2d[:, 0].astype(np.int32).min() < 0:
                print(f"Query point 2d is out of bounds of the depth map. This might be caused by incorrect camera extrinsics or intrinsics. Please check the data sample carefully. sample_id: {sample_id}, seq_name: {raw_slice_data.seq_name}, seq_id: {raw_slice_data.seq_id}")
                return None
            
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
                raw_slice_data.est_extrinsics[query_frames],
            )
            
            # est_query_point_2d = world_to_local(
            #     est_query_point[:, 1:],
            #     est_intrinsics[query_frames],
            #     raw_slice_data.est_extrinsics[query_frames],
            # )[..., :2]
            
            # st()
            # np.set_printoptions(precision=4, suppress=True)
            # zero_est_traj = (est_query_point[..., 1:].sum(-1) < 0.01)
            # if (np.abs(est_query_point_2d - gt_query_point_2d).sum(-1).max()) > 5:
            #     print("THIS MIGHT BE A BAD DATA SAMPLE")
            #     st()
        # pd_to_gt_scale = 1.0
        # pd_to_gt_scale_bg_masked = 1.0
        gt_depths_clone = np.copy(gt_depths)
        est_depths_clone = np.copy(est_depths)
        gt_depths_clone[~norm_mask] = np.nan
        est_depths_clone[~norm_mask] = np.nan

        pd_to_gt_scale = np.nanmedian(
            (gt_depths_clone / est_depths_clone).reshape(-1), axis=-1
        )
        if np.isnan(pd_to_gt_scale):
            pd_to_gt_scale = 1.0
            
        # print(f"pd to gt scale is {pd_to_gt_scale}")
            
        pd_to_gt_scale_bg_masked = pd_to_gt_scale


        # subsample the query points
        if self.subsample_query_trajs:
            select_indices = rng.permutation(len(gt_query_point))[: self.traj_sampling_cfg.num_traj]
            gt_query_point = gt_query_point[select_indices]
            est_query_point = est_query_point[select_indices]
            valids = valids[:, select_indices]
            visibs = visibs[:, select_indices]
            gt_traj_3d = gt_traj_3d[:, select_indices]
            gt_traj_2d = gt_traj_2d[:, select_indices]
            # gt_traj_local = gt_traj_local[:, select_indices]
            est_traj_3d = est_traj_3d[:, select_indices]
            gt_trajs_depths = gt_trajs_depths[:, select_indices]
            gt_queries = gt_queries[select_indices] if gt_queries is not None else None
            est_queries = est_queries[select_indices] if est_queries is not None else None
            is_dynamic = is_dynamic[select_indices] if is_dynamic is not None else None
        
        traj_norm_mask = (gt_trajs_depths < max_valid_depth) & (
            gt_trajs_depths > min_valid_depth
        )
        norm_mask = F.interpolate(
            torch.from_numpy(norm_mask_gt_only).unsqueeze(0).float(),
            scale_factor=1.0 / self.scale_factor,
            mode="nearest-exact"
        ).squeeze(0).bool().numpy()
        # norm_mask = norm_mask_gt_only[:, self.scale_factor//2 - 1::self.scale_factor, self.scale_factor//2 - 1::self.scale_factor].copy()
        
        if dense_valids is not None:
            dense_valids = dense_valids & norm_mask[..., None]
            dense_visibs = dense_visibs & norm_mask[..., None]
            dense_dynamic = dense_dynamic & (
                norm_mask[..., None] if dense_dynamic.ndim == 4 else norm_mask
            )
        
        valids = valids & traj_norm_mask
        visibs = visibs & traj_norm_mask
        
        return SliceData(
            rgbs=rearrange(torch.tensor(rgbs.copy()), "t h w c -> t c h w").to(
                torch.float32
            )
            / 255.0,
            gt_depths=torch.tensor(gt_depths.copy()),
            est_depths=torch.tensor(est_depths.copy()),
            gt_depths_downsampled=torch.tensor(gt_depths_downsampled),
            est_depths_downsampled=torch.tensor(est_depths_downsampled),
            gt_trajs_3d=torch.tensor(gt_traj_3d.copy()),
            gt_trajs_2d=torch.tensor(gt_traj_2d.copy()),
            est_trajs_3d=torch.tensor(est_traj_3d.copy()),
            visibs=torch.tensor(visibs.copy()),
            valids=torch.tensor(valids.copy()),
            seq_name=raw_slice_data.seq_name,
            seq_id=torch.tensor(raw_slice_data.seq_id, dtype=torch.int32),
            segmentation=torch.tensor(segmentation.copy())
            if segmentation is not None
            else None,
            gt_intrinsics=torch.tensor(gt_intrinsics.copy()),
            gt_intrinsics_downsampled=torch.tensor(gt_intrinsics_downsampled.copy()),
            est_intrinsics=torch.tensor(est_intrinsics.copy()),
            est_intrinsics_downsampled=torch.tensor(est_intrinsics_downsampled.copy()),
            gt_query_point=torch.tensor(gt_query_point.copy()).to(torch.float32),
            est_query_point=torch.tensor(est_query_point.copy()).to(torch.float32),
            orig_resolution=torch.tensor(raw_slice_data.orig_resolution.copy()),
            gt_extrinsics=torch.tensor(gt_extrinsics),
            est_extrinsics=torch.tensor(raw_slice_data.est_extrinsics.copy()),
            same_scale=raw_slice_data.same_scale,
            flags=raw_slice_data.flags.copy(),
            sample_id=torch.tensor(sample_id, dtype=torch.int32),
            est_depth_roi=torch.tensor(
                raw_slice_data.est_depth_roi, dtype=torch.float32
            )
            if raw_slice_data.est_depth_roi is not None
            else None,
            gt_depth_roi=torch.tensor(raw_slice_data.gt_depth_roi, dtype=torch.float32)
            if raw_slice_data.gt_depth_roi is not None
            else None,
            is_dynamic=torch.tensor(is_dynamic) if is_dynamic is not None else None,
            dense_trajs_3d=torch.from_numpy(dense_trajs_3d)
            if dense_trajs_3d is not None
            else None,
            est_dense_trajs_3d=torch.from_numpy(est_dense_trajs_3d)
            if est_dense_trajs_3d is not None
            else None,
            dense_valids=torch.from_numpy(dense_valids) if dense_valids is not None else None,
            dense_visibs=torch.from_numpy(dense_visibs) if dense_visibs is not None else None,
            dense_dynamic=torch.from_numpy(dense_dynamic) if dense_dynamic is not None else None,
            pd_to_gt_scale=torch.tensor(pd_to_gt_scale),
            pd_to_gt_scale_bg_masked=torch.tensor(pd_to_gt_scale_bg_masked),
            bg_mask=torch.from_numpy(bg_mask) if bg_mask is not None else None,
            bg_mask_downsampled=torch.from_numpy(bg_mask_downsampled)
            if bg_mask_downsampled is not None
            else None,
            query_indices=None,
            dense_norm_mask=torch.from_numpy(norm_mask[..., None]) if norm_mask is not None else None,
            traj_norm_mask=torch.from_numpy(traj_norm_mask) if traj_norm_mask is not None else None,
            tgt_time_step_scenes=torch.from_numpy(tgt_time_step_scenes) if tgt_time_step_scenes is not None else None,
            dense_sample_points=torch.from_numpy(dense_sample_points) if dense_sample_points is not None else None,
            dense_sample_trajs_3d=torch.from_numpy(dense_sample_trajs_3d) if dense_sample_trajs_3d is not None else None,
            dense_sample_visibs=torch.from_numpy(dense_sample_visibs) if dense_sample_visibs is not None else None,
            dense_sample_valids=torch.from_numpy(dense_sample_valids) if dense_sample_valids is not None else None,
            dense_sample_dynamic=torch.from_numpy(dense_sample_dynamic) if dense_sample_dynamic is not None else None,
            dataset_name=self.dataset_name,
            gt_norm_factor=torch.tensor(gt_norm_factor, dtype=torch.float32),
            est_norm_factor=torch.tensor(est_norm_factor, dtype=torch.float32),
            dynamic_filter_threshold=torch.tensor(
                self.dynamic_filter_threshold, dtype=torch.float32
            ),
        )
