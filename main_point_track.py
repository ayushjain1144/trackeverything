# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Main script for trajectory optimization."""

import gc
import logging
import os
import pickle
import re
import signal
import time
from datetime import timedelta
import ipdb

st = ipdb.set_trace
import fsspec

import hydra
import numpy as np
import torch
import copy

import torch.multiprocessing
import torch.nn.functional as F

from torch.amp import autocast
from torch.profiler import record_function


import wandb
from einops import rearrange
from iopath.common.file_io import PathManager
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

path_manager = PathManager()

from .training.datatypes import Prediction, TrainData

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["HYDRA_FULL_ERROR"] = "1"


import torch.profiler as profiler

from line_profiler import profile

from . import models
from .datasets.base_dataset import BaseDataset
from .datasets.datatypes import SliceData
from .evaluation import metrics_alltrack3r
from .evaluation.metric_display import CORE_DISPLAY_METRICS
from .training.criterion import AllTrajectoryCriterion
from .training.engine import BaseTrainTester, get_rank, log_multi_dataset_eval_table
from .utils import dataset_paths as _dataset_paths  # noqa: F401  (registers ${data_path} resolver)
from .utils.alltrack3r_utils import (
    prepare_dense_tracks_for_eval_with_v6,
    prepare_dense_sample_tracks_for_eval,
    prepare_query_through_dense_for_eval,
)
from .utils.common_utils import count_parameters, setup_logger
from .utils.vis_utils import (
    generate_videos,
    generate_dense_track_visualizations_with_rerun,
    generate_dense_track_visualizations_with_wandb,
    generate_query_dynamic_videos,
    generate_dense_dynamic_pred_videos,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.handlers.clear()
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _eval_console_log(msg: str) -> None:
    """Match eval_spatrackerv2 / main_point_track: visible in sweep logs (tee)."""
    logger.info("%s", msg)
    print(msg, flush=True)


class TrainTester(BaseTrainTester):
    """Train/test a trajectory optimization algorithm."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_criterion(self):
        return AllTrajectoryCriterion(
            train_iters=self.cfg.train.train_iters,
            eval_iters=self.cfg.train.eval_iters,
            evaluate_query_at_tgt=True,
            **self.cfg.criterion,
        )

    def get_datasets(self, start_iter=None):
        """Initialize datasets."""
        # Initialize datasets with arguments  
        train_dataset = None
        if not self.cfg.eval_only:
            train_dataset = {}
            # Unlock the parent so we can write merged cfgs back per-key below.
            if OmegaConf.is_config(self.cfg.train_dataset):
                OmegaConf.set_struct(self.cfg.train_dataset, False)
            for key, cfg in self.cfg.train_dataset.items():
                if self.cfg.get('train_datasets', None):
                    if key not in self.cfg.train_datasets:
                        continue
                cfg = cfg.copy()
                # Apply train-level overrides so flags can be set once for all datasets
                if OmegaConf.is_config(cfg) and self.cfg.train.get("dataset_overrides"):
                    OmegaConf.set_struct(cfg, False)
                    cfg = OmegaConf.merge(cfg, self.cfg.train.dataset_overrides)
                if OmegaConf.is_config(cfg) and "providers_config" in cfg and self.cfg.train.get("provider_overrides"):
                    for prov in cfg.providers_config:
                        if "config" in prov:
                            OmegaConf.set_struct(prov.config, False)
                            prov.config = OmegaConf.merge(prov.config, self.cfg.train.provider_overrides)
                if self.cfg.train.get("additional_transform", []):
                    if "transform" in cfg:
                        cfg.transform = cfg.transform + self.cfg.train.get(
                            "additional_transform", []
                        )
                        train_dataset[key] = BaseDataset.from_config(cfg)
                    else:
                        train_dataset[key] = BaseDataset.from_config(
                            cfg,
                            transform=self.cfg.train.get("additional_transform", []),
                        )
                else:
                    train_dataset[key] = BaseDataset.from_config(cfg)

                # Write the merged cfg back so downstream read-sites
                # (e.g. self.cfg.train_dataset[dataset_name]) see the effective config.
                if OmegaConf.is_config(cfg):
                    OmegaConf.set_struct(cfg, False)
                self.cfg.train_dataset[key] = cfg

                if start_iter is not None:
                    trained_epoch = start_iter * self.gradient_accumulation_steps // len(train_dataset[key])
                    train_dataset[key].set_epoch(trained_epoch)
            
            # merge all train datasets into one
            # total_length = sum([len(ds) for ds in train_dataset.values()])
            train_dataset = torch.utils.data.ConcatDataset(
                [ds for ds in train_dataset.values()]
            )
            
        test_datasets = {}
        eval_cfg = self.cfg.get("eval", {})
        # Unlock the parent so we can write merged cfgs back per-key below.
        if OmegaConf.is_config(self.cfg.test_datasets):
            OmegaConf.set_struct(self.cfg.test_datasets, False)
        for key, cfg in self.cfg.test_datasets.items():
            # Use `is not None` so eval_datasets=[] skips all eval sets (empty
            # list is falsy and previously loaded every test_datasets entry).
            eval_datasets = self.cfg.get("eval_datasets", None)
            if eval_datasets is not None:
                if key not in eval_datasets:
                    continue
            cfg = cfg.copy()
            # Apply eval-level overrides so flags can be set once for all eval datasets
            if OmegaConf.is_config(cfg) and eval_cfg.get("dataset_overrides"):
                OmegaConf.set_struct(cfg, False)
                cfg = OmegaConf.merge(cfg, eval_cfg.dataset_overrides)
            if OmegaConf.is_config(cfg) and eval_cfg.get("provider_overrides"):
                if "provider_config" in cfg and "config" in cfg.provider_config:
                    OmegaConf.set_struct(cfg.provider_config.config, False)
                    cfg.provider_config.config = OmegaConf.merge(
                        cfg.provider_config.config, eval_cfg.provider_overrides
                    )
                if "providers_config" in cfg:
                    for prov in cfg.providers_config:
                        if "config" in prov:
                            OmegaConf.set_struct(prov.config, False)
                            prov.config = OmegaConf.merge(prov.config, eval_cfg.provider_overrides)
            if self.cfg.get("eval", {}).get("additional_transforms", []):
                if "transform" in cfg:
                    cfg.transform = cfg.transform + self.cfg.eval.get(
                        "additional_transforms", []
                    )
                    test_datasets[key] = BaseDataset.from_config(cfg)
                else:
                    test_datasets[key] = BaseDataset.from_config(
                        cfg, transform=self.cfg.eval.get("additional_transforms", [])
                    )
            else:
                test_datasets[key] = BaseDataset.from_config(cfg)

            # Write the merged cfg back so downstream read-sites
            # (e.g. self.cfg.test_datasets[dataset_name]) see the effective config.
            if OmegaConf.is_config(cfg):
                OmegaConf.set_struct(cfg, False)
            self.cfg.test_datasets[key] = cfg
        return train_dataset, test_datasets

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        _model = models.from_config(
            self.cfg.model,
            image_size=tuple(self.cfg.resolution),
        )
        logger.info("Trainable parameters: %d", count_parameters(_model))
        return _model

    def _length_schedule_is_long_step(self, step_id: int, schedule_cfg) -> bool:
        pattern = schedule_cfg.get("pattern", "alternate")
        if pattern == "alternate":
            start_with_long = schedule_cfg.get("start_with_long", True)
            return (step_id % 2 == 0) == start_with_long
        if pattern == "random":
            long_prob = float(schedule_cfg.get("long_prob", 0.5))
            seed = int(self.cfg.get("seed", 0))
            rng = np.random.default_rng(seed + step_id)
            return bool(rng.random() < long_prob)
        raise ValueError(f"Unknown train.length_schedule.pattern: {pattern}")

    def _filter_query_trajectory_dim(
        self, sample: SliceData, start: int, end: int
    ) -> SliceData:
        query_times = sample.gt_query_point[..., 0]
        query_mask_by_batch = (query_times >= start) & (query_times < end)

        if query_mask_by_batch.ndim != 2:
            raise ValueError(
                f"Expected batched query points of shape (B, N, 4), got {sample.gt_query_point.shape}"
            )

        if query_mask_by_batch.shape[0] != 1 and not torch.all(
            query_mask_by_batch == query_mask_by_batch[:1]
        ):
            raise ValueError(
                "Length-schedule cropping with batch_size > 1 requires all batch items "
                "to keep the same query indices. Use batch_size=1 or query_mode=first."
            )

        query_mask = query_mask_by_batch[0]
        # query_mask[:] = False
        # if not bool(query_mask.any()):
        #     raise ValueError(
        #         f"Length-schedule crop [{start}, {end}) removed all query points. "
        #         "Use a larger short_frames value or a query sampler that queries early frames."
        #     )

        sample.gt_query_point = sample.gt_query_point[:, query_mask].clone()
        sample.est_query_point = sample.est_query_point[:, query_mask].clone()
        sample.gt_query_point[..., 0] -= start
        sample.est_query_point[..., 0] -= start

        sample.gt_trajs_3d = sample.gt_trajs_3d[:, :, query_mask]
        sample.est_trajs_3d = sample.est_trajs_3d[:, :, query_mask]
        sample.visibs = sample.visibs[:, :, query_mask]
        sample.valids = sample.valids[:, :, query_mask]

        if sample.gt_trajs_2d is not None and sample.gt_trajs_2d[0] is not None:
            sample.gt_trajs_2d = sample.gt_trajs_2d[:, :, query_mask]
        if sample.gt_query_point_2d is not None and sample.gt_query_point_2d[0] is not None:
            sample.gt_query_point_2d = sample.gt_query_point_2d[:, query_mask]
        if sample.traj_norm_mask is not None and sample.traj_norm_mask[0] is not None:
            sample.traj_norm_mask = sample.traj_norm_mask[:, :, query_mask]
        if sample.query_indices is not None and sample.query_indices[0] is not None:
            sample.query_indices = sample.query_indices[:, query_mask]
        if sample.is_dynamic is not None and sample.is_dynamic[0] is not None:
            sample.is_dynamic = sample.is_dynamic[:, query_mask]

        return sample

    def _crop_training_sample_for_length_schedule(
        self, sample: SliceData, start: int, end: int
    ) -> SliceData:
        total_frames = sample.rgbs.shape[1]
        if start == 0 and end == total_frames:
            return sample

        sample = sample.copy()
        sample.skip_post_init = True

        # Slice frame-indexed tensors first; query/trajectory filtering happens below.
        sample.rgbs = sample.rgbs[:, start:end]
        sample.gt_depths = sample.gt_depths[:, start:end]
        sample.est_depths = sample.est_depths[:, start:end]
        sample.gt_intrinsics = sample.gt_intrinsics[:, start:end]
        sample.est_intrinsics = sample.est_intrinsics[:, start:end]
        sample.gt_extrinsics = sample.gt_extrinsics[:, start:end]
        sample.est_extrinsics = sample.est_extrinsics[:, start:end]
        sample.gt_trajs_3d = sample.gt_trajs_3d[:, start:end]
        sample.est_trajs_3d = sample.est_trajs_3d[:, start:end]
        sample.visibs = sample.visibs[:, start:end]
        sample.valids = sample.valids[:, start:end]

        if sample.gt_trajs_2d is not None and sample.gt_trajs_2d[0] is not None:
            sample.gt_trajs_2d = sample.gt_trajs_2d[:, start:end]
        if sample.gt_depths_downsampled is not None and sample.gt_depths_downsampled[0] is not None:
            sample.gt_depths_downsampled = sample.gt_depths_downsampled[:, start:end]
        if sample.est_depths_downsampled is not None and sample.est_depths_downsampled[0] is not None:
            sample.est_depths_downsampled = sample.est_depths_downsampled[:, start:end]
        if sample.gt_intrinsics_downsampled is not None and sample.gt_intrinsics_downsampled[0] is not None:
            sample.gt_intrinsics_downsampled = sample.gt_intrinsics_downsampled[:, start:end]
        if sample.est_intrinsics_downsampled is not None and sample.est_intrinsics_downsampled[0] is not None:
            sample.est_intrinsics_downsampled = sample.est_intrinsics_downsampled[:, start:end]
        if sample.bg_mask is not None and sample.bg_mask[0] is not None:
            sample.bg_mask = sample.bg_mask[:, start:end]
        if sample.bg_mask_downsampled is not None and sample.bg_mask_downsampled[0] is not None:
            sample.bg_mask_downsampled = sample.bg_mask_downsampled[:, start:end]
        if sample.segmentation is not None and sample.segmentation[0] is not None:
            sample.segmentation = sample.segmentation[:, start:end]
        if sample.dense_norm_mask is not None and sample.dense_norm_mask[0] is not None:
            sample.dense_norm_mask = sample.dense_norm_mask[:, start:end]
        if sample.dense_dynamic is not None and sample.dense_dynamic[0] is not None:
            sample.dense_dynamic = sample.dense_dynamic[:, start:end]

        target_mask = None
        if sample.tgt_time_step_scenes is not None and sample.tgt_time_step_scenes[0] is not None:
            target_steps = sample.tgt_time_step_scenes[0]
            target_mask = (target_steps >= start) & (target_steps < end)
            if not bool(target_mask.any()):
                raise ValueError(
                    f"Length-schedule crop [{start}, {end}) removed all dense target timesteps."
                )
            sample.tgt_time_step_scenes = sample.tgt_time_step_scenes[:, target_mask] - start
            if (
                sample.dense_dynamic is not None
                and sample.dense_dynamic[0] is not None
                and sample.dense_dynamic.dim() == 5
                and sample.dense_dynamic.shape[-1] == target_mask.numel()
            ):
                sample.dense_dynamic = sample.dense_dynamic[..., target_mask]

        def _slice_dense_trajs(tensor):
            if tensor is None or tensor[0] is None:
                return tensor
            tensor = tensor[:, start:end]
            if target_mask is not None and tensor.shape[-2] == target_mask.numel():
                return tensor[..., target_mask, :]
            return tensor[..., start:end, :]

        def _slice_dense_mask(tensor):
            if tensor is None or tensor[0] is None:
                return tensor
            tensor = tensor[:, start:end]
            if target_mask is not None and tensor.shape[-1] == target_mask.numel():
                return tensor[..., target_mask]
            return tensor[..., start:end]

        sample.dense_trajs_3d = _slice_dense_trajs(sample.dense_trajs_3d)
        sample.est_dense_trajs_3d = _slice_dense_trajs(sample.est_dense_trajs_3d)
        sample.dense_valids = _slice_dense_mask(sample.dense_valids)
        sample.dense_visibs = _slice_dense_mask(sample.dense_visibs)

        return self._filter_query_trajectory_dim(sample, start, end)

    def _apply_length_schedule(
        self, step_id: int, sample: SliceData
    ) -> tuple[SliceData, int, int]:
        warmup_frames = int(self.cfg.train.get("warmup_frames", 0))
        schedule_cfg = self.cfg.train.get("length_schedule", None)
        if schedule_cfg is None or not schedule_cfg.get("enabled", False):
            return sample, warmup_frames, sample.rgbs.shape[1]

        total_frames = sample.rgbs.shape[1]
        is_long_step = self._length_schedule_is_long_step(step_id, schedule_cfg)
        long_frames = int(schedule_cfg.get("long_frames", total_frames))
        short_frames = int(schedule_cfg.get("short_frames", total_frames))
        active_frames = long_frames if is_long_step else short_frames
        if active_frames > total_frames:
            raise ValueError(
                f"train.length_schedule requested {active_frames} frames, "
                f"but the loaded sample only has {total_frames}. Set frames_per_sample to at least long_frames."
            )

        if is_long_step:
            warmup_frames = int(schedule_cfg.get("long_warmup_frames", warmup_frames))
        else:
            warmup_frames = int(schedule_cfg.get("short_warmup_frames", 0))

        crop_start = int(schedule_cfg.get("crop_start", 0))
        if crop_start + active_frames > total_frames:
            raise ValueError(
                f"train.length_schedule crop_start={crop_start} and active_frames={active_frames} "
                f"exceed the loaded {total_frames} frames."
            )
        sample = self._crop_training_sample_for_length_schedule(
            sample, crop_start, crop_start + active_frames
        )

        log_freq = int(schedule_cfg.get("log_freq", 100))
        if log_freq > 0 and step_id % log_freq == 0 and get_rank() == 0:
            logger.info(
                "length_schedule step=%d mode=%s active_frames=%d warmup_frames=%d",
                step_id,
                "long" if is_long_step else "short",
                active_frames,
                warmup_frames,
            )
            if self._should_log_wandb():
                wandb.log(
                    {
                        "train/active_frames": active_frames,
                        "train/warmup_frames": warmup_frames,
                        "train/is_long_length_step": int(is_long_step),
                    },
                    step=step_id,
                )

        return sample, warmup_frames, active_frames
    
    @profile
    def train_one_step(self, model, criterion, step_id, sample):
        """Run a single training step."""
        sample.skip_post_init = True
        sample, scheduled_warmup_frames, _ = self._apply_length_schedule(step_id, sample)
            
        est_sample = sample.with_annot_mode("est")

        streaming_loss_list: list | None = None
        streaming_info_list: list | None = None
        if self.cfg.train.multi_backward:
            pred = None
            train_data_list = []
            streaming_loss_list = []
            streaming_info_list = []

            warmup_frames = scheduled_warmup_frames

            with self.accelerator.autocast():
                with record_function("encode_rgbs"):
                    feats = model.encode_rgbs(
                        est_sample.rgbs,
                        warmup_frames=warmup_frames,
                    )

            with record_function("feats_detach_clone"):
                feats_param = torch.nn.Parameter(feats.detach().clone(), requires_grad=not model.freeze_encoder)

            streaming_forward = getattr(
                model,
                "streaming_forward",
                None,
            ) or model.streaming_forward_simple_regression_v6_windowed_overlap

            generator = streaming_forward(
                rgb_obs=est_sample.rgbs,
                depth_obs=est_sample.depths_downsampled,
                num_iters=self.cfg.train.train_iters,
                query_point=est_sample.query_point,
                intrinsics=est_sample.intrinsics_downsampled,
                extrinsics=est_sample.extrinsics,
                image_feats=feats_param,
                disable_backprop_through_time=self.cfg.train.disable_backprop_through_time,  # type: ignore
                dense_norm_mask=est_sample.dense_norm_mask,
                tgt_time_step_dense=est_sample.tgt_time_step_scenes,
                dense_sample_points=est_sample.dense_sample_points,
                warmup_frames=warmup_frames,
            )

            while True:
                with self.accelerator.autocast():
                    with record_function("streaming_forward_step"):
                        output = next(generator, None)
                if output is None:
                    break

                assert (
                    not torch.is_autocast_enabled()
                ), "AMP should not be enabled during loss computation"
                if isinstance(output, TrainData):
                    is_warmup = warmup_frames > 0 and output.frame_range[0] < warmup_frames
                    if is_warmup:
                        train_data_list.append(output.detach(clone=True))  # type: ignore
                        del output
                        continue

                    with record_function("compute_loss"):
                        with autocast("cuda", enabled=False):
                            loss, info = criterion.compute_loss(
                                sample=sample,
                                train_data=output,
                                bidirectional=model.bidirectional,
                                is_train=True,
                            )
                    try:
                        with record_function("backward"):
                            self.accelerator.backward(loss)
                    except Exception as e:
                        print(f"Error in backward pass: {e}")
                        nan_params = [(n, p) for n, p in model.named_parameters() if torch.isnan(p).any()]
                        nan_grads = [(n, p) for n, p in model.named_parameters() if p.grad is not None and torch.isnan(p.grad).any()]
                        print(f"  NaN params: {len(nan_params)}, NaN grads: {len(nan_grads)}")
                        for n, p in nan_params:
                            print(f"    NaN param: {n}, shape={p.shape}, nan_count={torch.isnan(p).sum().item()}")
                        for n, p in nan_grads:
                            print(f"    NaN grad: {n}, shape={p.shape}, nan_count={torch.isnan(p.grad).sum().item()}")
                        st()
                    if streaming_loss_list is not None:
                        # Capture detached loss/info so we can skip the
                        # duplicate compute_loss pass after the streaming loop.
                        # Numerically identical to running compute_loss on the
                        # detached train_data (which is what the second pass
                        # did previously).
                        streaming_loss_list.append(loss.detach())
                        streaming_info_list.append(
                            {
                                k: (v.detach() if torch.is_tensor(v) else v)
                                for k, v in info.items()
                            }
                        )
                    train_data_list.append(output.detach(clone=True))  # type: ignore
                    del output, loss, info
                elif isinstance(output, Prediction):
                    assert pred is None, "Internal error"
                    pred = output
                else:
                    raise ValueError(f"Unknown output type: {type(output)}")

            if not model.freeze_encoder:
                feats_grad = feats_param.grad
                del feats_param
                if feats_grad is not None:
                    assert not feats_grad.requires_grad, "feats_grad should not require grad"
                    with record_function("encoder_backward"):
                        self.accelerator.backward((feats_grad * feats).sum())
                    del feats_grad

            assert pred is not None, "Internal error"
        else:
            pred, train_data_list = model(
                rgb_obs=est_sample.rgbs,
                depth_obs=est_sample.depths_downsampled,
                num_iters=self.cfg.train.train_iters,
                query_point=est_sample.query_point,
                intrinsics=est_sample.intrinsics_downsampled,
                extrinsics=est_sample.extrinsics,
                disable_backprop_through_time=self.cfg.train.disable_backprop_through_time,
                dense_norm_mask=est_sample.dense_norm_mask,
                tgt_time_step_dense=est_sample.tgt_time_step_scenes,
                dense_sample_points=est_sample.dense_sample_points,
                warmup_frames=scheduled_warmup_frames,
            )

        # Loss for backward pass
        if (
            streaming_loss_list is not None
            and len(streaming_loss_list) == len(train_data_list)
        ):
            # Reuse the loss/info already produced inside the streaming loop;
            # running compute_loss again on the detached train_data would
            # produce identical values but doubles the work (~5 s/step).
            # The length check guards against the warmup case, where some
            # train_data entries were skipped in the streaming loop but the
            # old aggregation loop still computed loss on them for logging.
            loss_list = streaming_loss_list
            info_list = streaming_info_list
        else:
            loss_list, info_list = [], []

            for train_data in train_data_list:
                with autocast("cuda", enabled=False):
                    loss, info = criterion.compute_loss(
                        sample=sample,
                        train_data=train_data,
                        bidirectional=self.accelerator.unwrap_model(model).bidirectional,
                        is_train=True,
                    )
                loss_list.append(loss)
                info_list.append(info)
        loss = torch.sum(torch.stack(loss_list), dim=0)
        info = {
            k: torch.sum(torch.stack([info[k] for info in info_list]), dim=0)
            for k in info_list[0]
        }

        # self.output_stream.write(
        #     f"step: {step_id}, seq_name: {sample.seq_name}, loss: {loss.item()}\n"
        # )

        if self.cfg.train.multi_backward:
            assert not loss.requires_grad, "Internal error"
        else:
            assert loss.requires_grad, "Internal error"

        if (step_id + 1) % 100 == 0:
            logger.info("train-loss/total_loss: %f", loss.item())
            for k, v in info.items():
                logger.info(f"train-loss/{k}: {v.item()}")
            # self.output_stream.flush()

        if self._should_log() and (step_id + 1) % 100 == 0:
            if self._should_log_wandb():
                wandb.log(
                    {
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "train-loss/total_loss": loss,
                        "step": step_id,
                    },
                )
            if self._should_log_tb():
                self.tb_writer.add_scalar("train-loss/total_loss", loss.item(), step_id)  # type: ignore
                for k, v in info.items():
                    self.tb_writer.add_scalar(f"train-loss/{k}", v.item(), step_id)  # type: ignore

        return loss, pred, info

    @torch.no_grad()
    def evaluate_nsteps_alltrack3r(
        self, model, criterion, loader, step_id, dataset_name
    ):
        """Run a given number of evaluation steps."""
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process and self.cfg.train.get("save_preds", False):
            path = f"{self.output_dir}/pred_{step_id}"
            path_manager.mkdirs(path)

        self.accelerator.wait_for_everyone()

        values = {"seq_id": []}
        values_print_fixed: dict[str, list] = {}
        device = next(model.parameters()).device
        model.eval()

        log_eval_cuda_stats = bool(
            self.cfg.train.get(
                "log_eval_cuda_stats",
                OmegaConf.select(self.cfg, "eval_only", default=False),
            )
        )
        fwd_latency_clip_s: list[float] = []
        fwd_latency_per_frame_s: list[float] = []
        gpu_peak_mems_gb: list[float] = []
        cuda_timing_warmup_done = False

        bidirectional_eval: bool = self.cfg.train.get("bidirectional_eval", False)
        is_model_bidirectional: bool = getattr(
            self.accelerator.unwrap_model(model), "bidirectional", False
        )

        # ---- Guard against NCCL timeout killing training ----
        # Increase NCCL timeout during eval so a slow/hung sample doesn't
        # trigger SIGABRT from the NCCL watchdog.
        _orig_nccl_timeout = None
        _eval_nccl_timeout_s = self.cfg.train.get("eval_nccl_timeout", 7200)
        if torch.distributed.is_initialized() and _eval_nccl_timeout_s > 0:
            try:
                _pg = torch.distributed.distributed_c10d._get_default_group()
                _orig_nccl_timeout = _pg._get_backend(torch.device("cuda")).options._timeout
                torch.distributed.distributed_c10d._set_pg_timeout(
                    timedelta(seconds=_eval_nccl_timeout_s), _pg
                )
                logger.info("Increased NCCL timeout to %ds for evaluation", _eval_nccl_timeout_s)
            except Exception:
                logger.warning("Could not increase NCCL timeout for evaluation", exc_info=True)
                _orig_nccl_timeout = None

        # Per-sample wall-clock timeout (seconds). 0 = disabled.
        _eval_sample_timeout_s = self.cfg.train.get("eval_sample_timeout", 0)
        _old_sigalrm_handler = None
        if _eval_sample_timeout_s > 0 and hasattr(signal, "SIGALRM"):
            def _sigalrm_handler(signum, frame):
                raise TimeoutError(
                    f"Sample evaluation exceeded {_eval_sample_timeout_s}s timeout"
                )
            _old_sigalrm_handler = signal.signal(signal.SIGALRM, _sigalrm_handler)

        visited_indices = set()
        rank = get_rank()
        for i, sample in enumerate(
            tqdm(loader, position=1, desc=f"Evaluating {dataset_name}")
        ):        
            import time

            start_time = time.time()
            _sample_ok = True
            try:
                sample = sample.to(device)
                sample_id = sample.sample_id.item()
            except Exception:
                logger.exception("Failed to prepare sample %d on rank %d", i, rank)
                sample_id = -1
                _sample_ok = False

            if torch.distributed.is_initialized():
                sample_ids = [None for _ in range(self.accelerator.num_processes)]
                torch.distributed.all_gather_object(sample_ids, (rank, sample_id))
            else:
                sample_ids = [(rank, sample_id)]


            if not _sample_ok:
                continue
            
            # from .datasets.utils.geometry import voxelization
            # from torch_scatter import scatter_mean

            # tgt_time_step_scene = -1
            # h_, w_ = sample.dense_trajs_3d.shape[2:4]
            # points = sample.dense_trajs_3d[
            #     ..., tgt_time_step_scene, :
            # ].flatten(1, 3)
            # valids = sample.dense_valids[..., tgt_time_step_scene]
            # color = sample.rgbs
            # color = F.interpolate(
            #     color.flatten(0, 1),
            #     size=(h_, w_),
            #     mode="bilinear",
            #     align_corners=False,
            # )
            # color = rearrange(color, "t c h w -> 1 (t h w) c")

            # points = points[:, valids.flatten()]
            # color = color[:, valids.flatten()]
            # p2v = voxelization(
            #     points,
            #     self.cfg.model.voxel_size,
            # )
            # print("Compression in input gt geometry: ", p2v.max() / p2v.shape[1])
            # points = scatter_mean(
            #     points,
            #     p2v,
            #     dim=1
            # )
            # colors_voxelized = scatter_mean(
            #     color,
            #     p2v,
            #     dim=1
            # )
            # wandb.log(
            #     {f"{dataset_name}/3d_vis_input_gt_geometry_voxelized": wandb.Object3D(
            #         torch.cat([points.cpu(), colors_voxelized.cpu() * 255.0], dim=-1)[0].numpy(),
            #     )}
            # )
            # continue

            should_skip = False
            for r, idx in sample_ids:
                if idx in visited_indices and r == rank:
                    should_skip = True
                visited_indices.add(idx)

            if should_skip:
                logger.info(
                    f"Skipping sample {sample_id} because it has already been evaluated (likely a padded sample)"
                )
                continue
            try:
                if _eval_sample_timeout_s > 0 and hasattr(signal, "SIGALRM"):
                    signal.alarm(_eval_sample_timeout_s)

                logger.info(f"Evaluating sample {sample_id}, {i}, {get_rank()}")

                est_sample = sample.with_annot_mode("est")
                track_cuda = log_eval_cuda_stats and device.type == "cuda"
                if track_cuda:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)
                t_infer0 = time.perf_counter()
                preds, train_data = model(
                    rgb_obs=est_sample.rgbs,
                    depth_obs=est_sample.depths_downsampled,
                    num_iters=self.cfg.train.eval_iters,
                    query_point=est_sample.query_point,
                    intrinsics=est_sample.intrinsics_downsampled,
                    extrinsics=est_sample.extrinsics,
                    dense_norm_mask=est_sample.dense_norm_mask,
                    tgt_time_step_dense=est_sample.tgt_time_step_scenes,
                    dense_sample_points=est_sample.dense_sample_points,
                )
                if (
                    bidirectional_eval
                    and not is_model_bidirectional
                    and preds.query_coords is not None
                    and preds.query_coords.shape[1] == est_sample.rgbs.shape[1]
                ):
                    T = preds.query_coords.shape[1]
                    backward_query_point = est_sample.query_point.clone()
                    backward_query_point[..., 0] = (
                        est_sample.rgbs.shape[1] - 1 - backward_query_point[..., 0]
                    )

                    backward_tgt_time_step_dense = est_sample.tgt_time_step_scenes.clone()

                    preds_backward, _ = model(
                        rgb_obs=est_sample.rgbs.flip(dims=(1,)),
                        depth_obs=est_sample.depths_downsampled.flip(dims=(1,)),
                        num_iters=self.cfg.train.eval_iters,
                        query_point=backward_query_point,
                        intrinsics=est_sample.intrinsics_downsampled.flip(dims=(1,)),
                        extrinsics=est_sample.extrinsics.flip(dims=(1,)),
                        dense_norm_mask=(
                            est_sample.dense_norm_mask.flip(dims=(1,))
                            if est_sample.dense_norm_mask is not None
                            and est_sample.dense_norm_mask[0] is not None
                            else None
                        ),
                        tgt_time_step_dense=backward_tgt_time_step_dense,
                    )

                    query_frames = est_sample.query_point[..., 0].to(
                        device=preds.query_coords.device
                    )
                    use_backward = (
                        torch.arange(T, device=preds.query_coords.device)[
                            None, :, None
                        ]
                        < query_frames[:, None, :]
                    )
                    preds.query_coords = torch.where(
                        use_backward[..., None],
                        preds_backward.query_coords.flip(dims=(1,)),
                        preds.query_coords,
                    )
                    if (
                        preds.query_visibs is not None
                        and preds_backward.query_visibs is not None
                    ):
                        preds.query_visibs = torch.where(
                            use_backward,
                            preds_backward.query_visibs.flip(dims=(1,)),
                            preds.query_visibs,
                        )
                    del preds_backward

                infer_dt = time.perf_counter() - t_infer0
                if track_cuda:
                    torch.cuda.synchronize()
                peak_gb = (
                    torch.cuda.max_memory_allocated(device) / 1e9
                    if track_cuda
                    else 0.0
                )
                T_clip_inf = int(est_sample.rgbs.shape[1])
                if log_eval_cuda_stats:
                    if cuda_timing_warmup_done:
                        fwd_latency_clip_s.append(infer_dt)
                        fwd_latency_per_frame_s.append(
                            infer_dt / max(T_clip_inf, 1)
                        )
                        if track_cuda:
                            gpu_peak_mems_gb.append(peak_gb)
                    else:
                        cuda_timing_warmup_done = True

                if self.cfg.train.get("visualize_dense_track_with_rerun", False):
                    samples_cpu = sample.copy().to("cpu")
                    preds_cpu = preds.copy().to("cpu")
                    samples_cpu = samples_cpu.with_annot_mode("gt")
                    generate_dense_track_visualizations_with_rerun(
                        sample=samples_cpu,
                        preds=preds_cpu,
                        output_dir=self.output_dir,
                        step_id=step_id,
                        batch_idx=i,
                        dataset_name=dataset_name,
                        mode="voxel",
                        voxel_size=0.02,
                        trail_len=10,
                        max_points=60000,
                        visualize_est=True,
                    )
                    continue

                if self.cfg.train.get("visualize_dense_track_with_wandb", False):
                    samples_cpu = sample.copy().to("cpu")
                    preds_cpu = preds.copy().to("cpu")
                    samples_cpu = samples_cpu.with_annot_mode("gt")
                    generate_dense_track_visualizations_with_wandb(
                        sample=samples_cpu,
                        preds=preds_cpu,
                        output_dir=self.output_dir,
                        step_id=step_id,
                        batch_idx=i,
                        dataset_name=dataset_name,
                        trail_len=8,
                        max_points=20000,
                        dynamic_from_motion=self.cfg.train.get(
                            "dense_track_wandb_dynamic_from_motion", False
                        ),
                        motion_threshold=self.cfg.train.get(
                            "dense_track_wandb_motion_threshold", 0.1
                        ),
                    )
                    continue

                # Use forward prediction to compute loss
                loss_list, info_list = [], []
                if not self.cfg.model.get('mem_efficient', False):
                    for train_data_item in train_data:
                        with autocast("cuda", enabled=False):
                            visualize_geometry = self.cfg.train.get("visualize_geometry", False)
                            loss, info = criterion.compute_loss(
                                sample=sample,
                                train_data=train_data_item,
                                bidirectional=False,  # doesn't matter
                                is_train=False,
                                return_per_point_loss=visualize_geometry,
                            )

                            loss_list.append(loss)
                            info_list.append(info)

                per_point_loss = None
                per_point_valid = None
                if len(loss_list) > 0:
                    loss = torch.sum(torch.stack(loss_list), dim=0)
                    per_point_keys = {"per_point_loss", "per_point_valid"}
                    info = {
                        k: torch.sum(torch.stack([info[k] for info in info_list]), dim=0)
                        for k in info_list[-1] if k not in per_point_keys
                    }
                    if "per_point_loss" in info_list[-1]:
                        per_point_loss = info_list[-1]["per_point_loss"]
                        per_point_valid = info_list[-1]["per_point_valid"]
                else:
                    loss = torch.tensor(0.0, device=device)
                    info = {}
                
                assert sample.seq_id.shape[0] == 1, "batch size must be 1 for evaluation"

                values["seq_id"].append(sample.seq_id.squeeze(0))

                for n, l in info.items():
                    key = f"{dataset_name}-losses/mean/{n}"
                    if key not in values:
                        values[key] = []
                    values[key].append(l)
            
                eval_variants = {}
                eval_modes = copy.deepcopy(self.cfg.train.get("eval_modes", None))
                if eval_modes is None:
                    if self.cfg.train.get("evaluate_with_query_points", False):
                        eval_modes = ["query"]
                    else:
                        eval_modes = ["dense"]

                if sample.dense_trajs_3d[0] is None and eval_modes is not None:
                    if "dense" in eval_modes:
                        eval_modes.remove("dense")

                for mode in eval_modes:
                    if mode == "query":
                        sample_q = sample.copy()
                        preds_q = Prediction(
                            coords=preds.query_coords,
                            visibs=preds.query_visibs,
                            is_dynamic=preds.is_dynamic,
                        )
                        eval_variants["query"] = (
                            sample_q,
                            preds_q,
                            is_model_bidirectional or bidirectional_eval,
                        )
                    elif mode == "dense":
                        sample_d = sample.copy()
                        preds_d = preds.copy()
                        sample_d, preds_d = prepare_dense_tracks_for_eval_with_v6(
                            sample_d,
                            preds_d,
                        )
                        eval_variants["dense"] = (sample_d, preds_d, True)
                    elif mode == "query_through_dense":
                        if is_model_bidirectional or bidirectional_eval:
                            logger.warning("query_through_dense: skipped under bidirectional eval")
                            continue
                        preds_qtd = prepare_query_through_dense_for_eval(est_sample, preds)
                        if preds_qtd is None:
                            logger.warning(
                                "query_through_dense: no dense tracks; set "
                                "model.dense_tracking_at_eval=true"
                            )
                            continue
                        eval_variants["query_through_dense"] = (sample.copy(), preds_qtd, False)
                    elif mode == "dense_sample":
                        if preds.dense_sample_coords is None:
                            continue
                        sample_ds = sample.copy()
                        preds_ds = preds.copy()
                        sample_ds, preds_ds = prepare_dense_sample_tracks_for_eval(
                            sample_ds,
                            preds_ds,
                        )
                        eval_variants["dense_sample"] = (sample_ds, preds_ds, True)

                start_time = time.time()
                metrics_dict = {}
                eval_cfg = self.cfg.get("eval", {})
                use_fixed_metric_threshold = bool(
                    self.cfg.train.get("use_fixed_metric_threshold", False)
                    or eval_cfg.get("use_fixed_metric_threshold", False)
                )
                use_fixed_metric_threshold_startrack = bool(
                    self.cfg.train.get("use_fixed_metric_threshold_startrack", False)
                    or eval_cfg.get("use_fixed_metric_threshold_startrack", False)
                )
                for mode_name, (eval_sample, eval_preds, bidir) in eval_variants.items():
                    prefix = f"{mode_name}_" if mode_name != "query" else ""
                    with autocast("cuda", enabled=False):
                        mode_metrics = metrics_alltrack3r.compute_metrics(
                            sample=eval_sample,
                            preds=eval_preds,
                            bidirectional=bidir,
                            evaluate_with_query_points=False,
                            debug_mode=self.cfg.train.get("eval_debug_mode", False),
                            prefix=prefix,
                            track_all=mode_name == "dense",
                            use_fixed_metric_threshold=use_fixed_metric_threshold,
                            use_fixed_metric_threshold_startrack=use_fixed_metric_threshold_startrack,
                        )
                    metrics_dict.update(mode_metrics)

                    if mode_name in ("query", "query_through_dense") and not (
                        use_fixed_metric_threshold
                        and use_fixed_metric_threshold_startrack
                    ):
                        mode_metrics_fixed = metrics_alltrack3r.compute_metrics(
                            sample=eval_sample,
                            preds=eval_preds,
                            bidirectional=bidir,
                            evaluate_with_query_points=False,
                            debug_mode=self.cfg.train.get("eval_debug_mode", False),
                            prefix=prefix,
                            track_all=False,
                            use_fixed_metric_threshold=True,
                            use_fixed_metric_threshold_startrack=True,
                        )
                        for n, m in mode_metrics_fixed.items():
                            key = f"{dataset_name}-metrics-print-fixed/mean/{n}"
                            if key not in values_print_fixed:
                                values_print_fixed[key] = []
                            values_print_fixed[key].append(m)

                primary_mode = "query" if "query" in eval_variants else (eval_modes[0] if eval_modes else None)
                if primary_mode and primary_mode in eval_variants:
                    _, primary_preds, bidir = eval_variants[primary_mode]
                else:
                    primary_preds = preds
                    bidir = True

                if (
                    self._should_log_wandb()
                    and self.cfg.train.get("visualize_geometry", False)
                    and i < self.cfg.train.visualize_nbatches
                    and preds.scene_memory is not None
                ):   
                    static_points, static_colors, _, _ = preds.scene_memory
                    wandb.log(
                        {f"{dataset_name}/3d_vis_scene_memory": wandb.Object3D(
                            torch.cat([static_points.cpu(), static_colors.cpu() * 255.0], dim=-1).numpy(),
                        )}
                    )

                    from .datasets.utils.geometry import voxelization
                    from torch_scatter import scatter_mean

                    tgt_time_step_scene = -1
                    h_, w_ = sample.dense_trajs_3d.shape[2:4]
                    points = sample.dense_trajs_3d[
                        ..., tgt_time_step_scene, :
                    ].flatten(1, 3)
                    valids = sample.dense_valids[..., tgt_time_step_scene]
                    color = est_sample.rgbs
                    color = F.interpolate(
                        color.flatten(0, 1),
                        size=(h_, w_),
                        mode="bilinear",
                        align_corners=False,
                    )
                    color = rearrange(color, "t c h w -> 1 (t h w) c")
                    points = points[:, valids.flatten()]
                    color = color[:, valids.flatten()]
                    p2v = voxelization(
                        points,
                        self.cfg.model.voxel_size,
                    )
                    print("Compression in input gt geometry: ", p2v.max() / p2v.shape[1])
                    points = scatter_mean(
                        points,
                        p2v,
                        dim=1
                    )
                    colors_voxelized = scatter_mean(
                        color,
                        p2v,
                        dim=1
                    )
                    wandb.log(
                        {f"{dataset_name}/3d_vis_gt_memory": wandb.Object3D(
                            torch.cat([points.cpu(), colors_voxelized.cpu() * 255.0], dim=-1).numpy(),
                        )}
                    )
                

                if (
                    self._should_log_wandb()
                    and self.cfg.train.get("visualize_geometry", False)
                    and i < self.cfg.train.visualize_nbatches
                    and preds.coords is not None
                ):            
                    h_, w_ = train_data[0].coords[0].shape[2:4]
                    tgt_time_step_scene = -1
                    if sample.dense_valids[0] is not None:
                        valids = sample.dense_valids[..., tgt_time_step_scene]
                    points = preds.coords.flatten(1, 3)
                    color = est_sample.rgbs
                    color = F.interpolate(
                        color.flatten(0, 1),
                        size=(h_, w_),
                        mode="bilinear",
                        align_corners=False,
                    )
                    color = rearrange(color, "t c h w -> 1 (t h w) c")

                    if sample.dense_norm_mask[0] is not None:
                        points = points[:, sample.dense_norm_mask.flatten()]
                        color = color[:, sample.dense_norm_mask.flatten()]

                    from .datasets.utils.geometry import voxelization
                    from torch_scatter import scatter_mean
                    p2v = voxelization(
                        points,
                        self.cfg.model.voxel_size,
                    )
                    print("Compression in pred geometry: ", p2v.max() / p2v.shape[1])
                    points = scatter_mean(
                        points,
                        p2v,
                        dim=1
                    )
                    colors_voxelized = scatter_mean(
                        color,
                        p2v,
                        dim=1
                    )
                    geo_vis_objects = {}
                    geo_vis_objects[f"{dataset_name}/3d_vis_pred_geometry_voxelized"] = wandb.Object3D(
                        torch.cat([points.cpu(), colors_voxelized.cpu() * 255.0], dim=-1)[0].numpy(),
                    )
                
                    pred_points = torch.clone(points)
                    pred_colors_voxelized = torch.clone(colors_voxelized)

                    if per_point_loss is not None:
                        ppl = per_point_loss
                        ppv = per_point_valid.float()
                        if ppl.dim() > 4:
                            ppl = (ppl * ppv).sum(-1) / (ppv.sum(-1) + 1e-4)
                        loss_flat = ppl.flatten(1)  # (B, T*H*W)
                        if sample.dense_norm_mask[0] is not None:
                            loss_flat = loss_flat[:, sample.dense_norm_mask.flatten()]
                        loss_voxelized = scatter_mean(
                            loss_flat.unsqueeze(-1), p2v, dim=1
                        ).squeeze(-1)
                        lv = loss_voxelized[0].cpu()
                        lv_min, lv_max = lv.min(), lv.max()
                        lv_norm = (lv - lv_min) / (lv_max - lv_min + 1e-8)
                        import matplotlib.pyplot as plt
                        heatmap_colors = torch.from_numpy(
                            plt.cm.turbo(lv_norm.numpy())[..., :3]
                        ).float()
                        geo_vis_objects[f"{dataset_name}/3d_vis_pred_geometry_loss_heatmap"] = wandb.Object3D(
                            torch.cat([pred_points[0].cpu(), heatmap_colors * 255.0], dim=-1).numpy(),
                        )

                    from .utils.common_utils import batch_unproject

                    original_pcds = batch_unproject(
                        est_sample.depths,
                        est_sample.intrinsics,
                        est_sample.extrinsics
                    )
                    color = est_sample.rgbs
                    points = F.interpolate(
                        original_pcds.flatten(0, 1),
                        size=(h_, w_),
                        mode="nearest-exact",
                    )
                    # points = points.permute(0, 2, 3, 1)[valids[0]][None]
                    points = points.permute(0, 2, 3, 1).flatten(0, 2)[None]
                    color = F.interpolate(
                        color.flatten(0, 1),
                        size=(h_, w_),
                        mode="bilinear",
                        align_corners=False,
                    )
                    color = rearrange(color, "t c h w -> 1 (t h w) c")
                    p2v = voxelization(
                        points,
                        self.cfg.model.voxel_size,
                    )
                    print("Compression in input est geometry: ", p2v.max() / p2v.shape[1])
                    points = scatter_mean(
                        points,
                        p2v,
                        dim=1
                    )
                    colors_voxelized = scatter_mean(
                        color,
                        p2v,
                        dim=1
                    )
                    geo_vis_objects[f"{dataset_name}/3d_vis_input_est_geometry_voxelized"] = wandb.Object3D(
                        torch.cat([points.cpu(), colors_voxelized.cpu() * 255.0], dim=-1)[0].numpy(),
                    )
                    points_est = torch.clone(points)
                    colors_voxelized_est = torch.clone(colors_voxelized)

                    if sample.dense_trajs_3d[0] is not None:
                        points = sample.dense_trajs_3d[
                            ..., tgt_time_step_scene, :
                        ].flatten(1, 3)
                    else:
                        gt_sample = sample.with_annot_mode("gt")
                        original_pcds = batch_unproject(
                            gt_sample.depths[:, tgt_time_step_scene],
                            gt_sample.intrinsics[:, tgt_time_step_scene],
                            gt_sample.extrinsics[:, tgt_time_step_scene]
                        )
                        points = F.interpolate(
                            original_pcds,
                            size=(h_, w_),
                            mode="nearest-exact",
                        )
                        points = points.permute(0, 2, 3, 1).flatten(0, 2)[None]
                        color = gt_sample.rgbs[:, tgt_time_step_scene]
                        color = F.interpolate(
                            color,
                            size=(h_, w_),
                            mode="bilinear",
                            align_corners=False,
                        )
                        color = rearrange(color, "t c h w -> 1 (t h w) c")
                        valids = torch.ones_like(color[0, :, 0], dtype=bool)

                    points = points[:, valids.flatten()]
                    color = color[:, valids.flatten()]
                    p2v = voxelization(
                        points,
                        self.cfg.model.voxel_size,
                    )
                    print("Compression in input gt geometry: ", p2v.max() / p2v.shape[1])
                    points = scatter_mean(
                        points,
                        p2v,
                        dim=1
                    )
                    colors_voxelized = scatter_mean(
                        color,
                        p2v,
                        dim=1
                    )
                    geo_vis_objects[f"{dataset_name}/3d_vis_input_gt_geometry_voxelized"] = wandb.Object3D(
                        torch.cat([points.cpu(), colors_voxelized.cpu() * 255.0], dim=-1)[0].numpy(),
                    )
                
                    # combine gt and pred points and colors
                    points_combined = torch.cat([points, pred_points], dim=1)
                    colors_voxelized_combined = torch.cat([colors_voxelized, pred_colors_voxelized], dim=1)
                    geo_vis_objects[f"{dataset_name}/3d_vis_combined_gt_pred_geometry_voxelized"] = wandb.Object3D(
                        torch.cat([points_combined.cpu(), colors_voxelized_combined.cpu() * 255.0], dim=-1)[0].numpy(),
                    )
                
                    points_combined = torch.cat([points, points_est], dim=1)
                    colors_voxelized_combined = torch.cat([colors_voxelized, colors_voxelized_est], dim=1)
                    geo_vis_objects[f"{dataset_name}/3d_vis_combined_gt_est_geometry_voxelized"] = wandb.Object3D(
                        torch.cat([points_combined.cpu(), colors_voxelized_combined.cpu() * 255.0], dim=-1)[0].numpy(),
                    )
                
                    colors_voxelized_est[:] = torch.tensor([1.0, 0.0, 0.0])
                    colors_voxelized[:] = torch.tensor([0.0, 0.0, 1.0])
                    colors_voxelized_combined = torch.cat([colors_voxelized, colors_voxelized_est], dim=1)
                    geo_vis_objects[f"{dataset_name}/3d_vis_combined_gt_est_geometry_voxelized_red_blue"] = wandb.Object3D(
                        torch.cat([points_combined.cpu(), colors_voxelized_combined.cpu() * 255.0], dim=-1)[0].numpy(),
                    )
                
            
                
                
                    for k, v in geo_vis_objects.items():
                        wandb.log({k: v})

                del train_data
                gc.collect()
                
                end_time = time.time()
                logger.info(
                    f"Time taken for evaluation {i}: {end_time - start_time:.2f} seconds"
                )

                if self.cfg.train.get("save_preds", False):
                    metrics_dict_np = {
                        k: v.detach().cpu().reshape(-1).numpy()[0]
                        for k, v in metrics_dict.items()
                    }
                    preds_np = {
                        "coords": primary_preds.coords.cpu().numpy()[0],
                        "visibs": primary_preds.visibs.cpu().numpy()[0],
                    }

                    with fsspec.open(
                        f"{self.output_dir}/pred_{step_id}/{sample_id}.pkl", "wb"
                    ) as f:
                        pickle.dump(
                            {
                                "preds": preds_np,
                                "metrics": metrics_dict_np,
                            },
                            f,
                        )

                # Static/dynamic classification accuracy (query + dense) vs the
                # same GT used to supervise the dynamic BCE heads.
                try:
                    sd_kwargs = dict(
                        dynamic_filter_threshold=sample.dynamic_filter_threshold,
                        dynamic_threshold=self.cfg.model.get("dynamic_threshold", 0.5),
                        window_size=self.cfg.model.get("seq_len", None),
                    )
                    sd_metrics = metrics_alltrack3r.compute_static_dynamic_metrics(
                        sample=sample,
                        preds=preds,
                        tag="query",
                        **sd_kwargs,
                    )
                    metrics_dict.update(sd_metrics)
                    if "dense_sample" in eval_variants:
                        ds_sample, ds_preds, _ = eval_variants["dense_sample"]
                        sd_dense = metrics_alltrack3r.compute_static_dynamic_metrics(
                            sample=ds_sample,
                            preds=ds_preds,
                            tag="dense",
                            **sd_kwargs,
                        )
                        metrics_dict.update(sd_dense)
                except Exception:
                    logger.exception(
                        "static/dynamic classification metrics failed on sample %d",
                        sample_id,
                    )

                for n, m in metrics_dict.items():
                    key = f"{dataset_name}-metrics/mean/{n}"
                    if key not in values:
                        values[key] = []
                    values[key].append(m)

                if i < self.cfg.train.visualize_nbatches:
                    if self._should_log():
                        if step_id > -1:
                            vis_modes_to_show = self.cfg.train.get("visualize_eval_modes", None)
                            if vis_modes_to_show is None:
                                vis_modes_to_show = list(eval_variants.keys())

                            for vis_mode_name in vis_modes_to_show:
                                if vis_mode_name not in eval_variants:
                                    continue

                                vis_sample, vis_preds, _ = eval_variants[vis_mode_name]
                                samples_cpu = vis_sample.copy().to("cpu")
                                preds_cpu = vis_preds.copy().to("cpu")
                                
                                if not bidir:
                                    samples_cpu = samples_cpu.with_causal_mask()

                                samples_cpu = samples_cpu.with_annot_mode("est")

                                if preds_cpu.coords.shape[2] > 5000:
                                    valid_idx = samples_cpu.valids.any(dim=1)[0].nonzero().squeeze(1)
                                    subsample_idx = torch.randperm(valid_idx.shape[0])[:5000]
                                    subsample_idx = valid_idx[subsample_idx]
                                    preds_cpu.coords = preds_cpu.coords[:, :, subsample_idx]
                                    preds_cpu.visibs = preds_cpu.visibs[:, :, subsample_idx]
                                    samples_cpu.gt_trajs_3d = samples_cpu.gt_trajs_3d[:, :, subsample_idx]
                                    samples_cpu.est_trajs_3d = samples_cpu.est_trajs_3d[:, :, subsample_idx]
                                    samples_cpu.visibs = samples_cpu.visibs[:, :, subsample_idx]
                                    samples_cpu.valids = samples_cpu.valids[:, :, subsample_idx]
                                vis_dataset_name = f"{dataset_name}"
                                if vis_mode_name != "query":
                                    vis_dataset_name = f"{dataset_name}-{vis_mode_name}"

                                generate_videos(
                                    sample=samples_cpu,
                                    preds=preds_cpu,
                                    output_dir=self.output_dir,
                                    step_id=step_id,
                                    batch_idx=i,
                                    dataset_name=vis_dataset_name,
                                    log_wandb=self._should_log_wandb(),
                                    writer=self.tb_writer if not self._should_log_wandb() else None,
                                )

                            # Static/dynamic classification visualizations
                            # (query GT|PRED points; dense_sample GT|PRED points;
                            # full-grid dense PRED-only overlay).
                            if self.cfg.train.get("visualize_dynamic_classification", True):
                                try:
                                    sd_sample_cpu = sample.copy().to("cpu").with_annot_mode("est")
                                    sd_preds_cpu = preds.copy().to("cpu")
                                    dyn_thresh = self.cfg.model.get("dynamic_threshold", 0.5)
                                    dyn_filter = getattr(
                                        criterion, "dynamic_filter_threshold", 0.1
                                    )
                                    sd_window = self.cfg.model.get("seq_len", None)
                                    sd_log_wandb = self._should_log_wandb()
                                    sd_writer = self.tb_writer if not sd_log_wandb else None
                                    generate_query_dynamic_videos(
                                        sample=sd_sample_cpu,
                                        preds=sd_preds_cpu,
                                        output_dir=self.output_dir,
                                        step_id=step_id,
                                        batch_idx=i,
                                        dataset_name=dataset_name,
                                        dynamic_threshold=dyn_thresh,
                                        dynamic_filter_threshold=dyn_filter,
                                        log_wandb=sd_log_wandb,
                                        writer=sd_writer,
                                        window_size=sd_window,
                                    )
                                    if "dense_sample" in eval_variants:
                                        ds_sample, ds_preds, _ = eval_variants[
                                            "dense_sample"
                                        ]
                                        generate_query_dynamic_videos(
                                            sample=ds_sample.copy()
                                            .to("cpu")
                                            .with_annot_mode("est"),
                                            preds=ds_preds.copy().to("cpu"),
                                            output_dir=self.output_dir,
                                            step_id=step_id,
                                            batch_idx=i,
                                            dataset_name=dataset_name,
                                            dynamic_threshold=dyn_thresh,
                                            dynamic_filter_threshold=dyn_filter,
                                            log_wandb=sd_log_wandb,
                                            writer=sd_writer,
                                            window_size=sd_window,
                                            video_tag="dynamic-dense_sample",
                                        )
                                    generate_dense_dynamic_pred_videos(
                                        sample=sd_sample_cpu,
                                        preds=sd_preds_cpu,
                                        output_dir=self.output_dir,
                                        step_id=step_id,
                                        batch_idx=i,
                                        dataset_name=dataset_name,
                                        dynamic_threshold=dyn_thresh,
                                        log_wandb=sd_log_wandb,
                                        writer=sd_writer,
                                    )
                                except Exception:
                                    logger.exception(
                                        "static/dynamic classification viz failed on sample %d",
                                        sample_id,
                                    )

                if _eval_sample_timeout_s > 0 and hasattr(signal, "SIGALRM"):
                    signal.alarm(0)

                end_time = time.time()
                logger.info(
                    f"Time taken for batch {i}: {end_time - start_time:.2f} seconds"
                )
            except Exception:
                if _eval_sample_timeout_s > 0 and hasattr(signal, "SIGALRM"):
                    signal.alarm(0)
                logger.exception(
                    "Evaluation of sample %d failed on rank %d; skipping.",
                    sample_id, rank,
                )
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
                continue

        if (
            log_eval_cuda_stats
            and self.accelerator.is_main_process
            and len(fwd_latency_clip_s) > 0
        ):
            arr = np.asarray(fwd_latency_clip_s)
            pf = np.asarray(fwd_latency_per_frame_s)
            _eval_console_log(
                f"=== Ours ({dataset_name}) forward latency "
                f"(n_clips={len(fwd_latency_clip_s)}, warmup=1 dropped) ==="
            )
            _eval_console_log(f"  per-clip  mean   {float(arr.mean()):.4f} s")
            _eval_console_log(f"  per-clip  median {float(np.median(arr)):.4f} s")
            _eval_console_log(f"  per-clip  std    {float(arr.std()):.4f} s")
            _eval_console_log(f"  per-frame mean   {float(pf.mean()):.4f} s")
            _eval_console_log(f"  per-frame median {float(np.median(pf)):.4f} s")
            _eval_console_log(f"  per-frame std    {float(pf.std()):.4f} s")

        if (
            log_eval_cuda_stats
            and self.accelerator.is_main_process
            and len(gpu_peak_mems_gb) > 0
        ):
            mem_arr = np.asarray(gpu_peak_mems_gb)
            _eval_console_log(
                f"=== Ours ({dataset_name}) GPU peak memory "
                f"(n={len(gpu_peak_mems_gb)}, warmup=1 dropped) ==="
            )
            _eval_console_log(f"  mean   {float(mem_arr.mean()):.3f} GB")
            _eval_console_log(f"  median {float(np.median(mem_arr)):.3f} GB")
            _eval_console_log(f"  std    {float(mem_arr.std()):.3f} GB")
            _eval_console_log(f"  max    {float(mem_arr.max()):.3f} GB")

        # ---- Restore NCCL timeout and signal handler ----
        if _old_sigalrm_handler is not None:
            signal.signal(signal.SIGALRM, _old_sigalrm_handler)
        elif _eval_sample_timeout_s > 0 and hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, signal.SIG_DFL)

        if _orig_nccl_timeout is not None:
            try:
                _pg = torch.distributed.distributed_c10d._get_default_group()
                torch.distributed.distributed_c10d._set_pg_timeout(_orig_nccl_timeout, _pg)
                logger.info("Restored original NCCL timeout after evaluation")
            except Exception:
                pass

        # Synchronize between processes
        self.accelerator.wait_for_everyone()

        values = {k: torch.stack(v).cpu() for k, v in values.items() if len(v) > 0}
        print("number of values before dist init: ", len(values))
        if torch.distributed.is_initialized():
            values_list = [None for _ in range(self.accelerator.num_processes)]
            torch.distributed.all_gather_object(values_list, values)
        else:
            values_list = [values]

        # Eval metric keys are populated conditionally per sample (e.g.
        # dense_sample_* metrics only appear for some samples), so ranks do NOT
        # in general agree on which keys are present. Iterating the local
        # rank's keys and blindly indexing every other rank (the previous
        # behaviour) raised KeyError whenever a key was missing on some rank,
        # which silently skipped the entire eval dataset. Aggregate over the
        # union of keys instead, concatenating only the ranks that reported
        # each key. This is mean-equivalent to the old path when keys are
        # uniform and simply degrades gracefully when they are not.
        all_keys: set[str] = set()
        for vl in values_list:
            if vl:
                all_keys.update(vl.keys())

        missing_by_rank = {
            rank_i: sorted(all_keys - set(vl.keys()))
            for rank_i, vl in enumerate(values_list)
            if vl is not None and (all_keys - set(vl.keys()))
        }
        if missing_by_rank:
            logger.warning(
                "[%s] eval metric keys are not uniform across ranks; "
                "aggregating each key over only the ranks that reported it. "
                "Missing keys per rank: %s",
                dataset_name,
                missing_by_rank,
            )

        values_all = {
            k: torch.cat(
                [vl[k] for vl in values_list if vl is not None and k in vl]
            ).numpy()
            for k in all_keys
        }
        values = {k: np.nanmean(v) for k, v in values_all.items()}

        print("number of values before dist init: ", len(values))

        # Sanity check: a dataset that produced no metrics at all means every
        # sample failed (or the loader was empty). Previously this surfaced as
        # a confusing downstream KeyError/skip; make it explicit so eval
        # regressions are caught instead of silently dropping the dataset.
        if len(values) <= 1:  # only "seq_id" (or nothing) survived
            logger.error(
                "[%s] eval produced no usable metrics (only keys: %s). "
                "All samples likely failed or the eval loader is empty.",
                dataset_name,
                sorted(values.keys()),
            )

        vis_related_keys = set(
            [
                re.fullmatch(r"(.*)_visthr_.*", k).groups()[0]
                for k in values.keys()
                if re.fullmatch(r"(.*)_visthr_.*", k)
            ]
        )
        for k in vis_related_keys:
            values[k + "_best"] = np.max(
                [v for k_, v in values.items() if k_.startswith(f"{k}_visthr")]
            )

        # Log to wandb
        to_log = {
            "step": step_id,
        }

        base_metrics = dict(CORE_DISPLAY_METRICS)
        eval_modes_cfg = self.cfg.train.get("eval_modes", None)
        if eval_modes_cfg is None:
            if self.cfg.train.get("evaluate_with_query_points", False):
                eval_modes_cfg = ["query"]
            else:
                eval_modes_cfg = ["dense"]
        core_keys_mapping = {}
        for mode in eval_modes_cfg:
            prefix = f"{mode}_" if mode != "query" else ""
            display_prefix = f"{mode.capitalize()}-" if mode != "query" else ""
            for metric_key, display_name in base_metrics.items():
                core_keys_mapping[f"{prefix}{metric_key}"] = f"{display_prefix}{display_name}"
        core_keys = list(core_keys_mapping.keys())
        core_keys.sort()

        if self._should_log_wandb():
            table_content = []
            n_seq = len(values_all.get("seq_id", []))
            values_all_suffix = {k.split("/")[-1]: k for k in values_all.keys()}
            # Only include per-sample metrics that are aligned with seq_id.
            # With non-uniform keys across ranks a conditional metric can be
            # shorter than seq_id; indexing it per-sample would raise
            # IndexError, so drop such keys from the per-sequence table.
            available_core_keys = [
                k
                for k in core_keys
                if k in values_all_suffix
                and len(values_all[values_all_suffix[k]]) == n_seq
            ]
            for i in range(n_seq):
                table_content.append(
                    [int(values_all["seq_id"][i])]
                    + [
                        "%.4f" % float(values_all[values_all_suffix[k]][i])
                        for k in available_core_keys
                    ]
                )
            table_content.sort(key=lambda x: x[0])

        to_log.update(values)

        if self._should_log_wandb():
            wandb.log(to_log)

        eval_splits = ["", "static_only_", "dynamic_only_", "occ_only_"]

        if self._should_log_tb():
            for k, v in values.items():
                self.tb_writer.add_scalar(
                    f"{dataset_name}-metrics/mean/{k}", v, step_id
                )  # type: ignore

            for split in eval_splits:
                split_label = split.rstrip("_") if split else ""
                for metric_key, display_name in core_keys_mapping.items():
                    full_key = f"{dataset_name}-metrics/mean/{split}{metric_key}"
                    if full_key in values:
                        tag = f"{dataset_name}-Mean-{display_name}"
                        if split_label:
                            tag += f"-{split_label}"
                        self.tb_writer.add_scalar(tag, values[full_key], step_id)

        terminal_table: dict[str, float] = {}
        for split in eval_splits:
            split_label = split.rstrip("_") if split else ""
            for metric_key, display_name in core_keys_mapping.items():
                full_key = f"{dataset_name}-metrics/mean/{split}{metric_key}"
                if full_key in values:
                    label = (
                        f"{split_label} {display_name}"
                        if split_label
                        else display_name
                    ).strip()
                    terminal_table[label] = float(values[full_key])

        # Static/dynamic classification metrics are not part of core_keys_mapping;
        # surface them in the terminal table explicitly so they get printed.
        _sd_display = {
            "correct_all": "Acc",
            "correct_dynamic": "Dyn Acc",
            "correct_static": "Stat Acc",
            "dynamic_precision": "Dyn Prec",
            "dynamic_recall": "Dyn Recall",
            "dynamic_f1": "Dyn F1",
            "static_precision": "Stat Prec",
            "static_recall": "Stat Recall",
            "static_f1": "Stat F1",
            "gt_dynamic_frac": "GT Dyn Frac",
        }
        _sd_prefix = f"{dataset_name}-metrics/mean/"
        for full_key in sorted(values.keys()):
            if not full_key.startswith(_sd_prefix):
                continue
            suffix = full_key[len(_sd_prefix):]
            if suffix.startswith("sd_query_"):
                metric_name = suffix[len("sd_query_"):]
                scope = "Query SD"
            elif suffix.startswith("sd_dense_"):
                metric_name = suffix[len("sd_dense_"):]
                scope = "Dense SD"
            else:
                continue
            display = _sd_display.get(metric_name, metric_name)
            terminal_table[f"{scope} {display}"] = float(values[full_key])

        if terminal_table:
            log_multi_dataset_eval_table(step_id, {dataset_name: terminal_table})

        values_pf_mean: dict[str, float] = {}
        if values_print_fixed:
            vpf_stacked = {
                k: torch.stack(v).cpu()
                for k, v in values_print_fixed.items()
                if len(v) > 0
            }
            if torch.distributed.is_initialized():
                vpf_list = [None for _ in range(self.accelerator.num_processes)]
                torch.distributed.all_gather_object(vpf_list, vpf_stacked)
            else:
                vpf_list = [vpf_stacked]
            vpf_all = {
                k: torch.cat([vpf_list[i][k] for i in range(len(vpf_list))]).numpy()
                for k in vpf_stacked.keys()
            }
            values_pf_mean = {k: float(np.nanmean(v)) for k, v in vpf_all.items()}

        pfx_pf = f"{dataset_name}-metrics-print-fixed/mean/"
        terminal_table_fixed: dict[str, float] = {}
        if values_pf_mean:
            for split in eval_splits:
                split_label = split.rstrip("_") if split else ""
                for metric_key, display_name in core_keys_mapping.items():
                    full_key = f"{pfx_pf}{split}{metric_key}"
                    if full_key in values_pf_mean:
                        label = (
                            f"{split_label} {display_name}"
                            if split_label
                            else display_name
                        ).strip()
                        terminal_table_fixed[label] = values_pf_mean[full_key]

        ref_metric_candidates = []
        if "query" in eval_modes_cfg:
            ref_metric_candidates.append(
                ("tapvid3d_256_iso_average_pts_within_thresh", True)
            )
        for mode in eval_modes_cfg:
            if mode == "dense":
                ref_metric_candidates.append(
                    ("dense_tapvid3d_256_iso_average_pts_within_thresh", True)
                )

        ref_loss = None
        for metric_key, maximize in ref_metric_candidates:
            full_key = f"{dataset_name}-metrics/mean/{metric_key}"
            if full_key in values:
                ref_loss = -values[full_key] if maximize else values[full_key]
                break
        if ref_loss is None:
            ref_loss = 0.0
        return ref_loss, terminal_table, terminal_table_fixed


@hydra.main(config_path="configs", config_name="test", version_base="1.3")
def main(cfg: DictConfig):
    setup_logger()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logger.info(cfg)

    torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    train_tester = TrainTester(cfg)
    if cfg.get("profile", False):
        if cfg.train.num_workers > 0:
            logger.warning(
                "Profile is not supported with num_workers > 0, so we manually set it to 0. This will slow down the training."
            )
            cfg.train.num_workers = 0
            cfg.train.prefetch_factor = None
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
                profiler.ProfilerActivity.XPU,
            ],
            with_stack=True,
        ) as prof:
            train_tester.main(collate_fn=SliceData.collate)
        print(
            prof.key_averages(group_by_stack_n=1).table(
                sort_by="self_cpu_time_total", row_limit=20
            )
        )
        print(
            prof.key_averages(group_by_stack_n=1).table(
                sort_by="self_cuda_time_total", row_limit=20
            )
        )
    else:
        train_tester.main(collate_fn=SliceData.collate)


if __name__ == "__main__":
    main()
