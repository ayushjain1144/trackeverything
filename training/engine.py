# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared utilities for all main scripts."""

import contextlib
import json
import logging
import os
import signal
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import accelerate
import fsspec
import ipdb

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import wandb
from einops import rearrange, repeat
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

from torchinfo import summary
from viztracer import VizTracer

from line_profiler import profile
from omegaconf import DictConfig, OmegaConf

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm import tqdm
from transformers.optimization import get_constant_schedule_with_warmup

from ..utils.alltrack3r_utils import prepare_dense_tracks_for_eval
from ..utils.vis_utils import generate_videos

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
st = ipdb.set_trace


def log_multi_dataset_eval_table(
    step_id: int,
    dataset_columns: Dict[str, Dict[str, float]],
) -> None:
    """Print one table: metric names as rows, dataset names as columns."""
    if not dataset_columns or get_rank() != 0:
        return
    datasets = sorted(dataset_columns.keys())
    row_keys: set[str] = set()
    for col in dataset_columns.values():
        row_keys.update(col.keys())
    rows = sorted(row_keys)
    if not rows:
        return
    first_w = max(len("Metric"), max(len(r) for r in rows))
    col_widths: list[int] = []
    for d in datasets:
        w = len(d)
        for r in rows:
            v = dataset_columns[d].get(r)
            if v is not None:
                w = max(w, len(f"{v:.3f}"))
        col_widths.append(max(w, len(d)))
    sep = " | "
    header = "Metric".ljust(first_w) + sep + sep.join(
        datasets[i].ljust(col_widths[i]) for i in range(len(datasets))
    )
    rule = "-" * len(header)
    body_lines = [header, rule]
    for r in rows:
        cells = []
        for i, d in enumerate(datasets):
            v = dataset_columns[d].get(r)
            if v is None:
                cells.append("—".ljust(col_widths[i]))
            else:
                cells.append(f"{v:.3f}".ljust(col_widths[i]))
        body_lines.append(r.ljust(first_w) + sep + sep.join(cells))
    logger.info("Eval metrics (step %s):\n%s", step_id, "\n".join(body_lines))


def dump_eval_tables_json(
    point_thresh: Dict[str, Dict[str, float]],
    fixed_thresh: Dict[str, Dict[str, float]],
    step_id: int,
) -> None:
    """Write the eval tables to `$EVAL_METRICS_JSON`, if that env var is set.

    Opt-in and additive: the terminal tables are the only machine-readable
    record of an `eval_only` run otherwise, and scraping them back out of a
    slurm log is brittle. `point_thresh` is APD-P (depth-relative pixel
    thresholds), `fixed_thresh` is APD-M (fixed {0.1,0.3,0.5,1.0} m thresholds).
    """
    path = os.environ.get("EVAL_METRICS_JSON")
    if not path or get_rank() != 0:
        return
    payload = {
        "step": step_id,
        "point_thresh": point_thresh,
        "fixed_thresh": fixed_thresh,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    logger.info("Wrote eval metrics JSON to %s", path)


def trace_handler(prof: torch.profiler.profile):
    # Prefix for file names.
    # Construct the trace file.
    prof.export_chrome_trace(f"memory.json.gz")

    # Construct the memory timeline file.
    prof.export_memory_timeline(f"memory.html", device="cuda:0")


def chrome_trace_handler(prof: torch.profiler.profile):
    prof.export_chrome_trace("chrome_trace.json")


class BaseTrainTester:
    """Basic train/test class to be inherited."""

    def _should_log_wandb(self) -> bool:
        return self.cfg.wandb.enable and get_rank() == 0

    def _should_log_tb(self) -> bool:
        return self.cfg.tb.enable and get_rank() == 0

    def _should_log(self) -> bool:
        return self._should_log_wandb() or self._should_log_tb()

    def __init__(self, cfg: DictConfig):
        """Initialize."""
        self.cfg = cfg
        self.gradient_accumulation_steps = self.cfg.train.gradient_accumulation_steps

        self.num_trajs = []
        self.num_dynamic_trajs = []

    @staticmethod
    def get_datasets() -> Tuple[Dataset, Dict[str, Dataset]]:
        """Initialize datasets."""
        raise NotImplementedError()

    def get_loaders(
        self, collate_fn=default_collate, train_only=False
    ) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        """Initialize data loaders."""
        train_loader = None
        if self.train_dataset is not None:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.cfg.train.batch_size,
                shuffle=True,
                num_workers=self.cfg.train.num_workers,
                collate_fn=collate_fn,
                prefetch_factor=self.cfg.train.get("prefetch_factor", 2),
                pin_memory=True,
                drop_last=True,
            )
        test_loaders = (
            OrderedDict(
                {
                    key: DataLoader(
                        self.test_datasets[key],
                        batch_size=1,
                        shuffle=False,
                        num_workers=self.cfg.train.num_workers,
                        prefetch_factor=self.cfg.train.get("prefetch_factor", 2),
                        collate_fn=collate_fn,
                        pin_memory=True,
                        drop_last=False,
                    )
                    for key in self.test_datasets
                }
            )
            if not train_only
            else {}
        )
        return train_loader, test_loaders

    @staticmethod
    def get_model():
        """Initialize the model."""
        return None

    def get_criterion(self):
        """Get loss criterion for training."""
        # criterion is a class, must have compute_loss and compute_metrics
        return None

    def get_optimizer(self, model):
        """Initialize optimizer."""
        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": 0.0, "lr": self.cfg.train.lr},
            {
                "params": [],
                "weight_decay": self.cfg.train.weight_decay,
                "lr": self.cfg.train.lr,
            },
        ]
        no_decay = [
            "bias",
            "LayerNorm.weight",
            "LayerNorm.bias",
            "GroupNorm.weight",
            "GroupNorm.bias",
            "InstanceNorm.weight",
            "InstanceNorm.bias",
            "InstanceNorm1d.weight",
            "InstanceNorm1d.bias",
            "InstanceNorm2d.weight",
            "InstanceNorm2d.bias",
        ]
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                optimizer_grouped_parameters[0]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)
        self.trainable_parameters = list(
            filter(lambda p: p.requires_grad, model.parameters())
        )

        optimizer_name = self.cfg.train.get("optimizer_name", "adamw").lower()

        if optimizer_name == "adamw":
            adamw_kwargs = self.cfg.train.get("optimizer_kwargs", {}) or {}
            adamw_kwargs = OmegaConf.to_container(adamw_kwargs, resolve=True) if isinstance(adamw_kwargs, DictConfig) else dict(adamw_kwargs)
            if "betas" in adamw_kwargs and adamw_kwargs["betas"] is not None:
                adamw_kwargs["betas"] = tuple(adamw_kwargs["betas"])
            optimizer = optim.AdamW(optimizer_grouped_parameters, **adamw_kwargs)
        # elif optimizer_name == "adamw_schedulefree":
        #     assert (
        #         self.cfg.train.scheduler_name == "constant"
        #     ), "Please disable the learning rate schedule when using a schedule free optimizer"
        #     assert all(
        #         [not "BatchNorm" in name for name, param in model.named_parameters()]
        #     ), "BatchNorm is not supported when using a schedule free optimizer"
        #     assert (
        #         self.cfg.train.max_grad_norm is None
        #     ), "Gradient clipping should be disabled when using a schedule free optimizer"
        #     optimizer = AdamWScheduleFree(
        #         optimizer_grouped_parameters,
        #         **self.cfg.train.get("optimizer_kwargs", {}),
        #     )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        return optimizer

    @staticmethod
    def _set_optimizer_lrs(optimizer, lr):
        for group in optimizer.param_groups:
            group["lr"] = lr
            if "initial_lr" in group:
                group["initial_lr"] = lr

    def get_scheduler(self, optimizer):
        """Initialize scheduler."""
        if self.cfg.train.scheduler_name == "constant":
            self._set_optimizer_lrs(optimizer, self.cfg.train.lr)
            return None
        elif self.cfg.train.scheduler_name == "onecycle":
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.cfg.train.lr,
                total_steps=self.cfg.train.train_steps + 100,
                **self.cfg.train.scheduler_kwargs,
            )
        elif self.cfg.train.scheduler_name == "constant_with_warmup":
            return get_constant_schedule_with_warmup(
                optimizer, **self.cfg.train.scheduler_kwargs
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.cfg.train.scheduler_name}")

    def broadcast_model(self, model: torch.nn.Module):
        assert not isinstance(
            model, DistributedDataParallel
        ), "The model should not be manually broadcasted when using DDP"
        params = list(model.named_parameters())
        buffers = list(model.named_buffers())

        params.sort(key=lambda x: x[0])
        buffers.sort(key=lambda x: x[0])

        for name, param in params:
            accelerate.utils.broadcast(param.data)

        for name, buffer in buffers:
            accelerate.utils.broadcast(buffer.data)

    def sync_gradients(self, model: torch.nn.Module):
        assert not isinstance(
            model, DistributedDataParallel
        ), "Gradients should not be manually synced when using DDP"
        world_size = self.accelerator.num_processes

        grads = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grads.append((name, param.grad))
                else:
                    param.grad = torch.zeros_like(param.data)
                    grads.append((name, param.grad))
        grads.sort(key=lambda x: x[0])

        grad_tensors = [grad.view(-1) for name, grad in grads]
        flat_grads = torch.cat(grad_tensors)

        if world_size > 1:
            dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
            flat_grads /= world_size

        pointer = 0
        for _, grad in grads:
            numel = grad.numel()
            grad.copy_(flat_grads[pointer : pointer + numel].view_as(grad))
            pointer += numel

    def check_nan_gradients(self, model: torch.nn.Module):
        for _, param in model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and torch.isnan(param.grad).any()
            ):
                return False
        return True

    def visualize_bad_sample_data(
        self, loss, sample, step_id, pred, info, max_grad_norm
    ):
        log_string = (
            f"step_id: {step_id}, loss: {loss.item()}, max_grad_norm: {max_grad_norm}"
        )
        for key, value in info.items():
            log_string += f", {key}: {value.item()}"

        # find the min, max and mean of depths
        gt_sample = sample.with_annot_mode("gt")
        gt_depths = gt_sample.depths
        min_depth, max_depth, mean_depth = (
            gt_depths.min(),
            gt_depths.max(),
            gt_depths.mean(),
        )
        log_string += (
            f", for gt depth, min: {min_depth}, max: {max_depth}, mean: {mean_depth}"
        )

        # find the min, max and mean of estimated depths
        est_sample = sample.with_annot_mode("est")
        est_depths = est_sample.depths
        min_depth, max_depth, mean_depth = (
            est_depths.min(),
            est_depths.max(),
            est_depths.mean(),
        )
        log_string += (
            f", for est depth, min: {min_depth}, max: {max_depth}, mean: {mean_depth}"
        )

        # log things for the gt trajs
        trajs = sample.gt_trajs_3d
        valids = sample.valids
        visibs = sample.visibs
        # dynamic = sample.dense_dynamic.flatten()

        trajs = trajs[valids]
        visibs = visibs[valids]
        # dynamic = dynamic[sample.dense_valids[:, 0].flatten()]

        log_string += f", for gt:: max_traj: {trajs.max(0)[0].cpu()}, min_traj: {trajs.min(0)[0].cpu()}, mean_trajs: {trajs.mean(0).cpu()}"
        log_string += f", percent valids: {valids.float().mean().cpu()}, percent visibs: {visibs.float().mean().cpu()}"

        # log thinsg for the pred trajs
        # pred_trajs = pred_trajs = pred.coords.flatten(0, 4).coords.flatten(0, 4)
        pred_trajs = pred.query_coords
        pred_trajs = pred_trajs[valids]
        log_string += f", for pred:: max_traj: {pred_trajs.max(0)[0].cpu()}, min_traj: {pred_trajs.min(0)[0].cpu()}, mean_trajs: {pred_trajs.mean(0).cpu()}"

        # sample, preds = prepare_dense_tracks_for_eval(sample, pred)
        pred.coords = pred.query_coords
        pred.visibs = pred.query_visibs
        samples_cpu = sample.to("cpu")
        samples_cpu = samples_cpu.with_annot_mode("gt")
        preds_cpu = pred.to("cpu")

        if preds_cpu.coords.shape[2] > 5000:
            subsample_idx = torch.randperm(preds_cpu.coords.shape[2])[:5000]
            preds_cpu.coords = preds_cpu.coords[:, :, subsample_idx]
            preds_cpu.visibs = preds_cpu.visibs[:, :, subsample_idx]
            # preds_cpu.is_dynamic = preds_cpu.is_dynamic[:, subsample_idx]
            samples_cpu.est_trajs_3d = samples_cpu.est_trajs_3d[:, :, subsample_idx]
            samples_cpu.gt_trajs_3d = samples_cpu.gt_trajs_3d[:, :, subsample_idx]
            samples_cpu.visibs = samples_cpu.visibs[:, :, subsample_idx]
            samples_cpu.valids = samples_cpu.valids[:, :, subsample_idx]
            # samples_cpu.is_dynamic = samples_cpu.is_dynamic[:, subsample_idx]

        generate_videos(
            sample=samples_cpu,
            preds=preds_cpu,
            output_dir=self.output_dir,
            step_id=step_id,
            batch_idx=0,
            dataset_name="debug_bad_norms",
            log_wandb=self._should_log_wandb(),
            writer=self.tb_writer if not self._should_log_wandb() else None,
            scale=sample.pd_to_gt_scale.cpu(),
        )

        # log on wandb instead
        if self._should_log_wandb():
            wandb.log(
                {
                    "debug_bad_norms/loss": loss.item(),
                    "debug_bad_norms/max_grad_norm": max_grad_norm,
                    **{f"debug_bad_norms/{key}": value.item() for key, value in info.items()},
                },
                step=step_id,
            )
        else:
            self.tb_writer.add_text("debug_bad_norms/text_info", log_string, step_id)

        logger.info(f"finished logging bad sample")

    @profile
    def main(self, collate_fn=default_collate):
        """Run main training/testing pipeline."""
        set_seed(self.cfg.seed)

        self.max_log_sample = 0

        # Get model
        model = self.get_model()

        print(summary(model))

        # Get accelerator
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=True, static_graph=False
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_steps,
            mixed_precision=self.cfg.train.mixed_precision,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
        )

        logger.info(f"Mixed precision: {self.accelerator.mixed_precision}")

        # Get criterion
        criterion = self.get_criterion()

        # Get optimizer
        optimizer = self.get_optimizer(model)

        # Get scheduler
        scheduler = self.get_scheduler(optimizer)

        # Check for a checkpoint
        start_iter, best_ref_loss = 0, None

        experiment_id = self.cfg.get("experiment_id", None)
        output_root = self.cfg.get("output_dir", "outputs/tracking_logs")
        if experiment_id is None:
            experiment_id = f"{self.cfg.experiment_name}_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}"
            self.output_dir = f"{output_root}/auto_generated/{experiment_id}"
        else:
            self.output_dir = f"{output_root}/id_specified/{experiment_id}"

        if get_rank() == 0:
            if self.cfg.train.checkpoint is None:
                self.output_dir = Path(self.output_dir)
                if (self.output_dir / "last.pth").is_file():
                    logger.info(
                        f"Automatically resumed from {self.output_dir / 'last.pth'}"
                    )
                    self.cfg.train.checkpoint = str(self.output_dir / "last.pth")
                else:
                    logger.info(f"Starting new experiment {experiment_id}")
                    self.output_dir.mkdir(parents=True, exist_ok=True)

        _output_dir = [self.output_dir]
        _output_dir = accelerate.utils.broadcast_object_list(
            _output_dir, from_process=0
        )
        self.output_dir = _output_dir[0]

        self.accelerator.wait_for_everyone()

        if self._should_log_wandb():
            wandb.init(
                project=self.cfg.wandb.project,
                # entity=self.cfg.wandb.entity,
                name=experiment_id,
                resume="allow",
                id=experiment_id,
                sync_tensorboard=True,
                # dir=str(self.output_dir),
                config=OmegaConf.to_container(self.cfg),  # type: ignore
            )
            if self.cfg.wandb.log_code:
                logger.warning(
                    f"All code in {os.path.dirname(os.path.dirname(__file__))} will be logged to wandb. If this is not intended, set wandb.log_code to False."
                )
                wandb.run.log_code(  # type: ignore
                    str(os.path.dirname(os.path.dirname(__file__))),
                    include_fn=lambda x: x.endswith((".py", ".sh", ".yaml", ".json")),
                    exclude_fn=lambda path, root: os.path.relpath(
                        path, root
                    ).startswith(("outputs", ".git")),
                )
            wandb.define_metric("*", step_metric="step")

        if self._should_log_tb():
            from torch.utils.tensorboard import SummaryWriter

            self.tb_writer = SummaryWriter(
                log_dir=str(self.output_dir), purge_step=start_iter
            )

        # grad_file = open(self.output_dir / f"grad_norm_{get_rank()}.txt", "a")

        # self.output_stream = open(
        #     os.path.join(
        #         self.output_dir,
        #         f"output_localrank{os.environ.get('LOCAL_RANK', 0)}.txt",
        #     ),
        #     "a",
        # )

        checkpoint_list = [self.cfg.train.checkpoint]
        checkpoint_list = accelerate.utils.broadcast_object_list(
            checkpoint_list, from_process=0
        )
        self.cfg.train.checkpoint = checkpoint_list[0]
        if self.cfg.train.checkpoint:
            start_iter, best_ref_loss = self.load_checkpoint(
                model, optimizer, scheduler
            )
        elif self.cfg.train.get("pretrained", None):
            model_dict = torch.load(self.cfg.train.pretrained, map_location="cpu", weights_only=False)
            if self.cfg.train.get("_force_load_optimizer_state", False):
                raise Exception("Force loading optimizer state is deprecated.")
                logger.info("Force loading optimizer state")
                optimizer.load_state_dict(model_dict["optimizer"])

            # model.load_state_dict(model_dict["weight"], strict=True)
            # Allow shape mismatches: load only params that match; skip the rest
            ckpt = model_dict["weight"]
            model_sd = model.state_dict()
            load_sd = {
                k: v for k, v in ckpt.items()
                if k in model_sd and model_sd[k].shape == v.shape
            }
            if len(load_sd) < len(ckpt):
                skipped = set(ckpt.keys()) - set(load_sd.keys())
                logger.info(
                    "Pretrained load: skipped %d keys (missing or shape mismatch): %s",
                    len(skipped),
                    sorted(skipped)[:10] if len(skipped) > 10 else sorted(skipped),
                )
            model.load_state_dict(load_sd, strict=False)  # type: ignore

          
        # Wrap objects with accelerator
        if self.cfg.train.multi_backward:
            model = self.accelerator.prepare_model(model, evaluation_mode=True)  # type: ignore
            self.broadcast_model(model)
            for module in model.modules():  # type: ignore
                if isinstance(module, torch.nn.SyncBatchNorm):
                    raise ValueError(
                        "SyncBatchNorm is not supported for multi backward mode"
                    )
        else:
            model = self.accelerator.prepare_model(model, evaluation_mode=False)  # type: ignore

        self.train_dataset, self.test_datasets = self.get_datasets(start_iter=start_iter)

        # Get loaders
        train_loader, test_loaders = self.get_loaders(collate_fn)
        
        
        if train_loader is not None:
            optimizer, train_loader, scheduler = self.accelerator.prepare(
                optimizer, train_loader, scheduler, device_placement=[True, True, True]
            )
            self.optimizer = optimizer
                
            trained_epoch = (
                start_iter * self.gradient_accumulation_steps // len(train_loader)
            )
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(trained_epoch)
            else:
                train_loader.set_epoch(trained_epoch)
        
        test_loaders = {
            key: self.accelerator.prepare(test_loaders[key]) for key in test_loaders
        }

        if self.cfg.eval_only:
            model.eval()
            evaluate_nsteps = self.evaluate_nsteps_alltrack3r
            eval_table: Dict[str, Dict[str, float]] = {}
            eval_table_fixed: Dict[str, Dict[str, float]] = {}
            for key in test_loaders:
                out = evaluate_nsteps(
                    model, criterion, test_loaders[key], start_iter, dataset_name=key
                )
                if isinstance(out, tuple) and len(out) == 3:
                    _, col, col_fixed = out
                    if col:
                        eval_table[key] = col
                    if col_fixed:
                        eval_table_fixed[key] = col_fixed
                elif isinstance(out, tuple) and len(out) == 2:
                    _, col = out
                    if col:
                        eval_table[key] = col
            log_multi_dataset_eval_table(start_iter, eval_table)
            if eval_table_fixed:
                logger.info(
                    "Eval metrics (fixed metric thresholds; print-only) - full table:"
                )
                log_multi_dataset_eval_table(start_iter, eval_table_fixed)
            dump_eval_tables_json(
                point_thresh=eval_table, fixed_thresh=eval_table_fixed, step_id=start_iter
            )
            return model

        
        trained_batch_in_epoch = (
            start_iter * self.gradient_accumulation_steps % len(train_loader)
        )
        
        skipped_train_loader = self.accelerator.skip_first_batches(
            train_loader, num_batches=trained_batch_in_epoch
        )
        train_iter = iter(skipped_train_loader)

        # Training loop
        model.train()

        if hasattr(self.optimizer.optimizer, "train"):
            self.optimizer.optimizer.train()

        # SLURM preemption: save checkpoint when SIGUSR1 is received
        # (SLURM sends this before killing the job if --signal=USR1@<sec> is set)
        self._preemption_requested = False

        def _preemption_handler(signum, frame):
            logger.info(
                "Received SIGUSR1 — SLURM preemption imminent. "
                "Will save checkpoint after current step."
            )
            self._preemption_requested = True

        signal.signal(signal.SIGUSR1, _preemption_handler)

        if get_rank() == 0:
            pbar = tqdm(total=self.cfg.train.train_steps)
            pbar.update(start_iter)

        iter_times_freq = int(self.cfg.wandb.get("log_iter_times_freq", 10))
        track_iter_times = get_rank() == 0 and self._should_log() and iter_times_freq > 0
        
        for step_id in range(start_iter, self.cfg.train.train_steps):
            sum_data_s = sum_fwd_s = sum_bwd_s = 0.0
            for _ in range(self.gradient_accumulation_steps):
                t_batch_wait0 = time.perf_counter()
                # custom_viz_track_ = (
                #     VizTracer(output_file=f"new_profile.json")
                #     if self.cfg.train.get("use_viztracer", False) and step_id > 10
                #     else contextlib.nullcontext()
                # )
                
                try:
                    # with custom_viz_track_:
                    sample = next(train_iter)
                    # eval_iter = iter(test_loaders[list(test_loaders.keys())[0]])
                    # eval_sample = next(eval_iter)
                    # st()

                except StopIteration:
                    print("in stop iteration")
                    assert step_id % len(train_loader) == 0
                    train_loader.set_epoch(step_id // len(train_loader))
                    train_iter = iter(train_loader)
                    sample = next(train_iter)

                t_batch_wait1 = time.perf_counter()
                data_dt = t_batch_wait1 - t_batch_wait0
                # print(f"Data loading time: {data_dt}")
                
                if self.cfg.train.get("do_exploratory_visualization", False):
                    self._exploratory_visualization(sample, step_id)
                    if step_id >= self.cfg.train.get("exploratory_vis_samples", 10):
                        logger.info("Finished exploratory visualization.")
                        return
                    continue
                
                if self.cfg.train.get("profile_dataloader_only", False):
                    print(sample.dataset_name)
                    dataset_name = sample.dataset_name
                    continue
                    print("profiling dataloader")
                    logger.info(f"Loading time: {data_dt}")
                    
                    print("seq name: ", sample.seq_name[0].split('/')[-1])
                    from torch.nn import functional as F
                    rgbs = F.interpolate(
                        sample.rgbs[0],
                        size=sample.gt_depths_downsampled.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )[None]
                    from ..utils.vis_utils import visualize_rgbd_pc
                    try:
                        gt_pcs = visualize_rgbd_pc(
                            rgbs=rgbs.permute(0, 1, 3, 4, 2) * 255,
                            depths=sample.gt_depths_downsampled,
                            intrsinsics=sample.gt_intrinsics_downsampled,
                            extrinsics=sample.gt_extrinsics,
                            voxel_size=0.01,
                            legend="gt_rgbd_0.01",
                            valid_mask=sample.dense_norm_mask.squeeze(-1) if sample.dense_norm_mask[0] is not None else None
                        )
                    except Exception as e:
                        st()
                    print(f"number of points with voxel size 0.01: {gt_pcs[0].shape}")
                    continue
                
                    pred_pcs = visualize_rgbd_pc(
                        rgbs=rgbs.permute(0, 1, 3, 4, 2) * 255,
                        depths=sample.est_depths_downsampled,
                        intrsinsics=sample.est_intrinsics_downsampled,
                        extrinsics=sample.est_extrinsics,
                        voxel_size=0.01,
                        legend="est_rgbd_0.01",
                        valid_mask=sample.dense_norm_mask.squeeze(-1) if sample.dense_norm_mask[0] is not None else None
                    )
                    print(f"number of points with voxel size 0.01: {pred_pcs[0].shape}")
                    # gt_pcs = visualize_rgbd_pc(
                    #     rgbs=rgbs.permute(0, 1, 3, 4, 2) * 255,
                    #     depths=sample.gt_depths_downsampled,
                    #     intrsinsics=sample.gt_intrinsics_downsampled,
                    #     extrinsics=sample.gt_extrinsics,
                    #     voxel_size=0.02,
                    #     legend="gt_rgbd_0.02",
                    #     valid_mask=sample.dense_norm_mask.squeeze(-1)
                    # )
                    # print(f"number of points with voxel size 0.02: {gt_pcs[0].shape}")
                    # gt_pcs = visualize_rgbd_pc(
                    #     rgbs=rgbs.permute(0, 1, 3, 4, 2) * 255,
                    #     depths=sample.gt_depths_downsampled,
                    #     intrsinsics=sample.gt_intrinsics_downsampled,
                    #     extrinsics=sample.gt_extrinsics,
                    #     voxel_size=0.05,
                    #     legend="gt_rgbd_0.05",
                    #     valid_mask=sample.dense_norm_mask.squeeze(-1)
                    # )
                    # print(f"number of points with voxel size 0.05: {gt_pcs[0].shape}")
                    
                    # gt_pcs = visualize_rgbd_pc(
                    #     rgbs=sample.rgbs.permute(0, 1, 3, 4, 2)[:, 4:5] * 255,
                    #     depths=sample.gt_depths[:, 4:5],
                    #     intrsinsics=sample.gt_intrinsics[:, 4:5],
                    #     extrinsics=sample.gt_extrinsics[:, 4:5],
                    #     voxel_size=0.05,
                    #     legend="gt_rgbd_0.05",
                    #     valid_mask=sample.dense_norm_mask[:, 4:5]
                    # )
                    
                    # gt_pcs = visualize_rgbd_pc(
                    #     rgbs=sample.rgbs.permute(0, 1, 3, 4, 2)[:, 4:5] * 255,
                    #     depths=sample.gt_depths[:, 4:5],
                    #     intrsinsics=sample.gt_intrinsics[:, 4:5],
                    #     extrinsics=sample.gt_extrinsics[:, 4:5],
                    #     voxel_size=0.01,
                    #     legend="gt_rgbd_0.01",
                    #     valid_mask=sample.dense_norm_mask[:, 4:5]
                    # )
                    
                    
                    # gt_pcs = visualize_rgbd_pc(
                    #     rgbs=sample.rgbs.permute(0, 1, 3, 4, 2)[:, 4:5] * 255,
                    #     depths=sample.gt_depths[:, 4:5],
                    #     intrsinsics=sample.gt_intrinsics[:, 4:5],
                    #     extrinsics=sample.gt_extrinsics[:, 4:5],
                    #     voxel_size=0.02,
                    #     legend="gt_rgbd_0.02",
                    #     valid_mask=sample.dense_norm_mask[:, 4:5]
                    # )
                    
                    # pred_pcs = visualize_rgbd_pc(
                    #     rgbs=sample.rgbs.permute(0, 1, 3, 4, 2)[:, 4:5] * 255,
                    #     depths=sample.est_depths[:, 4:5],
                    #     intrsinsics=sample.est_intrinsics[:, 4:5],
                    #     extrinsics=sample.est_extrinsics[:, 4:5],
                    #     voxel_size=self.cfg.model.get("voxel_size", None),
                    #     legend="est_rgbd",
                    #     valid_mask=sample.dense_norm_mask[:, 4:5]
                    # )
                    
                    # # # combined pcs
                    pred_pcs[0] *= sample.pd_to_gt_scale[0].cpu().numpy()
                    combined_pcs = np.concatenate([gt_pcs[0], pred_pcs[0]], axis=0)
                    wandb.log(
                        {
                            "gt_est_rgbd": wandb.Object3D(
                                combined_pcs,
                            )
                        }
                    )
                    
                    # # # combined pc but gt in red, est in blue
                    gt_color = np.array([[255, 0, 0]])
                    est_color = np.array([[0, 0, 255]])
                    gt_pcs_colored = gt_pcs[0]
                    gt_pcs_colored[:, 3:6] = gt_color
                    pred_pcs_colored = pred_pcs[0]
                    pred_pcs_colored[:, 3:6] = est_color
                    combined_pcs_colored = np.concatenate(
                        [gt_pcs_colored, pred_pcs_colored], axis=0
                    )
                    wandb.log(
                        {
                            "gt_est_rgbd_colored": wandb.Object3D(
                                combined_pcs_colored,
                            )
                        }
                    )
                    continue
                    
                    # st()
                    

                    self.num_trajs.append(sample.gt_trajs_3d.shape[2])
                    self.num_dynamic_trajs.append(sample.is_dynamic.sum().item())

                    if step_id == 2000:
                        # st()
                        # draw a histogram of the num trajs and num dynamic trajs
                        import matplotlib.pyplot as plt

                        num_trajs = self.num_trajs
                        num_dynamic_trajs = self.num_dynamic_trajs

                        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

                        # First histogram
                        axes[0].hist(
                            num_trajs, bins="auto", color="skyblue", edgecolor="black"
                        )
                        axes[0].set_title("num_trajs")
                        axes[0].set_xlabel("Count")
                        axes[0].set_ylabel("Frequency")
                        axes[0].grid(True, linestyle="--", alpha=0.5)

                        # Second histogram
                        axes[1].hist(
                            num_dynamic_trajs,
                            bins="auto",
                            color="salmon",
                            edgecolor="black",
                        )
                        axes[1].set_title("num_dynamic_trajs")
                        axes[1].set_xlabel("Count")
                        axes[1].grid(True, linestyle="--", alpha=0.5)

                        plt.tight_layout()

                        plt.savefig("histogram.png")
                        st()
                        exit()

                    continue

                if track_iter_times:
                    sum_data_s += data_dt

                if self.cfg.train.get("profile_memory", False):
                    torch.cuda.memory._record_memory_history(max_entries=500000)

                # start = time.time()
                custom_viz_track = (
                    VizTracer(output_file=f"new_profile.json")
                    if self.cfg.train.get("use_viztracer", False) and step_id > 10
                    else contextlib.nullcontext()
                )

                custom_viz_track = (
                    torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        schedule=torch.profiler.schedule(
                            wait=0, warmup=0, active=5, repeat=1
                        ),
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=True,
                        on_trace_ready=trace_handler,
                    )
                    if self.cfg.train.get("profile_memory", False) and step_id > 5
                    else custom_viz_track
                )

                if self.cfg.train.get("use_torch_profiler", False) and step_id == 15:
                    custom_viz_track = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        record_shapes=True,
                        with_stack=True,
                        on_trace_ready=chrome_trace_handler,
                    )
                with custom_viz_track:
                    if not self.cfg.get("eval_only", False):
                        with self.accelerator.accumulate(model):
                            if track_iter_times:
                                t_fwd0 = time.perf_counter()
                            loss, pred, info = self.train_one_step(
                                model, criterion, step_id, sample
                            )
                            if track_iter_times:
                                sum_fwd_s += time.perf_counter() - t_fwd0

                            assert (
                                not torch.is_autocast_enabled()
                            ), "AMP should only be enabled in the train_one_step"
                            if not self.cfg.train.multi_backward:
                                if track_iter_times:
                                    t_bwd0 = time.perf_counter()
                                self.accelerator.backward(loss)
                                if track_iter_times:
                                    sum_bwd_s += time.perf_counter() - t_bwd0
                            else:
                                assert not loss.requires_grad  # type: ignore

                            loss = loss.detach().clone()

                            # Host-syncing full grad gather only when visualizing high-grad samples.
                            if (
                                self.cfg.train.get("visualize_high_grad_samples", False)
                                and step_id > 25000
                                and self.max_log_sample < 10
                                and self._should_log_tb()
                            ):
                                grads = [
                                    param.grad.detach().flatten()
                                    for param in model.parameters()
                                    if param.grad is not None
                                ]
                                norm = (
                                    torch.cat(grads).norm().item()
                                    if len(grads) > 0
                                    else 0.0
                                )
                                if norm > 2.0:
                                    self.visualize_bad_sample_data(
                                        loss, sample, step_id, pred, info, norm
                                    )
                                    self.max_log_sample += 1

                            if self.accelerator.sync_gradients:
                                if self.cfg.train.multi_backward:
                                    if track_iter_times:
                                        t_bwd0 = time.perf_counter()
                                    self.sync_gradients(model)  # type: ignore
                                    if track_iter_times:
                                        sum_bwd_s += time.perf_counter() - t_bwd0

                                max_grad_norm = (
                                    self.cfg.train.max_grad_norm
                                    if self.cfg.train.max_grad_norm is not None
                                    else float("inf")
                                )
                                # clip every step; only .item()/log every 100 steps
                                # (avoids an extra host sync on non-log steps).
                                grad_norm_t = self.accelerator.clip_grad_norm_(
                                    self.trainable_parameters,
                                    max_grad_norm,
                                )
                                if self._should_log() and (step_id + 1) % 100 == 0:
                                    grad_norm = float(grad_norm_t)
                                    logger.info(f"grad_norm: {grad_norm}")
                                    if self._should_log_tb():
                                        self.tb_writer.add_scalar(
                                            "grad_norm", grad_norm, step_id
                                        )  # type: ignore

                            optimizer.step()
                            if scheduler is not None:
                                scheduler.step()
                            optimizer.zero_grad()

                if self.cfg.train.get("use_viztracer", False) and step_id > 10:
                    print("Finished writing viztracer file for rank")
                    exit()

                if self.cfg.train.get("profile_memory", False) and step_id > 12:
                    torch.cuda.memory._dump_snapshot("memory.pickle")
                    torch.cuda.memory._record_memory_history(enabled=None)
                    print("Finished writing memory profiling")
                    exit()

                # end = time.time()
                # logger.info(f"Training time: {end - start}")

            if track_iter_times and (step_id + 1) % iter_times_freq == 0:
                perf_payload = {
                    "perf/data_time": sum_data_s,
                    "perf/forward_time": sum_fwd_s,
                    "perf/backward_time": sum_bwd_s,
                }
                if self._should_log_wandb():
                    wandb.log(perf_payload, step=step_id)
                if self._should_log_tb():
                    self.tb_writer.add_scalar("perf/data_time", sum_data_s, step_id)
                    self.tb_writer.add_scalar("perf/forward_time", sum_fwd_s, step_id)
                    self.tb_writer.add_scalar("perf/backward_time", sum_bwd_s, step_id)

            if get_rank() == 0:
                pbar.update(1)

            # if (step_id + 1) % 10 == 0:
            #     grad_file.flush()
            
            if (step_id + 1) % self.cfg.train.save_freq == 0:
                self.accelerator.wait_for_everyone()
                if get_rank() == 0:
                    self.save_checkpoint(
                        model, optimizer, scheduler, step_id, None, None
                    )

            if (step_id + 1) % self.cfg.train.get(
                "keep_freq", self.cfg.train.train_steps // 10
            ) == 0:
                self.accelerator.wait_for_everyone()
                if get_rank() == 0:
                    self.save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        step_id,
                        None,
                        None,
                        name=f"checkpoint_{step_id}",
                    )

            if (step_id + 1) % self.cfg.train.eval_freq == 0:
                model.eval()
                if hasattr(self.optimizer.optimizer, "eval"):
                    self.optimizer.optimizer.eval()

                ref_losses = []
                eval_table: Dict[str, Dict[str, float]] = {}
                eval_table_fixed: Dict[str, Dict[str, float]] = {}
                for key in sorted(test_loaders.keys()):
                    logger.info(f"Evaluating on {key}.......")
                    evaluate_nsteps = self.evaluate_nsteps_alltrack3r
                    try:
                        out = evaluate_nsteps(
                            model,
                            criterion,
                            test_loaders[key],
                            step_id,
                            dataset_name=key,
                        )
                    except Exception:
                        logger.exception(
                            "Evaluation failed for dataset %r; continuing with other eval sets.",
                            key,
                        )
                        continue
                    if isinstance(out, tuple) and len(out) == 3:
                        ref_loss, col, col_fixed = out
                        if col:
                            eval_table[key] = col
                        if col_fixed:
                            eval_table_fixed[key] = col_fixed
                    elif isinstance(out, tuple) and len(out) == 2:
                        ref_loss, col = out
                        if col:
                            eval_table[key] = col
                    else:
                        ref_loss = out
                    ref_losses.append(ref_loss)

                log_multi_dataset_eval_table(step_id, eval_table)
                if eval_table_fixed:
                    logger.info(
                        "Eval metrics (fixed metric thresholds; print-only) - full table:"
                    )
                    log_multi_dataset_eval_table(step_id, eval_table_fixed)
                if ref_losses:
                    mean_ref_loss = float(np.mean(ref_losses))
                else:
                    logger.warning(
                        "No eval datasets completed successfully; skipping mean ref loss / best checkpoint update for this eval round."
                    )
                    mean_ref_loss = None

                self.accelerator.wait_for_everyone()
                if get_rank() == 0:
                    best_ref_loss = self.save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        step_id,
                        mean_ref_loss,
                        best_ref_loss,
                    )

                if hasattr(self.optimizer.optimizer, "train"):
                    self.optimizer.optimizer.train()
                model.train()

                if self.cfg.get("eval_only", False):
                    break

            if self._preemption_requested:
                logger.info(
                    f"Preemption: saving checkpoint at step {step_id} before SLURM kills the job..."
                )
                self.accelerator.wait_for_everyone()
                if get_rank() == 0:
                    self.save_checkpoint(
                        model, optimizer, scheduler, step_id, None, None
                    )
                logger.info("Preemption checkpoint saved. Exiting.")
                sys.exit(0)

        return model

    @torch.no_grad()
    def _exploratory_visualization(self, sample, step_id):
        """Visualize multi-view consistency signals for noisy vs GT points."""
        from ..utils.vis_utils import (
            visualize_frame0_reconstruction,
            visualize_multiview_noisy_points,
        )

        seq_name = sample.seq_name[0].split("/")[-1] if isinstance(sample.seq_name, list) else sample.seq_name.split("/")[-1]
        log_step = step_id if self._should_log_wandb() else None

        _, noise_stats = visualize_multiview_noisy_points(sample, step_id=log_step, seq_name=seq_name)
        visualize_frame0_reconstruction(sample, step_id=log_step, seq_name=seq_name)

        if log_step is not None:
            logger.info(
                f"[Vis {step_id}] seq={seq_name}  "
                f"max_noise={noise_stats['max_noise']:.4f}  "
                f"median_noise={noise_stats['median_noise']:.4f}"
            )

    def train_one_step(self, model, criterion, step_id, sample):
        """Run a single training step."""
        pass

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, dataset_name):
        """Run a given number of evaluation steps."""
        return None

    def _diagnose_optimizer_state_dict_mismatch(
        self,
        model: torch.nn.Module,
        optimizer: optim.Optimizer,
        optim_sd: Dict,
        checkpoint_weight_keys: Optional[Set[str]] = None,
    ) -> None:
        """Log optimizer vs checkpoint details when load_state_dict fails (rank 0 only)."""
        if get_rank() != 0:
            return
        id_to_name = {id(p): n for n, p in model.named_parameters()}
        groups = optimizer.param_groups
        saved_groups = optim_sd.get("param_groups")
        if not isinstance(saved_groups, list):
            logger.error(
                "Optimizer checkpoint has no valid param_groups; keys=%s",
                list(optim_sd.keys()),
            )
            return

        def _group_param_names(
            param_list: List[torch.nn.Parameter],
        ) -> List[str]:
            return [id_to_name.get(id(p), f"<unknown_param_id={id(p)}>") for p in param_list]

        def _fmt_names(names: List[str], head: int = 24) -> str:
            if len(names) <= head:
                return str(names)
            return (
                f"{names[: head // 2]!s} ... ({len(names)} names, truncated) ... "
                f"{names[-(head // 2) :]!s}"
            )

        if len(groups) != len(saved_groups):
            logger.error(
                "Optimizer param group count mismatch: current=%d checkpoint=%d",
                len(groups),
                len(saved_groups),
            )
            for i, g in enumerate(groups):
                logger.error(
                    "  current group %d: n_params=%d names=%s",
                    i,
                    len(g["params"]),
                    _fmt_names(_group_param_names(g["params"])),
                )
            for i, sg in enumerate(saved_groups):
                pn = sg.get("param_names")
                logger.error(
                    "  checkpoint group %d: n_params=%d param_names=%s",
                    i,
                    len(sg.get("params", ())),
                    pn if pn is not None else "(not stored in ckpt)",
                )
            return

        for i, (g, sg) in enumerate(zip(groups, saved_groups)):
            n_cur = len(g["params"])
            n_ckpt = len(sg.get("params", ()))
            if n_cur == n_ckpt:
                continue
            cur_names = _group_param_names(g["params"])
            ckpt_names: Optional[List[str]] = sg.get("param_names")
            logger.error(
                "Optimizer param group %d size mismatch: current_n=%d checkpoint_n=%d",
                i,
                n_cur,
                n_ckpt,
            )
            if ckpt_names is not None:
                logger.error(
                    "  group %d name set diff only_in_current=%s only_in_checkpoint=%s",
                    i,
                    sorted(set(cur_names) - set(ckpt_names)),
                    sorted(set(ckpt_names) - set(cur_names)),
                )
                logger.error(
                    "  checkpoint group %d param_names: %s",
                    i,
                    _fmt_names(list(ckpt_names)),
                )
            elif checkpoint_weight_keys is not None:
                only_in_current_vs_weights = sorted(
                    n for n in cur_names if n not in checkpoint_weight_keys
                )
                if only_in_current_vs_weights:
                    logger.error(
                        "  group %d: current optimizer params missing from checkpoint "
                        "weight keys (likely new since ckpt): %s",
                        i,
                        only_in_current_vs_weights,
                    )
                else:
                    logger.error(
                        "  group %d: all names exist in checkpoint weight keys; mismatch "
                        "may be no_decay vs weight_decay regrouping or a different ckpt.",
                        i,
                    )

            logger.error(
                "  current group %d parameter names: %s",
                i,
                _fmt_names(cur_names),
            )

        flat_cur: List[str] = []
        for g in groups:
            flat_cur.extend(_group_param_names(g["params"]))
        flat_ckpt: List[str] = []
        has_ckpt_names = True
        for sg in saved_groups:
            pn = sg.get("param_names")
            if pn is None:
                has_ckpt_names = False
                break
            flat_ckpt.extend(pn)
        if has_ckpt_names and flat_ckpt:
            cur_set, ckpt_set = set(flat_cur), set(flat_ckpt)
            if cur_set != ckpt_set:
                logger.error(
                    "Optimizer parameter name set mismatch (union size %d): "
                    "only_in_current=%s only_in_checkpoint=%s",
                    len(cur_set | ckpt_set),
                    sorted(cur_set - ckpt_set),
                    sorted(ckpt_set - cur_set),
                )
            if len(flat_cur) == len(flat_ckpt):
                for j, (a, b) in enumerate(zip(flat_cur, flat_ckpt)):
                    if a != b:
                        logger.error(
                            "First flat-order name mismatch at index %d: current=%r checkpoint=%r",
                            j,
                            a,
                            b,
                        )
                        break
            else:
                logger.error(
                    "Flat optimizer param count: current=%d checkpoint_param_names=%d",
                    len(flat_cur),
                    len(flat_ckpt),
                )

    def load_checkpoint(self, model, optimizer, scheduler):
        """Load from checkpoint."""
        if get_rank() == 0:
            if Path(self.cfg.train.checkpoint).with_suffix(".pth_old").is_file():
                if Path(self.cfg.train.checkpoint).is_file():
                    os.unlink(self.cfg.train.checkpoint)
                os.rename(
                    Path(self.cfg.train.checkpoint).with_suffix(".pth_old"),
                    self.cfg.train.checkpoint,
                )

        self.accelerator.wait_for_everyone()

        assert Path(
            self.cfg.train.checkpoint
        ).is_file(), f"Checkpoint not found: {self.cfg.train.checkpoint}"

        logger.info("=> loading checkpoint '{}'".format(self.cfg.train.checkpoint))

        with fsspec.open(self.cfg.train.checkpoint, "rb") as f:
            model_dict = torch.load(f, map_location="cpu", weights_only=False)
        model.load_state_dict(model_dict["weight"], strict=False)
        resume_load_optimizer = self.cfg.train.get("resume_load_optimizer", True)
        # Only a *live* scheduler can consume ckpt scheduler state. If this run
        # uses train.scheduler_name=constant (scheduler is None), honor train.lr
        # even when the checkpoint was saved with a warmup/onecycle scheduler.
        restore_scheduler = (
            model_dict.get("scheduler") is not None and scheduler is not None
        )
        if "optimizer" in model_dict and resume_load_optimizer:
            try:
                optimizer.load_state_dict(model_dict["optimizer"])
            except ValueError as exc:
                if "parameter group" in str(exc) or "parameter groups" in str(exc):
                    self._diagnose_optimizer_state_dict_mismatch(
                        model,
                        optimizer,
                        model_dict["optimizer"],
                        checkpoint_weight_keys=(
                            set(model_dict["weight"].keys())
                            if isinstance(model_dict.get("weight"), dict)
                            else None
                        ),
                    )
                raise

            if restore_scheduler:
                logger.warning(
                    "Resuming scheduler from checkpoint; keeping optimizer LRs "
                    "from the checkpoint (train.lr=%s is ignored).",
                    self.cfg.train.lr,
                )
            else:
                ckpt_lrs = [group.get("lr") for group in optimizer.param_groups]
                self._set_optimizer_lrs(optimizer, self.cfg.train.lr)
                logger.info(
                    "Restored optimizer moments without a live scheduler. "
                    "Overwriting param group LRs %s -> train.lr=%s.",
                    ckpt_lrs,
                    self.cfg.train.lr,
                )
        elif "optimizer" in model_dict and not resume_load_optimizer:
            logger.info(
                "Skipping optimizer checkpoint restore (train.resume_load_optimizer=false); "
                "using freshly initialized optimizer. Setting all param group LRs to train.lr=%s.",
                self.cfg.train.lr,
            )
            self._set_optimizer_lrs(optimizer, self.cfg.train.lr)

        if restore_scheduler:
            scheduler.load_state_dict(model_dict["scheduler"])

        start_iter = model_dict.get("iter", 0)
        best_ref_loss = model_dict.get("best_ref_loss", None)

        logger.info(
            "=> loaded successfully '{}' (step {})".format(
                self.cfg.train.checkpoint, model_dict.get("iter", 0)
            )
        )
        del model_dict
        torch.cuda.empty_cache()
        return start_iter, best_ref_loss

    def _save_with_backup(self, data, path):
        path = Path(path)
        backup_path = path.with_suffix(".pth_old")
        if path.is_file():
            os.rename(path, backup_path)
        torch.save(data, path)
        if backup_path.is_file():
            os.unlink(backup_path)

    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        step_id,
        new_ref_loss,
        best_ref_loss,
        name=None,
    ):
        """Save checkpoint if requested."""
        assert get_rank() == 0, "Save checkpoint should only be called on rank 0!"
        logger.info("Saving checkpoint...")

        is_training = model.training

        if is_training:
            model.eval()
            if hasattr(optimizer.optimizer, "eval"):
                optimizer.optimizer.eval()
        if hasattr(optimizer.optimizer, "eval"):
            assert not optimizer.optimizer.param_groups[0][
                "train_mode"
            ], "Optimizer is not in eval mode"

        unwrapped_model = self.accelerator.unwrap_model(model)
        if name is None:
            if new_ref_loss is not None and (
                best_ref_loss is None or new_ref_loss <= best_ref_loss
            ):
                best_ref_loss = new_ref_loss
                self._save_with_backup(
                    {
                        "cfg": self.cfg,
                        "weight": unwrapped_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()
                        if scheduler is not None
                        else None,
                        "iter": step_id + 1,
                        "best_ref_loss": best_ref_loss,
                    },
                    f"{str(self.output_dir)}/best.pth",
                )
            self._save_with_backup(
                {
                    "cfg": self.cfg,
                    "weight": unwrapped_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                    if scheduler is not None
                    else None,
                    "iter": step_id + 1,
                    "best_ref_loss": best_ref_loss,
                },
                f"{str(self.output_dir)}/last.pth",
            )
        else:
            assert new_ref_loss is None and best_ref_loss is None
            self._save_with_backup(
                {
                    "cfg": self.cfg,
                    "weight": unwrapped_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                    if scheduler is not None
                    else None,
                    "iter": step_id + 1,
                    "best_ref_loss": best_ref_loss,
                },
                f"{str(self.output_dir)}/{name}.pth",
            )

        if is_training:
            model.train()
            if hasattr(optimizer.optimizer, "train"):
                optimizer.optimizer.train()
        return best_ref_loss


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
