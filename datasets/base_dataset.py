# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from re import S
import time
import typing
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict, Union
import ipdb

import av
import cv2
import numpy as np
import tap
import torch
from einops import rearrange, repeat
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..utils import dataset_paths as _dataset_paths  # noqa: F401  (registers ${data_path} resolver)
from ..utils.common_utils import setup_logger
from ..utils.rerun_visualizer import (
    destroy,
    log_trajectory,
    log_video,
    save_recording,
    setup_visualizer,
)
from .datatypes import RawSliceData, SliceData
from .providers.base_provider import BaseDataProvider
from .utils.crop_utils import CropArgs, get_crop_args
from .utils.random_utils import RandomMapping, WeightedSampler
from .utils.geometry import normalize_scene_geometry
from .dense_traj_gt import _unproject_dense_geometry

CONFIG_DIR = Path(__file__).parent.parent / "configs" / "dataset"
logger = logging.getLogger(__name__)
st = ipdb.set_trace


class BaseDataset(Dataset):
    def set_epoch(self, epoch: int) -> None:
        pass

    @staticmethod
    def safe_inverse(a):
        """Invert a batch of 4x4 homogeneous transformation matrices.
        a: numpy array of shape (B, 4, 4)
        """
        B, _, _ = a.shape
        inv = a.copy()
        r = a[:, :3, :3]
        r_transpose = np.transpose(r, (0, 2, 1))
        inv[:, :3, :3] = r_transpose
        inv[:, :3, 3:4] = -np.matmul(r_transpose, a[:, :3, 3:4])
        return inv

    def _apply_homo_transform(
        self, coords: np.ndarray, transform: np.ndarray, homo_input: bool = False
    ):
        if homo_input:
            coords_homo = coords
        else:
            coords_homo = np.concatenate(
                [coords, np.ones_like(coords[..., :1])], axis=-1
            )
        coords_transformed_homo = np.einsum("...ij,...j->...i", transform, coords_homo)
        coords_transformed = (
            coords_transformed_homo[..., :-1] / (coords_transformed_homo[..., -1:] + 1e-8)
        )
        return coords_transformed

    def _align_to_cam0(self, traj_3d, extrinsics, query_point=None, dense_trajs_3d=None):
        """Transform trajectories and extrinsics to camera 0 coordinate system.

        Returns:
            (traj_3d, extrinsics, query_point) all aligned to cam0 frame.
            query_point is None if input was None.
        """
        cam0_ext = extrinsics[0:1].copy()
        traj_3d = self._apply_homo_transform(traj_3d, cam0_ext)
        extrinsics = np.einsum(
            "bij,bjk->bik", extrinsics, self.safe_inverse(cam0_ext)
        )
        if query_point is not None:
            query_point = query_point.copy()
            query_point[..., 1:] = self._apply_homo_transform(
                query_point[..., 1:], cam0_ext[0]
            )
        if dense_trajs_3d is not None:
            dense_trajs_3d = self._apply_homo_transform(dense_trajs_3d, cam0_ext)
            return traj_3d, extrinsics, query_point, dense_trajs_3d
        return traj_3d, extrinsics, query_point

    def _compute_dense_trajs_and_unproject(
        self,
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
        time_steps=None,
        dynamic_filter_threshold=None,
        valids=None,
        window_size=None,
    ):
        from .dense_traj_gt import compute_dense_trajs_and_unproject

        return compute_dense_trajs_and_unproject(
            gt_traj_2d,
            gt_traj_3d,
            visibs,
            is_dynamic,
            H,
            W,
            scale_factor,
            compute_tmaps,
            gt_depths=gt_depths,
            gt_intrinsics=gt_intrinsics,
            gt_extrinsics=gt_extrinsics,
            gt_depths_downsampled=gt_depths_downsampled,
            gt_intrinsics_downsampled=gt_intrinsics_downsampled,
            subsamle_geometry_points=subsamle_geometry_points,
            make_all_dense_static=make_all_dense_static,
            lift_full_geometry=lift_full_geometry,
            time_steps=time_steps,
            dynamic_filter_threshold=dynamic_filter_threshold,
            valids=valids,
            window_size=window_size,
        )

    def _build_dense_sample_gt(
        self,
        dense_src_idx_map,
        gt_traj_3d,
        visibs,
        valids,
        gt_trajs_depths,
        is_dynamic,
        gt_depths_downsampled,
        gt_intrinsics_downsampled,
        gt_extrinsics,
        norm_mask_downsampled,
        num_points,
        rng,
        valid_depth_range=(0.1, 20.0),
    ):
        """Select ~num_points dense grid cells and build grid-anchored full GT tracks.

        A dense cell (src_frame, y, x) is anchored to the grid base geometry and
        carries the relative motion of the sparse track that won the scatter for
        that cell:
            track[t] = dense_geom[src_frame, y, x] + (traj_3d[t, n] - traj_3d[src, n])
        This is grid-level (not subpixel) and O(M*T); no large tensor is built.

        All position inputs must already be in the normalized frame so the output
        tracks are consistent with `gt_traj_3d`.

        Returns (or None if no valid cell / num_points <= 0):
            dense_sample_points:   (M, 3) int64 [src_frame, y, x] at grid resolution
            dense_sample_trajs_3d: (T, M, 3) float32
            dense_sample_visibs:   (T, M) bool
            dense_sample_valids:   (T, M) bool
            dense_sample_dynamic:  (M,) bool
        """
        if num_points is None or num_points <= 0 or dense_src_idx_map is None:
            return None

        S, H2, W2 = dense_src_idx_map.shape
        pool = dense_src_idx_map >= 0
        if norm_mask_downsampled is not None:
            pool = pool & norm_mask_downsampled.astype(bool)
        flat_pool = np.flatnonzero(pool.reshape(-1))
        if flat_pool.size == 0:
            return None

        # Always return exactly num_points so the batch collate can stack a fixed
        # M across samples. Sample without replacement when the pool is large
        # enough, otherwise pad by sampling with replacement (all indices stay
        # valid cells; duplicates are harmless for the loss).
        m = int(num_points)
        replace = flat_pool.size < m
        chosen = rng.choice(flat_pool, size=m, replace=replace)
        src_frame = (chosen // (H2 * W2)).astype(np.int64)
        rem = chosen % (H2 * W2)
        y = (rem // W2).astype(np.int64)
        x = (rem % W2).astype(np.int64)

        n_win = dense_src_idx_map[src_frame, y, x].astype(np.int64)  # (M,)

        # Normalized grid base geometry at the (source_frame, y, x) cell.
        dense_geom = _unproject_dense_geometry(
            gt_depths_downsampled, gt_intrinsics_downsampled, gt_extrinsics
        )  # (T, H2, W2, 3)
        base = dense_geom[src_frame, y, x, :]  # (M, 3)

        traj_at_src = gt_traj_3d[src_frame, n_win, :]  # (M, 3)
        rel = gt_traj_3d[:, n_win, :] - traj_at_src[None]  # (T, M, 3)
        dense_sample_trajs_3d = (base[None] + rel).astype(np.float32)  # (T, M, 3)

        dense_sample_visibs = visibs[:, n_win].astype(bool)  # (T, M)
        min_depth, max_depth = map(float, valid_depth_range)
        traj_depth_ok = (gt_trajs_depths < max_depth) & (
            gt_trajs_depths > min_depth
        )  # (T, N)
        dense_sample_valids = (valids[:, n_win] & traj_depth_ok[:, n_win]).astype(bool)
        if is_dynamic is not None:
            dense_sample_dynamic = np.asarray(is_dynamic)[n_win].astype(bool)  # (M,)
        else:
            dense_sample_dynamic = np.zeros((m,), dtype=bool)

        dense_sample_points = np.stack([src_frame, y, x], axis=-1).astype(np.int64)  # (M, 3)
        return (
            dense_sample_points,
            dense_sample_trajs_3d,
            dense_sample_visibs,
            dense_sample_valids,
            dense_sample_dynamic,
        )

    def _normalize_scene(
        self,
        extrinsics,
        depths,
        traj_3d,
        depths_downsampled,
        intrinsics_downsampled,
        norm_mask_downsampled,
        norm_mode="median_dis",
        query_point=None,
        dense_trajs_3d=None,
    ):
        """Normalize scene geometry by a depth-derived scale factor.

        norm_factor is computed from downsampled depths, intrinsics, and mask.
        Full-resolution depths and trajectories are scaled by the same factor.

        Modifies query_point in-place.

        Returns:
            (norm_factor, depths, depths_downsampled, traj_3d, extrinsics, dense_trajs_3d)
        """
        norm_factor = normalize_scene_geometry(
            intrinsics_downsampled,
            extrinsics,
            depths_downsampled,
            norm_mode=norm_mode,
            norm_mask=norm_mask_downsampled,
            subsample_factor=2,
        )
        if np.isnan(norm_factor):
            print("norm_factor is nan")
            norm_factor = normalize_scene_geometry(
                intrinsics_downsampled,
                    extrinsics,
                    depths_downsampled,
                    norm_mode=norm_mode,
                    norm_mask=norm_mask_downsampled,
                    subsample_factor=1,
                )
            if np.isnan(norm_factor):
                print("norm_factor is still nan")
                norm_factor = 1.0
        depths = depths / norm_factor
        depths_downsampled = depths_downsampled / norm_factor
        traj_3d = traj_3d / norm_factor
        extrinsics = extrinsics.copy()
        extrinsics[..., :3, 3] /= norm_factor
        if dense_trajs_3d is not None:
            dense_trajs_3d = dense_trajs_3d / norm_factor
        if query_point is not None:
            query_point[:, 1:] /= norm_factor
        return norm_factor, depths, depths_downsampled, traj_3d, extrinsics, dense_trajs_3d

    @classmethod
    def from_config(cls, cfg: Union[DictConfig, str], **kwargs) -> "BaseDataset":
        if isinstance(cfg, str):
            cfg = cls.load_config(cfg)
        assert all(
            key not in kwargs for key in cfg.keys()
        ), "Overwriting dataset config is not allowed"
        kwargs.update(cfg)

        from .eval_dataset import EvalDataset
        from .train_dataset import TrainDataset

        if cfg.type == "train":
            dataset_cls = TrainDataset
        elif cfg.type == "eval":
            dataset_cls = EvalDataset
        else:
            raise ValueError(f"Unknown dataset type: {cfg.type}")
        kwargs.pop("type")
        return dataset_cls(**kwargs)  # type: ignore

    @classmethod
    def load_config(cls, cfg: str) -> DictConfig:
        with initialize_config_dir(
            version_base=None, config_dir=str(CONFIG_DIR.absolute())
        ):
            cfg_dirs = cfg.split("/")
            cfg = compose(config_name=cfg)  # type: ignore
            for key in cfg_dirs[:-1]:
                cfg = cfg[key]  # type: ignore
        return typing.cast(DictConfig, cfg)


class Arguments(tap.Tap):
    task: str = "visualize"
    config: str = "dr_train"
    idx: Optional[int] = 0
    num_workers: int = 8
    batch_size: int = 1
    annot_mode: Literal["gt", "est"] = "gt"
    override_anno: Optional[str] = None


if __name__ == "__main__":
    setup_logger()
    args = Arguments().parse_args()
    dataset = BaseDataset.from_config(args.config)
    if args.task == "speed_test":
        dataset.set_epoch(0)

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=SliceData.collate,
        )

        start_time = time.time()
        num_batches = 0
        pbar = tqdm(dataloader, desc="Processing batches", unit="batch")
        for _ in pbar:
            num_batches += 1
            avg_time_per_batch = (time.time() - start_time) / num_batches
            pbar.set_postfix(avg_time=f"{avg_time_per_batch:.4f} seconds")
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_batch = total_time / num_batches

        print(f"Average time per batch: {avg_time_per_batch:.4f} seconds")
        print(f"Batches processed per second: {1/avg_time_per_batch:.2f}")
    elif args.task == "visualize":
        dataset.set_epoch(0)
        setup_visualizer(serve=False)

        assert args.idx is not None, "An index must be provided"
        item = dataset[args.idx].with_annot_mode(args.annot_mode)
        # item = item.time_slice(0, 12)

        log_video(
            entity_name=f"video_{args.idx}",
            rgb=item.rgbs,
            intrinsics=item.intrinsics,
            extrinsics=item.extrinsics,
            depth=item.depths,
            # uv_space=True,
        )

        log_trajectory(
            entity_name=f"video_{args.idx}",
            track_name="traj",
            intrinsics=item.intrinsics,
            extrinsics=item.extrinsics,
            trajs=item.trajs_3d,
            visibs=item.visibs,
            valids=item.valids,
            queries=item.query_point,
        )

        save_recording(Path("test.rrd"))
        destroy()

        video_path = Path("demo_rgb.mp4")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        container = av.open(str(video_path), mode="w")
        stream = container.add_stream("mpeg4", rate=5)
        stream.width = item.rgbs.shape[3]
        stream.height = item.rgbs.shape[2]
        stream.pix_fmt = "yuv420p"

        for rgb in item.rgbs:
            rgb = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            packet = stream.encode(frame)
            container.mux(packet)
        container.close()

        if item.segmentation is not None:
            video_path = Path("demo_segmentation.mp4")
            video_path.parent.mkdir(parents=True, exist_ok=True)
            import random

            import numpy as np

            unique_ids = np.unique(item.segmentation)
            color_map = {
                id_: [random.randint(0, 255) for _ in range(3)] for id_ in unique_ids
            }

            container = av.open(str(video_path), mode="w")
            stream = container.add_stream("mpeg4", rate=5)
            stream.width = item.segmentation.shape[2]
            stream.height = item.segmentation.shape[1]
            stream.pix_fmt = "yuv420p"

            for seg in item.segmentation:
                color_image = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
                for id_, color in color_map.items():
                    color_image[seg == id_] = color
                frame = av.VideoFrame.from_ndarray(color_image, format="rgb24")
                packet = stream.encode(frame)
                container.mux(packet)
            container.close()
    elif args.task == "get_scale":
        dataset.set_epoch(0)

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=SliceData.collate,
        )

        start_time = time.time()
        num_batches = 0
        train_iter = iter(dataloader)
        all_metrics = []
        for step in tqdm(range(5)):
            try:
                data = next(train_iter)
            except StopIteration:
                train_iter = iter(dataloader)
                data = next(train_iter)
            
            gt_norm_factor = normalize_scene_geometry(
                data.gt_intrinsics.squeeze(0).numpy(),
                data.gt_extrinsics.squeeze(0).numpy(),
                data.gt_depths.squeeze(0).numpy(),
                norm_mode="avg_dis"
            )
            all_metrics.append(gt_norm_factor)
        
        # calculate statistics of all_metrics
        all_metrics = np.array(all_metrics)
        mean_scale = np.mean(all_metrics)
        median_scale = np.median(all_metrics)
        std_scale = np.std(all_metrics)
        logger.info(f"Mean scale: {mean_scale:.4f}, Median scale: {median_scale:.4f}, Std scale: {std_scale:.4f}")
        
    else:
        raise ValueError(f"Unknown task: {args.task}")
