# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, List, Optional
from einops import rearrange, repeat

import ipdb

from iopath.common.file_io import PathManager

path_manager = PathManager()

import cv2
import numpy as np
from line_profiler import profile
from omegaconf import DictConfig

from ..datatypes import RawSliceData

from ..utils.geometry import batch_distance_to_depth_np

from .base_provider import BaseDataProvider

logger = logging.getLogger(__name__)
st = ipdb.set_trace


def _apd_value_to_percent(x: float) -> float:
    """Match main_point_track APD CSV / bracket logic: 0–1 fractions → 0–100."""
    if x <= 1.0 + 1e-6:
        return float(np.clip(x * 100.0, 0.0, 100.0))
    return float(np.clip(x, 0.0, 100.0))


def _load_scene_apd_percent_csv(csv_path: str) -> dict[str, float]:
    """CSV: scene_name,APD (second column from eval export, e.g. '3D APD Reconstruction Dense')."""
    out: dict[str, float] = {}
    with path_manager.open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            name, val_s = row[0].strip(), row[1].strip()
            if not name:
                continue
            out[name] = _apd_value_to_percent(float(val_s))
    return out


class KubricDataProvider(BaseDataProvider):
    def __init__(self, cfg: DictConfig, name:str, override_anno: Optional[str] = None):
        super().__init__(cfg, name, override_anno)
        self.data_root: str = cfg.data_root
        self.duplicate_factor = cfg.get("duplicate_factor", -1)
        self.max_num_frames_after_duplication = cfg.get("max_num_frames_after_duplication", -1)
        # commenting out for debugging now
        self.video_folders: List[str] = [
            os.path.join(self.data_root, f)
            for f in path_manager.ls(self.data_root)
            if path_manager.isdir(os.path.join(self.data_root, f)) and f != ".cache"
        ]

        self.video_folders = [
            scene for scene in self.video_folders if scene[-1].isdigit()
        ]
        
        if cfg.is_overfit:
            np.random.seed(42)
            self.video_folders = np.random.choice(
                self.video_folders, size=10, replace=False
            ).tolist()
            
            # self.video_folders = ['/data/user_data/ayushj2/tracking_datasets/kubric/010393']
            # self.video_folders = ["000001"]
        # elif cfg.is_train:
        #     self.video_folders = [
        #         scene
        #         for scene in self.video_folders
        #         if int(scene[-1]) in list(range(9))
        #     ]  # 0 to 8
        # else:
        #     self.video_folders = [
        #         scene for scene in self.video_folders if int(scene[-1]) in [9]
        #     ]

        # self.video_folders = [os.path.join(self.data_root, "0000")]
        
        # self.video_folders.remove('000000')
        # removing for now because my est depth for this is corrupted
        
        # scene_names_to_keep = ['000013', '000025', '000016', '000060', '000041', '000017', '000021', '000075', '000031', '000085']
        # scene_names_to_keep = ['000013']
        # scene_names_to_keep = ['000027', '000091']
        # self.video_folders = [scene for scene in self.video_folders if scene.split("/")[-1] in scene_names_to_keep]
        if f'{self.data_root}/000000' in self.video_folders:
            self.video_folders.remove(f'{self.data_root}/000000')
        
        start_apd = cfg.get("start_apd")
        end_apd = cfg.get("end_apd")
        if start_apd is not None or end_apd is not None:
            apd_csv = cfg.get("apd_scene_list_csv", "/data/user_data/ayushj2/tracking_datasets/apd_scene_list.csv")
            if not apd_csv:
                raise ValueError(
                    "apd_scene_list_csv must be set when start_apd or end_apd is set "
                    "(CSV: scene_name, APD column as written by eval, e.g. "
                    "'3D APD Reconstruction Dense')."
                )
            lo = 0.0 if start_apd is None else float(start_apd)
            hi = 100.0 if end_apd is None else float(end_apd)
            scene_apd = _load_scene_apd_percent_csv(apd_csv)
            before = len(self.video_folders)
            kept: List[str] = []
            for folder in self.video_folders:
                bn = os.path.basename(folder)
                if bn in scene_apd and lo <= scene_apd[bn] <= hi:
                    kept.append(folder)

            logger.info(
                "APD filter [%s, %s] via %s: %d -> %d scenes",
                lo,
                hi,
                apd_csv,
                before,
                len(kept),
            )
            rng = np.random.default_rng(42)
            self.video_folders = rng.choice(kept, size=before, replace=True).tolist()

        self.video_folders.sort()
        self.dynamic_filter_threshold: float = cfg.get("dynamic_filter_threshold", 0.0)
        self.keep_static_prob: float = cfg.get("keep_static_prob", 1.0)
        logger.info(
            f"Loaded Kubric data from {self.data_root} with {len(self.video_folders)} sequences"
        )
        
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
            coords_transformed_homo[..., :-1] / coords_transformed_homo[..., -1:]
        )
        return coords_transformed

    @profile
    def _load_slice(
        self,
        seq_id: int,
        start: int,
        length: int,
        stride: int,
        rng: np.random.Generator,
        executor: ThreadPoolExecutor,
    ) -> RawSliceData:
        annot_file = os.path.join(
            self.data_root,
            os.path.basename(self.video_folders[seq_id]),
            f"{os.path.basename(self.video_folders[seq_id])}.npy",
        )
        with path_manager.open(annot_file, "rb") as f:
            annot = np.load(f, allow_pickle=True).item()

        depth_range = annot["depth_range"]

        traj_2d = annot["coords"].transpose(1, 0, 2)[
            start : start + length * stride : stride
        ]
        traj_3d = annot["traj_3d"].transpose(1, 0, 2)[
            start : start + length * stride : stride
        ]
        visibs = ~annot["occluded"].transpose(1, 0)[
            start : start + length * stride : stride
        ]

        rgb_list: List[Future[np.ndarray]] = []
        distance_list: List[Future[np.ndarray]] = []
        for frame_id in range(start, start + length * stride, stride):
            rgb_path = os.path.join(self.data_root, os.path.basename(self.video_folders[seq_id]), "frames", f"{frame_id:03d}.png")
            depth_path = os.path.join(self.data_root, os.path.basename(self.video_folders[seq_id]), "frames", f"{frame_id:03d}_depth.png")
            rgb_list.append(executor.submit(lambda rgb_path=rgb_path: cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)))
            distance_list.append(
                executor.submit(
                    lambda depth_range=depth_range, depth_path=depth_path: 
                    depth_range[0] + cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) * (depth_range[1] - depth_range[0]) / 65535., 
                )
            )

        # The depth annotation in kubric is actually the distance from the camera
        rgbs: Any = [f.result() for f in rgb_list]
        distances = np.stack([f.result() for f in distance_list])

        # There are negative values in the intrinsics matrix.. We take the absolute value
        # To correct this effect, we need to modify the extrinsics matrix
        intrinsics = np.abs(
            annot["intrinsics"][start : start + length * stride : stride]
        )
        intrinsics[:, 0, :] *= rgbs[0].shape[1]
        intrinsics[:, 1, :] *= rgbs[0].shape[0]
        extrinsics = annot["matrix_world"][start : start + length * stride : stride]
        extrinsics = np.linalg.inv(extrinsics)
        extrinsics = np.einsum(
            "ij,bjk->bik",
            np.diag(np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32)),
            extrinsics,
        )

        visibs[traj_2d[:, :, 0] > rgbs[0].shape[1] - 1] = False
        visibs[traj_2d[:, :, 0] < 0] = False
        visibs[traj_2d[:, :, 1] > rgbs[0].shape[0] - 1] = False
        visibs[traj_2d[:, :, 1] < 0] = False

        if self.dynamic_filter_threshold > 0.0 and self.keep_static_prob < 1.0:
            max_xyz = np.max(traj_3d, axis=0)
            min_xyz = np.min(traj_3d, axis=0)
            diff_l2 = np.linalg.norm(max_xyz - min_xyz, axis=-1)
            dynamic_filter = diff_l2 > self.dynamic_filter_threshold
            keep_mask = (
                rng.random(traj_3d.shape[1], dtype=np.float32) < self.keep_static_prob
            )
            dynamic_filter[keep_mask] = True
            visibs = visibs[:, dynamic_filter]
            traj_3d = traj_3d[:, dynamic_filter]

        rgbs = np.stack(rgbs)
        depths = batch_distance_to_depth_np(distances, intrinsics)

        # if depths[depths > 30.0].any():
        #     print('avergae bad depth value:', depths[depths > 30.0].mean())
            
        # depths[depths > 30.0] = 0.0  # clip far away depth

        max_xyz = np.max(traj_3d, axis=0)
        min_xyz = np.min(traj_3d, axis=0)
        diff_l2 = np.linalg.norm(max_xyz - min_xyz, axis=-1)
        dynamic_filter = diff_l2 > self.dynamic_filter_threshold
        is_dynamic = dynamic_filter.copy()

        trajs_depths = self._apply_homo_transform(
            traj_3d,
            repeat(extrinsics, "t i j -> t n i j", n=traj_3d.shape[1]),
        )[..., 2]
        
        if self.duplicate_factor > 1:
            # we want to repeat trajs but alternatively reverse the trajs
            traj_3d = np.concatenate([traj_3d, traj_3d[::-1]], axis=0)
            traj_3d = np.repeat(traj_3d, self.duplicate_factor // 2, axis=0)
            extrinsics = np.concatenate([extrinsics, extrinsics[::-1]], axis=0)
            extrinsics = np.repeat(extrinsics, self.duplicate_factor // 2, axis=0)
            trajs_depths = np.concatenate([trajs_depths, trajs_depths[::-1]], axis=0)
            trajs_depths = np.repeat(trajs_depths, self.duplicate_factor // 2, axis=0)
            visibs = np.concatenate([visibs, visibs[::-1]], axis=0)
            visibs = np.repeat(visibs, self.duplicate_factor // 2, axis=0)
            rgbs = np.concatenate([rgbs, rgbs[::-1]], axis=0)
            rgbs = np.repeat(rgbs, self.duplicate_factor // 2, axis=0)
            depths = np.concatenate([depths, depths[::-1]], axis=0)
            depths = np.repeat(depths, self.duplicate_factor // 2, axis=0)
            intrinsics = np.concatenate([intrinsics, intrinsics[::-1]], axis=0)
            intrinsics = np.repeat(intrinsics, self.duplicate_factor // 2, axis=0)
            
            if self.max_num_frames_after_duplication > 0:
                traj_3d = traj_3d[:self.max_num_frames_after_duplication]
                extrinsics = extrinsics[:self.max_num_frames_after_duplication]
                trajs_depths = trajs_depths[:self.max_num_frames_after_duplication]
                visibs = visibs[:self.max_num_frames_after_duplication]
                rgbs = rgbs[:self.max_num_frames_after_duplication]
                depths = depths[:self.max_num_frames_after_duplication]
                intrinsics = intrinsics[:self.max_num_frames_after_duplication]
        
        # print(traj_3d.shape)
        return RawSliceData.create(
            seq_name=self.video_folders[seq_id],
            seq_id=seq_id,
            gt_trajs_2d=traj_2d,
            gt_trajs_3d=traj_3d,
            gt_trajs_depths=trajs_depths,
            visibs=visibs,
            valids=np.ones_like(visibs, dtype=np.bool_),
            rgbs=rgbs,
            gt_depths=depths,
            gt_query_point=None,
            gt_intrinsics=intrinsics,
            gt_extrinsics=extrinsics,
            orig_resolution=np.array(
                [rgbs[0].shape[0], rgbs[0].shape[1]], dtype=np.int32
            ),  # (h, w)
            copy_gt_to_est=True,
            is_dynamic=is_dynamic,
            # bg_mask=bg_mask,
        )

    def load_seq_lens(self) -> List[int]:
        seq_lens = []
        for f in self.video_folders:
            annot_file = os.path.join(
                self.data_root, os.path.basename(f), f"{os.path.basename(f)}.npy"
            )
            with path_manager.open(annot_file, "rb") as file:
                annot = np.load(file, allow_pickle=True).item()
            seq_lens.append(len(annot["intrinsics"]))
        return seq_lens
