# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Optional

import h5py

import ipdb
from einops import rearrange, repeat

from iopath.common.file_io import PathManager

path_manager = PathManager()

import numpy as np
import torch
import cv2
from line_profiler import profile
from omegaconf import DictConfig

from ...utils.data_utils import read_gen_normal as read_gen


from ..datatypes import RawSliceData

from ..utils.geometry import (
    batch_distance_to_depth_np,
    get_transformation_matrix,
    opencv_from_cameras_projection,
)

from .base_provider import BaseDataProvider

logger = logging.getLogger(__name__)
st = ipdb.set_trace


def safe_inverse(a):  # parallel version
    """
    Invert a batch of 4x4 homogeneous transformation matrices.
    a: numpy array of shape (B, 4, 4)
    """
    B, _, _ = a.shape
    inv = a.copy()

    r = a[:, :3, :3]  # rotation part
    r_transpose = np.transpose(r, (0, 2, 1))  # inverse of rotation

    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -np.matmul(r_transpose, a[:, :3, 3:4])

    return inv


class KubricAdamDataProvider(BaseDataProvider):
    def __init__(self, cfg: DictConfig, name:str, override_anno: Optional[str] = None):
        super().__init__(cfg, name, override_anno)
        self.data_root: str = cfg.data_root

        self.video_folders = sorted(path_manager.ls(self.data_root))
        self.video_folders = [
            scene for scene in self.video_folders if scene[-1].isdigit()
        ]
        if cfg.is_overfit:
            self.video_folders = ["0000"]
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

        # self.video_folders = ["0000"]
        # print("ONLY FOR DEBUGGING!!!!!!!!!!!!!!!!")
        # self.video_folders = [os.path.join(self.data_root, "0000")]

        # self.video_folders.sort()
        self.dynamic_filter_threshold: float = cfg.get("dynamic_filter_threshold", 0.0)
        self.keep_static_prob: float = cfg.get("keep_static_prob", 1.0)
        logger.info(
            f"Loaded Kubric data from {self.data_root} with {len(self.video_folders)} sequences"
        )

        # load the cached sequence lengths
        # with path_manager.open(self.data_root + "/seq_lens.npy", "rb") as f:
        self.seq_lens = [120] * len(self.video_folders)

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

        inds = np.arange(start, start + length * stride, stride)

        visibs = ~annot["visibility"].transpose(1, 0)[inds]

        # load rgb and depths
        rgb_list: List[Future[np.ndarray]] = []
        for frame_id in inds:
            rgb_path = os.path.join(
                self.data_root,
                os.path.basename(self.video_folders[seq_id]),
                "frames",
                f"{frame_id:03d}.png",
            )
            rgb_list.append(
                executor.submit(
                    lambda rgb_path=rgb_path: cv2.cvtColor(
                        cv2.imread(rgb_path), cv2.COLOR_BGR2RGB
                    )
                )
            )

        # assert stride == 1, "loading from mp4 is not implemented for stride > 1"

        depth_list: List[Future[np.ndarray]] = []
        for frame_id in range(start, start + length * stride, stride):
            depth_path = os.path.join(
                self.data_root,
                os.path.basename(self.video_folders[seq_id]),
                "depths",
                f"{frame_id:03d}.npy",
            )

            depth_list.append(executor.submit(read_gen, depth_path))

        # The depth annotation in kubric is actually the distance from the camera
        rgbs = [f.result() for f in rgb_list]
        depths = [f.result() for f in depth_list]

        camera = annot["camera"]

        # process intrinsics
        H, W = annot["metadata"]["height"], annot["metadata"]["width"]
        focal_length = camera["focal_length"]  # e.g., 33.637524
        sensor_width = camera["sensor_width"]  # e.g., 32
        f_x = focal_length / sensor_width
        f_y = focal_length / sensor_width * H / W
        p_x, p_y = 0.5, 0.5
        principal_point = np.array([p_x, p_y])
        focal_length = np.array([f_x, f_y])
        image_size = np.array([H, W])
        intrinsics = opencv_from_cameras_projection(
            focal_length, principal_point, image_size
        )

        bg_mask = (annot["segmentations"][inds] == 0).squeeze(-1)

        intrinsics = np.repeat(intrinsics[None], len(rgbs), axis=0).astype(np.float32)

        # process extrinsics
        camera_ts = camera["positions"][inds]  # T,3
        camera_quats = camera["quaternions"][inds]  # T,4
        world_T_cam = np.stack(
            [
                get_transformation_matrix(
                    camera_ts[i],
                    np.concatenate([camera_quats[i, 1:], camera_quats[i, 0:1]]),
                )
                for i in range(len(camera_ts))
            ]
        )  # this is world_T_cam
        # keeps x, flips y, and flips z
        world_T_cam = np.einsum(
            "bij,jk->bik",
            world_T_cam,
            np.diag(np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32)),
        )


        # process traj_3d
        trajs_2d = annot["coords"].transpose(1, 0, 2)[inds]  # T,N,2

        # trajs_2d = np.floor(trajs_2d).astype(np.float32)
        trajs_z = annot["coords_depth"].transpose(1, 0)[inds]  # T,N


        all_valid = np.nonzero(
            np.sum(np.sum(np.abs(trajs_2d), axis=-1) < 100000, axis=0) == length
        )[0]
        trajs_2d = trajs_2d[:, all_valid]
        trajs_z = trajs_z[:, all_valid]
        visibs = visibs[:, all_valid]

        visibs[trajs_2d[:, :, 0] > rgbs[0].shape[1] - 1] = False
        visibs[trajs_2d[:, :, 0] < 0] = False
        visibs[trajs_2d[:, :, 1] > rgbs[0].shape[0] - 1] = False
        visibs[trajs_2d[:, :, 1] < 0] = False

        K_inv = np.linalg.inv(intrinsics[..., :3, :3])
        trajs_2d_homo = np.concatenate(
            [trajs_2d, np.ones_like(trajs_2d[..., :1])],
            axis=-1,
        )
        camera_coords = np.einsum("...ij, ...j->...i", K_inv[:, None], trajs_2d_homo)
        trajs_depths = (
            camera_coords[..., -1] * trajs_z / np.sqrt((camera_coords**2).sum(-1))
        )

        camera_coords = camera_coords * trajs_depths[..., None]
        camera_coords = np.concatenate(
            (camera_coords, np.ones_like(camera_coords[..., :1])), axis=-1
        )
        traj_3d = np.einsum("tij,tnj->tni", world_T_cam, camera_coords)
        traj_3d = traj_3d[..., :3]

        max_xyz = np.max(traj_3d, axis=0)
        min_xyz = np.min(traj_3d, axis=0)
        diff_l2 = np.linalg.norm(max_xyz - min_xyz, axis=-1)

        dynamic_filter = diff_l2 > self.dynamic_filter_threshold
        is_dynamic = dynamic_filter.copy()

        if self.dynamic_filter_threshold > 0.0 and self.keep_static_prob < 1.0:
            keep_mask = (
                rng.random(traj_3d.shape[1], dtype=np.float32) < self.keep_static_prob
            )
            dynamic_filter[keep_mask] = True
            visibs = visibs[:, dynamic_filter]
            traj_3d = traj_3d[:, dynamic_filter]
            trajs_2d = trajs_2d[:, dynamic_filter]
            trajs_depths = trajs_depths[:, dynamic_filter]

        rgbs = np.stack(rgbs)
        depths = np.stack(depths).astype(np.float32)

        return RawSliceData.create(
            seq_name=self.video_folders[seq_id],
            seq_id=seq_id,
            gt_trajs_2d=trajs_2d,
            gt_trajs_3d=traj_3d,
            gt_trajs_depths=trajs_depths,
            visibs=visibs,
            valids=np.ones_like(visibs, dtype=np.bool_),
            rgbs=rgbs,
            gt_depths=depths,
            gt_query_point=None,
            gt_intrinsics=intrinsics[:, :3, :3],
            gt_extrinsics=safe_inverse(world_T_cam),  # thus this is cam_T_world
            orig_resolution=np.array(
                [rgbs[0].shape[0], rgbs[0].shape[1]], dtype=np.int32
            ),  # (h, w)
            copy_gt_to_est=True,
            is_dynamic=is_dynamic,
            bg_mask=bg_mask,
        )

    def load_seq_lens(self) -> List[int]:
        seq_lens = []
        for f in self.video_folders:
            annot_file = os.path.join(
                self.data_root, os.path.basename(f), f"{os.path.basename(f)}.npy"
            )
            with path_manager.open(annot_file, "rb") as file:
                annot = np.load(file, allow_pickle=True).item()
            seq_lens.append(len(annot["camera"]["quaternions"]))
        return seq_lens
