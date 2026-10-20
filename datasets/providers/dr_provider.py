# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import os
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from omegaconf import DictConfig
import torch
from einops import repeat

from .base_provider import BaseDataProvider
from ..datatypes import RawSliceData, SliceData
from PIL import Image
import cv2
import logging

from ..utils.dataclass_utils import load_dataclass
from dataclasses import dataclass

from ..utils.crop_utils import get_crop_args
import ipdb
st = ipdb.set_trace

logger = logging.getLogger(__name__)

@dataclass
class ImageAnnotation:
    # path to jpg file, relative w.r.t. dataset_root
    path: str
    # H x W
    size: Tuple[int, int]

@dataclass
class DynamicReplicaFrameAnnotation:
    """A dataclass used to load annotations from json."""

    # can be used to join with `SequenceAnnotation`
    sequence_name: str
    # 0-based, continuous frame number within sequence
    frame_number: int
    # timestamp in seconds from the video start
    frame_timestamp: float
    # camera viewpoint
    viewpoint: Dict[str, Any]
    # path to the depth file
    depth: Dict[str, Any]

    image: ImageAnnotation
    meta: Optional[Dict[str, Any]] = None

    camera_name: Optional[str] = None
    trajectories: Optional[str] = None

def load_16big_png_depth(depth_png: str) -> np.ndarray:
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth

def load_depth(path: str, scale_adjustment: float) -> np.ndarray:
    if path.lower().endswith(".exr"):
        # NOTE: environment variable OPENCV_IO_ENABLE_OPENEXR must be set to 1
        # You will have to accept these vulnerabilities by using OpenEXR:
        # https://github.com/opencv/opencv/issues/21326
        import cv2

        d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., 0]
        d[d > 1e9] = 0.0
    elif path.lower().endswith(".png"):
        d = load_16big_png_depth(path)
    else:
        raise ValueError('unsupported depth file name "%s"' % path)

    d = d * scale_adjustment

    d[~np.isfinite(d)] = 0.0
    return d[None]  # fake feature channel

class DynamicReplicaDataProvider(BaseDataProvider):
    def __init__(self, cfg: DictConfig, name:str, override_anno: Optional[str] = None):
        super().__init__(cfg, name, override_anno)
        self.split: str = cfg.split
        self.data_root: str = os.path.join(cfg.data_root, self.split)
        self.only_first_n_seqs: Optional[int] = cfg.get("only_first_n_seqs", None)

        self.dynamic_filter_threshold: float = cfg.get("dynamic_filter_threshold", 0.0)
        self.keep_static_prob: float = cfg.get("keep_static_prob", 1.0)

        self.max_trajs_per_sample: int = cfg.get("max_trajs_per_sample", 100000000)

        frame_annotations_file = f"frame_annotations_{self.split}.jgz"
        with gzip.open(
            os.path.join(self.data_root, frame_annotations_file), "rt", encoding="utf8"
        ) as zipfile:
            frame_annots_list = load_dataclass(
                zipfile, List[DynamicReplicaFrameAnnotation]
            )
        self.seq_annot = defaultdict(list)
        skipped_seqs: Set[str] = set()
        for frame_annot in frame_annots_list:
            # Trajectory annotations are only defined for the left camera
            if frame_annot.camera_name == "left":
                if not os.path.exists(os.path.join(self.data_root, frame_annot.image.path)):
                    skipped_seqs.add(frame_annot.sequence_name)
                    continue
                self.seq_annot[frame_annot.sequence_name].append(frame_annot)

        if len(skipped_seqs) > 0:
            logger.warning(f"The following sequences were skipped because the image was not found: {skipped_seqs}")

        self.seq_names = sorted(list(self.seq_annot.keys()))
        self.seq_lens = [len(self.seq_annot[seq_name]) for seq_name in self.seq_names]

        if self.only_first_n_seqs is not None:
            self.seq_names = self.seq_names[:self.only_first_n_seqs]
            self.seq_lens = self.seq_lens[:self.only_first_n_seqs]

        logger.info(f"Loaded DynamicReplica data from {self.data_root} with {len(self.seq_names)} sequences")

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
    
    def _load_slice(
        self,
        seq_id: int,
        start: int,
        length: int,
        stride: int,
        rng: np.random.Generator,
        executor: ThreadPoolExecutor,
    ) -> RawSliceData:
        sample: Any = self.seq_annot[self.seq_names[seq_id]][start:start+length*stride:stride]

        trajs: Any = []
        depths: Any = []

        for i in range(length):
            traj_path = os.path.join(
                self.data_root, sample[i].trajectories["path"]
            )
            trajs.append(executor.submit(torch.load, traj_path, weights_only=False)) # Dangerous

            depth_path = os.path.join(
                self.data_root, sample[i].depth["path"]
            )
            depths.append(executor.submit(load_depth, depth_path, sample[i].depth["scale_adjustment"]))

        rgbs: Any = []
        visibilities: Any = []
        traj_2d: Any = []
        traj_3d_world: Any = []
        intrinsics: Any = []
        extrinsics: Any = []

        for i in range(length):
            traj = trajs[i].result()

            visibilities.append(traj["verts_inds_vis"].numpy())
            rgbs.append(traj["img"].numpy())
            traj_2d.append(traj["traj_2d"].numpy()[..., :2])
            traj_3d_world.append(traj["traj_3d_world"].numpy())

            # Conversion from NDC to screen space. See https://pytorch3d.org/docs/cameras
            img_height, img_width, _ = traj['img'].shape
            s = min(img_height, img_width)
            assert sample[i].viewpoint['intrinsics_format'] == 'ndc_isotropic'
            fx_ndc, fy_ndc = sample[i].viewpoint['focal_length']
            px_ndc, py_ndc = sample[i].viewpoint['principal_point']
            fx_screen = fx_ndc * s / 2.0
            fy_screen = fy_ndc * s / 2.0
            px_screen = (- px_ndc * s / 2.0) + img_width / 2.0
            py_screen = (- py_ndc * s / 2.0) + img_height / 2.0

            # Not 100% sure about this
            px_screen = px_screen - 0.5
            py_screen = py_screen - 0.5

            intrinsics_ = np.array(
                [
                    [fx_screen, 0, px_screen],
                    [0, fy_screen, py_screen],
                    [0, 0, 1]
                ],
                dtype=np.float32
            )
            intrinsics.append(intrinsics_)

            R = np.array(sample[i].viewpoint['R'], dtype=np.float32)
            T = np.array(sample[i].viewpoint['T'], dtype=np.float32)

            extrinsics_ = np.eye(4, dtype=np.float32)

            # Pytorch3D uses a right-multiply format for R
            # See comments in https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/implicitron/dataset/types.py
            R = R.T

            # Since Pytorch3D assumes +Y up in the camera view space
            # We need to convert the coordinate system by adjusting the extrinsics
            # See https://pytorch3d.org/docs/cameras
            extrinsics_[:3, :3] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ R
            extrinsics_[:3, 3] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ T

            extrinsics.append(extrinsics_)

        depths = [depths[i].result()[0] for i in range(length)]

        traj_2d = np.stack(traj_2d)
        visibility = np.stack(visibilities)
        traj_3d_world = np.stack(traj_3d_world)
        intrinsics = np.stack(intrinsics)
        extrinsics = np.stack(extrinsics)

        if self.dynamic_filter_threshold > 0.0 and self.keep_static_prob < 1.0:
            max_xyz = np.max(traj_3d_world, axis=0)
            min_xyz = np.min(traj_3d_world, axis=0)
            diff_l2 = np.linalg.norm(max_xyz - min_xyz, axis=-1)
            dynamic_filter = diff_l2 > self.dynamic_filter_threshold
            keep_mask = rng.random(traj_3d_world.shape[1], dtype=np.float32) < self.keep_static_prob
            dynamic_filter[keep_mask] = True
            traj_2d = traj_2d[:, dynamic_filter]
            visibility = visibility[:, dynamic_filter]
            traj_3d_world = traj_3d_world[:, dynamic_filter]

        rgbs = np.stack(rgbs)
        depths = np.stack(depths)

        visibility[traj_2d[:, :, 0] > rgbs[0].shape[1] - 1] = False
        visibility[traj_2d[:, :, 0] < 0] = False
        visibility[traj_2d[:, :, 1] > rgbs[0].shape[0] - 1] = False
        visibility[traj_2d[:, :, 1] < 0] = False

        num_trajs = traj_3d_world.shape[1]
        perm = rng.permutation(num_trajs)[:self.max_trajs_per_sample]
        traj_3d_world = traj_3d_world[:, perm]
        visibility = visibility[:, perm]
        
        trajs_depths = self._apply_homo_transform(
            traj_3d_world,
            repeat(extrinsics, "t i j -> t n i j", n=traj_3d_world.shape[1]),
        )[..., 2]
        
        max_xyz = np.max(traj_3d_world, axis=0)
        min_xyz = np.min(traj_3d_world, axis=0)
        diff_l2 = np.linalg.norm(max_xyz - min_xyz, axis=-1)
        dynamic_filter = diff_l2 > self.dynamic_filter_threshold
        is_dynamic = dynamic_filter.copy()

        return RawSliceData.create(
            seq_name=sample[0].sequence_name,
            seq_id=seq_id,
            gt_trajs_3d=traj_3d_world,
            gt_trajs_depths=trajs_depths,
            visibs=visibility,
            valids=np.ones_like(visibility, dtype=np.bool_),
            rgbs=rgbs,
            gt_depths=depths,
            gt_intrinsics=intrinsics,
            gt_extrinsics=extrinsics,
            orig_resolution=np.array([rgbs[0].shape[0], rgbs[0].shape[1]], dtype=np.int32), # (h, w)
            copy_gt_to_est=True,
            is_dynamic=is_dynamic,
        )

    def load_seq_lens(self) -> List[int]:
        return self.seq_lens
