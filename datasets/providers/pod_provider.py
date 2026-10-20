# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from concurrent.futures import ThreadPoolExecutor
import os
import time
from typing import Any, List, Tuple, Optional
from einops import repeat
import numpy as np
from omegaconf import DictConfig
import torch
import ipdb

st = ipdb.set_trace

from .base_provider import BaseDataProvider
from ..datatypes import RawSliceData, SliceData
from ..utils.crop_utils import get_crop_args
from ..utils.dynamic_labels import dynamic_extent_l2
from ..utils.pod_corrections import (
    BUNDLED_MANIFEST_PATH,
    KNOWN_BAD_SEQUENCES,
    CorrectionSet,
    apply_camera_offset,
)
from PIL import Image
import cv2
import logging
logger = logging.getLogger(__name__)

class PointOdysseyDataProvider(BaseDataProvider):
    def __init__(self, cfg: DictConfig, name:str, override_anno: Optional[str] = None):
        super().__init__(cfg, name, override_anno)
        self.split: str = cfg.split
        self.is_overfit = self.cfg.is_overfit
        # self.is_overfit = True

        if self.is_overfit:
            self.split = "train"
        self.data_root: str = os.path.join(cfg.data_root, self.split)
        self.video_dirs: List[str] = sorted([os.path.join(self.data_root, d) for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d))])

        # Sequences whose annotations are unusable and cannot be repaired; the
        # default list lives in pod_corrections.KNOWN_BAD_SEQUENCES so it applies
        # without any config opt-in. Pass exclude_sequences to override (use [] to
        # keep everything). See docs/pod_known_bad_sequences.md for the evidence.
        excl = cfg.get("exclude_sequences") if "exclude_sequences" in cfg else KNOWN_BAD_SEQUENCES
        excl = {str(e).strip().rstrip("/") for e in (excl or []) if str(e).strip()}
        if excl:
            before = len(self.video_dirs)
            self.video_dirs = [
                d for d in self.video_dirs if os.path.basename(d.rstrip("/")) not in excl
            ]
            dropped = before - len(self.video_dirs)
            if dropped:
                logger.warning(
                    "PointOdyssey(%s): excluded %d sequence(s) flagged as unusable: %s",
                    self.split, dropped, sorted(excl),
                )
        
        if self.is_overfit:
            # np.random.seed(42)
            self.video_dirs = np.random.choice(
                self.video_dirs, size=10, replace=False
            ).tolist()
            self.video_dirs = [os.path.join(self.data_root, 'ani15_new_')]
        logger.info(f"Loaded PointOdyssey data from {self.data_root} with {len(self.video_dirs)} videos")
        self.valid_depth_range: Tuple[float, float] = cfg.valid_depth_range

        self.dynamic_filter_threshold: float = cfg.dynamic_filter_threshold
        self.keep_static_prob: float = cfg.keep_static_prob

        # Releasable annotation corrections (see datasets/utils/pod_corrections.py
        # and docs/pod_annotation_bugs.md). Chiefly a per-sequence camera-pose
        # frame offset: extrinsics[t] is the pose that rendered frame t-1 in part
        # of POD, which drags trajs_2d off its object whenever the camera moves
        # fast. Applied BY DEFAULT from the manifest bundled with the repo; set
        # correction_root to use a different manifest, or null for an A/B.
        if "correction_root" in cfg:
            correction_root = cfg.get("correction_root")
            correction_root = (
                None if correction_root in (None, "", "null") else str(correction_root)
            )
        else:
            correction_root = BUNDLED_MANIFEST_PATH
        self.correction_root: Optional[str] = correction_root
        self.corrections: Optional[CorrectionSet] = None
        if self.correction_root:
            if not os.path.exists(self.correction_root):
                raise FileNotFoundError(
                    f"POD corrections manifest not found at {self.correction_root}. "
                    "Point correction_root at a valid manifest, or set it to null to "
                    "train on uncorrected annotations."
                )
            self.corrections = CorrectionSet.load(self.correction_root)
            n_here = sum(1 for k in self.corrections.keys if k.startswith(f"{self.split}/"))
            logger.info(
                "POD corrections enabled from %s: %d sequences for split=%s",
                self.correction_root, n_here, self.split,
            )
        else:
            logger.warning(
                "POD corrections DISABLED (correction_root=null); annotations carry "
                "the known camera-offset defect. See docs/pod_annotation_bugs.md."
            )

        # Frames where the repair leaves POD's stored visibility meaningless
        # (POD derived visibs from the pre-repair pose). On by default; inert only
        # if corrections are disabled. Measured on 150 targeted windows: spikes
        # (loss > 2.0) 15 -> 5, p95 3.09 -> 1.73, p=9.6e-09, control group unmoved.
        # This switch is worth more than the camera repair itself.
        self.invalidate_unreliable_visibility: bool = bool(
            cfg.get("invalidate_unreliable_visibility", True)
        )

        # Stop supervising a track once its 2D projection leaves the image by
        # more than this many pixels (None/-1 disables). PIPs2 uses 64.
        _m = cfg.get("oob_valid_margin_px", 64.0)
        self.oob_valid_margin_px = None if _m in (None, "", "null") else float(_m)

        # PIPs2-style motion sanity caps on the 2D track, normalised by
        # max(H, W) (datasets/pointodysseydataset.py: "some of the data is a bit
        # crazy, so we will filter down based on motion"). Measured on 12.0M val
        # tracks the distribution is sharply bimodal -- p95 is ~0.09 and p99 is
        # ~22 -- so these thresholds sit in an empty gap and the exact value
        # barely matters: 1x drops 4.71% of tracks, 8x still drops 2.38%.
        # A track breaching any cap is invalidated for the whole window, as in
        # PIPs2 (`trajs = trajs[:, mot_ok]`). null disables a cap.
        def _f(key, default):
            v = cfg.get(key, default)
            return None if v in (None, "", "null") else float(v)

        self.motion_vel_max = _f("motion_vel_max", 0.4)
        self.motion_accel_max = _f("motion_accel_max", 0.3)
        self.motion_jerk_max = _f("motion_jerk_max", 0.1)
        # If a track goes far out of frame anywhere in the window, drop it for
        # the whole window rather than only for the excursion frames.
        self.oob_drop_track_in_window: bool = bool(
            cfg.get("oob_drop_track_in_window", False)
        )

        self._mmap_cache: dict[str, dict[str, np.ndarray]] = {}

    _ANNO_KEYS = ("trajs_2d", "trajs_3d", "visibs", "valids", "intrinsics", "extrinsics")

    def _get_mmap_arrays(self, video_dir: str) -> dict[str, np.ndarray]:
        if video_dir not in self._mmap_cache:
            split_dir = os.path.join(video_dir, "anno_split")
            self._mmap_cache[video_dir] = {
                k: np.load(os.path.join(split_dir, f"{k}.npy"), mmap_mode="r")
                for k in self._ANNO_KEYS
            }
        return self._mmap_cache[video_dir]

    def _apply_homo_transform(self, coords: np.ndarray, transform: np.ndarray, homo_input: bool = False):
        if homo_input:
            coords_homo = coords
        else:
            coords_homo = np.concatenate([coords, np.ones_like(coords[..., :1])], axis=-1)
        coords_transformed_homo = np.einsum("...ij,...j->...i", transform, coords_homo)
        coords_transformed = coords_transformed_homo[..., :-1] / coords_transformed_homo[..., -1:]
        return coords_transformed
        
    def _load_slice(
        self, 
        seq_id: int, 
        start: int, 
        length: int, 
        stride: int, 
        rng: np.random.Generator,
        executor: ThreadPoolExecutor
    ) -> RawSliceData:
        video_dir = self.video_dirs[seq_id]
        rgb_folder = os.path.join(video_dir, 'rgbs')
        depth_folder = os.path.join(video_dir, 'depths')
        # start = 147
        sl = slice(start, start + length * stride, stride)
        split_dir = os.path.join(video_dir, 'anno_split')
        if os.path.isdir(split_dir):
            anns = self._get_mmap_arrays(video_dir)
        else:
            annotation_path = os.path.join(video_dir, 'anno.npz')
            anns = np.load(annotation_path, mmap_mode='r')

        trajs_2d: np.ndarray = np.array(anns['trajs_2d'][sl])
        trajs_3d: Any = np.array(anns['trajs_3d'][sl])
        visibs = np.array(anns['visibs'][sl])
        valids = np.array(anns['valids'][sl])
        intrinsics = np.array(anns['intrinsics'][sl])
        extrinsics = np.array(anns['extrinsics'][sl]).reshape(-1, 4, 4)

        correction = None
        if self.corrections is not None:
            correction = self.corrections.get(
                self.split, os.path.basename(video_dir.rstrip("/"))
            )
        if correction is not None and correction.cam_offset != 0:
            frame_indices = np.arange(start, start + length * stride, stride)
            n_total = int(np.asarray(anns['extrinsics']).shape[0])
            trajs_2d, intrinsics, extrinsics, valids, _ = apply_camera_offset(
                correction, frame_indices, n_total, anns, trajs_3d, valids
            )
            if self.invalidate_unreliable_visibility:
                bad_vis = correction.unreliable_visibility_mask(frame_indices)
                if bad_vis.any():
                    valids = np.array(valids, copy=True)
                    valids[bad_vis] = False

        # Load RGB and depth frames
        rgb_obs: Any = []
        depth_obs: Any = []
        for frame_idx in range(start, start + length * stride, stride):
            rgb_path = os.path.join(rgb_folder, f'rgb_{frame_idx:05d}.jpg')
            depth_path = os.path.join(depth_folder, f'depth_{frame_idx:05d}.png')

            assert os.path.exists(rgb_path), f"RGB image {rgb_path} does not exist"
            assert os.path.exists(depth_path), f"Depth image {depth_path} does not exist"

            rgb_obs.append(executor.submit(lambda path: Image.open(path).convert('RGB'), rgb_path))
            depth_obs.append(
                executor.submit(
                    lambda path: cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0 * 1000.0,
                    depth_path
                )
            )

        rgb_obs = np.stack([rgb.result() for rgb in rgb_obs]) # (t, h, w, 3)
        depth_obs = np.stack([depth.result() for depth in depth_obs]) # (t, h, w)

        depth_obs[depth_obs < self.valid_depth_range[0]] = 0.0
        depth_obs[depth_obs > self.valid_depth_range[1]] = 0.0

        local_trajs_3d = self._apply_homo_transform(trajs_3d, repeat(extrinsics, 't i j -> t n i j', n=trajs_3d.shape[1]), homo_input=False)
        valids[local_trajs_3d[:, :, -1] < self.valid_depth_range[0]] = False
        valids[local_trajs_3d[:, :, -1] > self.valid_depth_range[1]] = False
        # Drop outlier annotations: any world xyz coordinate blowing far past the
        # depth range (~2x) indicates a bad/extrapolated 3D position.
        xyz_limit = 2.0 * self.valid_depth_range[1]
        valids[(np.abs(trajs_3d) > xyz_limit).any(axis=-1)] = False
        trajs_depths = local_trajs_3d[..., 2]

        img_h, img_w = rgb_obs[0].shape[0], rgb_obs[0].shape[1]
        visibs[trajs_2d[:, :, 0] > img_w - 1] = False
        visibs[trajs_2d[:, :, 0] < 0] = False
        visibs[trajs_2d[:, :, 1] > img_h - 1] = False
        visibs[trajs_2d[:, :, 1] < 0] = False

        # Points that leave the frame by a wide margin should not be supervised.
        # Only `visibs` was being cleared above, so a track whose GT sat hundreds
        # of pixels off-screen kept valids=1 and was still scored -- val
        # r1_new_f idx 396 track 39 swings to y = -2690 (it passes close to the
        # camera, so its projection blows up), comes back at frame 16, and the
        # model is penalised ~330 px for the rest of the clip for failing to
        # re-acquire something it never had visual evidence of.
        #
        # PIPs2 does the same thing at +-64 px (datasets/pointodysseydataset.py:
        # "when a point moves far oob, don't supervise with it").
        if self.oob_valid_margin_px is not None and self.oob_valid_margin_px >= 0:
            m = float(self.oob_valid_margin_px)
            very_oob = (
                (trajs_2d[:, :, 0] < -m) | (trajs_2d[:, :, 0] > img_w - 1 + m)
                | (trajs_2d[:, :, 1] < -m) | (trajs_2d[:, :, 1] > img_h - 1 + m)
            )
            valids[very_oob] = False

            # Clearing only the excursion frames is not enough: once a track has
            # left the frame the model has lost it, and the frames after it
            # returns are a re-detection problem, not a tracking one. PIPs2 gets
            # this for free -- its velocity/accel/jerk filter drops the whole
            # track from the clip, since such an excursion implies ~1500 px/frame
            # against their 384 px cap. We make it explicit instead of relying on
            # a motion threshold that would also discard genuinely fast objects.
            if self.oob_drop_track_in_window and very_oob.any():
                valids[:, very_oob.any(axis=0)] = False

        # Motion sanity. Computed on the raw (unclamped) 2D track, exactly as
        # PIPs2 does, so an excursion still registers even though the OOB frames
        # above have already been invalidated -- that is what makes it fire on
        # tracks that swing far off-screen and come back.
        caps = (self.motion_vel_max, self.motion_accel_max, self.motion_jerk_max)
        if any(c is not None for c in caps) and trajs_2d.shape[0] >= 2:
            norm = float(max(img_h, img_w))
            p = np.where(np.isfinite(trajs_2d), trajs_2d, np.nan) / norm
            bad = np.zeros(trajs_2d.shape[1], dtype=bool)
            d = p
            for cap in caps:
                if d.shape[0] < 2:
                    break
                d = np.diff(d, axis=0)
                if cap is None:
                    continue
                with np.errstate(invalid="ignore"):
                    mag = np.linalg.norm(d, axis=-1)
                    peak = np.nanmax(mag, axis=0) if mag.shape[0] else None
                if peak is not None:
                    bad |= np.nan_to_num(peak, nan=0.0) > cap
            if bad.any():
                valids[:, bad] = False

        visibs = visibs & valids

        assert rgb_obs.dtype == np.uint8

        # Static/dynamic GT is measured over VALID frames only: coordinates on
        # invalid frames are not meaningful, and one junk value stretches the
        # extent enough to mark a stationary point dynamic (5365 mislabelled
        # points in a single kg_ego2 window). See datasets/utils/dynamic_labels.py.
        diff_l2 = dynamic_extent_l2(trajs_3d, valids)
        dynamic_filter = diff_l2 > self.dynamic_filter_threshold
        is_dynamic = dynamic_filter.copy()
        
        if self.dynamic_filter_threshold > 0.0:
            keep_mask = rng.random(trajs_3d.shape[1], dtype=np.float32) < self.keep_static_prob
            dynamic_filter[keep_mask] = True
            trajs_2d = trajs_2d[:, dynamic_filter]
            trajs_3d = trajs_3d[:, dynamic_filter]
            visibs = visibs[:, dynamic_filter]
            valids = valids[:, dynamic_filter]
            is_dynamic = is_dynamic[dynamic_filter] 
            trajs_depths = trajs_depths[:, dynamic_filter]

        return RawSliceData.create(
            rgbs=rgb_obs,
            seq_name=video_dir,
            seq_id=seq_id,
            gt_trajs_2d=trajs_2d,
            gt_trajs_3d=trajs_3d,
            visibs=visibs,
            valids=valids,
            gt_query_point=None,
            gt_depths=depth_obs,
            gt_intrinsics=intrinsics,
            gt_extrinsics=extrinsics,
            orig_resolution=np.array([rgb_obs[0].shape[0], rgb_obs[0].shape[1]], dtype=np.int32), # (h, w)
            copy_gt_to_est=True,
            is_dynamic=is_dynamic,
            gt_trajs_depths=trajs_depths,
        )

    def load_seq_lens(self) -> List[int]:
        seq_lens = []
        for video_dir in self.video_dirs:
            info = np.load(os.path.join(video_dir, 'info.npz'))
            try:
                num_frames, num_points, _ = info['trajs_3d']
            except:
                print(f"Error loading {video_dir}/info.npz")
                seq_lens.append(0)
                continue
            seq_lens.append(num_frames)
        return seq_lens
