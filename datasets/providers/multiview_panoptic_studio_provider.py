# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
from einops import rearrange
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation

from ..datatypes import RawSliceData
from .base_provider import BaseDataProvider

logger = logging.getLogger(__name__)


def _parse_panoptic_name(name: str) -> Dict[str, Any]:
    """Parse a panoptic-multiview dataset name into config overrides.

    Supports the same suffix convention as the third-party
    ``PanopticStudioMultiViewDataset.from_name``.  Examples::

        multiview_panoptic_studio                          # base
        multiview_panoptic_studio-views27_16               # specific views
        multiview_panoptic_studio-views27_16_14_8          # four views

    Returns a dict with keys that can override provider config values:
      - views_to_return: Optional[List[int]]
    """
    result: Dict[str, Any] = {}

    # Strip the base provider name
    rest = re.sub(r"^multiview_panoptic_studio", "", name)
    if rest.startswith("-views"):
        match = re.match(r"-views((?:\d+_?)+)", rest)
        assert match is not None, f"Could not parse views from: {rest}"
        result["views_to_return"] = list(map(int, match.group(1).split("_")))
        rest = rest.replace(match.group(0), "", 1)

    assert rest == "", f"Unparsed portion of dataset name: {rest!r}"
    return result


class MultiviewPanopticStudioProvider(BaseDataProvider):
    """
    Provider for multiview Panoptic Studio data.

    Flattens (n_views, n_frames) into a single time dimension of length
    n_frames * n_views in time-major order, i.e. flat index
    ``t = f * n_views + v``; the model recovers the timestep index by integer
    division, so the ordering is part of the contract.
    ``RawSliceData.num_views`` carries n_views downstream.

    Configuration can come from the YAML config *or* from suffixes encoded in
    the provider ``name`` (name-based values take precedence).  Examples::

        name: multiview_panoptic_studio              # uses config.views_to_return
        name: multiview_panoptic_studio-views27_16   # overrides to views [27, 16]

    ``views_to_return`` entries are positions in the sorted list of view
    folders under ``ims/`` (which is also the annotation view order).
    """

    def __init__(
        self, cfg: DictConfig, name: str, override_anno: Optional[str] = None
    ):
        super().__init__(cfg, name, override_anno)
        self.data_root: str = cfg.data_root

        name_overrides = _parse_panoptic_name(name)

        self.views_to_return: Optional[List[int]] = name_overrides.get(
            "views_to_return",
            list(cfg.views_to_return) if cfg.get("views_to_return") else None,
        )
        self.traj_per_sample: int = cfg.get("traj_per_sample", 384)
        self.seed: Optional[int] = cfg.get("seed", 72)
        max_videos: Optional[int] = cfg.get("max_videos", None)

        self.seq_names = self._get_sequence_names(max_videos)
        self.video_folders = [
            os.path.join(self.data_root, s) for s in self.seq_names
        ]
        logger.info(
            "Loaded PanopticStudio data from %s with %d sequences (views=%s)",
            self.data_root,
            len(self.seq_names),
            self.views_to_return,
        )

    def _get_sequence_names(self, max_videos: Optional[int]) -> List[str]:
        seq_names = sorted(
            f
            for f in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, f))
            and not f.startswith(".")
            and not f.startswith("_")
        )
        valid: List[str] = []
        for s in seq_names:
            annot_path = os.path.join(self.data_root, s, "tapvid3d_annotations.npz")
            if not os.path.exists(annot_path):
                warnings.warn(f"Skipping {s}: no tapvid3d_annotations.npz")
                continue
            valid.append(s)
        if max_videos is not None:
            valid = valid[:max_videos]
        return valid

    def _load_slice(
        self,
        seq_id: int,
        start: int,
        length: int,
        stride: int,
        rng: np.random.Generator,
        executor: ThreadPoolExecutor,
    ) -> RawSliceData:
        seq_name = self.seq_names[seq_id]
        datapoint_path = os.path.join(self.data_root, seq_name)
        ims_path = os.path.join(datapoint_path, "ims")
        depths_path = os.path.join(datapoint_path, "dynamic3dgs_depth")

        # ── Load annotations ──
        annot = np.load(os.path.join(datapoint_path, "tapvid3d_annotations.npz"))
        traj3d_world_all = annot["trajectories"]          # (T_total, N_all, 3)
        traj2d_all = annot["trajectories_pixelspace"]      # (V_total, T_total, N_all, 2)
        vis_all = annot["per_view_visibilities"]            # (V_total, T_total, N_all)
        query_points_3d = annot["query_points_3d"]          # (N_all, 4)  (t, x, y, z)
        all_extrs = annot["extrinsics"]                     # (V_total, 4, 4)
        all_intrs = annot["intrinsics"]                     # (V_total, 3, 3)

        # ── Determine views ──
        view_folders = sorted(os.listdir(ims_path), key=lambda x: int(x))
        views = (
            self.views_to_return
            if self.views_to_return is not None
            else list(range(len(view_folders)))
        )
        assert all(0 <= v < len(view_folders) for v in views), (
            f"views_to_return={views} out of range for {len(view_folders)} views "
            f"in {ims_path}"
        )
        n_views = len(views)

        # ── Frame selection ──
        n_available = annot["trajectories"].shape[0]
        frame_indices = list(range(start, start + length * stride, stride))
        assert frame_indices and frame_indices[-1] < n_available, (
            f"requested frames up to {frame_indices[-1] if frame_indices else start} "
            f"but {seq_name} has only {n_available}"
        )
        n_frames = len(frame_indices)

        # ── Load RGBs (parallel I/O) ──
        rgb_futures: List[List[Future]] = []
        for v in views:
            rgb_folder = os.path.join(ims_path, view_folders[v])
            rgb_files = sorted(os.listdir(rgb_folder))
            vf: List[Future] = []
            for fi in frame_indices:
                p = os.path.join(rgb_folder, rgb_files[fi])
                vf.append(
                    executor.submit(lambda p=p: cv2.imread(p)[:, :, ::-1].copy())
                )
            rgb_futures.append(vf)

        # ── Load depths (parallel I/O) ──
        # Depth files are named by camera id, which is the view folder name.
        depth_futures: List[Future] = []
        for v in views:
            depth_file = os.path.join(
                depths_path, f"depths_{int(view_folders[v]):02d}.npy"
            )
            depth_futures.append(
                executor.submit(
                    lambda f=depth_file, fi=frame_indices: np.load(f)[fi]
                )
            )

        # ── Collect loaded data ──
        rgbs = np.stack(
            [np.stack([f.result() for f in vf]) for vf in rgb_futures]
        )  # (V, F, H, W, 3) uint8
        depths = np.stack(
            [f.result() for f in depth_futures]
        ).astype(np.float32)  # (V, F, H, W)
        _, _, h, w, _ = rgbs.shape

        # ── Camera parameters ──
        intrs = np.stack([all_intrs[v] for v in views]).astype(np.float32)
        intrs = np.tile(intrs[:, None], (1, n_frames, 1, 1))  # (V, F, 3, 3)

        # 3x4 extrinsics for camera-space projection
        extrs_3x4 = np.stack(
            [all_extrs[v][:3, :] for v in views]
        ).astype(np.float32)
        extrs_3x4 = np.tile(extrs_3x4[:, None], (1, n_frames, 1, 1))  # (V, F, 3, 4)

        # ── Slice annotations by view & frame ──
        traj2d = traj2d_all[views][:, frame_indices].copy()        # (V, F, N, 2)
        visibility = vis_all[views][:, frame_indices].copy()        # (V, F, N)
        traj3d_world = traj3d_world_all[frame_indices].copy()       # (F, N, 3)

        # ── Camera-space depths per view ──
        pt_homo = np.concatenate(
            [traj3d_world, np.ones_like(traj3d_world[..., :1])], axis=-1
        )  # (F, N, 4)
        pt_cam = np.einsum(
            "VFij,FNj->VFNi", extrs_3x4, pt_homo
        )  # (V, F, N, 3)
        traj2d_w_z = np.concatenate(
            [traj2d, pt_cam[..., 2:]], axis=-1
        )  # (V, F, N, 3)

        # ── Mask visibility by query-point appearance time ──
        abs_frame_times = np.array(frame_indices)
        query_timesteps = query_points_3d[:, 0].round().astype(int)
        visibility = visibility.astype(bool) & (
            abs_frame_times[None, :, None] >= query_timesteps[None, None, :]
        )

        # ── Track selection ──
        seed = self.seed if self.seed is not None else rng.integers(0, 2**32 - 1)
        rnd = np.random.RandomState(seed=seed)

        # Keep a track if the reference camera (view 0) sees it at least twice,
        # so the candidate pool does not grow with the view count.
        visible_ref_view = visibility[0]                    # (F, N)
        visible_enough = visible_ref_view.sum(axis=0) >= 2  # (N,)
        valid_track_ids = np.where(visible_enough)[0]

        perm = rnd.permutation(len(valid_track_ids))
        n_select = min(self.traj_per_sample, len(valid_track_ids))
        assert n_select > 0, f"No valid tracks found for sequence {seq_name}"
        selected = valid_track_ids[perm[:n_select]]

        traj2d = traj2d[:, :, selected]          # (V, F, n, 2)
        traj2d_w_z = traj2d_w_z[:, :, selected]  # (V, F, n, 3)
        traj3d_world = traj3d_world[:, selected]  # (F, n, 3)
        visibility = visibility[:, :, selected]   # (V, F, n)

        # Per-view valids (non-NaN per that specific view).
        # This guarantees visible ⊆ valid after flattening.
        valids = ~np.isnan(traj2d).any(axis=-1)   # (V, F, n)
        visibility = visibility & valids

        # Replace NaNs with zeros
        traj2d = np.nan_to_num(traj2d, nan=0.0)
        traj2d_w_z = np.nan_to_num(traj2d_w_z, nan=0.0)
        traj3d_world = np.nan_to_num(traj3d_world, nan=0.0)

        # ── Scene normalization (match Kubric scale / orientation) ──
        scale = 2.5
        rot_mat = Rotation.from_euler("x", -90, degrees=True).as_matrix()
        Rt = np.eye(4, dtype=np.float64)
        Rt[:3, :3] = rot_mat
        Rt_inv = np.linalg.inv(Rt)

        # Depths
        depths = depths * scale

        # Extrinsics (full 4x4): scale translation, then rotate world
        extrs_44 = np.stack([all_extrs[v] for v in views]).astype(np.float64)
        extrs_44 = np.tile(extrs_44[:, None], (1, n_frames, 1, 1))  # (V, F, 4, 4)
        extrs_44[:, :, :3, 3] *= scale
        extrs_44 = np.einsum("VFki,ij->VFkj", extrs_44, Rt_inv)

        # 3D world trajectories
        traj3d_scaled = traj3d_world.astype(np.float64) * scale
        traj3d_homo = np.concatenate(
            [traj3d_scaled, np.ones_like(traj3d_scaled[..., :1])], axis=-1
        )
        traj3d_trans = np.einsum("ij,FNj->FNi", Rt, traj3d_homo)[..., :3]

        # Camera-space depths (only z is affected by the scale)
        trajs_depths = traj2d_w_z[..., 2] * scale  # (V, F, n)

        # ── Flatten multiview time-major: (V, F, ...) → (F*V, ...) ──
        rgbs_flat = rearrange(rgbs, "v f h w c -> (f v) h w c")
        depths_flat = rearrange(depths, "v f h w -> (f v) h w").astype(np.float32)
        intrs_flat = rearrange(intrs, "v f i j -> (f v) i j").astype(np.float32)
        extrs_flat = rearrange(extrs_44, "v f i j -> (f v) i j").astype(np.float32)
        traj2d_flat = rearrange(traj2d, "v f n c -> (f v) n c").astype(np.float32)
        trajs_depths_flat = rearrange(trajs_depths, "v f n -> (f v) n").astype(np.float32)
        visibility_flat = rearrange(visibility, "v f n -> (f v) n").astype(np.bool_)
        valids_flat = rearrange(valids, "v f n -> (f v) n").astype(np.bool_)

        # 3D world trajs are view-independent → repeat each frame across views
        traj3d_flat = np.repeat(traj3d_trans, repeats=n_views, axis=0).astype(np.float32)
        return RawSliceData.create(
            seq_name=seq_name,
            seq_id=seq_id,
            gt_trajs_2d=traj2d_flat,
            gt_trajs_3d=traj3d_flat,
            gt_trajs_depths=trajs_depths_flat,
            visibs=visibility_flat,
            valids=valids_flat,
            rgbs=rgbs_flat,
            gt_depths=depths_flat,
            gt_query_point=None,
            gt_intrinsics=intrs_flat,
            gt_extrinsics=extrs_flat,
            orig_resolution=np.array([h, w], dtype=np.int32),
            copy_gt_to_est=True,
            is_dynamic=None,
            num_views=n_views,
        )

    def load_seq_lens(self) -> List[int]:
        seq_lens = []
        for seq_name in self.seq_names:
            annot_path = os.path.join(
                self.data_root, seq_name, "tapvid3d_annotations.npz"
            )
            annot = np.load(annot_path)
            seq_lens.append(annot["trajectories"].shape[0])
        return seq_lens
