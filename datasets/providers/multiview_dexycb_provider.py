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
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from einops import rearrange
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation

from ..datatypes import RawSliceData
from .base_provider import BaseDataProvider

logger = logging.getLogger(__name__)


def _parse_dexycb_name(name: str) -> Dict[str, Any]:
    """Parse multiview DexYCB provider name suffixes."""
    result: Dict[str, Any] = {}
    rest = re.sub(r"^multiview_dexycb", "", name)

    if rest.startswith("-views"):
        # Underscore-separated, matching the panoptic provider. A bare "-views12"
        # is two views (1 and 2), not view 12.
        match = re.match(r"-views((?:\d+_?)+)", rest)
        assert match is not None, f"Could not parse views from: {rest}"
        result["views_to_return"] = list(map(int, match.group(1).split("_")))
        rest = rest.replace(match.group(0), "", 1)

    if rest.startswith("-removehand"):
        result["remove_hand"] = True
        rest = rest.replace("-removehand", "", 1)

    assert rest == "", f"Unparsed portion of dataset name: {rest!r}"
    return result


class MultiviewDexYCBProvider(BaseDataProvider):
    """
    Provider for multiview DexYCB data.

    Flattened to a single time axis in time-major order:
    ``(n_views, n_frames, ...) -> (n_frames * n_views, ...)``, i.e. flat index
    ``t = f * n_views + v``; the model recovers the timestep index by integer
    division, so the ordering is part of the contract.

    ``views_to_return`` entries are positions in the sorted list of ``view_*``
    folders.
    """

    def __init__(
        self, cfg: DictConfig, name: str, override_anno: Optional[str] = None
    ):
        super().__init__(cfg, name, override_anno)
        self.data_root: str = cfg.data_root

        name_overrides = _parse_dexycb_name(name)
        self.views_to_return: Optional[List[int]] = name_overrides.get(
            "views_to_return",
            list(cfg.views_to_return) if cfg.get("views_to_return") else None,
        )
        self.remove_hand: bool = bool(
            name_overrides.get("remove_hand", cfg.get("remove_hand", False))
        )
        self.traj_per_sample: int = cfg.get("traj_per_sample", 384)
        self.seed: Optional[int] = cfg.get("seed", 72)
        max_videos: Optional[int] = cfg.get("max_videos", None)

        self.seq_names = self._get_sequence_names(max_videos)
        logger.info(
            "Loaded multiview DexYCB from %s with %d sequences (views=%s, remove_hand=%s)",
            self.data_root,
            len(self.seq_names),
            self.views_to_return,
            self.remove_hand,
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
        for seq_name in seq_names:
            scene_path = os.path.join(self.data_root, seq_name)
            has_views = any(
                os.path.isdir(os.path.join(scene_path, d)) and d.startswith("view_")
                for d in os.listdir(scene_path)
            )
            tracks_path = os.path.join(scene_path, "tracks_3d.npz")
            if not has_views or not os.path.exists(tracks_path):
                warnings.warn(f"Skipping {seq_name}: missing views or tracks_3d.npz")
                continue
            valid.append(seq_name)

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
        view_folders = sorted(
            [
                d
                for d in os.listdir(datapoint_path)
                if os.path.isdir(os.path.join(datapoint_path, d)) and d.startswith("view_")
            ]
        )
        views = (
            self.views_to_return
            if self.views_to_return is not None
            else sorted(list(range(len(view_folders))))
        )
        assert all(0 <= v < len(view_folders) for v in views), (
            f"views_to_return={views} out of range for {len(view_folders)} views "
            f"in {datapoint_path}"
        )
        n_views = len(views)

        # DexYCB sequences are short, so a longer request is clamped rather than
        # rejected.
        n_available = len(os.listdir(os.path.join(datapoint_path, view_folders[views[0]], "rgb")))
        length = min(length, n_available - start)
        frame_indices = list(range(start, start + length * stride, stride))
        assert frame_indices, f"no frames at start={start} in {seq_name}"
        n_frames = len(frame_indices)

        views_data: Dict[int, Dict[str, np.ndarray]] = {}
        for v in views:
            view_path = os.path.join(datapoint_path, view_folders[v])
            rgb_folder = os.path.join(view_path, "rgb")
            depth_folder = os.path.join(view_path, "depth")
            rgb_files = sorted(os.listdir(rgb_folder))
            depth_files = sorted(os.listdir(depth_folder))

            rgb_futures: List[Future] = []
            depth_futures: List[Future] = []
            for fi in frame_indices:
                rgb_futures.append(
                    executor.submit(
                        lambda p=os.path.join(rgb_folder, rgb_files[fi]): cv2.imread(p)[
                            :, :, ::-1
                        ].copy()
                    )
                )
                depth_futures.append(
                    executor.submit(
                        lambda p=os.path.join(depth_folder, depth_files[fi]): cv2.imread(
                            p, cv2.IMREAD_ANYDEPTH
                        ).copy()
                    )
                )

            params = np.load(os.path.join(view_path, "intrinsics_extrinsics.npz"))
            views_data[v] = {
                "rgb": np.stack([f.result() for f in rgb_futures]),
                "depth": np.stack([f.result() for f in depth_futures]),
                "intrinsics": params["intrinsics"][:3, :3].astype(np.float32),
                "extrinsics": params["extrinsics"][:3, :].astype(np.float32),
            }

        rgbs = np.stack([views_data[v]["rgb"] for v in views])  # (V, F, H, W, 3)
        n_views, n_frames, h, w, _ = rgbs.shape
        depths = (
            np.stack([views_data[v]["depth"] for v in views])[..., None].astype(np.float32)
            / 1000.0
        )  # (V, F, H, W, 1)
        intrs = np.stack([views_data[v]["intrinsics"] for v in views])[:, None].repeat(
            n_frames, axis=1
        )  # (V, F, 3, 3)
        extrs = np.stack([views_data[v]["extrinsics"] for v in views])[:, None].repeat(
            n_frames, axis=1
        )  # (V, F, 3, 4)

        tracks_3d_data = np.load(os.path.join(datapoint_path, "tracks_3d.npz"), allow_pickle=True)
        traj3d_world = tracks_3d_data["tracks_3d"][frame_indices].copy()  # (F, N, 3)
        traj2d = tracks_3d_data["tracks_2d"][views][:, frame_indices].copy()  # (V, F, N, 2)
        traj2d_w_z = np.concatenate(
            [
                traj2d,
                tracks_3d_data["tracks_2d_z"][views][:, frame_indices, :, None].copy(),
            ],
            axis=-1,
        )  # (V, F, N, 3)
        visibility = tracks_3d_data["tracks_2d_visibilities"][views][:, frame_indices].copy()

        object_id_to_name = tracks_3d_data["object_id_to_name"].item()
        traj_object_id = tracks_3d_data["object_ids"].copy()
        for object_name in object_id_to_name.values():
            assert object_name == "mano-right-hand" or object_name.startswith("ycb")

        avg_movement_per_object_id: Dict[int, float] = {}
        for object_id in np.unique(traj_object_id):
            object_mask = traj_object_id == object_id
            object_traj = traj3d_world[:, object_mask]
            avg_movement_per_object_id[int(object_id)] = float(
                np.linalg.norm(object_traj[1:] - object_traj[:-1], axis=-1).mean()
            )

        hand_id = {v: k for k, v in object_id_to_name.items()}["mano-right-hand"]
        non_hand_ids = [k for k in avg_movement_per_object_id.keys() if k != hand_id]
        dynamic_ycb_object_id = max(
            non_hand_ids, key=lambda k: avg_movement_per_object_id[k]
        )
        static_ycb_object_ids = [
            k for k in non_hand_ids if k != dynamic_ycb_object_id
        ]
        traj_object_id = (
            0 * (traj_object_id == hand_id)
            + 1 * (traj_object_id == dynamic_ycb_object_id)
            + 2 * np.isin(traj_object_id, static_ycb_object_ids)
        )

        if self.remove_hand:
            keep = traj_object_id > 0
            traj3d_world = traj3d_world[:, keep]
            traj2d = traj2d[:, :, keep]
            traj2d_w_z = traj2d_w_z[:, :, keep]
            visibility = visibility[:, :, keep]

        seed = self.seed if self.seed is not None else int(rng.integers(0, 2**32 - 1))
        rnd = np.random.RandomState(seed=seed)
        # Keep a track if the reference camera (view 0) sees it at least twice,
        # so the candidate pool does not grow with the view count.
        visible_for_at_least_two_frames = visibility[0].sum(axis=0) >= 2
        valid_track_ids = np.where(visible_for_at_least_two_frames)[0]

        n_select = min(self.traj_per_sample, len(valid_track_ids))
        assert n_select > 0, f"No valid tracks found for sequence {seq_name}"
        selected = valid_track_ids[rnd.permutation(len(valid_track_ids))[:n_select]]

        traj2d = traj2d[:, :, selected]
        traj2d_w_z = traj2d_w_z[:, :, selected]
        traj3d_world = traj3d_world[:, selected]
        visibility = visibility[:, :, selected]

        valids = ~np.isnan(traj2d).any(axis=-1)
        visibility = visibility.astype(bool) & valids

        traj2d = np.nan_to_num(traj2d, nan=0.0)
        traj2d_w_z = np.nan_to_num(traj2d_w_z, nan=0.0)
        traj3d_world = np.nan_to_num(traj3d_world, nan=0.0)

        scale = 6.0
        rot_x = Rotation.from_euler("x", 220, degrees=True).as_matrix()
        rot_y = Rotation.from_euler("y", 3, degrees=True).as_matrix()
        rot_z = Rotation.from_euler("z", -30, degrees=True).as_matrix()
        rot = rot_z @ rot_y @ rot_x
        translation = np.array([0.0, 0.0, 0.5], dtype=np.float64)

        Rt = np.eye(4, dtype=np.float64)
        Rt[:3, :3] = rot
        Rt[:3, 3] = translation
        Rt_inv = np.linalg.inv(Rt)

        depths = depths * scale
        traj2d_w_z[..., 2] *= scale

        extrs_44 = np.eye(4, dtype=np.float64)[None, None].repeat(n_views, axis=0).repeat(
            n_frames, axis=1
        )
        extrs_44[:, :, :3, :] = extrs.astype(np.float64)
        extrs_44[:, :, :3, 3] *= scale
        extrs_44 = np.einsum("VFki,ij->VFkj", extrs_44, Rt_inv)

        traj3d_scaled = traj3d_world.astype(np.float64) * scale
        traj3d_homo = np.concatenate(
            [traj3d_scaled, np.ones_like(traj3d_scaled[..., :1])], axis=-1
        )
        traj3d_trans = np.einsum("ij,FNj->FNi", Rt, traj3d_homo)[..., :3]

        # ── Flatten multiview time-major: (V, F, ...) → (F*V, ...) ──
        rgbs_flat = rearrange(rgbs, "v f h w c -> (f v) h w c")
        depths_flat = rearrange(depths, "v f h w c -> (f v) h w c")[..., 0].astype(
            np.float32
        )
        intrs_flat = rearrange(intrs, "v f i j -> (f v) i j").astype(np.float32)
        extrs_flat = rearrange(extrs_44, "v f i j -> (f v) i j").astype(np.float32)
        traj2d_flat = rearrange(traj2d, "v f n c -> (f v) n c").astype(np.float32)
        trajs_depths_flat = rearrange(traj2d_w_z[..., 2], "v f n -> (f v) n").astype(
            np.float32
        )
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
        seq_lens: List[int] = []
        for seq_name in self.seq_names:
            datapoint_path = os.path.join(self.data_root, seq_name)
            view_folders = sorted(
                [
                    d
                    for d in os.listdir(datapoint_path)
                    if os.path.isdir(os.path.join(datapoint_path, d)) and d.startswith("view_")
                ]
            )
            if not view_folders:
                seq_lens.append(0)
                continue
            rgb_folder = os.path.join(datapoint_path, view_folders[0], "rgb")
            seq_lens.append(len(sorted(os.listdir(rgb_folder))))
        return seq_lens
