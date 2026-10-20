# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
HuggingFace-compatible inference for TrackEverything.

Usage:
    from trackeverything.demo_hf import TrackEverything

    # Load model from HuggingFace
    model = TrackEverything.from_pretrained("katefgroup/TrackEverything")

    # Run inference on a video (geometry backend estimates depth/poses automatically)
    results = model.infer(video="path/to/video.mp4")

    # Run inference on a folder of images
    results = model.infer(video="path/to/image_folder/")

    # Provide your own geometry (skip Pi3 / VGGT-Omega)
    results = model.infer(
        video="path/to/video.mp4",
        depths=depths_array,          # (T, H, W) numpy or torch
        intrinsics=intrinsics_array,  # (T, 3, 3) numpy or torch
        extrinsics=extrinsics_array,  # (T, 4, 4) numpy or torch
    )

    # With 2D query points (unprojected to 3D automatically)
    results = model.infer(
        video="path/to/video.mp4",
        query_points_2d=[[0, 100, 200], [5, 150, 300]],  # [t, x, y]
    )

    # One track per scene point (linear in T); expand to per-pixel on demand
    tracks = results.tracks()          # tracks.coords: (N, T, 3)
    dense_coords, dense_visibs, dense_dyn = tracks.to_dense()

CLI (saves tracks; use demo.py to also visualize):
    python -m trackeverything.demo_hf --video path/to/video.mp4 --output_dir outputs/
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Data loading utilities
# --------------------------------------------------------------------------- #


def _read_video(video_path: str) -> np.ndarray:
    """Read video file into (T, H, W, 3) uint8 RGB array."""
    import av

    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames)


def _read_images_from_folder(folder_path: str) -> np.ndarray:
    """Load image sequence from directory, sorted by filename. Returns (T, H, W, 3) uint8 RGB."""
    folder = Path(folder_path).resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder_path}")
    paths = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS],
        key=lambda p: p.name,
    )
    if not paths:
        raise FileNotFoundError(
            f"No images with extensions {sorted(_IMAGE_EXTS)} found under {folder}"
        )
    frames = []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            logger.warning("Skipping unreadable image: %s", p)
            continue
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not frames:
        raise RuntimeError(f"Could not decode any images in {folder}")
    return np.stack(frames)


def _download_video(url: str, cache_dir: Optional[str] = None) -> str:
    """Download a video from a URL and return the local path."""
    import tempfile
    import urllib.request

    if cache_dir is None:
        cache_dir = tempfile.mkdtemp()
    out_path = Path(cache_dir) / Path(url).name
    if not out_path.exists():
        logger.info("Downloading video from %s ...", url)
        urllib.request.urlretrieve(url, str(out_path))
    return str(out_path)


def _load_video(video: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
    """Resolve video input to (T, H, W, 3) uint8 RGB numpy array."""
    if isinstance(video, torch.Tensor):
        if video.ndim == 4 and video.shape[1] == 3:
            video = video.permute(0, 2, 3, 1)
        arr = video.cpu().numpy()
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return arr
    if isinstance(video, np.ndarray):
        if video.dtype != np.uint8:
            video = (video * 255).clip(0, 255).astype(np.uint8)
        return video
    path_str = str(video)
    if path_str.startswith("http://") or path_str.startswith("https://"):
        path_str = _download_video(path_str)
    p = Path(path_str)
    if p.is_dir():
        return _read_images_from_folder(path_str)
    return _read_video(path_str)


# --------------------------------------------------------------------------- #
# Geometry utilities
# --------------------------------------------------------------------------- #


def _resize_depth_bilinear(depth: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
    is_valid = (depth > 0).astype(np.float32)
    depth_resized = cv2.resize(depth, new_shape, interpolation=cv2.INTER_LINEAR)
    is_valid_resized = cv2.resize(is_valid, new_shape, interpolation=cv2.INTER_LINEAR)
    depth_resized = depth_resized / (is_valid_resized + 1e-6)
    depth_resized[is_valid_resized <= 1e-6] = 0.0
    return depth_resized


def _normalize_scene(
    depths: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    norm_mask: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Normalize scene scale using median distance (median_dis mode)."""
    depths_s = depths[:, ::2, ::2]
    mask_s = norm_mask[:, ::2, ::2]
    intr_s = intrinsics.copy()
    intr_s[:, :2, :] /= 2.0

    T, H, W = depths_s.shape
    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij"
    )
    uv1 = np.stack([xx, yy, np.ones_like(xx)], axis=-1)

    inv_K = np.linalg.inv(intr_s)
    rays_cam = np.einsum("tij,hwj->thwi", inv_K, uv1)
    pts_cam = rays_cam * depths_s[..., None]
    pts_cam_h = np.concatenate([pts_cam, np.ones_like(pts_cam[..., :1])], axis=-1)
    c2w = np.linalg.inv(extrinsics)
    pts_world = np.einsum("tij,thwj->thwi", c2w, pts_cam_h)[..., :3]

    valid = (depths_s > 0) & mask_s
    dists = np.linalg.norm(pts_world.reshape(-1, 3), axis=-1)
    valid_flat = valid.reshape(-1)
    if valid_flat.sum() == 0:
        norm_factor = 1.0
    else:
        norm_factor = float(np.median(dists[valid_flat]))
        if not np.isfinite(norm_factor) or norm_factor < 1e-8:
            norm_factor = 1.0

    depths = depths / norm_factor
    extrinsics = extrinsics.copy()
    extrinsics[..., :3, 3] /= norm_factor
    return norm_factor, depths, extrinsics


def _unproject_2d_to_3d(
    query_points_2d: np.ndarray,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> torch.Tensor:
    """Unproject 2D query points [t, x, y] to 3D world coordinates [t, X, Y, Z].

    Args:
        query_points_2d: (N, 3) array with columns [frame_idx, x_pixel, y_pixel]
        depths: (T, H, W) tensor of depth maps
        intrinsics: (T, 3, 3) tensor of camera intrinsics
        extrinsics: (T, 4, 4) tensor of world-to-camera extrinsics

    Returns:
        query_points_3d: (1, N, 4) tensor with columns [t, X, Y, Z] in world frame
    """
    device = depths.device
    N = query_points_2d.shape[0]
    H, W = depths.shape[1], depths.shape[2]

    queries_3d = []
    for i in range(N):
        t = int(query_points_2d[i, 0])
        x = float(query_points_2d[i, 1])
        y = float(query_points_2d[i, 2])

        xi = int(round(x))
        yi = int(round(y))
        xi = max(0, min(xi, W - 1))
        yi = max(0, min(yi, H - 1))

        z = depths[t, yi, xi].item()
        if z <= 0:
            z = depths[t, max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3]
            valid = z[z > 0]
            z = valid.mean().item() if len(valid) > 0 else 1.0

        uv_h = torch.tensor([x, y, 1.0], device=device)
        inv_K = torch.linalg.inv(intrinsics[t])
        ray = inv_K @ uv_h
        pt_cam = ray * z
        pt_cam_h = torch.cat([pt_cam, torch.ones(1, device=device)])
        c2w = torch.linalg.inv(extrinsics[t])
        pt_world = (c2w @ pt_cam_h)[:3]

        queries_3d.append(torch.cat([torch.tensor([float(t)], device=device), pt_world]))

    return torch.stack(queries_3d, dim=0).unsqueeze(0)


# --------------------------------------------------------------------------- #
# Output dataclass
# --------------------------------------------------------------------------- #

from .training.datatypes import Prediction  # noqa: E402
from .utils.alltrack3r_utils import SceneTokens  # noqa: E402


@dataclass
class TrackEverythingOutput:
    """Output container for TrackEverything inference.

    The model's native output is ``prediction``: per-window track snapshots
    whose size grows linearly with the number of frames T. Two views are built
    from it on demand:

    * :meth:`tracks` returns one track per distinct scene point, plus the map
      from every pixel to its track. Size grows linearly with T; this is what
      most users want. ``tracks().to_dense()`` expands it to per-pixel tracks.
    * ``coords`` / ``visibs`` / ``dynamic_logits`` give every source pixel's
      exact position at every target frame. They are built on first access and
      grow with T squared, so only touch them on short clips.

    Attributes:
        prediction: Per-window snapshots (a slim ``Prediction`` carrying
            ``dense_track_3d``, ``dense_shape``, ``dense_extrinsics`` and
            ``dense_intrinsics``). ``None`` in sparse mode.
        query_coords: (B, T, N, 3) 3D positions of query points across time
            (only if query_points_2d was provided).
        query_visibs: (B, T, N) visibility of query points across time
            (only if query_points_2d was provided).
        norm_factor: Scene normalization factor applied to depths/extrinsics.
        depths: Preprocessed depth maps (T, H, W) at inference resolution.
        conf_mask: Geometry-backend per-pixel confidence (T, H, W) in ``[0, 1]``
            at the same resolution as ``depths``. ``None`` when geometry was
            supplied without a confidence map, or the backend returned none.
        intrinsics: Camera intrinsics (T, 3, 3) at inference resolution.
        extrinsics: Camera extrinsics (T, 4, 4) after cam0-alignment and normalization.
        rgb: Preprocessed RGB video (T, 3, H, W) float [0, 1] at inference resolution.
        dense_source_mask: Dense source validity mask at inference resolution after
            the 0.25x nearest downsample used for ``dense_norm_mask`` (T_src, H', W').
        voxel_size: Voxel size the model ran with.
        dynamic_threshold: Static/dynamic threshold the model ran with.
    """

    prediction: Optional["Prediction"] = None
    query_coords: Optional[torch.Tensor] = None
    query_visibs: Optional[torch.Tensor] = None
    norm_factor: float = 1.0
    depths: Optional[torch.Tensor] = None
    conf_mask: Optional[torch.Tensor] = None
    intrinsics: Optional[torch.Tensor] = None
    extrinsics: Optional[torch.Tensor] = None
    rgb: Optional[torch.Tensor] = None
    dense_source_mask: Optional[torch.Tensor] = None
    voxel_size: float = 0.02
    dynamic_threshold: float = 0.5
    _dense: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default=None, repr=False
    )

    def tracks(self, out_device: str = "cpu") -> SceneTokens:
        """One track per distinct scene point, as merged by the tracker.

        Returns a ``SceneTokens`` with, for N points over T frames: ``coords``
        (N, T, 3) world positions, ``visibs`` (N, T) visibility logits,
        ``dyn_logits`` (N, T) static/dynamic logits, ``src_time`` / ``src_y`` /
        ``src_x`` (N,) the frame and grid cell where each point first appears,
        and ``cell_to_token`` (T, H', W') the point each pixel of each frame
        belongs to. Entries before a point first appears are -1000.

        ``tracks().to_dense()`` expands this to per-pixel tracks with the layout
        of ``coords`` / ``visibs`` / ``dynamic_logits``. Pixels share their
        point's track, so positions are exact up to the voxel size; use the
        ``coords`` property for the exact dense tensor while the snapshots are
        still around.
        """
        if self.prediction is None:
            raise ValueError("tracks() needs dense mode (prediction is None)")
        from .utils.alltrack3r_utils import extract_merge_history_tokens

        return extract_merge_history_tokens(self.prediction, out_device=out_device)

    def _dense_tracks(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self._dense is None and self.prediction is not None:
            from .utils.alltrack3r_utils import reconstruct_dense_tracks_3d

            dense, dense_visibs, _confs, dense_logits = reconstruct_dense_tracks_3d(
                self.prediction,
                return_visibs_confs=True,
                dynamic_logits_by_target=True,
            )
            self._dense = (dense, dense_visibs, dense_logits)
        return self._dense

    @property
    def coords(self) -> Optional[torch.Tensor]:
        """(B, T_src, H, W, T_tgt, 3) world position of every source pixel at
        every target frame. Built on first access; quadratic in T."""
        dense = self._dense_tracks()
        return None if dense is None else dense[0]

    @property
    def visibs(self) -> Optional[torch.Tensor]:
        """(B, T_src, H, W, T_tgt) visibility. Built on first access."""
        dense = self._dense_tracks()
        return None if dense is None else dense[1]

    @property
    def dynamic_logits(self) -> Optional[torch.Tensor]:
        """(B, T_src, H, W, T_tgt) static/dynamic logits (higher = dynamic).
        Built on first access."""
        dense = self._dense_tracks()
        return None if dense is None else dense[2]


# --------------------------------------------------------------------------- #
# Main model class
# --------------------------------------------------------------------------- #


class TrackEverything:
    """HuggingFace-compatible inference wrapper for TrackEverything.

    Handles:
        - Loading model from HuggingFace Hub or local checkpoint
        - Running Pi3 or VGGT-Omega for geometry estimation (if depths/poses not provided)
        - Unprojecting 2D query points to 3D
        - Scene normalization and preprocessing
        - Model inference
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        geometry_backend: Literal["pi3", "vggt_omega"] = "vggt_omega",
        pi3_checkpoint: str = "yyfz233/Pi3",
        pi3_chunk_size: int = 60,
        pi3_overlap: int = 30,
        vggt_omega_checkpoint: Optional[str] = None,
        vggt_omega_chunk_size: int = 60,
        vggt_omega_overlap: int = 30,
        vggt_omega_image_resolution: int = 512,
    ):
        from .annotation.geometry_factory import DEFAULT_VGGT_OMEGA_CHECKPOINT

        self.model = model
        self.device = device
        self._geometry_backend = geometry_backend
        self._pi3_checkpoint = pi3_checkpoint
        self._pi3_chunk_size = pi3_chunk_size
        self._pi3_overlap = pi3_overlap
        self._vggt_omega_checkpoint = (
            vggt_omega_checkpoint
            if vggt_omega_checkpoint is not None
            else DEFAULT_VGGT_OMEGA_CHECKPOINT
        )
        self._vggt_omega_chunk_size = vggt_omega_chunk_size
        self._vggt_omega_overlap = vggt_omega_overlap
        self._vggt_omega_image_resolution = vggt_omega_image_resolution
        self._geometry = None

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str = "katefgroup/TrackEverything",
        *,
        device: str = "cuda",
        geometry_backend: Literal["pi3", "vggt_omega"] = "vggt_omega",
        pi3_checkpoint: str = "yyfz233/Pi3",
        pi3_chunk_size: int = 60,
        pi3_overlap: int = 30,
        vggt_omega_checkpoint: Optional[str] = None,
        vggt_omega_chunk_size: int = 60,
        vggt_omega_overlap: int = 30,
        vggt_omega_image_resolution: int = 512,
        revision: Optional[str] = None,
        token: Optional[str] = None,
    ) -> "TrackEverything":
        """Load TrackEverything from HuggingFace Hub or local checkpoint.

        Args:
            repo_id_or_path: HuggingFace repo ID (e.g. "katefgroup/TrackEverything")
                or local path to a .pth checkpoint file.
            device: Device to load model on ("cuda", "cpu", etc.).
            geometry_backend: "vggt_omega" (default) or "pi3" for automatic geometry.
            pi3_checkpoint: HuggingFace repo for Pi3 geometry model.
            pi3_chunk_size: Chunk size for Pi3 long-video processing.
            pi3_overlap: Overlap between Pi3 chunks.
            vggt_omega_checkpoint: Local path to VGGT-Omega .pt weights.
            vggt_omega_chunk_size: Chunk size for VGGT-Omega long-video processing.
            vggt_omega_overlap: Overlap between VGGT-Omega chunks.
            vggt_omega_image_resolution: Input resolution for VGGT-Omega.
            revision: Optional git revision for the HuggingFace repo.
            token: Optional HuggingFace API token for private repos.

        Returns:
            TrackEverything instance ready for inference.
        """
        from . import models
        from .models.track_everything import TrackEverything as TrackEverythingModel

        ckpt_path = Path(repo_id_or_path)
        if ckpt_path.exists() and ckpt_path.is_file():
            logger.info("Loading model from local checkpoint: %s", ckpt_path)
            model, _cfg = models.from_pretrained(str(ckpt_path))
        else:
            from huggingface_hub import hf_hub_download

            logger.info("Downloading model from HuggingFace Hub: %s", repo_id_or_path)
            local_path = hf_hub_download(
                repo_id=repo_id_or_path,
                filename="model.pth",
                revision=revision,
                token=token,
            )
            model, _cfg = models.from_pretrained(local_path)

        assert isinstance(model, TrackEverythingModel), (
            f"Expected TrackEverything model, got {type(model).__name__}"
        )
        if hasattr(model, "eval_mode"):
            model.set_eval_mode("raw")
        model.eval()
        model.to(device)

        return cls(
            model=model,
            device=device,
            geometry_backend=geometry_backend,
            pi3_checkpoint=pi3_checkpoint,
            pi3_chunk_size=pi3_chunk_size,
            pi3_overlap=pi3_overlap,
            vggt_omega_checkpoint=vggt_omega_checkpoint,
            vggt_omega_chunk_size=vggt_omega_chunk_size,
            vggt_omega_overlap=vggt_omega_overlap,
            vggt_omega_image_resolution=vggt_omega_image_resolution,
        )

    @property
    def geometry(self):
        """Lazy-load geometry estimator (Pi3 or VGGT-Omega)."""
        if self._geometry is None:
            from .annotation.geometry_factory import build_geometry_annotator

            logger.info(
                "Loading %s geometry estimator", self._geometry_backend
            )
            self._geometry = build_geometry_annotator(
                self._geometry_backend,
                device=self.device,
                pi3_checkpoint=self._pi3_checkpoint,
                pi3_chunk_size=self._pi3_chunk_size,
                pi3_overlap=self._pi3_overlap,
                vggt_omega_checkpoint=self._vggt_omega_checkpoint,
                vggt_omega_chunk_size=self._vggt_omega_chunk_size,
                vggt_omega_overlap=self._vggt_omega_overlap,
                vggt_omega_image_resolution=self._vggt_omega_image_resolution,
            )
        return self._geometry

    @property
    def pi3(self):
        """Backward-compatible alias for ``geometry`` when backend is Pi3."""
        return self.geometry

    def _estimate_geometry(
        self, video: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run geometry backend to estimate depths, intrinsics, extrinsics, conf masks.

        Args:
            video: (T, H, W, 3) uint8 RGB

        Returns:
            depths: (T, H, W) float32
            intrinsics: (T, 3, 3) float32
            extrinsics: (T, 4, 4) float32
            conf_mask: (T, H, W) float32
        """
        logger.info(
            "Estimating geometry with %s (%d frames)...",
            self._geometry_backend,
            video.shape[0],
        )
        depths, intrinsics, extrinsics, conf_mask = self.geometry.process_video(video)
        logger.info("Geometry estimation complete.")
        return depths, intrinsics, extrinsics, conf_mask

    def _preprocess(
        self,
        video: np.ndarray,
        depths: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        conf_mask: Optional[np.ndarray] = None,
        num_threads: int = 8,
        use_input_resolution: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Filter, normalize, and convert to tensors.

        By default resizes RGB/depths/conf/intrinsics to ``model.image_size``.
        With ``use_input_resolution=True``, keeps the input video resolution and
        asserts depths (and conf) already match that ``(H, W)``.

        Returns all tensors on self.device plus the norm_factor:
            video: (T, 3, H', W') float [0,1]
            depths: (T, H', W') float
            intrinsics: (T, 3, 3) float
            extrinsics: (T, 4, 4) float
            conf_mask: (T, H', W') float
            norm_factor: float
        """
        from .datasets.data_ops import _filter_one_depth

        original_res = video.shape[1:3]
        H, W = int(original_res[0]), int(original_res[1])

        if use_input_resolution:
            assert H % 4 == 0 and W % 4 == 0, (
                f"use_input_resolution requires H,W divisible by 4 (encoder stride); "
                f"got ({H},{W})"
            )
            assert depths.shape[-2:] == (H, W), (
                f"use_input_resolution: depths spatial shape {depths.shape[-2:]} "
                f"must match video ({H},{W}); geometry backends should return "
                "depths at the input RGB resolution"
            )
            if conf_mask is not None:
                assert conf_mask.shape[-2:] == (H, W), (
                    f"use_input_resolution: conf_mask spatial shape "
                    f"{conf_mask.shape[-2:]} must match video ({H},{W})"
                )
        else:
            inference_res = self.model.image_size
            intrinsics = np.asarray(intrinsics, dtype=np.float32).copy()
            intrinsics[:, 0, :] *= (inference_res[1] - 1) / (original_res[1] - 1)
            intrinsics[:, 1, :] *= (inference_res[0] - 1) / (original_res[0] - 1)

            with ThreadPoolExecutor(num_threads) as executor:
                video_futures = [
                    executor.submit(
                        cv2.resize, rgb, (inference_res[1], inference_res[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for rgb in video
                ]
                depth_futures = [
                    executor.submit(
                        _resize_depth_bilinear, depth, (inference_res[1], inference_res[0])
                    )
                    for depth in depths
                ]
                video = np.stack([f.result() for f in video_futures])
                depths = np.stack([f.result() for f in depth_futures])

                if conf_mask is not None:
                    # Plain bilinear: _resize_depth_bilinear renormalizes by the
                    # valid fraction, which turns a 0/1 mask into a dilated 0/1
                    # mask (every pixel touching a valid one becomes exactly
                    # 1.0), defeating any threshold.
                    conf_futures = [
                        executor.submit(
                            cv2.resize, cm, (inference_res[1], inference_res[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        for cm in conf_mask
                    ]
                    conf_mask = np.stack([f.result() for f in conf_futures])

        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        with ThreadPoolExecutor(num_threads) as executor:
            filter_futures = [
                executor.submit(_filter_one_depth, d, 0.08, 15.0, k)
                for d, k in zip(depths, intrinsics)
            ]
            depths = np.stack([f.result() for f in filter_futures])

        # VGGT-omega already emits unit-scale geometry in the cam0 frame.
        if self._geometry_backend != "vggt_omega":
            # Cam-0 alignment
            cam0_ext_inv = np.linalg.inv(extrinsics[0])
            extrinsics = np.einsum("bij,jk->bik", extrinsics, cam0_ext_inv)

            # Scene normalization: median distance over depth in (0.1, 15) and conf > 0.5
            est_norm_mask = (depths > 0.1) & (depths < 15.0)
            if conf_mask is not None:
                est_norm_mask = est_norm_mask & (conf_mask > 0.5)
            norm_factor, depths, extrinsics = _normalize_scene(
                depths, intrinsics, extrinsics, est_norm_mask
            )
        else:
            norm_factor = 1.0

        # Convert to tensors
        video_t = (
            torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
        ).to(self.device)
        depths_t = torch.from_numpy(depths).float().to(self.device)
        intrinsics_t = torch.from_numpy(intrinsics).float().to(self.device)
        extrinsics_t = torch.from_numpy(extrinsics).float().to(self.device)

        if conf_mask is not None:
            conf_mask_t = torch.from_numpy(conf_mask).float().to(self.device)
        else:
            conf_mask_t = torch.ones_like(depths_t)

        return video_t, depths_t, intrinsics_t, extrinsics_t, conf_mask_t, norm_factor

    def infer(
        self,
        video: Union[str, np.ndarray, torch.Tensor],
        *,
        depths: Optional[Union[np.ndarray, torch.Tensor]] = None,
        intrinsics: Optional[Union[np.ndarray, torch.Tensor]] = None,
        extrinsics: Optional[Union[np.ndarray, torch.Tensor]] = None,
        conf_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        query_points_2d: Optional[Union[np.ndarray, torch.Tensor, List[List[float]]]] = None,
        mode: Literal["dense", "sparse"] = "dense",
        num_iters: int = 3,
        voxel_size: float = 0.02,
        voxel_mode: str = "uniform",
        voxelize: bool = True,
        voxelize_on_endpoints: bool = True,
        compress_memory: Optional[int] = None,
        dynamic_threshold: float = 0.5,
        use_conf_mask: bool = False,
        start_frame: int = 0,
        max_frames: Optional[int] = None,
        frame_stride: int = 1,
        num_threads: int = 8,
        mem_efficient: bool = True,
        use_input_resolution: bool = False,
    ) -> TrackEverythingOutput:
        """Run TrackEverything inference.

        Args:
            video: Path to video file, URL, path to image folder, or numpy/torch array
                of shape (T, H, W, 3) in uint8 RGB or (T, 3, H, W) float [0,1].
            depths: Optional (T, H, W) depth maps. If provided along with intrinsics
                and extrinsics, geometry estimation is skipped.
            intrinsics: Optional (T, 3, 3) camera intrinsic matrices.
            extrinsics: Optional (T, 4, 4) world-to-camera extrinsic matrices.
            conf_mask: Optional (T, H, W) geometry confidence in ``[0, 1]``. Used
                when depths/poses are supplied so the output still carries the
                backend confidence; ignored when geometry is estimated here.
            query_points_2d: Optional (N, 3) array of 2D query points with format
                [frame_idx, x_pixel, y_pixel]. These are unprojected to 3D using the
                estimated or provided geometry.
            mode: "dense" for dense tracking, "sparse" for query-point-only tracking.
            num_iters: Number of refinement iterations.
            voxel_size: Voxel size for scene discretization.
            voxel_mode: ``uniform`` (Cartesian ``round(x/v)``) or ``signedlog``.
            voxelize: Merge scene points that land in the same voxel into one
                memory token (the de-duplication). Off keeps one token per pixel
                per frame, which is far slower and heavier.
            voxelize_on_endpoints: After the first refinement iteration, also
                merge points whose predicted positions share a voxel, so later
                iterations run on fewer tokens.
            compress_memory: If set, cap the static (never-dynamic) scene memory
                at roughly this many voxels by re-voxelizing it at 2x, then 4x,
                the voxel size. Dynamic points keep full resolution.
            dynamic_threshold: Threshold for static/dynamic classification.
            use_conf_mask: Whether to use geometry confidence mask for filtering.
            start_frame: Index of the first frame to process.
            max_frames: Limit number of frames processed (None = all).
            frame_stride: Subsample frames by this stride.
            num_threads: Number of threads for parallel preprocessing.
            mem_efficient: Run the model in memory-efficient mode (keeps only
                per-window snapshots instead of the full dense tensor).
            use_input_resolution: If True, skip resizing RGB/depths/intrinsics to
                ``model.image_size`` and run at the input video ``(H, W)``. Depths
                must already match that resolution (geometry backends map back to
                the input RGB size). ``H`` and ``W`` must be divisible by 4.

        Returns:
            TrackEverythingOutput. Use ``output.tracks()`` for a flat track set;
            the dense ``coords`` / ``visibs`` / ``dynamic_logits`` are built only
            when accessed.
        """
        if start_frame < 0:
            raise ValueError(f"start_frame must be >= 0, got {start_frame}")
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
        frames = slice(start_frame, None, frame_stride)

        # Load video
        video_np = _load_video(video)[frames]
        if max_frames is not None:
            video_np = video_np[:max_frames]
        if len(video_np) == 0:
            raise ValueError(
                f"No frames left after start_frame={start_frame}, frame_stride={frame_stride}"
            )

        logger.info(
            "Input video: %d frames at %dx%d (use_input_resolution=%s)",
            video_np.shape[0],
            video_np.shape[1],
            video_np.shape[2],
            use_input_resolution,
        )

        # Determine whether to run geometry backend or use provided geometry
        has_geometry = (
            depths is not None and intrinsics is not None and extrinsics is not None
        )

        if has_geometry:
            logger.info("Using provided geometry (skipping %s).", self._geometry_backend)
            depths_np = depths.cpu().numpy() if isinstance(depths, torch.Tensor) else np.asarray(depths)
            intr_np = intrinsics.cpu().numpy() if isinstance(intrinsics, torch.Tensor) else np.asarray(intrinsics)
            extr_np = extrinsics.cpu().numpy() if isinstance(extrinsics, torch.Tensor) else np.asarray(extrinsics)

            depths_np = depths_np[frames]
            intr_np = intr_np[frames]
            extr_np = extr_np[frames]
            if max_frames is not None:
                depths_np = depths_np[:max_frames]
                intr_np = intr_np[:max_frames]
                extr_np = extr_np[:max_frames]

            if conf_mask is not None:
                conf_mask_np = (
                    conf_mask.cpu().numpy()
                    if isinstance(conf_mask, torch.Tensor)
                    else np.asarray(conf_mask)
                )
                conf_mask_np = conf_mask_np[frames]
                if max_frames is not None:
                    conf_mask_np = conf_mask_np[:max_frames]
            else:
                conf_mask_np = None
        else:
            if depths is not None or intrinsics is not None or extrinsics is not None:
                logger.warning(
                    "Partial geometry provided. All of depths, intrinsics, and extrinsics "
                    "are required to skip geometry estimation. Running %s instead.",
                    self._geometry_backend,
                )
            depths_np, intr_np, extr_np, conf_mask_np = self._estimate_geometry(video_np)

        # Ensure shapes match
        assert video_np.shape[0] == depths_np.shape[0], (
            f"Frame count mismatch: video={video_np.shape[0]}, depths={depths_np.shape[0]}"
        )

        # Preprocess
        video_t, depths_t, intr_t, extr_t, conf_mask_t, norm_factor = self._preprocess(
            video_np, depths_np, intr_np, extr_np,
            conf_mask=conf_mask_np,
            num_threads=num_threads,
            use_input_resolution=use_input_resolution,
        )

        # Configure model for the requested mode
        if mode == "dense":
            self.model.dense_tracking_at_eval = True
            self.model.use_query_points = False
        else:
            self.model.use_only_query_points = True
            self.model.use_query_points = True

        self.model.mem_efficient = mem_efficient
        self.model.voxel_size = voxel_size
        self.model.voxel_mode = str(voxel_mode or "uniform")
        self.model.voxelize = voxelize
        self.model.voxelize_on_endpoints = voxelize_on_endpoints
        self.model.compress_memory = compress_memory
        self.model.dynamic_threshold = dynamic_threshold

        # Build query points
        if query_points_2d is not None:
            if isinstance(query_points_2d, list):
                query_points_2d = np.array(query_points_2d, dtype=np.float32)
            elif isinstance(query_points_2d, torch.Tensor):
                query_points_2d = query_points_2d.cpu().numpy()

            # Scale query point coordinates to inference resolution (identity when
            # use_input_resolution keeps pixels in the input video frame).
            original_res = video_np.shape[1:3]
            if use_input_resolution:
                query_points_2d_scaled = query_points_2d
            else:
                inference_res = self.model.image_size
                scale_x = (inference_res[1] - 1) / (original_res[1] - 1)
                scale_y = (inference_res[0] - 1) / (original_res[0] - 1)
                query_points_2d_scaled = query_points_2d.copy()
                query_points_2d_scaled[:, 1] *= scale_x
                query_points_2d_scaled[:, 2] *= scale_y

            query_point = _unproject_2d_to_3d(
                query_points_2d_scaled, depths_t, intr_t, extr_t
            )
            self.model.use_query_points = True
        else:
            query_point = torch.zeros((1, 0, 4), device=self.device)
            if mode == "dense":
                self.model.use_query_points = False

        # Build dense normalization mask
        dense_norm_mask = (depths_t > 0.1) & (depths_t < 15)
        if use_conf_mask and conf_mask_np is not None:
            dense_norm_mask = dense_norm_mask & (conf_mask_t > 0.5)
        dense_norm_mask = dense_norm_mask[None, ..., None]

        dense_norm_mask = F.interpolate(
            dense_norm_mask.squeeze(-1).flatten(0, 1).unsqueeze(1).float(),
            scale_factor=0.25,
            mode="nearest-exact",
        ).squeeze(1).bool()
        dense_norm_mask = dense_norm_mask[None, ..., None]

        # Run inference
        logger.info("Running model inference (%d frames, mode=%s)...", video_t.shape[0], mode)
        with torch.no_grad():
            preds, _ = self.model(
                rgb_obs=video_t[None],
                depth_obs=depths_t[None],
                intrinsics=intr_t[None],
                extrinsics=extr_t[None],
                query_point=query_point,
                num_iters=num_iters,
                dense_norm_mask=dense_norm_mask,
            )
        logger.info("Inference complete.")

        # Build output
        output = TrackEverythingOutput(
            norm_factor=norm_factor,
            depths=depths_t,
            conf_mask=conf_mask_t if conf_mask_np is not None else None,
            intrinsics=intr_t,
            extrinsics=extr_t,
            rgb=video_t,
            dense_source_mask=dense_norm_mask[0, ..., 0],
            voxel_size=voxel_size,
            dynamic_threshold=dynamic_threshold,
        )

        if mode == "dense" and preds.dense_track_3d is not None:
            output.prediction = Prediction(
                dense_track_3d=preds.dense_track_3d,
                dense_shape=preds.dense_shape,
                dense_extrinsics=preds.dense_extrinsics,
                dense_intrinsics=preds.dense_intrinsics,
            )

        if query_points_2d is not None:
            output.query_coords = preds.query_coords
            output.query_visibs = preds.query_visibs

        return output


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #


def _str2bool(v: Union[str, bool]) -> bool:
    """Parse a CLI bool from an explicit true/false string."""
    import argparse

    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {v!r}")


def add_inference_args(parser) -> None:
    """Arguments shared by this CLI and ``demo.py``."""
    import argparse

    from .annotation.geometry_factory import DEFAULT_VGGT_OMEGA_CHECKPOINT

    parser.add_argument(
        "--model", type=str, default="katefgroup/TrackEverything",
        help="HuggingFace repo ID or local checkpoint path",
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to video file, URL, or folder of images",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace API token")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--num_iters", type=int, default=3)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument(
        "--voxel_mode",
        type=str,
        default="uniform",
        choices=["uniform", "signedlog"],
        help="Scene discretization: uniform Cartesian or signedlog",
    )
    parser.add_argument(
        "--voxelize", type=_str2bool, nargs="?", const=True, default=True,
        help="Merge scene points sharing a voxel into one memory token (true/false, default: true)",
    )
    parser.add_argument(
        "--voxelize_on_endpoints", type=_str2bool, nargs="?", const=True, default=True,
        help="After iteration 0, also merge points whose predicted positions share a voxel "
        "(true/false, default: true)",
    )
    parser.add_argument(
        "--compress_memory", type=int, default=None,
        help="Cap static scene memory at ~N voxels by coarsening it 2x/4x (default: off)",
    )
    parser.add_argument("--dynamic_threshold", type=float, default=0.5)
    parser.add_argument("--use_conf_mask", action="store_true")
    parser.add_argument(
        "--mem_efficient", action=argparse.BooleanOptionalAction, default=True,
        help="Run the model in memory-efficient mode (default: enabled)",
    )

    # Optional geometry inputs (skip the geometry backend when all three are given)
    parser.add_argument("--depths_dir", type=str, default=None, help="Directory of .npy depth files")
    parser.add_argument("--intrinsics_path", type=str, default=None, help="Path to intrinsics .npy (T,3,3)")
    parser.add_argument("--extrinsics_path", type=str, default=None, help="Path to extrinsics .npy (T,4,4)")

    # Geometry backend config
    parser.add_argument(
        "--geometry_backend",
        type=str,
        default="vggt_omega",
        choices=["pi3", "vggt_omega"],
    )
    parser.add_argument("--pi3_checkpoint", type=str, default="yyfz233/Pi3")
    parser.add_argument("--pi3_chunk_size", type=int, default=60)
    parser.add_argument("--pi3_overlap", type=int, default=30)
    parser.add_argument(
        "--vggt_omega_checkpoint",
        type=str,
        default=DEFAULT_VGGT_OMEGA_CHECKPOINT,
    )
    parser.add_argument("--vggt_omega_chunk_size", type=int, default=60)
    parser.add_argument("--vggt_omega_overlap", type=int, default=30)
    parser.add_argument("--vggt_omega_image_resolution", type=int, default=512)


def load_tracker(args) -> TrackEverything:
    return TrackEverything.from_pretrained(
        args.model,
        device=args.device,
        geometry_backend=args.geometry_backend,
        pi3_checkpoint=args.pi3_checkpoint,
        pi3_chunk_size=args.pi3_chunk_size,
        pi3_overlap=args.pi3_overlap,
        vggt_omega_checkpoint=args.vggt_omega_checkpoint,
        vggt_omega_chunk_size=args.vggt_omega_chunk_size,
        vggt_omega_overlap=args.vggt_omega_overlap,
        vggt_omega_image_resolution=args.vggt_omega_image_resolution,
        token=args.token,
    )


def run_inference(
    tracker: TrackEverything,
    args,
    *,
    mode: Literal["dense", "sparse"] = "dense",
    query_points_2d: Optional[np.ndarray] = None,
) -> TrackEverythingOutput:
    """Load any geometry files named in ``args`` and run ``tracker.infer``."""
    depths = intrinsics = extrinsics = None
    if args.depths_dir is not None:
        depth_files = sorted(Path(args.depths_dir).glob("*.npy"))
        if depth_files:
            depths = np.stack([np.load(f) for f in depth_files])
            logger.info("Loaded %d depth maps from %s", len(depth_files), args.depths_dir)
    if args.intrinsics_path is not None:
        intrinsics = np.load(args.intrinsics_path)
        logger.info("Loaded intrinsics from %s", args.intrinsics_path)
    if args.extrinsics_path is not None:
        extrinsics = np.load(args.extrinsics_path)
        logger.info("Loaded extrinsics from %s", args.extrinsics_path)

    return tracker.infer(
        video=args.video,
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        query_points_2d=query_points_2d,
        mode=mode,
        num_iters=args.num_iters,
        voxel_size=args.voxel_size,
        voxel_mode=args.voxel_mode,
        voxelize=args.voxelize,
        voxelize_on_endpoints=args.voxelize_on_endpoints,
        compress_memory=args.compress_memory,
        dynamic_threshold=args.dynamic_threshold,
        use_conf_mask=args.use_conf_mask,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        mem_efficient=args.mem_efficient,
    )


def save_outputs(
    output: TrackEverythingOutput,
    output_dir: Path,
    *,
    save_dense: bool = False,
) -> None:
    """Write tracks.pt, query_tracks.pt, geometry.pt and optionally dense_tracks.pt.

    Reload the tracks with ``SceneTokens(**torch.load("tracks.pt"))``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if output.prediction is not None:
        tracks = output.tracks()
        torch.save(tracks._asdict(), output_dir / "tracks.pt")
        logger.info("Saved %d tracks to %s", tracks.coords.shape[0], output_dir / "tracks.pt")

    if save_dense and output.coords is not None:
        torch.save(
            {
                "coords": output.coords.cpu(),
                "visibs": output.visibs.cpu(),
                "dynamic_logits": output.dynamic_logits.cpu(),
            },
            output_dir / "dense_tracks.pt",
        )
        logger.info("Saved dense tracks to %s", output_dir / "dense_tracks.pt")

    if output.query_coords is not None:
        torch.save(
            {"coords": output.query_coords.cpu(), "visibs": output.query_visibs.cpu()},
            output_dir / "query_tracks.pt",
        )
        logger.info("Saved query tracks to %s", output_dir / "query_tracks.pt")

    torch.save(
        {
            "norm_factor": output.norm_factor,
            "depths": output.depths.cpu(),
            "intrinsics": output.intrinsics.cpu(),
            "extrinsics": output.extrinsics.cpu(),
        },
        output_dir / "geometry.pt",
    )
    logger.info("Saved geometry to %s", output_dir / "geometry.pt")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="TrackEverything inference: run the tracker and save its tracks. "
        "Use demo.py to also visualize them."
    )
    add_inference_args(parser)
    parser.add_argument("--mode", type=str, default="dense", choices=["dense", "sparse"])
    parser.add_argument(
        "--query_points", type=str, default=None,
        help="Path to .npy file of 2D query points (N,3) with [t, x, y] format",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/track_everything")
    parser.add_argument(
        "--save_dense", action="store_true",
        help="Also save the dense (T, H, W, T) tensors; size grows with T squared",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    query_points = None
    if args.query_points is not None:
        query_points = np.load(args.query_points)
        logger.info("Loaded %d query points from %s", len(query_points), args.query_points)

    tracker = load_tracker(args)
    output = run_inference(tracker, args, mode=args.mode, query_points_2d=query_points)

    save_outputs(output, Path(args.output_dir), save_dense=args.save_dense)
    logger.info("Done!")


if __name__ == "__main__":
    main()
