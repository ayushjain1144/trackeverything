# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Chunked long-video geometry annotation with VGGT-Omega.

Mirrors Pi3LongAnnotator: run overlapping chunks, align with confidence-weighted
Sim(3) on overlap world points, stitch depths / K / poses into chunk-0's frame.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tap
import torch
import torch.distributed as dist
from PIL import Image

from ..datasets.datatypes import RawSliceData
from ..third_party.pi3.utils.geometry import depth_edge
from .base_annotator import BaseAnnotator
from .sim3_align import (
    accumulate_sim3,
    apply_sim3_c2w,
    disable_autocast,
    weighted_align_point_maps,
)

_DEFAULT_CHECKPOINT = (
    "/data/group_data/katefgroup/language_grounding/tracking_ckpts/"
    "vggt_omega_1b_512.pt"
)

_VGGT_OMEGA_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "vggt_omega"
if str(_VGGT_OMEGA_ROOT) not in sys.path:
    sys.path.insert(0, str(_VGGT_OMEGA_ROOT))


def _w2c_3x4_to_4x4(extrinsics: torch.Tensor) -> torch.Tensor:
    """(T, 3, 4) OpenCV w2c -> (T, 4, 4)."""
    T = extrinsics.shape[0]
    out = torch.zeros(T, 4, 4, dtype=extrinsics.dtype, device=extrinsics.device)
    out[:, :3, :4] = extrinsics
    out[:, 3, 3] = 1.0
    return out


def _unproject_depth_to_world(
    depth: torch.Tensor,  # (T, H, W)
    w2c: torch.Tensor,  # (T, 4, 4)
    K: torch.Tensor,  # (T, 3, 3)
) -> torch.Tensor:
    """Unproject depth maps to world-frame points. Same math as VGGT-Omega demo_gradio."""
    T, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    xs = xs[None].expand(T, -1, -1)
    ys = ys[None].expand(T, -1, -1)

    fx = K[:, 0, 0][:, None, None]
    fy = K[:, 1, 1][:, None, None]
    cx = K[:, 0, 2][:, None, None]
    cy = K[:, 1, 2][:, None, None]

    cam_pts = torch.stack(
        [
            (xs - cx) / fx * depth,
            (ys - cy) / fy * depth,
            depth,
        ],
        dim=-1,
    )  # (T, H, W, 3)

    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    # X_world = R^T (X_cam - t)
    return torch.einsum(
        "tij,thwj->thwi",
        R.transpose(1, 2),
        cam_pts - t[:, None, None, :],
    )


def _crop_to_supported_aspect_ratio(
    image: Image.Image,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Center-crop extreme aspect ratios; return cropped image and (left, top, right, bottom) in original coords."""
    width, height = image.size
    aspect_ratio = height / max(width, 1)

    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        box = (left, 0, left + crop_width, height)
        return image.crop(box), box

    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        box = (0, top, width, top + crop_height)
        return image.crop(box), box

    return image, (0, 0, width, height)


def _balanced_target_shape(
    aspect_ratio: float, image_resolution: int, patch_size: int
) -> Tuple[int, int]:
    token_number = (image_resolution // patch_size) ** 2
    w_patches = np.sqrt(token_number / aspect_ratio)
    h_patches = token_number / w_patches
    w_patches = max(1, int(np.round(w_patches)))
    h_patches = max(1, int(np.round(h_patches)))
    return h_patches * patch_size, w_patches * patch_size


class VGGTOmegaLongAnnotator(BaseAnnotator):
    """VGGT-Omega with chunked long-sequence inference + Sim(3) stitching."""

    def __init__(
        self,
        checkpoint_path: str = _DEFAULT_CHECKPOINT,
        image_resolution: int = 512,
        patch_size: int = 16,
        chunk_size: int = 60,
        overlap: int = 30,
        conf_threshold_ratio: float = 0.1,
        sim3_device: Optional[str] = None,
        irls_delta: float = 0.1,
        irls_max_iters: int = 5,
        irls_tol: float = 1e-9,
        viz_conf_percentile: float = 20.0,
        viz_depth_edge_rtol: float = 0.03,
        use_gt_intrinsics: bool = False,
        num_workers: int = 8,
        autocast_dtype: str = "bfloat16",
    ):
        assert overlap < chunk_size, (
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )
        self.checkpoint_path = checkpoint_path
        self.image_resolution = image_resolution
        self.patch_size = patch_size
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.conf_threshold_ratio = conf_threshold_ratio
        self.sim3_device = sim3_device
        self.irls_delta = irls_delta
        self.irls_max_iters = irls_max_iters
        self.irls_tol = irls_tol
        self.viz_conf_percentile = viz_conf_percentile
        self.viz_depth_edge_rtol = viz_depth_edge_rtol
        self.use_gt_intrinsics = use_gt_intrinsics
        self.num_workers = num_workers
        self.autocast_dtype = autocast_dtype
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def device(self) -> str:
        return self._device

    def to(self, device: str) -> None:
        self._device = device
        if self._model is not None:
            self._model = self._model.to(device)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from vggt_omega.models import VGGTOmega

        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                f"VGGT-Omega checkpoint not found: {self.checkpoint_path}"
            )
        model = VGGTOmega().eval()
        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        self._model = model

    def _compute_target_size(self, h: int, w: int) -> Tuple[int, int]:
        aspect_ratio = h / max(w, 1)
        return _balanced_target_shape(
            aspect_ratio, self.image_resolution, self.patch_size
        )

    def _preprocess_rgbs(
        self, rgbs: np.ndarray
    ) -> Tuple[torch.Tensor, int, int, Tuple[int, int, int, int]]:
        """Resize/crop RGB frames for Omega. Returns (1,T,3,H,W), target_h, target_w, crop_box."""
        orig_h, orig_w = rgbs.shape[1], rgbs.shape[2]
        # Determine crop from first frame (all frames share the same size).
        first = Image.fromarray(rgbs[0])
        _, crop_box = _crop_to_supported_aspect_ratio(first)
        left, top, right, bottom = crop_box
        crop_h, crop_w = bottom - top, right - left
        target_h, target_w = self._compute_target_size(crop_h, crop_w)

        tensor_list = []
        for i in range(len(rgbs)):
            pil_img = Image.fromarray(rgbs[i]).crop(crop_box)
            pil_img = pil_img.resize((target_w, target_h), Image.Resampling.BICUBIC)
            arr = np.asarray(pil_img).astype(np.float32) / 255.0
            tensor_list.append(torch.from_numpy(arr).permute(2, 0, 1))
        imgs = torch.stack(tensor_list, dim=0).unsqueeze(0).to(self.device)
        return imgs, target_h, target_w, crop_box

    def _amp_dtype(self) -> torch.dtype:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.autocast_dtype]

    def _compute_chunk_indices(self, T: int) -> List[Tuple[int, int]]:
        if T <= self.chunk_size:
            return [(0, T)]
        step = self.chunk_size - self.overlap
        num_chunks = (T - self.overlap + step - 1) // step
        chunks: List[Tuple[int, int]] = []
        for i in range(num_chunks):
            start = i * step
            end = min(start + self.chunk_size, T)
            chunks.append((start, end))
        if len(chunks) >= 2 and (chunks[-1][1] - chunks[-1][0]) <= self.overlap:
            chunks[-2] = (chunks[-2][0], chunks[-1][1])
            chunks.pop()
        return chunks

    def _run_model_chunk(
        self, imgs_chunk: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Run VGGT-Omega on (1, T_chunk, 3, H, W); return CPU tensors for stitching."""
        from vggt_omega.utils.pose_enc import encoding_to_camera

        with torch.inference_mode():
            predictions = self._model(imgs_chunk)

        # Geometry / linalg post-processing must stay fp32 even if a parent
        # wraps this call in amp.autocast(bf16).
        with disable_autocast():
            pose_enc = predictions["pose_enc"].float()
            hw = predictions["images"].shape[-2:]
            extrinsics_3x4, intrinsics = encoding_to_camera(pose_enc, hw)
            # Drop batch dim.
            extrinsics_3x4 = extrinsics_3x4[0].float()  # (T, 3, 4)
            intrinsics = intrinsics[0].float()  # (T, 3, 3)
            depth = predictions["depth"][0].float()  # (T, H, W, 1) or (T, H, W)
            if depth.dim() == 4 and depth.shape[-1] == 1:
                depth = depth.squeeze(-1)
            conf = predictions["depth_conf"][0].float()
            if conf.dim() == 4 and conf.shape[-1] == 1:
                conf = conf.squeeze(-1)

            w2c = _w2c_3x4_to_4x4(extrinsics_3x4)
            c2w = torch.linalg.inv(w2c)
            world_points = _unproject_depth_to_world(depth, w2c, intrinsics)

        return {
            "camera_poses": c2w.cpu(),  # (T, 4, 4) cam2world
            "depths": depth.cpu(),  # (T, H, W)
            "intrinsics": intrinsics.cpu(),  # (T, 3, 3)
            "points": world_points.cpu(),  # (T, H, W, 3)
            "conf": conf.cpu(),  # (T, H, W)
        }

    def _resize_depth(self, depth: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        return cv2.resize(
            depth.astype(np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )

    def _resize_depths_to(
        self, depths_low: np.ndarray, target_w: int, target_h: int
    ) -> np.ndarray:
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._resize_depth, d, target_w, target_h)
                for d in depths_low
            ]
            return np.stack([f.result() for f in futures])

    def _rescale_intrinsics(
        self,
        K: np.ndarray,
        src_w: int,
        src_h: int,
        dst_w: int,
        dst_h: int,
    ) -> np.ndarray:
        K = K.copy()
        sx = dst_w / float(src_w)
        sy = dst_h / float(src_h)
        K[:, 0, :] *= sx
        K[:, 1, :] *= sy
        return K

    def _map_to_original_resolution(
        self,
        depths_crop: np.ndarray,
        K_crop: np.ndarray,
        conf_crop: np.ndarray,
        orig_h: int,
        orig_w: int,
        crop_box: Tuple[int, int, int, int],
        target_h: int,
        target_w: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resize model outputs to crop size, then paste into full original HxW."""
        left, top, right, bottom = crop_box
        crop_h, crop_w = bottom - top, right - left

        depths_at_crop = self._resize_depths_to(depths_crop, crop_w, crop_h)
        conf_at_crop = self._resize_depths_to(conf_crop, crop_w, crop_h)
        K_at_crop = self._rescale_intrinsics(
            K_crop, target_w, target_h, crop_w, crop_h
        )

        T = depths_at_crop.shape[0]
        depths_full = np.zeros((T, orig_h, orig_w), dtype=np.float32)
        conf_full = np.zeros((T, orig_h, orig_w), dtype=np.float32)
        depths_full[:, top:bottom, left:right] = depths_at_crop
        conf_full[:, top:bottom, left:right] = conf_at_crop

        K_full = K_at_crop.copy()
        K_full[:, 0, 2] += float(left)
        K_full[:, 1, 2] += float(top)
        return depths_full, K_full, conf_full

    def process_video(
        self,
        rgbs: np.ndarray,
        gt_intrinsics: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._load_model()

        T_total = rgbs.shape[0]
        orig_h, orig_w = rgbs.shape[1], rgbs.shape[2]
        imgs, target_h, target_w, crop_box = self._preprocess_rgbs(rgbs)

        chunk_indices = self._compute_chunk_indices(T_total)

        chunk_outputs: List[Dict[str, torch.Tensor]] = []
        for ci, (s_idx, e_idx) in enumerate(chunk_indices):
            chunk_imgs = imgs[:, s_idx:e_idx].contiguous()
            out = self._run_model_chunk(chunk_imgs)
            chunk_outputs.append(out)
            del chunk_imgs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        sim3_device = (
            self.sim3_device
            if self.sim3_device is not None
            else (str(self.device) if torch.cuda.is_available() else "cpu")
        )
        pair_sim3: List[Tuple[float, np.ndarray, np.ndarray]] = []
        for i in range(len(chunk_indices) - 1):
            pts_tgt = (
                chunk_outputs[i]["points"][-self.overlap:].numpy().astype(np.float32)
            )
            pts_src = (
                chunk_outputs[i + 1]["points"][: self.overlap].numpy().astype(np.float32)
            )
            conf_tgt = (
                chunk_outputs[i]["conf"][-self.overlap:].numpy().astype(np.float32)
            )
            conf_src = (
                chunk_outputs[i + 1]["conf"][: self.overlap].numpy().astype(np.float32)
            )
            conf_thresh = (
                min(float(np.median(conf_tgt)), float(np.median(conf_src)))
                * self.conf_threshold_ratio
            )
            s, R, t = weighted_align_point_maps(
                target_pts=pts_tgt,
                target_conf=conf_tgt,
                source_pts=pts_src,
                source_conf=conf_src,
                conf_threshold=conf_thresh,
                device=sim3_device,
                delta=self.irls_delta,
                max_iters=self.irls_max_iters,
                tol=self.irls_tol,
            )
            print(
                f"[vggt_omega_long] chunk {i + 1} -> {i}: s={s:.4f}, "
                f"|t|={np.linalg.norm(t):.4f}"
            )
            pair_sim3.append((s, R, t))

        cum_sim3 = accumulate_sim3(pair_sim3)

        depths_low = np.zeros((T_total, target_h, target_w), dtype=np.float32)
        K_low_full = np.zeros((T_total, 3, 3), dtype=np.float32)
        c2w_full = np.zeros((T_total, 4, 4), dtype=np.float32)
        conf_low = np.zeros((T_total, target_h, target_w), dtype=np.float32)

        for ci, (s_idx, e_idx) in enumerate(chunk_indices):
            depth_t = chunk_outputs[ci]["depths"]
            conf_t = chunk_outputs[ci]["conf"].clone()
            edge = depth_edge(depth_t, rtol=self.viz_depth_edge_rtol)
            conf_t[edge] = 0.0
            cam_poses = chunk_outputs[ci]["camera_poses"].numpy().astype(np.float32)
            K_chunk = chunk_outputs[ci]["intrinsics"].numpy().astype(np.float32)

            if ci == 0:
                cum_s = 1.0
                c2w_aligned = cam_poses
            else:
                cum_s, cum_R, cum_t = cum_sim3[ci - 1]
                c2w_aligned = apply_sim3_c2w(cam_poses, cum_s, cum_R, cum_t)

            depth_chunk = depth_t.numpy().astype(np.float32) * float(cum_s)

            depths_low[s_idx:e_idx] = depth_chunk
            K_low_full[s_idx:e_idx] = K_chunk
            c2w_full[s_idx:e_idx] = c2w_aligned
            conf_low[s_idx:e_idx] = conf_t.numpy().astype(np.float32)

        depths, K_full, conf_full = self._map_to_original_resolution(
            depths_low,
            K_low_full,
            conf_low,
            orig_h,
            orig_w,
            crop_box,
            target_h,
            target_w,
        )

        if self.use_gt_intrinsics:
            assert gt_intrinsics is not None, (
                "use_gt_intrinsics=True but gt_intrinsics is None"
            )
            intrinsics = gt_intrinsics.copy()
        else:
            intrinsics = K_full

        extrinsics = np.linalg.inv(c2w_full)  # world2cam

        valid = conf_full > 1e-5
        if np.any(valid) and self.viz_conf_percentile > 0:
            conf_threshold = float(
                np.percentile(conf_full[valid], self.viz_conf_percentile)
            )
            conf_mask_full = (conf_full >= conf_threshold).astype(np.float32)
        else:
            conf_mask_full = (conf_full > 1e-5).astype(np.float32)

        return (
            depths.astype(np.float32),
            intrinsics.astype(np.float32),
            extrinsics.astype(np.float32),
            conf_mask_full.astype(np.float32),
        )

    def __call__(self, data: RawSliceData) -> Dict[str, np.ndarray]:
        gt_K = data.gt_intrinsics if self.use_gt_intrinsics else None
        depths, intrinsics, extrinsics, _conf = self.process_video(
            data.rgbs, gt_intrinsics=gt_K
        )
        return dict(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            depths=depths,
        )


class Args(tap.Tap):
    output_path: str
    provider_cfg: str
    num_dataloader_threads: int = 8
    num_dataloader_workers: int = 4
    use_gt_intrinsics: bool = False
    mod10: Optional[int] = None
    checkpoint_path: str = _DEFAULT_CHECKPOINT
    image_resolution: int = 512
    autocast_dtype: str = "bfloat16"
    chunk_size: int = 60
    overlap: int = 30
    conf_threshold_ratio: float = 0.1
    irls_delta: float = 0.1
    irls_max_iters: int = 5
    irls_tol: float = 1e-9
    viz_conf_percentile: float = 20.0
    viz_depth_edge_rtol: float = 0.03
    annotator_name: str = "vggt_omega"


def main():
    if "LOCAL_RANK" in os.environ:
        try:
            dist.init_process_group(backend="nccl")
            print(
                f"Distributed initialization successful. Rank: {dist.get_rank()}, "
                f"World Size: {dist.get_world_size()}"
            )
        except Exception as e:
            print(f"Failed to initialize distributed process group: {e}")
    else:
        print(
            "LOCAL_RANK not found in environment variables. "
            "Running in non-distributed mode."
        )
    args = Args().parse_args()

    annotator = VGGTOmegaLongAnnotator(
        checkpoint_path=args.checkpoint_path,
        image_resolution=args.image_resolution,
        use_gt_intrinsics=args.use_gt_intrinsics,
        autocast_dtype=args.autocast_dtype,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        conf_threshold_ratio=args.conf_threshold_ratio,
        irls_delta=args.irls_delta,
        irls_max_iters=args.irls_max_iters,
        irls_tol=args.irls_tol,
        viz_conf_percentile=args.viz_conf_percentile,
        viz_depth_edge_rtol=args.viz_depth_edge_rtol,
    )

    from .utils import generate_annotations

    generate_annotations(
        args.output_path,
        annotator,
        args.provider_cfg,
        num_dataloader_threads=args.num_dataloader_threads,
        num_dataloader_workers=args.num_dataloader_workers,
        mod10=args.mod10,
        annotator_name=args.annotator_name,
    )


if __name__ == "__main__":
    main()
