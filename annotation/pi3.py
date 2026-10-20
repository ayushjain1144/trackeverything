# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import sys

# Ensure the real `wandb` PyPI package is imported, not a sibling `wandb/`
# log-cache directory that Python 3 would otherwise pick up as a namespace
# package (which has no `init` attribute and breaks wandb.init).
_cwd = os.getcwd()
if _cwd in sys.path:
    sys.path.remove(_cwd)
if "" in sys.path:
    sys.path.remove("")

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import ipdb
import numpy as np
import tap
import torch
import torch.distributed as dist
from PIL import Image

from ..datasets.datatypes import RawSliceData
from ..third_party.pi3.models.pi3 import Pi3
from ..third_party.pi3.utils.geometry import recover_intrinsic_from_rays_d
from .base_annotator import BaseAnnotator

st = ipdb.set_trace


# Pi3 emits per-pixel cam-frame xyz pointmaps and cam2world poses, but it does
# NOT directly emit camera intrinsics (unlike MegaSAM). We recover K by
# treating the camera-frame pointmap as a field of ray directions and
# least-squares fitting (fx, fy, cx, cy) — see recover_intrinsic_from_rays_d.
class Pi3BaseAnnotator(BaseAnnotator):
    def __init__(
        self,
        checkpoint_path: str = "yyfz233/Pi3",
        use_gt_intrinsics: bool = False,
        num_workers: int = 8,
        pixel_limit: int = 255000,
        patch_size: int = 14,
        autocast_dtype: str = "bfloat16",
        viz_dir: Optional[str] = "sreeharsha_tmp/est_geoms_pi3",
        max_viz_points: int = 200_000,
        viz_voxel_size: float = 0.02,
        viz_every_n: int = 1,
    ):
        self.checkpoint_path = checkpoint_path
        self.use_gt_intrinsics = use_gt_intrinsics
        self.num_workers = num_workers
        self.pixel_limit = pixel_limit
        self.patch_size = patch_size
        self.autocast_dtype = autocast_dtype
        self.viz_dir = viz_dir
        self.max_viz_points = max_viz_points
        self.viz_voxel_size = viz_voxel_size
        self.viz_every_n = viz_every_n
        self._viz_counter = -1
        self._model: Optional[Pi3] = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        model = Pi3.from_pretrained(self.checkpoint_path)
        model = model.to(self.device)
        model = model.eval()
        self._model = model

    def _compute_target_size(self, h: int, w: int) -> Tuple[int, int]:
        ps = self.patch_size
        scale = math.sqrt(self.pixel_limit / max(h * w, 1))
        target_w_f = w * scale
        target_h_f = h * scale
        k = round(target_w_f / ps)
        m = round(target_h_f / ps)
        while (k * ps) * (m * ps) > self.pixel_limit:
            if k / max(m, 1) > target_w_f / max(target_h_f, 1):
                k -= 1
            else:
                m -= 1
        target_h = max(1, m) * ps
        target_w = max(1, k) * ps
        return target_h, target_w

    def _preprocess_rgbs(
        self, rgbs: np.ndarray, target_h: int, target_w: int
    ) -> torch.Tensor:
        tensor_list = []
        for i in range(len(rgbs)):
            pil_img = Image.fromarray(rgbs[i])
            pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            arr = np.asarray(pil_img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            tensor_list.append(tensor)
        imgs = torch.stack(tensor_list, dim=0)
        imgs = imgs.unsqueeze(0)
        imgs = imgs.to(self.device)
        return imgs

    def _amp_dtype(self) -> torch.dtype:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.autocast_dtype]

    def _run_model(self, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
        amp_dtype = self._amp_dtype()
        on_cuda = torch.cuda.is_available() and str(self.device).startswith("cuda")
        use_amp = on_cuda and amp_dtype != torch.float32
        with torch.inference_mode():
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    return self._model(imgs)
            return self._model(imgs)

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

    def process_video(
        self,
        rgbs: np.ndarray,
        gt_intrinsics: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._load_model()

        orig_h, orig_w = rgbs.shape[1], rgbs.shape[2]
        target_h, target_w = self._compute_target_size(orig_h, orig_w)

        imgs = self._preprocess_rgbs(rgbs, target_h, target_w)
        out = self._run_model(imgs)

        local_points = out["local_points"][0].float().cpu()           # (T, H, W, 3)
        camera_poses = out["camera_poses"][0].float().cpu().numpy()   # (T, 4, 4) cam2world

        depths_low = local_points[..., 2].numpy()                     # (T, H, W)
        K_low = recover_intrinsic_from_rays_d(
            local_points,
            ndc_coords=False,
            force_center_principal_point=False,
        ).numpy()                                                     # (T, 3, 3)

        depths = self._resize_depths_to(depths_low, orig_w, orig_h)
        K_full = self._rescale_intrinsics(K_low, target_w, target_h, orig_w, orig_h)

        if self.use_gt_intrinsics:
            assert gt_intrinsics is not None, "use_gt_intrinsics=True but gt_intrinsics is None"
            intrinsics = gt_intrinsics.copy()
        else:
            intrinsics = K_full

        extrinsics = np.linalg.inv(camera_poses)  # world2cam

        return (
            depths.astype(np.float32),
            intrinsics.astype(np.float32),
            extrinsics.astype(np.float32),
        )

    def __call__(self, data: RawSliceData) -> Dict[str, np.ndarray]:
        gt_K = data.gt_intrinsics if self.use_gt_intrinsics else None
        depths, intrinsics, extrinsics = self.process_video(data.rgbs, gt_intrinsics=gt_K)
        # self._visualize_est_geoms(data, depths, intrinsics, extrinsics)
        return dict(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            depths=depths,
        )

    def _visualize_est_geoms(
        self,
        data: RawSliceData,
        depths: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ) -> None:
        """Unproject every frame's depth into world coords, transform the union
        into cam-0's frame (so each sequence lives in the same canonical
        coordinate system), voxelize, subsample, and emit ONE (N, 6) xyz+rgb
        cloud per sequence — to both wandb (if a run exists) and disk. Controlled
        by viz_dir (disables both outputs when empty), viz_voxel_size, max_viz_points,
        and viz_every_n."""
        # if not self.viz_dir:
        #     return
        self._viz_counter += 1
        if self.viz_every_n > 1 and (self._viz_counter % self.viz_every_n) != 0:
            return

        from torch.nn import functional as F
        from ..utils.common_utils import batch_unproject
        from ..datasets.utils.geometry import voxelization
        from torch_scatter import scatter_mean

        rgbs_np = data.rgbs                                           # (T, H, W, 3) uint8
        rgbs_t = torch.from_numpy(rgbs_np).float()                    # (T, H, W, 3) in [0, 255]
        rgbs_t = rgbs_t.permute(0, 3, 1, 2).contiguous()              # (T, 3, H, W)

        depths_t = torch.from_numpy(depths)                           # (T, H, W)
        intrinsics_t = torch.from_numpy(intrinsics)                   # (T, 3, 3)
        extrinsics_t = torch.from_numpy(extrinsics)                   # (T, 4, 4) world2cam

        # Resize RGBs to match depth resolution so each pixel has one color.
        # rgbs_t = F.interpolate(
        #     rgbs_t,
        #     size=depths_t.shape[-2:],
        #     mode="bilinear",
        #     align_corners=False,
        # )                                                              # (T, 3, H, W)

        # Unproject to world frame, then transform into cam-0 frame so every
        # sequence has a canonical origin (cam-0 at identity). extrinsics are
        # world2cam, so world2cam_0 = extrinsics_t[0]; applying it to world
        # points gives cam-0-frame points.
        pcd_world = batch_unproject(
            depths_t, intrinsics_t, extrinsics_t, channel_last=True
        )                                                              # (T, H, W, 3) in world frame

        T, H, W, _ = pcd_world.shape
        ones = torch.ones(T, H, W, 1, dtype=pcd_world.dtype, device=pcd_world.device)
        pcd_world_h = torch.cat([pcd_world, ones], dim=-1)             # (T, H, W, 4)
        w2c0 = extrinsics_t[0]                                         # (4, 4)
        pcd_cam0 = torch.einsum("ij, thwj -> thwi", w2c0, pcd_world_h)[..., :3]

        valid = depths_t > 0                                           # (T, H, W)
        pc = pcd_cam0[valid]                                           # (M, 3)
        color = rgbs_t.permute(0, 2, 3, 1)[valid]                      # (M, 3) in [0, 255]

        if pc.numel() == 0:
            return

        if self.viz_voxel_size and self.viz_voxel_size > 0:
            p2v = voxelization(pc[None], self.viz_voxel_size).squeeze(0)  # (M,)
            pc = scatter_mean(pc, p2v, dim=0)
            color = scatter_mean(color, p2v, dim=0)

        if self.max_viz_points and pc.shape[0] > self.max_viz_points:
            idx = np.random.choice(pc.shape[0], self.max_viz_points, replace=False)
            pc = pc[idx]
            color = color[idx]

        viz_pc = torch.cat([pc.cpu(), color.cpu()], dim=-1).numpy().astype(np.float32)

        legend = f"est_geoms_pi3"

        import wandb
        if wandb.run is not None:
            wandb.log({legend: wandb.Object3D(viz_pc)})

        # save_dir = Path(self.viz_dir)
        # save_dir.mkdir(parents=True, exist_ok=True)
        # np.save(save_dir / f"{seq_slug}.npy", viz_pc)

    @property
    def device(self):
        return self._device

    def to(self, device: str):
        self._device = device
        if self._model is not None:
            self._model = self._model.to(device)


class Args(tap.Tap):
    output_path: str
    provider_cfg: str
    num_dataloader_threads: int = 8
    num_dataloader_workers: int = 4
    use_gt_intrinsics: bool = False
    mod10: Optional[int] = None
    checkpoint_path: str = "yyfz233/Pi3"
    pixel_limit: int = 255000
    autocast_dtype: str = "bfloat16"
    viz_dir: Optional[str] = "sreeharsha_tmp/est_geoms_pi3"
    max_viz_points: int = 200_000
    viz_voxel_size: float = 0.02
    viz_every_n: int = 1
    wandb_project: Optional[str] = "tapip3d_pi3_annotations"
    wandb_entity: Optional[str] = "katef-study"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    annotator_name: str = "pi3"


def main():
    if "LOCAL_RANK" in os.environ:
        try:
            dist.init_process_group(backend="nccl")
            print(
                f"Distributed initialization successful. Rank: {dist.get_rank()}, World Size: {dist.get_world_size()}"
            )
        except Exception as e:
            print(f"Failed to initialize distributed process group: {e}")
    else:
        print(
            "LOCAL_RANK not found in environment variables. Running in non-distributed mode."
        )
    args = Args().parse_args()

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if args.wandb_project and rank == 0:
        import wandb
        if not hasattr(wandb, "init"):
            raise RuntimeError(
                f"Imported 'wandb' has no init() — likely a local wandb/ log dir "
                f"shadowing the real package. wandb.__file__={getattr(wandb, '__file__', None)}, "
                f"wandb.__path__={getattr(wandb, '__path__', None)}. "
                f"Fix: set WANDB_DIR to an out-of-repo path, delete ./wandb, "
                f"or run from a cwd without a wandb/ directory."
            )
        run_name = args.wandb_run_name or f"pi3_{args.provider_cfg}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            mode=args.wandb_mode,
            config=dict(
                provider_cfg=args.provider_cfg,
                checkpoint_path=args.checkpoint_path,
                use_gt_intrinsics=args.use_gt_intrinsics,
                pixel_limit=args.pixel_limit,
                autocast_dtype=args.autocast_dtype,
            ),
        )

    annotator = Pi3BaseAnnotator(
        checkpoint_path=args.checkpoint_path,
        use_gt_intrinsics=args.use_gt_intrinsics,
        pixel_limit=args.pixel_limit,
        autocast_dtype=args.autocast_dtype,
        viz_dir=args.viz_dir,
        max_viz_points=args.max_viz_points,
        viz_voxel_size=args.viz_voxel_size,
        viz_every_n=args.viz_every_n,
    )

    from .utils import generate_annotations

    generate_annotations(
        args.output_path,
        annotator,
        args.provider_cfg,
        num_dataloader_threads=args.num_dataloader_threads,
        num_dataloader_workers=args.num_dataloader_workers,
        mod10=args.mod10,
        annotator_name="pi3",
    )


if __name__ == "__main__":
    main()