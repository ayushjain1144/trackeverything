# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

# Ensure the real `wandb` PyPI package is imported, not a sibling `wandb/`
# log-cache directory that Python 3 would otherwise pick up as a namespace
# package (which has no `init` attribute and breaks wandb.init).
# _cwd = os.getcwd()
# if _cwd in sys.path:
#     sys.path.remove(_cwd)
# if "" in sys.path:
#     sys.path.remove("")

from typing import Dict, List, Optional, Tuple

import cv2
import ipdb
import numpy as np
import tap
import torch
import torch.distributed as dist

from ..third_party.pi3.utils.geometry import depth_edge, recover_intrinsic_from_rays_d
from .pi3 import Pi3BaseAnnotator
from .sim3_align import (
    accumulate_sim3 as _accumulate_sim3,
    apply_sim3_c2w as _apply_sim3_c2w,
    disable_autocast,
    weighted_align_point_maps as _weighted_align_point_maps,
)

st = ipdb.set_trace


# --------------------------------------------------------------------------- #
# Annotator
# --------------------------------------------------------------------------- #


class Pi3LongAnnotator(Pi3BaseAnnotator):
    """Pi3 with chunked long-sequence inference.

    Runs Pi3 on overlapping chunks of the input video, aligns adjacent
    chunks with a robust confidence-weighted Sim(3) on their shared overlap,
    accumulates the transforms back to chunk-0's coordinate frame, and then
    emits per-frame (depths, intrinsics, extrinsics) just like Pi3BaseAnnotator.

    Intrinsics are recovered per-chunk from cam-frame rays (scale-invariant);
    depths are the cam-frame z of local_points rescaled by the cumulative
    Sim(3) scale so all chunks live in chunk-0's metric; extrinsics come
    from inverting the sim3-aligned cam2world.

    Loop-closure from Pi-Long is NOT included here (it requires DBoW / SALAD
    retrieval weights and the sim3solve/pypose/triton/numba stack). If you
    need it, wire it in by producing loop pairs and extra sim3 constraints,
    then feeding them through a Sim3 pose-graph optimizer.
    """

    def __init__(
        self,
        chunk_size: int = 60,
        overlap: int = 30,
        # Relative conf threshold used to pre-filter overlap pixels before
        # IRLS. Matches Pi-Long: `conf_thresh = min(median(conf1), median(conf2)) * conf_threshold_ratio`.
        conf_threshold_ratio: float = 0.1,
        sim3_device: Optional[str] = None,
        irls_delta: float = 0.1,
        irls_max_iters: int = 5,
        irls_tol: float = 1e-9,
        # Visualization filters — mirror third_party/pi3/demo_gradio.py, which
        # (a) zeros out conf at depth discontinuities via depth_edge(..., rtol),
        # and then (b) keeps only points with conf >= viz_conf_threshold.
        viz_conf_threshold: float = 0.2,
        viz_depth_edge_rtol: float = 0.03,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert overlap < chunk_size, (
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.conf_threshold_ratio = conf_threshold_ratio
        self.sim3_device = sim3_device
        self.irls_delta = irls_delta
        self.irls_max_iters = irls_max_iters
        self.irls_tol = irls_tol
        self.viz_conf_threshold = viz_conf_threshold
        self.viz_depth_edge_rtol = viz_depth_edge_rtol

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
        # Last chunk might be shorter than chunk_size; ensure it still has
        # at least `overlap`+1 frames of overlap with the prior chunk,
        # otherwise alignment is ill-conditioned. If not, merge into prior.
        if len(chunks) >= 2 and (chunks[-1][1] - chunks[-1][0]) <= self.overlap:
            chunks[-2] = (chunks[-2][0], chunks[-1][1])
            chunks.pop()
        return chunks

    def _run_model_chunk(
        self, imgs_chunk: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Run the base Pi3 model on a (1, T_chunk, 3, H, W) batch and return
        CPU-tensor outputs we need for alignment + annotation."""
        out = self._run_model(imgs_chunk)
        # Pi3 emits raw logits for conf; Pi-Long sigmoids before alignment.
        conf = torch.sigmoid(out["conf"])        # (1, T, H, W, 1) or (1, T, H, W)
        conf = conf[0].float()
        if conf.dim() == 4 and conf.shape[-1] == 1:
            conf = conf.squeeze(-1)              # (T, H, W)
        return {
            "camera_poses": out["camera_poses"][0].float().cpu(),   # (T, 4, 4) cam2world
            "local_points": out["local_points"][0].float().cpu(),   # (T, H, W, 3) cam-frame
            "points":       out["points"][0].float().cpu(),         # (T, H, W, 3) world-frame
            "conf":         conf.cpu(),                             # (T, H, W) in [0, 1]
        }
    
    def _visualize_est_geoms(
        self,
        rgbs: np.ndarray,
        depths: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
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

        rgbs_np = rgbs                                           # (T, H, W, 3) uint8
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
        if valid_mask is not None:
            # Extra filter mirroring demo_gradio.py: drop pixels whose
            # confidence (already zeroed at depth edges) is below threshold.
            valid = valid & torch.from_numpy(valid_mask).to(valid.device)
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

        # save_dir = ...
        # save_dir.mkdir(parents=True, exist_ok=True)
        # np.save(save_dir / f"{seq_slug}.npy", viz_pc)

    def _log_conf_overlay_wandb(
        self,
        rgbs: np.ndarray,
        conf_low: np.ndarray,
        orig_w: int,
        orig_h: int,
        log_key: str = "pi3_long/conf_overlay",
        fps: float = 15.0,
        heatmap_alpha: float = 0.55,
    ) -> None:
        """Log a wandb.Video of RGB + colormapped confidence (no disk write).

        `conf_low` is model resolution (T, H_low, W_low); it is nearest-neighbor
        upsampled to match `rgbs` (T, orig_h, orig_w, 3) before blending.
        """
        import wandb

        if wandb.run is None or rgbs.shape[0] == 0:
            return

        conf_full = self._resize_depths_to(conf_low, orig_w, orig_h)
        a = float(np.clip(heatmap_alpha, 0.0, 1.0))
        # wandb.Video expects (time, channel, height, width).
        frames = np.empty((rgbs.shape[0], 3, int(orig_h), int(orig_w)), dtype=np.uint8)
        for t in range(rgbs.shape[0]):
            rgb = rgbs[t]
            rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
            c = np.clip(conf_full[t].astype(np.float32), 0.0, 1.0)
            conf_u8 = (c * 255.0).round().astype(np.uint8)
            heat_bgr = cv2.applyColorMap(conf_u8, cv2.COLORMAP_INFERNO)
            bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
            blended_bgr = (
                a * heat_bgr.astype(np.float32) + (1.0 - a) * bgr.astype(np.float32)
            ).clip(0, 255).astype(np.uint8)
            blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
            frames[t] = np.transpose(blended_rgb, (2, 0, 1))

        wandb.log(
            {log_key: wandb.Video(frames, fps=int(round(fps)), format="mp4")}
        )

    def process_video(
        self,
        rgbs: np.ndarray,
        gt_intrinsics: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._load_model()

        T_total = rgbs.shape[0]
        orig_h, orig_w = rgbs.shape[1], rgbs.shape[2]
        target_h, target_w = self._compute_target_size(orig_h, orig_w)

        # All chunks share the same TARGET_H/W because it is determined from
        # the first frame; thus intrinsics are consistent across chunks.
        imgs = self._preprocess_rgbs(rgbs, target_h, target_w)   # (1, T, 3, H, W)

        chunk_indices = self._compute_chunk_indices(T_total)

        # 1) Run model on every chunk independently (drop each chunk's
        #    intermediate CUDA memory before moving to the next).
        chunk_outputs: List[Dict[str, torch.Tensor]] = []
        for ci, (s_idx, e_idx) in enumerate(chunk_indices):
            chunk_imgs = imgs[:, s_idx:e_idx].contiguous()
            out = self._run_model_chunk(chunk_imgs)
            chunk_outputs.append(out)
            del chunk_imgs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 2) Estimate Sim(3) between every adjacent pair of chunks on their
        #    shared overlap. transform[i] maps chunk i+1 -> chunk i.
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
            s, R, t = _weighted_align_point_maps(
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
                f"[pi3_long] chunk {i+1} -> {i}: s={s:.4f}, "
                f"|t|={np.linalg.norm(t):.4f}"
            )
            pair_sim3.append((s, R, t))

        # cum_sim3[k] maps chunk k+1 -> chunk 0 (chunk 0 itself is identity).
        cum_sim3 = _accumulate_sim3(pair_sim3)

        # 3) Stitch per-frame outputs. Later chunks overwrite overlap frames
        #    (this matches Pi-Long's save_camera_poses).
        depths_low = np.zeros((T_total, target_h, target_w), dtype=np.float32)
        K_low_full = np.zeros((T_total, 3, 3), dtype=np.float32)
        c2w_full = np.zeros((T_total, 4, 4), dtype=np.float32)
        # conf lives in [0, 1]; we zero-out pixels on depth discontinuities
        # per-chunk (mirrors demo_gradio.run_model) so they're trivially
        # filtered by any positive conf threshold downstream.
        conf_low = np.zeros((T_total, target_h, target_w), dtype=np.float32)

        for ci, (s_idx, e_idx) in enumerate(chunk_indices):
            local_pts_t = chunk_outputs[ci]["local_points"]             # (T_c, H, W, 3)
            conf_t = chunk_outputs[ci]["conf"].clone()  
            # conf_t[:] = 1.0 # (T_c, H, W)
            # depth_edge / recover_intrinsic stay fp32 under ambient autocast.
            with disable_autocast():
                edge = depth_edge(local_pts_t[..., 2].float(), rtol=self.viz_depth_edge_rtol)
            conf_t[edge] = 0.0
            cam_poses = chunk_outputs[ci]["camera_poses"].numpy().astype(np.float32)

            if ci == 0:
                cum_s, cum_R, cum_t = 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
                c2w_aligned = cam_poses
            else:
                cum_s, cum_R, cum_t = cum_sim3[ci - 1]
                c2w_aligned = _apply_sim3_c2w(cam_poses, cum_s, cum_R, cum_t)

            # Cam-frame z rescaled by cumulative Sim(3) scale so every chunk
            # reports depth in chunk-0's metric. (See derivation: under a
            # Sim(3) S=(s,R,t) applied to world coords, new X_cam = s * old X_cam.)
            depth_chunk = (
                local_pts_t[..., 2].numpy().astype(np.float32) * float(cum_s)
            )

            # K recovered from cam-frame rays is scale-invariant, so we can
            # safely take it from the raw local_points per chunk.
            with disable_autocast():
                K_chunk = (
                    recover_intrinsic_from_rays_d(
                        local_pts_t.float(),
                        ndc_coords=False,
                        force_center_principal_point=False,
                    )
                    .numpy()
                    .astype(np.float32)
                )

            depths_low[s_idx:e_idx] = depth_chunk
            K_low_full[s_idx:e_idx] = K_chunk
            c2w_full[s_idx:e_idx] = c2w_aligned
            conf_low[s_idx:e_idx] = conf_t.numpy().astype(np.float32)

        depths = self._resize_depths_to(depths_low, orig_w, orig_h)
        K_full = self._rescale_intrinsics(K_low_full, target_w, target_h, orig_w, orig_h)

        if self.use_gt_intrinsics:
            assert gt_intrinsics is not None, (
                "use_gt_intrinsics=True but gt_intrinsics is None"
            )
            intrinsics = gt_intrinsics.copy()
        else:
            intrinsics = K_full

        extrinsics = np.linalg.inv(c2w_full)  # world2cam
        conf_mask_low = (conf_low >= self.viz_conf_threshold).astype(np.float32)
        conf_mask_full = self._resize_depths_to(conf_mask_low, orig_w, orig_h)
        # st()
        # self._log_conf_overlay_wandb(rgbs, conf_low, orig_w, orig_h)
        if False:
            # Threshold conf first (while it's still at low-res), then nearest-
            # neighbor upsample the binary mask — this avoids bleeding the
            # zeroed-out depth-edge pixels into their full-res neighbors.
            conf_mask_low = (conf_low >= self.viz_conf_threshold).astype(np.float32)
            conf_mask_full = self._resize_depths_to(conf_mask_low, orig_w, orig_h) > 0.5
            self._visualize_est_geoms(
                rgbs, depths, intrinsics, extrinsics, valid_mask=conf_mask_full
            )
        return (
            depths.astype(np.float32),
            intrinsics.astype(np.float32),
            extrinsics.astype(np.float32),
            conf_mask_full.astype(np.float32),
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


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

    # Chunked-inference / alignment knobs (defaults mirror Pi-Long's
    # configs/pi3-sim3-30.yaml with loop closure disabled).
    chunk_size: int = 60
    overlap: int = 30
    conf_threshold_ratio: float = 0.1
    irls_delta: float = 0.1
    irls_max_iters: int = 5
    irls_tol: float = 1e-9

    viz_dir: Optional[str] = "sreeharsha_tmp/est_geoms_pi3_long"
    max_viz_points: int = 200_000
    viz_voxel_size: float = 0.02
    viz_every_n: int = 1
    viz_conf_threshold: float = 0.2
    viz_depth_edge_rtol: float = 0.03
    wandb_project: Optional[str] = "tapip3d_pi3_long_annotations"
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
    
    # if args.wandb_project and rank == 0:
    #     import wandb

    #     if not hasattr(wandb, "init"):
    #         raise RuntimeError(
    #             f"Imported 'wandb' has no init() — likely a local wandb/ log dir "
    #             f"shadowing the real package. wandb.__file__={getattr(wandb, '__file__', None)}, "
    #             f"wandb.__path__={getattr(wandb, '__path__', None)}. "
    #             f"Fix: set WANDB_DIR to an out-of-repo path, delete ./wandb, "
    #             f"or run from a cwd without a wandb/ directory."
    #         )
    #     run_name = args.wandb_run_name or f"pi3_long_{args.provider_cfg}"
    #     wandb.init(
    #         project=args.wandb_project,
    #         entity=args.wandb_entity,
    #         name=run_name,
    #         mode=args.wandb_mode,
    #         config=dict(
    #             provider_cfg=args.provider_cfg,
    #             checkpoint_path=args.checkpoint_path,
    #             use_gt_intrinsics=args.use_gt_intrinsics,
    #             pixel_limit=args.pixel_limit,
    #             autocast_dtype=args.autocast_dtype,
    #             chunk_size=args.chunk_size,
    #             overlap=args.overlap,
    #             conf_threshold_ratio=args.conf_threshold_ratio,
    #             irls_delta=args.irls_delta,
    #             irls_max_iters=args.irls_max_iters,
    #             irls_tol=args.irls_tol,
    #         ),
    #     )

    annotator = Pi3LongAnnotator(
        checkpoint_path=args.checkpoint_path,
        use_gt_intrinsics=args.use_gt_intrinsics,
        pixel_limit=args.pixel_limit,
        autocast_dtype=args.autocast_dtype,
        viz_dir=args.viz_dir,
        max_viz_points=args.max_viz_points,
        viz_voxel_size=args.viz_voxel_size,
        viz_every_n=args.viz_every_n,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        conf_threshold_ratio=args.conf_threshold_ratio,
        irls_delta=args.irls_delta,
        irls_max_iters=args.irls_max_iters,
        irls_tol=args.irls_tol,
        viz_conf_threshold=args.viz_conf_threshold,
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
