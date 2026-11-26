# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Track a grid of query points through a video and visualize them.

Query points are tracked individually and never merged into the scene
representation, which makes this the cheapest way to follow a fixed set of
pixels. Pass ``--mode dense`` to also build the full scene representation.

Usage:
    python -m trackeverything.demo_query --video path/to/video.mp4 --grid_size 32

Writes ``output.rrd`` (open with ``rerun output.rrd``) and
``query_tracks_2d.mp4`` under ``--output_dir/<timestamp>/``. Pass
``--use_wandb`` to also log the 2D overlay to Weights & Biases.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np

from .demo_hf import (
    _load_video,
    _str2bool,
    add_inference_args,
    load_tracker,
    run_inference,
    save_outputs,
)
from .utils import rerun_visualizer as visualizer
from .utils.common_utils import setup_logger

logger = logging.getLogger(__name__)


def grid_query_points(
    resolution: Tuple[int, int],
    grid_size: int,
    frame_idx: int = 0,
) -> np.ndarray:
    """Build a ``grid_size x grid_size`` grid of query points inset from the border.

    Args:
        resolution: ``(height, width)`` of the input video in pixels.
        grid_size: Number of points per side.
        frame_idx: Frame the grid is placed on, indexed into the processed
            (post ``start_frame`` / ``frame_stride``) sequence.

    Returns:
        ``(grid_size ** 2, 3)`` array of ``[frame_idx, x, y]`` rows, in input-video
        pixels. ``TrackEverything.infer`` rescales them to the inference resolution.
    """
    height, width = resolution
    ys = np.linspace(0, height, grid_size + 2, dtype=np.float32)[1:-1]
    xs = np.linspace(0, width, grid_size + 2, dtype=np.float32)[1:-1]
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    frame = np.full(grid_x.size, frame_idx, dtype=np.float32)
    return np.stack([frame, grid_x.ravel(), grid_y.ravel()], axis=1)


def visualize_query_tracks(args, output, query_points, output_dir) -> None:
    """Log the query trajectories, RGB video and depth point cloud to Rerun."""
    from .utils.vis_utils import log_query_3d_tracks_rerun

    save_path = output_dir / "output.rrd"
    visualizer.setup_visualizer(app_name="Visualization", serve=False)
    visualizer.save_recording(save_path)

    log_query_3d_tracks_rerun(
        rgb=output.rgb,
        depth=output.depths,
        intrinsics=output.intrinsics,
        extrinsics=output.extrinsics,
        query_coords=output.query_coords[0],
        query_visibs=output.query_visibs[0],
        query_birth_steps=query_points[:, 0].astype(np.int64),
        entity="video",
        trail_len=args.trail_len,
        max_points=args.max_viz_points,
        log_backdrop=True,
    )
    logger.info(f"Saved results to {save_path.resolve()}")


def render_query_tracks_2d(output, query_points, *, trail_len: int = 8) -> np.ndarray:
    """Project the query trajectories into each frame and draw them over the RGB video.

    Points are colored by query index with the same ``turbo`` ramp as the Rerun
    view, and drawn from the frame each query is born on.

    Returns:
        ``(T, H, W, 3)`` uint8 RGB frames.
    """
    import cv2
    import matplotlib

    from .utils.common_utils import batch_project

    coords = output.query_coords[0].detach().float().cpu()
    num_frames, num_points, _ = coords.shape
    intrinsics = output.intrinsics.detach().float().cpu()
    extrinsics = output.extrinsics.detach().float().cpu()

    uv = batch_project(
        coords,
        intrinsics[:, None].expand(-1, num_points, -1, -1),
        extrinsics[:, None].expand(-1, num_points, -1, -1),
    ).numpy()
    drawable = np.isfinite(uv).all(axis=-1)
    uv = np.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0).clip(-1e4, 1e4)
    uv = np.round(uv).astype(np.int32)

    rgb = (output.rgb.detach().float().cpu().permute(0, 2, 3, 1).numpy() * 255)
    rgb = rgb.clip(0, 255).astype(np.uint8)
    height, width = rgb.shape[1:3]
    radius = max(2, min(height, width) // 160)

    births = query_points[:, 0].astype(np.int64)
    ramp = matplotlib.colormaps["turbo"](np.linspace(0, 1, num_points))
    colors = (ramp[:, :3] * 255).astype(np.uint8)

    frames = []
    for t in range(num_frames):
        frame = rgb[t].copy()
        for n in np.nonzero(births <= t)[0]:
            color = tuple(int(c) for c in colors[n])
            for s in range(max(int(births[n]), t - trail_len + 1), t):
                if drawable[s, n] and drawable[s + 1, n]:
                    cv2.line(frame, tuple(uv[s, n]), tuple(uv[s + 1, n]), color, 1, cv2.LINE_AA)
            x, y = uv[t, n]
            if drawable[t, n] and 0 <= x < width and 0 <= y < height:
                cv2.circle(frame, (x, y), radius, color, -1)
        frames.append(frame)

    return np.stack(frames, axis=0)


def save_query_2d_video(args, output, query_points, output_dir, wandb_run, seq_name) -> None:
    """Write the 2D query-track overlay and log it to W&B."""
    from .utils.vis_utils import save_dense_2d_tracks_video

    frames = render_query_tracks_2d(output, query_points, trail_len=args.trail_len)
    mp4_path = output_dir / "query_tracks_2d.mp4"
    save_dense_2d_tracks_video(frames, mp4_path, fps=args.query_2d_fps)
    logger.info(f"Saved 2D query-track video to {mp4_path}")

    if wandb_run is not None:
        import wandb

        wandb.log({
            f"query_tracks_2d/{seq_name}": wandb.Video(
                str(mp4_path), fps=args.query_2d_fps, format="mp4"
            )
        })
        logger.info(f"Logged 2D query-track video to {wandb_run.url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track a grid of query points on a video and visualize the tracks."
    )
    add_inference_args(parser)
    parser.add_argument("--output_dir", type=str, default="outputs/query_inference")
    parser.add_argument(
        "--grid_size", type=int, default=32,
        help="Track a grid_size x grid_size grid of points (default: 32)",
    )
    parser.add_argument(
        "--query_frame", type=int, default=0,
        help="Frame the grid is placed on, in processed-frame indices",
    )
    parser.add_argument(
        "--mode", type=str, default="sparse", choices=["dense", "sparse"],
        help="'sparse' tracks only the query points; 'dense' also tracks every pixel",
    )
    parser.add_argument(
        "--trail_len", type=int, default=8,
        help="Trail length (frames) for the 2D and 3D track visualizations",
    )
    parser.add_argument("--max_viz_points", type=int, default=4096)
    parser.add_argument(
        "--save_tracks", action="store_true",
        help="Also save query_tracks.pt and geometry.pt",
    )
    parser.add_argument(
        "--use_rerun", type=_str2bool, nargs="?", const=True, default=True,
        help="Write tracks to output.rrd via rerun (true/false, default: true)",
    )
    parser.add_argument(
        "--save_2d_video", type=_str2bool, nargs="?", const=True, default=True,
        help="Write the 2D query-track overlay mp4 (true/false, default: true)",
    )
    parser.add_argument("--query_2d_fps", type=int, default=12)
    parser.add_argument(
        "--use_wandb", type=_str2bool, nargs="?", const=True, default=False,
        help="Log the 2D query-track video to Weights & Biases (true/false, default: false)",
    )
    parser.add_argument("--wandb_project", type=str, default="from_scratch")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def main():
    setup_logger()
    args = parse_args()
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_name = Path(args.video).resolve().name

    video = _load_video(args.video)
    num_frames = len(video[args.start_frame::args.frame_stride])
    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)
    if not 0 <= args.query_frame < num_frames:
        raise ValueError(
            f"--query_frame {args.query_frame} is outside the {num_frames} frames "
            f"left after start_frame={args.start_frame}, frame_stride={args.frame_stride}"
        )

    query_points = grid_query_points(video.shape[1:3], args.grid_size, args.query_frame)
    logger.info(
        "Tracking %d query points placed on frame %d", len(query_points), args.query_frame
    )

    wandb_run = None
    if args.use_wandb:
        import wandb

        init_kwargs = dict(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name or output_dir.name,
            dir=str(output_dir),
            config=vars(args),
        )
        if args.wandb_run_id:
            init_kwargs["id"] = args.wandb_run_id
            init_kwargs["resume"] = "allow"
        wandb_run = wandb.init(**init_kwargs)

    tracker = load_tracker(args)
    infer_args = argparse.Namespace(**vars(args))
    infer_args.video = video
    output = run_inference(tracker, infer_args, mode=args.mode, query_points_2d=query_points)

    if args.save_tracks:
        save_outputs(output, output_dir)
    if args.use_rerun:
        visualize_query_tracks(args, output, query_points, output_dir)
    if args.save_2d_video or wandb_run is not None:
        save_query_2d_video(args, output, query_points, output_dir, wandb_run, seq_name)

    if wandb_run is not None:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
