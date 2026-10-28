# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run TrackEverything on several synchronized cameras and visualize the tracks.

Tracking itself goes through ``demo_hf_multiview.MultiViewTracker`` (use that
module directly to get tracks back in your own code); this script adds Rerun /
video / W&B visualization on top.

Usage:
    python -m trackeverything.demo_multiview --video cam0.mp4,cam1.mp4

Writes ``output.rrd`` (open with ``rerun output.rrd``) under
``--output_dir/<timestamp>/``, plus one 2D overlay video per camera with
``--save_dense_2d_video``. A single overlay video covering every camera would
jump viewpoint each frame, so the cameras are written separately.

Only the per-window track visualization is available here: the dense
(T, H, W, T) tensor grows with (frames x cameras) squared.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from .demo_hf import _str2bool, add_inference_args, run_inference, save_outputs
from .demo_hf_multiview import load_tracker, split_streams
from .utils import rerun_visualizer as visualizer
from .utils.common_utils import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TrackEverything on several synchronized cameras and "
        "visualize the tracks. Pass one --video path per camera, comma-separated.",
    )
    add_inference_args(parser)
    parser.add_argument("--output_dir", type=str, default="outputs/inference_mv")
    parser.add_argument("--max_viz_points", type=int, default=16000)
    parser.add_argument("--max_dynamic_points", type=int, default=4000)
    parser.add_argument(
        "--dynamic_from_motion", type=_str2bool, nargs="?", const=True, default=True,
        help="Color dynamic tracks from motion magnitude (true/false, default: true)",
    )
    parser.add_argument("--motion_threshold", type=float, default=0.01)
    parser.add_argument(
        "--trail_len", type=int, default=8,
        help="Trail length (frames) for the 2D and 3D track visualizations",
    )
    parser.add_argument(
        "--save_dense_2d_video", action="store_true",
        help="Also write the per-camera 2D track and static/dynamic overlay videos (mp4)",
    )
    parser.add_argument("--dense_2d_max_points", type=int, default=20000)
    parser.add_argument("--dense_2d_fps", type=int, default=12)
    parser.add_argument(
        "--save_tracks", action="store_true",
        help="Also save tracks.pt and geometry.pt like demo_hf_multiview",
    )
    parser.add_argument(
        "--use_rerun", type=_str2bool, nargs="?", const=True, default=True,
        help="Write tracks to output.rrd via rerun (true/false, default: true)",
    )
    parser.add_argument(
        "--use_wandb", type=_str2bool, nargs="?", const=True, default=False,
        help="Log the 2D dense-track videos to Weights & Biases (true/false, default: false)",
    )
    parser.add_argument("--wandb_project", type=str, default="from_scratch")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def _tracks_for_view(flat, view: int, num_views: int):
    """Restrict CompressedTracks to one camera's frames.

    The target-time axis is sliced to that camera's images and birth times are
    converted from the flat image index to the timestep index the sliced video
    is indexed by. Tracks born in other cameras are kept.
    """
    if num_views == 1:
        return flat
    return flat._replace(
        coords=flat.coords[:, view::num_views],
        visibs=flat.visibs[:, view::num_views],
        dyn_logits=flat.dyn_logits[:, view::num_views],
        src_time=torch.div(flat.src_time, num_views, rounding_mode="floor"),
    )


def visualize_tracks(args, output, window_len, output_dir, wandb_run, seq_name):
    """Visualize a capped sample of tracks read from the per-window snapshots."""
    from .utils.alltrack3r_utils import (
        compressed_dynamic_logits_diagonal,
        extract_compressed_tracks,
    )
    from .utils.vis_utils import (
        _log_flat_2d_tracks_video_rerun,
        _log_flat_tracks_rerun,
        _render_flat_2d_tracks_frames,
        generate_dense_dynamic_pred_videos,
        log_rgb_frames_rerun,
        save_dense_2d_tracks_video,
    )

    video = output.rgb
    num_views = output.num_views
    _, _, Hf, Wf = output.prediction.dense_shape
    tracks = extract_compressed_tracks(
        output.prediction,
        max_points=args.max_viz_points,
        max_dynamic_points=args.max_dynamic_points,
        dynamic_thresh=args.dynamic_threshold,
        dense_source_mask=output.dense_source_mask,
        voxel_size=args.voxel_size,
    )
    flat = tracks.to(video.device)

    def render_2d(view=None):
        """Overlay frames for one camera, or for every image when view is None."""
        sel = slice(None) if view is None else slice(view, None, num_views)
        return _render_flat_2d_tracks_frames(
            video[sel],
            output.intrinsics[sel],
            output.extrinsics[sel],
            flat if view is None else _tracks_for_view(flat, view, num_views),
            trail_len=args.trail_len,
            max_points=args.dense_2d_max_points,
            dynamic_thresh=args.dynamic_threshold,
            dynamic_from_motion=args.dynamic_from_motion,
            motion_threshold=args.motion_threshold,
            window_len=window_len,
        )

    hard_2d_frames = None
    if args.use_rerun:
        # Every camera in one recording, on the interleaved time axis.
        save_path = output_dir / "output.rrd"
        visualizer.setup_visualizer(app_name="Visualization", serve=False)
        visualizer.save_recording(save_path)

        _log_flat_tracks_rerun(
            rgb=video,
            depth=output.depths,
            intrinsics=output.intrinsics,
            extrinsics=output.extrinsics,
            flat=tracks,
            grid_size=(Hf, Wf),
            dense_source_mask=output.dense_source_mask.cpu(),
            entity="video",
            trail_len=args.trail_len,
            max_points=args.max_viz_points,
            max_dynamic_points=args.max_dynamic_points,
            dynamic_thresh=args.dynamic_threshold,
            dynamic_from_motion=args.dynamic_from_motion,
            motion_threshold=args.motion_threshold,
            color_by="rainbow",
            window_len=window_len,
        )
        _log_flat_2d_tracks_video_rerun(
            video,
            output.intrinsics,
            output.extrinsics,
            flat,
            grid_size=(Hf, Wf),
            entity="video",
            dynamic_thresh=args.dynamic_threshold,
            dynamic_from_motion=args.dynamic_from_motion,
            motion_threshold=args.motion_threshold,
            trail_len=args.trail_len,
            window_len=window_len,
            max_dynamic_points=args.max_dynamic_points,
            color_by="rainbow",
        )
        hard_2d_frames = render_2d()
        log_rgb_frames_rerun(
            hard_2d_frames,
            entity="video",
            image_path="dense_tracks_2d_hard",
            intrinsics=output.intrinsics,
        )
        logger.info(f"Saved results to {save_path.resolve()}")

    if not (args.save_dense_2d_video or wandb_run is not None):
        return

    mp4_path = None
    for view in range(num_views):
        path = (
            output_dir / "dense_tracks_2d.mp4"
            if num_views == 1
            else output_dir / f"dense_tracks_2d_view{view}.mp4"
        )
        frames = (
            hard_2d_frames
            if num_views == 1 and hard_2d_frames is not None
            else render_2d(view)
        )
        save_dense_2d_tracks_video(frames, path, fps=args.dense_2d_fps)
        if args.save_dense_2d_video:
            logger.info(f"Saved 2D dense-track video to {path}")
        if mp4_path is None:
            mp4_path = path

    # Secondary diagnostic; camera 0 only.
    cam0 = slice(None, None, num_views)
    dyn_paths = generate_dense_dynamic_pred_videos(
        sample=SimpleNamespace(rgbs=output.rgb[None, cam0].detach().cpu()),
        preds=SimpleNamespace(
            dense_dynamic_logits=compressed_dynamic_logits_diagonal(
                output.prediction
            )[:, cam0]
        ),
        output_dir=output_dir,
        step_id=0,
        batch_idx=0,
        dataset_name=seq_name,
        dynamic_threshold=args.dynamic_threshold,
        log_wandb=False,
        writer=None,
        fps=args.dense_2d_fps,
        source_mask=output.dense_source_mask[cam0],
    )

    if wandb_run is not None:
        import wandb

        log_payload = {
            f"dense_tracks_2d/{seq_name}": wandb.Video(
                str(mp4_path), fps=args.dense_2d_fps, format="mp4"
            )
        }
        if dyn_paths:
            log_payload[f"dynamic_dense_pred/{seq_name}"] = wandb.Video(
                str(dyn_paths[0]), fps=args.dense_2d_fps, format="mp4"
            )
        wandb.log(log_payload)


def main():
    setup_logger()
    args = parse_args()
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_name = Path(split_streams(args.video)[0]).resolve().name

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
    output = run_inference(tracker, args)
    window_len = int(tracker.model.seq_len)
    logger.info("Tracked %d camera(s)", output.num_views)

    if args.save_tracks:
        save_outputs(output, output_dir)

    if not (args.use_rerun or wandb_run is not None or args.save_dense_2d_video):
        logger.info("Skipping visualization (rerun/wandb/2d-video all off)")
    else:
        visualize_tracks(args, output, window_len, output_dir, wandb_run, seq_name)

    if wandb_run is not None:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
