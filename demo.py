# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Run TrackEverything on a video and visualize the tracks.

Tracking itself goes through ``demo_hf.TrackEverything`` (use that module
directly to get tracks back in your own code); this script adds Rerun / video /
W&B visualization on top.

Usage:
    python -m trackeverything.demo --video path/to/video.mp4

Writes ``output.rrd`` (open with ``rerun output.rrd``) under
``--output_dir/<timestamp>/``, plus 2D overlay videos with
``--save_dense_2d_video``.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from .demo_hf import (
    _str2bool,
    add_inference_args,
    load_tracker,
    run_inference,
    save_outputs,
)
from .utils import rerun_visualizer as visualizer
from .utils.common_utils import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TrackEverything on a video and visualize the tracks."
    )
    add_inference_args(parser)
    parser.add_argument("--output_dir", type=str, default="outputs/inference")
    parser.add_argument(
        "--long_video", type=_str2bool, nargs="?", const=True, default=True,
        help="Visualize a capped sample of per-window tracks instead of the dense "
        "(T, H, W, T) tensor, which grows with T squared (true/false, default: true)",
    )
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
        help="Also write 2D track and static/dynamic overlay videos (mp4)",
    )
    parser.add_argument("--dense_2d_max_points", type=int, default=20000)
    parser.add_argument("--dense_2d_fps", type=int, default=12)
    parser.add_argument(
        "--save_tracks", action="store_true",
        help="Also save tracks.pt and geometry.pt like demo_hf",
    )
    parser.add_argument(
        "--use_rerun", type=_str2bool, nargs="?", const=True, default=True,
        help="Write tracks to output.rrd via rerun (true/false, default: true)",
    )
    parser.add_argument(
        "--use_wandb", type=_str2bool, nargs="?", const=True, default=False,
        help="Log 2D dense-track videos to Weights & Biases (true/false, default: false)",
    )
    parser.add_argument("--wandb_project", type=str, default="from_scratch")
    parser.add_argument("--wandb_entity", type=str, default=None)
    # If set, resume/join this fixed run id so multiple invocations
    # (e.g. a DAVIS loop) log into the same wandb run.
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def _save_2d_videos(args, output, hard_2d_frames, dynamic_logits, output_dir, wandb_run, seq_name):
    """Write the 2D track overlay and the static/dynamic overlay; log both to W&B."""
    from .utils.vis_utils import generate_dense_dynamic_pred_videos, save_dense_2d_tracks_video

    mp4_path = output_dir / "dense_tracks_2d.mp4"
    save_dense_2d_tracks_video(hard_2d_frames, mp4_path, fps=args.dense_2d_fps)
    if args.save_dense_2d_video:
        logger.info(f"Saved 2D dense-track video to {mp4_path}")

    dyn_paths = generate_dense_dynamic_pred_videos(
        sample=SimpleNamespace(rgbs=output.rgb[None].detach().cpu()),
        preds=SimpleNamespace(dense_dynamic_logits=dynamic_logits),
        output_dir=output_dir,
        step_id=0,
        batch_idx=0,
        dataset_name=seq_name,
        dynamic_threshold=args.dynamic_threshold,
        log_wandb=False,
        writer=None,
        fps=args.dense_2d_fps,
        source_mask=output.dense_source_mask,
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


def visualize_tracks(args, output, window_len, output_dir, wandb_run, seq_name):
    """Visualize a capped sample of tracks read from the per-window snapshots;
    linear in T."""
    from .utils.alltrack3r_utils import (
        compressed_dynamic_logits_diagonal,
        extract_compressed_tracks,
    )
    from .utils.vis_utils import (
        _log_flat_2d_tracks_video_rerun,
        _log_flat_tracks_rerun,
        _render_flat_2d_tracks_frames,
        log_rgb_frames_rerun,
    )

    video = output.rgb
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

    def render_2d():
        return _render_flat_2d_tracks_frames(
            video,
            output.intrinsics,
            output.extrinsics,
            flat,
            trail_len=args.trail_len,
            max_points=args.dense_2d_max_points,
            dynamic_thresh=args.dynamic_threshold,
            dynamic_from_motion=args.dynamic_from_motion,
            motion_threshold=args.motion_threshold,
            window_len=window_len,
        )

    hard_2d_frames = None
    if args.use_rerun:
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

    if args.save_dense_2d_video or wandb_run is not None:
        if hard_2d_frames is None:
            hard_2d_frames = render_2d()
        _save_2d_videos(
            args, output, hard_2d_frames,
            compressed_dynamic_logits_diagonal(output.prediction),
            output_dir, wandb_run, seq_name,
        )


def visualize_dense_tracks(args, output, window_len, output_dir, wandb_run, seq_name):
    """Visualize the dense (T, H, W, T) tracks; short clips only."""
    from .utils.vis_utils import (
        log_dense_2d_tracks_video_rerun,
        log_dense_3d_tracks_rerun,
        log_rgb_frames_rerun,
        render_dense_2d_tracks_frames,
    )

    video = output.rgb
    dense, dense_visibs, dense_logits = output.coords, output.visibs, output.dynamic_logits

    def render_2d():
        return render_dense_2d_tracks_frames(
            rgb=video,
            intrinsics=output.intrinsics,
            extrinsics=output.extrinsics,
            dense_coords=dense[0],
            dense_visibs=dense_visibs[0],
            dense_source_mask=output.dense_source_mask,
            trail_len=args.trail_len,
            max_points=args.dense_2d_max_points,
            dense_logits=dense_logits[0],
            dynamic_thresh=args.dynamic_threshold,
            dynamic_from_motion=args.dynamic_from_motion,
            motion_threshold=args.motion_threshold,
            window_len=window_len,
        )

    hard_2d_frames = None
    if args.use_rerun:
        # Attach file sink BEFORE any rerun.log() calls.
        save_path = output_dir / "output.rrd"
        visualizer.setup_visualizer(app_name="Visualization", serve=False)
        visualizer.save_recording(save_path)

        log_dense_3d_tracks_rerun(
            rgb=video,
            depth=output.depths,
            intrinsics=output.intrinsics,
            extrinsics=output.extrinsics,
            dense_coords=dense[0],
            dense_visibs=dense_visibs[0],
            dense_logits=dense_logits[0],
            entity="video",
            mode="voxel",
            voxel_size=args.voxel_size,
            trail_len=args.trail_len,
            max_points=args.max_viz_points,
            max_dynamic_points=args.max_dynamic_points,
            dynamic_thresh=args.dynamic_threshold,
            dynamic_from_motion=args.dynamic_from_motion,
            motion_threshold=args.motion_threshold,
            dense_source_mask=output.dense_source_mask,
            color_by="rainbow",
            window_len=window_len,
        )
        log_dense_2d_tracks_video_rerun(
            rgb=video,
            intrinsics=output.intrinsics,
            extrinsics=output.extrinsics,
            dense_coords=dense[0],
            dense_visibs=dense_visibs[0],
            dense_logits=dense_logits[0],
            entity="video",
            dynamic_thresh=args.dynamic_threshold,
            dynamic_from_motion=args.dynamic_from_motion,
            motion_threshold=args.motion_threshold,
            dense_source_mask=output.dense_source_mask,
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

    if args.save_dense_2d_video or wandb_run is not None:
        if hard_2d_frames is None:
            hard_2d_frames = render_2d()
        # mem_efficient inference leaves no per-pixel logits on the model output;
        # use the dense logits reconstructed from the per-window snapshots.
        _save_2d_videos(
            args, output, hard_2d_frames, dense_logits.detach().cpu(),
            output_dir, wandb_run, seq_name,
        )


def main():
    setup_logger()
    args = parse_args()
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_name = Path(args.video).resolve().name

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

    if args.save_tracks:
        save_outputs(output, output_dir)

    if not (args.use_rerun or wandb_run is not None or args.save_dense_2d_video):
        logger.info("Skipping visualization (rerun/wandb/2d-video all off)")
    elif args.long_video:
        visualize_tracks(args, output, window_len, output_dir, wandb_run, seq_name)
    else:
        visualize_dense_tracks(args, output, window_len, output_dir, wandb_run, seq_name)

    if wandb_run is not None:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
