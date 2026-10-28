# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-camera inference for TrackEverything.

Several synchronized cameras are laid out on the single time axis the
single-camera model already uses, in time-major order: flat image index
``t = f * V + v`` for ``V`` cameras and ``T = F * V`` images over ``F``
timesteps. The streams are interleaved here and handed to the geometry backend
as one image sequence, so the reconstruction is globally consistent across
cameras.

Usage:
    from trackeverything.demo_hf_multiview import MultiViewTracker

    tracker = MultiViewTracker.from_pretrained("katefgroup/TrackEverything")
    output = tracker.infer(video=["cam0.mp4", "cam1.mp4", "cam2.mp4"])

    # Already interleaved into one sequence, so state the camera count
    output = tracker.infer(video=interleaved_array, num_views=3)

CLI (saves tracks; use demo_multiview.py to also visualize them):
    python -m trackeverything.demo_hf_multiview \\
        --video cam0.mp4,cam1.mp4 --output_dir outputs/
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange

from .demo_hf import (
    TrackEverything,
    TrackEverythingOutput,
    _load_video,
    add_inference_args,
    run_inference,
    save_outputs,
)

logger = logging.getLogger(__name__)


def split_streams(video) -> List:
    """Split a multi-camera ``video`` argument into one entry per camera."""
    if isinstance(video, str) and "," in video:
        return [p.strip() for p in video.split(",") if p.strip()]
    if isinstance(video, (list, tuple)):
        return list(video)
    return [video]


def load_views(
    video, *, frames: slice = slice(None), max_frames: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """Resolve one or more synchronized streams into one time-major array.

    ``video`` is a single video (path, URL, image folder or array) or several of
    them, as a list or a comma-separated string of paths. Several streams are
    truncated to their common length and interleaved so that flat image index
    ``t = f * V + v``. ``frames`` and ``max_frames`` select timesteps, so they
    are applied per stream.

    Returns the (T, H, W, 3) uint8 RGB array and ``V``.
    """
    streams = split_streams(video)
    loaded = [_load_video(stream)[frames] for stream in streams]
    if max_frames is not None:
        loaded = [v[:max_frames] for v in loaded]
    if len(loaded) == 1:
        return loaded[0], 1

    shapes = {v.shape[1:] for v in loaded}
    if len(shapes) != 1:
        raise ValueError(f"All camera streams must share a resolution, got {shapes}")
    lengths = [len(v) for v in loaded]
    n_steps = min(lengths)
    if len(set(lengths)) != 1:
        logger.warning(
            "Camera streams have different lengths %s; truncating to %d",
            lengths,
            n_steps,
        )
    stacked = np.stack([v[:n_steps] for v in loaded], axis=0)
    return rearrange(stacked, "v f h w c -> (f v) h w c"), len(loaded)


@dataclasses.dataclass
class MultiViewOutput(TrackEverythingOutput):
    """:class:`TrackEverythingOutput` plus the camera count it was run at.

    When ``num_views > 1``, the time axis of every field holds ``num_views``
    images per timestep in time-major order, so camera ``v``'s images are
    ``[v::num_views]``.
    """

    num_views: int = 1


class MultiViewTracker(TrackEverything):
    """Inference wrapper for several synchronized cameras.

    Loads the weights into ``TrackEverythingMultiView`` and interleaves the
    camera streams onto one time axis; everything else is the single-camera
    pipeline unchanged.
    """

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
    ) -> "MultiViewTracker":
        """Load the multiview model from HuggingFace Hub or a local checkpoint.

        Takes the same arguments as :meth:`TrackEverything.from_pretrained`.
        The two model classes share their parameters, so any checkpoint works.
        """
        from . import models
        from .models.track_everything_multiview import TrackEverythingMultiView

        ckpt_path = Path(repo_id_or_path)
        if ckpt_path.exists() and ckpt_path.is_file():
            logger.info("Loading model from local checkpoint: %s", ckpt_path)
            model, _cfg = models.from_pretrained(str(ckpt_path), multiview=True)
        else:
            from huggingface_hub import hf_hub_download

            logger.info("Downloading model from HuggingFace Hub: %s", repo_id_or_path)
            local_path = hf_hub_download(
                repo_id=repo_id_or_path,
                filename="model.pth",
                revision=revision,
                token=token,
            )
            model, _cfg = models.from_pretrained(local_path, multiview=True)

        assert isinstance(model, TrackEverythingMultiView), (
            f"Expected a multiview model, got {type(model).__name__}"
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

    def _check_chunking(self, num_views: int) -> None:
        """Reject chunk sizes that would split a timestep across chunks.

        The geometry backends reconstruct in overlapping chunks and stitch them
        with a Sim(3) fit, so a chunk boundary inside a timestep would align
        partial timesteps against each other.
        """
        if self._geometry_backend == "pi3":
            chunk, overlap = self._pi3_chunk_size, self._pi3_overlap
        else:
            chunk, overlap = self._vggt_omega_chunk_size, self._vggt_omega_overlap
        if chunk % num_views or overlap % num_views:
            raise ValueError(
                f"{self._geometry_backend} chunk_size={chunk} and "
                f"overlap={overlap} must both be multiples of "
                f"num_views={num_views}; try chunk_size="
                f"{chunk // num_views * num_views} and overlap="
                f"{overlap // num_views * num_views}"
            )

    def infer(
        self,
        video: Union[str, List, np.ndarray, torch.Tensor],
        *,
        num_views: Optional[int] = None,
        depths: Optional[Union[np.ndarray, torch.Tensor]] = None,
        intrinsics: Optional[Union[np.ndarray, torch.Tensor]] = None,
        extrinsics: Optional[Union[np.ndarray, torch.Tensor]] = None,
        conf_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        start_frame: int = 0,
        frame_stride: int = 1,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> MultiViewOutput:
        """Run inference on one or more synchronized camera streams.

        Args:
            video: One video per camera (list, or comma-separated path string),
                or a single already-interleaved sequence together with
                ``num_views``. Each entry takes the same forms
                :meth:`TrackEverything.infer` accepts.
            num_views: Cameras per timestep. Defaults to the number of streams
                in ``video``; pass it only for already-interleaved input.
            depths, intrinsics, extrinsics, conf_mask: Optional geometry,
                indexed on the flat image axis, so already interleaved
                time-major to match the streams.
            start_frame, max_frames: Counted in timesteps, not images.
            frame_stride: Must be 1 when there is more than one camera.
            **kwargs: Forwarded to :meth:`TrackEverything.infer`. Query points
                in ``query_points_2d`` carry a flat image index, so a query on
                camera ``v`` at timestep ``f`` is ``f * num_views + v``.

        Returns:
            :class:`MultiViewOutput`.
        """
        frames = slice(start_frame, None, frame_stride)
        video_np, stream_views = load_views(
            video, frames=frames, max_frames=max_frames
        )
        if num_views is None:
            num_views = stream_views
        elif stream_views > 1 and num_views != stream_views:
            raise ValueError(
                f"num_views={num_views} does not match the {stream_views} "
                "streams passed as video"
            )
        num_views = int(num_views)
        if num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {num_views}")

        if num_views > 1:
            if frame_stride != 1:
                raise ValueError(
                    f"frame_stride must be 1 with num_views={num_views}, got "
                    f"{frame_stride}"
                )
            if video_np.shape[0] % num_views:
                raise ValueError(
                    f"{video_np.shape[0]} images is not a whole number of "
                    f"timesteps for num_views={num_views}"
                )

        # Supplied geometry is indexed on the flat image axis, so the timestep
        # window selected above spans num_views times as many entries. The
        # single-camera case keeps the base class's own selection.
        if num_views == 1:
            geom_frames, geom_max = frames, max_frames
        else:
            geom_frames = slice(start_frame * num_views, None, 1)
            geom_max = None if max_frames is None else max_frames * num_views

        def _cut(x):
            if x is None:
                return None
            x = x[geom_frames]
            return x if geom_max is None else x[:geom_max]

        has_geometry = (
            depths is not None and intrinsics is not None and extrinsics is not None
        )
        if num_views > 1 and not has_geometry:
            self._check_chunking(num_views)

        self.model.num_views = num_views
        try:
            output = super().infer(
                video=video_np,
                depths=_cut(depths),
                intrinsics=_cut(intrinsics),
                extrinsics=_cut(extrinsics),
                conf_mask=_cut(conf_mask),
                start_frame=0,
                frame_stride=1,
                max_frames=None,
                **kwargs,
            )
        finally:
            self.model.num_views = 1

        fields = {f.name: getattr(output, f.name) for f in dataclasses.fields(output)}
        return MultiViewOutput(num_views=num_views, **fields)


def load_tracker(args) -> MultiViewTracker:
    return MultiViewTracker.from_pretrained(
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


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-camera TrackEverything inference: run the tracker "
        "and save its tracks. Use demo_multiview.py to also visualize them. "
        "Pass one --video path per camera, comma-separated.",
    )
    add_inference_args(parser)
    parser.add_argument("--mode", type=str, default="dense", choices=["dense", "sparse"])
    parser.add_argument(
        "--query_points", type=str, default=None,
        help="Path to .npy file of 2D query points (N,3) with [t, x, y], where t "
        "is a flat image index (frame * num_views + view)",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/track_everything_mv")
    parser.add_argument(
        "--save_dense", action="store_true",
        help="Also save the dense (T, H, W, T) tensors; grows with (frames x cameras) squared",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

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
