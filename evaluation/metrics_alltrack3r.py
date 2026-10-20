# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Dict, Optional

import ipdb
import numpy as np
import torch
from einops import rearrange, repeat
from torch.nn import functional as F

from ..datasets.datatypes import SliceData
from ..training.datatypes import Prediction
from ..utils.common_utils import ensure_float32
from . import tapvid2d_metrics, tapvid3d_metrics
from .alltrack3r_metric_utils import evaluate_mobility

st = ipdb.set_trace


def apply_intrinsics(pts: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    pts_proj = torch.einsum("...ij,...j->...i", intrinsics, pts)
    return pts_proj[..., :2] / pts_proj[..., 2:3]


def get_new_hw_with_given_smallest_side_length(
    *, orig_resolution: torch.Tensor, smallest_side_length: int = 256
):
    scaling_factor = (
        torch.full_like(orig_resolution[..., 0], smallest_side_length)
        / orig_resolution.min(dim=-1).values
    )
    return scaling_factor


def _restore_original_metric_scale(
    sample: SliceData,
    gt_tracks: torch.Tensor,
    pred_tracks: torch.Tensor,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not enabled:
        return gt_tracks, pred_tracks

    def _scale_tracks(tracks: torch.Tensor, norm_factor) -> torch.Tensor:
        if not isinstance(norm_factor, torch.Tensor):
            return tracks
        scale = norm_factor.to(device=tracks.device, dtype=tracks.dtype)
        while scale.ndim < tracks.ndim:
            scale = scale[..., None]
        return tracks * scale

    return (
        _scale_tracks(gt_tracks, sample.gt_norm_factor),
        _scale_tracks(pred_tracks, sample.est_norm_factor),
    )


def _compute_metrics_quick(
    sample: SliceData,
    preds: Prediction,
    bidirectional: bool,
    evaluate_with_query_points: bool = False,
    debug_mode: bool = False,
    prefix: str = "",
    gt_extrinsics = None,
    track_all: bool = False,
    use_fixed_metric_threshold: bool = False,
    use_fixed_metric_threshold_startrack: bool = False,
) -> Dict[str, torch.Tensor]:
    sample = sample.copy().with_annot_mode(None)
    gt_sample = sample.with_annot_mode("gt")

    est_sample = sample.with_annot_mode("est")

    # things should be in cam0 coordinate frame, especialy when dealing with predicted geometry
    # try:
    #     assert torch.allclose(
    #             est_sample.extrinsics[:, 0],
    #             torch.eye(4, device=est_sample.extrinsics.device)[None].expand(
    #                 est_sample.extrinsics.shape[0], -1, -1
    #             ),
    #             atol=1e-1,
    #     )
    # except:
    #     st()

    # assert torch.allclose(
    #     est_sample.extrinsics[:, 0],
    #     torch.eye(4, device=est_sample.extrinsics.device)[None].expand(
    #         est_sample.extrinsics.shape[0], -1, -1
    #     ),
    #     atol=1e-1,
    # )

    # diff = ((gt_tracks - pred_tracks) ** 2).sum(-1)[gt_sample.valids].mean()
    # gt_sample.gt_extrinsics = torch.einsum(
    #     'b t i j, b t j k -> b t i k',
    #     gt_sample.extrinsics[0:1],
    #     gt_sample.extrinsics.inverse()
    # )
    gt_tracks = torch.einsum(
        "b t i j, b t n j -> b t n i",
        gt_sample.extrinsics,
        torch.cat(
            [gt_sample.trajs_3d, torch.ones_like(gt_sample.trajs_3d[..., :1])], dim=-1
        ),
    )[..., :3]

    # use gt extrinsics for predicted tracks too, but before that scale the trajs
    # pred_trajs = preds.coords * sample.pd_to_gt_scale
    pred_trajs = preds.coords
    pred_tracks = torch.einsum(
        "b t i j, b t n j -> b t n i",
        gt_extrinsics if gt_extrinsics is not None else est_sample.extrinsics,
        torch.cat([pred_trajs, torch.ones_like(pred_trajs[..., :1])], dim=-1),
    )[..., :3]

    B, T, C, H, W = sample.rgbs.shape

    scaling_factor = get_new_hw_with_given_smallest_side_length(
        orig_resolution=sample.orig_resolution,
        smallest_side_length=256,
    )

    gt_resized_intrinsics = gt_sample.intrinsics.clone()
    gt_resized_intrinsics[..., 0, 0] *= (256) / (W)
    gt_resized_intrinsics[..., 1, 1] *= (256) / (H)
    gt_resized_intrinsics[..., 0, 2] *= (256) / (W)
    gt_resized_intrinsics[..., 1, 2] *= (256) / (H)

    est_resized_intrinsics = est_sample.intrinsics.clone()
    est_resized_intrinsics[..., 0, 0] *= (256) / (W)
    est_resized_intrinsics[..., 1, 1] *= (256) / (H)
    est_resized_intrinsics[..., 0, 2] *= (256) / (W)
    est_resized_intrinsics[..., 1, 2] *= (256) / (H)

    # to match exactly with the implementation in https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid3d/evaluation/evaluate_model.py
    gt_resized_intrinsics_iso = gt_sample.intrinsics.clone()
    gt_resized_intrinsics_iso[..., 0, 0] *= (
        ((sample.orig_resolution[..., 1]) / (W)) * scaling_factor
    )[:, None]
    gt_resized_intrinsics_iso[..., 1, 1] *= (
        ((sample.orig_resolution[..., 0]) / (H)) * scaling_factor
    )[:, None]
    gt_resized_intrinsics_iso[..., 0, 2] *= (
        ((sample.orig_resolution[..., 1]) / (W)) * scaling_factor
    )[:, None]
    gt_resized_intrinsics_iso[..., 1, 2] *= (
        ((sample.orig_resolution[..., 0]) / (H)) * scaling_factor
    )[:, None]

    gt_resized_intrinsics_params_iso = torch.stack(
        [
            gt_resized_intrinsics_iso[..., 0, 0],
            gt_resized_intrinsics_iso[..., 1, 1],
            gt_resized_intrinsics_iso[..., 0, 2],
            gt_resized_intrinsics_iso[..., 1, 2],
        ],
        dim=-1,
    )

    gt_query_points = gt_sample.query_point.clone()
    est_query_points = est_sample.query_point.clone()
    gt_query_frames = gt_query_points[..., 0].long()
    est_query_frames = est_query_points[..., 0].long()
    gt_extrinsics_at_query_frames = gt_sample.extrinsics[
        repeat(torch.arange(B), "b -> b n", n=gt_query_frames.shape[1]),
        gt_query_frames,
    ]
    if gt_extrinsics is not None:
        est_extrinsics_at_query_frames = gt_extrinsics[
            repeat(torch.arange(B), "b -> b n", n=gt_query_frames.shape[1]),
            gt_query_frames,
        ]
    else:
        est_extrinsics_at_query_frames = est_sample.extrinsics[
            repeat(torch.arange(B), "b -> b n", n=est_query_frames.shape[1]),
            est_query_frames,
        ]
    gt_resized_intrinsics_at_query_frames = gt_resized_intrinsics[
        repeat(torch.arange(B), "b -> b n", n=gt_query_frames.shape[1]),
        gt_query_frames,
    ]
    est_resized_intrinsics_at_query_frames = est_resized_intrinsics[
        repeat(torch.arange(B), "b -> b n", n=est_query_frames.shape[1]),
        est_query_frames,
    ]
    gt_query_points[..., 1:] = torch.einsum(
        "b n i j, b n j -> b n i",
        gt_extrinsics_at_query_frames,
        torch.cat(
            [gt_query_points[..., 1:], torch.ones_like(gt_query_points[..., :1])],
            dim=-1,
        ),
    )[..., :3]
    est_query_points[..., 1:] = torch.einsum(
        "b n i j, b n j -> b n i",
        est_extrinsics_at_query_frames,
        torch.cat(
            [est_query_points[..., 1:], torch.ones_like(est_query_points[..., :1])],
            dim=-1,
        ),
    )[..., :3]
    gt_query_points_2d_resized = torch.cat(
        [
            gt_query_points[..., :1],
            apply_intrinsics(
                gt_query_points[..., 1:], gt_resized_intrinsics_at_query_frames
            ),
        ],
        dim=-1,
    )

    gt_tracks_2d_resized = apply_intrinsics(
        gt_tracks,
        repeat(gt_resized_intrinsics, "b t i j -> b t n i j", n=gt_tracks.shape[2]),
    )
    pred_tracks_2d_resized = apply_intrinsics(
        pred_tracks,
        repeat(est_resized_intrinsics, "b t i j -> b t n i j", n=pred_tracks.shape[2]),
    )

    metrics_dict = {}

    gt_occluded = ~sample.visibs
    valids = sample.valids
    visib_probs = torch.sigmoid(preds.visibs)
    vis_related_keys = set()

    threshold = 0.5
    pred_occluded = visib_probs <= threshold
    use_metric_scale_thresholds = (
        use_fixed_metric_threshold or use_fixed_metric_threshold_startrack
    )

    # print("CHEATING FOR NOW")
    # pred_occluded = gt_occluded
    metrics_dict_ = {}
    # valids_ = valids.clone()
    # valids_[:, :-1] = False
    # gt_occluded_ = gt_occluded.clone()
    # gt_occluded_[:, :-1] = True
    # tapvid3d_256_iso_metrics_cuda: Dict[str, np.ndarray] = (
    #     tapvid3d_metrics.compute_tapvid3d_metrics_torch(
    #             gt_occluded=gt_occluded_,
    #             gt_tracks=gt_tracks,
    #             pred_occluded=pred_occluded,
    #             pred_tracks=pred_tracks,
    #             query_points=gt_query_points,
    #             intrinsics_params=gt_resized_intrinsics_params_iso,
    #             valids=valids_,
    #             order="b t n",
    #             bidirectional=bidirectional,
    #             debug_mode=debug_mode,
    #             # scaling="none",
    #         )
    #     )
    # )  # type: ignore
    tapvid3d_gt_tracks, tapvid3d_pred_tracks = _restore_original_metric_scale(
        sample,
        gt_tracks,
        pred_tracks,
        use_metric_scale_thresholds,
    )
    tapvid3d_256_iso_metrics_cuda: Dict[str, np.ndarray] = (
        tapvid3d_metrics.compute_tapvid3d_metrics_torch(
            gt_occluded=gt_occluded,
            gt_tracks=tapvid3d_gt_tracks,
            pred_occluded=pred_occluded,
            pred_tracks=tapvid3d_pred_tracks,
            query_points=gt_query_points,
            intrinsics_params=gt_resized_intrinsics_params_iso,
            valids=valids,
            order="b t n",
            bidirectional=bidirectional,
            debug_mode=debug_mode,
            use_fixed_metric_threshold=use_fixed_metric_threshold,
            use_fixed_metric_threshold_startrack=use_fixed_metric_threshold_startrack,
        )
    )  # type: ignore
    metrics_dict_.update(
        {f"{prefix}tapvid3d_256_iso_{k}": v for k, v in tapvid3d_256_iso_metrics_cuda.items()}
    )

    # torch version of tapvid2d_metrics.compute_tapvid_metrics
    try:
        if track_all:
            query_mode = "all"
        else:
            query_mode = "strided" if bidirectional else "first"
        tapvid2d_256_metrics_cuda: Dict[str, np.ndarray] = (
            tapvid2d_metrics.compute_tapvid_metrics_torch(
                query_points=gt_query_points_2d_resized[..., [0, 2, 1]],  # [t, x, y] -> [t, y, x]
                gt_occluded=rearrange(gt_occluded, "b t n -> b n t"),
                gt_tracks=rearrange(gt_tracks_2d_resized, "b t n i -> b n t i"),
                pred_occluded=rearrange(pred_occluded, "b t n -> b n t"),
                pred_tracks=rearrange(pred_tracks_2d_resized, "b t n i -> b n t i"),
                # query_mode="all",
                query_mode=query_mode,
                valids=rearrange(valids, "b t n -> b n t"),
                get_trackwise_metrics=False,
            )
        )  # type: ignore
    except:
        st()

    metrics_dict_.update(
        {f"{prefix}tapvid2d_256_{k}": v for k, v in tapvid2d_256_metrics_cuda.items()}
    )

    vis_related_keys_ = [
        k for k in metrics_dict_.keys() if "occlusion_accuracy" in k or "jaccard" in k
    ]
    vis_related_keys.update(vis_related_keys_)
    vis_related_metrics = {k: metrics_dict_.pop(k) for k in vis_related_keys_}
    if len(metrics_dict) == 0:
        metrics_dict = metrics_dict_
    for k, v in vis_related_metrics.items():
        metrics_dict[f"{k}_visthr_{threshold:.2f}"] = v

    metrics_dict = {
        k: torch.tensor(v, device=sample.rgbs.device, dtype=torch.float32)
        for k, v in metrics_dict.items()
    }
    return metrics_dict


# @ensure_float32(allow_cast=False)
def _compute_metrics(
    sample: SliceData, preds: Prediction, bidirectional: bool
) -> Dict[str, torch.Tensor]:
    sample = sample.copy().with_annot_mode(None)
    gt_sample = sample.with_annot_mode("gt")

    # for now atleast
    est_sample = gt_sample
    # est_sample = sample.with_annot_mode("est")

    gt_tracks = torch.einsum(
        "b t i j, b t n j -> b t n i",
        gt_sample.extrinsics,
        torch.cat(
            [gt_sample.trajs_3d, torch.ones_like(gt_sample.trajs_3d[..., :1])], dim=-1
        ),
    )[..., :3]
    pred_tracks = torch.einsum(
        "b t i j, b t n j -> b t n i",
        est_sample.extrinsics,
        torch.cat([preds.coords, torch.ones_like(preds.coords[..., :1])], dim=-1),
    )[..., :3]

    # gt_intrinsics_params = torch.stack([
    #     gt_sample.intrinsics[..., 0, 0],
    #     gt_sample.intrinsics[..., 1, 1],
    #     gt_sample.intrinsics[..., 0, 2],
    #     gt_sample.intrinsics[..., 1, 2],
    # ], dim=-1).cpu().numpy()

    B, T, C, H, W = sample.rgbs.shape

    scaling_factor = get_new_hw_with_given_smallest_side_length(
        orig_resolution=sample.orig_resolution,
        smallest_side_length=256,
    )

    gt_resized_intrinsics = gt_sample.intrinsics.clone()
    gt_resized_intrinsics[..., 0, 0] *= (256 - 1) / (W - 1)
    gt_resized_intrinsics[..., 1, 1] *= (256 - 1) / (H - 1)
    gt_resized_intrinsics[..., 0, 2] *= (256 - 1) / (W - 1)
    gt_resized_intrinsics[..., 1, 2] *= (256 - 1) / (H - 1)

    est_resized_intrinsics = est_sample.intrinsics.clone()
    est_resized_intrinsics[..., 0, 0] *= (256 - 1) / (W - 1)
    est_resized_intrinsics[..., 1, 1] *= (256 - 1) / (H - 1)
    est_resized_intrinsics[..., 0, 2] *= (256 - 1) / (W - 1)
    est_resized_intrinsics[..., 1, 2] *= (256 - 1) / (H - 1)

    # to match exactly with the implementation in https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid3d/evaluation/evaluate_model.py
    gt_resized_intrinsics_iso = gt_sample.intrinsics.clone()
    gt_resized_intrinsics_iso[..., 0, 0] *= (
        ((sample.orig_resolution[..., 1] - 1) / (W - 1)) * scaling_factor
    )[:, None]
    gt_resized_intrinsics_iso[..., 1, 1] *= (
        ((sample.orig_resolution[..., 0] - 1) / (H - 1)) * scaling_factor
    )[:, None]
    gt_resized_intrinsics_iso[..., 0, 2] *= (
        ((sample.orig_resolution[..., 1] - 1) / (W - 1)) * scaling_factor
    )[:, None]
    gt_resized_intrinsics_iso[..., 1, 2] *= (
        ((sample.orig_resolution[..., 0] - 1) / (H - 1)) * scaling_factor
    )[:, None]

    gt_resized_intrinsics_params_iso = (
        torch.stack(
            [
                gt_resized_intrinsics_iso[..., 0, 0],
                gt_resized_intrinsics_iso[..., 1, 1],
                gt_resized_intrinsics_iso[..., 0, 2],
                gt_resized_intrinsics_iso[..., 1, 2],
            ],
            dim=-1,
        )
        .cpu()
        .numpy()
    )

    gt_tracks_2d_resized = apply_intrinsics(
        gt_tracks,
        repeat(gt_resized_intrinsics, "b t i j -> b t n i j", n=gt_tracks.shape[2]),
    )
    pred_tracks_2d_resized = apply_intrinsics(
        pred_tracks,
        repeat(est_resized_intrinsics, "b t i j -> b t n i j", n=pred_tracks.shape[2]),
    )

    # gt_query_points = gt_query_points.cpu().numpy()
    # est_query_points = est_query_points.cpu().numpy()
    gt_tracks = gt_tracks.cpu().numpy()
    pred_tracks = pred_tracks.cpu().numpy()
    gt_tracks_2d_resized = gt_tracks_2d_resized.cpu().numpy()
    pred_tracks_2d_resized = pred_tracks_2d_resized.cpu().numpy()

    metrics_dict = {}

    gt_occluded = ~sample.visibs.cpu().numpy()
    valids = sample.valids.cpu().numpy()
    visib_probs = (torch.sigmoid(preds.visibs)).cpu().numpy()
    vis_related_keys = set()

    for threshold in np.linspace(0, 1, 11)[1:-1]:
        pred_occluded = visib_probs <= threshold
        metrics_dict_ = {}

        start_time = time.time()

        # with torch.device("cuda"):
        tapvid3d_256_iso_metrics: Dict[str, np.ndarray] = (
            tapvid3d_metrics.compute_tapvid3d_metrics(
                gt_occluded=gt_occluded,
                gt_tracks=gt_tracks,
                pred_occluded=pred_occluded,
                pred_tracks=pred_tracks,
                query_points=None,
                intrinsics_params=gt_resized_intrinsics_params_iso,
                valids=valids,
                order="b t n",
                bidirectional=bidirectional,
            )
        )  # type: ignore
        metrics_dict_.update(
            {f"tapvid3d_256_iso_{k}": v for k, v in tapvid3d_256_iso_metrics.items()}
        )

        end_time = time.time()
        print(f"Time taken for tapvid3d_256_iso_metrics: {end_time - start_time}")

        start_time = time.time()
        tapvid2d_256_metrics: Dict[str, np.ndarray] = (
            tapvid2d_metrics.compute_tapvid_metrics(
                query_points=None,  # [t, x, y] -> [t, y, x]
                gt_occluded=rearrange(gt_occluded, "b t n -> b n t"),
                gt_tracks=rearrange(gt_tracks_2d_resized, "b t n i -> b n t i"),
                pred_occluded=repeat(pred_occluded, "b t n -> b n t"),
                pred_tracks=rearrange(pred_tracks_2d_resized, "b t n i -> b n t i"),
                # query_mode="all",
                query_mode="strided" if bidirectional else "first",
                valids=rearrange(valids, "b t n -> b n t"),
                get_trackwise_metrics=False,
            )
        )  # type: ignore
        metrics_dict_.update(
            {f"tapvid2d_256_{k}": v for k, v in tapvid2d_256_metrics.items()}
        )
        end_time = time.time()
        print(f"Time taken for tapvid2d_256_metrics: {end_time - start_time}")

        vis_related_keys_ = [
            k
            for k in metrics_dict_.keys()
            if "occlusion_accuracy" in k or "jaccard" in k
        ]
        vis_related_keys.update(vis_related_keys_)
        vis_related_metrics = {k: metrics_dict_.pop(k) for k in vis_related_keys_}
        if len(metrics_dict) == 0:
            metrics_dict = metrics_dict_
        for k, v in vis_related_metrics.items():
            metrics_dict[f"{k}_visthr_{threshold:.2f}"] = v

    metrics_dict = {
        k: torch.tensor(v, device=sample.rgbs.device, dtype=torch.float32)
        for k, v in metrics_dict.items()
    }
    return metrics_dict


def _window_dynamic_from_trajs(
    trajs_3d: torch.Tensor, threshold, norm_factor=None, valids=None
) -> torch.Tensor:
    """Per-point window motion label, matching the criterion's supervision GT.

    trajs_3d: (B, T, N, 3). Returns (B, N) bool: True if the point's xyz
    bounding-box diagonal over the window exceeds `threshold`.

    ``valids`` (B, T, N) restricts the extent to frames where the point is valid,
    matching the criterion and the providers. Coordinates on invalid frames are
    not meaningful and one junk value labels a stationary point dynamic; without
    this the metric would score against a different GT than training used.
    See docs/static_dynamic_labels.md.

    If ``norm_factor`` is given, ``trajs_3d`` is assumed scene-normalized and the
    motion is rescaled back to metric (pre-normalization) units so ``threshold``
    is interpreted in the same metric space as the dense GT. ``threshold`` may be
    a scalar or a per-sample (B,) tensor.
    """
    if trajs_3d.shape[1] <= 1:
        return torch.zeros(
            trajs_3d.shape[0], trajs_3d.shape[2],
            device=trajs_3d.device, dtype=torch.bool,
        )
    if valids is None:
        max_xyz = torch.amax(trajs_3d, dim=1)
        min_xyz = torch.amin(trajs_3d, dim=1)
        diff_l2 = torch.linalg.norm(max_xyz - min_xyz, dim=-1)  # (B, N)
    else:
        v = valids.bool()
        any_valid = v.any(dim=1)
        # Points with no valid frame: keep every frame so the reduction stays
        # finite, then force them static below (they are never supervised).
        keep = (v | (~any_valid)[:, None, :])[..., None]
        big = torch.finfo(trajs_3d.dtype).max
        max_xyz = torch.where(keep, trajs_3d, -big).amax(dim=1)
        min_xyz = torch.where(keep, trajs_3d, big).amin(dim=1)
        diff_l2 = torch.linalg.norm(max_xyz - min_xyz, dim=-1)  # (B, N)
        diff_l2 = torch.where(any_valid, diff_l2, torch.zeros_like(diff_l2))
    if norm_factor is not None:
        diff_l2 = diff_l2 * norm_factor.reshape(-1, 1).to(diff_l2)
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.reshape(-1, 1).to(diff_l2)
    return diff_l2 > threshold


def _window_dynamic_from_trajs_per_frame(
    trajs_3d: torch.Tensor, threshold, window_size, norm_factor=None, valids=None
) -> torch.Tensor:
    """Per-frame per-point window motion label.

    trajs_3d: (B, T, N, 3). Windows are the fixed non-overlapping tiling induced
    by ``window_size`` (``[0, w), [w, 2w), ...``, trailing window clipped to T),
    matching the model's window schedule. Each frame ``t`` is assigned the motion
    label of the window it belongs to. Returns (B, T, N) bool. Falls back to a
    single whole-clip window when ``window_size`` is None (i.e. legacy per-video).
    """
    Bn, T, N, _ = trajs_3d.shape
    out = torch.zeros(Bn, T, N, dtype=torch.bool, device=trajs_3d.device)
    if not window_size or window_size <= 0:
        window_size = T
    w = int(window_size)
    for ws in range(0, T, w):
        we = min(ws + w, T)
        lab = _window_dynamic_from_trajs(
            trajs_3d[:, ws:we], threshold, norm_factor=norm_factor,
            valids=None if valids is None else valids[:, ws:we],
        )  # (B, N)
        out[:, ws:we] = lab[:, None]
    return out


def _mobility_metrics_from_logits(
    pred_logits: torch.Tensor,
    gt_dynamic: torch.Tensor,
    valid: torch.Tensor,
    threshold: float,
) -> Dict[str, torch.Tensor]:
    """Threshold-aware version of evaluate_mobility.

    Reports overall / dynamic / static accuracy, plus precision / recall / F1
    for both classes, binarizing predictions at
    ``sigmoid(logit) > threshold`` (the model's configured dynamic_threshold).
    Also reports the GT dynamic fraction for context. All values are scalar
    tensors; empty splits yield NaN so downstream nanmean skips them.
    """
    device = pred_logits.device
    nan = torch.tensor(float("nan"), device=device)

    pred_logits = pred_logits.float()[valid]
    gt_dynamic = gt_dynamic[valid].bool()

    empty = {
        "correct_all": nan,
        "correct_dynamic": nan,
        "correct_static": nan,
        "dynamic_precision": nan,
        "dynamic_recall": nan,
        "dynamic_f1": nan,
        "static_precision": nan,
        "static_recall": nan,
        "static_f1": nan,
        "gt_dynamic_frac": nan,
    }
    if gt_dynamic.numel() == 0:
        return empty

    pred_dynamic = pred_logits.sigmoid() > threshold
    pred_static = ~pred_dynamic
    gt_static = ~gt_dynamic
    correct = (pred_dynamic == gt_dynamic).float()

    has_dyn = gt_dynamic.any()
    has_stat = gt_static.any()
    has_pred_dyn = pred_dynamic.any()
    has_pred_stat = pred_static.any()

    tp_dyn = (pred_dynamic & gt_dynamic).sum().float()
    tp_stat = (pred_static & gt_static).sum().float()

    dynamic_precision = (
        tp_dyn / pred_dynamic.sum().clamp(min=1) if has_pred_dyn else nan
    )
    dynamic_recall = (
        tp_dyn / gt_dynamic.sum().clamp(min=1) if has_dyn else nan
    )
    static_precision = (
        tp_stat / pred_static.sum().clamp(min=1) if has_pred_stat else nan
    )
    static_recall = (
        tp_stat / gt_static.sum().clamp(min=1) if has_stat else nan
    )

    def _f1(precision: torch.Tensor, recall: torch.Tensor) -> torch.Tensor:
        if torch.isnan(precision) or torch.isnan(recall):
            return nan
        denom = precision + recall
        return (2 * precision * recall / denom) if denom > 0 else nan

    return {
        "correct_all": correct.mean(),
        "correct_dynamic": correct[gt_dynamic].mean() if has_dyn else nan,
        "correct_static": correct[gt_static].mean() if has_stat else nan,
        "dynamic_precision": dynamic_precision,
        "dynamic_recall": dynamic_recall,
        "dynamic_f1": _f1(dynamic_precision, dynamic_recall),
        "static_precision": static_precision,
        "static_recall": static_recall,
        "static_f1": _f1(static_precision, static_recall),
        "gt_dynamic_frac": gt_dynamic.float().mean(),
    }


def _sd_metrics_for_point_tracks(
    trajs_3d: torch.Tensor,
    valids: torch.Tensor,
    pred_logits: torch.Tensor,
    filter_threshold,
    window_size: Optional[int],
    norm_factor,
    dynamic_threshold: float,
) -> Optional[Dict[str, torch.Tensor]]:
    """Static/dynamic metrics for query-shaped point tracks.

    GT is per-window motion of ``trajs_3d`` via ``_window_dynamic_from_trajs_per_frame``.
    Returns None if pred logits cannot be aligned to the GT shape.
    """
    gt_dynamic = _window_dynamic_from_trajs_per_frame(
        trajs_3d,
        filter_threshold,
        window_size,
        norm_factor=norm_factor,
        valids=valids,
    )  # (B, T, N)

    return _mobility_metrics_from_logits(
        pred_logits, gt_dynamic, valids, dynamic_threshold
    )


def compute_static_dynamic_metrics(
    sample: SliceData,
    preds: Prediction,
    dynamic_threshold: float = 0.5,
    dynamic_filter_threshold: float = 0.1,
    prefix: str = "",
    window_size: Optional[int] = None,
    tag: str = "query",
) -> Dict[str, torch.Tensor]:
    """Static/dynamic classification metrics for query-shaped point tracks.

    Compares ``preds.query_dynamic_logits`` against per-window GT motion of
    ``sample.trajs_3d`` (``_window_dynamic_from_trajs_per_frame`` in metric
    units, using the fixed ``window_size`` tiling).

    Call on the raw sample/preds with ``tag="query"``, or on a prepared
    dense_sample eval variant (from ``prepare_dense_sample_tracks_for_eval``)
    with ``tag="dense"``.

    The GT uses the per-sample metric threshold ``sample.dynamic_filter_threshold``
    (falling back to the passed ``dynamic_filter_threshold``) and rescales
    normalized trajectories via ``sample.gt_norm_factor``. Predictions are
    binarized at the model's ``dynamic_threshold``. When ``window_size`` is None
    the GT collapses to a single whole-clip window (legacy per-video behavior).

    Returns scalar tensors keyed ``{prefix}sd_{tag}_*`` covering accuracy,
    precision, recall, and F1 for both classes.
    """
    sample = sample.copy().with_annot_mode(None)
    gt_sample = sample.with_annot_mode("gt")
    out: Dict[str, torch.Tensor] = {}

    filter_threshold = dynamic_filter_threshold
    if gt_sample.dynamic_filter_threshold is not None:
        filter_threshold = gt_sample.dynamic_filter_threshold

    if preds.query_dynamic_logits is not None and gt_sample.trajs_3d is not None:
        q_metrics = _sd_metrics_for_point_tracks(
            gt_sample.trajs_3d,
            sample.valids,
            preds.query_dynamic_logits,
            filter_threshold,
            window_size,
            gt_sample.gt_norm_factor,
            dynamic_threshold,
        )
        if q_metrics is not None:
            out.update({f"{prefix}sd_{tag}_{k}": v for k, v in q_metrics.items()})

    return out


@ensure_float32(allow_cast=False)
def compute_metrics(
    sample: SliceData,
    preds: Prediction,
    bidirectional: bool,
    evaluate_with_query_points: bool = False,
    dynamic_filter_threshold: float = 0.1,
    debug_mode: bool = False,
    prefix: str = "",
    gt_extrinsics = None,
    track_all: bool = False,
    use_fixed_metric_threshold: bool = False,
    use_fixed_metric_threshold_startrack: bool = False,
) -> Dict[str, torch.Tensor]:
    compute_metrics = _compute_metrics_quick
    # if True:
    #     compute_metrics = _compute_metrics_quick
    # else:
    #     compute_metrics = _compute_metrics
    # evaluation on all points

    # sample, preds = find_annotated_tracks_from_all_tracks(sample, preds)
    metrics_dict_base = compute_metrics(
        sample, preds, bidirectional, evaluate_with_query_points, debug_mode=debug_mode,
        prefix=prefix, 
        gt_extrinsics=gt_extrinsics,
        track_all=track_all,
        use_fixed_metric_threshold=use_fixed_metric_threshold,
        use_fixed_metric_threshold_startrack=use_fixed_metric_threshold_startrack,
    )

    # evaluation on visible points
    # occ_sample = sample.copy()
    # occ_sample.valids = sample.visibs & sample.valids
    # occ_metrics_dict = compute_metrics(
    #     occ_sample, preds, bidirectional, evaluate_with_query_points
    # )
    # occ_metrics_dict = {f"vis_only_{k}": v for k, v in occ_metrics_dict.items()}

    # metrics_dict_base.update(occ_metrics_dict)

    # evaluation on occluded points
    occ_sample = sample.copy()
    occ_sample.visibs = ~sample.visibs & sample.valids
    if occ_sample.dense_visibs is not None and occ_sample.dense_visibs[0] is not None:
        occ_sample.dense_visibs = ~occ_sample.dense_visibs & occ_sample.dense_valids
    occ_metrics_dict = compute_metrics(
        occ_sample, preds, bidirectional, evaluate_with_query_points,
        prefix=prefix,
        gt_extrinsics=gt_extrinsics,
        track_all=track_all,
        use_fixed_metric_threshold=use_fixed_metric_threshold,
        use_fixed_metric_threshold_startrack=use_fixed_metric_threshold_startrack,
    )
    occ_metrics_dict = {f"occ_only_{k}": v for k, v in occ_metrics_dict.items()}

    metrics_dict_base.update(occ_metrics_dict)

    # Eval variants can lack per-point dynamic labels (e.g. dense_sample on a
    # dataset without dense_sample_dynamic), in which case there is no split to
    # report. The caller does not wrap this in try/except, so returning here
    # keeps one unlabeled dataset from aborting the whole eval.
    if sample.is_dynamic is None:
        return metrics_dict_base

    dynamic_sample = sample.copy()
    dynamic_sample.valids = sample.is_dynamic & sample.valids
    if dynamic_sample.dense_valids is not None and dynamic_sample.dense_valids[0] is not None and not track_all:
        dense_dynamic = dynamic_sample.dense_dynamic
        if dense_dynamic.dim() == dynamic_sample.dense_valids.dim():
            dense_dynamic = dense_dynamic.any(dim=-1)
        dynamic_sample.dense_valids = dense_dynamic[..., None] & dynamic_sample.dense_valids
    dynamic_metrics_dict = compute_metrics(
        dynamic_sample, preds, bidirectional, evaluate_with_query_points,
        prefix=prefix,
        gt_extrinsics=gt_extrinsics,
        track_all=track_all,
        use_fixed_metric_threshold=use_fixed_metric_threshold,
        use_fixed_metric_threshold_startrack=use_fixed_metric_threshold_startrack,
    )
    dynamic_metrics_dict = {
        f"dynamic_only_{k}": v for k, v in dynamic_metrics_dict.items()
    }
    static_sample = sample.copy()
    static_sample.valids = ~sample.is_dynamic & sample.valids
    if static_sample.dense_valids is not None and static_sample.dense_valids[0] is not None and not track_all:
        dense_dynamic = static_sample.dense_dynamic
        if dense_dynamic.dim() == static_sample.dense_valids.dim():
            dense_dynamic = dense_dynamic.any(dim=-1)
        static_sample.dense_valids = ~dense_dynamic[..., None] & static_sample.dense_valids
    static_metrics_dict = compute_metrics(
        static_sample, preds, bidirectional, evaluate_with_query_points,
        prefix=prefix,
        gt_extrinsics=gt_extrinsics,
        track_all=track_all,
        use_fixed_metric_threshold=use_fixed_metric_threshold,
        use_fixed_metric_threshold_startrack=use_fixed_metric_threshold_startrack,
    )
    static_metrics_dict = {
        f"static_only_{k}": v for k, v in static_metrics_dict.items()
    }

    metrics_dict_base.update(dynamic_metrics_dict)
    metrics_dict_base.update(static_metrics_dict)

    return metrics_dict_base

    # evaluation of mobility

    # from explicit prediction
    # mobility_eval_dict = evaluate_mobility(
    #     preds.is_dynamic,
    #     sample.is_dynamic,
    # )
    # metrics_dict_base.update(mobility_eval_dict)

    # # mobility directly from the predicted tracks

    # max_xyz = preds.coords.abs().max(dim=1).values
    # min_xyz = preds.coords.abs().min(dim=1).values
    # diff_l2 = torch.linalg.norm(max_xyz - min_xyz, dim=-1)
    # pred_mobility = diff_l2 > dynamic_filter_threshold

    # mobility_eval_dict = evaluate_mobility(
    #     pred_mobility,
    #     sample.is_dynamic,
    # )
    # metrics_dict_base.update(
    #     {f"mobility_from_trajs_{k}": v for k, v in mobility_eval_dict.items()}
    # )

    # return metrics_dict_base
