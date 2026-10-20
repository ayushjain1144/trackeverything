# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation dataset creation functions."""

from typing import Mapping

import numpy as np
import torch


def compute_tapvid_metrics_torch(
    query_points: torch.Tensor,
    gt_occluded: torch.Tensor,
    gt_tracks: torch.Tensor,
    pred_occluded: torch.Tensor,
    pred_tracks: torch.Tensor,
    valids: torch.Tensor,
    query_mode: str,
    get_trackwise_metrics: bool = False,
) -> Mapping[str, torch.Tensor]:
    """Torch version of compute_tapvid_metrics.

    All inputs should be torch tensors instead of numpy arrays.
    Returns torch tensors instead of numpy arrays.

    Args:
       query_points: The query points, an in the format [t, y, x].  Its size is
         [b, n, 3], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number of
         frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is in the
         format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same format
         as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the same
         format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.
       get_trackwise_metrics: if True, the metrics will be computed for every
         track (rather than every video, which is the default).  This means
         every output tensor will have an extra axis [batch, num_tracks] rather
         than simply (batch).

    Returns:
        A dict with the following keys:

        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """
    summing_axis = (2,) if get_trackwise_metrics else (1, 2)

    metrics = {}

    eye = torch.eye(gt_tracks.shape[2], dtype=torch.int32, device=gt_tracks.device)
    if query_mode == "first":
        # evaluate frames after the query frame
        query_frame_to_eval_frames = torch.cumsum(eye, dim=1) - eye
    elif query_mode == "strided":
        # evaluate all frames except the query frame
        query_frame_to_eval_frames = 1 - eye
    elif query_mode == "all":
        pass
    else:
        raise ValueError("Unknown query mode " + query_mode)

    if query_mode != "all":
        query_frame = query_points[..., 0]
        query_frame = torch.round(query_frame).to(torch.int32)
        evaluation_points = query_frame_to_eval_frames[query_frame] > 0
        evaluation_points = torch.logical_and(evaluation_points, valids)
    else:
        evaluation_points = valids

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    occ_acc = torch.sum(
        (pred_occluded == gt_occluded) & evaluation_points,
        dim=summing_axis,
    ) / torch.sum(evaluation_points, dim=summing_axis)
    metrics["occlusion_accuracy"] = occ_acc

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = torch.logical_not(gt_occluded)
    pred_visible = torch.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []

    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = torch.sum(
            torch.square(pred_tracks - gt_tracks),
            dim=-1,
        ) < torch.square(torch.tensor(thresh, device=gt_tracks.device))
        is_correct = torch.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = torch.sum(
            is_correct & evaluation_points,
            dim=summing_axis,
        )
        count_visible_points = torch.sum(visible & evaluation_points, dim=summing_axis)
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = torch.sum(
            is_correct & pred_visible & evaluation_points, dim=summing_axis
        )

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = torch.sum(visible & evaluation_points, dim=summing_axis)
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = torch.sum(
            false_positives & evaluation_points, dim=summing_axis
        )
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)

    metrics["average_jaccard"] = torch.mean(
        torch.stack(all_jaccard, dim=1),
        dim=1,
    )
    metrics["average_pts_within_thresh"] = torch.mean(
        torch.stack(all_frac_within, dim=1),
        dim=1,
    )
    return metrics


def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    valids: np.ndarray,
    query_mode: str,
    get_trackwise_metrics: bool = False,
) -> Mapping[str, np.ndarray]:
    """Computes TAP-Vid metrics (Jaccard, Pts.

    Within Thresh, Occ.

    Acc.)

    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.

    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.

    Args:
       query_points: The query points, an in the format [t, y, x].  Its size is
         [b, n, 3], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number of
         frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is in the
         format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same format
         as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the same
         format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.
       get_trackwise_metrics: if True, the metrics will be computed for every
         track (rather than every video, which is the default).  This means
         every output tensor will have an extra axis [batch, num_tracks] rather
         than simply (batch).

    Returns:
        A dict with the following keys:

        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """
    #   import ipdb; ipdb.set_trace()
    summing_axis = (2,) if get_trackwise_metrics else (1, 2)

    metrics = {}

    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)
    if query_mode == "first":
        # evaluate frames after the query frame
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == "strided":
        # evaluate all frames except the query frame
        query_frame_to_eval_frames = 1 - eye

    # newly added
    elif query_mode == "all":
        pass
    else:
        raise ValueError("Unknown query mode " + query_mode)

    if query_mode != "all":
        query_frame = query_points[..., 0]
        query_frame = np.round(query_frame).astype(np.int32)
        evaluation_points = query_frame_to_eval_frames[query_frame] > 0

        evaluation_points = np.logical_and(evaluation_points, valids)
    else:
        evaluation_points = valids

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=summing_axis,
    ) / np.sum(evaluation_points, axis=summing_axis)
    metrics["occlusion_accuracy"] = occ_acc

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=summing_axis,
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=summing_axis)
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=summing_axis
        )

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=summing_axis)
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=summing_axis)
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    return metrics
