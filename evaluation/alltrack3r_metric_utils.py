# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def evaluate_mobility(mobility_pred, mobility_gt):
    """Binary static/dynamic classification metrics.

    ``mobility_pred`` is treated as a hard prediction after ``> 0`` (True =
    dynamic). Returns accuracy, precision, recall, and F1 for both classes.
    Empty class splits yield NaN.
    """
    pred_dynamic = (mobility_pred > 0).bool()
    gt_dynamic = mobility_gt.bool()
    pred_static = ~pred_dynamic
    gt_static = ~gt_dynamic

    correct = (pred_dynamic == gt_dynamic).float()
    nan = correct.new_tensor(float("nan"))

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

    def _f1(precision, recall):
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
    }
