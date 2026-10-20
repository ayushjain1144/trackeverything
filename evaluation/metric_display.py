# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared display names for the core point-tracking metrics.

Keyed by the metric suffix as it appears in the aggregated ``values`` dict, i.e.
without the ``{dataset}-metrics/mean/`` namespace, without the eval-mode prefix
(``dense_``, ...) and without the split prefix (``static_only_``, ...).

Jaccard and occlusion accuracy are reported at the standard fixed visibility
threshold of 0.5, matching the official TAP-Vid / TAP-Vid-3D evaluators. The
``_best`` keys that the training loops derive by maximizing over the
``_visthr_`` sweep are oracle threshold selection and are deliberately not
displayed.
"""

CORE_DISPLAY_METRICS: dict[str, str] = {
    "tapvid3d_256_iso_average_pts_within_thresh": "3D-APD",
    "tapvid2d_256_average_pts_within_thresh": "2D-APD",
    "tapvid3d_256_iso_average_jaccard_visthr_0.50": "3D-AJ",
    "tapvid3d_256_iso_occlusion_accuracy_visthr_0.50": "OA",
    "tapvid3d_256_iso_epe3d": "3D-EPE",
}
