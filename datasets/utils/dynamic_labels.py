# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Static/dynamic ground-truth labelling for trajectories.

A point is labelled dynamic when its 3D position moves more than a threshold
over the window, measured as the L2 norm of the per-axis (max - min) extent.

The subtlety this module exists for: **the extent must be measured over valid
frames only**. Coordinates on invalid frames are not meaningful -- providers
leave whatever the annotation happened to contain there, which can be a filler
value, a stale pose or a wild extrapolation. Taking min/max across all frames
lets a single junk coordinate stretch the extent and mislabel a stationary point
as dynamic. Measured on Point Odyssey: 5365 mislabelled points in one `kg_ego2`
window.

Providers whose ``valids`` is all-true (most of them build it as
``np.ones_like(visibs)``) are unaffected either way, and can pass ``valids=None``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def dynamic_extent_l2(
    trajs_3d: np.ndarray, valids: Optional[np.ndarray] = None
) -> np.ndarray:
    """L2 extent of each trajectory over the window.

    Args:
        trajs_3d: ``(T, N, 3)`` world-space positions.
        valids: ``(T, N)`` boolean. Frames that are False are excluded from the
            extent. ``None`` treats every frame as valid.

    Returns:
        ``(N,)`` float array. Points with no valid frame at all get 0.0, i.e.
        they are labelled static -- they are never supervised, so calling them
        dynamic would only pollute the dynamic class.
    """
    if trajs_3d.ndim != 3 or trajs_3d.shape[-1] != 3:
        raise ValueError(f"trajs_3d must be (T, N, 3), got {trajs_3d.shape}")

    if valids is None:
        max_xyz = np.max(trajs_3d, axis=0)
        min_xyz = np.min(trajs_3d, axis=0)
        return np.linalg.norm(max_xyz - min_xyz, axis=-1)

    valids = np.asarray(valids, dtype=bool)
    if valids.shape != trajs_3d.shape[:2]:
        raise ValueError(
            f"valids must be (T, N) matching trajs_3d, got {valids.shape} "
            f"vs {trajs_3d.shape[:2]}"
        )

    any_valid = valids.any(axis=0)
    masked = np.where(valids[..., None], trajs_3d, np.nan)
    # Substitute frame 0 for points with no valid frame at all so nanmax/nanmin
    # do not warn; their result is overwritten via `any_valid` below.
    masked = np.where(any_valid[None, :, None], masked, trajs_3d[:1])
    max_xyz = np.nanmax(masked, axis=0)
    min_xyz = np.nanmin(masked, axis=0)
    diff_l2 = np.linalg.norm(max_xyz - min_xyz, axis=-1)
    return np.where(any_valid, diff_l2, 0.0)
