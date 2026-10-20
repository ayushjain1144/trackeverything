# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Apply releasable Point Odyssey annotation corrections at load time.

The artifact this reads is deliberately tiny -- a JSON keyed by ``split/seq``
carrying an integer camera offset and a few un-repairable frame ranges. Nothing
from Point Odyssey is redistributed; users apply the patch to their own copy.

Correction: camera-pose frame offset
------------------------------------
For the affected sequences the pose stored at ``extrinsics[t]`` is the pose that
rendered frame ``t-1``. ``trajs_3d`` is correctly indexed. Repair is therefore:

* take camera arrays from ``t + k`` instead of ``t``
* recompute ``trajs_2d`` from the corrected camera and the (unchanged) 3D points
* invalidate the final ``k`` frames, which have no ``t + k`` to borrow from

The manifest ships inside this repo (``pod_corrections_data/``), so the
corrections are applied by default -- no flag and no external path required.
``PointOdysseyDataProvider`` falls back to :data:`BUNDLED_MANIFEST_PATH` when the
config does not name one; pass ``correction_root`` only to point at a different
manifest, or ``correction_root: null`` to run uncorrected for an A/B.

See ``scripts/pod_diagnostics/pod_build_corrections.py`` for how the artifact is
produced, ``scripts/pod_diagnostics/lib/pod_alignment.py`` for how the offset is
measured, and ``docs/pod_annotation_bugs.md`` for the full list of defects.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MANIFEST_NAME = "pod_corrections.json"
FORMAT_VERSION = 1

# Shipped with the repo so corrections are on by default. This is our own
# measurement output (~30 KB of integer offsets and frame ranges) -- no Point
# Odyssey data is redistributed.
BUNDLED_MANIFEST_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pod_corrections_data"
)
BUNDLED_MANIFEST_PATH = os.path.join(BUNDLED_MANIFEST_DIR, MANIFEST_NAME)

# Sequences whose annotations are unusable and cannot be repaired by re-indexing.
# scene_recording_20210910_S05_S06_0_ego1 has *temporally incoherent* extrinsics:
# the best sub-frame alignment wanders over -1.0..+1.5 and 40% of probe frames
# need |alpha| >= 0.5, versus 3% and 0% in comparable sequences. Its median is 0,
# so integer-offset tests call it "aligned" while no single offset can fix it.
# See docs/pod_known_bad_sequences.md.
#
# NOTE: dropping a sequence changes the (seed, index) -> window mapping, because
# sampling runs over the surviving sequence list. Window indices recorded before
# an exclusion do not refer to the same clips afterwards.
KNOWN_BAD_SEQUENCES: Tuple[str, ...] = ("scene_recording_20210910_S05_S06_0_ego1",)


@dataclass
class SequenceCorrection:
    seq: str
    split: str
    cam_offset: int = 0
    invalidate_frames: List[Tuple[int, int]] = field(default_factory=list)
    visibility_unreliable_frames: List[Tuple[int, int]] = field(default_factory=list)
    evidence: Dict = field(default_factory=dict)

    def unreliable_visibility_mask(self, frame_indices: np.ndarray) -> np.ndarray:
        """Frames where the repair invalidates POD's stored visibility.

        POD derived ``visibs[t]`` from the pre-repair pose, so wherever the
        repair moves points appreciably those flags describe a camera that never
        rendered anything -- in ani15_new_ at t=163 all 135 deer tracks are
        marked invisible while the deer is plainly in frame. They are not
        rewritten: re-deriving visibility from the depth buffer disagrees with
        visibs on 8-17% of cells even with no camera change, so the estimator is
        too noisy to substitute.
        """
        bad = np.zeros(len(frame_indices), dtype=bool)
        for lo, hi in self.visibility_unreliable_frames:
            bad |= (frame_indices >= lo) & (frame_indices <= hi)
        return bad

    def invalid_mask(self, frame_indices: np.ndarray, n_frames: int) -> np.ndarray:
        """Frames that cannot be repaired, as a boolean over ``frame_indices``."""
        bad = np.zeros(len(frame_indices), dtype=bool)
        for lo, hi in self.invalidate_frames:
            bad |= (frame_indices >= lo) & (frame_indices <= hi)
        if self.cam_offset > 0:
            bad |= frame_indices + self.cam_offset > n_frames - 1
        elif self.cam_offset < 0:
            bad |= frame_indices + self.cam_offset < 0
        return bad


class CorrectionSet:
    """All per-sequence corrections from one manifest."""

    def __init__(self, manifest: Dict, root: Optional[str] = None):
        self.root = root
        self.version = int(manifest.get("version", FORMAT_VERSION))
        self.meta = {k: v for k, v in manifest.items() if k != "sequences"}
        self._by_key: Dict[str, SequenceCorrection] = {}
        for key, entry in (manifest.get("sequences") or {}).items():
            split, _, seq = key.partition("/")
            self._by_key[key] = SequenceCorrection(
                seq=seq,
                split=split,
                cam_offset=int(entry.get("cam_offset", 0)),
                invalidate_frames=[tuple(x) for x in entry.get("invalidate_frames", [])],
                visibility_unreliable_frames=[
                    tuple(x) for x in entry.get("visibility_unreliable_frames", []) or []
                ],
                evidence=entry.get("evidence", {}) or {},
            )

    @classmethod
    def load(cls, path: str) -> "CorrectionSet":
        p = path
        if os.path.isdir(p):
            p = os.path.join(p, MANIFEST_NAME)
        with open(p) as f:
            manifest = json.load(f)
        cs = cls(manifest, root=os.path.dirname(p))
        logger.info(
            "Loaded POD corrections v%s from %s: %d sequences",
            cs.version, p, len(cs._by_key),
        )
        return cs

    def get(self, split: str, seq: str) -> Optional[SequenceCorrection]:
        return self._by_key.get(f"{split}/{seq}")

    def __len__(self) -> int:
        return len(self._by_key)

    @property
    def keys(self) -> List[str]:
        return sorted(self._by_key)


def project_points(
    trajs_3d: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    """``(T, N, 3)`` world points -> ``(T, N, 2)`` pixels, per-frame camera."""
    cam = np.einsum("tij,tnj->tni", extrinsics[:, :3, :3], trajs_3d) + extrinsics[:, None, :3, 3]
    uvw = np.einsum("tij,tnj->tni", intrinsics, cam)
    return uvw[..., :2] / np.where(np.abs(uvw[..., 2:3]) < 1e-6, 1e-6, uvw[..., 2:3])


def apply_camera_offset(
    correction: SequenceCorrection,
    frame_indices: np.ndarray,
    n_frames: int,
    anns,
    trajs_3d: np.ndarray,
    valids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Return ``(trajs_2d, intrinsics, extrinsics, valids, info)`` after repair.

    ``frame_indices`` are absolute indices into the sequence. Frames without a
    ``t + k`` counterpart are clamped for indexing and then invalidated, so the
    arrays keep their shape and no out-of-range read occurs.
    """
    k = int(correction.cam_offset)
    shifted = frame_indices + k
    unrepairable = correction.invalid_mask(frame_indices, n_frames)
    shifted_clamped = np.clip(shifted, 0, n_frames - 1)

    intrinsics = np.asarray(anns["intrinsics"][shifted_clamped]).astype(np.float32)
    extrinsics = np.asarray(anns["extrinsics"][shifted_clamped]).reshape(-1, 4, 4).astype(np.float32)
    trajs_2d = project_points(trajs_3d, extrinsics, intrinsics).astype(np.float32)

    valids = np.array(valids, copy=True)
    if unrepairable.any():
        valids[unrepairable] = False

    info = {
        "cam_offset": k,
        "n_unrepairable_frames": int(unrepairable.sum()),
    }
    return trajs_2d, intrinsics, extrinsics, valids, info
