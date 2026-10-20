# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Factory for geometry backends used by demos (Pi3 / VGGT-Omega)."""

from __future__ import annotations

from typing import Any, Literal, Optional

GeometryBackend = Literal["pi3", "vggt_omega"]

DEFAULT_VGGT_OMEGA_CHECKPOINT = (
    "/data/group_data/katefgroup/language_grounding/tracking_ckpts/"
    "vggt_omega_1b_512.pt"
)


def build_geometry_annotator(
    backend: GeometryBackend = "pi3",
    *,
    device: str = "cuda",
    pi3_checkpoint: str = "yyfz233/Pi3",
    pi3_chunk_size: int = 60,
    pi3_overlap: int = 30,
    vggt_omega_checkpoint: str = DEFAULT_VGGT_OMEGA_CHECKPOINT,
    vggt_omega_chunk_size: int = 60,
    vggt_omega_overlap: int = 30,
    vggt_omega_image_resolution: int = 512,
) -> Any:
    """Construct a long-video geometry annotator with a shared process_video API."""
    if backend == "pi3":
        from .pi3_long import Pi3LongAnnotator

        annotator = Pi3LongAnnotator(
            checkpoint_path=pi3_checkpoint,
            chunk_size=pi3_chunk_size,
            overlap=pi3_overlap,
            viz_dir=None,
        )
    elif backend == "vggt_omega":
        from .vggt_omega_long import VGGTOmegaLongAnnotator

        annotator = VGGTOmegaLongAnnotator(
            checkpoint_path=vggt_omega_checkpoint,
            image_resolution=vggt_omega_image_resolution,
            chunk_size=vggt_omega_chunk_size,
            overlap=vggt_omega_overlap,
        )
    else:
        raise ValueError(
            f"Unknown geometry_backend={backend!r}; expected 'pi3' or 'vggt_omega'"
        )
    annotator.to(device)
    return annotator
