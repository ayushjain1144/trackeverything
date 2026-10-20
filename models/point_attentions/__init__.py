# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from omegaconf import DictConfig


def load_point_attention(config: DictConfig):
    name = config.name
    kwargs = {k: v for k, v in config.items() if k != "name"}
    if name == "spatio_temporal_v2":
        from .spatio_temporal_point_attentions import SpatioTemporalPointAttentionsv2

        return SpatioTemporalPointAttentionsv2(**kwargs)
    elif name == "spatio_temporal_dino":
        from .spatio_temporal_point_attentions import (
            SpatioTemporalPointAttentionsDino,
        )

        return SpatioTemporalPointAttentionsDino(**kwargs)
    elif name == "spatial_dino":
        from .spatio_temporal_point_attentions import SpatialPointAttentionsDino

        return SpatialPointAttentionsDino(**kwargs)
    else:
        raise ValueError(f"Unknown point attention: {name}")
