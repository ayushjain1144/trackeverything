# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Protocol, Tuple

import torch
from omegaconf import DictConfig


class EncoderProtocol(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def feature_size(self) -> Tuple[int, int]:
        pass

    @property
    def embedding_dim(self) -> int:
        pass


def load_encoder(resolution: Tuple[int, int], config: DictConfig) -> EncoderProtocol:
    name = config.name
    kwargs = {k: v for k, v in config.items() if k != "name"}
    if name == "dinov3_vit_adapter":
        from .dinov3_encoder import Dinov3Encoder

        return Dinov3Encoder(**kwargs)
    else:
        raise ValueError(f"Backbone {name} not supported")
