# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from omegaconf import DictConfig

from .endpoint_decoder import EndpointDecoder


def load_point_updater(config: DictConfig) -> nn.Module:
    name = config.name
    kwargs = {k: v for k, v in config.items() if k != "name"}
    if name == "endpoint_decoder":
        return EndpointDecoder(**kwargs)
    raise ValueError(f"Unknown point updater: {name}")
