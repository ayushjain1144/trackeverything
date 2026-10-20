# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.amp import autocast


class EndpointDecoder(nn.Module):
    """MLP head for the endpoint refiner.

    Decodes, per point: a 3D flow residual, a visibility logit, a confidence
    scalar, and a static/dynamic logit. Unlike the older updater this head does
    not predict any geometry (source-position) refinement.
    """

    def __init__(
        self,
        input_dim=320,
        hidden_size=384,
        output_dim=5,
        conf_activation="expp1",
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.conf_activation = conf_activation

        self.input_transform = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_size, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
        )
        # flow residual (3) is decoded in fp32; visibility + confidence share a head.
        self.flow_head = torch.nn.Linear(hidden_size, output_dim - 2, bias=True)
        self.vis_conf_head = torch.nn.Linear(hidden_size, 2, bias=True)

        # separate branch for the static/dynamic classification logit
        self.classification_input_transform = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_size, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.classification_head = torch.nn.Linear(hidden_size, 1, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            torch.nn.init.trunc_normal_(self.flow_head.weight, std=0.001)
            torch.nn.init.trunc_normal_(self.vis_conf_head.weight, std=0.001)

        self.apply(_basic_init)

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: (B, N, C) per-point features from the endpoint refiner.

        Returns:
            flow: (B, N, output_dim) with [dx, dy, dz, vis_logit, conf]
            dynamic_logits: (B, N, 1) static/dynamic logit
        """
        tokens = self.input_transform(input_tensor)

        with autocast("cuda", enabled=False):
            flow = self.flow_head(tokens.to(torch.float32)).to(input_tensor.dtype)

        vis_conf = self.vis_conf_head(tokens)
        vis_logit = vis_conf[..., :1]
        conf_raw = vis_conf[..., 1:]

        if self.conf_activation == "expp1":
            conf_activated = 1 + conf_raw.exp()
        elif self.conf_activation == "expp0":
            conf_activated = conf_raw.exp()
        elif self.conf_activation == "sigmoid":
            conf_activated = torch.sigmoid(conf_raw)
        else:
            conf_activated = conf_raw

        flow = torch.cat([flow, vis_logit, conf_activated], dim=-1)

        cls_tokens = self.classification_input_transform(input_tensor)
        dynamic_logits = self.classification_head(cls_tokens)

        return flow, dynamic_logits
