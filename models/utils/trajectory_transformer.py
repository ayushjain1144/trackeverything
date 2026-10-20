# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ipdb
import torch
import torch.nn as nn

from .position_encodings import SinusoidalPosEmb


st = ipdb.set_trace


class TrajectoryTransformerEncoder(nn.Module):
    """Encodes a variable-length trajectory (T×C) into a fixed embedding."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        output_dim: int,
        num_layers: int = 2,
        nhead: int = 4,
        train_traj_len: int = 8,
    ):
        super().__init__()
        # dim * 2 because we just want a small transformer
        self.input_proj = nn.Linear(input_dim, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=latent_dim * 2,
            batch_first=True,
        )
        self.nhead = nhead
        self.train_traj_len = train_traj_len
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.time_pos_embed = SinusoidalPosEmb(latent_dim)
        self.output_dim = output_dim
        self.proj = nn.Linear(latent_dim, output_dim)

    def forward(self, traj: torch.Tensor):
        """
        traj: [N, T, C_input]  (sequence per point)
        Returns: [N, C_ouptut]
        """
        if traj.shape[0] == 0:
            return torch.zeros((0, self.output_dim), device=traj.device)
        # Add simple sinusoidal positional encoding
        pos = torch.arange(traj.size(1), device=traj.device).unsqueeze(0).unsqueeze(-1)
        if traj.size(1) > self.train_traj_len:
            pos = (pos / (traj.size(1) - 1)) * (self.train_traj_len - 1)
            assert pos.max() <= self.train_traj_len - 1
        pe = self.time_pos_embed(pos).squeeze(0)  # [1, T, C]
        traj = self.input_proj(traj)  # [N, T, C]
        traj = traj + pe  # [N, T, C]

        # Encode and mean-pool across time
        max_chunk = 60000
        bs = traj.size(0)
        if bs <= max_chunk:
            out = self.encoder(traj)  # [N, T, C]
        else:
            chunks = []
            for start in range(0, bs, max_chunk):
                end = min(start + max_chunk, bs)
                out_chunk = self.encoder(traj[start:end])  # [N, T, C]
                chunks.append(out_chunk)
            out = torch.cat(chunks, dim=0)
        out = out.mean(dim=1)  # [N, C]
        return self.proj(out)
