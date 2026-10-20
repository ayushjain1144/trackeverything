# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from einops import repeat
from torch.amp import autocast

from ..point_attentions.spatio_temporal_point_attentions import TemporalPointAttentionsDino
from ..utils.position_encodings import PositionEmbeddingNeRF, SinusoidalPosEmb
from ..waft_warp import WAFTWarpv2
import ipdb
st = ipdb.set_trace

class TransformerTrajDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim=384,
        output_dim=5,
        num_layers=6,
        num_heads=6,
        posenc_3d_min_deg=-2,
        posenc_3d_max_deg=14,
        warp_fn="warp_2d",
    ):
        super().__init__()
        supported_hidden_dims = {384: "vits14", 768: "vitb14", 1024: "vitl14", 1536: "vitg14"}
        vit_name = supported_hidden_dims.get(hidden_dim)
        if vit_name is None:
            raise ValueError(
                f"Unsupported hidden_dim={hidden_dim}. Expected one of {sorted(supported_hidden_dims)}."
            )
        self.hidden_dim = hidden_dim

        self.pe_layer_3d_warped = PositionEmbeddingNeRF(
            input_dim=3,
            min_deg=posenc_3d_min_deg,
            max_deg=posenc_3d_max_deg,
            output_dim=hidden_dim,
        )

        self.pe_layer_3d_trajs = PositionEmbeddingNeRF(
            input_dim=3,
            min_deg=posenc_3d_min_deg,
            max_deg=posenc_3d_max_deg,
            output_dim=hidden_dim,
        )
        self.temporal_point_attentions = TemporalPointAttentionsDino(
            latent_dim=hidden_dim,
            nheads=num_heads,
            num_layers=num_layers,
            vit_name=vit_name,
        )

        self.input_transform = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

        self.flow_head = torch.nn.Linear(hidden_dim, output_dim - 2, bias=True)
        self.vis_conf_head = torch.nn.Linear(hidden_dim, 2, bias=True)

        self.waft = WAFTWarpv2(
            dim=hidden_dim,
            warp_fn=warp_fn,
            return_warped_coords=True,
        )
        
        sinusoidal_pos_emb_theta = 10000
        sinu_pos_emb = SinusoidalPosEmb(
            hidden_dim, theta=sinusoidal_pos_emb_theta
        )
        self.time_encoder = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(hidden_dim, hidden_dim),
        )



    def forward(
        self,
        traj_feats_state,
        src_time_step_coords,
        tgt_time_step_coords,
        src_time_step,
        tgt_time_step,
        scene_context,
        image_feats,
        window_len,
        extrinsics,
        intrinsics_scaled,
        waft_feats,
        scene_coords,
        initial_traj=None,
        initial_traj_feats=None,
    ):
        """
        Args:
            traj_feats_state: [N, hidden_dim]
            src_time_step_coords: [N, 3]
            tgt_time_step_coords: [N, 3]
            src_time_step: int or [N]
            tgt_time_step: int or [N]
            scene_context: [N_s, hidden_dim]

        Returns:
            trajs: [N, T, output_dim]
        """
        if traj_feats_state.shape[0] == 0:
            return traj_feats_state.new_zeros((0, window_len, self.flow_head.out_features + 2)), traj_feats_state, None
        if initial_traj is None or initial_traj_feats is None:
            if not torch.is_tensor(src_time_step) or src_time_step.ndim == 0:
                src_time_step = torch.full(
                    (traj_feats_state.shape[0],),
                    fill_value=src_time_step,
                    device=traj_feats_state.device,
                    dtype=src_time_step_coords.dtype,
                )
            else:
                src_time_step = src_time_step.to(device=traj_feats_state.device, dtype=src_time_step_coords.dtype)

            if not torch.is_tensor(tgt_time_step) or tgt_time_step.ndim == 0:
                tgt_time_step = torch.full(
                    (traj_feats_state.shape[0],),
                    fill_value=tgt_time_step,
                    device=traj_feats_state.device,
                    dtype=src_time_step_coords.dtype,
                )
            else:
                tgt_time_step = tgt_time_step.to(device=traj_feats_state.device, dtype=src_time_step_coords.dtype)

            # first build a constant velocity initial trajectory
            delta_t = (tgt_time_step - src_time_step)
            delta_t = torch.where(delta_t == 0, torch.ones_like(delta_t), delta_t)
            velocity = (tgt_time_step_coords - src_time_step_coords) / delta_t[:, None]
            
            # where tgt_time_step and src_time_step are same, make velocity 0
            velocity = torch.where((torch.isclose(tgt_time_step, src_time_step))[:, None], torch.zeros_like(velocity), velocity)
            time_steps = torch.arange(
                window_len, device=traj_feats_state.device, dtype=src_time_step_coords.dtype
            )[None, :, None]
            initial_traj = src_time_step_coords[:, None] + velocity[:, None] * (
                time_steps - src_time_step[:, None, None]
            )
            initial_traj_feats = repeat(traj_feats_state, "n c -> n t c", t=window_len)

        # then we do waft
        initial_traj_feats, warped_coords = self.waft(
            image_feats,
            initial_traj[None].permute(0, 2, 1, 3),
            initial_traj_feats,
            extrinsics,
            intrinsics_scaled,
            waft_feats[None],
            scene_coords,
        )
        # add back traj state
        traj_feats = torch.cat([traj_feats_state[:, None], initial_traj_feats], dim=1)
        warped_coords = warped_coords.squeeze(0)

        # encode positional embeddings
        with autocast("cuda", enabled=False):
            pos_embeds = self.pe_layer_3d_trajs(
                initial_traj.float()
            ).to(traj_feats.dtype) + self.pe_layer_3d_warped(
                warped_coords.float()
            ).to(traj_feats.dtype) + self.time_encoder(
                torch.arange(window_len, device=traj_feats.device, dtype=torch.float32)
            )[None].to(traj_feats.dtype)
        pos_embeds = torch.cat([
            torch.zeros_like(traj_feats_state)[:, None],
            pos_embeds,
        ], dim=1)

        # apply temporal point attentions
        traj_feats = self.temporal_point_attentions(
            traj_feats,
            pos_embeds,
            scene_context,
        )
        traj_feats_seq = traj_feats[:, 1:]
        traj_feats_state = traj_feats[:, 0]

        traj_feats_decoder = self.input_transform(traj_feats_seq)
        with autocast("cuda", enabled=False):
            flow = self.flow_head(traj_feats_decoder.to(torch.float32)).to(traj_feats_decoder.dtype)
        
        traj_coords = initial_traj + flow
        vis_conf = self.vis_conf_head(traj_feats_decoder)
        vis_logit = vis_conf[..., :1]
        conf_raw = vis_conf[..., 1:]

        conf_activated = 1 + conf_raw.exp()
        traj_decoder_out = torch.cat([traj_coords, vis_logit, conf_activated], dim=-1)
        return traj_decoder_out, traj_feats_state, traj_feats_seq
