# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ipdb
import torch
from torch import nn
from einops import rearrange, repeat

from ..utils.warp_utils import warp_2d
from .utils.trajectory_transformer import TrajectoryTransformerEncoder
from .utils.position_encodings import (
    PositionEmbeddingCoordsSine,
    PositionEmbeddingNeRF,
)

st = ipdb.set_trace


class WAFTWarpv2(nn.Module):
    def __init__(
        self,
        dim: int,
        static: bool = False,
        warp_fn: str = "warp_2d",
        seq_len=8,
        use_transformer_traj_encoder=False,
        add_pos=False,
        use_nerf_pe: bool = False,
        posenc_3d_min_deg: int = -2,
        posenc_3d_max_deg: int = 14,
        return_warped_coords=False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.static = static
        self.use_transformer_traj_encoder = use_transformer_traj_encoder
        self.add_pos = add_pos
        self.return_warped_coords = return_warped_coords
        if self.static:
            if use_transformer_traj_encoder:
                self.condense_traj = TrajectoryTransformerEncoder(
                    input_dim=dim * 2,
                    latent_dim=64,
                    output_dim=dim * 2,
                    nhead=4,
                    num_layers=2,
                    train_traj_len=seq_len,
                )
            else:
                self.condense_traj = nn.Sequential(
                    nn.Linear(dim * 2 * seq_len, dim * 2),
                    nn.SiLU(),
                    nn.Linear(dim * 2, dim * 2),
                )

        if warp_fn != "warp_2d":
            raise NotImplementedError(f"{warp_fn} is not implemented")
        self.warp_fn = warp_2d

        self.project_warped_feats = nn.Linear(dim * 3, dim)
        
        if self.add_pos:
            if use_nerf_pe:
                self.pe_layer_3d = PositionEmbeddingNeRF(
                    input_dim=3,
                    min_deg=posenc_3d_min_deg,
                    max_deg=posenc_3d_max_deg,
                    output_dim=dim,
                )
            else:
                self.pe_layer_3d = PositionEmbeddingCoordsSine(
                    pos_type="sine", d_pos=dim
                )

    def forward(
        self,
        image_feats,
        coords_3d,
        scene_feats,
        cam_T_world=None,
        intrinsics=None,
        waft_image_feats=None,
        scene_coords=None,
        query_coords=None,
    ):
        """
        Inputs:
            feats: B X T X C X H X W
            coords_3d: B X T X N X 3
            scene_feats: N X C (static) or N X T X C (dynamic)
            world_T_cam: B X T X 4 X 4
            intrinsics: B X T X 3 X 3
            waft_image_feats: B X N X C
            scene_coords: B X T X H X W X 3
            query_coords: B X N X 3
        Outputs:
            warped_feats: B X T X H X W X T X C
        """
        assert image_feats.shape[0] == 1, "Only support batch size 1 for now"
        if self.return_warped_coords:
            warped_feats, warped_coords = self.warp_fn(
                image_feats, coords_3d, cam_T_world, intrinsics, scene_coords=scene_coords
            )
        else:
            warped_feats = self.warp_fn(
                image_feats, coords_3d, cam_T_world, intrinsics
            )  # B X N X T X C
            
        if self.add_pos:
            B, N, T, C = warped_feats.shape
            if coords_3d.shape[1] != T:
                assert coords_3d.shape[1] == 1, "coords_3d should have T or 1 time dimension"
                coords_3d = coords_3d.repeat(1, T, 1, 1)
            
            flow = coords_3d - query_coords.unsqueeze(1)  # B X T X N X 3
            warped_coords_pos = self.pe_layer_3d(
                rearrange(flow, 'b t n c -> b (t n) c')
            )
            # warped_coords_3d = rearrange(warped_coords_3d, 'b n t c -> b t n c')
            # warped_coords_pos = self.pe_layer_3d(
            #     rearrange(coords_3d - warped_coords_3d, 'b t n c -> b (t n) c')
            # )
            warped_coords_pos = rearrange(
                warped_coords_pos, 'b (t n) c -> b n t c', t=T, n=N
            )
            warped_feats = warped_feats + warped_coords_pos

            waft_image_pos = self.pe_layer_3d(query_coords - query_coords)

            waft_image_feats = waft_image_feats + waft_image_pos
       
        warped_feats = torch.cat([
            warped_feats,
            repeat(waft_image_feats, "b n c -> b n t c", t=warped_feats.shape[2]),
        ], dim=-1)
        warped_feats = warped_feats.squeeze(0)  # N X T X C
        if self.static:
            # warped_feats = warped_feats.mean(1)  # N X C
            if self.use_transformer_traj_encoder:
                warped_feats = self.condense_traj(warped_feats)  # N X C
            else:
                warped_feats = self.condense_traj(warped_feats.flatten(1, 2))  # N X C
        if self.return_warped_coords:
            return self.project_warped_feats(torch.cat([scene_feats, warped_feats], dim=-1)), warped_coords
        else:
            return self.project_warped_feats(torch.cat([scene_feats, warped_feats], dim=-1))


class WAFTWarpv3(nn.Module):
    def __init__(
        self,
        dim: int,
        warp_fn: str = "warp_2d",
        seq_len=8,
        use_nerf_pe: bool = False,
        posenc_3d_min_deg: int = -2,
        posenc_3d_max_deg: int = 14,
        return_warped_coords=False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.return_warped_coords = return_warped_coords
        if warp_fn != "warp_2d":
            raise NotImplementedError(f"{warp_fn} is not implemented")
        self.warp_fn = warp_2d

        self.project_warped_feats = nn.Linear(dim * 3, dim)

    def forward(
        self,
        image_feats,
        coords_3d,
        scene_feats,
        cam_T_world=None,
        intrinsics=None,
        waft_image_feats=None,
        scene_coords=None,
        query_coords=None,
        tgt_time_step_scene=None,
    ):
        """
        Inputs:
            feats: B X T X C X H X W
            coords_3d: B X T X N X 3
            scene_feats: N X C (static) or N X T X C (dynamic)
            world_T_cam: B X T X 4 X 4
            intrinsics: B X T X 3 X 3
            waft_image_feats: B X N X C
            scene_coords: B X T X H X W X 3
            query_coords: B X N X 3
        Outputs:
            warped_feats: B X T X H X W X T X C
        """
        assert image_feats.shape[0] == 1, "Only support batch size 1 for now"
        if self.return_warped_coords:
            warped_feats, warped_coords = self.warp_fn(
                image_feats,
                coords_3d,
                cam_T_world,
                intrinsics,
                scene_coords=scene_coords,
            )
        else:
            warped_feats = self.warp_fn(
                image_feats, coords_3d, cam_T_world, intrinsics
            )  # B X N X T X C
        warped_feats = warped_feats[:, :, tgt_time_step_scene]
        if self.return_warped_coords:
            warped_coords = warped_coords[:, :, tgt_time_step_scene]

        warped_feats = torch.cat([
            warped_feats.squeeze(0),
            waft_image_feats.squeeze(0),
            scene_feats,
        ], dim=-1)
        if self.return_warped_coords:
            return self.project_warped_feats(warped_feats), warped_coords
        else:
            return self.project_warped_feats(warped_feats)
    
    
