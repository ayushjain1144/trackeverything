# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ipdb
import torch
from einops import rearrange, repeat

from ..datasets.utils.geometry import project
from ..models.utils.cotracker_utils import bilinear_sampler

st = ipdb.set_trace


# simplest version -- pick up features from 2D feature maps
def warp_2d(feats, coords_3d, cam_T_world, intrinsics, scene_coords=None):
    """
    Inputs:
        feats: B X T X C X H X W
        coords_3d: B X T X N X 3
        world_T_cam: B X T X 4 X 4
        intrinsics: B X T X 3 X 3
    Outputs:
        warped_feats: B X N X T X C
    """
    # coords_3d = rearrange(coords_3d, "b t h w t1 c -> b (t h w) t1 c")
    image_size = feats.shape[-2], feats.shape[-1]
    with torch.autocast(device_type="cuda", enabled=False):
        coords_2d = project(
            coords_3d.to(torch.float32),
            intrinsics.unsqueeze(2),
            cam_T_world.unsqueeze(2),
            image_size,
        )[..., :2]  # B X T X N X 2
        coords_2d = coords_2d.to(feats.dtype)

    T = coords_2d.shape[1]
    query_time = repeat(
        torch.arange(T, device=coords_3d.device),
        "t1 -> b t1 n",
        b=coords_3d.shape[0],
        n=coords_3d.shape[2],
    )  # B X T X N
    coords_2d = torch.cat(
        (query_time[..., None], coords_2d), dim=-1
    )  # B X (T H W) X T X 3
    interpolated_feats = bilinear_sampler(
        rearrange(feats, "b t c h w -> b c t h w"),
        rearrange(coords_2d, "b t1 n d -> b (t1 n) 1 1 d"),
    )  # b c (n t1) 1 1
    interpolated_feats = rearrange(
        interpolated_feats, "b c (t1 n) 1 1 -> b n t1 c", n=coords_3d.shape[2]
    )  # b n t1 c
    
    if scene_coords is not None:
        interpolated_coords = bilinear_sampler(
            rearrange(scene_coords, "b t h w c -> b c t h w"),
            rearrange(coords_2d, "b t1 n d -> b (t1 n) 1 1 d"),
            mode="nearest"
        )  # b c (n t1) 1 1
        interpolated_coords = rearrange(
            interpolated_coords, "b c (t1 n) 1 1 -> b n t1 c", n=coords_3d.shape[2]
        )  # b n t1 c
        return interpolated_feats, interpolated_coords
    return interpolated_feats
