# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import fields, is_dataclass
from functools import wraps
from typing import Dict

import ipdb

import numpy as np
import torch
from box import Box
from opt_einsum import contract
from rich.logging import RichHandler

Instructions = Dict[str, Dict[int, torch.Tensor]]
logger = logging.getLogger(__name__)
st = ipdb.set_trace


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_local_debug_logging() -> None:
    """Enable DEBUG on the root logger when TAPIP3D_LOCAL_DEBUG=1 (see configure_local in setup.sh)."""
    if os.environ.get("TAPIP3D_LOCAL_DEBUG", "0") != "1":
        return
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in root.handlers:
        handler.setLevel(logging.DEBUG)
    package = __name__.split(".")[0]
    for name in (
        f"{package}.main_point_track",
        f"{package}.models.encoders.dinov3_encoder",
    ):
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        for handler in lg.handlers:
            handler.setLevel(logging.DEBUG)


def setup_logger():
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    apply_local_debug_logging()
    # PIL logs every PNG chunk at DEBUG when root logging is DEBUG (TAPIP3D_LOCAL_DEBUG).
    logging.getLogger("PIL").setLevel(logging.WARNING)


def convert_tensor(x, allow_cast: bool):
    if not isinstance(x, torch.Tensor):
        return x
    if x.dtype not in [torch.float16, torch.bfloat16]:
        return x
    if allow_cast:
        return x.to(dtype=torch.float32)
    else:
        raise TypeError(f"Tensor of dtype {x.dtype} is not allowed. Expected float32.")
    return x


def convert_tensor_struct(data, allow_cast: bool):
    if isinstance(data, torch.Tensor):
        return convert_tensor(data, allow_cast)
    elif isinstance(data, list):
        return [convert_tensor_struct(item, allow_cast) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_tensor_struct(item, allow_cast) for item in data)
    elif isinstance(data, Box):
        return Box({key: process_input(value) for key, value in data.items()})
    elif isinstance(data, dict):
        return {
            key: convert_tensor_struct(value, allow_cast) for key, value in data.items()
        }
    elif is_dataclass(data):
        return type(data)(
            **{
                field.name: convert_tensor_struct(getattr(data, field.name), allow_cast)
                for field in fields(data)
            }
        )
    else:
        return data


def ensure_float32(allow_cast: bool = True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            new_args = tuple(convert_tensor_struct(arg, allow_cast) for arg in args)
            new_kwargs = {
                key: convert_tensor_struct(value, allow_cast)
                for key, value in kwargs.items()
            }
            with torch.autocast(device_type="cuda", enabled=False):
                return convert_tensor_struct(func(*new_args, **new_kwargs), True)

        return wrapper

    return decorator


def cast_float32():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            new_args = tuple(convert_tensor_struct(arg, True) for arg in args)
            new_kwargs = {
                key: convert_tensor_struct(value, True) for key, value in kwargs.items()
            }
            return convert_tensor_struct(func(*new_args, **new_kwargs), True)

        return wrapper

    return decorator


def unproject(intrinsics, poses, depths):
    """
    Inputs:
        intrinsics: B X V X 3 X 3
        poses: B X V X 4 X 4 (torch.tensor)
        depths: B X V X H X W (torch.tensor)

    Outputs:
        world_coords: B X V X H X W X 3 (all valid 3D points)
        valid: B X V X H X W (bool to indicate valid points)
                can be used to index into RGB images
                to get N X 3 valid RGB values
    """
    B, V, H, W = depths.shape
    fx, fy, px, py = (
        intrinsics[..., 0, 0][..., None],
        intrinsics[..., 1, 1][..., None],
        intrinsics[..., 0, 2][..., None],
        intrinsics[..., 1, 2][..., None],
    )

    y = torch.arange(0, H).to(depths.device)
    x = torch.arange(0, W).to(depths.device)
    y, x = torch.meshgrid(y, x)

    x = x[None, None].repeat(B, V, 1, 1).flatten(2)
    y = y[None, None].repeat(B, V, 1, 1).flatten(2)
    z = depths.flatten(2)
    x = (x - px) * z / fx
    y = (y - py) * z / fy
    cam_coords = torch.stack([x, y, z, torch.ones_like(x)], -1)

    world_coords = (poses @ cam_coords.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    world_coords = world_coords[..., :3] / world_coords[..., 3][..., None]

    world_coords = world_coords.reshape(B, V, H, W, 3)

    return world_coords


def safe_inverse(a):  # parallel version
    # B, _, _ = list(a.shape)
    inv = a.clone()
    r_transpose = a[..., :3, :3].transpose(-1, -2)  # inverse of rotation matrix

    inv[..., :3, :3] = r_transpose
    inv[..., :3, 3:4] = -torch.matmul(r_transpose, a[..., :3, 3:4])

    return inv

def safe_inverse_np(a):
    inv = a.copy()
    r_transpose = np.swapaxes(a[..., :3, :3], -1, -2)  # inverse of rotation matrix
    inv[..., :3, :3] = r_transpose
    inv[..., :3, 3:4] = -np.matmul(r_transpose, a[..., :3, 3:4])
    return inv


@ensure_float32(allow_cast=False)
def batch_unproject(
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,  # this is cam_T_world
    channel_last: bool = False,
) -> torch.Tensor:
    if depth.ndim == 4:
        pcds = batch_unproject(
            depth.view(-1, *depth.shape[2:]),
            intrinsic.view(-1, *intrinsic.shape[2:]),
            extrinsic.view(-1, *extrinsic.shape[2:]),
            channel_last,
        )
        return pcds.reshape(depth.shape[:2] + pcds.shape[1:])
    assert depth.ndim == 3, "depth must be a 3D array"
    t, h, w = depth.shape
    assert intrinsic.shape == (
        t,
        3,
        3,
    ), f"intrinsic must be a 3D array of shape (t, 3, 3), got {intrinsic.shape}"
    assert extrinsic.shape == (
        t,
        4,
        4,
    ), f"extrinsic must be a 3D array of shape (t, 4, 4), got {extrinsic.shape}"

    # Generate 3D point cloud in world coordinates
    v, u = torch.meshgrid(
        torch.arange(h, device=depth.device) + 0.5,
        torch.arange(w, device=depth.device) + 0.5,
        indexing="ij",
    )
    uv_homogeneous = torch.stack((u, v, torch.ones_like(u)), dim=-1).float()
    K_inv = torch.linalg.inv(intrinsic)
    camera_coords = contract("nij, xyj -> nxyi", K_inv, uv_homogeneous)
    camera_coords = camera_coords * depth[..., None]
    camera_coords = torch.cat(
        (camera_coords, torch.ones_like(camera_coords[..., :1])), dim=-1
    )

    # inv_extrinsics = torch.linalg.inv(extrinsic)
    inv_extrinsics = safe_inverse(extrinsic)
    # inv_extrinsics = torch.from_numpy(np.linalg.inv(extrinsic.cpu().numpy())).to(extrinsic)
    # inv_extrinsics = torch.from_numpy(np.linalg.inv(extrinsic.cpu().numpy())).to(extrinsic)
    world_coordinates = contract("nij, nxyj -> nxyi", inv_extrinsics, camera_coords)

    if channel_last:
        return world_coordinates[..., :3]
    else:
        return world_coordinates[..., :3].permute(0, 3, 1, 2)


@ensure_float32(allow_cast=False)
def batch_unproject_uv_space(
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    channel_last: bool = False,
    eps: float = 1e-5,
) -> torch.Tensor:
    if depth.ndim == 4:
        pcds = batch_unproject_uv_space(
            depth.view(-1, *depth.shape[2:]),
            intrinsic.view(-1, *intrinsic.shape[2:]),
            channel_last,
        )
        return pcds.reshape(depth.shape[:2] + pcds.shape[1:])
    assert depth.ndim == 3, "depth must be a 3D array"
    t, h, w = depth.shape
    assert intrinsic.shape == (
        t,
        3,
        3,
    ), f"intrinsic must be a 3D array of shape (t, 3, 3), got {intrinsic.shape}"

    v, u = torch.meshgrid(
        torch.arange(h, device=depth.device),
        torch.arange(w, device=depth.device),
        indexing="ij",
    )
    logd = torch.log(torch.clamp(depth, min=eps))
    focals = torch.sqrt(intrinsic[..., 0, 0] * intrinsic[..., 1, 1])

    u_v_logd = torch.stack(
        (
            u[None].expand(t, -1, -1),
            v[None].expand(t, -1, -1),
            focals[:, None, None] * logd,
        ),
        dim=-1,
    )

    if channel_last:
        return u_v_logd
    else:
        return u_v_logd.permute(0, 3, 1, 2)


@ensure_float32(allow_cast=False)
def batch_project(
    pts3d: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> torch.Tensor:
    pts3d_homogeneous = torch.cat(
        (pts3d, torch.ones_like(pts3d[..., :1])), dim=-1
    )  # (n, 4)

    pts_camera_homogeneous = torch.einsum(
        "...ij, ...j -> ...i", extrinsics, pts3d_homogeneous
    )  # (n, 4)
    pts_camera = (
        pts_camera_homogeneous[..., :3] / pts_camera_homogeneous[..., 3:]
    )  # (n, 3)

    pts_image_homogeneous = torch.einsum(
        "...ij, ...j -> ...i", intrinsics, pts_camera
    )  # (n, 3)
    pts_image = (
        pts_image_homogeneous[..., :2] / pts_image_homogeneous[..., 2:]
    )  # (n, 2)

    return pts_image


@ensure_float32(allow_cast=False)
def apply_homo_transform(
    coords: torch.Tensor, transform: torch.Tensor, pad_ones: bool = True
):
    if pad_ones:
        coords_homo = torch.cat([coords, torch.ones_like(coords[..., :1])], dim=-1)
    else:
        coords_homo = coords
    coords_transformed_homo = torch.einsum("...ij,...j->...i", transform, coords_homo)
    coords_transformed = (
        coords_transformed_homo[..., :-1] / coords_transformed_homo[..., -1:]
    )
    return coords_transformed


    # if is_changed:
    # checkpoint.pop("optimizer_states", None)
