# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Depth and normal edge helpers used by the training transforms.

Copied from MoGe's utils3d so the release does not depend on that package.
"""

from functools import wraps
import warnings
from numbers import Number
from typing import Tuple, Union

import numpy as np


def _no_warnings(category=RuntimeWarning):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=category)
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def sliding_window_1d(x: np.ndarray, window_size: int, stride: int, axis: int = -1):
    assert x.shape[axis] >= window_size, (
        f"kernel_size ({window_size}) is larger than axis_size ({x.shape[axis]})"
    )
    axis = axis % x.ndim
    shape = (
        *x.shape[:axis],
        (x.shape[axis] - window_size + 1) // stride,
        *x.shape[axis + 1 :],
        window_size,
    )
    strides = (
        *x.strides[:axis],
        stride * x.strides[axis],
        *x.strides[axis + 1 :],
        x.strides[axis],
    )
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def sliding_window_nd(x: np.ndarray, window_size: Tuple[int, ...], stride: Tuple[int, ...], axis: Tuple[int, ...]) -> np.ndarray:
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    for i in range(len(axis)):
        x = sliding_window_1d(x, window_size[i], stride[i], axis[i])
    return x


def sliding_window_2d(
    x: np.ndarray,
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    axis: Tuple[int, int] = (-2, -1),
) -> np.ndarray:
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    return sliding_window_nd(x, window_size, stride, axis)


def max_pool_1d(x: np.ndarray, kernel_size: int, stride: int, padding: int = 0, axis: int = -1):
    axis = axis % x.ndim
    if padding > 0:
        fill_value = np.nan if x.dtype.kind == "f" else np.iinfo(x.dtype).min
        padding_arr = np.full(
            (*x.shape[:axis], padding, *x.shape[axis + 1 :]),
            fill_value=fill_value,
            dtype=x.dtype,
        )
        x = np.concatenate([padding_arr, x, padding_arr], axis=axis)
    a_sliding = sliding_window_1d(x, kernel_size, stride, axis)
    return np.nanmax(a_sliding, axis=-1)


def max_pool_2d(
    x: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]],
    axis: Tuple[int, int] = (-2, -1),
):
    if isinstance(kernel_size, Number):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, Number):
        stride = (stride, stride)
    if isinstance(padding, Number):
        padding = (padding, padding)
    for i, ax in enumerate(axis):
        x = max_pool_1d(x, kernel_size[i], stride[i], padding[i], ax)
    return x


@_no_warnings()
def depth_edge(
    depth: np.ndarray,
    atol: float = None,
    rtol: float = None,
    kernel_size: int = 3,
    mask: np.ndarray = None,
) -> np.ndarray:
    if mask is None:
        diff = max_pool_2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + max_pool_2d(
            -depth, kernel_size, stride=1, padding=kernel_size // 2
        )
    else:
        diff = max_pool_2d(
            np.where(mask, depth, -np.inf), kernel_size, stride=1, padding=kernel_size // 2
        ) + max_pool_2d(
            np.where(mask, -depth, -np.inf), kernel_size, stride=1, padding=kernel_size // 2
        )

    edge = np.zeros_like(depth, dtype=bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= diff / depth > rtol
    return edge


@_no_warnings()
def normals_edge(
    normals: np.ndarray, tol: float, kernel_size: int = 3, mask: np.ndarray = None
) -> np.ndarray:
    assert normals.ndim >= 3 and normals.shape[-1] == 3, "normal should be of shape (..., height, width, 3)"
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)

    padding = kernel_size // 2
    normals_window = sliding_window_2d(
        np.pad(
            normals,
            (*([(0, 0)] * (normals.ndim - 3)), (padding, padding), (padding, padding), (0, 0)),
            mode="edge",
        ),
        window_size=kernel_size,
        stride=1,
        axis=(-3, -2),
    )
    if mask is None:
        angle_diff = np.arccos((normals[..., None, None] * normals_window).sum(axis=-3)).max(axis=(-2, -1))
    else:
        mask_window = sliding_window_2d(
            np.pad(
                mask,
                (*([(0, 0)] * (mask.ndim - 3)), (padding, padding), (padding, padding)),
                mode="edge",
            ),
            window_size=kernel_size,
            stride=1,
            axis=(-3, -2),
        )
        angle_diff = np.where(
            mask_window, np.arccos((normals[..., None, None] * normals_window).sum(axis=-3)), 0
        ).max(axis=(-2, -1))

    angle_diff = max_pool_2d(angle_diff, kernel_size, stride=1, padding=kernel_size // 2)
    return angle_diff > np.deg2rad(tol)


@_no_warnings()
def points_to_normals(point: np.ndarray, mask: np.ndarray = None):
    height, width = point.shape[-3:-1]
    has_mask = mask is not None

    if mask is None:
        mask = np.ones_like(point[..., 0], dtype=bool)
    mask_pad = np.zeros((height + 2, width + 2), dtype=bool)
    mask_pad[1:-1, 1:-1] = mask
    mask = mask_pad

    pts = np.zeros((height + 2, width + 2, 3), dtype=point.dtype)
    pts[1:-1, 1:-1, :] = point
    up = pts[:-2, 1:-1, :] - pts[1:-1, 1:-1, :]
    left = pts[1:-1, :-2, :] - pts[1:-1, 1:-1, :]
    down = pts[2:, 1:-1, :] - pts[1:-1, 1:-1, :]
    right = pts[1:-1, 2:, :] - pts[1:-1, 1:-1, :]
    normal = np.stack(
        [
            np.cross(up, left, axis=-1),
            np.cross(left, down, axis=-1),
            np.cross(down, right, axis=-1),
            np.cross(right, up, axis=-1),
        ]
    )
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)
    valid = np.stack(
        [
            mask[:-2, 1:-1] & mask[1:-1, :-2],
            mask[1:-1, :-2] & mask[2:, 1:-1],
            mask[2:, 1:-1] & mask[1:-1, 2:],
            mask[1:-1, 2:] & mask[:-2, 1:-1],
        ]
    ) & mask[None, 1:-1, 1:-1]
    normal = (normal * valid[..., None]).sum(axis=0)
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)

    if has_mask:
        normal_mask = valid.any(axis=0)
        normal = np.where(normal_mask[..., None], normal, 0)
        return normal, normal_mask
    return normal
