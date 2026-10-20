# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, List, Optional, Tuple

import cv2
import ipdb
import numpy as np
import torch
import torchvision.transforms.functional as TF
try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func
from omegaconf import DictConfig, ListConfig
from PIL import Image
from scipy import ndimage
from scipy.spatial.transform import Rotation as R
from torchvision.transforms import ColorJitter, GaussianBlur

from ..models.utils.cotracker_utils import bilinear_sampler

from .depth_edges import depth_edge, normals_edge, points_to_normals
from .datatypes import RawSliceData, SliceData
from .utils.crop_utils import CropArgs

st = ipdb.set_trace


# def compose(*ops: Callable[..., RawSliceData]) -> Callable[..., RawSliceData]:
#     def composed(
#         data: RawSliceData, executor: ThreadPoolExecutor, **kwargs
#     ) -> RawSliceData:
#         for op in ops:
#             data = op(data=data, executor=executor, **kwargs)
#         return data

#     return composed


def composed(
    data: RawSliceData,
    executor: ThreadPoolExecutor,
    ops: List[Callable[..., RawSliceData]],
    **kwargs,
) -> RawSliceData:
    for op in ops:
        data = op(data=data, executor=executor, **kwargs)
    return data


def compose(*ops: Callable[..., RawSliceData]) -> Callable[..., RawSliceData]:
    return partial(composed, ops=ops)


all_transforms = {}


def register_transform(
    name: str,
) -> Callable[[Callable[..., RawSliceData]], Callable[..., RawSliceData]]:
    def wrapper(func: Callable[..., RawSliceData]) -> Callable[..., RawSliceData]:
        all_transforms[name] = func
        return func

    return wrapper


class TemporarySeed:
    def __init__(self, seed):
        self.seed = seed
        self.random_state = None
        self.torch_rng_state = None

    def __enter__(self):
        self.random_state = random.getstate()
        self.torch_rng_state = torch.get_rng_state()
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self.random_state)  # type: ignore
        torch.set_rng_state(self.torch_rng_state)  # type: ignore


def apply_extrinsics(extrinsics: np.ndarray, coords: np.ndarray):
    coords_homo = np.concatenate([coords, np.ones_like(coords[..., :1])], axis=-1)
    coords_transformed = np.einsum("...ij,...j->...i", extrinsics, coords_homo)
    coords_transformed = coords_transformed[..., :3] / coords_transformed[..., 3:4]
    return coords_transformed


@register_transform("eraser")
def eraser_transform(
    data: RawSliceData,
    executor: ThreadPoolExecutor,
    rng: np.random.Generator,
    prob: float,
    bounds: List[int],
    max: int,
    _ver2: bool = False,
    **kwargs,
) -> RawSliceData:
    data = data.copy()
    # https://github.com/facebookresearch/co-tracker/blob/main/cotracker//kubric_movif_dataset.py#L19
    S, H, W = data.rgbs.shape[:3]
    rgbs = [rgb.astype(np.float32).copy() for rgb in data.rgbs]

    gt_trajs_local = np.einsum(
        "tij,tnj->tni",
        data.gt_extrinsics,
        np.concatenate(
            [data.gt_trajs_3d, np.ones_like(data.gt_trajs_3d[..., :1])], axis=-1
        ),
    )[..., :3]
    gt_trajs_2d = np.einsum("tij,tnj->tni", data.gt_intrinsics, gt_trajs_local)
    gt_trajs_2d = gt_trajs_2d[..., :2] / gt_trajs_2d[..., 2:3]
    # make sure we do not accidentally modify the original data
    data.visibs = data.visibs.copy()
    data.est_depths = data.est_depths.copy()
    for i in range(1, S):
        if rng.random() < prob:
            for _ in range(rng.integers(1, max + 1)):  # number of times to occlude
                xc = rng.integers(0, W)
                yc = rng.integers(0, H)
                dx = rng.integers(bounds[0], bounds[1])
                dy = rng.integers(bounds[0], bounds[1])
                x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                mean_color = np.mean(rgbs[i][y0:y1, x0:x1, :].reshape(-1, 3), axis=0)
                rgbs[i][y0:y1, x0:x1, :] = mean_color

                # only est_depths will be provided to the model
                mean_depths = np.mean(data.est_depths[i][y0:y1, x0:x1], axis=0)
                data.est_depths[i][y0:y1, x0:x1] = mean_depths

                if _ver2:
                    target_depth = np.min(data.gt_depths[i][y0:y1, x0:x1], axis=0)
                    target_depth = target_depth * rng.uniform(0.8, 1.0)
                    data.est_depths[i][y0:y1, x0:x1] = target_depth

                occ_inds = np.logical_and(
                    np.logical_and(
                        gt_trajs_2d[i, :, 0] >= x0, gt_trajs_2d[i, :, 0] < x1
                    ),
                    np.logical_and(
                        gt_trajs_2d[i, :, 1] >= y0, gt_trajs_2d[i, :, 1] < y1
                    ),
                )
                data.visibs[i, occ_inds] = 0

    rgbs = [rgb.astype(np.uint8) for rgb in rgbs]
    data.rgbs = np.stack(rgbs)
    return data


@register_transform("replace")
def replace_transform(
    data: RawSliceData,
    executor: ThreadPoolExecutor,
    rng: np.random.Generator,
    prob: float,
    bounds: List[int],
    max: int,
    **kwargs,
) -> RawSliceData:
    data = data.copy()
    # https://github.com/facebookresearch/co-tracker/blob/main/cotracker//kubric_movif_dataset.py#L19
    S, H, W = data.rgbs.shape[:3]
    rgbs = [rgb.astype(np.float32).copy() for rgb in data.rgbs]
    gt_trajs_local = np.einsum(
        "tij,tnj->tni",
        data.gt_extrinsics,
        np.concatenate(
            [data.gt_trajs_3d, np.ones_like(data.gt_trajs_3d[..., :1])], axis=-1
        ),
    )[..., :3]
    gt_trajs_2d = np.einsum("tij,tnj->tni", data.gt_intrinsics, gt_trajs_local)
    gt_trajs_2d = gt_trajs_2d[..., :2] / gt_trajs_2d[..., 2:3]
    # make sure we do not accidentally modify the original data
    data.visibs = data.visibs.copy()
    data.est_depths = data.est_depths.copy()
    for i in range(1, S):
        if rng.random() < prob:
            for _ in range(rng.integers(1, max + 1)):  # number of times to occlude
                xc = rng.integers(0, W)
                yc = rng.integers(0, H)
                dx = rng.integers(bounds[0], bounds[1])
                dy = rng.integers(bounds[0], bounds[1])
                x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                wid = x1 - x0
                hei = y1 - y0
                y00 = rng.integers(0, H - hei)
                x00 = rng.integers(0, W - wid)
                fr = rng.integers(0, S)
                rep_rgb = rgbs[fr][y00 : y00 + hei, x00 : x00 + wid, :]
                rgbs[i][y0:y1, x0:x1, :] = rep_rgb

                rep_depth = data.est_depths[fr][y00 : y00 + hei, x00 : x00 + wid]
                rep_depth = (
                    rep_depth / rep_depth.max() * data.est_depths[i][y0:y1, x0:x1].min()
                )
                rep_depth = rng.uniform(0.8, 1.0) * rep_depth
                data.est_depths[i][y0:y1, x0:x1] = rep_depth

                occ_inds = np.logical_and(
                    np.logical_and(
                        gt_trajs_2d[i, :, 0] >= x0, gt_trajs_2d[i, :, 0] < x1
                    ),
                    np.logical_and(
                        gt_trajs_2d[i, :, 1] >= y0, gt_trajs_2d[i, :, 1] < y1
                    ),
                )
                data.visibs[i, occ_inds] = 0

    rgbs = [rgb.astype(np.uint8) for rgb in rgbs]
    data.rgbs = np.stack(rgbs)
    return data


@register_transform("color_jitter")
def color_jitter_transform(
    data: RawSliceData,
    executor: ThreadPoolExecutor,
    rng: np.random.Generator,
    prob: float,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
    **kwargs,
) -> RawSliceData:
    # threaded execution will break the reproducibility
    if rng.random() < prob:
        data = data.copy()

        with TemporarySeed(int(rng.integers(0, 2**32 - 1))):
            photo_aug = ColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
            )
            rgbs = [
                np.array(photo_aug(Image.fromarray(rgb.copy())), dtype=np.uint8)
                for rgb in data.rgbs
            ]
            data.rgbs = np.stack(rgbs)

    return data


@register_transform("blur")
def blur_transform(
    data: RawSliceData,
    executor: ThreadPoolExecutor,
    rng: np.random.Generator,
    prob: float,
    kernel_size: int,
    sigma: List[float],
    **kwargs,
) -> RawSliceData:
    if rng.random() < prob:
        data = data.copy()

        with TemporarySeed(int(rng.integers(0, 2**32 - 1))):
            photo_aug = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            rgbs = [
                np.array(photo_aug(Image.fromarray(rgb.copy())), dtype=np.uint8)
                for rgb in data.rgbs
            ]
            data.rgbs = np.stack(rgbs)

    return data


@register_transform("blur_depth")
def blur_depth_transform(
    data: RawSliceData,
    executor: ThreadPoolExecutor,
    rng: np.random.Generator,
    prob: float,
    gn_kernel: Optional[tuple] = None,
    gn_sigma: Optional[tuple] = None,
    gn_kernel_range: Optional[tuple] = None,
    gn_sigma_range: Optional[tuple] = None,
    **kwargs,
) -> RawSliceData:
    if rng.random() > prob:
        return data

    data = data.copy()

    if gn_kernel is None:
        gn_kernel = rng.integers(
            (gn_kernel_range[0] - 1 + 1) // 2, (gn_kernel_range[1] - 1) // 2 + 1
        )  # type: ignore
        gn_kernel = 2 * gn_kernel + 1  # type: ignore
        gn_kernel = (gn_kernel, gn_kernel)
    if gn_sigma is None and gn_sigma_range is not None:
        gn_sigma = rng.uniform(gn_sigma_range[0], gn_sigma_range[1])  # type: ignore
        gn_sigma = (gn_sigma, gn_sigma)

    gn_kernel = list(gn_kernel)  # type: ignore
    if gn_sigma is not None:
        gn_sigma = list(gn_sigma)  # type: ignore

    mask = data.est_depths > 0
    est_depths_torch = torch.tensor(data.est_depths.copy())
    est_depths_torch = TF.gaussian_blur(
        est_depths_torch, kernel_size=gn_kernel, sigma=gn_sigma
    )  # type: ignore
    est_depths = est_depths_torch.numpy().copy()

    est_depths[~mask] = 0.0
    data.est_depths = est_depths

    return data


@register_transform("spatial")
def spatial_transform(
    data: RawSliceData,
    executor: ThreadPoolExecutor,
    rng: np.random.Generator,
    target_resolution: List[int],
    pad_bounds: List[int],
    resize_lim: List[float],
    resize_delta: float,
    max_crop_offset: int,
    h_flip_prob: float,
    v_flip_prob: float,
    t_flip_prob: float,
    **kwargs,
) -> RawSliceData:
    data = data.copy()
    data.rgbs = data.rgbs.copy()
    data.gt_depths = data.gt_depths.copy()
    data.est_depths = data.est_depths.copy()
    data.gt_intrinsics = data.gt_intrinsics.copy()
    data.est_intrinsics = data.est_intrinsics.copy()
    data.gt_extrinsics = data.gt_extrinsics.copy()
    data.est_extrinsics = data.est_extrinsics.copy()

    T, N, __ = data.gt_trajs_3d.shape

    S = len(data.rgbs)
    H, W = data.rgbs[0].shape[:2]
    assert S == T

    rgbs = [rgb.astype(np.float32).copy() for rgb in data.rgbs]

    ############ spatial transform ############

    # padding
    pad_x0 = rng.integers(pad_bounds[0], pad_bounds[1])
    pad_x1 = rng.integers(pad_bounds[0], pad_bounds[1])
    pad_y0 = rng.integers(pad_bounds[0], pad_bounds[1])
    pad_y1 = rng.integers(pad_bounds[0], pad_bounds[1])

    rgbs = [np.pad(rgb, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for rgb in rgbs]
    gt_depths = [
        np.pad(depth, ((pad_y0, pad_y1), (pad_x0, pad_x1))) for depth in data.gt_depths
    ]
    est_depths = [
        np.pad(depth, ((pad_y0, pad_y1), (pad_x0, pad_x1))) for depth in data.est_depths
    ]
    if data.segmentation is not None:
        segmentation = [
            np.pad(
                segmentation, ((pad_y0, pad_y1), (pad_x0, pad_x1)), constant_values=-1
            )
            for segmentation in data.segmentation
        ]
    else:
        segmentation = None

    if data.bg_mask is not None:
        bg_mask = [
            np.pad(mask, ((pad_y0, pad_y1), (pad_x0, pad_x1)), constant_values=0)
            for mask in data.bg_mask
        ]
    else:
        bg_mask = None

    data.gt_intrinsics[:, 0, 2] += pad_x0
    data.gt_intrinsics[:, 1, 2] += pad_y0
    data.est_intrinsics[:, 0, 2] += pad_x0
    data.est_intrinsics[:, 1, 2] += pad_y0

    H, W = rgbs[0].shape[:2]

    # scaling + stretching
    scale = rng.uniform(resize_lim[0], resize_lim[1])
    scale_x = scale
    scale_y = scale
    H_new = H
    W_new = W

    scale_delta_x = 0.0
    scale_delta_y = 0.0

    rgbs_scaled = []
    gt_depths_scaled = []
    est_depths_scaled = []
    segmentation_scaled = []
    bg_mask_scaled = []
    for s in range(S):
        if s == 1:
            scale_delta_x = rng.uniform(-resize_delta, resize_delta)
            scale_delta_y = rng.uniform(-resize_delta, resize_delta)
        elif s > 1:
            scale_delta_x = (
                scale_delta_x * 0.8 + rng.uniform(-resize_delta, resize_delta) * 0.2
            )
            scale_delta_y = (
                scale_delta_y * 0.8 + rng.uniform(-resize_delta, resize_delta) * 0.2
            )
        scale_x = scale_x + scale_delta_x
        scale_y = scale_y + scale_delta_y

        # bring h/w closer
        scale_xy = (scale_x + scale_y) * 0.5
        scale_x = scale_x * 0.5 + scale_xy * 0.5
        scale_y = scale_y * 0.5 + scale_xy * 0.5

        # don't get too crazy
        scale_x = np.clip(scale_x, 0.2, 2.0)
        scale_y = np.clip(scale_y, 0.2, 2.0)

        H_new = int(H * scale_y)
        W_new = int(W * scale_x)

        # make it at least slightly bigger than the crop area,
        # so that the random cropping can add diversity
        H_new = np.clip(H_new, target_resolution[0] + 10, None)
        W_new = np.clip(W_new, target_resolution[1] + 10, None)
        # recompute scale in case we clipped
        scale_x = (W_new) / float(W)
        scale_y = (H_new) / float(H)
        rgbs_scaled.append(
            cv2.resize(rgbs[s], (W_new, H_new), interpolation=cv2.INTER_LINEAR)
        )
        # INTER_NEAREST_EXACT_exact doesn't work on meta environment
        gt_depths_scaled.append(
            cv2.resize(gt_depths[s], (W_new, H_new), interpolation=cv2.INTER_NEAREST_EXACT)
        )
        est_depths_scaled.append(
            cv2.resize(est_depths[s], (W_new, H_new), interpolation=cv2.INTER_NEAREST_EXACT)
        )
        if segmentation is not None:
            segmentation_scaled.append(
                cv2.resize(
                    segmentation[s],
                    (W_new, H_new),
                    interpolation=cv2.INTER_NEAREST_EXACT,
                )
            )
        if bg_mask is not None:
            bg_mask_scaled.append(
                (
                    cv2.resize(
                        bg_mask[s].astype(np.float32),
                        (W_new, H_new),
                        interpolation=cv2.INTER_NEAREST_EXACT,
                    )
                ).astype(bool)
            )
        data.gt_intrinsics[s, 0, :] *= scale_x
        data.gt_intrinsics[s, 1, :] *= scale_y
        data.est_intrinsics[s, 0, :] *= scale_x
        data.est_intrinsics[s, 1, :] *= scale_y

    rgbs = rgbs_scaled
    gt_depths = gt_depths_scaled
    est_depths = est_depths_scaled
    segmentation = segmentation_scaled if segmentation is not None else None
    bg_mask = bg_mask_scaled if bg_mask is not None else None

    ok_inds = data.visibs[0, :] > 0
    vis_trajs = data.gt_trajs_3d[:, ok_inds]  # S,?,2

    if vis_trajs.shape[1] > 0:
        mid_x = np.mean(vis_trajs[0, :, 0])
        mid_y = np.mean(vis_trajs[0, :, 1])
    else:
        mid_y = target_resolution[0]
        mid_x = target_resolution[1]

    x0 = int(mid_x - target_resolution[1] // 2)
    y0 = int(mid_y - target_resolution[0] // 2)

    offset_x = 0
    offset_y = 0

    for s in range(S):
        # on each frame, shift a bit more
        if s == 1:
            offset_x = rng.integers(-max_crop_offset, max_crop_offset)
            offset_y = rng.integers(-max_crop_offset, max_crop_offset)
        elif s > 1:
            offset_x = int(
                offset_x * 0.8
                + rng.integers(-max_crop_offset, max_crop_offset + 1) * 0.2
            )
            offset_y = int(
                offset_y * 0.8
                + rng.integers(-max_crop_offset, max_crop_offset + 1) * 0.2
            )
        x0 = x0 + offset_x
        y0 = y0 + offset_y

        H_new, W_new = rgbs[s].shape[:2]
        if H_new == target_resolution[0]:
            y0 = 0
        else:
            y0 = min(max(0, y0), H_new - target_resolution[0])

        if W_new == target_resolution[1]:
            x0 = 0
        else:
            x0 = min(max(0, x0), W_new - target_resolution[1])

        rgbs[s] = rgbs[s][
            y0 : y0 + target_resolution[0], x0 : x0 + target_resolution[1]
        ]
        gt_depths[s] = gt_depths[s][
            y0 : y0 + target_resolution[0], x0 : x0 + target_resolution[1]
        ]
        est_depths[s] = est_depths[s][
            y0 : y0 + target_resolution[0], x0 : x0 + target_resolution[1]
        ]
        if segmentation is not None:
            segmentation[s] = segmentation[s][
                y0 : y0 + target_resolution[0], x0 : x0 + target_resolution[1]
            ]
        if bg_mask is not None:
            bg_mask[s] = bg_mask[s][
                y0 : y0 + target_resolution[0], x0 : x0 + target_resolution[1]
            ]

        data.gt_intrinsics[s, 0, 2] -= x0
        data.gt_intrinsics[s, 1, 2] -= y0
        data.est_intrinsics[s, 0, 2] -= x0
        data.est_intrinsics[s, 1, 2] -= y0

    H_new = target_resolution[0]
    W_new = target_resolution[1]

    if data.gt_query_point is not None:
        if type(data.gt_query_point) == np.ndarray:
            data.gt_query_point = torch.from_numpy(data.gt_query_point)
        gt_query_frames = data.gt_query_point[..., 0].long()  # type: ignore
        gt_query_coords = data.gt_query_point[..., 1:]  # type: ignore
        gt_query_coords_local = apply_extrinsics(
            data.gt_extrinsics[gt_query_frames], gt_query_coords
        )
    if data.est_query_point is not None:
        if type(data.est_query_point) == np.ndarray:
            data.est_query_point = torch.from_numpy(data.est_query_point)
        est_query_frames = data.est_query_point[..., 0].long()  # type: ignore
        est_query_coords = data.est_query_point[..., 1:]  # type: ignore
        est_query_coords_local = apply_extrinsics(
            data.est_extrinsics[est_query_frames], est_query_coords
        )

    gt_trajs_3d_local = apply_extrinsics(data.gt_extrinsics[:, None], data.gt_trajs_3d)
    est_trajs_3d_local = apply_extrinsics(
        data.est_extrinsics[:, None], data.est_trajs_3d
    )

    if rng.random() < h_flip_prob:
        rgbs = [rgb[:, ::-1] for rgb in rgbs]
        gt_depths = [depth[:, ::-1] for depth in gt_depths]
        est_depths = [depth[:, ::-1] for depth in est_depths]
        if segmentation is not None:
            segmentation = [seg[:, ::-1] for seg in segmentation]
        if bg_mask is not None:
            bg_mask = [mask[:, ::-1] for mask in bg_mask]
        gt_trajs_3d_local[..., 0] *= -1
        est_trajs_3d_local[..., 0] *= -1
        if data.gt_query_point is not None:
            gt_query_coords_local[..., 0] *= -1
        if data.est_query_point is not None:
            est_query_coords_local[..., 0] *= -1
        data.gt_intrinsics[:, 0, 2] = W_new - data.gt_intrinsics[:, 0, 2]
        data.est_intrinsics[:, 0, 2] = W_new - data.est_intrinsics[:, 0, 2]
        data.gt_extrinsics[..., 0, :] *= -1
        data.est_extrinsics[..., 0, :] *= -1
        data.gt_extrinsics[..., :, 2] *= -1
        data.est_extrinsics[..., :, 2] *= -1

    # v flip
    if rng.random() < v_flip_prob:
        rgbs = [rgb[::-1] for rgb in rgbs]
        gt_depths = [depth[::-1] for depth in gt_depths]
        est_depths = [depth[::-1] for depth in est_depths]
        if segmentation is not None:
            segmentation = [seg[::-1] for seg in segmentation]
        if bg_mask is not None:
            bg_mask = [mask[::-1] for mask in bg_mask]
        gt_trajs_3d_local[..., 1] *= -1
        est_trajs_3d_local[..., 1] *= -1
        if data.gt_query_point is not None:
            # dataset with query point has never been tested....
            import ipdb

            ipdb.set_trace()
            gt_query_coords_local[..., 1] *= -1
        if data.est_query_point is not None:
            est_query_coords_local[..., 1] *= -1
        data.gt_intrinsics[:, 1, 2] = H_new - data.gt_intrinsics[:, 1, 2]
        data.est_intrinsics[:, 1, 2] = H_new - data.est_intrinsics[:, 1, 2]
        data.gt_extrinsics[..., 1, :] *= -1
        data.est_extrinsics[..., 1, :] *= -1
        data.gt_extrinsics[..., :, 2] *= -1
        data.est_extrinsics[..., :, 2] *= -1

    # t flip
    if rng.random() < t_flip_prob:
        rgbs = rgbs[::-1]
        gt_depths = gt_depths[::-1]
        est_depths = est_depths[::-1]
        if segmentation is not None:
            segmentation = segmentation[::-1]
        if bg_mask is not None:
            bg_mask = bg_mask[::-1]
        gt_trajs_3d_local = gt_trajs_3d_local[::-1]
        est_trajs_3d_local = est_trajs_3d_local[::-1]
        if data.gt_query_point is not None:
            gt_query_frames = T - 1 - gt_query_frames  # type: ignore
        if data.est_query_point is not None:
            est_query_frames = T - 1 - est_query_frames  # type: ignore
        data.visibs = data.visibs[::-1]
        data.valids = data.valids[::-1]
        data.gt_intrinsics = data.gt_intrinsics[::-1]
        data.est_intrinsics = data.est_intrinsics[::-1]
        data.gt_extrinsics = data.gt_extrinsics[::-1]
        data.est_extrinsics = data.est_extrinsics[::-1]

    inv_gt_extrinsics = np.linalg.inv(data.gt_extrinsics)
    inv_est_extrinsics = np.linalg.inv(data.est_extrinsics)
    data.gt_trajs_3d = apply_extrinsics(inv_gt_extrinsics[:, None], gt_trajs_3d_local)
    data.est_trajs_3d = apply_extrinsics(
        inv_est_extrinsics[:, None], est_trajs_3d_local
    )
    if data.gt_query_point is not None:
        gt_query_coords = apply_extrinsics(
            inv_gt_extrinsics[gt_query_frames], gt_query_coords
        )
        data.gt_query_point = np.concatenate(
            [gt_query_frames.astype(np.float32), gt_query_coords], axis=-1
        )  # type: ignore
    if data.est_query_point is not None:
        est_query_coords = apply_extrinsics(
            inv_est_extrinsics[est_query_frames], est_query_coords
        )
        data.est_query_point = np.concatenate(
            [est_query_frames.astype(np.float32), est_query_coords], axis=-1
        )  # type: ignore

    data.rgbs = np.stack(rgbs).astype(np.uint8)
    data.gt_depths = np.stack(gt_depths)
    data.est_depths = np.stack(est_depths)
    if segmentation is not None:
        data.segmentation = np.stack(segmentation)
    if bg_mask is not None:
        data.bg_mask = np.stack(bg_mask)

    data.__post_init__()  # verify the data is valid
    return data


def transform_query_point(
    query_point: np.ndarray, extrinsics: np.ndarray
) -> np.ndarray:
    query_frames = query_point[..., 0].astype(np.int64)
    query_coords = query_point[..., 1:]
    extrinsics_at_query_frames = extrinsics[query_frames]
    query_coords_transformed = apply_extrinsics(
        extrinsics_at_query_frames, query_coords
    )
    return np.concatenate(
        [query_frames[..., None].astype(np.float32), query_coords_transformed], axis=-1
    )


@register_transform("to_local")
@profile
def to_local(
    data: RawSliceData, executor: ThreadPoolExecutor, rng: np.random.Generator, **kwargs
) -> RawSliceData:
    data = data.copy()
    data.flags = data.flags.copy()
    data.flags.append("local")

    T, N, _ = data.est_trajs_3d.shape

    extrinsics = data.est_extrinsics
    inv_extrinsics = np.linalg.inv(extrinsics)
    data.est_extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], T, axis=0)
    data.gt_extrinsics = np.einsum("tij,tjk->tik", data.gt_extrinsics, inv_extrinsics)
    data.est_trajs_3d = apply_extrinsics(extrinsics[:, None], data.est_trajs_3d)
    data.gt_trajs_3d = apply_extrinsics(extrinsics[:, None], data.gt_trajs_3d)

    if data.est_query_point is not None:
        data.est_query_point = transform_query_point(data.est_query_point, extrinsics)
    if data.gt_query_point is not None:
        data.gt_query_point = transform_query_point(data.gt_query_point, extrinsics)

    return data

@register_transform("to_local_alltracker")
@profile
def to_local_alltracker(
    data: RawSliceData, executor: ThreadPoolExecutor, rng: np.random.Generator, **kwargs
) -> RawSliceData:
    """Torch-native local transform for alltracker fields.

    Expects `data.est_extrinsics`, `data.est_trajs_3d`, and optional
    `data.est_query_point` to be `torch.Tensor`.
    """
    data = data.copy()
    # data.flags = data.flags.copy()
    # data.flags.append("local")

    T, _, _ = data.est_trajs_3d.shape

    extrinsics_t = data.est_extrinsics.to(torch.float32)
    est_trajs_3d_t = data.est_trajs_3d.to(torch.float32)

    data.est_extrinsics = torch.eye(
        4, dtype=torch.float32, device=extrinsics_t.device
    )[None].repeat(T, 1, 1)

    est_trajs_homo_t = torch.cat(
        [est_trajs_3d_t, torch.ones_like(est_trajs_3d_t[..., :1])], dim=-1
    )
    est_trajs_transformed_t = torch.einsum(
        "...ij,...j->...i", extrinsics_t[:, None], est_trajs_homo_t
    )
    data.est_trajs_3d = est_trajs_transformed_t[..., :3] / est_trajs_transformed_t[
        ..., 3:4
    ]
    # data.gt_trajs_3d = apply_extrinsics(extrinsics[:, None], data.gt_trajs_3d)

    if data.est_query_point is not None:
        est_query_point_t = data.est_query_point.to(torch.float32)
        query_frames_t = est_query_point_t[..., 0].to(torch.int64)
        query_coords_t = est_query_point_t[..., 1:]
        extrinsics_at_query_frames_t = extrinsics_t[query_frames_t]

        query_coords_homo_t = torch.cat(
            [query_coords_t, torch.ones_like(query_coords_t[..., :1])], dim=-1
        )
        query_coords_transformed_t = torch.einsum(
            "...ij,...j->...i", extrinsics_at_query_frames_t, query_coords_homo_t
        )
        query_coords_transformed_t = (
            query_coords_transformed_t[..., :3] / query_coords_transformed_t[..., 3:4]
        )
        data.est_query_point = torch.cat(
            [query_frames_t[..., None].to(torch.float32), query_coords_transformed_t],
            dim=-1,
        )
    
    # if data.gt_query_point is not None:
        # data.gt_query_point = transform_query_point(data.gt_query_point, extrinsics)

    return data


# https://github.com/prs-eth/Marigold/blob/62413d56099d36573b2de1eb8c429839734b7782/src/util/multi_res_noise.py#L8
def multi_res_noise_like_np(
    depths: np.ndarray,
    power=5,
    downscale_strategy="original",
    rng: np.random.Generator = None,
):
    if rng is None:
        rng = np.random.default_rng()

    if isinstance(power, np.ndarray):
        strength = strength.reshape((1, 1, 1))

    t, h, w = depths.shape
    assert t < 512, "cv2.resize does not support more than 512 frames"

    noise = rng.standard_normal(size=(h, w, t), dtype=np.float32)

    w_cur, h_cur = w, h

    if downscale_strategy == "original":
        for i in range(10):
            r = rng.random() * 2 + 2
            w_cur = max(1, int(w_cur / (r**i)))
            h_cur = max(1, int(h_cur / (r**i)))

            new_noise = rng.standard_normal(size=(h_cur, w_cur, t), dtype=np.float32)
            resized_noise = cv2.resize(
                new_noise, (w, h), interpolation=cv2.INTER_LINEAR
            )
            noise += resized_noise * (power**i)

            if w_cur == 1 or h_cur == 1:
                break

    elif downscale_strategy == "every_layer":
        num_layers = int(math.log2(min(w, h)))
        for i in range(num_layers):
            w_cur = max(1, w_cur // 2)
            h_cur = max(1, h_cur // 2)

            new_noise = rng.standard_normal(size=(h_cur, w_cur, t), dtype=np.float32)
            resized_noise = cv2.resize(
                new_noise, (w, h), interpolation=cv2.INTER_LINEAR
            )
            noise += resized_noise * (power**i)

    elif downscale_strategy == "power_of_two":
        for i in range(10):
            w_cur = max(1, int(w_cur / (2**i)))
            h_cur = max(1, int(h_cur / (2**i)))

            new_noise = rng.standard_normal(size=(h_cur, w_cur, t), dtype=np.float32)
            resized_noise = cv2.resize(
                new_noise, (w, h), interpolation=cv2.INTER_LINEAR
            )
            noise += resized_noise * (power**i)

            if w_cur == 1 or h_cur == 1:
                break

    elif downscale_strategy == "random_step":
        for i in range(10):
            r = rng.random() * 2 + 2
            w_cur = max(1, int(w_cur / r))
            h_cur = max(1, int(h_cur / r))

            new_noise = rng.standard_normal(size=(h_cur, w_cur, t), dtype=np.float32)
            resized_noise = cv2.resize(
                new_noise, (w, h), interpolation=cv2.INTER_LINEAR
            )
            noise += resized_noise * (power**i)

            if w_cur == 1 or h_cur == 1:
                break

    else:
        raise ValueError(f"Unknown downscale strategy: {downscale_strategy}")

    noise_std = noise.std()
    if noise_std > 0:
        noise /= noise_std

    noise = noise.transpose(2, 0, 1)

    return noise


@register_transform("depth_noise")
def depth_noise_transform(
    data: RawSliceData,
    executor: ThreadPoolExecutor,
    rng: np.random.Generator,
    prob: float,
    power_range: Tuple[float, float],
    strength_range: Tuple[float, float],
    **kwargs,
) -> RawSliceData:
    if rng.random() > prob:
        return data

    T, H, W = data.est_depths.shape
    _depths = data.est_depths.copy()
    _depths[_depths == 0] = np.nan
    std = rng.uniform(*strength_range) * np.nanstd(_depths.reshape(T, -1), axis=-1)
    power = rng.uniform(*power_range)
    noise = std[:, None, None] * multi_res_noise_like_np(
        data.est_depths, power=power, rng=rng
    )
    invalid_mask = data.est_depths == 0
    data.est_depths = data.est_depths + noise
    data.est_depths[invalid_mask] = 0
    data.est_depths = np.clip(data.est_depths, 0, None)
    return data


@register_transform("delta_depth_aug")
def delta_depth_aug(
    data: RawSliceData,
    executor: ThreadPoolExecutor,
    rng: np.random.Generator,
    prob: float = 0.5,
    grid: tuple = (8, 8),
    scale: tuple = (0.85, 1.15),
    shift: tuple = (-0.05, 0.05),
    gn_kernel: tuple = (7, 7),
    gn_sigma: tuple = (2.0, 2.0),
    **kwargs,
) -> RawSliceData:
    if rng.random() > prob:
        return data

    data = data.copy()

    mask = data.est_depths != 0
    T, H, W = data.est_depths.shape
    scale_maps = []
    shift_maps = []
    for _ in range(T):
        scale_map = rng.uniform(scale[0], scale[1], (grid[0], grid[1]))
        shift_map = rng.uniform(shift[0], shift[1], (grid[0], grid[1]))
        scale_maps.append(scale_map)
        shift_maps.append(shift_map)
    scale_map_futures = []
    shift_map_futures = []
    for scale_map, shift_map in zip(scale_maps, shift_maps):
        scale_map_futures.append(
            executor.submit(
                lambda img: cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR),
                scale_map,
            )
        )
        shift_map_futures.append(
            executor.submit(
                lambda img: cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR),
                shift_map,
            )
        )
    scale_maps = [future.result() for future in scale_map_futures]
    shift_maps = [future.result() for future in shift_map_futures]
    scale_maps = np.stack(scale_maps)
    shift_maps = np.stack(shift_maps)

    est_depths = data.est_depths.copy()

    shift_scale = est_depths[mask].mean()
    # import ipdb; ipdb.set_trace()
    est_depths[mask] = (est_depths[mask] * scale_maps[mask]) + shift_maps[
        mask
    ] * shift_scale

    est_depths_torch = torch.tensor(est_depths)
    est_depths_torch = TF.gaussian_blur(
        est_depths_torch, kernel_size=list(gn_kernel), sigma=list(gn_sigma)
    )  # type: ignore
    est_depths = est_depths_torch.numpy().copy()

    est_depths[~mask] = 0.0
    data.est_depths = est_depths

    est_trajs_local = np.einsum(
        "tij,tnj->tni",
        data.est_extrinsics,
        np.concatenate(
            [data.est_trajs_3d, np.ones_like(data.est_trajs_3d[..., :1])], axis=-1
        ),
    )[..., :3]
    est_trajs_2d = np.einsum("tij,tnj->tni", data.est_intrinsics, est_trajs_local)
    est_trajs_2d = est_trajs_2d[..., :2] / est_trajs_2d[..., 2:3]

    est_trajs_scales = (
        bilinear_sampler(
            torch.tensor(scale_maps - 1.0, dtype=torch.float32)[:, None],
            torch.tensor(est_trajs_2d)[:, None],
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0, 0].numpy()
        + 1.0
    ).copy()

    est_trajs_shifts = (
        bilinear_sampler(
            torch.tensor(shift_maps, dtype=torch.float32)[:, None],
            torch.tensor(est_trajs_2d)[:, None],
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0, 0]
        .numpy()
        .copy()
    )

    est_trajs_depths = (
        est_trajs_local[..., -1] * est_trajs_scales + est_trajs_shifts * shift_scale
    )
    est_trajs_local = est_trajs_local * (
        est_trajs_depths[..., None] / est_trajs_local[..., -1:]
    )
    est_trajs_local_homo = np.concatenate(
        [est_trajs_local, np.ones_like(est_trajs_local[..., :1])], axis=-1
    )
    est_trajs_world = np.einsum(
        "tij,tnj->tni", np.linalg.inv(data.est_extrinsics), est_trajs_local_homo
    )
    est_trajs_world = est_trajs_world[..., :3]

    data.est_trajs_3d = est_trajs_world
    # print (data.est_depths.min(), data.est_depths.max())
    if data.est_depths.min() < 0:
        import ipdb

        ipdb.set_trace()

    return data


def _skew_symmetric3(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def _se3_exp_sophus(tangent6: np.ndarray) -> np.ndarray:
    """Group exponential SE3: tangent = [upsilon(3), omega(3)], matches Sophus SE3::exp."""
    a = np.asarray(tangent6, dtype=np.float64).reshape(6)
    upsilon = a[:3]
    omega = a[3:]
    theta_sq = float(np.dot(omega, omega))
    omega_hat = _skew_symmetric3(omega)
    omega_hat_sq = omega_hat @ omega_hat
    eps = 1e-12
    if theta_sq < eps * eps:
        v = np.eye(3, dtype=np.float64) + 0.5 * omega_hat
    else:
        theta = math.sqrt(theta_sq)
        v = (
            np.eye(3, dtype=np.float64)
            + (1.0 - math.cos(theta)) / theta_sq * omega_hat
            + (theta - math.sin(theta)) / (theta_sq * theta) * omega_hat_sq
        )
    r = R.from_rotvec(omega)
    t = v @ upsilon
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r.as_matrix()
    out[:3, 3] = t
    return out


def random_poses(
    T: int,
    rng: np.random.Generator,
    init_vel_range: Tuple[float, float],
    init_ang_range: Tuple[float, float],
    scene_scale: float,
    vel_noise: float,
    ang_noise: float,
    alpha: float,
    keep_first: bool = False,
) -> np.ndarray:
    """
    pose means the local to world matrix
    """
    init_SE3 = np.eye(4, dtype=np.float32)
    if not keep_first:
        rand_quat = rng.normal(size=(4,))
        rand_quat /= np.linalg.norm(rand_quat)
        rand_quat = R.from_quat(rand_quat)
        init_SE3[:3, :3] = rand_quat.as_matrix()

    vel = rng.uniform(*init_vel_range, size=(3,))
    ang = rng.normal(size=(3,))
    ang = ang / np.linalg.norm(ang) * rng.uniform(*init_ang_range)

    cur_SE3 = init_SE3
    ret = [init_SE3]
    for i in range(1, T):
        if i > 1:
            vel = alpha * vel + (1 - alpha) * rng.normal(size=(3,)) * vel_noise
            ang = alpha * ang + (1 - alpha) * rng.normal(size=(3,)) * ang_noise
        vel_scaled = vel * scene_scale
        delta = _se3_exp_sophus(np.concatenate([vel_scaled, ang], axis=-1)).astype(
            np.float32, copy=False
        )
        cur_SE3 = cur_SE3 @ delta
        ret.append(cur_SE3)

    return np.stack(ret).astype(np.float32)


@torch.no_grad()
def scene_transform_augs(
    data: SliceData,
    rng: np.random.Generator,
    prob: float = 1.0,
    rot_x_range: Tuple[float, float] = (-15.0, 15.0),
    rot_y_range: Tuple[float, float] = (-15.0, 15.0),
    rot_z_range: Tuple[float, float] = (0.0, 0.0),
    scale_range: Tuple[float, float] = (0.8, 1.5),
    translate_range: Tuple[float, float] = (-0.5, 0.5),
) -> SliceData:
    if rng.random() > prob:
        return data

    data = data.copy()

    gt_trajs = data.gt_trajs_3d.clone()
    valid_mask = data.valids
    gt_trajs[~valid_mask] = float("nan")
    scene_center = gt_trajs.nanmean(dim=(0, 1))  # (3,)
    scene_scale: float = gt_trajs[valid_mask].std().item()

    rot_x_angle = rng.uniform(*rot_x_range)
    rot_y_angle = rng.uniform(*rot_y_range)
    rot_z_angle = rng.uniform(*rot_z_range)
    scale = rng.uniform(*scale_range)
    translate = rng.uniform(*translate_range, size=(3,)) * scene_scale

    rot = (
        R.from_euler("z", rot_z_angle, degrees=True)
        * R.from_euler("y", rot_y_angle, degrees=True)
        * R.from_euler("x", rot_x_angle, degrees=True)
    ).as_matrix()

    # center -> rotate+scale -> uncenter + translate
    T_center = np.eye(4, dtype=np.float64)
    T_center[:3, 3] = -scene_center.numpy().astype(np.float64)
    T_rot_scale = np.eye(4, dtype=np.float64)
    T_rot_scale[:3, :3] = rot * scale
    T_uncenter = np.eye(4, dtype=np.float64)
    T_uncenter[:3, 3] = scene_center.numpy().astype(np.float64) + translate

    T_np = T_uncenter @ T_rot_scale @ T_center
    T = torch.from_numpy(T_np).float()
    T_inv = torch.from_numpy(np.linalg.inv(T_np)).float()

    def _transform_pts(points: torch.Tensor) -> torch.Tensor:
        """Apply T to world-space points of shape (..., 3)."""
        orig_shape = points.shape
        pts = points.reshape(-1, 3)
        pts_homo = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=-1)
        out = (T @ pts_homo.T).T[:, :3]
        return out.reshape(orig_shape)

    # world-space trajectories: p' = T @ p
    data.gt_trajs_3d = _transform_pts(data.gt_trajs_3d)
    data.est_trajs_3d = _transform_pts(data.est_trajs_3d)

    # extrinsics (world-to-cam): E' = E @ T^{-1}
    data.gt_extrinsics = data.gt_extrinsics @ T_inv
    data.est_extrinsics = data.est_extrinsics @ T_inv

    # query points (frame_idx, x, y, z)
    gt_qp = data.gt_query_point.clone()
    gt_qp[..., 1:] = _transform_pts(gt_qp[..., 1:])
    data.gt_query_point = gt_qp

    est_qp = data.est_query_point.clone()
    est_qp[..., 1:] = _transform_pts(est_qp[..., 1:])
    data.est_query_point = est_qp
     
    # dense trajectories (t, h, w, t, 3) — world-space per-pixel positions
    if data.dense_trajs_3d is not None:
        data.dense_trajs_3d = _transform_pts(data.dense_trajs_3d)

    if data.est_dense_trajs_3d is not None:
        data.est_dense_trajs_3d = _transform_pts(data.est_dense_trajs_3d)

    return data


@register_transform("camera_aug")
def camera_aug(
    data: RawSliceData,
    executor: ThreadPoolExecutor,
    rng: np.random.Generator,
    init_vel_range: Tuple[float, float],
    init_ang_range: Tuple[float, float],
    vel_noise: float,
    ang_noise: float,
    alpha: float,
    keep_first: bool = False,
    **kwargs,
) -> RawSliceData:
    # assert "local" in data.flags, "camera_aug should be applied after to_local"
    data = data.copy()

    gt_trajs_3d = data.gt_trajs_3d.copy()
    gt_trajs_3d[~data.valids] = np.nan
    centers = np.nanmean(gt_trajs_3d, axis=1)
    gt_trajs_3d = gt_trajs_3d - np.nanmean(gt_trajs_3d, axis=(1, 2), keepdims=True)
    scene_scale: float = np.nanstd(gt_trajs_3d)  # type: ignore

    T, N, _ = gt_trajs_3d.shape
    poses = random_poses(
        T,
        rng=rng,
        init_vel_range=init_vel_range,
        init_ang_range=init_ang_range,
        scene_scale=scene_scale,
        vel_noise=vel_noise,
        ang_noise=ang_noise,
        alpha=alpha,
        keep_first=keep_first,
    )

    centering = np.repeat(np.eye(4, dtype=np.float32)[None], T, axis=0)
    centering[:, :3, 3] = -centers
    poses = poses @ centering

    extrinsics = np.linalg.inv(poses)

    data.est_extrinsics = extrinsics
    data.gt_extrinsics = np.einsum("tij,tjk->tik", data.gt_extrinsics, extrinsics)
    data.gt_trajs_3d = apply_extrinsics(poses[:, None], data.gt_trajs_3d)
    data.est_trajs_3d = apply_extrinsics(poses[:, None], data.est_trajs_3d)
    if data.est_query_point is not None:
        data.est_query_point = transform_query_point(data.est_query_point, poses)
    if data.gt_query_point is not None:
        data.gt_query_point = transform_query_point(data.gt_query_point, poses)
    return data


@profile
def _filter_one_depth(
    depth: np.ndarray, depth_rtol: float, normal_tol: float, intrinsics: np.ndarray
) -> np.ndarray:
    inv_intrinsics = np.linalg.inv(intrinsics)
    uv_grid = np.meshgrid(
        np.arange(depth.shape[1]), np.arange(depth.shape[0]), indexing="xy"
    )
    uv_homo = np.stack([uv_grid[0], uv_grid[1], np.ones_like(uv_grid[0])], axis=-1)
    xyz_homo = np.einsum("ij,uvj->uvi", inv_intrinsics, uv_homo)
    xyz_homo = xyz_homo * depth[..., None]
    valid_mask = depth > 0.0
    normals, normals_mask = points_to_normals(xyz_homo, mask=valid_mask)
    depth_in = depth.copy().astype(np.float32)
    assert np.all(depth_in < 1e9)
    depth_in[~valid_mask] = 1e9
    edge_mask = depth_edge(depth_in, rtol=depth_rtol, mask=valid_mask) & normals_edge(
        normals, tol=normal_tol, mask=normals_mask
    )
    distance, indices = ndimage.distance_transform_edt(
        edge_mask | (~valid_mask), return_indices=True
    )  # type: ignore
    filled_depth = depth.copy()
    filled_depth[edge_mask] = depth[tuple(indices)][edge_mask]
    return filled_depth


@register_transform("resize")
def resize(
    data: RawSliceData,
    executor: ThreadPoolExecutor,
    rng: np.random.Generator,
    target_hw: Tuple[int, int],
    **kwargs,
) -> RawSliceData:
    data = data.copy()
    orig_hw = data.est_depths.shape[1:]
    crop_args = CropArgs(
        crop_start=(0, 0),
        crop_end=(orig_hw[0], orig_hw[1]),
        src_resolution=(orig_hw[0], orig_hw[1]),
        target_resolution=target_hw,
    )
    gt_intrinsics = crop_args.update_intrinsics(data.gt_intrinsics)
    est_intrinsics = crop_args.update_intrinsics(data.est_intrinsics)
    rgbs = crop_args.crop_img(
        data.rgbs, interp_mode=cv2.INTER_LANCZOS4, executor=executor
    )
    gt_depths = crop_args.crop_img(
        data.gt_depths, interp_mode=cv2.INTER_NEAREST_EXACT, executor=executor
    )
    est_depths = crop_args.crop_img(
        data.est_depths, interp_mode=cv2.INTER_NEAREST_EXACT, executor=executor
    )
    if data.segmentation is not None:
        segmentation = crop_args.crop_img(
            data.segmentation, interp_mode=cv2.INTER_NEAREST_EXACT, executor=executor
        )
    else:
        segmentation = None
    data.rgbs = rgbs
    data.gt_depths = gt_depths
    data.est_depths = est_depths
    data.gt_intrinsics = gt_intrinsics
    data.est_intrinsics = est_intrinsics
    data.segmentation = segmentation
    return data


@register_transform("filter_edge")
@profile
def filter_edge(
    data: RawSliceData,
    depth_rtol: float,
    normal_tol: float,
    executor: ThreadPoolExecutor,
    **kwargs,
) -> RawSliceData:
    data = data.copy()

    futures = []
    for i in range(data.est_depths.shape[0]):
        futures.append(
            executor.submit(
                _filter_one_depth,
                data.est_depths[i],
                depth_rtol,
                normal_tol,
                data.est_intrinsics[i],
            )
        )
    data.est_depths = np.stack([future.result() for future in futures])
    return data


@register_transform("add_flag")
def add_flag(
    data: RawSliceData,
    executor: ThreadPoolExecutor,
    rng: np.random.Generator,
    flag: str,
    **kwargs,
) -> RawSliceData:
    data = data.copy()
    data.flags = data.flags.copy()
    data.flags.append(flag)
    return data


def from_config(config: ListConfig) -> Callable[..., RawSliceData]:
    transforms = []
    for op in config:
        transforms.append(partial(all_transforms[op.name], **op.kwargs))  # type: ignore
    return compose(*transforms)


@register_transform("choice")
def choice(
    data: RawSliceData,
    executor: ThreadPoolExecutor,
    rng: np.random.Generator,
    transforms: List[DictConfig],
    **kwargs,
) -> RawSliceData:
    transforms = [from_config(transform) for transform in transforms]  # type: ignore
    transform = rng.choice(transforms)  # type: ignore
    return transform(data, executor, rng, **kwargs)


@register_transform("set_roi")
def set_roi(
    data: RawSliceData,
    executor: ThreadPoolExecutor,
    rng: np.random.Generator,
    ratio: float = 2.0,
    **kwargs,
) -> RawSliceData:
    depth_est = data.est_depths.copy()
    depth_est[depth_est == 0] = np.nan
    depth_gt = data.gt_depths.copy()
    depth_gt[depth_gt == 0] = np.nan

    data = data.copy()

    gt_trajs = data.gt_trajs_3d
    est_trajs = data.est_trajs_3d
    gt_trajs_homo = np.concatenate([gt_trajs, np.ones_like(gt_trajs[..., :1])], axis=-1)
    est_trajs_homo = np.concatenate(
        [est_trajs, np.ones_like(est_trajs[..., :1])], axis=-1
    )
    gt_trajs_local = np.einsum("tij,tnj->tni", data.gt_extrinsics, gt_trajs_homo)
    est_trajs_local = np.einsum("tij,tnj->tni", data.est_extrinsics, est_trajs_homo)
    gt_trajs_local = gt_trajs_local[..., :3]
    est_trajs_local = est_trajs_local[..., :3]

    # Prefer visible+valid traj depths; fall back to depth maps when filters
    # (POD OOB / motion caps / correction visibility drops) leave none — otherwise
    # `.max()` on an empty mask crashes the DataLoader worker.
    mask = data.valids & data.visibs
    if mask.any():
        gt_z_max = float(gt_trajs_local[..., -1][mask].max())
        est_z_max = float(est_trajs_local[..., -1][mask].max())
    else:
        gt_finite = np.isfinite(depth_gt)
        est_finite = np.isfinite(depth_est)
        gt_z_max = float(np.nanmax(depth_gt)) if gt_finite.any() else 1.0
        est_z_max = float(np.nanmax(depth_est)) if est_finite.any() else 1.0

    data.gt_depth_roi = np.array([0.0, gt_z_max * ratio], dtype=np.float32)
    data.est_depth_roi = np.array([0.0, est_z_max * ratio], dtype=np.float32)

    return data


@register_transform("prob_drop")
def prob_drop(
    data: RawSliceData,
    executor: ThreadPoolExecutor,
    rng: np.random.Generator,
    prob: float,
    transforms: List[DictConfig],
    **kwargs,
) -> RawSliceData:
    if rng.random() > prob:
        return data
    transform = from_config(transforms)  # type: ignore
    return transform(data=data, executor=executor, rng=rng, **kwargs)
