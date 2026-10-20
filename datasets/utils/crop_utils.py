# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclasses.dataclass
class CropArgs:
    crop_start: Tuple[int, int]
    crop_end: Tuple[int, int]
    src_resolution: Tuple[int, int]
    target_resolution: Tuple[int, int]

    def update_traj_2d(self, traj_2d: np.ndarray) -> np.ndarray:
        traj_2d = traj_2d.copy()

        crop_start_y, crop_start_x = self.crop_start
        crop_end_y, crop_end_x = self.crop_end
        crop_height = crop_end_y - crop_start_y
        crop_width = crop_end_x - crop_start_x
        target_height, target_width = self.target_resolution

        scale_y = target_height / crop_height
        scale_x = target_width / crop_width

        traj_2d[..., 0] = (traj_2d[..., 0] - crop_start_x) * scale_x
        traj_2d[..., 1] = (traj_2d[..., 1] - crop_start_y) * scale_y

        return traj_2d

    def update_intrinsics(self, intrinsics: np.ndarray) -> np.ndarray:
        intrinsics = intrinsics.copy()

        crop_start_y, crop_start_x = self.crop_start
        crop_end_y, crop_end_x = self.crop_end
        crop_height = crop_end_y - crop_start_y
        crop_width = crop_end_x - crop_start_x
        target_height, target_width = self.target_resolution

        scale_y = (target_height) / (crop_height)
        scale_x = (target_width) / (crop_width)

        intrinsics[:, 0, 2] -= crop_start_x
        intrinsics[:, 1, 2] -= crop_start_y

        intrinsics[:, 0, 0] *= scale_x
        intrinsics[:, 1, 1] *= scale_y
        intrinsics[:, 0, 2] *= scale_x
        intrinsics[:, 1, 2] *= scale_y

        return intrinsics

    def _process_img(
        self,
        img: np.ndarray,
        crop_start_y: int,
        crop_start_x: int,
        crop_end_y: int,
        crop_end_x: int,
        target_height: int,
        target_width: int,
        interp_mode: int,
    ) -> np.ndarray:
        to_resize = img[crop_start_y:crop_end_y, crop_start_x:crop_end_x].copy()
        return cv2.resize(
            to_resize, (target_width, target_height), interpolation=interp_mode
        )

    def crop_img(
        self,
        imgs: Union[np.ndarray, List[np.ndarray]],
        interp_mode: int,
        executor: ThreadPoolExecutor,
    ) -> np.ndarray:
        crop_start_y, crop_start_x = self.crop_start
        crop_end_y, crop_end_x = self.crop_end
        target_height, target_width = self.target_resolution

        processed_imgs = []

        for img in imgs:
            future = executor.submit(
                self._process_img,
                img,
                crop_start_y,
                crop_start_x,
                crop_end_y,
                crop_end_x,
                target_height,
                target_width,
                interp_mode,
            )
            processed_imgs.append(future)

        return np.stack([f.result() for f in processed_imgs])


def get_crop_args(
    target_resolution: Tuple[int, int],
    src_resolution: Tuple[int, int],
    aug_crop: Tuple[float, float],
    rng: np.random.Generator,
) -> CropArgs:
    """
    Get a random crop region from the image

    This function first figures out the maximum possible crop size that
        has the same aspect ratio as the target image.
    Then it randomly scales the crop size within the user-specified range.

    Args:
        target_resolution (Tuple[int, int]): Target resolution [height, width]
        src_resolution (Tuple[int, int]): Source resolution [height, width]
        aug_crop (Tuple[float]): The user-specified range mentioned above
        rng (np.random.Generator): Random number generator

    Returns:
        Arguments for cropping
    """

    target_h, target_w = target_resolution
    src_h, src_w = src_resolution
    min_scale, max_scale = aug_crop

    assert min_scale <= max_scale, "min_scale must be less than or equal to max_scale"
    assert 0 < min_scale <= max_scale <= 1, "scale must be between 0 and 1"

    # Calculate max possible crop size with same aspect ratio
    scale = min(src_h / target_h, src_w / target_w)
    max_crop_h = int(target_h * scale)
    max_crop_w = int(target_w * scale)

    # Randomly scale crop size within specified range
    random_scale = rng.uniform(min_scale, max_scale)
    crop_h = int(max_crop_h * random_scale)
    crop_w = int(max_crop_w * random_scale)

    # Randomly select crop region start position
    i_start = rng.integers(0, src_h - crop_h + 1)
    j_start = rng.integers(0, src_w - crop_w + 1)

    i_end = i_start + crop_h
    j_end = j_start + crop_w

    return CropArgs(
        crop_start=(i_start, j_start),
        crop_end=(i_end, j_end),
        src_resolution=src_resolution,
        target_resolution=target_resolution,
    )
