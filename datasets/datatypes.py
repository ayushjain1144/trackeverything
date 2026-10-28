# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses
from typing import Any, List, Literal, Optional, Union

import ipdb

import numpy as np
import torch
from einops import rearrange
try:
    from line_profiler import profile
except ImportError:  # optional; only used for profiling decorators
    def profile(func):
        return func

st = ipdb.set_trace

EPS = 1e-6


@dataclasses.dataclass
class RawSliceData:
    seq_name: str
    seq_id: int
    visibs: np.ndarray  # (t, n)
    valids: np.ndarray  # (t, n)
    rgbs: np.ndarray  # (t, h, w, 3)

    orig_resolution: np.ndarray  # (2)

    gt_intrinsics: np.ndarray  # (t, 3, 3)
    gt_extrinsics: np.ndarray  # (t, 4, 4)x
    gt_depths: np.ndarray  # (t, h, w)

    est_intrinsics: np.ndarray  # (t, 3, 3)
    est_extrinsics: np.ndarray  # (t, 4, 4)
    est_depths: np.ndarray  # (t, h, w)
    same_scale: bool

    est_trajs_3d: Optional[np.ndarray] = None  # (t, n, 3)
    gt_trajs_3d: Optional[np.ndarray] = None  # (t, n, 3)
    est_trajs_depths: Optional[np.ndarray] = None  # (t, n, 3)
    gt_trajs_depths: Optional[np.ndarray] = None  # (t, n, 3)
    is_dynamic: Optional[np.ndarray] = None  # (t)
    gt_trajs_2d: Optional[np.ndarray] = None  # (t, n, 2)
    est_trajs_2d: Optional[np.ndarray] = None  # (t, n, 2)
    gt_depth_roi: Optional[np.ndarray] = None  # (2,)
    est_depth_roi: Optional[np.ndarray] = None  # (2,)
    bg_mask: Optional[np.ndarray] = None

    segmentation: Optional[np.ndarray] = None  # (t, h, w)

    gt_query_point: Optional[np.ndarray] = None  # (n, 3)
    est_query_point: Optional[np.ndarray] = None  # (n, 3)

    num_views: Optional[int] = None

    flags: List[str] = dataclasses.field(default_factory=list)
    skip_post_init: bool = False
    annotated: bool = False

    @classmethod
    def create(cls, copy_gt_to_est: bool = False, **kwargs) -> "RawSliceData":
        kwargs = kwargs.copy()
        if copy_gt_to_est:
            if not "same_scale" in kwargs:
                kwargs["same_scale"] = True
            for key in kwargs.copy():
                if key.startswith("gt_"):
                    kwargs[key.replace("gt_", "est_")] = copy.deepcopy(kwargs[key])
        return cls(**kwargs)

    def __post_init__(self):
        if self.skip_post_init:
            return
        assert isinstance(self.seq_name, str), "seq_name must be a string"
        assert isinstance(self.seq_id, int), "seq_id must be an integer"

        assert (
            isinstance(self.visibs, np.ndarray) and self.visibs.dtype == np.bool_
        ), "visibs must be a boolean numpy array"
        assert (
            isinstance(self.valids, np.ndarray) and self.valids.dtype == np.bool_
        ), "valids must be a boolean numpy array"
        assert (
            isinstance(self.rgbs, np.ndarray) and self.rgbs.dtype == np.uint8
        ), "rgbs must be a uint8 numpy array"
        assert (
            isinstance(self.orig_resolution, np.ndarray)
            and self.orig_resolution.dtype == np.int32
        ), "orig_resolution must be a int32 numpy array"
        if self.segmentation is not None:
            assert (
                isinstance(self.segmentation, np.ndarray)
                and self.segmentation.dtype == np.int32
            ), "segmentation must be a uint8 numpy array"

        assert (
            len(self.rgbs.shape) == 4 and self.rgbs.shape[-1] == 3
        ), "the shape of rgbs must be (t, h, w, 3)"
        assert (
            len(self.orig_resolution.shape) == 1 and self.orig_resolution.shape[0] == 2
        ), "the shape of orig_resolution must be (2,)"

        # gt
        assert (
            isinstance(self.gt_depths, np.ndarray)
            and self.gt_depths.dtype == np.float32
        ), "gt_depths must be a float32 numpy array"
        assert (
            isinstance(self.gt_intrinsics, np.ndarray)
            and self.gt_intrinsics.dtype == np.float32
        ), "gt_intrinsics must be a float32 numpy array"
        assert (
            isinstance(self.gt_extrinsics, np.ndarray)
            and self.gt_extrinsics.dtype == np.float32
        ), "gt_extrinsics must be a float32 numpy array"
        assert (
            isinstance(self.gt_trajs_3d, np.ndarray)
            and self.gt_trajs_3d.dtype == np.float32
        ), "gt_trajs_3d must be a float32 numpy array"
        assert len(self.gt_intrinsics.shape) == 3 and self.gt_intrinsics.shape[-2:] == (
            3,
            3,
        ), "the shape of gt_intrinsics must be (t, 3, 3)"
        assert (
            len(self.gt_depths.shape) == 3
        ), "the shape of gt_depths must be (t, h, w)"
        if self.gt_query_point is not None:
            assert (
                isinstance(self.gt_query_point, np.ndarray)
                and self.gt_query_point.dtype == np.float32
            ), "gt_query_point must be a float32 numpy array"

        # est
        assert (
            isinstance(self.est_depths, np.ndarray)
            and self.est_depths.dtype == np.float32
        ), "est_depths must be a float32 numpy array"
        assert (
            isinstance(self.est_intrinsics, np.ndarray)
            and self.est_intrinsics.dtype == np.float32
        ), "est_intrinsics must be a float32 numpy array"
        assert (
            isinstance(self.est_extrinsics, np.ndarray)
            and self.est_extrinsics.dtype == np.float32
        ), "est_extrinsics must be a float32 numpy array"
        assert (
            isinstance(self.est_trajs_3d, np.ndarray)
            and self.est_trajs_3d.dtype == np.float32
        ), "est_trajs_3d must be a float32 numpy array"
        assert len(self.est_intrinsics.shape) == 3 and self.est_intrinsics.shape[
            -2:
        ] == (3, 3), "the shape of est_intrinsics must be (t, 3, 3)"
        assert (
            len(self.est_depths.shape) == 3
        ), "the shape of est_depths must be (t, h, w)"
        if self.est_query_point is not None:
            assert (
                isinstance(self.est_query_point, np.ndarray)
                and self.est_query_point.dtype == np.float32
            ), "est_query_point must be a float32 numpy array"

        assert id(self.gt_extrinsics) != id(
            self.est_extrinsics
        ), "gt_extrinsics and est_extrinsics must be different objects"
        assert id(self.gt_intrinsics) != id(
            self.est_intrinsics
        ), "gt_intrinsics and est_intrinsics must be different objects"
        assert id(self.gt_trajs_3d) != id(
            self.est_trajs_3d
        ), "gt_trajs_3d and est_trajs_3d must be different objects"
        assert id(self.gt_depths) != id(
            self.est_depths
        ), "gt_depths and est_depths must be different objects"
        if self.gt_query_point is not None and self.est_query_point is not None:
            assert id(self.gt_query_point) != id(
                self.est_query_point
            ), "gt_query_point and est_query_point must be different objects"

        assert (
            self.rgbs.shape[:3] == self.est_depths.shape[:3]
        ), "rgbs and est_depths must have the same size"
        assert (
            self.est_depths.shape == self.gt_depths.shape
        ), "est_depths and gt_depths must have the same shape"

        assert not (self.visibs & ~self.valids).any(), "visible points should be valid"

        if self.gt_depth_roi is not None:
            assert (
                isinstance(self.gt_depth_roi, np.ndarray)
                and self.gt_depth_roi.dtype == np.float32
            ), "gt_depth_roi must be a float32 numpy array"
            assert self.gt_depth_roi.shape == (
                2,
            ), "the shape of gt_depth_roi must be (2,)"
        if self.est_depth_roi is not None:
            assert (
                isinstance(self.est_depth_roi, np.ndarray)
                and self.est_depth_roi.dtype == np.float32
            ), "est_depth_roi must be a float32 numpy array"
            assert self.est_depth_roi.shape == (
                2,
            ), "the shape of est_depth_roi must be (2,)"

    def copy(self) -> "RawSliceData":
        return RawSliceData(**self.__dict__)


@dataclasses.dataclass
class SliceData:
    seq_name: Union[str, List[str]]
    seq_id: torch.Tensor  # (b)
    sample_id: torch.Tensor  # (b)
    visibs: torch.Tensor  # (t, n)
    valids: torch.Tensor  # (t, n)
    rgbs: torch.Tensor  # (t, 3, h, w)
    orig_resolution: torch.Tensor  # (2)

    gt_query_point: torch.Tensor  # (n, 3)
    gt_intrinsics: torch.Tensor  # (t, 3, 3)
    gt_extrinsics: torch.Tensor  # (t, 4, 4)
    gt_trajs_3d: torch.Tensor  # (t, n, 3)
    gt_depths: torch.Tensor  # (t, h, w)

    est_query_point: torch.Tensor  # (n, 3)
    est_intrinsics: torch.Tensor  # (t, 3, 3)
    est_extrinsics: torch.Tensor  # (t, 4, 4)
    est_trajs_3d: torch.Tensor  # (t, n, 3)
    est_depths: torch.Tensor  # (t, h, w)
    dataset_name: str = ""

    est_depths_downsampled: Optional[torch.Tensor] = None
    est_intrinsics_downsampled: Optional[torch.Tensor] = None  # (t, 3, 3)
    gt_intrinsics_downsampled: Optional[torch.Tensor] = None  # (t, 3, 3)
    gt_depths_downsampled: Optional[torch.Tensor] = None
    same_scale: Optional[bool] = None
    pd_to_gt_scale: Optional[torch.Tensor] = None  # (b)
    gt_norm_factor: Optional[torch.Tensor] = None  # (b)
    est_norm_factor: Optional[torch.Tensor] = None  # (b)
    # Per-sample metric (pre-normalization) motion threshold used to define
    # static/dynamic GT for this dataset. Mirrors the dataset's
    # ``dynamic_filter_threshold`` so the query dynamic label can be computed in
    # metric units (via gt_norm_factor) consistently with the dense label.
    dynamic_filter_threshold: Optional[torch.Tensor] = None  # (b)
    pd_to_gt_scale_bg_masked: Optional[torch.Tensor] = None  # (b)
    bg_mask: Optional[torch.Tensor] = None  # (t, h, w)
    bg_mask_downsampled: Optional[torch.Tensor] = None  # (t, h, w)
    query_indices: Optional[torch.Tensor] = None  # (b, t, n)
    dense_trajs_3d: Optional[torch.Tensor] = None  # (t, t, 3, h, w)
    est_dense_trajs_3d: Optional[torch.Tensor] = None  # (t, t, 3, h, w)
    dense_valids: Optional[torch.Tensor] = None  # (t, t, h, w)
    dense_visibs: Optional[torch.Tensor] = None  # (t, t, h, w)
    dense_dynamic: Optional[torch.Tensor] = None  # (t, h, w) or (t, h, w, k)
    is_dynamic: Optional[torch.Tensor] = None  # (t)
    gt_trajs_2d: Optional[torch.Tensor] = None  # (t,n,2)
    est_depth_roi: Optional[torch.Tensor] = None  # (2,)
    gt_depth_roi: Optional[torch.Tensor] = None  # (2,)
    annot_mode: Literal["gt", "est", None] = None
    segmentation: Optional[torch.Tensor] = None  # (t, h, w)
    gt_query_point_2d: Optional[torch.Tensor] = None  # (n, 2)
    dense_norm_mask: Optional[torch.Tensor] = None  # (t, h, w)
    traj_norm_mask: Optional[torch.Tensor] = None  # (t, n)
    tgt_time_step_scenes: Optional[torch.Tensor] = None  # (t)
    # Subsampled dense grid cells tracked as full trajectories (grid-anchored GT).
    dense_sample_points: Optional[torch.Tensor] = None  # (m, 3) int [src_frame, y, x] at grid res
    dense_sample_trajs_3d: Optional[torch.Tensor] = None  # (t, m, 3)
    dense_sample_visibs: Optional[torch.Tensor] = None  # (t, m)
    dense_sample_valids: Optional[torch.Tensor] = None  # (t, m)
    dense_sample_dynamic: Optional[torch.Tensor] = None  # (m,)
    # Cameras per timestep. When > 1 the time axis of every (t, ...) field above
    # holds num_views images per timestep, in time-major order
    # (t = frame * num_views + view). 1 or None means single-camera.
    num_views: Optional[int] = None
    flags: List[str] = dataclasses.field(default_factory=list)
    skip_post_init: bool = False

    @torch.no_grad()
    @profile
    def __post_init__(self):
        if self.skip_post_init:
            return
        assert isinstance(
            self.seq_name, (str, list)
        ), "seq_name must be a string or a list of strings"
        assert (
            isinstance(self.seq_id, torch.Tensor) and self.seq_id.dtype == torch.int32
        ), "seq_id must be a int32 tensor"
        assert (
            isinstance(self.sample_id, torch.Tensor)
            and self.sample_id.dtype == torch.int32
        ), "sample_id must be a int32 tensor"
        assert (
            isinstance(self.visibs, torch.Tensor) and self.visibs.dtype == torch.bool
        ), "visibs must be a boolean tensor"
        assert (
            isinstance(self.valids, torch.Tensor) and self.valids.dtype == torch.bool
        ), "valids must be a boolean tensor"
        assert (
            isinstance(self.rgbs, torch.Tensor) and self.rgbs.dtype == torch.float32
        ), "rgbs must be a float32 tensor"
        assert (
            isinstance(self.orig_resolution, torch.Tensor)
            and self.orig_resolution.dtype == torch.int32
        ), "orig_resolution must be a int32 tensor"

        if len(self.rgbs.shape) == 4:
            assert (
                self.rgbs.shape[1] == 3 and len(self.rgbs.shape) == 4
            ), "the shape of rgbs must be (t, 3, h, w)"
        else:
            assert (
                self.rgbs.shape[2] == 3 and len(self.rgbs.shape) == 5
            ), "the shape of rgbs must be (b, t, 3, h, w)"
        assert (
            self.rgbs.max() <= 1.0 + EPS and self.rgbs.min() >= -EPS
        ), "rgbs must be in the range of [0, 1]"

        assert (
            isinstance(self.gt_trajs_3d, torch.Tensor)
            and self.gt_trajs_3d.dtype == torch.float32
        ), "gt_trajs_3d must be a float32 tensor"
        assert (
            isinstance(self.est_trajs_3d, torch.Tensor)
            and self.est_trajs_3d.dtype == torch.float32
        ), "est_trajs_3d must be a float32 tensor"

        assert (
            isinstance(self.gt_depths, torch.Tensor)
            and self.gt_depths.dtype == torch.float32
        ), "gt_depths must be a float32 tensor"
        assert (
            isinstance(self.est_depths, torch.Tensor)
            and self.est_depths.dtype == torch.float32
        ), "est_depths must be a float32 tensor"

        assert (
            isinstance(self.gt_intrinsics, torch.Tensor)
            and self.gt_intrinsics.dtype == torch.float32
        ), "gt_intrinsics must be a float32 tensor"
        assert (
            isinstance(self.gt_extrinsics, torch.Tensor)
            and self.gt_extrinsics.dtype == torch.float32
        ), "gt_extrinsics must be a float32 tensor"
        assert (
            isinstance(self.est_intrinsics, torch.Tensor)
            and self.est_intrinsics.dtype == torch.float32
        ), "est_intrinsics must be a float32 tensor"
        assert (
            isinstance(self.est_extrinsics, torch.Tensor)
            and self.est_extrinsics.dtype == torch.float32
        ), "est_extrinsics must be a float32 tensor"
        assert (
            isinstance(self.gt_query_point, torch.Tensor)
            and self.gt_query_point.dtype == torch.float32
        ), f"gt_query_point must be a float32 tensor but got {self.gt_query_point.dtype}"
        assert (
            isinstance(self.est_query_point, torch.Tensor)
            and self.est_query_point.dtype == torch.float32
        ), "est_query_point must be a float32 tensor but got {self.est_query_point.dtype}"

        if self.segmentation is not None:
            assert (
                isinstance(self.segmentation, torch.Tensor)
                and self.segmentation.dtype == torch.int32
            ), "segmentation must be a int32 tensor"

        assert torch.allclose(
            self.gt_query_point[..., 0], torch.round(self.gt_query_point[..., 0])
        ), "the first dimension of gt_query_point must be integer"

        assert torch.allclose(
            self.est_query_point[..., 0], torch.round(self.est_query_point[..., 0])
        ), "the first dimension of est_query_point must be integer"
        assert torch.allclose(
            self.gt_query_point[..., 0], self.est_query_point[..., 0]
        ), "gt_query_point and est_query_point must have the same query frame"

        # import ipdb; ipdb.set_trace()
        if not torch.isfinite(self.est_depths).all():
            import ipdb

            ipdb.set_trace()
            # import ipdb; ipdb.set_trace()
            # self.est_depths.zero_()
            raise Exception("est_depths is not finite")
            import ipdb

            ipdb.set_trace()
        assert torch.isfinite(self.gt_depths).all(), "gt_depths must be finite"
        assert torch.isfinite(self.est_depths).all(), "est_depths must be finite"
        assert self.gt_depths.min() >= 0.0, "gt_depths must be non-negative"
        assert self.est_depths.min() >= 0.0, "est_depths must be non-negative"

        assert (
            self.rgbs.shape[-2:] == self.est_depths.shape[-2:]
        ), "rgbs and est_depths must have the same size"
        assert (
            self.est_depths.shape == self.gt_depths.shape
        ), "est_depths and gt_depths must have the same shape"

        if self.gt_depth_roi is not None:
            assert (
                self.gt_depth_roi.shape[-1] == 2
            ), "the shape of gt_depth_roi must be (2,) or (b, 2)"
        if self.est_depth_roi is not None:
            assert (
                self.est_depth_roi.shape[-1] == 2
            ), "the shape of est_depth_roi must be (2,) or (b, 2)"

    @staticmethod
    def collate(batch: List["SliceData"]) -> "SliceData":
        kwargs = {}
        for field in dataclasses.fields(SliceData):
            if isinstance(getattr(batch[0], field.name), torch.Tensor):
                kwargs[field.name] = torch.stack(
                    [getattr(item, field.name) for item in batch]
                )
            elif field.name == "segmentation" and batch[0].segmentation is None:
                assert all(item.segmentation is None for item in batch)
                kwargs[field.name] = None
            elif field.name in [
                "annot_mode",
                "same_scale",
                "flags",
                "skip_post_init",
                "num_views",
            ]:
                assert all(
                    getattr(item, field.name) == batch[0].__dict__[field.name]
                    for item in batch
                ), "all items in the batch must have the same value for the field"
                kwargs[field.name] = getattr(batch[0], field.name)
            elif field.name == "gt_depth_roi" or field.name == "est_depth_roi":
                if all(getattr(item, field.name) is None for item in batch):
                    kwargs[field.name] = None
                else:
                    assert all(
                        getattr(item, field.name) is not None for item in batch
                    ), "all items in the batch must either be None or not None"
                    kwargs[field.name] = torch.stack(
                        [getattr(item, field.name) for item in batch]
                    )
            else:
                kwargs[field.name] = [getattr(item, field.name) for item in batch]
        return SliceData(**kwargs)

    def cuda(self) -> "SliceData":
        def _to_cuda(x):
            if isinstance(x, torch.Tensor):
                return x.cuda()
            return x

        return SliceData(**{k: _to_cuda(v) for k, v in self.__dict__.items()})  # type: ignore

    def to(
        self,
        device: Union[str, torch.device],
        non_blocking: bool = False,
    ) -> "SliceData":
        def _to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(device, non_blocking=non_blocking)
            return x

        return SliceData(**{k: _to_device(v) for k, v in self.__dict__.items()})  # type: ignore

    def copy(self) -> "SliceData":
        d = self.__dict__.copy()
        if "skip_post_init" in d:
            d["skip_post_init"] = True
        r = SliceData(**d)
        r.skip_post_init = self.skip_post_init
        return r

    # Fields whose time axis is axis 1 when batched (axis 0 otherwise).
    _TIME_AXIS_FIELDS = (
        "visibs",
        "valids",
        "rgbs",
        "gt_trajs_3d",
        "gt_trajs_2d",
        "est_trajs_3d",
        "gt_depths",
        "gt_depths_downsampled",
        "gt_intrinsics",
        "gt_intrinsics_downsampled",
        "gt_extrinsics",
        "est_depths",
        "est_depths_downsampled",
        "est_intrinsics",
        "est_intrinsics_downsampled",
        "est_extrinsics",
        "bg_mask",
        "bg_mask_downsampled",
        "query_indices",
        "traj_norm_mask",
        "dense_norm_mask",
        "segmentation",
        "dense_sample_trajs_3d",
        "dense_sample_visibs",
        "dense_sample_valids",
    )

    def select_camera(self, view: int, num_views: int) -> "SliceData":
        """Keep one camera's images from a multiview time axis.

        With ``num_views`` cameras the time axis holds ``num_views`` images per
        timestep in time-major order, so camera ``view`` is
        ``[view::num_views]``. Query times are image indices and are divided
        down to timestep indices, which is exact because ``EvalDataset`` authors
        every multiview query in camera 0. ``is_dynamic`` is per-track, not
        per-frame, so it is left alone.

        The result has ``num_views=1`` and is indistinguishable from a genuine
        single-camera sample.
        """
        if num_views == 1:
            return self.copy()

        batched = len(self.rgbs.shape) == 5
        T = self.rgbs.shape[1 if batched else 0]
        assert T % num_views == 0, f"time axis {T} not divisible by {num_views}"
        assert 0 <= view < num_views, f"view {view} out of range for {num_views}"

        for name in (
            "dense_trajs_3d",
            "est_dense_trajs_3d",
            "dense_valids",
            "dense_visibs",
            "dense_dynamic",
            "tgt_time_step_scenes",
            "dense_sample_points",
        ):
            v = getattr(self, name)
            # These index time twice, or index into time, so a single-axis
            # stride would silently corrupt them.
            assert v is None or (isinstance(v, list) and v[0] is None), (
                f"select_camera does not support {name}; it is set on this sample"
            )

        sel = slice(view, None, num_views)
        d = self.__dict__.copy()
        for name in self._TIME_AXIS_FIELDS:
            v = d.get(name)
            if not isinstance(v, torch.Tensor):
                continue
            d[name] = v[:, sel] if batched else v[sel]

        for name in ("gt_query_point", "est_query_point"):
            v = d.get(name)
            if not isinstance(v, torch.Tensor):
                continue
            v = v.clone()
            v[..., 0] = torch.div(v[..., 0], num_views, rounding_mode="floor")
            d[name] = v

        d["num_views"] = 1
        d["skip_post_init"] = True
        r = SliceData(**d)
        r.skip_post_init = self.skip_post_init
        return r

    def time_slice(self, start: int, end: int) -> "SliceData":
        if len(self.rgbs.shape) == 5:
            assert (
                start >= 0 and end <= self.rgbs.shape[1] and start < end
            ), "the range of start and end is out of bounds"
            _keep_dense_full = (
                self.tgt_time_step_scenes is not None
                and self.tgt_time_step_scenes[0] is not None
            )

            def _dense_trajs_time_slice(traj):
                if traj is None or traj[0] is None:
                    return None
                return traj if _keep_dense_full else traj[..., start:end, :]

            def _dense_mask_time_slice(traj):
                if traj is None or traj[0] is None:
                    return None
                return traj if _keep_dense_full else traj[..., start:end]

            return SliceData(
                seq_name=self.seq_name,
                seq_id=self.seq_id,
                sample_id=self.sample_id,
                gt_trajs_3d=self.gt_trajs_3d[:, start:end],
                gt_trajs_2d=self.gt_trajs_2d[:, start:end]
                if self.gt_trajs_2d[0] is not None
                else None,
                est_trajs_3d=self.est_trajs_3d[:, start:end],
                visibs=self.visibs[:, start:end],
                valids=self.valids[:, start:end],
                rgbs=self.rgbs[:, start:end],
                gt_depths=self.gt_depths[:, start:end],
                gt_depths_downsampled=self.gt_depths_downsampled[:, start:end]
                if self.gt_depths_downsampled[0] is not None
                else None,
                gt_intrinsics_downsampled=self.gt_intrinsics_downsampled[:, start:end]
                if self.gt_intrinsics_downsampled[0] is not None
                else None,
                est_depths=self.est_depths[:, start:end],
                est_depths_downsampled=self.est_depths_downsampled[:, start:end]
                if self.est_depths_downsampled[0] is not None
                else None,
                est_intrinsics_downsampled=self.est_intrinsics_downsampled[:, start:end]
                if self.est_intrinsics_downsampled[0] is not None
                else None,
                gt_query_point=self.gt_query_point,
                est_query_point=self.est_query_point,
                gt_intrinsics=self.gt_intrinsics[:, start:end],
                est_intrinsics=self.est_intrinsics[:, start:end],
                gt_extrinsics=self.gt_extrinsics[:, start:end],
                est_extrinsics=self.est_extrinsics[:, start:end],
                orig_resolution=self.orig_resolution,
                annot_mode=self.annot_mode,
                same_scale=self.same_scale,
                skip_post_init=self.skip_post_init,
                flags=self.flags,
                gt_depth_roi=self.gt_depth_roi,
                est_depth_roi=self.est_depth_roi,
                is_dynamic=self.is_dynamic if self.is_dynamic is not None else None,
                dense_trajs_3d=_dense_trajs_time_slice(self.dense_trajs_3d),
                est_dense_trajs_3d=_dense_trajs_time_slice(self.est_dense_trajs_3d),
                dense_valids=_dense_mask_time_slice(self.dense_valids),
                dense_visibs=_dense_mask_time_slice(self.dense_visibs),
                dense_dynamic=self.dense_dynamic
                if self.dense_dynamic[0] is not None
                else None,
                pd_to_gt_scale=self.pd_to_gt_scale,
                gt_norm_factor=self.gt_norm_factor,
                est_norm_factor=self.est_norm_factor,
                dynamic_filter_threshold=self.dynamic_filter_threshold,
                pd_to_gt_scale_bg_masked=self.pd_to_gt_scale_bg_masked,
                bg_mask=self.bg_mask[:, start:end]
                if self.bg_mask[0] is not None
                else None,
                bg_mask_downsampled=self.bg_mask_downsampled[:, start:end]
                if self.bg_mask_downsampled[0] is not None
                else None,
                query_indices=self.query_indices[:, start:end]
                if self.query_indices[0] is not None
                else None,
                traj_norm_mask=self.traj_norm_mask[:, start:end] if self.traj_norm_mask[0] is not None else None,
                dense_norm_mask=self.dense_norm_mask if self.dense_norm_mask[0] is not None else None,
                tgt_time_step_scenes=self.tgt_time_step_scenes if self.tgt_time_step_scenes[0] is not None else None,
                dense_sample_points=self.dense_sample_points if self.dense_sample_points[0] is not None else None,
                dense_sample_trajs_3d=self.dense_sample_trajs_3d[:, start:end] if self.dense_sample_trajs_3d[0] is not None else None,
                dense_sample_visibs=self.dense_sample_visibs[:, start:end] if self.dense_sample_visibs[0] is not None else None,
                dense_sample_valids=self.dense_sample_valids[:, start:end] if self.dense_sample_valids[0] is not None else None,
                dense_sample_dynamic=self.dense_sample_dynamic if self.dense_sample_dynamic[0] is not None else None,
            )
        elif len(self.rgbs.shape) == 4:
            assert (
                start >= 0 and end <= self.rgbs.shape[0] and start < end
            ), "the range of start and end is out of bounds"
            return SliceData(
                seq_name=self.seq_name,
                seq_id=self.seq_id,
                sample_id=self.sample_id,
                gt_trajs_3d=self.gt_trajs_3d[start:end],
                est_trajs_3d=self.est_trajs_3d[start:end],
                visibs=self.visibs[start:end],
                valids=self.valids[start:end],
                rgbs=self.rgbs[start:end],
                gt_depths=self.gt_depths[start:end],
                gt_depths_downsampled=self.gt_depths_downsampled[start:end],
                est_depths=self.est_depths[start:end],
                est_depths_downsampled=self.est_depths_downsampled[start:end],
                gt_query_point=self.gt_query_point,
                est_query_point=self.est_query_point,
                gt_intrinsics=self.gt_intrinsics[start:end],
                est_intrinsics=self.est_intrinsics[start:end],
                gt_extrinsics=self.gt_extrinsics[start:end],
                est_extrinsics=self.est_extrinsics[start:end],
                orig_resolution=self.orig_resolution,
                annot_mode=self.annot_mode,
                same_scale=self.same_scale,
                skip_post_init=self.skip_post_init,
                flags=self.flags,
                gt_depth_roi=self.gt_depth_roi,
                est_depth_roi=self.est_depth_roi,
                is_dynamic=self.is_dynamic if self.is_dynamic is not None else None,
                dense_trajs_3d=self.dense_trajs_3d
                if self.dense_trajs_3d[0] is not None
                else None,
                est_dense_trajs_3d=self.est_dense_trajs_3d[start:end]
                if self.est_dense_trajs_3d[0] is not None
                else None,
                dense_valids=self.dense_valids
                if self.dense_valids[0] is not None
                else None,
                dense_visibs=self.dense_visibs
                if self.dense_visibs[0] is not None
                else None,
                dense_dynamic=self.dense_dynamic
                if self.dense_dynamic[0] is not None
                else None,
                pd_to_gt_scale=self.pd_to_gt_scale,
                gt_norm_factor=self.gt_norm_factor,
                est_norm_factor=self.est_norm_factor,
                dynamic_filter_threshold=self.dynamic_filter_threshold,
                pd_to_gt_scale_bg_masked=self.pd_to_gt_scale_bg_masked,
                bg_mask=self.bg_mask[start:end]
                if self.bg_mask[0] is not None
                else None,
                bg_mask_downsampled=self.bg_mask_downsampled[:, start:end]
                if self.bg_mask_downsampled[0] is not None
                else None,
                query_indices=self.query_indices[start:end]
                if self.query_indices[0] is not None
                else None,
                traj_norm_mask=self.traj_norm_mask[:, start:end] if self.traj_norm_mask[0] is not None else None,
                dense_norm_mask=self.dense_norm_mask if self.dense_norm_mask[0] is not None else None,
            )
        else:
            raise ValueError(f"Unknown shape of rgbs: {self.rgbs.shape}")

    def with_annot_mode(self, annot_mode: Literal["gt", "est", None]) -> "SliceData":
        assert annot_mode in [
            "gt",
            "est",
            None,
        ], "annot_mode must be either 'gt', 'est', or None"
        ret = self.copy()
        ret.annot_mode = annot_mode
        return ret

    def __getattr__(self, name: str) -> Any:
        if not f"gt_{name}" in self.__dict__:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )
        if self.annot_mode is None:
            raise ValueError(
                f"You must call with_annot_mode(annot_mode) before accessing {name}"
            )
        if self.annot_mode == "gt":
            return getattr(self, f"gt_{name}")
        else:
            return getattr(self, f"est_{name}")

    def with_causal_mask(self) -> "SliceData":
        sample = self.copy()

        query_frame = sample.gt_query_point[..., 0].long()
        eye = torch.eye(
            sample.valids.shape[-2], dtype=query_frame.dtype, device=query_frame.device
        )
        query_frame_to_eval_frames = torch.cumsum(eye, dim=1)
        mask = rearrange(query_frame_to_eval_frames[query_frame] > 0, "b n t -> b t n")

        sample.valids = sample.valids & mask

        return sample
