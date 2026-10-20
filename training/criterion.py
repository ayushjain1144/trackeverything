# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Trajectory criterion for TrackEverything.

Drops geometry and flow auxiliary losses (the model does not predict geometry,
and the flow loss is a redundant reparametrization of the trajectory loss).

Losses kept:
- 3D trajectory/coords L1 (visible / invisible / all, with static-vs-dynamic
  reweighting), on the dense scene grid and on query points;
- 2D reprojection loss on the same;
- visibility BCE (dense + query);
- static/dynamic classification BCE (dense + query, with static-vs-dynamic
  reweighting);
- confidence regression loss (kept although not described in the paper);
- query coordinate-at-target loss.
"""

import os
from typing import Any, Dict, Tuple

import ipdb
import torch

from einops import rearrange
from line_profiler import profile
from torch.nn import functional as F

from ..datasets.datatypes import SliceData
from ..datasets.utils.geometry import apply_log_to_norm, project_trajs_3d_to_2d

from ..utils.common_utils import ensure_float32
from .datatypes import TrainData
from .vggt_losses import regression_loss_flexible

st = ipdb.set_trace


# UV bounds in normalized coords (same as calc_loss_2d): [-1, 2] => pixel u in [-W, 2*W], v in [-H, 2*H]
UV_NORM_LOW, UV_NORM_HIGH = -1.0, 2.0


def _world_trajs_to_uvd_with_valid(
    pd_traj,
    gt_traj,
    extrinsics,
    intrinsics,
    valids,
    depth_min,
    depth_max,
    image_size=None,
):
    """Convert world trajectories to UV (pixel) + inverse depth and compute valid mask.

    Only supervise where GT is in-frame and in front of camera (gt_depth > 0).
    """
    pd_2d, pd_depth = project_trajs_3d_to_2d(pd_traj, extrinsics, intrinsics)
    gt_2d, gt_depth = project_trajs_3d_to_2d(gt_traj, extrinsics, intrinsics)
    pd_uv = pd_2d[..., :2]
    gt_uv = gt_2d[..., :2]

    pd_inv_depth = 1.0 / torch.clamp(pd_depth, min=depth_min, max=depth_max)
    gt_inv_depth = 1.0 / torch.clamp(gt_depth, min=depth_min, max=depth_max)

    depth_ok = gt_depth > 0
    uvd_valid = depth_ok & valids

    if image_size is not None:
        h, w = image_size[0], image_size[1]
        uv_in_bounds = (
            (gt_uv[..., 0] >= UV_NORM_LOW * w) & (gt_uv[..., 0] <= UV_NORM_HIGH * w)
            & (gt_uv[..., 1] >= UV_NORM_LOW * h) & (gt_uv[..., 1] <= UV_NORM_HIGH * h)
        )
        uvd_valid = uvd_valid & uv_in_bounds

    return pd_uv, gt_uv, pd_inv_depth, gt_inv_depth, uvd_valid


class AllTrajectoryCriterion:
    def __init__(
        self,
        gamma: float,
        bce_weight: float,
        train_iters: int,
        eval_iters: int,
        traj_weight: float = 0.0,
        conf_weight: float = 0.0,
        loss_type: str = "l21",
        traj_visible_weight: float = 0.0,
        traj_invisible_weight: float = 0.0,
        depth_min: float = 0.1,
        depth_max: float = 100.0,
        dynamic_bce_weight: float = 0.0,
        log_loss: bool = False,
        compute_2d_loss: bool = False,
        compute_3d_loss: bool = True,
        traj_2d_weight: float = 0.0,
        topk_loss_fraction: float = 1.0,
        use_conf_regression_loss: bool = False,
        conf_regression_weight: float = 1.0,
        conf_regression_gamma: float = 1.0,
        conf_regression_alpha: float = 0.2,
        conf_regression_gradient_loss_fn: str = "normal",
        conf_regression_valid_range: float = 0.98,
        dynamic_loss_weight: float = 1.0,
        query_loss_weight: float = 1.0,
        evaluate_query_at_tgt: bool = False,
        query_coord_weight: float = 1.0,
        dynamic_filter_threshold: float = 0.01,
        skip_dense_loss: bool = False,
    ):
        self.gamma = gamma
        self.traj_weight = traj_weight
        self.traj_2d_weight = traj_2d_weight
        self.traj_visible_weight = traj_visible_weight
        self.traj_invisible_weight = traj_invisible_weight
        self.bce_weight = bce_weight
        self.train_iters = train_iters
        self.eval_iters = eval_iters
        self.loss_type = loss_type
        self.conf_weight = conf_weight
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.dynamic_bce_weight = dynamic_bce_weight
        self.log_loss = log_loss
        self.compute_2d_loss = compute_2d_loss
        self.compute_3d_loss = compute_3d_loss
        self.topk_loss_fraction = topk_loss_fraction
        self.use_conf_regression_loss = use_conf_regression_loss
        self.conf_regression_weight = conf_regression_weight
        self.conf_regression_gamma = conf_regression_gamma
        self.conf_regression_alpha = conf_regression_alpha
        self.conf_regression_gradient_loss_fn = conf_regression_gradient_loss_fn
        self.conf_regression_valid_range = conf_regression_valid_range
        self.dynamic_loss_weight = dynamic_loss_weight
        self.query_loss_weight = query_loss_weight
        self.evaluate_query_at_tgt = evaluate_query_at_tgt
        self.query_coord_weight = query_coord_weight
        self.dynamic_filter_threshold = dynamic_filter_threshold
        self.skip_dense_loss = skip_dense_loss

    @staticmethod
    def _align_valids_to_window(valids, trajs_3d, window_trajs, start, end):
        """Return ``valids`` sliced to the window, whatever time extent it has."""
        if valids.shape[1] == trajs_3d.shape[1]:
            valids = valids[:, start:end]
        elif valids.shape[1] != window_trajs.shape[1]:
            raise ValueError(
                f"valids time extent {valids.shape[1]} matches neither the full "
                f"trajectory ({trajs_3d.shape[1]}) nor the window "
                f"({window_trajs.shape[1]})"
            )
        if valids.shape[-1] != window_trajs.shape[2]:
            raise ValueError(
                f"valids has {valids.shape[-1]} points, trajs have "
                f"{window_trajs.shape[2]}"
            )
        return valids.bool()

    def _compute_window_dynamic_from_trajs(
        self, trajs_3d, frame_range, norm_factor=None, threshold=None, valids=None
    ):
        """Static/dynamic GT for the window: the L2 extent of each trajectory.

        ``valids`` (B, T, N) restricts the extent to frames where the point is
        actually valid. Coordinates on invalid frames are not meaningful, and one
        junk value stretches the extent enough to label a stationary point
        dynamic. Mirrors datasets/utils/dynamic_labels.dynamic_extent_l2.

        ``trajs_3d`` is the FULL-length trajectory (captured before the sample is
        time-sliced) and is sliced here by ``frame_range``. ``valids`` may be
        either full-length or already sliced to the window, so its time extent is
        checked rather than assumed.
        """
        start, end = frame_range
        window_trajs = trajs_3d[:, start:end]
        if window_trajs.shape[1] <= 1:
            return torch.zeros(
                trajs_3d.shape[0],
                trajs_3d.shape[2],
                device=trajs_3d.device,
                dtype=torch.bool,
            )
        if valids is None:
            max_xyz = torch.amax(window_trajs, dim=1)
            min_xyz = torch.amin(window_trajs, dim=1)
            diff_l2 = torch.linalg.norm(max_xyz - min_xyz, dim=-1)  # (B, N)
            all_valid_pts = None
        else:
            w_valid = self._align_valids_to_window(
                valids, trajs_3d, window_trajs, start, end
            )
            all_valid_pts = w_valid.any(dim=1)
            # Points with no valid frame at all: keep every frame so the
            # reduction stays finite, then force them static below.
            keep = (w_valid | (~all_valid_pts)[:, None, :])[..., None]
            big = torch.finfo(window_trajs.dtype).max
            max_xyz = torch.where(keep, window_trajs, -big).amax(dim=1)
            min_xyz = torch.where(keep, window_trajs, big).amin(dim=1)
            diff_l2 = torch.linalg.norm(max_xyz - min_xyz, dim=-1)  # (B, N)
            diff_l2 = torch.where(all_valid_pts, diff_l2, torch.zeros_like(diff_l2))
        if threshold is None:
            # Legacy fallback: threshold the (normalized) motion directly.
            return diff_l2 > self.dynamic_filter_threshold
        # trajs_3d are scene-normalized (divided by gt_norm_factor). Rescale the
        # motion back to metric (pre-normalization) units so the threshold matches
        # the dense GT, which is thresholded per-dataset in metric space.
        if norm_factor is not None:
            diff_l2 = diff_l2 * norm_factor.reshape(-1, 1).to(diff_l2)
        return diff_l2 > threshold.reshape(-1, 1).to(diff_l2)

    @profile
    def _traj_loss_func(self, gt, pd):
        if self.loss_type == "l21":
            loss_3d = (pd - gt).norm(dim=-1)
        elif self.loss_type == "l1":
            loss_3d = (pd - gt).abs().sum(dim=-1)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        return loss_3d

    def _topk_masked_mean(self, loss, mask, no_topk=False):
        """Mean of the top-k fraction of loss values within mask."""
        if self.topk_loss_fraction >= 1.0 or no_topk:
            return (loss * mask.float()).sum() / (mask.sum() + 1e-4)
        masked_loss = loss[mask]
        if masked_loss.numel() == 0:
            return loss.new_zeros(())
        k = max(1, int(masked_loss.numel() * self.topk_loss_fraction))
        topk_vals, _ = torch.topk(masked_loss, k)
        return topk_vals.mean()

    def calc_loss_2d(
        self, pd_traj, gt_traj, gt_sample,
        visible_mask, invisible_mask, valids,
        num_iters, iter_idx,
        static_mask=None, dynamic_mask=None,
        extrinsics=None, intrinsics=None,
        image_size=None,
    ):
        """2D trajectory loss using same valid mask as UVD (in-frame, depth in range)."""
        if intrinsics is None or extrinsics is None:
            intrinsics = gt_sample.intrinsics
            extrinsics = gt_sample.extrinsics
        assert image_size is not None, "calc_loss_2d requires image_size"
        assert pd_traj.shape[-1] == 3 and gt_traj.shape[-1] == 3, "calc_loss_2d expects 3D world trajs"

        pd_uv, gt_uv, _, _, uvd_valid = _world_trajs_to_uvd_with_valid(
            pd_traj, gt_traj, extrinsics, intrinsics, valids,
            self.depth_min, self.depth_max, image_size=image_size,
        )
        pd_uv_norm = pd_uv.clone()
        pd_uv_norm[..., 0] /= image_size[1]
        pd_uv_norm[..., 1] /= image_size[0]
        pd_uv_norm = pd_uv_norm.clamp(UV_NORM_LOW, UV_NORM_HIGH)
        gt_uv_norm = gt_uv.clone()
        gt_uv_norm[..., 0] /= image_size[1]
        gt_uv_norm[..., 1] /= image_size[0]

        if self.topk_loss_fraction < 1.0:
            pd_sel = pd_uv_norm[uvd_valid]
            gt_sel = gt_uv_norm[uvd_valid]
            loss_2d_sel = self._traj_loss_func(gt=gt_sel, pd=pd_sel)
            if loss_2d_sel.numel() > 0:
                k = max(1, int(loss_2d_sel.numel() * self.topk_loss_fraction))
                topk_vals, _ = torch.topk(loss_2d_sel, k)
                loss_2d_all = topk_vals.mean()
            else:
                loss_2d_all = pd_uv_norm.new_zeros(())
            loss_2d = self._traj_loss_func(gt=gt_uv_norm, pd=pd_uv_norm)
        else:
            loss_2d = self._traj_loss_func(gt=gt_uv_norm, pd=pd_uv_norm)
            valid_f = uvd_valid.float()
            loss_2d_all = (loss_2d * valid_f).sum() / valid_f.sum().clamp(min=1)

        if dynamic_mask is not None:
            dynamic_valid = dynamic_mask & uvd_valid
            static_valid = static_mask & uvd_valid
            loss_2d_dynamic = self._topk_masked_mean(loss_2d, dynamic_valid)
            loss_2d_static = self._topk_masked_mean(loss_2d, static_valid)
            loss_2d_all = loss_2d_static + loss_2d_dynamic * self.dynamic_loss_weight
        else:
            loss_2d_dynamic = loss_2d_static = None

        ret = {
            "2d_all": loss_2d_all,
            "2d_dynamic": loss_2d_dynamic if dynamic_mask is not None else None,
            "2d_static": loss_2d_static if dynamic_mask is not None else None,
        }
        gamma_weight = self.gamma ** (num_iters - 1 - iter_idx)
        for k, v in ret.items():
            if ret[k] is not None:
                ret[k] = gamma_weight * v
        return ret

    def calc_loss(
        self, pd_traj, gt_traj, gt_sample,
        visible_mask, invisible_mask, valids,
        num_iters, iter_idx,
        static_mask=None, dynamic_mask=None,
        extrinsics=None, intrinsics=None,
        traj_norm_mask=None,
        no_topk=False,
        return_per_point_loss=False,
    ):
        loss_3d = self._traj_loss_func(gt=gt_traj, pd=pd_traj)
        if traj_norm_mask is not None:
            valids = valids & traj_norm_mask
            visible_mask = visible_mask & valids
            invisible_mask = invisible_mask & valids

        loss_3d_visible = self._topk_masked_mean(loss_3d, visible_mask, no_topk=no_topk)
        loss_3d_invisible = self._topk_masked_mean(loss_3d, invisible_mask, no_topk=no_topk)
        loss_3d_all = self._topk_masked_mean(loss_3d, valids, no_topk=no_topk)

        if dynamic_mask is not None:
            loss_3d_dynamic = self._topk_masked_mean(loss_3d, dynamic_mask, no_topk=no_topk)
            loss_3d_static = self._topk_masked_mean(loss_3d, static_mask, no_topk=no_topk)
            # keeps the loss from being dominated by the (many) static points
            loss_3d_all = loss_3d_static + loss_3d_dynamic * self.dynamic_loss_weight

        ret = {
            "3d_visible": loss_3d_visible,
            "3d_invisible": loss_3d_invisible,
            "3d_all": loss_3d_all,
            "3d_dynamic": loss_3d_dynamic if dynamic_mask is not None else None,
            "3d_static": loss_3d_static if dynamic_mask is not None else None,
        }

        gamma_weight = self.gamma ** (num_iters - 1 - iter_idx)
        for k, v in ret.items():
            if ret[k] is not None:
                ret[k] = gamma_weight * v

        if return_per_point_loss:
            ret["per_point_loss"] = loss_3d.detach()
            ret["per_point_valid"] = valids.detach()

        return ret

    @profile
    def _compute_bce_loss(
        self, gt_visibs, pred_visibs, iter_idx, num_iters: int, gt_valids=None
    ) -> torch.Tensor:
        if gt_valids is None:
            traj_mask = torch.ones_like(gt_visibs)
        else:
            traj_mask = gt_valids

        gamma_weight = self.gamma ** (num_iters - 1 - iter_idx)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_visibs, gt_visibs.float(), reduction="none"
        )
        bce_loss = (bce_loss * traj_mask.float()).sum() / (traj_mask.sum() + 1e-4)
        return gamma_weight * bce_loss

    def _compute_balanced_dynamic_bce(
        self, pred_logits, gt_dynamic, valid, iter_idx, num_iters: int
    ) -> torch.Tensor:
        """Static/dynamic BCE with per-class means so static points don't dominate."""
        gamma_weight = self.gamma ** (num_iters - 1 - iter_idx)
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, gt_dynamic.float(), reduction="none"
        )
        dynamic_mask = valid & (gt_dynamic > 0.5)
        static_mask = valid & (gt_dynamic <= 0.5)
        loss_dynamic = (bce * dynamic_mask.float()).sum() / (dynamic_mask.sum() + 1e-4)
        loss_static = (bce * static_mask.float()).sum() / (static_mask.sum() + 1e-4)
        return gamma_weight * (loss_static + loss_dynamic * self.dynamic_loss_weight)

    @ensure_float32(allow_cast=False)
    @profile
    def compute_loss(
        self,
        sample: SliceData,
        train_data: TrainData,
        bidirectional: bool,
        is_train: bool,
        return_per_point_loss: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = sample.copy().with_annot_mode(None)
        extrinsics = sample.gt_extrinsics
        intrinsics = sample.gt_intrinsics
        orig_gt_trajs_3d = sample.gt_trajs_3d.clone()

        sample = sample.time_slice(train_data.frame_range[0], train_data.frame_range[1])

        num_iters = self.train_iters if is_train else self.eval_iters

        gt_sample = sample.with_annot_mode("gt")

        tgt_scenes_cpu = (
            gt_sample.tgt_time_step_scenes.squeeze(0).tolist()
            if gt_sample.tgt_time_step_scenes is not None
            else None
        )

        traj_loss_dict = {}
        bce_loss = torch.zeros((), device=extrinsics.device)
        query_coord_loss = None
        query_coord_loss_2d = None

        # on the dense grid points, we only care to supervise on the tgt_time_step_scene
        skip_dense_eval = (
            not is_train
            and tgt_scenes_cpu is not None
            and (
                train_data.tgt_time_step_scene is None
                or gt_sample.dense_trajs_3d is None
                or train_data.tgt_time_step_scene not in tgt_scenes_cpu
                or gt_sample.dense_trajs_3d.shape[-2] < len(tgt_scenes_cpu)
            )
        )

        if gt_sample.dense_trajs_3d is not None and not skip_dense_eval and not self.skip_dense_loss:
            gt_dense_trajs_3d = gt_sample.dense_trajs_3d
            gt_dense_trajs_valid = gt_sample.dense_valids
            gt_dense_visibs = gt_sample.dense_visibs
            gt_dense_dynamic = gt_sample.dense_dynamic

            tgt_time_step_scene = train_data.tgt_time_step_scene
            assert tgt_time_step_scene is not None, "tgt_time_step_scene is not set"
            if tgt_scenes_cpu is not None:
                tgt_key = (
                    int(tgt_time_step_scene)
                    if isinstance(tgt_time_step_scene, torch.Tensor)
                    else tgt_time_step_scene
                )
                if os.environ.get("TAPIP3D_LOCAL_DEBUG", "0") == "1":
                    if tgt_key not in tgt_scenes_cpu:
                        st()
                    assert tgt_key in tgt_scenes_cpu, "tgt_time_step_scene is not in gt_sample.tgt_time_step_scenes"
                    assert gt_sample.tgt_time_step_scenes.shape[0] == 1, "only one batch supported for now"
                index = tgt_scenes_cpu.index(tgt_key)
            else:
                index = tgt_time_step_scene

            gt_dense_trajs_3d = gt_dense_trajs_3d[..., index, :]
            gt_dense_trajs_valid = gt_dense_trajs_valid[..., index]
            gt_dense_visibs = gt_dense_visibs[..., index]
            if gt_dense_dynamic.dim() == gt_dense_trajs_valid.dim() + 1:
                gt_dense_dynamic = gt_dense_dynamic[..., index]

            pred_trajs = train_data.coords[0]

            window_size = pred_trajs.shape[1]

            gt_dense_trajs_3d = gt_dense_trajs_3d[:, :window_size]
            gt_dense_trajs_valid = gt_dense_trajs_valid[:, :window_size]
            gt_dense_visibs = gt_dense_visibs[:, :window_size]
            gt_dense_dynamic = gt_dense_dynamic[:, :window_size]
            extrinsics = extrinsics[:, :window_size]
            intrinsics = intrinsics[:, :window_size]
            if gt_sample.dense_norm_mask is not None:
                dense_norm_mask = gt_sample.dense_norm_mask[:, :window_size]
            else:
                dense_norm_mask = None

            if train_data.sampled_mask is not None:
                sm = train_data.sampled_mask[:, :window_size]
                gt_dense_trajs_valid = gt_dense_trajs_valid & sm

            if self.log_loss:
                gt_dense_trajs_3d_log = apply_log_to_norm(gt_dense_trajs_3d)
                pred_trajs_log = apply_log_to_norm(pred_trajs)

            visible_mask = gt_dense_visibs & gt_dense_trajs_valid
            invisible_mask = ~gt_dense_visibs & gt_dense_trajs_valid
            dynamic_mask = gt_dense_dynamic & gt_dense_trajs_valid
            static_mask = ~gt_dense_dynamic & gt_dense_trajs_valid

            use_log_traj = self.log_loss
            pred_for_traj = pred_trajs_log if use_log_traj else pred_trajs
            gt_for_traj = gt_dense_trajs_3d_log if use_log_traj else gt_dense_trajs_3d
            dense_valids = gt_dense_trajs_valid
            dense_traj_norm_mask = dense_norm_mask.squeeze(-1) if dense_norm_mask is not None else None

            if self.use_conf_regression_loss:
                gamma_weight = self.gamma ** (num_iters - 1 - train_data.iter_idx)
                valid_mask = dense_valids & dense_traj_norm_mask if dense_traj_norm_mask is not None else dense_valids

                dense_conf_loss, dense_grad_loss, dense_reg_loss = regression_loss_flexible(
                    pred_for_traj,
                    gt_for_traj,
                    valid_mask,
                    conf=train_data.confs,
                    gradient_loss_fn=self.conf_regression_gradient_loss_fn,
                    gamma=self.conf_regression_gamma,
                    alpha=self.conf_regression_alpha,
                    valid_range=self.conf_regression_valid_range,
                    dynamic_weight=self.dynamic_loss_weight,
                    dynamic_mask=dynamic_mask,
                )
                traj_loss_dict = {
                    "3d_all": gamma_weight * (dense_conf_loss + dense_grad_loss + dense_reg_loss),
                    "3d_visible": torch.zeros((), device=pred_for_traj.device),
                    "3d_invisible": torch.zeros((), device=pred_for_traj.device),
                    "_conf_dense_conf_loss": dense_conf_loss.detach(),
                    "_conf_dense_grad_loss": dense_grad_loss.detach() if isinstance(dense_grad_loss, torch.Tensor) else torch.tensor(dense_grad_loss, device=pred_for_traj.device),
                    "_conf_dense_reg_loss": dense_reg_loss.detach(),
                }
                if return_per_point_loss:
                    pp_reg = torch.norm(gt_for_traj - pred_for_traj, dim=-1)
                    if train_data.confs is not None:
                        pp_conf = self.conf_regression_gamma * pp_reg * train_data.confs \
                            - self.conf_regression_alpha * torch.log(train_data.confs.clamp(min=1e-8))
                        pp_combined = pp_conf + pp_reg
                    else:
                        pp_combined = pp_reg
                    traj_loss_dict["per_point_loss"] = (pp_combined * valid_mask.float()).detach()
                    traj_loss_dict["per_point_valid"] = valid_mask.detach()
            else:
                traj_loss_dict = self.calc_loss(
                    pred_for_traj,
                    gt_for_traj,
                    gt_sample,
                    visible_mask,
                    invisible_mask,
                    dense_valids,
                    num_iters,
                    train_data.iter_idx,
                    static_mask,
                    dynamic_mask,
                    extrinsics,
                    intrinsics,
                    traj_norm_mask=dense_traj_norm_mask,
                    return_per_point_loss=return_per_point_loss,
                )

            if self.compute_2d_loss:
                image_size = gt_sample.rgbs.shape[-2:]
                # Model may leave tgt_time_step_scene as float (e.g. under autocast);
                # advanced indexing requires integer/bool indices.
                tgt_idx = (
                    tgt_time_step_scene.long()
                    if torch.is_tensor(tgt_time_step_scene)
                    else tgt_time_step_scene
                )
                extrinsics_ = extrinsics[:, tgt_idx, None, None, None]
                intrinsics_ = intrinsics[:, tgt_idx, None, None, None]

                if dense_norm_mask is not None:
                    gt_dense_trajs_valid = gt_dense_trajs_valid & dense_norm_mask.squeeze(-1)
                loss_2d_dict = self.calc_loss_2d(
                    pred_trajs,
                    gt_dense_trajs_3d,
                    gt_sample,
                    visible_mask,
                    invisible_mask,
                    gt_dense_trajs_valid,
                    num_iters,
                    train_data.iter_idx,
                    static_mask,
                    dynamic_mask,
                    extrinsics_,
                    intrinsics_,
                    image_size=image_size
                )
                if traj_loss_dict is None:
                    traj_loss_dict = {}
                traj_loss_dict.update(loss_2d_dict)

            if dense_norm_mask is not None:
                valids = gt_dense_trajs_valid & dense_norm_mask.squeeze(-1)
            else:
                valids = gt_dense_trajs_valid
            if train_data.sampled_mask is not None:
                valids = valids & train_data.sampled_mask[:, :window_size]
            bce_loss = self._compute_bce_loss(
                gt_valids=valids,
                gt_visibs=gt_dense_visibs,
                pred_visibs=train_data.visibs,
                iter_idx=train_data.iter_idx,
                num_iters=num_iters,
            )

            if train_data.dense_dynamic_logits is not None and self.dynamic_bce_weight > 0:
                gt_dynamic_target = gt_dense_dynamic[:, :window_size].float()
                pred_dynamic_logits = train_data.dense_dynamic_logits[:, :window_size]
                valid = gt_dense_trajs_valid[:, :window_size]
                if train_data.sampled_mask is not None:
                    valid = valid & train_data.sampled_mask[:, :window_size]

                dense_dynamic_bce = self._compute_balanced_dynamic_bce(
                    pred_dynamic_logits,
                    gt_dynamic_target,
                    valid,
                    train_data.iter_idx,
                    num_iters,
                )
                traj_loss_dict["dense_dynamic_bce"] = dense_dynamic_bce

        # also calculate loss on query coords
        if train_data.query_coords is not None:
            pred_trajs = train_data.query_coords[0]
            gt_trajs = gt_sample.trajs_3d

            if self.log_loss:
                gt_trajs_log = apply_log_to_norm(gt_trajs)
                pred_trajs_log = apply_log_to_norm(pred_trajs)

            # mask out queries which are not seen yet
            valids = gt_sample.valids.clone()
            mask = gt_sample.gt_query_point[..., 0] >= train_data.frame_range[1]
            valids[:, :, mask.squeeze(0)] = False

            visible_mask = gt_sample.visibs & valids
            invisible_mask = ~gt_sample.visibs & valids

            use_log_traj = self.log_loss
            pred_query_for_traj = pred_trajs_log[None] if use_log_traj else pred_trajs[None]
            gt_query_for_traj = gt_trajs_log if use_log_traj else gt_trajs
            query_traj_norm_mask = gt_sample.traj_norm_mask

            query_visible = visible_mask & gt_sample.traj_norm_mask if gt_sample.traj_norm_mask is not None else visible_mask
            query_valids = valids & gt_sample.traj_norm_mask if gt_sample.traj_norm_mask is not None else valids
            query_invisible = invisible_mask & gt_sample.traj_norm_mask if gt_sample.traj_norm_mask is not None else invisible_mask

            if self.use_conf_regression_loss:
                gamma_weight = self.gamma ** (num_iters - 1 - train_data.iter_idx)
                query_conf_loss, _, query_reg_loss = regression_loss_flexible(
                    pred_query_for_traj,
                    gt_query_for_traj,
                    query_valids,
                    conf=train_data.query_confs,
                    gradient_loss_fn="none",
                    gamma=self.conf_regression_gamma,
                    alpha=self.conf_regression_alpha,
                    valid_range=self.conf_regression_valid_range,
                )
                query_conf_total = gamma_weight * (query_conf_loss + query_reg_loss)
                traj_loss_dict["3d_query_loss"] = query_conf_total.clone().detach()
                traj_loss_dict["_conf_query_conf_loss"] = query_conf_loss.detach()
                traj_loss_dict["_conf_query_reg_loss"] = query_reg_loss.detach()
                if "3d_all" in traj_loss_dict:
                    traj_loss_dict['3d_grid_query_loss'] = traj_loss_dict["3d_all"].clone().detach()
                    traj_loss_dict["3d_all"] = traj_loss_dict["3d_all"] + query_conf_total * self.query_loss_weight
                else:
                    traj_loss_dict["3d_all"] = query_conf_total * self.query_loss_weight
                    traj_loss_dict["3d_visible"] = torch.zeros((), device=pred_query_for_traj.device)
                    traj_loss_dict["3d_invisible"] = torch.zeros((), device=pred_query_for_traj.device)
            else:
                query_loss_dict = self.calc_loss(
                    pd_traj=pred_query_for_traj,
                    gt_traj=gt_query_for_traj,
                    gt_sample=gt_sample,
                    visible_mask=query_visible,
                    valids=query_valids,
                    invisible_mask=query_invisible,
                    num_iters=num_iters,
                    iter_idx=train_data.iter_idx,
                    traj_norm_mask=query_traj_norm_mask,
                    no_topk=True,
                )
                traj_loss_dict["3d_query_loss"] = query_loss_dict["3d_all"].clone().detach()
                qw = self.query_loss_weight
                if "3d_visible" in traj_loss_dict:
                    traj_loss_dict['3d_grid_query_loss'] = traj_loss_dict["3d_all"].clone().detach()
                    traj_loss_dict["3d_visible"] += query_loss_dict["3d_visible"] * qw
                    traj_loss_dict["3d_invisible"] += query_loss_dict["3d_invisible"] * qw
                    traj_loss_dict["3d_all"] += query_loss_dict["3d_all"] * qw
                else:
                    traj_loss_dict["3d_visible"] = query_loss_dict["3d_visible"] * qw
                    traj_loss_dict["3d_invisible"] = query_loss_dict["3d_invisible"] * qw
                    traj_loss_dict["3d_all"] = query_loss_dict["3d_all"] * qw

            if self.compute_2d_loss:
                image_size = gt_sample.rgbs.shape[-2:]
                extrinsics_ = extrinsics[:, train_data.frame_range[0]:train_data.frame_range[1]]
                intrinsics_ = intrinsics[:, train_data.frame_range[0]:train_data.frame_range[1]]
                loss_2d_dict = self.calc_loss_2d(
                    pd_traj=rearrange(pred_trajs[None], "b t n c -> b n t c"),
                    gt_traj=rearrange(gt_trajs, "b t n c -> b n t c"),
                    gt_sample=gt_sample,
                    visible_mask=rearrange(visible_mask, "b t n -> b n t"),
                    invisible_mask=rearrange(invisible_mask, "b t n -> b n t"),
                    valids=rearrange(valids, "b t n -> b n t"),
                    num_iters=num_iters,
                    iter_idx=train_data.iter_idx,
                    extrinsics=extrinsics_,
                    intrinsics=intrinsics_,
                    image_size=image_size
                )

                traj_loss_dict["2d_query_loss"] = loss_2d_dict["2d_all"].clone().detach()
                if '2d_all' in traj_loss_dict:
                    traj_loss_dict['2d_grid_query_loss'] = traj_loss_dict["2d_all"].clone().detach()
                    traj_loss_dict["2d_all"] += loss_2d_dict["2d_all"] * self.query_loss_weight
                else:
                    traj_loss_dict["2d_all"] = loss_2d_dict["2d_all"] * self.query_loss_weight

            if train_data.query_coord_at_tgt is not None:
                pred_coords = train_data.query_coord_at_tgt
                window_num = train_data.window_num
                tgt_time_step_coord = gt_sample.tgt_time_step_scenes[:, window_num]
                if torch.is_tensor(tgt_time_step_coord):
                    tgt_time_step_coord = tgt_time_step_coord.long()
                gt_coords = orig_gt_trajs_3d[:, tgt_time_step_coord]
                use_log_traj = self.log_loss
                if use_log_traj:
                    gt_coords_ = apply_log_to_norm(gt_coords)
                    pred_coords_ = apply_log_to_norm(pred_coords)
                else:
                    gt_coords_ = gt_coords
                    pred_coords_ = pred_coords
                query_coord_loss = self._traj_loss_func(gt=gt_coords_, pd=pred_coords_)
                query_valids_ = query_valids[:, 0]
                query_coord_loss = self._topk_masked_mean(query_coord_loss, query_valids_, no_topk=True)
                gamma_weight = self.gamma ** (num_iters - 1 - train_data.iter_idx)
                query_coord_loss = gamma_weight * query_coord_loss
                traj_loss_dict["3d_query_coord_loss"] = query_coord_loss.clone().detach()

                if self.compute_2d_loss:
                    image_size = gt_sample.rgbs.shape[-2:]
                    extrinsics_coord = extrinsics[:, tgt_time_step_coord]
                    intrinsics_coord = intrinsics[:, tgt_time_step_coord]
                    pd_coords_2d = pred_coords.unsqueeze(1)
                    query_coord_loss_2d = self.calc_loss_2d(
                        pd_traj=rearrange(pd_coords_2d, "b t n c -> b n t c"),
                        gt_traj=rearrange(gt_coords, "b t n c -> b n t c"),
                        gt_sample=gt_sample,
                        visible_mask=None,
                        invisible_mask=None,
                        valids=query_valids_[:, :, None],
                        num_iters=num_iters,
                        iter_idx=train_data.iter_idx,
                        extrinsics=extrinsics_coord,
                        intrinsics=intrinsics_coord,
                        image_size=image_size,
                    )["2d_all"]
                    traj_loss_dict["2d_query_coord_loss"] = query_coord_loss_2d.clone().detach()

                query_coord_loss = query_coord_loss * self.query_loss_weight
                if query_coord_loss_2d is not None:
                    query_coord_loss_2d = query_coord_loss_2d * self.query_loss_weight

            query_bce_loss = self._compute_bce_loss(
                gt_valids=query_valids,
                gt_visibs=gt_sample.visibs,
                pred_visibs=train_data.query_visibs,
                iter_idx=train_data.iter_idx,
                num_iters=num_iters,
            )
            traj_loss_dict["3d_query_bce_loss"] = query_bce_loss.clone().detach()

            if sample.dense_trajs_3d is not None and not skip_dense_eval:
                traj_loss_dict["3d_grid_query_bce_loss"] = bce_loss.clone().detach()
                bce_loss += query_bce_loss * self.query_loss_weight
            else:
                bce_loss = query_bce_loss * self.query_loss_weight

            if train_data.query_dynamic_logits is not None and self.dynamic_bce_weight > 0:
                query_len = train_data.query_coords[0].shape[0]
                gt_query_dynamic = self._compute_window_dynamic_from_trajs(
                    orig_gt_trajs_3d,
                    train_data.frame_range,
                    norm_factor=gt_sample.gt_norm_factor,
                    threshold=gt_sample.dynamic_filter_threshold,
                    valids=gt_sample.valids,
                ).float()[..., None].expand(-1, -1, query_len)
                mask = gt_sample.gt_query_point[..., 0] < train_data.frame_range[1]
                query_valid = gt_sample.valids & mask[:, None]
                query_valid = rearrange(query_valid, "b t n -> b n t")

                pred_qdl = train_data.query_dynamic_logits
                if pred_qdl.dim() == 2:
                    pred_qdl = pred_qdl[..., None].expand(-1, -1, query_len)
                elif pred_qdl.shape[1:] != query_valid.shape[1:]:
                    pred_qdl = rearrange(pred_qdl, "b t n -> b n t")

                query_dynamic_bce = self._compute_balanced_dynamic_bce(
                    pred_qdl,
                    gt_query_dynamic,
                    query_valid,
                    train_data.iter_idx,
                    num_iters,
                )
                traj_loss_dict["query_dynamic_bce"] = query_dynamic_bce.clone().detach()
                if "dense_dynamic_bce" in traj_loss_dict:
                    traj_loss_dict["dense_grid_query_dynamic_bce"] = traj_loss_dict["dense_dynamic_bce"].clone()
                    traj_loss_dict["dense_dynamic_bce"] = (
                        traj_loss_dict["dense_dynamic_bce"] + query_dynamic_bce * self.query_loss_weight
                    )
                else:
                    traj_loss_dict["dense_dynamic_bce"] = query_dynamic_bce * self.query_loss_weight

        # also calculate full-track loss on the subsampled dense grid cells.
        # These are grid-anchored GT tracks (grid-level, not subpixel) so they add
        # dense tracking supervision without the fine-grained misalignment that
        # query tracks would introduce. Reuses query_loss_weight (no new config).
        if (
            train_data.dense_sample_coords is not None
            and gt_sample.dense_sample_trajs_3d is not None
            and torch.is_tensor(gt_sample.dense_sample_trajs_3d)
        ):
            pred_ds = train_data.dense_sample_coords[0]  # (t_win, M, 3)
            ds_window = pred_ds.shape[0]
            gt_ds = gt_sample.dense_sample_trajs_3d[:, :ds_window]  # (B, t_win, M, 3)
            ds_visibs = gt_sample.dense_sample_visibs[:, :ds_window]  # (B, t_win, M)
            ds_valids = gt_sample.dense_sample_valids[:, :ds_window].clone()  # (B, t_win, M)

            # invalidate cells not yet born by the end of this window
            born_mask = gt_sample.dense_sample_points[..., 0] >= train_data.frame_range[1]  # (B, M)
            ds_valids[:, :, born_mask.squeeze(0)] = False
            # invalidate cells not tracked this window (dropped voxels, etc.)
            if train_data.dense_sample_valid_cells is not None:
                ds_valids[:, :, ~train_data.dense_sample_valid_cells] = False

            if self.log_loss:
                gt_ds_for_traj = apply_log_to_norm(gt_ds)
                pred_ds_for_traj = apply_log_to_norm(pred_ds)[None]
            else:
                gt_ds_for_traj = gt_ds
                pred_ds_for_traj = pred_ds[None]

            ds_visible = ds_visibs & ds_valids
            ds_invisible = ~ds_visibs & ds_valids

            ds_loss_dict = self.calc_loss(
                pd_traj=pred_ds_for_traj,
                gt_traj=gt_ds_for_traj,
                gt_sample=gt_sample,
                visible_mask=ds_visible,
                valids=ds_valids,
                invisible_mask=ds_invisible,
                num_iters=num_iters,
                iter_idx=train_data.iter_idx,
                traj_norm_mask=None,
                no_topk=True,
            )
            dsw = self.query_loss_weight
            traj_loss_dict["3d_dense_sample_loss"] = ds_loss_dict["3d_all"].clone().detach()
            if "3d_visible" in traj_loss_dict:
                traj_loss_dict["3d_visible"] = traj_loss_dict["3d_visible"] + ds_loss_dict["3d_visible"] * dsw
                traj_loss_dict["3d_invisible"] = traj_loss_dict["3d_invisible"] + ds_loss_dict["3d_invisible"] * dsw
                traj_loss_dict["3d_all"] = traj_loss_dict["3d_all"] + ds_loss_dict["3d_all"] * dsw
            else:
                traj_loss_dict["3d_visible"] = ds_loss_dict["3d_visible"] * dsw
                traj_loss_dict["3d_invisible"] = ds_loss_dict["3d_invisible"] * dsw
                traj_loss_dict["3d_all"] = ds_loss_dict["3d_all"] * dsw

            if train_data.dense_sample_visibs is not None:
                ds_bce_loss = self._compute_bce_loss(
                    gt_valids=ds_valids,
                    gt_visibs=ds_visibs,
                    pred_visibs=train_data.dense_sample_visibs,
                    iter_idx=train_data.iter_idx,
                    num_iters=num_iters,
                )
                traj_loss_dict["3d_dense_sample_bce_loss"] = ds_bce_loss.clone().detach()
                bce_loss = bce_loss + ds_bce_loss * dsw

        total_loss = (bce_loss * self.bce_weight)
        if self.compute_3d_loss:
            total_loss += (
                traj_loss_dict["3d_all"] * self.traj_weight
                + traj_loss_dict["3d_visible"] * self.traj_visible_weight
                + traj_loss_dict["3d_invisible"] * self.traj_invisible_weight
                + (query_coord_loss * self.query_coord_weight if query_coord_loss is not None else 0.0)
            )

        if self.compute_2d_loss:
            total_loss += (
                traj_loss_dict["2d_all"] * self.traj_2d_weight
                + (query_coord_loss_2d * self.query_coord_weight if query_coord_loss_2d is not None else 0.0)
            )

        if "dense_dynamic_bce" in traj_loss_dict:
            total_loss += traj_loss_dict["dense_dynamic_bce"] * self.dynamic_bce_weight

        train_info = {}
        if self.traj_weight > 0:
            train_info["traj_loss_all"] = traj_loss_dict["3d_all"]
        if self.bce_weight > 0:
            train_info["bce_loss"] = bce_loss
        train_info["traj_loss_visible"] = traj_loss_dict["3d_visible"]
        train_info["traj_loss_invisible"] = traj_loss_dict["3d_invisible"]
        if '3d_dynamic' in traj_loss_dict:
            train_info["traj_loss_dynamic"] = traj_loss_dict["3d_dynamic"]
        if '3d_static' in traj_loss_dict:
            train_info["traj_loss_static"] = traj_loss_dict["3d_static"]
        if train_data.query_coords is not None:
            train_info['3d_query_loss'] = traj_loss_dict["3d_query_loss"]
            train_info['3d_query_bce_loss'] = traj_loss_dict["3d_query_bce_loss"]
            if '3d_grid_query_loss' in traj_loss_dict:
                train_info['3d_grid_query_loss'] = traj_loss_dict["3d_grid_query_loss"]
                train_info['3d_grid_query_bce_loss'] = traj_loss_dict["3d_grid_query_bce_loss"]
        if train_data.query_coord_at_tgt is not None and '3d_query_coord_loss' in traj_loss_dict:
            train_info['3d_query_coord_loss'] = traj_loss_dict["3d_query_coord_loss"]
        if self.compute_2d_loss:
            train_info["traj_2d_loss_all"] = traj_loss_dict["2d_all"]
            if train_data.query_coords is not None:
                train_info['2d_query_loss'] = traj_loss_dict["2d_query_loss"]
                if '2d_grid_query_loss' in traj_loss_dict:
                    train_info['2d_grid_query_loss'] = traj_loss_dict["2d_grid_query_loss"]
            if train_data.query_coord_at_tgt is not None and '2d_query_coord_loss' in traj_loss_dict:
                train_info['2d_query_coord_loss'] = traj_loss_dict["2d_query_coord_loss"]
        if 'dense_dynamic_bce' in traj_loss_dict:
            train_info['dense_dynamic_bce'] = traj_loss_dict['dense_dynamic_bce']
        if 'query_dynamic_bce' in traj_loss_dict:
            train_info['query_dynamic_bce'] = traj_loss_dict['query_dynamic_bce']
        if 'dense_grid_query_dynamic_bce' in traj_loss_dict:
            train_info['dense_grid_query_dynamic_bce'] = traj_loss_dict['dense_grid_query_dynamic_bce']
        if '3d_dense_sample_loss' in traj_loss_dict:
            train_info['3d_dense_sample_loss'] = traj_loss_dict['3d_dense_sample_loss']
        if '3d_dense_sample_bce_loss' in traj_loss_dict:
            train_info['3d_dense_sample_bce_loss'] = traj_loss_dict['3d_dense_sample_bce_loss']
        # Log conf regression diagnostic info (stored in traj_loss_dict with _ prefix)
        for k, v in traj_loss_dict.items():
            if k.startswith("_conf_"):
                train_info[k[1:]] = v

        if return_per_point_loss and "per_point_loss" in traj_loss_dict:
            train_info["per_point_loss"] = traj_loss_dict["per_point_loss"]
            train_info["per_point_valid"] = traj_loss_dict["per_point_valid"]

        for k in train_info:
            if isinstance(train_info[k], torch.Tensor) and train_info[k].requires_grad:
                train_info[k] = train_info[k].detach()
        return total_loss, train_info
