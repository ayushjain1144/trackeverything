# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared Sim(3) alignment utilities for long-video geometry annotators.

Ported from third_party/Pi-Long/loop_utils/sim3utils.py + alignment_torch.py,
distilled down to the robust weighted IRLS sim3 solver that pi_long.py uses
to align adjacent chunks. Re-implemented in torch to avoid the heavy
Pi-Long dep chain (numba / triton / pypose / sim3solve / fastloop / DBoW2).
"""

from __future__ import annotations

import contextlib
import math
from typing import List, Tuple

import numpy as np
import torch


@contextlib.contextmanager
def disable_autocast():
    """Force fp32 math: disable any ambient autocast (CPU + CUDA)."""
    with torch.autocast(device_type="cpu", enabled=False), torch.autocast(
        device_type="cuda", enabled=False
    ):
        yield


def weighted_estimate_sim3_torch(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form weighted Sim(3) that maps source -> target.

    All inputs live on the same device; returns (s, R, t) as torch tensors.
    """
    with disable_autocast():
        source = source.float()
        target = target.float()
        weights = weights.float()

        total_weight = weights.sum()
        if total_weight < 1e-6:
            raise ValueError("Total weight too small for meaningful Sim(3) estimation")
        w = weights / total_weight

        mu_src = (w[:, None] * source).sum(dim=0)
        mu_tgt = (w[:, None] * target).sum(dim=0)
        src_c = source - mu_src
        tgt_c = target - mu_tgt

        scale_src = torch.sqrt((w * (src_c ** 2).sum(dim=1)).sum()).clamp_min(1e-12)
        scale_tgt = torch.sqrt((w * (tgt_c ** 2).sum(dim=1)).sum())
        s = scale_tgt / scale_src

        sqw = torch.sqrt(w)[:, None]
        H = (s * src_c * sqw).T @ (tgt_c * sqw)
        U, _, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T
        if torch.linalg.det(R) < 0:
            Vt = Vt.clone()
            Vt[2] = -Vt[2]
            R = Vt.T @ U.T
        t = mu_tgt - s * (R @ mu_src)
        return s, R, t


def robust_weighted_sim3(
    source: np.ndarray,
    target: np.ndarray,
    init_weights: np.ndarray,
    delta: float,
    max_iters: int,
    tol: float,
    device: str,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """IRLS with Huber weights on top of the closed-form weighted sim3.

    Equivalent to Pi-Long's robust_weighted_estimate_sim3_torch(align_method='sim3').
    """
    src = torch.from_numpy(source).to(device).float()
    tgt = torch.from_numpy(target).to(device).float()
    w0 = torch.from_numpy(init_weights).to(device).float()

    with disable_autocast():
        s, R, t = weighted_estimate_sim3_torch(src, tgt, w0)
        prev_err = float("inf")

        for _ in range(max_iters):
            transformed = s * (src @ R.T) + t
            residuals = (tgt - transformed).norm(dim=1)

            huber = torch.ones_like(residuals)
            big = residuals > delta
            huber[big] = delta / residuals[big]
            combined = w0 * huber
            combined = combined / (combined.sum() + 1e-12)

            s_new, R_new, t_new = weighted_estimate_sim3_torch(src, tgt, combined)

            dparam = (s_new - s).abs() + (t_new - t).norm()
            cos_ang = ((torch.trace(R_new @ R.T) - 1) * 0.5).clamp(-1.0, 1.0)
            dang = torch.acos(cos_ang).item()

            huber_loss = torch.where(
                residuals <= delta,
                0.5 * residuals ** 2,
                delta * (residuals - 0.5 * delta),
            )
            cur_err = (huber_loss * w0).sum().item()

            s, R, t = s_new, R_new, t_new
            if dparam.item() < tol and dang < math.radians(0.1):
                break
            if abs(prev_err - cur_err) < tol * max(prev_err, 1.0):
                break
            prev_err = cur_err

    return (
        float(s.item()),
        R.detach().cpu().numpy().astype(np.float32),
        t.detach().cpu().numpy().astype(np.float32),
    )


def weighted_align_point_maps(
    target_pts: np.ndarray,  # (T, H, W, 3) — chunk i's last `overlap` frames
    target_conf: np.ndarray,  # (T, H, W)
    source_pts: np.ndarray,  # (T, H, W, 3) — chunk i+1's first `overlap` frames
    source_conf: np.ndarray,  # (T, H, W)
    conf_threshold: float,
    device: str,
    delta: float,
    max_iters: int,
    tol: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Returns (s, R, t) s.t. target ≈ s R source + t. Mirrors Pi-Long's
    weighted_align_point_maps(point_map1=target, point_map2=source)."""
    T = min(target_pts.shape[0], source_pts.shape[0])
    tgt_list, src_list, wt_list = [], [], []
    for i in range(T):
        m1 = target_conf[i] > conf_threshold
        m2 = source_conf[i] > conf_threshold
        valid = m1 & m2
        if not np.any(valid):
            continue
        idx = np.where(valid)
        tgt_list.append(target_pts[i][idx])
        src_list.append(source_pts[i][idx])
        # Same confidence weighting Pi-Long uses (sqrt of product).
        wt_list.append(np.sqrt(target_conf[i][idx] * source_conf[i][idx]))

    if not tgt_list:
        raise RuntimeError(
            "No matching confident point pairs found across chunk overlap; "
            "cannot estimate Sim(3) alignment."
        )
    tgt = np.concatenate(tgt_list, axis=0).astype(np.float32)
    src = np.concatenate(src_list, axis=0).astype(np.float32)
    wt = np.concatenate(wt_list, axis=0).astype(np.float32)

    return robust_weighted_sim3(
        src,
        tgt,
        wt,
        delta=delta,
        max_iters=max_iters,
        tol=tol,
        device=device,
    )


def accumulate_sim3(
    transforms: List[Tuple[float, np.ndarray, np.ndarray]],
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """Composes adjacent sim3 transforms into cumulative-to-frame-0 ones.

    If `transforms[i] = (s_i, R_i, t_i)` maps chunk i+1 -> chunk i, then the
    returned list has cum[k] mapping chunk k+1 -> chunk 0. This matches
    Pi-Long's accumulate_sim3_transforms exactly.
    """
    if not transforms:
        return []
    cum = [transforms[0]]
    for i in range(1, len(transforms)):
        s_p, R_p, t_p = cum[-1]
        s_n, R_n, t_n = transforms[i]
        cum.append((s_p * s_n, R_p @ R_n, s_p * (R_p @ t_n) + t_p))
    return cum


def apply_sim3_c2w(
    c2w: np.ndarray,
    s: float,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Apply S = [[sR, t], [0, 1]] to each cam2world, then normalize rotation
    by 1/s so the resulting c2w stays orthonormal. Matches Pi-Long's
    save_camera_poses convention."""
    S = np.eye(4, dtype=np.float32)
    S[:3, :3] = (s * R).astype(np.float32)
    S[:3, 3] = t.astype(np.float32)
    new_c2w = np.einsum("ij, tjk -> tik", S, c2w.astype(np.float32))
    new_c2w = new_c2w.copy()
    new_c2w[:, :3, :3] = new_c2w[:, :3, :3] / float(s)
    return new_c2w
