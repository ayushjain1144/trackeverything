# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ipdb
import logging
import torch
import wandb
from torch.nn import functional as F
import cv2
import numpy as np
from pathlib import Path

from einops import rearrange, repeat
from iopath.common.file_io import PathManager

path_manager = PathManager()

from .common_utils import batch_project
from .alltrack3r_utils import (
    CompressedTracks,
    TRACK_VALID_THRESH,
    _subsample_independent_caps,
    dense_dynamic_logits_diagonal,
)
from .cotracker.visualizer import Visualizer


st = ipdb.set_trace
logger = logging.getLogger(__name__)


COLORMAP_FILE = Path(__file__).resolve().parent / "bremm.png"


def _synthetic_bremm_colormap(h=256, w=256):
    """HSV sheet: hue from x, value from y (used when bremm.png is missing)."""
    xs = np.linspace(0, 1, w, dtype=np.float32)
    ys = np.linspace(0, 1, h, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = (xv * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (yv * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


class ColorMap2d:
    def __init__(self, filename=None):
        path = Path(filename) if filename is not None else COLORMAP_FILE
        if path.is_file():
            img = cv2.imread(str(path))
            if img is None:
                self._img = _synthetic_bremm_colormap()
            else:
                self._img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            self._img = _synthetic_bremm_colormap()
        
        self._height = self._img.shape[0]
        self._width = self._img.shape[1]

    def __call__(self, X):
        assert len(X.shape) == 2
        output = np.zeros((X.shape[0], 3), dtype=np.uint8)
        for i in range(X.shape[0]):
            x, y = X[i, :]
            xp = int((self._width-1) * x)
            yp = int((self._height-1) * y)
            xp = np.clip(xp, 0, self._width-1)
            yp = np.clip(yp, 0, self._height-1)
            output[i, :] = self._img[yp, xp]
        return output
    
def get_2d_colors(xys, H, W):
    N,D = xys.shape
    assert(D==2)
    bremm = ColorMap2d()
    xys[:,0] /= float(W-1)
    xys[:,1] /= float(H-1)
    colors = bremm(xys)
    # print('colors', colors)
    # colors = (colors[0]*255).astype(np.uint8) 
    # colors = (int(colors[0]),int(colors[1]),int(colors[2]))
    return colors


def get_cotracker_rainbow_colors(xys, H=None, W=None):
    """CoTracker-style colors: matplotlib ``gist_rainbow`` by birth-frame y.

    Matches ``cotracker.utils.visualizer.Visualizer`` with ``mode="rainbow"``:
    each track gets a constant color from normalizing its y coordinate.
    ``H``/``W`` are unused (kept for call-site symmetry with ``get_2d_colors``).
    """
    del H, W
    xys = np.asarray(xys, dtype=np.float64)
    if xys.ndim != 2 or xys.shape[1] != 2:
        raise ValueError(f"xys must have shape (N, 2), got {tuple(xys.shape)}")
    from matplotlib import cm
    from matplotlib import colors as mcolors

    y = xys[:, 1]
    y_min = float(np.nanmin(y)) if y.size else 0.0
    y_max = float(np.nanmax(y)) if y.size else 1.0
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        y_min, y_max = 0.0, 1.0
    norm = mcolors.Normalize(vmin=y_min, vmax=y_max)
    cmap = cm.get_cmap("gist_rainbow")
    rgba = cmap(norm(y))  # (N, 4) in 0..1
    return (rgba[:, :3] * 255.0).astype(np.uint8)


def draw_pts_hard(rgbs, trajs, visibs, colormap, bkg_opacity=1.0, point_radius: int = 0):
    """Write crisp opaque dots. ``point_radius=0`` -> 1px; ``>0`` -> filled circle.

    CoTracker uses ``radius = linewidth * 2`` (default linewidth=2 => radius=4)
    with a filled PIL ellipse; we match that look via ``cv2.circle``.
    """
    device = rgbs.device
    T, C, H, W = rgbs.shape
    trajs = trajs.permute(1, 0, 2)  # N,T,2
    visibs = visibs.permute(1, 0)  # N,T
    colors = torch.as_tensor(colormap, dtype=torch.float32, device=device)
    rgbs = rgbs * float(bkg_opacity)
    radius = max(0, int(point_radius))
    if radius <= 0:
        for t in range(T):
            mask = visibs[:, t]
            if not bool(mask.any()):
                continue
            xy = trajs[mask, t]
            cx = xy[:, 0].round().long().clamp(0, W - 1)
            cy = xy[:, 1].round().long().clamp(0, H - 1)
            colors_now = colors[mask]
            flat = (cy * W + cx).long()
            img_flat = rgbs[t].view(C, -1)
            img_flat[:, flat] = colors_now.t()
        return rgbs.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()

    # Filled circles on CPU (matches CoTracker PIL ellipse look).
    frames = (
        rgbs.detach().cpu().float().permute(0, 2, 3, 1).contiguous().numpy()
    )  # T,H,W,3
    trajs_np = trajs.detach().cpu().numpy()
    visibs_np = visibs.detach().cpu().numpy().astype(bool)
    colors_np = colors.detach().cpu().numpy()
    for t in range(T):
        mask = visibs_np[:, t]
        if not mask.any():
            continue
        img = np.clip(frames[t], 0, 255).astype(np.uint8)
        xy = trajs_np[mask, t]
        cols = np.clip(np.round(colors_np[mask]), 0, 255).astype(np.int32)
        for i in range(xy.shape[0]):
            c = (int(cols[i, 0]), int(cols[i, 1]), int(cols[i, 2]))
            cv2.circle(
                img,
                (int(round(xy[i, 0])), int(round(xy[i, 1]))),
                radius,
                c,
                thickness=-1,
                lineType=cv2.LINE_8,
            )
        frames[t] = img.astype(np.float32)
    return np.clip(frames, 0, 255).astype(np.uint8)


def draw_pts_gpu(rgbs, trajs, visibs, colormap, rate=1, bkg_opacity=0.5):
    device = rgbs.device
    T, C, H, W = rgbs.shape
    trajs = trajs.permute(1,0,2) # N,T,2
    visibs = visibs.permute(1,0) # N,T
    N = trajs.shape[0]
    colors = torch.as_tensor(colormap, dtype=torch.float32, device=device)  # [N,3]

    rgbs = rgbs * bkg_opacity # darken, to see the point tracks better
    
    opacity = 1.0
    if rate==1:
        radius = 1
        opacity = 0.9
    elif rate==2:
        radius = 1
    elif rate== 4:
        radius = 2
    elif rate== 8:
        radius = 4
    else:
        radius = 6
    sharpness = 0.15 + 0.05 * np.log2(rate)
    
    D = radius * 2 + 1
    y = torch.arange(D, device=device).float()[:, None] - radius
    x = torch.arange(D, device=device).float()[None, :] - radius
    dist2 = x**2 + y**2
    icon = torch.clamp(1 - (dist2 - (radius**2) / 2.0) / (radius * 2 * sharpness), 0, 1)  # [D,D]
    icon = icon.view(1, D, D)
    dx = torch.arange(-radius, radius + 1, device=device)
    dy = torch.arange(-radius, radius + 1, device=device)
    disp_y, disp_x = torch.meshgrid(dy, dx, indexing="ij")  # [D,D]
    for t in range(T):
        mask = visibs[:, t]  # [N]
        if mask.sum() == 0:
            continue
        xy = trajs[mask, t] + 0.5  # [N,2]
        xy[:, 0] = xy[:, 0].clamp(0, W - 1)
        xy[:, 1] = xy[:, 1].clamp(0, H - 1)
        colors_now = colors[mask]  # [N,3]
        N = xy.shape[0]
        cx = xy[:, 0].long()  # [N]
        cy = xy[:, 1].long()
        x_grid = cx[:, None, None] + disp_x  # [N,D,D]
        y_grid = cy[:, None, None] + disp_y  # [N,D,D]
        valid = (x_grid >= 0) & (x_grid < W) & (y_grid >= 0) & (y_grid < H)
        x_valid = x_grid[valid]  # [K]
        y_valid = y_grid[valid]
        icon_weights = icon.expand(N, D, D)[valid]  # [K]
        colors_valid = colors_now[:, :, None, None].expand(N, 3, D, D).permute(1, 0, 2, 3)[
            :, valid
        ]  # [3, K]
        idx_flat = (y_valid * W + x_valid).long()  # [K]

        accum = torch.zeros_like(rgbs[t])  # [3, H, W]
        weight = torch.zeros(1, H * W, device=device)  # [1, H*W]
        img_flat = accum.view(C, -1)  # [3, H*W]
        weighted_colors = colors_valid * icon_weights  # [3, K]
        img_flat.scatter_add_(1, idx_flat.unsqueeze(0).expand(C, -1), weighted_colors)
        weight.scatter_add_(1, idx_flat.unsqueeze(0), icon_weights.unsqueeze(0))
        weight = weight.view(1, H, W)

        alpha = weight.clamp(0, 1) * opacity
        accum = accum / (weight + 1e-6)  # [3, H, W]
        rgbs[t] = rgbs[t] * (1 - alpha) + accum * alpha
    rgbs = rgbs.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy() # T,H,W,3
    if bkg_opacity==0.0:
        for t in range(T):
            hsv_frame = cv2.cvtColor(rgbs[t], cv2.COLOR_RGB2HSV)
            saturation_factor = 1.5
            hsv_frame[..., 1] = np.clip(hsv_frame[..., 1] * saturation_factor, 0, 255)
            rgbs[t] = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)
    return rgbs


# def generate_dense_track_vis(
#     sample,
#     preds,
#     output_dir,
#     step_id,
#     batch_idx,
#     dataset_name,
#     log_wandb,
#     writer,
#     scale=1.0,
# ):
#     # use draw_pts_gpu to generate the video

def generate_videos(
    sample,
    preds,
    output_dir,
    step_id,
    batch_idx,
    dataset_name,
    log_wandb,
    writer,
    scale=1.0,
    stop=False,
):
    for i in range(len(sample.rgbs)):
        save_dir = f"{output_dir}/visualization_{dataset_name}"
        filename = f"video_{step_id}_{batch_idx}_{i}"
        path_manager.mkdirs(save_dir)
        # save_dir.mkdir(parents=True, exist_ok=True)
        vis = Visualizer(
            save_dir=save_dir, pad_value=120, linewidth=0.5, show_first_frame=0
        )
        tracks_3d = preds.coords[i : i + 1]

        # we are using pd_to_gt_scale to scale the predicted 3D points to the GT scale
        if scale is not None:
            tracks_3d = tracks_3d * scale

        tracks_2d = batch_project(
            tracks_3d,
            repeat(
                sample.intrinsics[i : i + 1],
                "b t i j -> b t n i j",
                n=tracks_3d.shape[2],
            ),
            repeat(
                sample.extrinsics[i : i + 1],
                "b t i j -> b t n i j",
                n=tracks_3d.shape[2],
            ),
        )
        gt_tracks_3d = sample.trajs_3d[i : i + 1]

        gt_tracks_2d = batch_project(
            gt_tracks_3d,
            repeat(
                sample.intrinsics[i : i + 1],
                "b t i j -> b t n i j",
                n=gt_tracks_3d.shape[2],
            ),
            repeat(
                sample.extrinsics[i : i + 1],
                "b t i j -> b t n i j",
                n=gt_tracks_3d.shape[2],
            ),
        )
        # import ipdb
        # st = ipdb.set_trace
        vis.linewidth = 1.0
        res_video = vis.visualize(
            video=(sample.rgbs[i : i + 1] * 255).to(torch.uint8),
            tracks=gt_tracks_2d,
            visibility=(sample.visibs[i : i + 1] > 0.0),
            # valids=torch.ones_like(sample.valids[i : i + 1]),
            valids=sample.valids[i : i + 1],
            filename=f"{dataset_name}-viz-gt/viz-{batch_idx}-{i}",
            writer=writer,
            step=step_id,
        )

        vis.linewidth = 0.5
        # overlay_video = res_video[
        #     ..., vis.pad_value : -vis.pad_value, vis.pad_value : -vis.pad_value
        # ]
            
        # replace all infs with 0
        res_video = vis.visualize(
            video=res_video.to(torch.uint8),
            tracks=tracks_2d,
            visibility=(preds.visibs[i : i + 1] > 0.0),
            # valids=torch.ones_like(sample.valids[i : i + 1]),
            valids=sample.valids[i : i + 1],
            filename=f"{dataset_name}-viz-gt_pred-new/viz-{batch_idx}-{i}",
            writer=writer,
            step=step_id,
            already_padded=True,
        )

        res_video = vis.visualize(
            video=(sample.rgbs[i : i + 1] * 255).to(torch.uint8),
            tracks=tracks_2d,
            visibility=(preds.visibs[i : i + 1] > 0.0),
            # valids=torch.ones_like(sample.valids[i : i + 1]),
            valids=sample.valids[i : i + 1],
            filename=f"{dataset_name}-viz_pred-new/viz-{batch_idx}-{i}",
            writer=writer,
            step=step_id,
        )

        if log_wandb:
            # viz_key = f"{dataset_name}-viz/viz-{batch_idx}-{i}"
            viz_keys = [
                f"{dataset_name}-viz-gt/viz-{batch_idx}-{i}",
                f"{dataset_name}-viz-gt_pred-new/viz-{batch_idx}-{i}",
                f"{dataset_name}-viz_pred-new/viz-{batch_idx}-{i}",
            ]
            for viz_key in viz_keys:
                wandb.log(
                    {
                        "step": step_id,
                        viz_key: wandb.Video(str(f"{save_dir}/{viz_key}.mp4")),
                    }
                )
        # else:
        #     writer.add_video(
        #         f"{dataset_name}-viz/viz-{batch_idx}-{i}",
        #         f"{save_dir}/{filename}.mp4",
        #         step_id,
        #     )


# Static/dynamic overlay colors (RGB). Dynamic = red, static = blue.
_SD_DYNAMIC_COLOR = (220, 40, 40)
_SD_STATIC_COLOR = (40, 90, 220)


def _sd_label_color(is_dynamic: bool):
    return _SD_DYNAMIC_COLOR if is_dynamic else _SD_STATIC_COLOR


def _window_dynamic_from_trajs_np(trajs_3d, threshold, norm_factor=None, valids=None):
    """Per-point window-motion label matching the criterion's supervision GT.

    trajs_3d: (T, N, 3) tensor. Returns (N,) bool tensor: True if the xyz
    bounding-box diagonal over the window exceeds `threshold`. If ``norm_factor``
    is given, the (scene-normalized) motion is rescaled to metric units so
    ``threshold`` is interpreted in the same metric space as the dense GT.

    ``valids`` (T, N) restricts the extent to frames where the point is valid, so
    this overlay shows the same GT the criterion supervises against. Without it
    the visualisation silently disagrees with training.
    See docs/static_dynamic_labels.md.
    """
    if trajs_3d.shape[0] <= 1:
        return torch.zeros(trajs_3d.shape[1], dtype=torch.bool, device=trajs_3d.device)
    if valids is None:
        max_xyz = torch.amax(trajs_3d, dim=0)
        min_xyz = torch.amin(trajs_3d, dim=0)
        diff_l2 = torch.linalg.norm(max_xyz - min_xyz, dim=-1)
    else:
        v = valids.bool()
        any_valid = v.any(dim=0)
        keep = (v | (~any_valid)[None, :])[..., None]
        big = torch.finfo(trajs_3d.dtype).max
        max_xyz = torch.where(keep, trajs_3d, -big).amax(dim=0)
        min_xyz = torch.where(keep, trajs_3d, big).amin(dim=0)
        diff_l2 = torch.linalg.norm(max_xyz - min_xyz, dim=-1)
        diff_l2 = torch.where(any_valid, diff_l2, torch.zeros_like(diff_l2))
    if norm_factor is not None:
        diff_l2 = diff_l2 * float(norm_factor)
    if isinstance(threshold, torch.Tensor):
        threshold = float(threshold.reshape(-1)[0])
    return diff_l2 > threshold


def _window_dynamic_per_frame_np(trajs_3d, threshold, window_size, norm_factor=None,
                                 valids=None):
    """Per-frame per-point window-motion label.

    trajs_3d: (T, N, 3). Windows are the fixed non-overlapping tiling induced by
    ``window_size`` (``[0, w), [w, 2w), ...``, trailing window clipped to T),
    matching the model's window schedule. Frame ``t`` gets the motion label of
    the window it belongs to. Returns (T, N) bool tensor. Falls back to a single
    whole-clip window when ``window_size`` is None.
    """
    T, N, _ = trajs_3d.shape
    out = torch.zeros(T, N, dtype=torch.bool, device=trajs_3d.device)
    if not window_size or window_size <= 0:
        window_size = T
    w = int(window_size)
    for ws in range(0, T, w):
        we = min(ws + w, T)
        lab = _window_dynamic_from_trajs_np(
            trajs_3d[ws:we], threshold, norm_factor=norm_factor,
            valids=None if valids is None else valids[ws:we],
        )  # (N,)
        out[ws:we] = lab[None]
    return out


def _sticky_dynamic_mask(dyn: torch.Tensor, time_dim: int) -> torch.Tensor:
    """Once a track is dynamic at some frame, keep it dynamic for later frames.

    ``dyn`` is a bool mask with a time axis at ``time_dim`` (``(N, T)`` or
    ``(T, N)``). Does not paint frames *before* the first dynamic hit.
    """
    return dyn.cummax(dim=time_dim).values


def _sd_stack_panels(left, right, left_label="GT", right_label="PRED", gap=6):
    """Stack two (H, W, 3) uint8 RGB panels side by side with titles."""
    left = np.ascontiguousarray(left)
    right = np.ascontiguousarray(right)
    cv2.putText(left, left_label, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(right, right_label, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    sep = np.full((left.shape[0], gap, 3), 255, dtype=np.uint8)
    return np.concatenate([left, sep, right], axis=1)


def _sd_dense_overlay(rgb, dyn_mask, valid_mask, alpha=0.5):
    """Alpha-blend a red(dynamic)/blue(static) color layer onto an RGB frame.

    rgb: (H, W, 3) uint8. dyn_mask / valid_mask: (h, w) bool (any resolution;
    nearest-resized to the frame). Only valid pixels are tinted.
    """
    H, W = rgb.shape[:2]
    dyn = cv2.resize(dyn_mask.astype(np.uint8), (W, H),
                     interpolation=cv2.INTER_NEAREST).astype(bool)
    val = cv2.resize(valid_mask.astype(np.uint8), (W, H),
                     interpolation=cv2.INTER_NEAREST).astype(bool)
    color = np.zeros_like(rgb)
    color[val & dyn] = _SD_DYNAMIC_COLOR
    color[val & (~dyn)] = _SD_STATIC_COLOR
    out = rgb.copy()
    out[val] = (rgb[val] * (1 - alpha) + color[val] * alpha).astype(np.uint8)
    return out


def _sd_write_and_log_video(frames, save_dir, filename, log_wandb, writer, step_id, fps=10):
    """frames: (T, H, W, 3) uint8 RGB. Writes mp4 and logs to wandb/tensorboard."""
    import os
    import imageio

    save_path = f"{save_dir}/{filename}.mp4"
    path_manager.mkdirs(os.path.dirname(save_path))
    video_writer = imageio.get_writer(save_path, fps=fps)
    for frame in frames:
        video_writer.append_data(frame)
    video_writer.close()

    if log_wandb:
        wandb.log({"step": step_id, filename: wandb.Video(str(save_path))})
    elif writer is not None:
        vid = torch.from_numpy(frames).permute(0, 3, 1, 2)[None]  # (1, T, 3, H, W)
        writer.add_video(filename, vid.to(torch.uint8), global_step=step_id, fps=fps)
    return save_path


@torch.no_grad()
def generate_query_dynamic_videos(
    sample,
    preds,
    output_dir,
    step_id,
    batch_idx,
    dataset_name,
    dynamic_threshold=0.5,
    dynamic_filter_threshold=0.1,
    log_wandb=False,
    writer=None,
    fps=10,
    window_size=None,
    video_tag="dynamic-query",
):
    """Two-panel (GT | PRED) video of query-point static/dynamic labels.

    GT positions (projected GT query tracks) are drawn on every frame, colored
    red=dynamic / blue=static. Left panel colors by the per-window motion GT used
    for supervision; right panel colors by the model's predicted label. Both
    labels are per-window (computed over the fixed ``window_size`` tiling), so a
    point can flip static/dynamic across windows within the same clip. This
    isolates classification quality (positions are identical across panels).

    ``video_tag`` controls the subdirectory under ``visualization_{dataset_name}``
    (e.g. ``dynamic-query`` or ``dynamic-dense_sample``).
    """
    if preds.query_dynamic_logits is None:
        return

    gt_sample = sample.with_annot_mode("gt")
    trajs_3d = gt_sample.trajs_3d  # (B, T, N, 3)
    intr = gt_sample.intrinsics    # (B, T, 3, 3)
    extr = gt_sample.extrinsics    # (B, T, 4, 4)
    valids = sample.valids         # (B, T, N)
    visibs = sample.visibs         # (B, T, N)
    B, T, N, _ = trajs_3d.shape

    pred_logits = preds.query_dynamic_logits
    if pred_logits.dim() == 2:  # (B, N) -> broadcast to per-frame (legacy models)
        pred_logits = pred_logits[:, None].expand(-1, T, -1)
    if pred_logits.shape[-1] != N or pred_logits.shape[1] != T:
        return
    pred_dynamic = pred_logits.sigmoid() > dynamic_threshold  # (B, T, N)

    tracks_2d = batch_project(
        trajs_3d,
        repeat(intr, "b t i j -> b t n i j", n=N),
        repeat(extr, "b t i j -> b t n i j", n=N),
    )  # (B, T, N, 2)

    # Prefer the dataset's per-sample metric threshold + norm factor so the GT
    # panel matches the metric-space label used for training/eval.
    gt_norm_factor = getattr(gt_sample, "gt_norm_factor", None)
    sample_thr = getattr(gt_sample, "dynamic_filter_threshold", None)

    save_dir = f"{output_dir}/visualization_{dataset_name}"
    for b in range(B):
        thr_b = dynamic_filter_threshold
        if sample_thr is not None:
            thr_b = sample_thr[b] if sample_thr.ndim > 0 else sample_thr
        nf_b = None
        if gt_norm_factor is not None:
            nf_b = gt_norm_factor[b] if gt_norm_factor.ndim > 0 else gt_norm_factor
        gt_dynamic = _window_dynamic_per_frame_np(
            trajs_3d[b], thr_b, window_size, norm_factor=nf_b,
            valids=None if valids is None else valids[b],
        )  # (T, N)
        rgbs = (sample.rgbs[b].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        H, W = rgbs.shape[1:3]
        radius = max(2, min(H, W) // 120)
        gt_lab = gt_dynamic.cpu().numpy()       # (T, N)
        pr_lab = pred_dynamic[b].cpu().numpy()  # (T, N)
        pts = tracks_2d[b].cpu().numpy()  # (T, N, 2)
        draw_mask = (valids[b] & visibs[b]).cpu().numpy()  # (T, N)

        frames = []
        for t in range(T):
            gt_panel = rgbs[t].copy()
            pr_panel = rgbs[t].copy()
            for n in range(N):
                if not draw_mask[t, n]:
                    continue
                x, y = int(round(pts[t, n, 0])), int(round(pts[t, n, 1]))
                if not (0 <= x < W and 0 <= y < H):
                    continue
                cv2.circle(gt_panel, (x, y), radius, _sd_label_color(bool(gt_lab[t, n])), -1)
                cv2.circle(pr_panel, (x, y), radius, _sd_label_color(bool(pr_lab[t, n])), -1)
            frames.append(_sd_stack_panels(gt_panel, pr_panel))
        frames = np.stack(frames, axis=0)
        _sd_write_and_log_video(
            frames, save_dir, f"{dataset_name}-{video_tag}/viz-{batch_idx}-{b}",
            log_wandb, writer, step_id, fps,
        )


def _per_frame_dense_mask(mask, Tn):
    """Reduce GT dense (B, T, H, W[, K]) to (B, Tn, H, W) bool.

    5D tensors whose last axis equals T are treated as source→target and reduced
    to the source=target diagonal (same convention as predicted logits). Any
    other trailing axis is OR-reduced.
    """
    if mask is None or not torch.is_tensor(mask):
        return None
    m = mask
    if m.dim() == 5:
        if m.shape[-1] == m.shape[1]:
            m = torch.stack([m[:, t, :, :, t] for t in range(m.shape[1])], dim=1)
        else:
            m = m.any(dim=-1)
    if m.dim() != 4:
        return None
    return m[:, :Tn].bool()


@torch.no_grad()
def generate_dense_dynamic_pred_videos(
    sample,
    preds,
    output_dir,
    step_id,
    batch_idx,
    dataset_name,
    dynamic_threshold=0.5,
    log_wandb=False,
    writer=None,
    fps=10,
    alpha=0.5,
    source_mask=None,
    include_gt=False,
):
    """Full-grid overlay of dense static/dynamic logits.

    Default is PRED-only: ``sigmoid(dense_dynamic_logits) > threshold`` on all
    dense pixels (red=dynamic, blue=static). With ``include_gt=True`` and
    ``sample.dense_dynamic`` present, writes a side-by-side GT | PRED mask
    video instead.

    ``source_mask`` is the model's ``dense_norm_mask``, (B, T, H, W) or
    (T, H, W) bool at dense-grid resolution. Cells outside it are left
    untinted: the model displaces them to a sentinel coordinate rather than
    dropping them, so it still emits a logit there, but that logit is
    meaningless (all such cells in a frame collapse into one voxel and share
    its prediction).
    """
    if preds.dense_dynamic_logits is None:
        return []

    # Overlay uses source=target diagonal of the (B, T, H, W, T) track logits.
    pred_logits = dense_dynamic_logits_diagonal(preds.dense_dynamic_logits)  # (B, T, H, W)
    Tn = min(pred_logits.shape[1], sample.rgbs.shape[1])
    pred_dyn = pred_logits[:, :Tn].sigmoid() > dynamic_threshold
    # Pixels with finite logits are shown, minus any the caller masks out.
    valid = torch.isfinite(pred_logits[:, :Tn])
    if source_mask is not None:
        sm = torch.as_tensor(source_mask).detach().bool().to(valid.device)
        if sm.ndim == 3:
            sm = sm[None]
        if sm.ndim != 4 or sm.shape[-2:] != valid.shape[-2:] or sm.shape[1] < Tn:
            raise ValueError(
                f"source_mask must be (B, T, H, W) or (T, H, W) with T >= {Tn} "
                f"and H, W == {tuple(valid.shape[-2:])}; got {tuple(sm.shape)}"
            )
        valid = valid & sm[:, :Tn]

    gt_dyn = gt_valid = None
    if include_gt:
        gt_dyn = _per_frame_dense_mask(getattr(sample, "dense_dynamic", None), Tn)
        gt_valid = _per_frame_dense_mask(getattr(sample, "dense_valids", None), Tn)
        if gt_dyn is not None:
            if gt_valid is None:
                gt_valid = torch.ones_like(gt_dyn)
            else:
                gt_valid = gt_valid.to(gt_dyn.device)
            if source_mask is not None:
                sm = torch.as_tensor(source_mask).detach().bool().to(gt_dyn.device)
                if sm.ndim == 3:
                    sm = sm[None]
                if sm.ndim == 4 and sm.shape[-2:] == gt_dyn.shape[-2:] and sm.shape[1] >= Tn:
                    gt_valid = gt_valid & sm[:, :Tn]
            if gt_dyn.shape[-2:] != pred_dyn.shape[-2:]:
                # Nearest-resize handled per-frame in _sd_dense_overlay; keep as-is.
                pass

    tag = "dynamic-dense_gt_pred" if gt_dyn is not None else "dynamic-dense_pred"
    B = sample.rgbs.shape[0]
    save_dir = f"{output_dir}/visualization_{dataset_name}"
    save_paths = []
    for b in range(B):
        rgbs = (sample.rgbs[b, :Tn].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        pr_b = pred_dyn[b].cpu().numpy()
        vl_b = valid[b].cpu().numpy()
        gt_b = gt_dyn[b].cpu().numpy() if gt_dyn is not None else None
        gv_b = gt_valid[b].cpu().numpy() if gt_valid is not None else None

        frames = []
        for t in range(Tn):
            pr_panel = _sd_dense_overlay(rgbs[t], pr_b[t], vl_b[t], alpha)
            if gt_b is None:
                panel = np.ascontiguousarray(pr_panel)
            else:
                gt_panel = _sd_dense_overlay(rgbs[t], gt_b[t], gv_b[t], alpha)
                panel = _sd_stack_panels(gt_panel, pr_panel)
            frames.append(panel)
        frames = np.stack(frames, axis=0)
        save_paths.append(
            _sd_write_and_log_video(
                frames, save_dir, f"{dataset_name}-{tag}/viz-{batch_idx}-{b}",
                log_wandb, writer, step_id, fps,
            )
        )
    return save_paths


def visualize_pcs(
    pc, color, voxel_size=None, legend='pc'
):
    """
        pc: (N, 3)
        color: (N, 3), in [0, 255]
        voxel_size: float or None
    """
    from ..datasets.utils.geometry import voxelization
    from torch_scatter import scatter_mean
    
    if voxel_size:
        p2v = voxelization(
            pc[None],
            voxel_size,
        ).squeeze(0)  # (N,)
        pc = scatter_mean(
            pc, p2v, dim=0
        )
        color = scatter_mean(
            color, p2v, dim=0
        )
    
    viz_pc = torch.cat([
                pc.cpu(),
                color.cpu(),
            ], dim=-1).numpy()
    wandb.log({
        legend: wandb.Object3D(
            viz_pc,
        )
    })
    return viz_pc


_INVALID_XYZ = 900.0


def visualize_rgbd_pc(
    rgbs,
    depths,
    intrsinsics,
    extrinsics,
    shape=None,
    voxel_size=None,
    legend='rgbd_pc',
    valid_mask=None,
):
    """
        rgbs: (B, T, H, W, 3) in [0, 255]
        depths: (B, T, H, W)
        intrsinsics: (B, T, 3, 3)
        extrinsics: (B, T, 4, 4)
        shape: (new_h, new_w) or None
        voxel_size: float or None
        valid_mask: (B, T, H, W) or None
        legend: str
    """
    from .common_utils import batch_unproject
    pcd = batch_unproject(
        depths,
        intrsinsics,
        extrinsics,
    )  # (B, T, 3, H, W)
    pcd = rearrange(pcd, "b t c h w -> b t h w c")
    if shape is not None:
        rgbs = F.interpolate(
            rearrange(rgbs, "b t h w c -> (b t) c h w"),
            size=shape,
            mode="bilinear",
            align_corners=False,
        )
        rgbs = rearrange(
            rgbs, "(b t) c h w -> b t h w c", b=rgbs.shape[0] // depths.shape[1]
        )
        pcd = F.interpolate(
            rearrange(pcd, "b t h w c -> (b t) c h w"),
            size=shape,
            mode="nearest",
        )
        pcd = rearrange(
            pcd, "(b t) c h w -> b t h w c", b=pcd.shape[0] // depths.shape[1]
        )

    viz_pcs = []
    for b in range(rgbs.shape[0]):
        pc = rearrange(pcd[b], "t h w c -> (t h w) c")
        color = rearrange(rgbs[b], "t h w c -> (t h w) c")

        if valid_mask is not None:
            mask = rearrange(valid_mask[b], "t h w -> (t h w)")
            pc = pc[mask]
            color = color[mask]

        viz_pc = visualize_pcs(
            pc,
            color,
            voxel_size=voxel_size,
            legend=f"{legend}_{b}",
        )
        viz_pcs.append(viz_pc)
    return viz_pcs
    

@torch.no_grad()
def visualize_multiview_noisy_points(sample, step_id=None, seq_name=None, num_vis_points=5):
    """Pick the noisiest points in frame 0 (est vs GT) and project them into
    every frame as red (est) / green (GT) circles.  Logs to wandb if available.

    Args:
        sample: SliceData with gt/est depths, intrinsics, extrinsics, rgbs.
        step_id: training step (used for wandb caption). None to skip logging.
        seq_name: sequence name (used for wandb caption).
        num_vis_points: how many of the noisiest points to show.

    Returns:
        multiview_img: np.ndarray (H, T*W, 3) uint8 – annotated image strip.
        noise_stats: dict with max_noise, median_noise, topk_noise values.
    """
    from ..datasets.utils.geometry import project
    from .common_utils import batch_unproject

    B, T, C, H, W = sample.rgbs.shape

    # unproject GT and est depths (full res) to world-space 3D
    gt_pcds = batch_unproject(
        sample.gt_depths, sample.gt_intrinsics, sample.gt_extrinsics
    )  # B, T, 3, H, W
    est_pcds = batch_unproject(
        sample.est_depths, sample.est_intrinsics, sample.est_extrinsics
    )  # B, T, 3, H, W

    # scale est to GT if needed
    if not sample.same_scale and sample.pd_to_gt_scale is not None:
        est_pcds = est_pcds * sample.pd_to_gt_scale[:, None, None, None, None]

    gt_valid = sample.dense_valids
    # gt_valid = sample.gt_depths[:, 0] > 0   # B, H, W
    # est_valid = sample.est_depths[:, 0] > 0  # B, H, W
    both_valid = gt_valid[:, 0]
    both_valid = F.interpolate(both_valid.permute(0, 3, 1, 2).float(), size=(H, W), mode="nearest-exact").bool().permute(0, 2, 3, 1)

    # per-pixel noise in frame 0
    gt_pts_0 = rearrange(gt_pcds[:, 0], "b c h w -> b (h w) c")   # B, HW, 3
    est_pts_0 = rearrange(est_pcds[:, 0], "b c h w -> b (h w) c") # B, HW, 3
    noise = (est_pts_0 - gt_pts_0).norm(dim=-1)  # B, HW
    noise[~both_valid.reshape(B, -1)] = -1.0  # mask invalid

    _, topk_idx = noise[0].topk(num_vis_points * 50)
    
    # randomly sample num_vis_points from topk_idx
    topk_idx = topk_idx[torch.randperm(topk_idx.shape[0])[:num_vis_points]]

    noisy_est_3d = est_pts_0[0, topk_idx]  # num_vis_points, 3
    noisy_gt_3d = gt_pts_0[0, topk_idx]    # num_vis_points, 3

    # project into every frame
    image_size = (H, W)
    est_proj_all = []
    gt_proj_all = []
    for t in range(T):
        est_2d = project(
            noisy_est_3d.unsqueeze(0),
            sample.gt_intrinsics[0:1, t:t+1],
            sample.gt_extrinsics[0:1, t:t+1],
            image_size,
        )[0, :, :2]  # N, 2  (x, y)
        gt_2d = project(
            noisy_gt_3d.unsqueeze(0),
            sample.gt_intrinsics[0:1, t:t+1],
            sample.gt_extrinsics[0:1, t:t+1],
            image_size,
        )[0, :, :2]
        est_proj_all.append(est_2d.cpu())
        gt_proj_all.append(gt_2d.cpu())

    # build annotated images
    rgbs_np = (sample.rgbs[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8).copy()
    radius = max(3, min(H, W) // 80)
    point_colors = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
    ]

    vis_frames = []
    for t in range(T):
        img = rgbs_np[t].copy()
        for pi in range(num_vis_points):
            color = point_colors[pi % len(point_colors)]
            ex, ey = int(est_proj_all[t][pi, 0].item()), int(est_proj_all[t][pi, 1].item())
            gx, gy = int(gt_proj_all[t][pi, 0].item()), int(gt_proj_all[t][pi, 1].item())
            if 0 <= ex < W and 0 <= ey < H:
                cv2.circle(img, (ex, ey), radius, (0, 0, 255), -1)       # red filled
                cv2.circle(img, (ex, ey), radius, color, 2)               # colored ring
            if 0 <= gx < W and 0 <= gy < H:
                cv2.circle(img, (gx, gy), radius, (0, 255, 0), -1)       # green filled
                cv2.circle(img, (gx, gy), radius, color, 2)               # colored ring
            if 0 <= ex < W and 0 <= ey < H and 0 <= gx < W and 0 <= gy < H:
                cv2.line(img, (ex, ey), (gx, gy), color, 1)
        vis_frames.append(img)

    multiview_img = np.concatenate(vis_frames, axis=1)  # H x (T*W) x 3

    valid_noise = noise[0][noise[0] >= 0]
    noise_stats = {
        "max_noise": noise[0, topk_idx].max().item(),
        "median_noise": valid_noise.median().item() if valid_noise.numel() > 0 else 0.0,
        "topk_noise": noise[0, topk_idx].cpu().tolist(),
    }

    if step_id is not None:
        wandb.log({
            "exploratory/multiview_noisy_points": wandb.Image(
                multiview_img,
                caption=f"step={step_id} seq={seq_name} | red=est, green=GT, lines=est→GT  "
                        f"top-{num_vis_points} noisiest pts from frame-0  "
                        f"(max noise={noise_stats['max_noise']:.4f})",
            ),
            "step": step_id,
        })

    return multiview_img, noise_stats


@torch.no_grad()
def visualize_frame0_reconstruction(sample, step_id=None, seq_name=None):
    """Warp frame-0 point cloud into all other frames using GT and est depths/poses.

    Uses downsampled depths for tractable reconstruction.  Assumes a static scene
    (frame-0 3D positions are re-used for every target frame).  Logs to wandb if
    step_id is provided.

    Args:
        sample: SliceData with gt/est depths_downsampled, intrinsics, extrinsics, rgbs.
        step_id: training step (used for wandb caption). None to skip logging.
        seq_name: sequence name (used for wandb caption).

    Returns:
        recon_grid: np.ndarray (3*Hd, T*Wd, 3) uint8 – three rows:
                    original | GT recon | est recon.  None if data unavailable.
    """
    from .common_utils import batch_unproject
    from .warp_utils import warp_2d

    if sample.gt_depths_downsampled is None:
        return None

    B, T, C, H, W = sample.rgbs.shape
    Hd = sample.gt_depths_downsampled.shape[-2]
    Wd = sample.gt_depths_downsampled.shape[-1]
    intr_ds = sample.gt_intrinsics_downsampled
    if intr_ds is None:
        return None

    # GT reconstruction
    gt_pcds_ds = batch_unproject(
        sample.gt_depths_downsampled, intr_ds, sample.gt_extrinsics
    )  # B, T, 3, Hd, Wd
    pts_frame0_gt = rearrange(gt_pcds_ds[:, 0], "b c h w -> b (h w) c")
    valid_frame0_gt = (sample.gt_depths_downsampled[:, 0] > 0).reshape(B, -1)

    pts_repeated_gt = pts_frame0_gt.unsqueeze(1).expand(-1, T, -1, -1)  # B,T,N,3

    imgs_ds = F.interpolate(
        sample.rgbs[0], size=(Hd, Wd), mode="bilinear", align_corners=False
    ).unsqueeze(0)  # B, T, C, Hd, Wd

    warped_colors_gt = warp_2d(
        imgs_ds, pts_repeated_gt, sample.gt_extrinsics, intr_ds
    )  # B, N, T, C

    # Est reconstruction
    est_pcds_ds = batch_unproject(
        sample.est_depths_downsampled,
        sample.est_intrinsics_downsampled,
        sample.est_extrinsics,
    )
    if not sample.same_scale and sample.pd_to_gt_scale is not None:
        est_pcds_ds = est_pcds_ds * sample.pd_to_gt_scale[:, None, None, None, None]
    pts_frame0_est = rearrange(est_pcds_ds[:, 0], "b c h w -> b (h w) c")
    valid_frame0_est = (sample.est_depths_downsampled[:, 0] > 0).reshape(B, -1)
    pts_repeated_est = pts_frame0_est.unsqueeze(1).expand(-1, T, -1, -1)

    warped_colors_est = warp_2d(
        imgs_ds, pts_repeated_est, sample.gt_extrinsics, intr_ds
    )  # B, N, T, C

    # Build grid: row0 = original frames, row1 = GT recon, row2 = est recon
    def _to_img_row(warped, valid_mask):
        row = []
        for t in range(T):
            frame = warped[0, :, t, :].clone()  # N, C
            frame[~valid_mask[0]] = 0.0
            frame = frame.reshape(Hd, Wd, C)
            frame = (frame.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            row.append(frame)
        return np.concatenate(row, axis=1)

    orig_row = np.concatenate(
        [(imgs_ds[0, t].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8) for t in range(T)],
        axis=1,
    )
    gt_recon_row = _to_img_row(warped_colors_gt, valid_frame0_gt)
    est_recon_row = _to_img_row(warped_colors_est, valid_frame0_est)
    recon_grid = np.concatenate([orig_row, gt_recon_row, est_recon_row], axis=0)

    if step_id is not None:
        wandb.log({
            "exploratory/frame0_reconstruction": wandb.Image(
                recon_grid,
                caption=f"step={step_id} seq={seq_name} | "
                        f"row0=original, row1=GT recon, row2=est recon  "
                        f"(frame-0 warped to all views, static assumption)",
            ),
            "step": step_id,
        })

    return recon_grid


def depth_to_colormap(depth, vmin=None, vmax=None, cmap=cv2.COLORMAP_TURBO):
    depth = depth.copy()

    # Mask invalid values
    valid = depth > 0

    if vmin is None:
        vmin = depth[valid].min()
    if vmax is None:
        vmax = depth[valid].max()

    depth_norm = np.zeros_like(depth, dtype=np.float32)
    depth_norm[valid] = (depth[valid] - vmin) / (vmax - vmin + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)

    depth_u8 = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cmap)

    # Optional: make invalid pixels black
    depth_color[~valid] = 0

    return depth_color


def visualize_rgbd(rgbs, depths, legend=None):
    """
        rgbs: (T, H, W, 3) in [0, 255]
        depths: (T, H, W)
        legend: str or None
    """
    # use wandb to visualize rgb and depth images in wandb
    for t in range(rgbs.shape[0]):
        rgb = rgbs[t]
        depth = depths[t]
        
        # convert depth to a nice colormap for visualization
        depth = depth_to_colormap(depth)
        
        if legend is not None:
            wandb.log({
                f"{legend}/rgb_{t}": wandb.Image(rgb),
                f"{legend}/depth_{t}": wandb.Image(depth),
            })
    return


import matplotlib.pyplot as plt


def log_depth_diff(depth_est, depth_gt, mask=None, max_diff=None, step=None, log_name="Depth_Analysis"):
    """
    Computes and logs the difference between estimated and GT depth maps to W&B.
    
    Args:
        depth_est (np.ndarray): Estimated depth map (H, W).
        depth_gt (np.ndarray): Ground truth depth map (H, W).
        mask (np.ndarray, optional): Boolean mask for valid GT pixels.
        max_diff (float, optional): Cap the colorbar range for better visualization.
        step (int, optional): Current training/eval step for the W&B timeline.
        log_name (str): The name of the panel in the W&B workspace.
    """
    # Create a mask for valid pixels if not provided
    if mask is None:
        mask = (depth_gt > 0) & (depth_est > 0)
        
    # Calculate the signed difference
    depth_diff = np.zeros_like(depth_gt, dtype=np.float32)
    depth_diff[mask] = depth_est[mask] - depth_gt[mask]
    
    # Calculate error metrics
    mae = np.mean(np.abs(depth_diff[mask]))
    bias = np.mean(depth_diff[mask])
    
    # Setup the plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Ground Truth
    im0 = axes[0].imshow(np.where(mask, depth_gt, np.nan), cmap='viridis')
    axes[0].set_title('Ground Truth Depth')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 2. Estimated Depth
    im1 = axes[1].imshow(np.where(mask, depth_est, np.nan), cmap='viridis')
    axes[1].set_title('Estimated Depth')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Difference Heatmap (Est - GT)
    if max_diff is None:
        # Robust max for colorbar: 95th percentile to ignore outliers
        max_diff = np.percentile(np.abs(depth_diff[mask]), 95)
        
    im2 = axes[2].imshow(np.where(mask, depth_diff, np.nan), 
                         cmap='RdBu_r', 
                         vmin=-max_diff, 
                         vmax=max_diff)
    axes[2].set_title(f'Difference (Est - GT)\nMAE: {mae:.3f}, Bias: {bias:.3f}')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    
    # --- W&B Logging ---
    # Convert the matplotlib figure to a W&B Image
    wandb_image = wandb.Image(fig)
    
    # Log to wandb, handling the optional step parameter
    log_dict = {log_name: wandb_image, 
                f"{log_name}_MAE": mae, 
                f"{log_name}_Bias": bias}
    
    if step is not None:
        wandb.log(log_dict, step=step)
    else:
        wandb.log(log_dict)
        
    # CRITICAL: Close the figure to prevent memory leaks in loops
    plt.close(fig)

    return depth_diff


# ---------------------------------------------------------------------------
# Paper-worthy rerun visualization for dense static/dynamic 3D tracks.
#
# Semantics of the input tensors (one batch item):
#   dense_coords[t_src, h, w, t_tgt, :]  = predicted 3D world position at target
#                                           frame t_tgt of the pixel observed
#                                           at (t_src, h, w)
#   dense_visibs[t_src, h, w, t_tgt]     = visibility at t_tgt of the same src
#                                           pixel (float in [0,1] or bool)
#   dense_logits[t_src, h, w, t_tgt]     = static/dynamic motion logit for the
#                                           target frame; dynamic when logits >
#                                           dynamic_thresh
#
# We render:
#   - the current RGB+depth point cloud + camera (the backdrop)
#   - a growing cloud of *static* source points (born at t_src, shown at their
#     static 3D position for every t >= t_src)
#   - a growing cloud of *dynamic* source points animated along their
#     predicted trajectory with a trailing LineStrips3D trail
#
# Two modes:
#   mode="voxel" (default, efficient):
#       voxelize source points by their 3D birth position and keep one
#       representative per voxel (earliest t_src). Uses the same
#       `voxelization` helper the model itself uses, so the viz is "in the
#       p2v space" conceptually.
#   mode="dense":
#       render every source pixel. Higher quality, much heavier.
# ---------------------------------------------------------------------------


@torch.no_grad()
def log_dense_3d_tracks_rerun(
    rgb,
    depth,
    intrinsics,
    extrinsics,
    dense_coords,
    dense_visibs,
    dense_logits,
    dense_source_mask=None,
    *,
    entity: str = "video",
    mode: str = "voxel",
    voxel_size: float = 0.03,
    trail_len: int = 10,
    visib_thresh: float = 0.5,
    dynamic_thresh: float = 0.5,
    dynamic_from_motion: bool = False,
    motion_threshold: float = 0.1,
    dynamic_sticky: bool = False,
    static_radius: float = 0.003,
    dynamic_radius: float = 0.003,
    trail_radius: float = 0.001,
    color_by: str = "source_rgb",
    colormap_name: str = "turbo",
    static_mix: float = 0.6,
    occluded_alpha: float = 0.25,
    log_backdrop: bool = True,
    pcd_lighten: float = 0.0,
    max_points: int = 60000,
    max_dynamic_points: int = 20000,
    scale: float = 1.0,
    window_len: int | None = None,
    originate_before: int | None = None,
    show_frame_number: bool = False,
    trail_dim: float = 1.0,
    trail_alpha: float = 1.0,
):
    """Log a dense static/dynamic 3D track visualization to the active rerun
    recording.

    Args:
        rgb: (T, 3, H, W) in [0,1] — RGB video.
        depth: (T, H, W).
        intrinsics: (T, 3, 3).
        extrinsics: (T, 4, 4).
        dense_coords: (T_src, Hf, Wf, T_tgt, 3) — must have T_src == T_tgt == T.
        dense_visibs: (T_src, Hf, Wf, T_tgt).
        dense_logits: (T_src, Hf, Wf, T_tgt).
        dense_source_mask: optional (T_src, Hf, Wf) bool mask over source pixels.
        entity: root entity under which the tracks are logged.
        mode: "voxel" or "dense".
        voxel_size: used only in "voxel" mode.
        trail_len: number of trailing target frames drawn behind dynamic
            points (trail_len <= 1 disables trails).
        visib_thresh: threshold on `dense_visibs` to call a point visible.
        dynamic_thresh: threshold on `dense_logits` to call a point dynamic.
        dynamic_from_motion: if True, classify static/dynamic from the actual
            per-frame motion of each track (distance from its birth position)
            rather than from the model's predicted `dense_logits`. A point is
            called dynamic at target frame `t` when its displacement from the
            source (birth) position exceeds `motion_threshold`.
        motion_threshold: distance (in the same units as `dense_coords`, after
            `scale`) above which a point is considered dynamic when
            `dynamic_from_motion` is True.
        dynamic_sticky: if True, once a track is dynamic at any target frame
            it stays dynamic for all later frames (cummax along time).
        static_radius / dynamic_radius / trail_radius: rerun point/line radii.
        color_by: ``"source_rgb"`` to color from source-frame RGB at the
            source pixel; ``"source_time"`` for bremm colors by birth ``(x, y)``;
            ``"rainbow"`` for CoTracker ``gist_rainbow`` by birth y (same as
            the wandb 2D overlay).
        colormap_name: unused for current modes (kept for API compatibility).
        static_mix: in [0,1], amount of desaturation applied to static points
            when ``color_by`` is not ``"source_rgb"``
            (1 -> fully gray, 0 -> full colormap color).
        occluded_alpha: alpha multiplier for occluded points.
        log_backdrop: if True, also log the RGB+depth point cloud + camera
            pose + image stream (the "video" scaffolding).
        pcd_lighten: in [0,1], mix the RGBD backdrop colors toward white
            (0 = full RGB, 1 = pure white).
        max_points: max never-dynamic source points to render (random subsample
            when there are more).
        max_dynamic_points: max ever-dynamic source points to render (random
            subsample when there are more). Static and dynamic caps are
            independent.
        scale: multiplicative factor applied to `dense_coords` (to match GT
            scale at eval time).
        window_len: if set, at each output frame `t` only points whose source
            (birth) frame `t_src` lies in the same length-`window_len` block as
            `t` are drawn, i.e. ``(t_src // window_len) == (t // window_len)``.
            Still combined with the usual ``t_src <= t`` (static/dynamic)
            visibility rules.
        originate_before: if set, only tracks with birth frame
            ``t_src < originate_before`` are drawn at every output frame
            (combined with ``window_len`` and ``t_src <= t`` when those apply).
    """
    flat, Hf, Wf, source_mask = _dense_tracks_to_flat(
        dense_coords,
        dense_visibs,
        dense_logits,
        dense_source_mask,
        mode=mode,
        voxel_size=voxel_size,
        scale=scale,
    )
    _log_flat_tracks_rerun(
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        flat=flat,
        grid_size=(Hf, Wf),
        dense_source_mask=source_mask,
        entity=entity,
        trail_len=trail_len,
        visib_thresh=visib_thresh,
        dynamic_thresh=dynamic_thresh,
        dynamic_from_motion=dynamic_from_motion,
        motion_threshold=motion_threshold,
        dynamic_sticky=dynamic_sticky,
        static_radius=static_radius,
        dynamic_radius=dynamic_radius,
        trail_radius=trail_radius,
        color_by=color_by,
        static_mix=static_mix,
        occluded_alpha=occluded_alpha,
        log_backdrop=log_backdrop,
        pcd_lighten=pcd_lighten,
        max_points=max_points,
        max_dynamic_points=max_dynamic_points,
        window_len=window_len,
        originate_before=originate_before,
        show_frame_number=show_frame_number,
        trail_dim=trail_dim,
        trail_alpha=trail_alpha,
    )


@torch.no_grad()
def _dense_tracks_to_flat(
    dense_coords,
    dense_visibs,
    dense_logits,
    dense_source_mask=None,
    *,
    mode: str = "voxel",
    voxel_size: float = 0.03,
    scale: float = 1.0,
):
    """Collapse a dense (T_src, Hf, Wf, T_tgt, ...) track tensor into flat tracks.

    This is the half of the dense visualizers that only reshapes and subsamples;
    the rendering half takes the flat form and is shared with the compressed
    long-video path (see ``extract_compressed_tracks``).

    Returns ``(CompressedTracks, Hf, Wf, dense_source_mask)`` with raw (pre-sigmoid)
    visibility and dynamic logits.
    """
    assert mode in ("voxel", "dense"), f"unknown mode: {mode}"

    dense_coords = torch.as_tensor(dense_coords).detach().float().cpu() * float(scale)
    dense_visibs = torch.as_tensor(dense_visibs).detach().float().cpu()
    dense_logits = torch.as_tensor(dense_logits).detach().float().cpu()
    T_src, Hf, Wf, T_tgt, _ = dense_coords.shape
    assert T_src == T_tgt, (
        f"expected square (T, T) dense tracks, got T_src={T_src}, T_tgt={T_tgt}"
    )
    T = T_src
    if tuple(dense_logits.shape) != (T, Hf, Wf, T):
        raise ValueError(
            "dense_logits must have shape (T_src, Hf, Wf, T_tgt) for rerun "
            f"visualization, got {tuple(dense_logits.shape)}"
        )
    if dense_source_mask is not None:
        dense_source_mask = torch.as_tensor(dense_source_mask).detach().bool().cpu()
        if dense_source_mask.shape != (T, Hf, Wf):
            raise ValueError(
                f"dense_source_mask must have shape {(T, Hf, Wf)}, "
                f"got {tuple(dense_source_mask.shape)}"
            )

    # Flatten source pixels to (N, T, 3) and build per-point metadata.
    tracks = dense_coords.reshape(-1, T, 3)
    visibs = dense_visibs.reshape(-1, T)
    dyn_logits = dense_logits.reshape(-1, T)
    src_time = (
        torch.arange(T).view(T, 1, 1).expand(T, Hf, Wf).reshape(-1)
    )  # (N,) long
    src_y = torch.arange(Hf).view(1, Hf, 1).expand(T, Hf, Wf).reshape(-1)  # (N,) long
    src_x = torch.arange(Wf).view(1, 1, Wf).expand(T, Hf, Wf).reshape(-1)  # (N,) long
    if dense_source_mask is not None:
        keep = dense_source_mask.reshape(-1)
        tracks = tracks[keep]
        visibs = visibs[keep]
        dyn_logits = dyn_logits[keep]
        src_time = src_time[keep]
        src_y = src_y[keep]
        src_x = src_x[keep]

    # Birth position = trajectory at t_src. For static points this equals the
    # model's static scene position; for dynamic points it's where they were
    # first observed.
    birth_pos = tracks.gather(
        1, src_time.view(-1, 1, 1).expand(-1, 1, 3)
    ).squeeze(1)  # (N, 3)

    if mode == "voxel" and tracks.shape[0] > 0:
        from ..datasets.utils.geometry import voxelization

        p2v = voxelization(birth_pos[None].contiguous(), voxel_size).squeeze(0)
        # Choose one representative per voxel: the earliest src_time.
        # Sort by (voxel_id, src_time), then keep the first occurrence of each
        # voxel_id. This avoids a hard dependency on torch_scatter.
        sort_keys = p2v.to(torch.int64) * (T + 1) + src_time.to(torch.int64)
        sort_idx = torch.argsort(sort_keys)
        p2v_sorted = p2v[sort_idx]
        first_mask = torch.ones_like(p2v_sorted, dtype=torch.bool)
        first_mask[1:] = p2v_sorted[1:] != p2v_sorted[:-1]
        chosen = sort_idx[first_mask]
        tracks = tracks[chosen]
        visibs = visibs[chosen]
        dyn_logits = dyn_logits[chosen]
        src_time = src_time[chosen]
        src_y = src_y[chosen]
        src_x = src_x[chosen]

    flat = CompressedTracks(
        coords=tracks,
        visibs=visibs,
        dyn_logits=dyn_logits,
        src_time=src_time,
        src_y=src_y,
        src_x=src_x,
    )
    return flat, Hf, Wf, dense_source_mask


@torch.no_grad()
def _log_flat_tracks_rerun(
    rgb,
    depth,
    intrinsics,
    extrinsics,
    flat: CompressedTracks,
    grid_size,
    dense_source_mask=None,
    *,
    entity: str = "video",
    trail_len: int = 10,
    visib_thresh: float = 0.5,
    dynamic_thresh: float = 0.5,
    dynamic_from_motion: bool = False,
    motion_threshold: float = 0.1,
    dynamic_sticky: bool = False,
    static_radius: float = 0.003,
    dynamic_radius: float = 0.003,
    trail_radius: float = 0.001,
    color_by: str = "source_rgb",
    static_mix: float = 0.6,
    occluded_alpha: float = 0.25,
    log_backdrop: bool = True,
    pcd_lighten: float = 0.0,
    max_points: int = 60000,
    max_dynamic_points: int = 20000,
    window_len: int | None = None,
    originate_before: int | None = None,
    show_frame_number: bool = False,
    trail_dim: float = 1.0,
    trail_alpha: float = 1.0,
):
    """Render a flat track set to rerun as a static/dynamic 3D animation.

    Shared by the dense path (via ``_dense_tracks_to_flat``) and the long-video
    path (via ``extract_compressed_tracks``), so both produce the same look.

    ``grid_size`` is the ``(Hf, Wf)`` of the dense feature grid the birth indices
    in ``flat.src_y`` / ``flat.src_x`` refer to.

    ``trail_dim`` multiplies trail RGB (point heads stay full brightness).
    ``trail_alpha`` optionally appends an alpha channel to trail colors.
    """
    import rerun
    from . import rerun_visualizer as _rv

    assert color_by in ("source_rgb", "source_time", "rainbow"), (
        f"unknown color_by: {color_by}"
    )

    Hf, Wf = grid_size
    rgb = torch.as_tensor(rgb).detach().float().cpu()
    tracks = flat.coords
    T = tracks.shape[1]
    visibs = flat.visibs.sigmoid() > visib_thresh
    dyn_logits = flat.dyn_logits.sigmoid()
    src_time, src_y, src_x = flat.src_time, flat.src_y, flat.src_x

    # Cap ever-dynamic and never-dynamic tracks independently.
    # `max_dynamic_points` caps points dynamic in any target frame;
    # `max_points` caps points that are never dynamic.
    dyn_by_frame = dyn_logits > dynamic_thresh
    n_logit_ever = int(dyn_by_frame.any(dim=1).sum().item())
    n_logit_slots = int(dyn_by_frame.sum().item())
    if dynamic_from_motion:
        # AND-suppress the per-target logit label with the per-window bbox
        # extent (same metric as the loss / the 2D overlay path). This removes
        # points the model flags dynamic but that do not actually move within
        # their window (e.g. window-boundary false-positive flips).
        if window_len is None or window_len < 1:
            raise ValueError(
                "window_len must be set (>= 1) when dynamic_from_motion=True, "
                f"got {window_len}"
            )
        coords_tnc = tracks.permute(1, 0, 2).contiguous()  # (T, N, 3)
        # Restrict the extent to written entries: sentinel positions sit far
        # outside the scene, so an unmasked bbox marks every track dynamic.
        motion_dyn = _window_dynamic_per_frame_np(
            coords_tnc,
            float(motion_threshold),
            window_len,
            valids=(flat.visibs > TRACK_VALID_THRESH).permute(1, 0),
        )  # (T, N) bool
        dyn_by_frame = dyn_by_frame & motion_dyn.permute(1, 0)
    if dynamic_sticky:
        dyn_by_frame = _sticky_dynamic_mask(dyn_by_frame, time_dim=1)
    ever_dyn = dyn_by_frame.any(dim=1)
    dyn_idx = torch.nonzero(ever_dyn, as_tuple=False).view(-1)
    stat_idx = torch.nonzero(~ever_dyn, as_tuple=False).view(-1)
    n_dyn = dyn_idx.numel()
    n_stat = stat_idx.numel()
    n_motion_drop_tracks = n_logit_ever - n_dyn
    n_motion_drop_slots = n_logit_slots - int(dyn_by_frame.sum().item())
    n_dyn_sub_drop = max(0, n_dyn - max_dynamic_points)
    n_stat_sub_drop = max(0, n_stat - max_points)
    logger.info(
        "rerun 3D dynamic trails: logit ever-dyn=%d (%d frame-slots); "
        "motion-gate dropped %d tracks / %d frame-slots; "
        "subsample dropped %d/%d ever-dyn (cap %d) and %d/%d never-dyn (cap %d); "
        "kept %d dynamic",
        n_logit_ever,
        n_logit_slots,
        n_motion_drop_tracks,
        n_motion_drop_slots,
        n_dyn_sub_drop,
        n_dyn,
        max_dynamic_points,
        n_stat_sub_drop,
        n_stat,
        max_points,
        min(n_dyn, max_dynamic_points),
    )
    need_subsample = False
    g = torch.Generator().manual_seed(0)
    if n_dyn > max_dynamic_points:
        dyn_idx = dyn_idx[torch.randperm(n_dyn, generator=g)[:max_dynamic_points]]
        need_subsample = True
    if n_stat > max_points:
        stat_idx = stat_idx[torch.randperm(n_stat, generator=g)[:max_points]]
        need_subsample = True
    if need_subsample:
        perm = torch.cat([dyn_idx, stat_idx])
        perm = perm[torch.randperm(perm.numel(), generator=g)]
        tracks = tracks[perm]
        visibs = visibs[perm]
        dyn_by_frame = dyn_by_frame[perm]
        src_time = src_time[perm]
        src_y = src_y[perm]
        src_x = src_x[perm]

    tracks_np = tracks.numpy()                # (N, T, 3)
    visibs_np = visibs.numpy()                # (N, T) bool
    dyn_np = dyn_by_frame.numpy()             # (N, T) bool
    src_np = src_time.numpy().astype(np.int64)

    if rgb.ndim != 4 or rgb.shape[0] != T or rgb.shape[1] != 3:
        raise ValueError(
            f"rgb must have shape (T, 3, H, W) with T={T}, got {tuple(rgb.shape)}"
        )

    if color_by == "source_rgb":
        _, _, Hr, Wr = rgb.shape
        # Map source-grid (Hf, Wf) indices to RGB-grid (Hr, Wr) via nearest-neighbor.
        src_y_img = torch.round(
            src_y.float() * (float(Hr - 1) / float(max(Hf - 1, 1)))
        ).long().clamp_(0, Hr - 1)
        src_x_img = torch.round(
            src_x.float() * (float(Wr - 1) / float(max(Wf - 1, 1)))
        ).long().clamp_(0, Wr - 1)
        base_colors_t = rgb[src_time, :, src_y_img, src_x_img]
        if base_colors_t.numel() > 0 and float(base_colors_t.max()) <= 1.0 + 1e-6:
            base_colors_t = base_colors_t * 255.0
        base_colors = base_colors_t.clamp(0.0, 255.0).to(torch.uint8).numpy()
    else:
        xy0 = np.stack([src_x.numpy().astype(np.float64),
                        src_y.numpy().astype(np.float64)], axis=-1)
        if color_by == "rainbow":
            base_colors = get_cotracker_rainbow_colors(xy0, Hf, Wf)
        else:
            base_colors = get_2d_colors(xy0, Hf, Wf)

    if color_by == "source_rgb":
        static_colors = base_colors
    else:
        gray = np.array([160, 160, 160], dtype=np.float32)
        static_colors = (
            (1.0 - static_mix) * base_colors.astype(np.float32)
            + static_mix * gray
        ).clip(0, 255).astype(np.uint8)

    dim = float(np.clip(trail_dim, 0.0, 1.0))
    trail_colors = np.clip(
        base_colors.astype(np.float32) * dim, 0, 255
    ).astype(np.uint8)
    if trail_alpha < 1.0:
        a = np.full(
            (trail_colors.shape[0], 1),
            int(round(255 * float(np.clip(trail_alpha, 0.0, 1.0)))),
            dtype=np.uint8,
        )
        trail_colors = np.concatenate([trail_colors, a], axis=1)

    # Optional backdrop (RGB+depth pcd + camera + pinhole + image stream).
    rerun.log("/", rerun.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    if log_backdrop:
        _rv.log_video(
            entity_name=entity,
            rgb=rgb,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            depth=depth,
            show_frame_number=show_frame_number,
            dense_source_mask=dense_source_mask,
            pcd_lighten=pcd_lighten,
        )

    static_path = f"/{entity}/world/tracks/static"
    dynamic_path = f"/{entity}/world/tracks/dynamic"
    trails_path = f"/{entity}/world/tracks/dynamic_trails"

    def _apply_occlusion(rgb_u8: np.ndarray, visible: np.ndarray) -> np.ndarray:
        """Return (N, 4) uint8 RGBA; occluded points dimmed + low alpha."""
        rgba = np.concatenate(
            [rgb_u8, np.full((rgb_u8.shape[0], 1), 255, dtype=np.uint8)],
            axis=1,
        ).astype(np.float32)
        # if (~visible).any():
        #     rgba[~visible, :3] *= 0.45
        #     rgba[~visible, 3] = 255.0 * occluded_alpha
        return rgba.clip(0, 255).astype(np.uint8)

    if window_len is not None:
        if window_len < 1:
            raise ValueError(f"window_len must be >= 1, got {window_len}")
    if originate_before is not None and originate_before < 0:
        raise ValueError(
            f"originate_before must be >= 0 when set, got {originate_before}"
        )

    for t in range(T):
        rerun.set_time("frameid", sequence=t)
        # wk = t // window_len if window_len is not None else None

        point_active = src_np <= t
        # if wk is not None:
        #     point_active = point_active & (src_np // window_len == wk)
        if originate_before is not None:
            point_active = point_active & (src_np < originate_before)

        # --- Static cloud: points not classified dynamic for this target frame.
        sm = point_active & ~dyn_np[:, t]
        if sm.any():
            pts = tracks_np[sm, t]                # (Ns_t, 3)
            vis = visibs_np[sm, t]                # (Ns_t,)
            col = _apply_occlusion(static_colors[sm], vis)
            rerun.log(
                static_path,
                rerun.Points3D(pts, colors=col, radii=static_radius),
            )
        else:
            rerun.log(static_path, rerun.Clear(recursive=False))

        # --- Dynamic points at time t.
        dm = point_active & dyn_np[:, t]
        # dm_new = dm & (src_np == t)
        # dm_old = dm & (src_np != t)
        # active_times = src_np[dm]
        if dm.any():
            pts = tracks_np[dm, t]
            vis = visibs_np[dm, t]
            col = _apply_occlusion(base_colors[dm], vis)
            rerun.log(
                dynamic_path,
                rerun.Points3D(pts, colors=col, radii=dynamic_radius),
            )
        else:
            rerun.log(dynamic_path, rerun.Clear(recursive=False))

        # --- Dynamic trails: last trail_len positions per dynamic point.
        if trail_len > 1 and t > 0:
            t0 = max(0, t - trail_len + 1)
            tm = (src_np < t) & dyn_np[:, t]  # need at least one past frame
            # if wk is not None:
            #     tm = tm & (src_np // window_len == wk)
            if originate_before is not None:
                tm = tm & (src_np < originate_before)
            if tm.any():
                tracks_tm = tracks_np[tm]
                d_src_tm = src_np[tm]
                d_col_tm = trail_colors[tm]
                strip_list = []
                strip_cols = []
                for n, t_src in enumerate(d_src_tm):
                    start = max(t0, int(t_src))
                    seg = tracks_tm[n, start : t + 1]
                    if seg.shape[0] >= 2:
                        strip_list.append(seg)
                        strip_cols.append(d_col_tm[n])
                rerun.log(
                    trails_path,
                    rerun.LineStrips3D(
                        strip_list,
                        colors=np.stack(strip_cols, axis=0),
                        radii=trail_radius,
                    ),
                )
            else:
                rerun.log(trails_path, rerun.Clear(recursive=False))
        elif t == 0:
            # Ensure any stale trails from previous runs are cleared.
            rerun.log(trails_path, rerun.Clear(recursive=False))


@torch.no_grad()
def log_rgb_frames_rerun(
    frames,
    *,
    entity: str = "video",
    image_path: str = "dense_tracks_2d_hard",
    intrinsics=None,
):
    """Log a pre-rendered ``(T, H, W, 3)`` uint8 RGB video to the active rerun recording.

    Used to put the hard-dot / trail overlay from ``render_dense_2d_tracks_frames``
    (and the flat long-video equivalent) into the same ``.rrd`` as the soft-splat
    ``log_dense_2d_tracks_video_rerun`` path, under a separate image entity.
    """
    import rerun

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must have shape (T, H, W, 3); got {frames.shape}")
    T, H, W, _ = frames.shape
    overlay_path = f"/{entity}/world/camera/{image_path}"

    K = None
    if intrinsics is not None:
        K = torch.as_tensor(intrinsics).detach().cpu().numpy()
        if K.ndim != 3 or K.shape[0] < T:
            raise ValueError(
                f"intrinsics must have shape (T, 3, 3) with T >= {T}; got {K.shape}"
            )

    for t in range(T):
        rerun.set_time("frameid", sequence=t)
        if K is not None:
            rerun.log(
                overlay_path,
                rerun.Pinhole(
                    image_from_camera=K[t],
                    resolution=[W, H],
                    camera_xyz=rerun.ViewCoordinates.RDF,
                ),
            )
        rerun.log(overlay_path, rerun.Image(np.ascontiguousarray(frames[t])))


@torch.no_grad()
def log_dense_2d_tracks_video_rerun(
    rgb,
    intrinsics,
    extrinsics,
    dense_coords,
    dense_visibs,
    *,
    dense_source_mask=None,
    entity: str = "video",
    visib_thresh: float = 0.5,
    rate: int = 1,
    bkg_opacity: float = 0.5,
    scale: float = 1.0,
    window_len: int | None = None,
    originate_before: int | None = None,
    image_path: str = "dense_tracks_2d",
    colormap_name: str = "turbo",
    color_by: str = "rainbow",
    dense_logits: torch.Tensor | None = None,
    dynamic_thresh: float = 0.5,
    dynamic_from_motion: bool = False,
    motion_threshold: float = 0.1,
    dynamic_sticky: bool = False,
    dynamic_image_path: str | None = None,
    separately_log_2d_dynamic: bool = False,
    trail_len: int = 3,
    trail_radius: float = 0.5,
    seq_len: int = None,
    max_dynamic_points: int = 20000,
    static_lighten: float = 0.65,
    trail_dim: float = 1.0,
    trail_alpha: float = 1.0,
):
    """Project dense 3D tracks to each target image and log a 2D overlay video.

    When ``dense_logits`` is provided with shape ``(T, Hf, Wf, T)`` (same layout
    as ``log_dense_3d_tracks_rerun``), dynamic classification at the current
    target frame (``sigmoid(logit) > dynamic_thresh``) is used for 2D trail
    filtering on the main overlay.

    If ``separately_log_2d_dynamic`` is True, a second overlay is also logged
    under ``dynamic_image_path`` (default ``{image_path}_dynamic``) that draws
    only points classified as dynamic.

    If ``trail_len > 1`` and ``dense_logits`` is provided, 2D line-strip trails
    are logged for dynamic points showing the last ``trail_len`` positions.
    Ever-dynamic tracks used for trails / the optional dynamic overlay are
    capped at ``max_dynamic_points`` (seeded random subsample). The main
    all-visible soft-splat overlay is not capped.

    ``color_by`` controls track colors: ``"rainbow"`` (default, CoTracker
    ``gist_rainbow`` by birth y — same as wandb), ``"source_time"`` (bremm by
    birth xy), or ``"source_rgb"`` (RGB at birth pixel).

    When ``dense_logits`` is set, static points on the main overlay are mixed
    toward white by ``static_lighten`` (same formula as the wandb overlay);
    dynamic points keep full colormap color.

    When ``dynamic_from_motion`` is True, ``window_len`` must be set and the
    logit-based dynamic mask is AND-suppressed with a per-window motion gate:
    a track is only kept dynamic at frames in a window if the L2 extent of its
    xyz bbox over that window (``||amax - amin||``) exceeds ``motion_threshold``.
    Motion never invents dynamic labels; it can only force static.
    """
    dense_coords = torch.as_tensor(dense_coords).detach().float()
    device = dense_coords.device

    T_src, Hf, Wf, T_tgt, _ = dense_coords.shape
    if T_src != T_tgt:
        raise ValueError(
            f"expected square dense tracks for 2D rerun overlay, got "
            f"T_src={T_src}, T_tgt={T_tgt}"
        )
    T = T_src
    dense_visibs = torch.as_tensor(dense_visibs).detach().float().to(device)
    if dense_visibs.shape != (T, Hf, Wf, T):
        raise ValueError(
            f"dense_visibs must have shape {(T, Hf, Wf, T)}, "
            f"got {tuple(dense_visibs.shape)}"
        )
    if dense_logits is not None:
        dense_logits = torch.as_tensor(dense_logits).detach().float().to(device)
        if tuple(dense_logits.shape) != (T, Hf, Wf, T):
            raise ValueError(
                "dense_logits must have shape (T_src, Hf, Wf, T_tgt); "
                f"got {tuple(dense_logits.shape)}"
            )
        dense_logits = dense_logits.reshape(-1, T)

    src_valid = None
    if dense_source_mask is not None:
        dense_source_mask = torch.as_tensor(dense_source_mask).detach().bool().to(device)
        if dense_source_mask.shape != (T, Hf, Wf):
            raise ValueError(
                f"dense_source_mask must have shape {(T, Hf, Wf)}, "
                f"got {tuple(dense_source_mask.shape)}"
            )
        src_valid = dense_source_mask.reshape(-1)

    N = T * Hf * Wf
    flat = CompressedTracks(
        coords=dense_coords.reshape(N, T, 3),
        visibs=dense_visibs.reshape(N, T),
        dyn_logits=(
            dense_logits
            if dense_logits is not None
            else torch.full((N, T), -1000.0, device=device)
        ),
        src_time=torch.arange(T, device=device).view(T, 1, 1).expand(T, Hf, Wf).reshape(-1),
        src_y=torch.arange(Hf, device=device).view(1, Hf, 1).expand(T, Hf, Wf).reshape(-1),
        src_x=torch.arange(Wf, device=device).view(1, 1, Wf).expand(T, Hf, Wf).reshape(-1),
    )
    return _log_flat_2d_tracks_video_rerun(
        rgb,
        intrinsics,
        extrinsics,
        flat,
        grid_size=(Hf, Wf),
        src_valid=src_valid,
        has_logits=dense_logits is not None,
        entity=entity,
        visib_thresh=visib_thresh,
        rate=rate,
        bkg_opacity=bkg_opacity,
        scale=scale,
        window_len=window_len,
        originate_before=originate_before,
        image_path=image_path,
        colormap_name=colormap_name,
        color_by=color_by,
        dynamic_thresh=dynamic_thresh,
        dynamic_from_motion=dynamic_from_motion,
        motion_threshold=motion_threshold,
        dynamic_sticky=dynamic_sticky,
        dynamic_image_path=dynamic_image_path,
        separately_log_2d_dynamic=separately_log_2d_dynamic,
        trail_len=trail_len,
        trail_radius=trail_radius,
        seq_len=seq_len,
        max_dynamic_points=max_dynamic_points,
        static_lighten=static_lighten,
        trail_dim=trail_dim,
        trail_alpha=trail_alpha,
    )


@torch.no_grad()
def _log_flat_2d_tracks_video_rerun(
    rgb,
    intrinsics,
    extrinsics,
    flat: CompressedTracks,
    grid_size,
    *,
    src_valid=None,
    has_logits: bool = True,
    entity: str = "video",
    visib_thresh: float = 0.5,
    rate: int = 1,
    bkg_opacity: float = 0.5,
    scale: float = 1.0,
    window_len: int | None = None,
    originate_before: int | None = None,
    image_path: str = "dense_tracks_2d",
    colormap_name: str = "turbo",
    color_by: str = "rainbow",
    dynamic_thresh: float = 0.5,
    dynamic_from_motion: bool = False,
    motion_threshold: float = 0.1,
    dynamic_sticky: bool = False,
    dynamic_image_path: str | None = None,
    separately_log_2d_dynamic: bool = False,
    trail_len: int = 3,
    trail_radius: float = 0.5,
    seq_len: int = None,
    max_dynamic_points: int = 20000,
    static_lighten: float = 0.65,
    trail_dim: float = 1.0,
    trail_alpha: float = 1.0,
):
    """Log a flat track set as a 2D overlay video to the active rerun recording.

    Shared by the dense path (via ``log_dense_2d_tracks_video_rerun``) and the
    long-video path (via ``extract_compressed_tracks``).

    When ``has_logits`` is True, static points are mixed toward white by
    ``static_lighten``; dynamic points keep full colormap color. Soft-splat
    points are otherwise unchanged. ``LineStrips2D`` trails are multiplied by
    ``trail_dim`` (and optionally given ``trail_alpha``) so they read softer
    against the darkened background.
    """
    import rerun

    assert color_by in ("source_rgb", "source_time", "rainbow"), (
        f"unknown color_by: {color_by}"
    )

    Hf, Wf = grid_size
    dense_coords = flat.coords.float() * float(scale)
    device = dense_coords.device
    N, T, _ = dense_coords.shape

    rgb = torch.as_tensor(rgb).detach().float().to(device)
    intrinsics = torch.as_tensor(intrinsics).detach().float().to(device)
    extrinsics = torch.as_tensor(extrinsics).detach().float().to(device)
    if rgb.ndim != 4 or rgb.shape[0] != T or rgb.shape[1] != 3:
        raise ValueError(
            f"rgb must have shape (T, 3, H, W) with T={T}, got {tuple(rgb.shape)}"
        )

    _, _, H, W = rgb.shape
    rgb_255 = rgb
    if rgb_255.numel() > 0 and float(rgb_255.max()) <= 1.0 + 1e-6:
        rgb_255 = rgb_255 * 255.0
    rgb_255 = rgb_255.clamp(0.0, 255.0)

    tracks_3d = dense_coords.permute(1, 0, 2).contiguous()
    tracks_2d = batch_project(
        tracks_3d,
        intrinsics[:, None].expand(T, N, 3, 3),
        extrinsics[:, None].expand(T, N, 4, 4),
    )

    src_time = flat.src_time.to(device)
    track_idx = torch.arange(N, device=device)
    xy0 = tracks_2d[src_time, track_idx].detach().cpu().numpy().copy()
    # Invalid projections are masked below; keep color lookup from failing on them.
    xy0[:, 0] = np.nan_to_num(xy0[:, 0], nan=0.0, posinf=float(W - 1), neginf=0.0)
    xy0[:, 1] = np.nan_to_num(xy0[:, 1], nan=0.0, posinf=float(H - 1), neginf=0.0)
    xy0[:, 0] = np.clip(xy0[:, 0], 0.0, float(W - 1))
    xy0[:, 1] = np.clip(xy0[:, 1], 0.0, float(H - 1))
    if color_by == "source_rgb":
        src_y_img = torch.round(
            flat.src_y.to(device).float() * (float(H - 1) / float(max(Hf - 1, 1)))
        ).long().clamp_(0, H - 1)
        src_x_img = torch.round(
            flat.src_x.to(device).float() * (float(W - 1) / float(max(Wf - 1, 1)))
        ).long().clamp_(0, W - 1)
        base_colors_t = rgb_255[src_time, :, src_y_img, src_x_img]
        colors = base_colors_t.clamp(0.0, 255.0)
    elif color_by == "rainbow":
        colors = torch.from_numpy(
            get_cotracker_rainbow_colors(xy0, H, W).astype(np.float32)
        ).to(device)
    else:
        colors = torch.from_numpy(
            get_2d_colors(xy0, H, W).astype(np.float32)
        ).to(device)

    visibs = flat.visibs.to(device).permute(1, 0).sigmoid() > visib_thresh
    if src_valid is not None:
        visibs = visibs & src_valid.to(device).view(1, N)
    target_time = torch.arange(T, device=device).view(T, 1)
    active = src_time.view(1, N) <= target_time
    # if window_len is not None:
    #     if window_len < 1:
    #         raise ValueError(f"window_len must be >= 1, got {window_len}")
    #     active = active & (src_time.view(1, N) // window_len == target_time // window_len)
    if originate_before is not None:
        if originate_before < 0:
            raise ValueError(
                f"originate_before must be >= 0 when set, got {originate_before}"
            )
        active = active & (src_time.view(1, N) < originate_before)

    finite = torch.isfinite(tracks_2d).all(dim=-1)
    in_bounds = (
        finite
        & (tracks_2d[..., 0] >= 0)
        & (tracks_2d[..., 0] < W)
        & (tracks_2d[..., 1] >= 0)
        & (tracks_2d[..., 1] < H)
    )
    visibs = visibs & active & in_bounds

    overlay_path = f"/{entity}/world/camera/{image_path}"

    visibs_dynamic = None
    frames_dynamic = None
    overlay_dynamic_path = None
    if has_logits:
        if dynamic_from_motion:
            if window_len is None or window_len < 1:
                raise ValueError(
                    "window_len must be set (>= 1) when dynamic_from_motion=True, "
                    f"got {window_len}"
                )
        dyn_at_tgt = (
            flat.dyn_logits.to(device).permute(1, 0).sigmoid() > float(dynamic_thresh)
        )
        if dynamic_from_motion:
            # AND-suppress with per-window bbox extent (same metric as the loss).
            coords_tnc = tracks_3d
            motion_dyn = _window_dynamic_per_frame_np(
                coords_tnc,
                float(motion_threshold),
                window_len,
                valids=(flat.visibs.to(device) > TRACK_VALID_THRESH).permute(1, 0),
            )
            dyn_at_tgt = dyn_at_tgt & motion_dyn
        if dynamic_sticky:
            dyn_at_tgt = _sticky_dynamic_mask(dyn_at_tgt, time_dim=0)
        # Main overlay uses the full (uncapped) dynamic mask for static lighten.
        visibs_static = visibs & ~dyn_at_tgt
        lighten = float(np.clip(static_lighten, 0.0, 1.0))
        colors_static = colors * (1.0 - lighten) + 255.0 * lighten
        # Two-pass soft-splat: lightened static first, then full-color dynamic.
        frames = draw_pts_gpu(
            rgb_255,
            tracks_2d,
            visibs_static,
            colors_static,
            rate=rate,
            bkg_opacity=bkg_opacity,
        )
        base2 = (
            torch.from_numpy(frames)
            .to(device)
            .float()
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        frames = draw_pts_gpu(
            base2,
            tracks_2d,
            visibs & dyn_at_tgt,
            colors,
            rate=rate,
            bkg_opacity=1.0,
        )
        # Cap ever-dynamic tracks for trails / optional dynamic overlay only;
        # the main all-visible soft-splat overlay stays uncapped.
        ever_dyn = dyn_at_tgt.any(dim=0)
        n_dyn = int(ever_dyn.sum().item())
        dyn_for_trails = dyn_at_tgt
        n_dyn_sub_drop = max(0, n_dyn - max_dynamic_points)
        logger.info(
            "rerun 2D dynamic trails: ever-dyn=%d, subsample dropped %d (cap %d); kept %d",
            n_dyn,
            n_dyn_sub_drop,
            max_dynamic_points,
            min(n_dyn, max_dynamic_points),
        )
        if n_dyn > max_dynamic_points:
            g = torch.Generator().manual_seed(0)
            dyn_idx = torch.nonzero(ever_dyn, as_tuple=False).view(-1)
            keep_idx = dyn_idx[
                torch.randperm(n_dyn, generator=g)[:max_dynamic_points].to(dyn_idx.device)
            ]
            keep_dyn = torch.zeros(N, dtype=torch.bool, device=device)
            keep_dyn[keep_idx] = True
            dyn_for_trails = dyn_at_tgt & keep_dyn.view(1, N)
        visibs_dynamic = visibs & dyn_for_trails
        if separately_log_2d_dynamic:
            frames_dynamic = draw_pts_gpu(
                rgb_255,
                tracks_2d,
                visibs_dynamic,
                colors,
                rate=rate,
                bkg_opacity=bkg_opacity,
            )
            dyn_path = (
                dynamic_image_path
                if dynamic_image_path is not None
                else f"{image_path}_dynamic"
            )
            overlay_dynamic_path = f"/{entity}/world/camera/{dyn_path}"
    elif dynamic_from_motion:
        raise ValueError(
            "dense_logits is required when dynamic_from_motion=True "
            "(motion only AND-suppresses the logit mask)"
        )
    else:
        frames = draw_pts_gpu(
            rgb_255,
            tracks_2d,
            visibs,
            colors,
            rate=rate,
            bkg_opacity=bkg_opacity,
        )

    # Pre-compute numpy arrays for 2D trail construction.
    draw_trails = trail_len > 1 and visibs_dynamic is not None
    if draw_trails:
        tracks_2d_np = tracks_2d.detach().cpu().numpy()  # (T, N, 2)
        src_time_np = src_time.detach().cpu().numpy()    # (N,)
        dim = float(np.clip(trail_dim, 0.0, 1.0))
        trail_colors_np = np.clip(
            colors.detach().cpu().numpy() * dim, 0, 255
        ).astype(np.uint8)
        if trail_alpha < 1.0:
            a = np.full(
                (N, 1),
                int(round(255 * float(np.clip(trail_alpha, 0.0, 1.0)))),
                dtype=np.uint8,
            )
            trail_colors_np = np.concatenate([trail_colors_np, a], axis=1)
        trails_entities = [f"{overlay_path}/trails"]
        if overlay_dynamic_path is not None:
            trails_entities.append(f"{overlay_dynamic_path}/trails")

    for t in range(T):
        rerun.set_time("frameid", sequence=t)
        rerun.log(
            overlay_path,
            rerun.Pinhole(
                image_from_camera=intrinsics[t].detach().cpu().numpy(),
                resolution=[W, H],
                camera_xyz=rerun.ViewCoordinates.RDF,
            ),
        )
        rerun.log(overlay_path, rerun.Image(frames[t]))
        if frames_dynamic is not None and overlay_dynamic_path is not None:
            rerun.log(
                overlay_dynamic_path,
                rerun.Pinhole(
                    image_from_camera=intrinsics[t].detach().cpu().numpy(),
                    resolution=[W, H],
                    camera_xyz=rerun.ViewCoordinates.RDF,
                ),
            )
            rerun.log(overlay_dynamic_path, rerun.Image(frames_dynamic[t]))

        if seq_len is not None:
            if t % seq_len == 0:
                src_time_np[src_time_np < t] = t
        # Log 2D trails for dynamic points on the main overlay (and optional
        # separate dynamic overlay when separately_log_2d_dynamic=True).
        if draw_trails and t > 0:
            t0 = max(0, t - trail_len + 1)
            vis_mask = visibs_dynamic[t].cpu().numpy()
            eligible = vis_mask & (src_time_np < t)
            if eligible.any():
                idxs = np.where(eligible)[0]
                strip_list = []
                strip_cols = []
                for n in idxs:
                    start = max(t0, int(src_time_np[n]))
                    seg = tracks_2d_np[start : t + 1, n]
                    if seg.shape[0] >= 2:
                        strip_list.append(seg)
                        strip_cols.append(trail_colors_np[n])
                if strip_list:
                    strips_arch = rerun.LineStrips2D(
                        strip_list,
                        colors=np.stack(strip_cols, axis=0),
                        radii=trail_radius,
                    )
                    for te in trails_entities:
                        rerun.log(te, strips_arch)
                else:
                    for te in trails_entities:
                        rerun.log(te, rerun.Clear(recursive=False))
            else:
                for te in trails_entities:
                    rerun.log(te, rerun.Clear(recursive=False))
        elif draw_trails and t == 0:
            for te in trails_entities:
                rerun.log(te, rerun.Clear(recursive=False))


@torch.no_grad()
def generate_dense_track_visualizations_with_rerun(
    sample,
    preds,
    output_dir: str,
    step_id,
    batch_idx,
    dataset_name: str,
    *,
    mode: str = "voxel",
    voxel_size: float = 0.03,
    trail_len: int = 10,
    scale: float = 1.0,
    visualize_est: bool = False,
    max_points: int = 60000,
    app_name: str = "DenseTracks",
    window_len: int | None = None,
    originate_before: int | None = None,
    visualize_query_coords: bool = True,
    visualize_dense_2d: bool = True,
    dense_2d_rate: int = 1,
    dense_2d_bkg_opacity: float = 0.5,
    dense_2d_colormap_name: str = "turbo",
    show_frame_number: bool = True,
    seq_len: int = None,
    trail_dim: float = 1.0,
    trail_alpha: float = 1.0,
):
    """End-to-end wrapper: reconstructs dense 3D tracks from `preds` and logs
    a paper-worthy rerun recording per batch item.

    Expects `preds.dense_track_3d` to be populated (i.e. the model ran with
    `dense_tracking_at_eval=True`) and `preds.coords` / `preds.visibs` to be
    present.

    When ``window_len`` / ``originate_before`` are set, see
    ``log_dense_3d_tracks_rerun`` for filtering semantics.

    Writes one `.rrd` per batch item to
      `{output_dir}/visualization_{dataset_name}_dense/{step_id}_{batch_idx}_{i}.rrd`
    """
    from . import rerun_visualizer as visualizer
    from .alltrack3r_utils import reconstruct_dense_tracks_3d

    dense, dense_visibs, _dense_confs, dense_logits = reconstruct_dense_tracks_3d(
        preds,
        return_visibs_confs=True,
        dynamic_logits_by_target=True,
    )
    # st()

    query_coords = getattr(preds, "query_coords", None)
    query_visibs = getattr(preds, "query_visibs", None)
    qp = getattr(sample, "gt_query_point", None)
    dense_norm_mask = getattr(sample, "dense_norm_mask", None)

    def _batch_dense_source_mask(batch_idx: int, target_shape: tuple[int, int, int]):
        if dense_norm_mask is None or dense_norm_mask[0] is None:
            return None
        mask = torch.as_tensor(dense_norm_mask[batch_idx]).detach().bool()
        if mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        if tuple(mask.shape) == target_shape:
            return mask
        if mask.ndim != 3 or mask.shape[0] != target_shape[0]:
            raise ValueError(
                f"dense_norm_mask for batch item {batch_idx} must have shape "
                f"{target_shape} or (T, H, W, 1), got {tuple(mask.shape)}"
            )
        return (
            F.interpolate(
                mask.float().unsqueeze(1),
                size=target_shape[1:],
                mode="nearest",
            )
            .squeeze(1)
            .bool()
        )

    B = dense.shape[0]
    for i in range(B):
        dense_source_mask = _batch_dense_source_mask(i, tuple(dense.shape[1:4]))
        query_birth_steps = (
            None
            if qp is None
            else torch.as_tensor(qp[i, :, 0]).detach().long().reshape(-1)
        )
        visualizer.setup_visualizer(app_name=app_name, serve=False)

        # Attach the file sink BEFORE any rerun.log() calls. rerun.save() does
        # not retroactively capture log entries emitted before the sink was
        # set; doing this after logging produces a ~1 KB header-only .rrd.
        save_path = (
            f"{output_dir}/visualization_{dataset_name}_dense/"
            f"{step_id}_{batch_idx}_{i}_{mode}.rrd"
        )
        visualizer.save_recording(save_path)
        # log_dense_3d_tracks_rerun(
        #     rgb=sample.rgbs[i],
        #     depth=sample.depths[i],
        #     intrinsics=sample.intrinsics[i],
        #     extrinsics=sample.extrinsics[i],
        #     dense_coords=dense[i],
        #     dense_visibs=dense_visibs[i],
        #     dense_logits=dense_logits[i],
        #     entity="video",
        #     mode=mode,
        #     voxel_size=voxel_size,
        #     trail_len=trail_len,
        #     max_points=max_points,
        #     scale=scale,
        #     window_len=window_len,
        #     originate_before=originate_before,
        # )
        # if visualize_query_coords and query_coords is not None:
        #     log_query_3d_tracks_rerun(
        #         rgb=sample.rgbs[i],
        #         depth=sample.depths[i],
        #         intrinsics=sample.intrinsics[i],
        #         extrinsics=sample.extrinsics[i],
        #         query_coords=query_coords[i],
        #         query_visibs=query_visibs[i] if query_visibs is not None else None,
        #         query_birth_steps=query_birth_steps,
        #         entity="video",
        #     )
        entity = "video_est_pi3"
        if visualize_est and getattr(sample, "est_depths", None) is not None:
            # Overlay an "estimated camera" branch (same tracks, different
            # camera/depth backdrop) so reviewers can compare.
            log_dense_3d_tracks_rerun(
                rgb=sample.rgbs[i],
                depth=sample.est_depths[i],
                intrinsics=sample.est_intrinsics[i],
                extrinsics=sample.est_extrinsics[i],
                dense_coords=dense[i],
                dense_visibs=dense_visibs[i],
                dense_logits=dense_logits[i],
                dense_source_mask=dense_source_mask,
                entity=entity,
                mode=mode,
                voxel_size=voxel_size,
                trail_len=trail_len,
                max_points=max_points,
                scale=scale,
                window_len=window_len,
                originate_before=originate_before,
                show_frame_number=show_frame_number,
                trail_dim=trail_dim,
                trail_alpha=trail_alpha,
            )
            if visualize_dense_2d:
                log_dense_2d_tracks_video_rerun(
                    rgb=sample.rgbs[i],
                    intrinsics=sample.est_intrinsics[i],
                    extrinsics=sample.est_extrinsics[i],
                    dense_coords=dense[i],
                    dense_visibs=dense_visibs[i],
                    dense_source_mask=dense_source_mask,
                    entity=entity,
                    rate=dense_2d_rate,
                    bkg_opacity=dense_2d_bkg_opacity,
                    scale=scale,
                    window_len=window_len,
                    originate_before=originate_before,
                    colormap_name=dense_2d_colormap_name,
                    dense_logits=dense_logits[i],
                    trail_len=3,
                    seq_len=seq_len,
                    trail_dim=trail_dim,
                    trail_alpha=trail_alpha,
                )
            if visualize_query_coords and query_coords is not None:
                log_query_3d_tracks_rerun(
                    rgb=sample.rgbs[i],
                    depth=sample.est_depths[i],
                    intrinsics=sample.est_intrinsics[i],
                    extrinsics=sample.est_extrinsics[i],
                    query_coords=query_coords[i],
                    query_visibs=query_visibs[i] if query_visibs is not None else None,
                    query_birth_steps=query_birth_steps,
                    entity=entity,
                )

                # # also log the sample.gt_trajs_3d
                # log_query_3d_tracks_rerun(
                #     rgb=sample.rgbs[i],
                #     depth=sample.depths[i],
                #     intrinsics=sample.intrinsics[i],
                #     extrinsics=sample.extrinsics[i],
                #     query_coords=sample.gt_trajs_3d[i],
                #     query_visibs=sample.visibs[i],
                #     query_birth_steps=query_birth_steps,
                #     entity="video_gt_orig_scale",
                # )

                # # Median norm scaling (same masking/medians as tapvid3d_metrics scaling="median"),
                # # but scale GT by pred/gt so GT aligns to pred (inverse of metric's pred scale).
                # gt_tracks = sample.gt_trajs_3d[i].float()
                # pred_tracks = query_coords[i].float().to(gt_tracks.device)
                # pred_norms = torch.sqrt(
                #     torch.clamp(
                #         torch.sum(torch.square(pred_tracks), dim=-1), min=1e-12
                #     )
                # )
                # gt_norms = torch.sqrt(
                #     torch.clamp(
                #         torch.sum(torch.square(gt_tracks), dim=-1), min=1e-12
                #     )
                # )
                # _vis_gt = sample.visibs[i]
                # if _vis_gt.dtype == torch.bool:
                #     gt_occluded = ~_vis_gt.to(gt_tracks.device)
                # else:
                #     gt_occluded = _vis_gt.float().to(gt_tracks.device) <= 0.5
                # if query_visibs is not None:
                #     pred_occluded = (
                #         torch.sigmoid(query_visibs[i].float().to(gt_tracks.device))
                #         <= 0.5
                #     )
                # else:
                #     pred_occluded = torch.zeros_like(gt_occluded)
                # _valids = getattr(sample, "valids", None)
                # if _valids is not None:
                #     evaluation_points = _valids[i].to(gt_tracks.device)
                # else:
                #     evaluation_points = torch.ones_like(
                #         gt_occluded, dtype=torch.bool, device=gt_tracks.device
                #     )
                # either_occluded = torch.logical_or(gt_occluded, pred_occluded)
                # either_occluded = torch.logical_or(either_occluded, ~evaluation_points)
                # nan_val = torch.tensor(
                #     float("nan"), dtype=pred_norms.dtype, device=pred_norms.device
                # )
                # pred_norms = torch.where(either_occluded, nan_val, pred_norms)
                # gt_norms = torch.where(either_occluded, nan_val, gt_norms)
                # gt_median = torch.nanmedian(
                #     gt_norms.flatten(-2, -1), dim=-1, keepdim=True
                # )[0]
                # pred_median = torch.nanmedian(
                #     pred_norms.flatten(-2, -1), dim=-1, keepdim=True
                # )[0]
                # scale_factor = pred_median / torch.clamp(gt_median, min=1e-12)
                # if not torch.isfinite(scale_factor).item():
                #     gt_trajs_scaled = gt_tracks
                # else:
                #     gt_trajs_scaled = gt_tracks * scale_factor

                # log_query_3d_tracks_rerun(
                #     rgb=sample.rgbs[i],
                #     depth=sample.depths[i],
                #     intrinsics=sample.intrinsics[i],
                #     extrinsics=sample.extrinsics[i],
                #     query_coords=gt_trajs_scaled,
                #     query_visibs=sample.visibs[i],
                #     query_birth_steps=query_birth_steps,
                #     entity="video_gt_est_scale",
                # )
                

        # Close the file sink so the .rrd is fully flushed to disk before
        # the next iteration creates a new recording stream.
        import rerun as _rerun
        _rerun.disconnect()


def _draw_trails_lines(
    base: torch.Tensor,
    tracks_2d: torch.Tensor,
    head_vis: torch.Tensor,
    valid_pos: torch.Tensor,
    src_time: torch.Tensor,
    colors: torch.Tensor,
    *,
    trail_len: int = 8,
    trail_opacity: float = 1.0,
    linewidth: int = 1,
    line_aa: bool = False,
    trail_fade: str = "linear",
):
    """Draw fading comet trails as real line segments (CPU).

    For each display frame ``t`` and each dynamic track visible at ``t``, draws
    ``cv2.line`` segments over ``[t-trail_len+1, t]``. Default styling matches
    the locked dense wandb look (``linewidth=1``, ``trail_fade="linear"``,
    ``line_aa=False``, ``trail_opacity=1.0``).
    """
    T, C, H, W = base.shape
    device = base.device
    lw = max(1, int(linewidth))
    opacity = float(np.clip(trail_opacity, 0.0, 1.0))
    line_type = cv2.LINE_AA if line_aa else cv2.LINE_8
    fade_mode = str(trail_fade).lower().strip()
    if fade_mode not in ("quad", "linear", "flat"):
        raise ValueError(f"trail_fade must be quad|linear|flat, got {trail_fade!r}")

    xy = torch.nan_to_num(
        tracks_2d, nan=0.0, posinf=float(W - 1), neginf=0.0
    )
    xy_np = xy.detach().cpu().numpy()
    head_np = head_vis.detach().cpu().numpy().astype(bool)
    valid_np = valid_pos.detach().cpu().numpy().astype(bool)
    src_np = src_time.detach().cpu().numpy().astype(np.int64)
    colors_np = colors.detach().cpu().numpy().astype(np.float32)
    # OpenCV wants BGR for drawing into a BGR buffer; we work in RGB and convert.
    base_cpu = (
        base.detach().cpu().float().permute(0, 2, 3, 1).contiguous().numpy()
    )  # (T, H, W, 3)

    for t in range(T):
        t0 = max(0, t - trail_len + 1)
        if t <= t0:
            continue
        head = head_np[t]
        if not head.any():
            continue
        T_win = t - t0 + 1
        rgb = np.clip(base_cpu[t], 0.0, 255.0).astype(np.uint8)
        for s_idx, b in enumerate(range(t0 + 1, t + 1)):
            a = b - 1
            m = head & valid_np[a] & valid_np[b] & (src_np <= a)
            if not m.any():
                continue
            # Segment fade toward the trail head.
            if fade_mode == "flat":
                alpha = opacity
            elif fade_mode == "linear":
                denom = max(1, T_win - 1)
                alpha = (s_idx / float(denom)) * opacity
            else:  # quad (CoTracker)
                alpha = ((s_idx / float(T_win)) ** 2) * opacity
            if alpha <= 1e-6:
                continue
            original = rgb.copy()
            overlay = rgb  # draw in place, then blend with original
            idxs = np.flatnonzero(m)
            p0 = np.round(xy_np[a, idxs]).astype(np.int32)
            p1 = np.round(xy_np[b, idxs]).astype(np.int32)
            p0[:, 0] = np.clip(p0[:, 0], 0, W - 1)
            p0[:, 1] = np.clip(p0[:, 1], 0, H - 1)
            p1[:, 0] = np.clip(p1[:, 0], 0, W - 1)
            p1[:, 1] = np.clip(p1[:, 1], 0, H - 1)
            cols = np.clip(np.round(colors_np[idxs]), 0, 255).astype(np.int32)
            for i in range(idxs.shape[0]):
                c = (int(cols[i, 0]), int(cols[i, 1]), int(cols[i, 2]))
                cv2.line(
                    overlay,
                    (int(p0[i, 0]), int(p0[i, 1])),
                    (int(p1[i, 0]), int(p1[i, 1])),
                    c,
                    thickness=lw,
                    lineType=line_type,
                )
            rgb = (
                overlay.astype(np.float32) * alpha
                + original.astype(np.float32) * (1.0 - alpha)
            ).astype(np.uint8)
        base_cpu[t] = rgb.astype(np.float32)

    base.copy_(
        torch.from_numpy(base_cpu)
        .to(device=device, dtype=base.dtype)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    return base


# Alias kept for older call sites.
_draw_trails_gpu = _draw_trails_lines


@torch.no_grad()
def render_dense_2d_tracks_frames(
    rgb,
    intrinsics,
    extrinsics,
    dense_coords,
    dense_visibs,
    *,
    dense_source_mask=None,
    visib_thresh: float = 0.5,
    bkg_opacity: float = 0.5,
    scale: float = 1.0,
    window_len: int | None = None,
    originate_before: int | None = None,
    trail_len: int = 8,
    max_points: int = 20000,
    dense_logits: torch.Tensor | None = None,
    dynamic_thresh: float = 0.5,
    dynamic_from_motion: bool = False,
    motion_threshold: float = 0.1,
    dynamic_sticky: bool = False,
    static_lighten: float = 0.65,
    linewidth: int = 1,
    trail_opacity: float = 1.0,
    point_radius: int = 2,
    line_aa: bool = False,
    trail_fade: str = "linear",
    # Unused; kept so older callers that passed soft-splat ``rate`` do not break.
    rate: int = 1,
) -> np.ndarray:
    """Project dense 3D tracks to 2D and render the locked wandb overlay look.

    Look (``try_lw1_r2_linear``):
      - CoTracker ``gist_rainbow`` colors by birth y
      - Static: lightened 1px hard dots (no trails)
      - Dynamic: ``cv2.line`` trails (``linewidth=1``, linear fade, no AA,
        full opacity) + filled head circles (``point_radius=2``)

    Dynamic classification per target frame comes from ``dense_logits``
    (``sigmoid > dynamic_thresh``). When ``dynamic_from_motion`` is True,
    ``window_len`` must be set and the logit mask is AND-suppressed with a
    per-window motion gate: a track is only kept dynamic in a window if the L2
    extent of its xyz bbox over that window (``||amax - amin||``) exceeds
    ``motion_threshold``. Motion never invents dynamic labels.

    Returns:
        frames: ``(T, H, W, 3)`` uint8 numpy array.
    """
    del rate
    dense_coords = torch.as_tensor(dense_coords).detach().float()
    device = dense_coords.device

    T_src, Hf, Wf, T_tgt, _ = dense_coords.shape
    if T_src != T_tgt:
        raise ValueError(
            f"expected square dense tracks for 2D wandb overlay, got "
            f"T_src={T_src}, T_tgt={T_tgt}"
        )
    T = T_src
    dense_visibs = torch.as_tensor(dense_visibs).detach().float().to(device)
    if dense_visibs.shape != (T, Hf, Wf, T):
        raise ValueError(
            f"dense_visibs must have shape {(T, Hf, Wf, T)}, "
            f"got {tuple(dense_visibs.shape)}"
        )
    if dense_logits is None:
        raise ValueError(
            "dense_logits is required for static/dynamic classification in "
            "render_dense_2d_tracks_frames"
        )
    dense_logits = torch.as_tensor(dense_logits).detach().float().to(device)
    if tuple(dense_logits.shape) != (T, Hf, Wf, T):
        raise ValueError(
            "dense_logits must have shape (T_src, Hf, Wf, T_tgt); "
            f"got {tuple(dense_logits.shape)}"
        )

    src_valid = None
    if dense_source_mask is not None:
        dense_source_mask = torch.as_tensor(dense_source_mask).detach().bool().to(device)
        if dense_source_mask.shape != (T, Hf, Wf):
            raise ValueError(
                f"dense_source_mask must have shape {(T, Hf, Wf)}, "
                f"got {tuple(dense_source_mask.shape)}"
            )
        src_valid = dense_source_mask.reshape(-1)

    N = T * Hf * Wf
    flat = CompressedTracks(
        coords=dense_coords.reshape(N, T, 3),
        visibs=dense_visibs.reshape(N, T),
        dyn_logits=dense_logits.reshape(N, T),
        src_time=torch.arange(T, device=device).view(T, 1, 1).expand(T, Hf, Wf).reshape(-1),
        src_y=torch.arange(Hf, device=device).view(1, Hf, 1).expand(T, Hf, Wf).reshape(-1),
        src_x=torch.arange(Wf, device=device).view(1, 1, Wf).expand(T, Hf, Wf).reshape(-1),
    )
    return _render_flat_2d_tracks_frames(
        rgb,
        intrinsics,
        extrinsics,
        flat,
        src_valid=src_valid,
        visib_thresh=visib_thresh,
        bkg_opacity=bkg_opacity,
        scale=scale,
        window_len=window_len,
        originate_before=originate_before,
        trail_len=trail_len,
        max_points=max_points,
        dynamic_thresh=dynamic_thresh,
        dynamic_from_motion=dynamic_from_motion,
        motion_threshold=motion_threshold,
        dynamic_sticky=dynamic_sticky,
        static_lighten=static_lighten,
        linewidth=linewidth,
        trail_opacity=trail_opacity,
        point_radius=point_radius,
        line_aa=line_aa,
        trail_fade=trail_fade,
    )


@torch.no_grad()
def _render_flat_2d_tracks_frames(
    rgb,
    intrinsics,
    extrinsics,
    flat: CompressedTracks,
    *,
    src_valid=None,
    visib_thresh: float = 0.5,
    bkg_opacity: float = 0.5,
    scale: float = 1.0,
    window_len: int | None = None,
    originate_before: int | None = None,
    trail_len: int = 8,
    max_points: int = 20000,
    dynamic_thresh: float = 0.5,
    dynamic_from_motion: bool = False,
    motion_threshold: float = 0.1,
    dynamic_sticky: bool = False,
    static_lighten: float = 0.65,
    linewidth: int = 1,
    trail_opacity: float = 1.0,
    point_radius: int = 2,
    line_aa: bool = False,
    trail_fade: str = "linear",
) -> np.ndarray:
    """Project a flat track set to 2D and render the locked wandb overlay look.

    Shared by the dense path (via ``render_dense_2d_tracks_frames``) and the
    long-video path (via ``extract_compressed_tracks``).

    ``src_valid`` is an optional (N,) bool gating which tracks may be drawn; the
    dense path derives it from ``dense_source_mask``.
    """
    coords_ntc = flat.coords.float() * float(scale)
    device = coords_ntc.device
    N, T, _ = coords_ntc.shape

    rgb = torch.as_tensor(rgb).detach().float().to(device)
    intrinsics = torch.as_tensor(intrinsics).detach().float().to(device)
    extrinsics = torch.as_tensor(extrinsics).detach().float().to(device)
    if rgb.ndim != 4 or rgb.shape[0] != T or rgb.shape[1] != 3:
        raise ValueError(
            f"rgb must have shape (T, 3, H, W) with T={T}, got {tuple(rgb.shape)}"
        )

    _, _, H, W = rgb.shape
    rgb_255 = rgb
    if rgb_255.numel() > 0 and float(rgb_255.max()) <= 1.0 + 1e-6:
        rgb_255 = rgb_255 * 255.0
    rgb_255 = rgb_255.clamp(0.0, 255.0)

    tracks_3d = coords_ntc.permute(1, 0, 2).contiguous()
    tracks_2d = batch_project(
        tracks_3d,
        intrinsics[:, None].expand(T, N, 3, 3),
        extrinsics[:, None].expand(T, N, 4, 4),
    )

    src_time = flat.src_time.to(device)
    track_idx = torch.arange(N, device=device)
    xy0 = tracks_2d[src_time, track_idx].detach().cpu().numpy().copy()
    xy0[:, 0] = np.nan_to_num(xy0[:, 0], nan=0.0, posinf=float(W - 1), neginf=0.0)
    xy0[:, 1] = np.nan_to_num(xy0[:, 1], nan=0.0, posinf=float(H - 1), neginf=0.0)
    xy0[:, 0] = np.clip(xy0[:, 0], 0.0, float(W - 1))
    xy0[:, 1] = np.clip(xy0[:, 1], 0.0, float(H - 1))

    visibs = flat.visibs.to(device).permute(1, 0).sigmoid() > visib_thresh
    if src_valid is not None:
        visibs = visibs & src_valid.to(device).view(1, N)
    target_time = torch.arange(T, device=device).view(T, 1)
    active = src_time.view(1, N) <= target_time
    # if window_len is not None:
    #     if window_len < 1:
    #         raise ValueError(f"window_len must be >= 1, got {window_len}")
    #     active = active & (src_time.view(1, N) // window_len == target_time // window_len)
    if originate_before is not None:
        if originate_before < 0:
            raise ValueError(
                f"originate_before must be >= 0 when set, got {originate_before}"
            )
        active = active & (src_time.view(1, N) < originate_before)

    finite = torch.isfinite(tracks_2d).all(dim=-1)
    valid_pos = (
        finite
        & (tracks_2d[..., 0] >= 0)
        & (tracks_2d[..., 0] < W)
        & (tracks_2d[..., 1] >= 0)
        & (tracks_2d[..., 1] < H)
    )
    visibs = visibs & active & valid_pos

    if dynamic_from_motion:
        if window_len is None or window_len < 1:
            raise ValueError(
                "window_len must be set (>= 1) when dynamic_from_motion=True, "
                f"got {window_len}"
            )
    dyn_at_tgt = (
        flat.dyn_logits.to(device).permute(1, 0).sigmoid() > float(dynamic_thresh)
    )
    if dynamic_from_motion:
        # AND-suppress with per-window bbox extent (same metric as the loss).
        coords_tnc = coords_ntc.permute(1, 0, 2).contiguous()
        motion_dyn = _window_dynamic_per_frame_np(
            coords_tnc,
            float(motion_threshold),
            window_len,
            valids=(flat.visibs.to(device) > TRACK_VALID_THRESH).permute(1, 0),
        )
        dyn_at_tgt = dyn_at_tgt & motion_dyn

    if dynamic_sticky:
        dyn_at_tgt = _sticky_dynamic_mask(dyn_at_tgt, time_dim=0)

    visibs_dynamic = visibs & dyn_at_tgt
    visibs_static = visibs & ~dyn_at_tgt

    colors_dyn = torch.from_numpy(
        get_cotracker_rainbow_colors(xy0.copy(), H, W).astype(np.float32)
    ).to(device)
    lighten = float(np.clip(static_lighten, 0.0, 1.0))
    colors_static = colors_dyn * (1.0 - lighten) + 255.0 * lighten

    if N > max_points:
        g = torch.Generator().manual_seed(0)
        keep = torch.randperm(N, generator=g)[:max_points].to(device)
        tracks_2d = tracks_2d[:, keep]
        visibs_dynamic = visibs_dynamic[:, keep]
        visibs_static = visibs_static[:, keep]
        valid_pos = valid_pos[:, keep]
        colors_dyn = colors_dyn[keep]
        colors_static = colors_static[keep]
        src_time = src_time[keep]
        N = int(max_points)

    base = (rgb_255 * float(bkg_opacity)).clone()
    if trail_len > 1:
        base = _draw_trails_lines(
            base,
            tracks_2d,
            visibs_dynamic,
            valid_pos,
            src_time,
            colors_dyn,
            trail_len=trail_len,
            trail_opacity=trail_opacity,
            linewidth=linewidth,
            line_aa=line_aa,
            trail_fade=trail_fade,
        )
    # Static: tiny 1px dots; dynamic heads: filled circles of ``point_radius``.
    frames_static = draw_pts_hard(
        base, tracks_2d, visibs_static, colors_static, bkg_opacity=1.0, point_radius=0
    )
    base2 = (
        torch.from_numpy(frames_static)
        .to(device)
        .float()
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    return draw_pts_hard(
        base2,
        tracks_2d,
        visibs_dynamic,
        colors_dyn,
        bkg_opacity=1.0,
        point_radius=int(point_radius),
    )


@torch.no_grad()
def log_dense_2d_tracks_video_wandb(
    rgb,
    intrinsics,
    extrinsics,
    dense_coords,
    dense_visibs,
    *,
    dense_source_mask=None,
    visib_thresh: float = 0.5,
    bkg_opacity: float = 0.5,
    scale: float = 1.0,
    window_len: int | None = None,
    originate_before: int | None = None,
    trail_len: int = 8,
    max_points: int = 20000,
    fps: int = 12,
    dense_logits: torch.Tensor | None = None,
    dynamic_thresh: float = 0.5,
    dynamic_from_motion: bool = False,
    motion_threshold: float = 0.1,
    dynamic_sticky: bool = False,
    key: str = "dense_tracks_2d",
    rate: int = 1,
    mode: str = "dense",
    voxel_size: float = 0.02,
    max_dynamic_points: int = 4000,
    max_static_points: int = 16000,
):
    """Render the dense 2D tracks overlay and log it as a wandb H.264 video.

    ``mode="dense"`` (default) flattens every source cell then uniformly caps at
    ``max_points``. ``mode="voxel"`` is the ``demo.py --long_video`` overlay:
    one track per birth voxel, independent ever-dynamic / never-dynamic caps
    (``max_dynamic_points`` / ``max_static_points``), then the same 2D look.

    No-op if wandb is inactive. Writes a temp mp4 via ``save_dense_2d_tracks_video``
    so the logged file matches the local playable encode.
    """
    if wandb.run is None:
        return
    import tempfile

    if mode == "voxel":
        if dense_logits is None:
            raise ValueError("mode='voxel' requires dense_logits")
        flat, _, _, _ = _dense_tracks_to_flat(
            dense_coords,
            dense_visibs,
            dense_logits,
            dense_source_mask,
            mode="voxel",
            voxel_size=float(voxel_size),
            scale=scale,
        )
        ever_dyn = (
            flat.dyn_logits.float().sigmoid() > float(dynamic_thresh)
        ).any(dim=1)
        keep = _subsample_independent_caps(
            ever_dyn,
            max_points=int(max_static_points),
            max_dynamic_points=int(max_dynamic_points),
        )
        flat = CompressedTracks(
            coords=flat.coords[keep],
            visibs=flat.visibs[keep],
            dyn_logits=flat.dyn_logits[keep],
            src_time=flat.src_time[keep],
            src_y=flat.src_y[keep],
            src_x=flat.src_x[keep],
        )
        device = torch.as_tensor(rgb).device
        frames = _render_flat_2d_tracks_frames(
            rgb,
            intrinsics,
            extrinsics,
            flat.to(device),
            visib_thresh=visib_thresh,
            bkg_opacity=bkg_opacity,
            scale=1.0,
            window_len=window_len,
            originate_before=originate_before,
            trail_len=trail_len,
            max_points=max_points,
            dynamic_thresh=dynamic_thresh,
            dynamic_from_motion=dynamic_from_motion,
            motion_threshold=motion_threshold,
            dynamic_sticky=dynamic_sticky,
        )
    elif mode == "dense":
        frames = render_dense_2d_tracks_frames(
            rgb=rgb,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            dense_coords=dense_coords,
            dense_visibs=dense_visibs,
            dense_source_mask=dense_source_mask,
            visib_thresh=visib_thresh,
            bkg_opacity=bkg_opacity,
            scale=scale,
            window_len=window_len,
            originate_before=originate_before,
            trail_len=trail_len,
            max_points=max_points,
            dense_logits=dense_logits,
            dynamic_thresh=dynamic_thresh,
            dynamic_from_motion=dynamic_from_motion,
            motion_threshold=motion_threshold,
            dynamic_sticky=dynamic_sticky,
            rate=rate,
        )
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'dense' or 'voxel'")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    try:
        save_dense_2d_tracks_video(frames, tmp.name, fps=int(fps))
        wandb.log({key: wandb.Video(tmp.name, fps=int(fps), format="mp4")})
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except OSError:
            pass


def save_dense_2d_tracks_video(
    frames: np.ndarray,
    path,
    fps: int = 12,
) -> str:
    """Write ``(T, H, W, 3)`` uint8 RGB frames to an mp4 file at ``path``.

    Prefers imageio + ffmpeg H.264 (``libx264``) so the file plays in Cursor /
    VS Code / browsers. Falls back to OpenCV ``mp4v`` if imageio/ffmpeg is
    unavailable. Returns the written path.
    """
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"frames must have shape (T, H, W, 3), got {tuple(frames.shape)}"
        )
    path = str(path)
    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            path,
            fps=float(fps),
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv420p",
            macro_block_size=1,
            ffmpeg_params=["-crf", "18"],
        )
        try:
            for t in range(frames.shape[0]):
                writer.append_data(frames[t])
        finally:
            writer.close()
        return path
    except Exception:
        # Fallback: OpenCV mp4v (may not preview in some IDEs).
        T, H, W, _ = frames.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, float(fps), (W, H))
        if not writer.isOpened():
            raise RuntimeError(f"failed to open VideoWriter for {path}")
        try:
            for t in range(T):
                writer.write(cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        return path


@torch.no_grad()
def generate_dense_track_visualizations_with_wandb(
    sample,
    preds,
    output_dir: str,
    step_id,
    batch_idx,
    dataset_name: str,
    *,
    scale: float = 1.0,
    trail_len: int = 8,
    max_points: int = 20000,
    rate: int = 1,
    bkg_opacity: float = 0.5,
    window_len: int | None = None,
    originate_before: int | None = None,
    fps: int = 12,
    visib_thresh: float = 0.5,
    dynamic_thresh: float = 0.5,
    dynamic_from_motion: bool = False,
    motion_threshold: float = 0.1,
):
    """Reconstruct dense 3D tracks from ``preds`` and log a 2D overlay video to
    wandb (static dots + dynamic trails).

    Mirrors ``generate_dense_track_visualizations_with_rerun`` inputs but emits
    a wandb video instead of a ``.rrd`` recording. Uses the estimated camera when
    available (matching the est branch of the rerun path).
    """
    if wandb.run is None:
        return

    from .alltrack3r_utils import reconstruct_dense_tracks_3d

    dense, dense_visibs, _dense_confs, dense_logits = reconstruct_dense_tracks_3d(
        preds,
        return_visibs_confs=True,
        dynamic_logits_by_target=True,
    )

    dense_norm_mask = getattr(sample, "dense_norm_mask", None)

    def _batch_dense_source_mask(bi: int, target_shape: tuple[int, int, int]):
        if dense_norm_mask is None or dense_norm_mask[0] is None:
            return None
        mask = torch.as_tensor(dense_norm_mask[bi]).detach().bool()
        if mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        if tuple(mask.shape) == target_shape:
            return mask
        if mask.ndim != 3 or mask.shape[0] != target_shape[0]:
            raise ValueError(
                f"dense_norm_mask for batch item {bi} must have shape "
                f"{target_shape} or (T, H, W, 1), got {tuple(mask.shape)}"
            )
        return (
            F.interpolate(
                mask.float().unsqueeze(1),
                size=target_shape[1:],
                mode="nearest",
            )
            .squeeze(1)
            .bool()
        )

    use_est = (
        getattr(sample, "est_intrinsics", None) is not None
        and getattr(sample, "est_extrinsics", None) is not None
    )
    intr_all = sample.est_intrinsics if use_est else sample.intrinsics
    extr_all = sample.est_extrinsics if use_est else sample.extrinsics

    B = dense.shape[0]
    for i in range(B):
        dense_source_mask = _batch_dense_source_mask(i, tuple(dense.shape[1:4]))
        log_dense_2d_tracks_video_wandb(
            rgb=sample.rgbs[i],
            intrinsics=intr_all[i],
            extrinsics=extr_all[i],
            dense_coords=dense[i],
            dense_visibs=dense_visibs[i],
            dense_source_mask=dense_source_mask,
            visib_thresh=visib_thresh,
            rate=rate,
            bkg_opacity=bkg_opacity,
            scale=scale,
            window_len=window_len,
            originate_before=originate_before,
            trail_len=trail_len,
            max_points=max_points,
            fps=fps,
            dense_logits=dense_logits[i] if dense_logits is not None else None,
            dynamic_thresh=dynamic_thresh,
            dynamic_from_motion=dynamic_from_motion,
            motion_threshold=motion_threshold,
            key=f"{dataset_name}/dense_tracks_2d_{batch_idx}_{i}",
        )


@torch.no_grad()
def log_query_3d_tracks_rerun(
    rgb,
    depth,
    intrinsics,
    extrinsics,
    query_coords,
    query_visibs=None,
    *,
    query_birth_steps=None,
    entity: str = "video",
    visib_thresh: float = 0.5,
    point_radius: float = 0.008,
    trail_radius: float = 0.003,
    trail_len: int = 10,
    colormap_name: str = "turbo",
    max_points: int = 1000,
    scale: float = 1.0,
    log_backdrop: bool = False,
):
    """Log query-point trajectories to the active rerun recording.

    Args:
        query_coords: (T, N, 3) query trajectories in world coordinates.
        query_visibs: optional (T, N) visibility logits (unused for display; all points
            are shown after birth, including when the model marks them occluded).
        query_birth_steps: optional (N,) int tensor or array of local frame indices
            when each query first appears; points and trails are omitted for t < birth.
        visib_thresh: unused; kept for call-site compatibility.
    """
    import rerun
    import matplotlib
    from . import rerun_visualizer as _rv

    _ = visib_thresh  # query tracks are not filtered by visibility

    query_coords = torch.as_tensor(query_coords).detach().float().cpu() * float(scale)
    assert query_coords.ndim == 3 and query_coords.shape[-1] == 3, (
        f"expected query_coords with shape (T, N, 3), got {tuple(query_coords.shape)}"
    )
    T, N, _ = query_coords.shape

    if query_visibs is not None:
        query_visibs = torch.as_tensor(query_visibs).detach().float().cpu()
        assert query_visibs.shape == (T, N), (
            f"expected query_visibs with shape {(T, N)}, got {tuple(query_visibs.shape)}"
        )

    if query_birth_steps is not None:
        query_birth_steps = torch.as_tensor(query_birth_steps).detach().long().reshape(-1)
        assert query_birth_steps.shape[0] == N, (
            f"query_birth_steps length {query_birth_steps.shape[0]} != N={N}"
        )

    if N > max_points:
        g = torch.Generator().manual_seed(0)
        keep = torch.randperm(N, generator=g)[:max_points]
        query_coords = query_coords[:, keep]
        if query_visibs is not None:
            query_visibs = query_visibs[:, keep]
        if query_birth_steps is not None:
            query_birth_steps = query_birth_steps[keep]
        N = query_coords.shape[1]

    tracks = query_coords.permute(1, 0, 2).numpy()  # (N, T, 3)
    query_birth_steps_np = (
        None
        if query_birth_steps is None
        else query_birth_steps.detach().cpu().numpy().astype(np.int64, copy=False)
    )

    cmap = matplotlib.colormaps[colormap_name]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(N - 1, 1))
    colors = (cmap(norm(np.arange(N)))[:, :3] * 255.0).astype(np.uint8)  # (N, 3)

    rerun.log("/", rerun.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    if log_backdrop:
        _rv.log_video(
            entity_name=entity,
            rgb=rgb,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            depth=depth,
        )

    pts_path = f"/{entity}/world/query_tracks/points"
    trails_path = f"/{entity}/world/query_tracks/trails"

    for t in range(T):
        rerun.set_time("frameid", sequence=t)
        if query_birth_steps_np is None:
            rerun.log(
                pts_path,
                rerun.Points3D(tracks[:, t], colors=colors[:], radii=point_radius),
            )
            if trail_len > 1 and t > 0:
                t0 = max(0, t - trail_len + 1)
                strips = tracks[:, t0 : t + 1]
                rerun.log(
                    trails_path,
                    rerun.LineStrips3D(strips, colors=colors, radii=trail_radius),
                )
            elif t == 0:
                rerun.log(trails_path, rerun.Clear(recursive=False))
            continue

        birth_ok = t >= query_birth_steps_np
        show = birth_ok
        if show.any():
            rerun.log(
                pts_path,
                rerun.Points3D(
                    tracks[show, t],
                    colors=colors[show],
                    radii=point_radius,
                ),
            )
        else:
            rerun.log(pts_path, rerun.Clear(recursive=False))

        if trail_len > 1 and t > 0:
            t0 = max(0, t - trail_len + 1)
            strip_list = []
            strip_cols = []
            for n in range(N):
                if not show[n]:
                    continue
                start = max(t0, int(query_birth_steps_np[n]))
                seg = tracks[n, start : t + 1]
                if seg.shape[0] >= 2:
                    strip_list.append(seg)
                    strip_cols.append(colors[n])
            if strip_list:
                rerun.log(
                    trails_path,
                    rerun.LineStrips3D(
                        strip_list,
                        colors=np.stack(strip_cols, axis=0),
                        radii=trail_radius,
                    ),
                )
            else:
                rerun.log(trails_path, rerun.Clear(recursive=False))
        elif t == 0:
            rerun.log(trails_path, rerun.Clear(recursive=False))


        
        
def _log_conf_overlay_wandb(
    rgbs: np.ndarray,
    conf_full: np.ndarray,
    fps: float = 15.0,
    heatmap_alpha: float = 0.55,
) -> None:
    """Log a wandb.Video of RGB + colormapped confidence (no disk write).

    `conf_low` is model resolution (T, H_low, W_low); it is nearest-neighbor
    upsampled to match `rgbs` (T, orig_h, orig_w, 3) before blending.
    """
    a = float(np.clip(heatmap_alpha, 0.0, 1.0))
    # wandb.Video expects (time, channel, height, width).
    frames = np.empty((rgbs.shape[0], 3, rgbs.shape[1], rgbs.shape[2]), dtype=np.uint8)
    for t in range(rgbs.shape[0]):
        rgb = rgbs[t]
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
        c = np.clip(conf_full[t].astype(np.float32), 0.0, 1.0)
        conf_u8 = (c * 255.0).round().astype(np.uint8)
        heat_bgr = cv2.applyColorMap(conf_u8, cv2.COLORMAP_INFERNO)
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        blended_bgr = (
            a * heat_bgr.astype(np.float32) + (1.0 - a) * bgr.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)
        blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
        frames[t] = np.transpose(blended_rgb, (2, 0, 1))

    wandb.log(
        {"conf_overlay": wandb.Video(frames, fps=int(round(fps)), format="mp4")}
    )


def log_norm_mask_overlay_wandb(
    rgbs: np.ndarray,
    norm_mask: np.ndarray,
    fps: float = 15.0,
    heatmap_alpha: float = 0.5,
    key: str = "norm_mask_overlay",
) -> None:
    """Log a wandb.Video of RGB blended with a colormapped norm mask (bool or 0/1).

    No-op when wandb is not initialized. Intended for occasional debugging (can be
    very slow if called every sample); pair with env ``WANDB_LOG_NORM_MASK=1`` from
    the caller when desired.
    """
    if wandb.run is None:
        return
    a = float(np.clip(heatmap_alpha, 0.0, 1.0))
    frames = np.empty((rgbs.shape[0], 3, rgbs.shape[1], rgbs.shape[2]), dtype=np.uint8)
    for t in range(rgbs.shape[0]):
        rgb = rgbs[t]
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
        m = norm_mask[t].astype(np.float32)
        if m.max() > 1.0 or m.min() < 0.0:
            m = np.clip(m, 0.0, 1.0)
        m_u8 = (m * 255.0).round().astype(np.uint8)
        heat_bgr = cv2.applyColorMap(m_u8, cv2.COLORMAP_INFERNO)
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        blended_bgr = (
            a * heat_bgr.astype(np.float32) + (1.0 - a) * bgr.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)
        blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
        frames[t] = np.transpose(blended_rgb, (2, 0, 1))

    wandb.log({key: wandb.Video(frames, fps=int(round(fps)), format="mp4")})


