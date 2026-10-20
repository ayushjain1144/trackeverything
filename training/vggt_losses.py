# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from math import ceil, floor
import ipdb

st = ipdb.set_trace


def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.
    
    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside 
                                  [-hard_max, hard_max] will be clamped. If None, 
                                  no clamping is performed. Defaults to 100.
    """
    if input_tensor is None:
        return input_tensor
    
    # Check for inf/nan values
    has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
    if has_inf_nan:
        print(f"Tensor {loss_name} contains inf or nan values. Replacing with zeros.")
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor),
            torch.zeros_like(input_tensor),
            input_tensor
        )

    # Apply hard clamping if specified
    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor

def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.
    
    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.
    
    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization
    
    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    with torch.amp.autocast("cuda", enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Get neighboring points for each pixel
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Compute direction vectors from center to neighbors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Compute four cross products for different normal directions
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity masks - require both direction pixels to be valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack normals and validity masks
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize normal vectors
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids

def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.
    
    This helps remove outliers that could destabilize training.
    
    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss
    
    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor

    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.
    
    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.
    
    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    # Extract valid normals
    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    dot = torch.sum(pred_normals * gt_normals, dim=-1)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            # Apply confidence weighting
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss


def regression_loss_flexible(pred, gt, mask, conf=None, gradient_loss_fn="none",
                              gamma=1.0, alpha=0.2, valid_range=-1, dynamic_weight=1.0, dynamic_mask=None):
    """
    Shape-flexible regression loss with confidence weighting and optional gradient loss.

    Supports arbitrary input shapes where the last dim is the feature dim (C) and
    mask/conf have one fewer dim than pred/gt. Specifically handles:
      - (B, T1, H, W, T, C): dense grid with spatial structure -> gradient loss supported
      - (B, T, N, C) or (N, T, C): point trajectories -> no gradient loss

    The confidence formula: gamma * ||pred - gt|| * conf - alpha * log(conf)
    Expects already-activated conf (positive values). Activation (e.g. expp1, sigmoid)
    should be applied in the model architecture, not here.

    Args:
        pred: (..., C) predicted values
        gt: (..., C) ground truth values, same shape as pred
        mask: (...) boolean mask, same shape as pred without last dim
        conf: (...) already-activated confidence (positive values, same shape as mask),
              or None. Activation (expp1, expp0, sigmoid) is done in the model.
        gradient_loss_fn: str — "none", "grad", "normal", "grad_conf", "normal_conf"
        gamma: weight for confidence-scaled loss term
        alpha: weight for confidence regularization (encourages not-too-low confidence)
        valid_range: quantile for outlier filtering (-1 to disable)
        dynamic_weight: relative weight for dynamic points in final reduction
        dynamic_mask: optional (...) boolean mask indicating dynamic points

    Returns:
        loss_conf: confidence-weighted loss (scalar)
        loss_grad: gradient/normal loss (scalar, or 0 if not applicable)
        loss_reg:  plain L2 loss without confidence weighting (scalar)
    """
    masked_pred = pred[mask]
    masked_gt = gt[mask]

    if masked_pred.numel() == 0:
        zero = (0.0 * pred).mean()
        return zero, 0, zero

    loss_reg = torch.norm(masked_gt - masked_pred, dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    if conf is not None:
        masked_conf = conf[mask]
        loss_conf = gamma * loss_reg * masked_conf - alpha * torch.log(masked_conf.clamp(min=1e-8))
        loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")
    else:
        loss_conf = loss_reg.clone()

    # --- Gradient loss (spatial structure required) ---
    loss_grad = 0
    if gradient_loss_fn is not None and gradient_loss_fn != "none":
        if pred.ndim == 6:
            # Dense grid: (B, T1, H, W, T, C) -> (B*T1*T, H, W, C)
            bb, t1, hh, ww, tt, nc = pred.shape
            pred_sp = pred.permute(0, 1, 4, 2, 3, 5).reshape(bb * t1 * tt, hh, ww, nc)
            gt_sp = gt.permute(0, 1, 4, 2, 3, 5).reshape(bb * t1 * tt, hh, ww, nc)
            mask_sp = mask.permute(0, 1, 4, 2, 3).reshape(bb * t1 * tt, hh, ww)

            to_feed_conf = None
            if conf is not None and "conf" in gradient_loss_fn:
                conf_sp = conf.permute(0, 1, 4, 2, 3).reshape(bb * t1 * tt, hh, ww)
                to_feed_conf = conf_sp

            if "normal" in gradient_loss_fn:
                loss_grad = gradient_loss_multi_scale_wrapper(
                    pred_sp, gt_sp, mask_sp,
                    gradient_loss_fn=normal_loss, scales=3, conf=to_feed_conf,
                )
            elif "grad" in gradient_loss_fn:
                loss_grad = gradient_loss_multi_scale_wrapper(
                    pred_sp, gt_sp, mask_sp,
                    gradient_loss_fn=gradient_loss, conf=to_feed_conf,
                )
        elif pred.ndim == 5:
            # Original VGGT format: (B, S, H, W, C)
            bb, ss, hh, ww, nc = pred.shape
            pred_sp = pred.reshape(bb * ss, hh, ww, nc)
            gt_sp = gt.reshape(bb * ss, hh, ww, nc)
            mask_sp = mask.reshape(bb * ss, hh, ww)

            to_feed_conf = None
            if conf is not None and "conf" in gradient_loss_fn:
                conf_sp = conf.reshape(bb * ss, hh, ww)
                to_feed_conf = conf_sp

            if "normal" in gradient_loss_fn:
                loss_grad = gradient_loss_multi_scale_wrapper(
                    pred_sp, gt_sp, mask_sp,
                    gradient_loss_fn=normal_loss, scales=3, conf=to_feed_conf,
                )
            elif "grad" in gradient_loss_fn:
                loss_grad = gradient_loss_multi_scale_wrapper(
                    pred_sp, gt_sp, mask_sp,
                    gradient_loss_fn=gradient_loss, conf=to_feed_conf,
                )
        # For lower-dim inputs (4D, 3D) there is no spatial H, W -> skip gradient loss
    masked_dynamic = None
    if dynamic_mask is not None:
        masked_dynamic = dynamic_mask.bool()[mask]

    def _reduce_loss(loss_vec, name):
        if loss_vec.numel() == 0:
            return (0.0 * pred).mean()

        if masked_dynamic is not None and masked_dynamic.shape == loss_vec.shape:
            static_vals = loss_vec[~masked_dynamic]
            dynamic_vals = loss_vec[masked_dynamic]

            if valid_range > 0:
                if static_vals.numel() > 0:
                    static_vals = filter_by_quantile(static_vals, valid_range)
                if dynamic_vals.numel() > 0:
                    dynamic_vals = filter_by_quantile(dynamic_vals, valid_range)

            if static_vals.numel() > 0:
                static_vals = check_and_fix_inf_nan(static_vals, f"{name}_static")
            if dynamic_vals.numel() > 0:
                dynamic_vals = check_and_fix_inf_nan(dynamic_vals, f"{name}_dynamic")
            static_mean = static_vals.mean() if static_vals.numel() > 0 else (0.0 * pred).mean()
            dynamic_mean = dynamic_vals.mean() if dynamic_vals.numel() > 0 else (0.0 * pred).mean()
            return static_mean + dynamic_weight * dynamic_mean

        if valid_range > 0:
            loss_vec = filter_by_quantile(loss_vec, valid_range)
        loss_vec = check_and_fix_inf_nan(loss_vec, name)
        return loss_vec.mean()

    # --- Reduce losses ---
    loss_conf = _reduce_loss(loss_conf, "loss_conf_final")
    loss_reg = _reduce_loss(loss_reg, "loss_reg_final")

    return loss_conf, loss_grad, loss_reg