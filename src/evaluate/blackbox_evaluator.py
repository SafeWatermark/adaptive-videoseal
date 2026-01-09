"""
Blackbox attacks evaluator.

Evaluates watermarking models under blackbox attacks (Square Attack, Triangle Attack)
for both removal and forgery attacks with all aggregation strategies.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from typing import Dict, Optional, Iterator, Tuple
import pandas as pd
import logging
from tqdm import tqdm
from scipy.stats import binom

try:
    import torch_dct
except ImportError:
    raise ImportError(
        "torch_dct is required for Triangle Attack. "
        "Please install it: pip install torch-dct"
    )
import random
import time

# Add submodule to path
submodule_path = Path(__file__).parent.parent / "videomarkbench"
sys.path.insert(0, str(submodule_path))

from aggregation import (  # noqa: E402
    logit_mean as _logit_mean,
    logit_g_median as _logit_g_median,
    bit_majority as _bit_majority,
    bit_g_median as _bit_g_median,
    bitacc_mean as _bitacc_mean,
    bitacc_median as _bitacc_median,
    detection_threshold as _detection_threshold,
    detection_majority as _detection_majority,
)

# Import common aggregation utilities
from .aggregation_utils import (
    call_with_watermark_gt_tensor,
)

# Aggregation strategies
AGGREGATION_STRATEGIES = {
    "logit_mean": _logit_mean,
    "logit_g_median": _logit_g_median,
    "bit_majority": _bit_majority,
    "bit_g_median": _bit_g_median,
    "bitacc_mean": _bitacc_mean,
    "bitacc_median": _bitacc_median,
    "detection_threshold": _detection_threshold,
    "detection_majority": _detection_majority,
}


def p_selection(p_init: float, it: int, n_iters: int) -> float:
    """Piece-wise constant schedule for p (the fraction of pixels changed on every iteration)."""
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p


def get_tau_for_aggregation(aggregation: str, n_frames: int) -> float:
    """Get threshold tau for given aggregation strategy."""
    if aggregation in [
        "ba-mean",
        "ba-median",
        "bit-median",
        "logit-mean",
        "logit-median",
    ]:
        return 29 / 96
    elif aggregation == "detection-threshold":
        p = 0.0000661
        threshold = 10e-4
        for i in range(n_frames + 1):
            if binom.sf(i - 1, n_frames, p) <= threshold:
                k = i
                break
        return 1 - k / n_frames
    elif aggregation == "detection-median":
        return 0.5
    else:
        return 29 / 96  # Default


def square_attack_linf(
    model,
    video: torch.Tensor,
    attack_type: str,
    watermark_gt: Optional[np.ndarray] = None,
    eps: float = 0.05,
    n_iters: int = 1000,
    p_init: float = 0.05,
    device: torch.device = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Square Attack (Linf) for video watermarking.

    Uses ba-mean aggregation for attack optimization, then evaluates with all aggregation strategies.

    Args:
        model: VideoMarkBenchModel wrapper instance
        video: Video tensor of shape (T, C, H, W) or (1, T, C, H, W)
        attack_type: 'removal' or 'forgery'
        watermark_gt: Ground truth watermark pattern (if None, will be inferred from logits)
        eps: Perturbation bound (default: 0.05)
        n_iters: Maximum iterations (default: 1000)
        p_init: Initial probability of changing pixels (default: 0.05)
        device: Device to run on
        logger: Optional logger

    Returns:
        Tuple of (perturbed_video, n_queries)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if device is None:
        device = model.device

    # Ensure video is (T, C, H, W)
    if video.dim() == 5:
        video = video.squeeze(0)
    video = video.to(device)

    T, C, H, W = video.shape
    video_reshaped = video.reshape(1, T * C, H, W)  # (1, F*C, H, W)

    # Keep on GPU for better performance
    video_tensor = video_reshaped  # Keep as tensor

    # Get threshold (use ba-mean for attack optimization)
    tau = get_tau_for_aggregation("ba-mean", T)

    min_val, max_val = 0.0, 1.0
    c, h, w = video_tensor.shape[1:]

    # Initialize perturbation on GPU
    torch.manual_seed(0)
    init_delta = torch.randint(0, 2, size=(1, c, 1, w), device=device).float()
    init_delta = (init_delta * 2 - 1) * eps  # Convert 0/1 to -eps/eps
    x_best = torch.clamp(video_tensor + init_delta, min_val, max_val)

    # Initial detection - use ba-mean for attack optimization
    x_best_reshaped = x_best.reshape(1, T, C, H, W).squeeze(0)  # (T, C, H, W)
    logits = model.extract(x_best_reshaped)  # (T, K)

    # Get aggregated score using ba-mean (used for attack optimization)
    # Use provided watermark_gt or infer from logits size
    if watermark_gt is None:
        K = logits.shape[1]
        watermark_gt = np.array([0, 1] * (K // 2))  # Alternating pattern
        if K % 2 == 1:
            watermark_gt = np.append(watermark_gt, 0)  # Add one more if odd

    # Compute on GPU for better performance (reuse in loop)
    watermark_gt_tensor = torch.from_numpy(watermark_gt).float().to(device)  # (K,)
    watermark_gt_bits = (
        watermark_gt_tensor.unsqueeze(0).expand(logits.shape[0], -1).long()
    )  # (T, K)
    predicted_bits = (logits >= 0).long()  # (T, K)
    bitacc = (predicted_bits == watermark_gt_bits).float().mean().item()
    score = bitacc

    loss_min = score if attack_type == "forgery" else -score
    n_queries = 1
    no_improvement_count = 0
    max_no_improvement = 50  # Early stop if no improvement for 50 iterations

    # Attack loop
    for i_iter in range(n_iters - 1):
        # Check if attack succeeded
        if attack_type == "removal" and score < tau:
            break
        elif attack_type == "forgery" and score >= tau:
            break

        # Early stopping if no improvement
        if no_improvement_count >= max_no_improvement:
            if logger:
                logger.debug(
                    f"Early stopping at iteration {i_iter} (no improvement for {max_no_improvement} iterations)"
                )
            break

        x_curr = x_best.clone()
        deltas = x_best - video_tensor

        p = p_selection(p_init, i_iter, n_iters)
        n_features = c * h * w
        s = int(round(np.sqrt(p * n_features / c)))
        s = min(max(s, 1), h - 1)

        center_h = torch.randint(0, h - s, (1,), device=device).item()
        center_w = torch.randint(0, w - s, (1,), device=device).item()

        # Update perturbation in square region (on GPU)
        # Optimize: avoid unnecessary operations
        delta_slice = deltas[0, :, center_h : center_h + s, center_w : center_w + s]
        x_curr_slice = x_curr[0, :, center_h : center_h + s, center_w : center_w + s]
        x_best_slice = x_best[0, :, center_h : center_h + s, center_w : center_w + s]

        # Check if delta would change anything (optimize)
        clipped = torch.clamp(x_curr_slice + delta_slice, min_val, max_val)
        diff = torch.abs(clipped - x_best_slice)
        if (diff < 1e-7).all():
            # Generate new random delta only if needed
            max_retries = 10  # Limit retries
            for _ in range(max_retries):
                new_delta = torch.randint(0, 2, size=(c, 1, 1), device=device).float()
                new_delta = (new_delta * 2 - 1) * eps
                deltas[
                    0, :, center_h : center_h + s, center_w : center_w + s
                ] = new_delta
                delta_slice = deltas[
                    0, :, center_h : center_h + s, center_w : center_w + s
                ]
                clipped = torch.clamp(x_curr_slice + delta_slice, min_val, max_val)
                diff = torch.abs(clipped - x_best_slice)
                if not (diff < 1e-7).all():
                    break

        x_new = torch.clamp(video_tensor + deltas, min_val, max_val)

        # Evaluate new perturbation - use ba-mean for attack optimization
        x_new_reshaped = x_new.reshape(1, T, C, H, W).squeeze(0)  # (T, C, H, W)
        logits = model.extract(x_new_reshaped)  # (T, K)

        # Get aggregated score using ba-mean (used for attack optimization)
        # Compute on GPU
        predicted_bits = (logits >= 0).long()  # (T, K)
        bitacc = (predicted_bits == watermark_gt_bits).float().mean().item()
        score_new = bitacc

        loss_new = score_new if attack_type == "forgery" else -score_new

        # Update if improved
        improved = False
        if attack_type == "forgery":
            if loss_new > loss_min:
                x_best = x_new.clone()  # Use clone to avoid reference issues
                loss_min = loss_new
                score = score_new
                improved = True
        else:  # removal
            if loss_new < loss_min:
                x_best = x_new.clone()  # Use clone to avoid reference issues
                loss_min = loss_new
                score = score_new
                improved = True

        if improved:
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        n_queries += 1

        if (i_iter + 1) % 100 == 0 and logger:
            logger.debug(
                f"Iteration {i_iter+1}: score={score:.4f}, queries={n_queries}, no_improvement={no_improvement_count}"
            )

    # Convert back to video format (already on GPU)
    x_best_reshaped = x_best.reshape(1, T, C, H, W).squeeze(0)  # (T, C, H, W)

    return x_best_reshaped, n_queries


# Triangle Attack helper functions (from original videomarkbench implementation)
def get_difference(x_o: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
    """Compute the difference between original and adversarial example."""
    difference = x_adv - x_o
    if torch.norm(difference, p=2) == 0:
        raise RuntimeError("difference is zero vector!")
    return difference


def rotate_in_2d(
    x_o2x_adv: torch.Tensor, direction: torch.Tensor, theta: float = np.pi / 8
) -> torch.Tensor:
    """Rotate direction in 2D plane."""
    alpha = torch.sum(x_o2x_adv * direction) / torch.sum(x_o2x_adv * x_o2x_adv)
    orthogonal = direction - alpha * x_o2x_adv
    norm_orthogonal = torch.norm(orthogonal, p=2)
    if norm_orthogonal == 0:
        return direction
    direction_theta = x_o2x_adv * np.cos(theta) + torch.norm(
        x_o2x_adv, p=2
    ) / norm_orthogonal * orthogonal * np.sin(theta)
    direction_theta = (
        direction_theta / torch.norm(direction_theta, p=2) * torch.norm(x_o2x_adv, p=2)
    )
    return direction_theta


def get_orthogonal_1d_in_subspace(
    x_o2x_adv: torch.Tensor,
    n: int,
    ratio_size_mask: float,
    H: int,
    W: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Obtain the mask in the low frequency (orthogonal direction)."""
    random.seed(int(time.time() * 1000) % 2**32)
    zero_mask = torch.zeros(size=[H, W], device=device)
    size_mask = int(W * ratio_size_mask)
    zero_mask[:size_mask, :size_mask] = 1

    to_choose = torch.where(zero_mask == 1)
    x = to_choose[0]
    y = to_choose[1]

    masks_per_frame = []
    num_frames = x_o2x_adv.shape[1] // 3
    for _ in range(num_frames):
        # mask1
        select = np.random.choice(len(x), size=n, replace=False)
        mask1 = torch.zeros_like(zero_mask)
        mask1[x[select], y[select]] = 1
        # mask2
        select = np.random.choice(len(x), size=n, replace=False)
        mask2 = torch.zeros_like(zero_mask)
        mask2[x[select], y[select]] = 1
        # mask3
        select = np.random.choice(len(x), size=n, replace=False)
        mask3 = torch.zeros_like(zero_mask)
        mask3[x[select], y[select]] = 1

        masks_per_frame.append(torch.stack([mask1, mask2, mask3], dim=0))  # (3, H, W)
    mask = torch.cat(masks_per_frame, dim=0).unsqueeze(0)  # (1, F*3, H, W)

    mask *= torch.randn_like(mask, device=device)
    direction = rotate_in_2d(x_o2x_adv, mask, theta=np.pi / 2)
    return direction / torch.norm(direction, p=2) * torch.norm(x_o2x_adv, p=2), mask


def get_x_hat_in_2d(
    x_o: torch.Tensor,
    x_adv: torch.Tensor,
    axis_unit1: torch.Tensor,
    axis_unit2: torch.Tensor,
    model,
    video_shape: Tuple[int, int, int, int],  # (T, C, H, W)
    aggregation: str,
    watermark_gt: Optional[np.ndarray],
    queries: int,
    original_label: bool,
    max_iter: int = 2,
    plus_learning_rate: float = 0.1,
    minus_learning_rate: float = 0.005,
    half_range: float = 0.1,
    init_alpha: float = np.pi / 2,
    device: torch.device = None,
) -> Tuple[torch.Tensor, int, bool]:
    """
    Compute the best adversarial example in the 2D surface.

    Returns:
        Tuple of (x_hat, queries, changed)
    """
    if not hasattr(get_x_hat_in_2d, "alpha"):
        get_x_hat_in_2d.alpha = init_alpha
    upper = np.pi / 2 + half_range
    lower = np.pi / 2 - half_range

    T, C, H, W = video_shape

    # Helper function to get label
    def get_label(video_tensor: torch.Tensor) -> bool:
        """Get binary label: True if watermark detected, False otherwise."""
        video_detect = video_tensor.reshape(1, T, C, H, W).squeeze(0)  # (T, C, H, W)
        logits = model.extract(video_detect)  # (T, K)

        if watermark_gt is None:
            K = logits.shape[1]
            watermark_gt_local = np.array([0, 1] * (K // 2))
            if K % 2 == 1:
                watermark_gt_local = np.append(watermark_gt_local, 0)
        else:
            watermark_gt_local = watermark_gt

        watermark_gt_tensor = torch.from_numpy(watermark_gt_local).float().to(device)
        watermark_gt_tiled = watermark_gt_tensor.unsqueeze(0).expand(
            logits.shape[0], -1
        )
        predicted_bits = (logits >= 0).long()
        watermark_gt_bits = (watermark_gt_tiled >= 0.5).long()
        bitacc = (predicted_bits == watermark_gt_bits).float().mean().item()
        tau = get_tau_for_aggregation(aggregation, T)
        return bitacc >= tau

    d = torch.norm(x_adv - x_o, p=2)

    theta = max(np.pi - 2 * get_x_hat_in_2d.alpha, 0) + min(
        np.pi / 16, get_x_hat_in_2d.alpha / 2
    )
    x_hat = torch_dct.idct_2d(x_adv)
    right_theta = np.pi - get_x_hat_in_2d.alpha
    x = x_o + d * (axis_unit1 * np.cos(theta) + axis_unit2 * np.sin(theta)) / np.sin(
        get_x_hat_in_2d.alpha
    ) * np.sin(get_x_hat_in_2d.alpha + theta)
    x = torch_dct.idct_2d(x)
    if not hasattr(get_x_hat_in_2d, "total"):
        get_x_hat_in_2d.total = 0
    if not hasattr(get_x_hat_in_2d, "clamp"):
        get_x_hat_in_2d.clamp = 0
    get_x_hat_in_2d.total += 1
    get_x_hat_in_2d.clamp += torch.sum(x > 1) + torch.sum(x < 0)
    x = torch.clamp(x, 0, 1)
    label = get_label(x)
    queries += 1
    if label != original_label:
        x_hat = x
        left_theta = theta
        flag = 1
    else:
        get_x_hat_in_2d.alpha -= minus_learning_rate
        get_x_hat_in_2d.alpha = max(lower, get_x_hat_in_2d.alpha)
        theta = max(theta, np.pi - 2 * get_x_hat_in_2d.alpha + np.pi / 64)

        x = x_o + d * (
            axis_unit1 * np.cos(theta) - axis_unit2 * np.sin(theta)
        ) / np.sin(get_x_hat_in_2d.alpha) * np.sin(get_x_hat_in_2d.alpha + theta)
        x = torch_dct.idct_2d(x)
        get_x_hat_in_2d.total += 1
        get_x_hat_in_2d.clamp += torch.sum(x > 1) + torch.sum(x < 0)
        x = torch.clamp(x, 0, 1)
        label = get_label(x)
        queries += 1
        if label != original_label:
            x_hat = x
            left_theta = theta
            flag = -1
        else:
            get_x_hat_in_2d.alpha -= minus_learning_rate
            get_x_hat_in_2d.alpha = max(get_x_hat_in_2d.alpha, lower)
            return x_hat, queries, False

    # binary search for beta
    theta = (left_theta + right_theta) / 2
    for i in range(max_iter):
        x = x_o + d * (
            axis_unit1 * np.cos(theta) + flag * axis_unit2 * np.sin(theta)
        ) / np.sin(get_x_hat_in_2d.alpha) * np.sin(get_x_hat_in_2d.alpha + theta)
        x = torch_dct.idct_2d(x)
        get_x_hat_in_2d.total += 1
        get_x_hat_in_2d.clamp += torch.sum(x > 1) + torch.sum(x < 0)
        x = torch.clamp(x, 0, 1)
        label = get_label(x)
        queries += 1
        if label != original_label:
            left_theta = theta
            x_hat = x
            get_x_hat_in_2d.alpha += plus_learning_rate
            return x_hat, queries, True
        else:
            get_x_hat_in_2d.alpha -= minus_learning_rate
            get_x_hat_in_2d.alpha = max(lower, get_x_hat_in_2d.alpha)
            theta = max(theta, np.pi - 2 * get_x_hat_in_2d.alpha + np.pi / 64)

            flag = -flag
            x = x_o + d * (
                axis_unit1 * np.cos(theta) + flag * axis_unit2 * np.sin(theta)
            ) / np.sin(get_x_hat_in_2d.alpha) * np.sin(get_x_hat_in_2d.alpha + theta)
            x = torch_dct.idct_2d(x)
            get_x_hat_in_2d.total += 1
            get_x_hat_in_2d.clamp += torch.sum(x > 1) + torch.sum(x < 0)
            x = torch.clamp(x, 0, 1)
            label = get_label(x)
            queries += 1
            if label != original_label:
                left_theta = theta
                x_hat = x
                get_x_hat_in_2d.alpha += plus_learning_rate
                get_x_hat_in_2d.alpha = min(upper, get_x_hat_in_2d.alpha)
                return x_hat, queries, True
            else:
                get_x_hat_in_2d.alpha -= minus_learning_rate
                get_x_hat_in_2d.alpha = max(lower, get_x_hat_in_2d.alpha)
                left_theta = max(np.pi - 2 * get_x_hat_in_2d.alpha, 0) + min(
                    np.pi / 16, get_x_hat_in_2d.alpha / 2
                )
                right_theta = theta
        theta = (left_theta + right_theta) / 2
    get_x_hat_in_2d.alpha += plus_learning_rate
    get_x_hat_in_2d.alpha = min(upper, get_x_hat_in_2d.alpha)
    return x_hat, queries, True


def get_x_hat_arbitary(
    x_o: torch.Tensor,
    model,
    video_shape: Tuple[int, int, int, int],
    aggregation: str,
    watermark_gt: Optional[np.ndarray],
    original_label: bool,
    init_x: torch.Tensor,
    dim_num: int,
    ratio_mask: float,
    max_queries: int,
    max_iter_num_in_2d: int,
    plus_learning_rate: float,
    minus_learning_rate: float,
    half_range: float,
    init_alpha: float,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """
    Main Triangle Attack loop.

    Returns:
        Tuple of (x_hat, queries)
    """
    T, C, H, W = video_shape

    # Helper function to get label
    def get_label(video_tensor: torch.Tensor) -> bool:
        video_detect = video_tensor.reshape(1, T, C, H, W).squeeze(0)
        logits = model.extract(video_detect)

        if watermark_gt is None:
            K = logits.shape[1]
            watermark_gt_local = np.array([0, 1] * (K // 2))
            if K % 2 == 1:
                watermark_gt_local = np.append(watermark_gt_local, 0)
        else:
            watermark_gt_local = watermark_gt

        watermark_gt_tensor = torch.from_numpy(watermark_gt_local).float().to(device)
        watermark_gt_tiled = watermark_gt_tensor.unsqueeze(0).expand(
            logits.shape[0], -1
        )
        predicted_bits = (logits >= 0).long()
        watermark_gt_bits = (watermark_gt_tiled >= 0.5).long()
        bitacc = (predicted_bits == watermark_gt_bits).float().mean().item()
        tau = get_tau_for_aggregation(aggregation, T)
        return bitacc >= tau

    if get_label(x_o) != original_label:
        return x_o, max_queries + 1

    x_adv = init_x
    x_hat = x_adv
    queries = 0

    while queries < max_queries:
        x_o2x_adv = torch_dct.dct_2d(get_difference(x_o, x_adv))
        axis_unit1 = x_o2x_adv / torch.norm(x_o2x_adv, p=2)
        direction, mask = get_orthogonal_1d_in_subspace(
            x_o2x_adv, dim_num, ratio_mask, H, W, device
        )
        axis_unit2 = direction / torch.norm(direction, p=2)
        x_hat, queries, changed = get_x_hat_in_2d(
            torch_dct.dct_2d(x_o),
            torch_dct.dct_2d(x_adv),
            axis_unit1,
            axis_unit2,
            model,
            video_shape,
            aggregation,
            watermark_gt,
            queries,
            original_label,
            max_iter=max_iter_num_in_2d,
            plus_learning_rate=plus_learning_rate,
            minus_learning_rate=minus_learning_rate,
            half_range=half_range,
            init_alpha=init_alpha,
            device=device,
        )
        x_adv = x_hat

        if queries >= max_queries:
            break

    return x_hat, queries


def triangle_attack(
    model,
    video: torch.Tensor,
    attack_type: str,
    watermark_gt: Optional[np.ndarray] = None,
    aggregation: str = "ba-mean",
    max_queries: int = 1000,
    ratio_mask: float = 0.1,
    dim_num: int = 1,
    max_iter_num_in_2d: int = 2,
    plus_learning_rate: float = 0.1,
    minus_learning_rate: float = 0.005,
    half_range: float = 0.1,
    init_alpha: float = np.pi / 2,
    device: torch.device = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Triangle Attack for video watermarking (original videomarkbench implementation).

    Triangle Attack is a label-based black-box attack that maintains the label
    while reducing perturbation size using DCT-based 2D subspace search.

    Args:
        model: VideoMarkBenchModel wrapper instance
        video: Video tensor of shape (T, C, H, W) or (1, T, C, H, W)
        attack_type: 'removal' or 'forgery'
        watermark_gt: Ground truth watermark pattern (if None, will be inferred)
        aggregation: Aggregation strategy name
        max_queries: Maximum number of queries (default: 1000)
        ratio_mask: Ratio of mask size (default: 0.1)
        dim_num: Number of picked dimensions (default: 1)
        max_iter_num_in_2d: Maximum iteration number in 2D subspace (default: 2)
        plus_learning_rate: Learning rate when success (default: 0.1)
        minus_learning_rate: Learning rate when fail (default: 0.005)
        half_range: Half range of alpha from pi/2 (default: 0.1)
        init_alpha: Initial angle of alpha (default: pi/2)
        device: Device to run on
        logger: Optional logger

    Returns:
        Tuple of (perturbed_video, n_queries)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if device is None:
        device = model.device

    # Ensure video is (T, C, H, W)
    if video.dim() == 5:
        video = video.squeeze(0)
    video = video.to(device)

    T, C, H, W = video.shape

    # Reshape video to [1, F*C, H, W] for Triangle Attack
    video_reshaped = video.reshape(1, T * C, H, W)  # (1, F*C, H, W)

    # Get threshold
    tau = get_tau_for_aggregation(aggregation, T)

    # Helper function to get label (True if watermark detected, False otherwise)
    def get_label(video_tensor: torch.Tensor) -> bool:
        """Get binary label: True if watermark detected, False otherwise."""
        video_detect = video_tensor.reshape(1, T, C, H, W).squeeze(0)  # (T, C, H, W)
        logits = model.extract(video_detect)  # (T, K)

        if watermark_gt is None:
            K = logits.shape[1]
            watermark_gt_local = np.array([0, 1] * (K // 2))
            if K % 2 == 1:
                watermark_gt_local = np.append(watermark_gt_local, 0)
        else:
            watermark_gt_local = watermark_gt

        watermark_gt_tensor = torch.from_numpy(watermark_gt_local).float().to(device)
        watermark_gt_tiled = watermark_gt_tensor.unsqueeze(0).expand(
            logits.shape[0], -1
        )
        predicted_bits = (logits >= 0).long()
        watermark_gt_bits = (watermark_gt_tiled >= 0.5).long()
        bitacc = (predicted_bits == watermark_gt_bits).float().mean().item()
        return bitacc >= tau

    # For removal: we want label to be False (no watermark)
    # For forgery: we want label to be True (has watermark)
    target_label = False if attack_type == "removal" else True

    # Initialize get_x_hat_in_2d state
    get_x_hat_in_2d.alpha = init_alpha
    get_x_hat_in_2d.total = 0
    get_x_hat_in_2d.clamp = 0

    # Random initialization
    def random_init(
        x: torch.Tensor, max_trials: int = 1000, sigma: float = 0.1
    ) -> torch.Tensor:
        """Find initial adversarial example."""
        if attack_type == "removal":
            seed = x.clone()
            # Try with increasing noise levels
            for sigma_level in [0.1, 0.2, 0.3, 0.5]:
                for _ in range(max_trials // 4):
                    noise = torch.randn_like(seed) * sigma_level
                    seed_test = torch.clamp(seed + noise, 0, 1)
                    if not get_label(seed_test):
                        return seed_test
            # If still not found, try with larger perturbation
            for _ in range(max_trials):
                noise = torch.randn_like(seed) * 0.1
                seed = torch.clamp(seed + noise, 0, 1)
                if not get_label(seed):
                    return seed
            # Last resort: return original with large noise
            logger.warning(
                "Could not find adversarial seed with small noise, using larger perturbation"
            )
            large_noise = torch.randn_like(seed) * 0.5
            return torch.clamp(seed + large_noise, 0, 1)
        else:  # forgery
            # For forgery, embed watermark into random video
            x_reshaped = x.reshape(1, T, C, H, W).squeeze(0)  # (T, C, H, W)
            seed = torch.rand_like(x_reshaped)
            # Embed watermark
            with torch.no_grad():
                if watermark_gt is None:
                    K = model.extract(seed).shape[1]
                    msg = torch.zeros(1, K, device=device)
                    msg[:, 1::2] = 1
                else:
                    msg = torch.from_numpy(watermark_gt).float().unsqueeze(0).to(device)
                embed_result = model.embed(seed, message=msg)
                seed = embed_result["watermarked_video"]  # (T, C, H, W)
            seed = seed.reshape(1, T * C, H, W)  # (1, F*C, H, W)
            return seed

    x_init = random_init(video_reshaped)

    # Main attack loop
    x_hat, n_queries = get_x_hat_arbitary(
        video_reshaped,
        model,
        (T, C, H, W),
        aggregation,
        watermark_gt,
        target_label,
        x_init,
        dim_num,
        ratio_mask,
        max_queries,
        max_iter_num_in_2d,
        plus_learning_rate,
        minus_learning_rate,
        half_range,
        init_alpha,
        device,
    )

    # Convert back to video format
    x_hat_reshaped = x_hat.reshape(1, T, C, H, W).squeeze(0)  # (T, C, H, W)

    return x_hat_reshaped, n_queries


def evaluate_blackbox_attacks(
    model,
    dataset: Iterator[Tuple[torch.Tensor, Dict]],
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
    max_videos: Optional[int] = None,
    max_frames_per_video: Optional[int] = None,
    attack_method: str = "square",
    evaluate_forgery: bool = True,
    evaluate_removal: bool = True,
    kinetics_dataset: Optional[Iterator[Tuple[torch.Tensor, Dict]]] = None,
    eps: float = 0.05,
    n_iters: int = 1000,
    p_init: float = 0.05,
) -> pd.DataFrame:
    """
    Evaluate blackbox attacks on watermarking model.

    Args:
        model: VideoMarkBenchModel wrapper instance
        dataset: Iterator of (video_tensor, metadata_dict) tuples
        output_dir: Directory to save results
        logger: Optional logger
        max_videos: Maximum number of videos to evaluate (None = all)
        max_frames_per_video: Maximum frames per video (None = all, default: 14 for forgery)
        attack_method: 'square' or 'triangle' (default: 'square')
        evaluate_forgery: Whether to evaluate forgery attacks (default: True)
        evaluate_removal: Whether to evaluate removal attacks (default: True)
        kinetics_dataset: Optional iterator for forgery attacks (uses same dataset if None)
        eps: Perturbation bound (default: 0.05)
        n_iters: Maximum iterations (default: 1000)
        p_init: Initial probability of changing pixels (default: 0.05)

    Returns:
        DataFrame with evaluation results
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if attack_method not in ["square", "triangle"]:
        raise ValueError(
            f"Unknown attack_method: {attack_method}. Must be 'square' or 'triangle'"
        )

    # Triangle Attack is now implemented

    results = []
    device = model.device

    # Get actual watermark size from model
    try:
        test_msg = model.model.get_random_msg()  # (1, K)
        secret_size = test_msg.shape[1]
    except Exception as e:
        logger.warning(
            f"Could not determine model's secret size, using default 96: {e}"
        )
        secret_size = 96

    # Generate ground truth watermark pattern (alternating 0,1)
    gt_watermark = torch.zeros(1, secret_size, device=device)
    gt_watermark[:, 1::2] = 1  # Set every other bit to 1: [0,1,0,1,...]
    embedded_message_long = gt_watermark.squeeze(0).long()  # Shape: (K,)

    video_count = 0
    kinetics_iter = iter(kinetics_dataset) if kinetics_dataset else None

    for video, metadata in tqdm(dataset, desc="Blackbox evaluation"):
        if max_videos is not None and video_count >= max_videos:
            break

        video_name = metadata.get("video_name", f"video_{video_count}")
        video = video.to(device)

        # Limit frames if specified
        if max_frames_per_video is not None:
            video = video[:max_frames_per_video]

        T = video.shape[0]
        logger.debug(f"Processing {video_name} with {T} frames")

        # ===== REMOVAL ATTACK =====
        if evaluate_removal:
            # Embed watermark first
            with torch.no_grad():
                embed_result = model.embed(video, message=gt_watermark)
                watermarked_video = embed_result["watermarked_video"]  # (T, C, H, W)

            # Apply attack
            try:
                # Convert gt_watermark to numpy for attack
                watermark_gt_np = gt_watermark.squeeze(0).cpu().numpy()  # (K,)

                if attack_method == "square":
                    perturbed_video, n_queries = square_attack_linf(
                        model,
                        watermarked_video,
                        "removal",
                        watermark_gt=watermark_gt_np,
                        eps=eps,
                        n_iters=n_iters,
                        p_init=p_init,
                        device=device,
                        logger=logger,
                    )
                elif attack_method == "triangle":
                    perturbed_video, n_queries = triangle_attack(
                        model,
                        watermarked_video,
                        "removal",
                        watermark_gt=watermark_gt_np,
                        aggregation="ba-mean",
                        max_queries=n_iters,
                        device=device,
                        logger=logger,
                    )
                else:
                    raise ValueError(f"Unsupported attack method: {attack_method}")

                # Extract logits from perturbed video
                with torch.no_grad():
                    logits = model.extract(perturbed_video)  # (T, K)

                # Apply aggregation strategies
                for agg_name, agg_func in AGGREGATION_STRATEGIES.items():
                    try:
                        detected = call_with_watermark_gt_tensor(
                            agg_func, logits, embedded_message_long
                        )
                        results.append(
                            {
                                "video": video_name,
                                "attack_type": "removal",
                                "attack_method": attack_method,
                                "aggregation": agg_name,
                                "detected": bool(detected),
                                "n_queries": n_queries,
                                "type": "blackbox",
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error in {agg_name} for {video_name} (removal): {e}"
                        )

            except Exception as e:
                logger.warning(f"Error in removal attack for {video_name}: {e}")

        # ===== FORGERY ATTACK =====
        if evaluate_forgery:
            # Use kinetics dataset for forgery attack
            if kinetics_iter is not None:
                try:
                    original_video, _ = next(kinetics_iter)
                    original_video = original_video.to(device)
                    if max_frames_per_video is not None:
                        original_video = original_video[:max_frames_per_video]
                    video_name_forgery = f"kinetics_{video_count}"
                except StopIteration:
                    logger.warning(
                        f"Kinetics dataset exhausted, skipping forgery for {video_name}"
                    )
                    video_count += 1
                    continue
            else:
                original_video = video
                video_name_forgery = video_name

            # Apply attack
            try:
                if attack_method == "square":
                    # For forgery, we don't know the exact watermark, so use None
                    # (will be inferred from logits as alternating pattern)
                    perturbed_video, n_queries = square_attack_linf(
                        model,
                        original_video,
                        "forgery",
                        watermark_gt=None,
                        eps=eps,
                        n_iters=n_iters,
                        p_init=p_init,
                        device=device,
                        logger=logger,
                    )
                elif attack_method == "triangle":
                    # For forgery, we don't know the exact watermark, so use None
                    perturbed_video, n_queries = triangle_attack(
                        model,
                        original_video,
                        "forgery",
                        watermark_gt=None,
                        aggregation="ba-mean",
                        max_queries=n_iters,
                        device=device,
                        logger=logger,
                    )
                else:
                    raise ValueError(f"Unsupported attack method: {attack_method}")

                # Extract logits from perturbed video
                with torch.no_grad():
                    logits = model.extract(perturbed_video)  # (T, K)

                # Apply aggregation strategies
                for agg_name, agg_func in AGGREGATION_STRATEGIES.items():
                    try:
                        detected = call_with_watermark_gt_tensor(
                            agg_func, logits, embedded_message_long
                        )
                        results.append(
                            {
                                "video": video_name_forgery,
                                "attack_type": "forgery",
                                "attack_method": attack_method,
                                "aggregation": agg_name,
                                "detected": bool(detected),
                                "n_queries": n_queries,
                                "type": "blackbox",
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error in {agg_name} for {video_name_forgery} (forgery): {e}"
                        )

            except Exception as e:
                logger.warning(f"Error in forgery attack for {video_name_forgery}: {e}")

        video_count += 1

    # Save results
    df = pd.DataFrame(results)
    if len(df) > 0:
        results_file = output_dir / "blackbox_results.csv"
        df.to_csv(results_file, index=False)
        logger.info(f"Saved blackbox results to {results_file}")

        # Calculate and save statistics
        from evaluate.utils import calculate_statistics

        stats = calculate_statistics(df, logger)
        stats_file = output_dir / "blackbox_stats.json"
        import json

        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved blackbox statistics to {stats_file}")
    else:
        logger.warning("No results to save")

    return df
