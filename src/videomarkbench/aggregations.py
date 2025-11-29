"""
Aggregation strategies for combining frame-level watermark logits.

Based on the original VideoMarkBench implementation.
Each aggregation function returns a boolean indicating detection success.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom
from typing import Optional
import torch


def logit_mean(decode: np.ndarray, watermark_gt: Optional[np.ndarray] = None) -> bool:
    """
    Mean aggregation: Average logits across frames.

    Args:
        decode: Frame-level logits of shape (T, K) where T is number of frames
                and K is number of bits
        watermark_gt: Ground truth watermark bits. If None, uses default pattern [0, 1] * 48

    Returns:
        Boolean indicating if detection is successful (bit accuracy >= 67/96)
    """
    if watermark_gt is None:
        watermark_gt = np.array([0, 1] * 48)

    decoded = np.mean(decode, axis=0)
    bitacc = np.sum((watermark_gt == (decoded >= 0))) / len(watermark_gt)
    return bitacc >= 67 / 96


def logit_g_median(
    decode: np.ndarray, watermark_gt: Optional[np.ndarray] = None
) -> bool:
    """
    Geometric median aggregation: Find geometric median of logits.

    Args:
        decode: Frame-level logits of shape (T, K)
        watermark_gt: Ground truth watermark bits. If None, uses default pattern [0, 1] * 48

    Returns:
        Boolean indicating if detection is successful
    """
    if watermark_gt is None:
        watermark_gt = np.array([0, 1] * 48)

    def objective(median):
        return np.sum(np.linalg.norm(decode - median, axis=1))

    initial_guess = np.mean(decode, axis=0)
    decoded = minimize(objective, initial_guess, method="Powell", tol=1e-5)
    bitacc = np.sum((watermark_gt == (decoded.x >= 0))) / len(watermark_gt)
    return bitacc >= 67 / 96


def bit_majority(decode: np.ndarray, watermark_gt: Optional[np.ndarray] = None) -> bool:
    """
    Majority vote aggregation: Hard predictions per frame, then majority vote.

    Args:
        decode: Frame-level logits of shape (T, K)
        watermark_gt: Ground truth watermark bits. If None, uses default pattern [0, 1] * 48

    Returns:
        Boolean indicating if detection is successful
    """
    if watermark_gt is None:
        watermark_gt = np.array([0, 1] * 48)

    decode = decode >= 0
    decoded = np.sum(decode, axis=0) > (len(decode) / 2)
    bitacc = np.sum((watermark_gt == decoded)) / len(watermark_gt)
    return bitacc >= 67 / 96


def bit_g_median(decode: np.ndarray, watermark_gt: Optional[np.ndarray] = None) -> bool:
    """
    Bit-level geometric median aggregation.

    Args:
        decode: Frame-level logits of shape (T, K)
        watermark_gt: Ground truth watermark bits. If None, uses default pattern [0, 1] * 48

    Returns:
        Boolean indicating if detection is successful
    """
    if watermark_gt is None:
        watermark_gt = np.array([0, 1] * 48)

    # Binarize logits: >= 0 -> 1, < 0 -> 0
    decode = decode.copy()
    decode = (decode >= 0).astype(np.float32)

    # Compute median for each bit position across frames
    decoded = np.median(decode, axis=0)

    # Convert median to binary: > 0.5 -> 1, <= 0.5 -> 0
    decoded = (decoded > 0.5).astype(np.int32)

    bitacc = np.sum((watermark_gt == decoded)) / len(watermark_gt)
    return bitacc >= 67 / 96


def bitacc_mean(decode: np.ndarray, watermark_gt: Optional[np.ndarray] = None) -> bool:
    """
    Mean bit accuracy aggregation: Compute bit accuracy per frame, then average.

    Args:
        decode: Frame-level logits of shape (T, K)
        watermark_gt: Ground truth watermark bits. If None, uses default pattern [0, 1] * 48

    Returns:
        Boolean indicating if detection is successful
    """
    if watermark_gt is None:
        watermark_gt = np.array([0, 1] * 48)

    watermark_gt = np.tile(watermark_gt, (decode.shape[0], 1))
    bitacc = np.mean((watermark_gt == (decode >= 0)))
    return bitacc >= 67 / 96


def bitacc_median(
    decode: np.ndarray, watermark_gt: Optional[np.ndarray] = None
) -> bool:
    """
    Median bit accuracy aggregation: Compute bit accuracy per frame, then take median.

    Args:
        decode: Frame-level logits of shape (T, K)
        watermark_gt: Ground truth watermark bits. If None, uses default pattern [0, 1] * 48

    Returns:
        Boolean indicating if detection is successful
    """
    if watermark_gt is None:
        watermark_gt = np.array([0, 1] * 48)

    bitacc = np.mean((decode >= 0) == watermark_gt, axis=1)
    median_bitacc = np.median(bitacc)
    return median_bitacc >= 67 / 96


def detection_threshold(
    decode: np.ndarray, watermark_gt: Optional[np.ndarray] = None
) -> bool:
    """
    Detection threshold aggregation: Count frames with bit accuracy >= threshold.

    Uses binomial distribution to determine threshold.

    Args:
        decode: Frame-level logits of shape (T, K)
        watermark_gt: Ground truth watermark bits. If None, uses default pattern [0, 1] * 48

    Returns:
        Boolean indicating if detection is successful
    """
    if watermark_gt is None:
        watermark_gt = np.array([0, 1] * 48)

    bitacc = np.mean((decode >= 0) == watermark_gt, axis=1)
    detected_frames = np.sum(bitacc >= 67 / 96)

    n = decode.shape[0]
    p = 0.0000661
    threshold = 10e-4
    k = n + 1
    for i in range(n + 1):
        if binom.sf(i - 1, n, p) <= threshold:
            k = i
            break

    return detected_frames >= k


def detection_majority(
    decode: np.ndarray, watermark_gt: Optional[np.ndarray] = None
) -> bool:
    """
    Detection majority aggregation: Count frames with bit accuracy >= threshold.
    Simple majority vote on frame-level detections.

    Args:
        decode: Frame-level logits of shape (T, K)
        watermark_gt: Ground truth watermark bits. If None, uses default pattern [0, 1] * 48

    Returns:
        Boolean indicating if detection is successful
    """
    if watermark_gt is None:
        watermark_gt = np.array([0, 1] * 48)

    bitacc = np.mean((decode >= 0) == watermark_gt, axis=1)
    detected_frames = np.sum(bitacc >= 67 / 96)

    return detected_frames > (decode.shape[0] // 2)


# Registry of aggregation strategies (matching original VideoMarkBench)
AGGREGATION_STRATEGIES = {
    "logit_mean": logit_mean,
    "logit_g_median": logit_g_median,
    "bit_majority": bit_majority,
    "bit_g_median": bit_g_median,
    "bitacc_mean": bitacc_mean,
    "bitacc_median": bitacc_median,
    "detection_threshold": detection_threshold,
    "detection_majority": detection_majority,
}


def get_aggregation_strategy(name: str):
    """
    Get an aggregation strategy function by name.

    Args:
        name: Name of the aggregation strategy

    Returns:
        Aggregation function
    """
    if name not in AGGREGATION_STRATEGIES:
        raise ValueError(
            f"Unknown aggregation strategy: {name}. "
            f"Available: {list(AGGREGATION_STRATEGIES.keys())}"
        )
    return AGGREGATION_STRATEGIES[name]


def aggregate(
    logits: torch.Tensor,
    strategy: str = "logit_mean",
    watermark_gt: Optional[np.ndarray] = None,
) -> bool:
    """
    Aggregate frame-level logits using specified strategy.

    Args:
        logits: Frame-level logits of shape (T, K) as torch.Tensor
        strategy: Name of aggregation strategy
        watermark_gt: Optional ground truth watermark bits

    Returns:
        Boolean indicating if detection is successful
    """
    # Convert torch tensor to numpy
    if isinstance(logits, torch.Tensor):
        decode = logits.detach().cpu().numpy()
    else:
        decode = logits

    agg_func = get_aggregation_strategy(strategy)
    return agg_func(decode, watermark_gt)
