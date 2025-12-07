"""
Common aggregation utilities for watermark evaluation.

This module provides shared functions for aggregation strategies used in both
common perturbations and whitebox attack evaluations.
"""

import torch
import numpy as np
from typing import Callable
from scipy.optimize import minimize
from scipy.stats import binom

# Constants for aggregation thresholds
BIT_ACCURACY_THRESHOLD = (
    67 / 96
)  # Minimum bit accuracy for detection (67 out of 96 bits)
RANDOM_DETECTION_PROBABILITY = 0.0000661  # Probability of random detection
DETECTION_THRESHOLD_P_VALUE = 10e-4  # P-value threshold for detection_threshold method


def call_with_watermark_gt_tensor(
    func: Callable, decode: torch.Tensor, watermark_gt: torch.Tensor
) -> bool:
    """
    Call aggregation function with custom watermark_gt (GPU-optimized tensor version).

    Args:
        func: Aggregation function (for name lookup)
        decode: Logits tensor of shape (T, K) on GPU
        watermark_gt: Ground truth watermark tensor of shape (K,) on GPU

    Returns:
        Boolean indicating if watermark is detected
    """
    func_name = func.__name__

    # Ensure both tensors are on the same device
    device = decode.device
    watermark_gt = watermark_gt.to(device)

    # Convert logits to bits: positive -> 1, negative/zero -> 0
    # VideoSeal convention: logits > 0 -> bit 1, logits <= 0 -> bit 0
    predicted_bits = (decode > 0).long()  # 1 for positive, 0 for negative/zero

    if func_name == "logit_mean":
        decoded = torch.mean(decode, dim=0)
        predicted = (decoded > 0).long()
        bitacc = torch.sum((watermark_gt == predicted).float()) / len(watermark_gt)
        return (bitacc >= BIT_ACCURACY_THRESHOLD).item()

    elif func_name == "logit_g_median":
        # This requires scipy optimization, so convert to CPU for this step
        decode_np = decode.detach().cpu().numpy()
        watermark_gt_np = watermark_gt.detach().cpu().numpy()

        def objective(median):
            return np.sum(np.linalg.norm(decode_np - median, axis=1))

        initial_guess = np.mean(decode_np, axis=0)
        decoded = minimize(objective, initial_guess, method="Powell", tol=1e-5)
        predicted = (decoded.x > 0).astype(int)
        bitacc = np.sum((watermark_gt_np == predicted)) / len(watermark_gt_np)
        return bitacc >= BIT_ACCURACY_THRESHOLD

    elif func_name == "bit_majority":
        # Majority vote across frames
        decoded = (torch.sum(predicted_bits, dim=0) > (decode.shape[0] / 2)).long()
        bitacc = torch.sum((watermark_gt == decoded).float()) / len(watermark_gt)
        return (bitacc >= BIT_ACCURACY_THRESHOLD).item()

    elif func_name == "bit_g_median":
        decoded = torch.median(predicted_bits.float(), dim=0)[0].long()
        bitacc = torch.sum((watermark_gt == decoded).float()) / len(watermark_gt)
        return (bitacc >= BIT_ACCURACY_THRESHOLD).item()

    elif func_name == "bitacc_mean":
        watermark_gt_tiled = watermark_gt.unsqueeze(0).expand(decode.shape[0], -1)
        bitacc = torch.mean((watermark_gt_tiled == predicted_bits).float())
        return (bitacc >= BIT_ACCURACY_THRESHOLD).item()

    elif func_name == "bitacc_median":
        watermark_gt_tiled = watermark_gt.unsqueeze(0).expand(decode.shape[0], -1)
        bitacc = torch.mean((watermark_gt_tiled == predicted_bits).float(), dim=1)
        median_bitacc = torch.median(bitacc)
        return (median_bitacc >= BIT_ACCURACY_THRESHOLD).item()

    elif func_name == "detection_threshold":
        watermark_gt_tiled = watermark_gt.unsqueeze(0).expand(decode.shape[0], -1)
        bitacc = torch.mean((watermark_gt_tiled == predicted_bits).float(), dim=1)
        detected_frames = torch.sum((bitacc >= BIT_ACCURACY_THRESHOLD).long()).item()

        n = decode.shape[0]
        p = RANDOM_DETECTION_PROBABILITY
        threshold = DETECTION_THRESHOLD_P_VALUE
        k = n  # Default if no threshold found
        for i in range(n + 1):
            if binom.sf(i - 1, n, p) <= threshold:
                k = i
                break

        return detected_frames >= k

    elif func_name == "detection_majority":
        watermark_gt_tiled = watermark_gt.unsqueeze(0).expand(decode.shape[0], -1)
        bitacc = torch.mean((watermark_gt_tiled == predicted_bits).float(), dim=1)
        detected_frames = torch.sum((bitacc >= BIT_ACCURACY_THRESHOLD).long()).item()
        return detected_frames > (decode.shape[0] // 2)

    else:
        # Fallback: convert to numpy and use numpy version
        decode_np = decode.detach().cpu().numpy()
        watermark_gt_np = watermark_gt.detach().cpu().numpy()
        return call_with_watermark_gt_numpy(func, decode_np, watermark_gt_np)


def call_with_watermark_gt_numpy(
    func: Callable, decode: np.ndarray, watermark_gt: np.ndarray
) -> bool:
    """
    Call aggregation function with custom watermark_gt (CPU/numpy version for fallback).

    Args:
        func: Aggregation function (for name lookup)
        decode: Logits numpy array of shape (T, K)
        watermark_gt: Ground truth watermark numpy array of shape (K,)

    Returns:
        Boolean indicating if watermark is detected
    """
    func_name = func.__name__

    # Convert logits to bits: positive -> 1, negative/zero -> 0
    # VideoSeal convention: logits > 0 -> bit 1, logits <= 0 -> bit 0
    predicted_bits = (decode > 0).astype(int)  # 1 for positive, 0 for negative/zero

    if func_name == "logit_mean":
        decoded = np.mean(decode, axis=0)
        predicted = (decoded > 0).astype(int)
        bitacc = np.sum((watermark_gt == predicted)) / len(watermark_gt)
        return bitacc >= BIT_ACCURACY_THRESHOLD

    elif func_name == "logit_g_median":

        def objective(median):
            return np.sum(np.linalg.norm(decode - median, axis=1))

        initial_guess = np.mean(decode, axis=0)
        decoded = minimize(objective, initial_guess, method="Powell", tol=1e-5)
        predicted = (decoded.x > 0).astype(int)
        bitacc = np.sum((watermark_gt == predicted)) / len(watermark_gt)
        return bitacc >= BIT_ACCURACY_THRESHOLD

    elif func_name == "bit_majority":
        decoded = (np.sum(predicted_bits, axis=0) > (len(decode) / 2)).astype(int)
        bitacc = np.sum((watermark_gt == decoded)) / len(watermark_gt)
        return bitacc >= BIT_ACCURACY_THRESHOLD

    elif func_name == "bit_g_median":
        decoded = np.median(predicted_bits, axis=0).astype(int)
        bitacc = np.sum((watermark_gt == decoded)) / len(watermark_gt)
        return bitacc >= BIT_ACCURACY_THRESHOLD

    elif func_name == "bitacc_mean":
        watermark_gt_tiled = np.tile(watermark_gt, (decode.shape[0], 1))
        bitacc = np.mean((watermark_gt_tiled == predicted_bits))
        return bitacc >= BIT_ACCURACY_THRESHOLD

    elif func_name == "bitacc_median":
        watermark_gt_tiled = np.tile(watermark_gt, (decode.shape[0], 1))
        bitacc = np.mean((watermark_gt_tiled == predicted_bits), axis=1)
        median_bitacc = np.median(bitacc)
        return median_bitacc >= BIT_ACCURACY_THRESHOLD

    elif func_name == "detection_threshold":
        watermark_gt_tiled = np.tile(watermark_gt, (decode.shape[0], 1))
        bitacc = np.mean((watermark_gt_tiled == predicted_bits), axis=1)
        detected_frames = np.sum(bitacc >= BIT_ACCURACY_THRESHOLD)

        n = decode.shape[0]
        p = RANDOM_DETECTION_PROBABILITY
        threshold = DETECTION_THRESHOLD_P_VALUE
        k = n
        for i in range(n + 1):
            if binom.sf(i - 1, n, p) <= threshold:
                k = i
                break

        return detected_frames >= k

    elif func_name == "detection_majority":
        watermark_gt_tiled = np.tile(watermark_gt, (decode.shape[0], 1))
        bitacc = np.mean((watermark_gt_tiled == predicted_bits), axis=1)
        detected_frames = np.sum(bitacc >= BIT_ACCURACY_THRESHOLD)
        return detected_frames > (decode.shape[0] // 2)

    else:
        # Fallback to original function
        return func(decode)
