"""
Whitebox attacks evaluator.

Evaluates watermarking models under whitebox attacks (WEvade) using both
standard and adaptive versions. Supports both removal and forgery attacks
with all aggregation strategies from the original VideoMarkBench.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Iterator, Tuple
import pandas as pd
import logging
from tqdm import tqdm
from easydict import EasyDict

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

# Import original Wevade functions from videomarkbench
from whitebox_perturbations.VideoSeal.Wevade import (
    removal as wevade_removal,
    forgery as wevade_forgery,
)  # noqa: E402
from whitebox_perturbations.VideoSeal.Wevade_adaptive import (
    removal as wevade_adaptive_removal,
    forgery as wevade_adaptive_forgery,
)  # noqa: E402

# Import common aggregation utilities
from .aggregation_utils import (
    call_with_watermark_gt_tensor,
    call_with_watermark_gt_numpy,
)


# Original Wevade functions are imported directly from videomarkbench
# No need for custom implementations - we use the original functions


def _call_with_watermark_gt_tensor(
    func, decode: torch.Tensor, watermark_gt: torch.Tensor
) -> bool:
    """
    Call aggregation function with custom watermark_gt (GPU-optimized tensor version).

    Wrapper around common aggregation_utils for backward compatibility.

    Args:
        func: Aggregation function
        decode: Logits tensor of shape (T, K)
        watermark_gt: Ground truth watermark tensor of shape (K,)

    Returns:
        Boolean indicating if watermark is detected
    """
    return call_with_watermark_gt_tensor(func, decode, watermark_gt)


def _call_with_watermark_gt_numpy(
    func, decode: np.ndarray, watermark_gt: np.ndarray
) -> bool:
    """
    Call aggregation function with custom watermark_gt (CPU/numpy version for fallback).

    Wrapper around common aggregation_utils for backward compatibility.

    Args:
        func: Aggregation function
        decode: Logits numpy array of shape (T, K)
        watermark_gt: Ground truth watermark numpy array of shape (K,)

    Returns:
        Boolean indicating if watermark is detected
    """
    return call_with_watermark_gt_numpy(func, decode, watermark_gt)


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


def evaluate_whitebox_attacks(
    model,
    dataset: Iterator[Tuple[torch.Tensor, Dict]],
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
    max_videos: Optional[int] = None,
    max_frames_per_video: Optional[int] = None,
    wevade_version: str = "standard",
    wevade_iteration: int = 200,
    wevade_epsilon: float = 4 / 96,
    wevade_alpha: float = 0.01,
    wevade_rb: float = 0.1,
    wevade_type: str = "WEvade-W-I",
    secret_size: int = 96,
    evaluate_forgery: bool = True,
    kinetics_dataset: Optional[Iterator[Tuple[torch.Tensor, Dict]]] = None,
) -> pd.DataFrame:
    """
    Evaluate whitebox attacks (WEvade) on watermarking model.

    Supports both standard and adaptive WEvade implementations:
    - Standard: Full PGD-style attack with MSE loss, configurable parameters
    - Adaptive: Simplified attack with fixed parameters (20 iterations, lr=0.02)

    Args:
        model: VideoMarkBenchModel wrapper instance
        dataset: Iterator of (video_tensor, metadata_dict) tuples
        output_dir: Directory to save results
        logger: Optional logger
        max_videos: Maximum number of videos to evaluate (None = all)
        max_frames_per_video: Maximum frames per video (None = all)
        wevade_version: 'standard' or 'adaptive' (default: 'standard')
        wevade_iteration: Max iterations for standard version (default: 200)
        wevade_epsilon: Early stopping threshold (default: 4/96)
        wevade_alpha: Learning rate (default: 0.01)
        wevade_rb: Perturbation bound (default: 0.1)
        wevade_type: 'WEvade-W-I' or 'WEvade-W-II' (default: 'WEvade-W-I')
        secret_size: Watermark size (default: 96)
        evaluate_forgery: Whether to evaluate forgery attacks (default: True)

    Returns:
        DataFrame with evaluation results
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Prepare attack arguments as EasyDict (required by original Wevade functions)
    attack_args = EasyDict(
        {
            "iteration": wevade_iteration,
            "epsilon": wevade_epsilon,
            "alpha": wevade_alpha,
            "rb": wevade_rb,
            "WEvade_type": wevade_type,
            "secret_size": secret_size,
        }
    )

    # Select attack functions based on version (using original functions from videomarkbench)
    if wevade_version == "standard":
        removal_fn = wevade_removal
        forgery_fn = wevade_forgery
        criterion = nn.MSELoss().to(model.device)
        logger.info(
            f"Using original standard WEvade (iterations={wevade_iteration}, alpha={wevade_alpha}, rb={wevade_rb})"
        )
    elif wevade_version == "adaptive":
        removal_fn = wevade_adaptive_removal
        forgery_fn = wevade_adaptive_forgery
        criterion = None  # Not used in adaptive version
        logger.info("Using original adaptive WEvade (fixed 20 iterations, lr=0.02)")
    else:
        raise ValueError(
            f"Unknown wevade_version: {wevade_version}. Must be 'standard' or 'adaptive'"
        )

    # Get actual model (unwrap VideoMarkBenchModel wrapper if needed)
    actual_model = model.model if hasattr(model, "model") else model

    results = []
    device = model.device

    # Get actual watermark size from model
    try:
        # Try to get random message from model to determine actual size
        test_msg = model.model.get_random_msg()  # (1, K)
        actual_secret_size = test_msg.shape[1]
        if secret_size != actual_secret_size:
            logger.warning(
                f"Secret size mismatch: requested {secret_size}, but model uses {actual_secret_size}. "
                f"Using model's actual size: {actual_secret_size}"
            )
            secret_size = actual_secret_size
            # Update attack_args
            attack_args["secret_size"] = secret_size
    except Exception as e:
        logger.warning(
            f"Could not determine model's secret size, using provided {secret_size}: {e}"
        )

    # Generate ground truth watermark pattern (alternating 0,1)
    gt_watermark = torch.zeros(1, secret_size, device=device)
    gt_watermark[:, 1::2] = 1  # Set every other bit to 1: [0,1,0,1,...]

    # Convert to long format for aggregation: (K,)
    embedded_message_long = gt_watermark.squeeze(0).long()  # Shape: (K,)

    video_count = 0
    for video, metadata in tqdm(dataset, desc="Whitebox evaluation"):
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
        # Embed watermark first
        with torch.no_grad():
            embed_result = model.embed(video, message=gt_watermark)
            watermarked_video = embed_result["watermarked_video"]  # (T, C, H, W)

        # Attack each frame
        attacked_frames_logits = []
        for frame_idx in range(T):
            # Clone frame and enable gradient for attack
            frame = (
                watermarked_video[frame_idx : frame_idx + 1].clone().detach()
            )  # (1, C, H, W)

            try:
                if wevade_version == "standard":
                    # Original function returns: (bit_acc_groundtruth, bound, ssim, psnr, success, decoded_logits)
                    bit_acc, bound, ssim, psnr, success, decoded_logits = removal_fn(
                        frame, gt_watermark, actual_model, criterion, attack_args
                    )
                else:
                    # Adaptive function returns: (bit_acc_groundtruth, bound, ssim, psnr, decoded_logits)
                    bit_acc, bound, ssim, psnr, decoded_logits = removal_fn(
                        frame, gt_watermark, actual_model, attack_args
                    )

                # Convert decoded_logits to tensor if needed
                if isinstance(decoded_logits, np.ndarray):
                    decoded_logits = torch.from_numpy(decoded_logits)
                elif isinstance(decoded_logits, torch.Tensor):
                    decoded_logits = decoded_logits.cpu()
                else:
                    decoded_logits = torch.tensor(decoded_logits).cpu()

                # Ensure shape is (K,)
                if decoded_logits.dim() > 1:
                    decoded_logits = decoded_logits.squeeze()

                attacked_frames_logits.append(decoded_logits)

            except Exception as e:
                logger.warning(
                    f"Error in removal attack for {video_name} frame {frame_idx}: {e}"
                )
                # Use original frame's logits as fallback
                with torch.no_grad():
                    logits = model.extract(frame)  # (1, K)
                    attacked_frames_logits.append(logits.squeeze(0).cpu())

        # Aggregate logits across frames
        if len(attacked_frames_logits) > 0:
            all_logits = torch.stack(attacked_frames_logits)  # (T, K)

            # Apply aggregation strategies
            for agg_name, agg_func in AGGREGATION_STRATEGIES.items():
                try:
                    detected = _call_with_watermark_gt_tensor(
                        agg_func, all_logits, embedded_message_long
                    )
                    results.append(
                        {
                            "video": video_name,
                            "attack_type": "removal",
                            "wevade_version": wevade_version,
                            "aggregation": agg_name,
                            "detected": bool(detected),
                            "type": "whitebox",
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Error in {agg_name} for {video_name} (removal): {e}"
                    )

        # ===== FORGERY ATTACK =====
        if evaluate_forgery:
            # Use kinetics dataset for forgery attack (as per original implementation)
            # Note: kinetics_dataset iterator is recreated in main.py before calling this function
            # to avoid exhaustion from common_evaluator. If provided, use it; otherwise use the same dataset
            if kinetics_dataset is not None:
                # Get next video from kinetics dataset
                try:
                    kinetics_video, kinetics_metadata = next(kinetics_dataset)
                    original_video = kinetics_video.to(device)
                    video_name_forgery = kinetics_metadata.get(
                        "video_name", f"kinetics_{video_count}"
                    )
                    T_forgery = original_video.shape[0]
                    if max_frames_per_video is not None:
                        T_forgery = min(T_forgery, max_frames_per_video)
                        original_video = original_video[:T_forgery]
                except StopIteration:
                    logger.warning(
                        f"Kinetics dataset exhausted, skipping forgery for {video_name}"
                    )
                    continue
            else:
                # Fallback: use same dataset (for backward compatibility)
                original_video = video
                video_name_forgery = video_name
                T_forgery = T

            # Attack each frame
            attacked_frames_logits = []
            for frame_idx in range(T_forgery):
                # Clone frame and enable gradient for attack
                frame = (
                    original_video[frame_idx : frame_idx + 1].clone().detach()
                )  # (1, C, H, W)

                try:
                    if wevade_version == "standard":
                        # Original function returns: (bit_acc_groundtruth, bound, ssim, psnr, success, decoded_logits)
                        (
                            bit_acc,
                            bound,
                            ssim,
                            psnr,
                            success,
                            decoded_logits,
                        ) = forgery_fn(
                            frame,
                            gt_watermark,
                            actual_model,
                            criterion,
                            attack_args,
                        )
                    else:
                        # Adaptive function returns: (bit_acc_groundtruth, bound, ssim, psnr, decoded_logits)
                        bit_acc, bound, ssim, psnr, decoded_logits = forgery_fn(
                            frame, gt_watermark, actual_model, attack_args
                        )

                    # Convert decoded_logits to tensor if needed
                    if isinstance(decoded_logits, np.ndarray):
                        decoded_logits = torch.from_numpy(decoded_logits)
                    elif isinstance(decoded_logits, torch.Tensor):
                        decoded_logits = decoded_logits.cpu()
                    else:
                        decoded_logits = torch.tensor(decoded_logits).cpu()

                    # Ensure shape is (K,)
                    if decoded_logits.dim() > 1:
                        decoded_logits = decoded_logits.squeeze()

                    attacked_frames_logits.append(decoded_logits)

                except Exception as e:
                    logger.warning(
                        f"Error in forgery attack for {video_name_forgery} frame {frame_idx}: {e}"
                    )
                    # Use original frame's logits as fallback
                    with torch.no_grad():
                        logits = model.extract(frame)  # (1, K)
                        attacked_frames_logits.append(logits.squeeze(0).cpu())

            # Aggregate logits across frames
            if len(attacked_frames_logits) > 0:
                all_logits = torch.stack(attacked_frames_logits)  # (T, K)

                # Apply aggregation strategies
                for agg_name, agg_func in AGGREGATION_STRATEGIES.items():
                    try:
                        detected = _call_with_watermark_gt_tensor(
                            agg_func, all_logits, embedded_message_long
                        )
                        results.append(
                            {
                                "video": video_name_forgery,
                                "attack_type": "forgery",
                                "wevade_version": wevade_version,
                                "aggregation": agg_name,
                                "detected": bool(detected),
                                "type": "whitebox",
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error in {agg_name} for {video_name} (forgery): {e}"
                        )

        video_count += 1

    # Save results
    df = pd.DataFrame(results)
    if len(df) > 0:
        results_file = output_dir / "whitebox_results.csv"
        df.to_csv(results_file, index=False)
        logger.info(f"Saved whitebox results to {results_file}")

        # Calculate and save statistics
        from evaluate.utils import calculate_statistics

        stats = calculate_statistics(df, logger)
        stats_file = output_dir / "whitebox_stats.json"
        import json

        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved whitebox statistics to {stats_file}")
    else:
        logger.warning("No results to save")

    return df
