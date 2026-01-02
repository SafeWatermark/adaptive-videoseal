"""
Common perturbations evaluator.

Evaluates watermarking models under common perturbations (JPEG, Gaussian, Crop, etc.)
using all aggregation strategies from the original VideoMarkBench.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Iterator, Tuple
import pandas as pd
import logging
from tqdm import tqdm
import tempfile
import cv2
import ffmpeg

# Import albumentations for faster image processing (required)
import albumentations as A

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

from common_perturabtions.crop import Crop  # noqa: E402
from common_perturabtions.frame_average import FrameAverage  # noqa: E402
from common_perturabtions.frame_remove import FrameRemove  # noqa: E402
from common_perturabtions.frame_switch import FrameSwitch  # noqa: E402
from common_perturabtions.gaussian import Gaussian  # noqa: E402
from common_perturabtions.gaussian_blur import GaussianBlur  # noqa: E402
from common_perturabtions.jpeg import JPEG  # noqa: E402


def _video_to_numpy_uint8(video: torch.Tensor) -> np.ndarray:
    """
    Convert video tensor to numpy array in albumentations format.

    Args:
        video: Video tensor of shape (T, C, H, W) in [0, 1]

    Returns:
        Video numpy array of shape (T, H, W, C) in uint8 [0, 255]
    """
    return (video.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)


def _numpy_uint8_to_video(
    video_np: np.ndarray, original_device: torch.device
) -> torch.Tensor:
    """
    Convert numpy array back to video tensor.

    Args:
        video_np: Video numpy array of shape (T, H, W, C) in uint8 [0, 255]
        original_device: Device to move tensor to

    Returns:
        Video tensor of shape (T, C, H, W) in [0, 1]
    """
    return torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0


def _apply_albumentations_transform(
    video: torch.Tensor, transform: A.BasicTransform, clamp: bool = False
) -> torch.Tensor:
    """
    Apply albumentations transform to video using video augmentation API.

    Uses `images` (plural) argument to process all frames at once with consistent parameters.
    Reference: https://albumentations.ai/docs/3-basic-usage/video-augmentation/

    Args:
        video: Video tensor of shape (T, C, H, W) in [0, 1]
        transform: albumentations transform to apply
        clamp: Whether to clamp output to [0, 1] range

    Returns:
        Perturbed video tensor of shape (T, C, H, W) in [0, 1]
    """
    # Convert to numpy format
    video_np = _video_to_numpy_uint8(video)

    # Apply transform using video augmentation API
    augmented = transform(images=video_np)
    perturbed_np = augmented["images"]

    # Convert back to tensor
    perturbed_tensor = _numpy_uint8_to_video(perturbed_np, video.device)

    if clamp:
        perturbed_tensor = perturbed_tensor.clamp(0.0, 1.0)

    return perturbed_tensor.to(video.device)


class MPEG4Wrapper(nn.Module):
    """
    Wrapper for MPEG4 perturbation that fixes bugs in the original submodule.

    The original MPEG4 class has several issues:
    1. Missing import torch.nn as nn
    2. Wrong super() call (uses Crop instead of MPEG4)
    3. Doesn't return the processed video

    This wrapper fixes all these issues.
    """

    def __init__(
        self, quality: int, codec: str = "libx264", temp_dir: Optional[Path] = None
    ):
        """
        Initialize MPEG4 wrapper.

        Args:
            quality: CRF quality value (lower = higher quality, typically 18-28)
            codec: Video codec to use (default: libx264)
            temp_dir: Directory for temporary files (default: system temp)
        """
        super(MPEG4Wrapper, self).__init__()
        self.quality = quality
        self.codec = codec
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply MPEG4 compression to video (optimized version).

        Args:
            video: Video tensor of shape (T, C, H, W) in [0, 1]

        Returns:
            Compressed video tensor of same shape
        """
        # Create unique temporary file
        import uuid

        temp_file = self.temp_dir / f"mpeg4_{uuid.uuid4().hex}.mp4"

        try:
            # Convert video to numpy array (batch conversion is faster)
            frames = (video * 255).byte().cpu().numpy()
            frames = np.transpose(frames, (0, 2, 3, 1))  # (T, H, W, C)

            # Get video dimensions
            T, H, W, C = frames.shape
            fps = 25  # Default FPS

            # Optimized ffmpeg pipeline with faster encoding settings
            process = (
                ffmpeg.input(
                    "pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{W}x{H}", r=fps
                )
                .output(
                    str(temp_file),
                    vcodec=self.codec,
                    pix_fmt="yuv420p",
                    crf=self.quality,
                    preset="ultrafast",  # Faster encoding
                    movflags="faststart",  # Optimize for streaming
                    **{"threads": "0"},  # Use all available threads
                )
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stderr=True, quiet=True)
            )

            # Write all frames at once (faster than frame-by-frame)
            frames_bytes = frames.tobytes()
            process.stdin.write(frames_bytes)
            process.stdin.close()
            process.wait()

            # Read compressed video back using decord if available (faster than cv2)
            try:
                from decord import VideoReader, cpu

                vr = VideoReader(
                    str(temp_file), width=W, height=H, num_threads=4, ctx=cpu(0)
                )
                num_frames = len(vr)
                if num_frames > 0:
                    # Get all frames at once
                    vid_np = vr.get_batch(
                        range(num_frames)
                    ).asnumpy()  # (T, H, W, C) in RGB
                    # Convert to tensor: (T, H, W, C) -> (T, C, H, W)
                    compressed_video = torch.from_numpy(vid_np).float()
                    compressed_video = (
                        compressed_video.permute(0, 3, 1, 2) / 255.0
                    )  # (T, C, H, W)
                else:
                    raise ValueError("No frames read from compressed video")
            except (ImportError, Exception):
                # Fallback to cv2 if decord is not available
                cap = cv2.VideoCapture(str(temp_file))
                compressed_frames = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert BGR to RGB and resize if needed
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if frame.shape[:2] != (H, W):
                        frame = cv2.resize(frame, (W, H))

                    # Convert to tensor: (H, W, C) -> (C, H, W)
                    frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
                    compressed_frames.append(frame)

                cap.release()

                if len(compressed_frames) == 0:
                    raise ValueError("No frames read from compressed video")

                # Stack frames: (T, C, H, W)
                compressed_video = torch.stack(compressed_frames, dim=0)

            # Ensure same number of frames (handle frame count mismatch)
            if compressed_video.shape[0] != video.shape[0]:
                # Interpolate or pad if needed
                if compressed_video.shape[0] < video.shape[0]:
                    # Repeat last frame if compressed video is shorter
                    last_frame = compressed_video[-1:].repeat(
                        video.shape[0] - compressed_video.shape[0], 1, 1, 1
                    )
                    compressed_video = torch.cat([compressed_video, last_frame], dim=0)
                else:
                    # Take first T frames if compressed video is longer
                    compressed_video = compressed_video[: video.shape[0]]

            return compressed_video.to(video.device)

        finally:
            # Clean up temporary file
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass


# Import common aggregation utilities
from .aggregation_utils import (
    call_with_watermark_gt_tensor,
    call_with_watermark_gt_numpy,
)


def _call_with_watermark_gt_tensor(
    func, decode: torch.Tensor, watermark_gt: torch.Tensor
) -> bool:
    """
    GPU-optimized version of aggregation function using PyTorch tensors.

    Wrapper around common aggregation_utils for backward compatibility.

    Args:
        func: Aggregation function (for name lookup)
        decode: Logits tensor of shape (T, K) on GPU
        watermark_gt: Ground truth watermark tensor of shape (K,) on GPU

    Returns:
        Boolean indicating if watermark is detected
    """
    return call_with_watermark_gt_tensor(func, decode, watermark_gt)


def _call_with_watermark_gt(func, decode: np.ndarray, watermark_gt: np.ndarray) -> bool:
    """
    Call aggregation function with custom watermark_gt (CPU/numpy version for fallback).

    Wrapper around common aggregation_utils for backward compatibility.

    Args:
        func: Aggregation function (for name lookup)
        decode: Logits numpy array of shape (T, K)
        watermark_gt: Ground truth watermark numpy array of shape (K,)

    Returns:
        Boolean indicating if watermark is detected
    """
    return call_with_watermark_gt_numpy(func, decode, watermark_gt)


# Aggregation strategies - will be wrapped with actual watermark_gt at runtime
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


def get_common_perturbations(temp_dir: Optional[Path] = None) -> List[torch.nn.Module]:
    """
    Get list of common perturbations.

    Args:
        temp_dir: Directory for temporary files (for MPEG4)

    Returns:
        List of perturbation instances
    """
    perturbations = []
    # JPEG
    perturbations.append(JPEG(quality=90))
    perturbations.append(JPEG(quality=75))
    perturbations.append(JPEG(quality=50))
    # Gaussian
    perturbations.append(Gaussian(std=0.01))
    perturbations.append(Gaussian(std=0.05))
    # GaussianBlur
    perturbations.append(GaussianBlur(std=1.0))
    perturbations.append(GaussianBlur(std=2.0))
    # Crop
    perturbations.append(Crop(crop_ratio=0.9))
    perturbations.append(Crop(crop_ratio=0.8))
    # MPEG4
    perturbations.append(MPEG4Wrapper(quality=23, temp_dir=temp_dir))
    perturbations.append(MPEG4Wrapper(quality=28, temp_dir=temp_dir))
    perturbations.append(MPEG4Wrapper(quality=33, temp_dir=temp_dir))
    # FrameAverage
    perturbations.append(FrameAverage(num_frames=2))
    perturbations.append(FrameAverage(num_frames=3))
    # FrameRemove
    perturbations.append(FrameRemove(p=0.1))
    perturbations.append(FrameRemove(p=0.2))
    # FrameSwitch
    perturbations.append(FrameSwitch(p=0.1))
    perturbations.append(FrameSwitch(p=0.2))

    return perturbations


def evaluate_common_perturbations(
    model,
    dataset: Iterator[Tuple[torch.Tensor, Dict]],
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
    max_videos: Optional[int] = None,
    max_frames_per_video: Optional[int] = None,
    evaluate_forgery: bool = True,
    kinetics_dataset: Optional[Iterator[Tuple[torch.Tensor, Dict]]] = None,
) -> pd.DataFrame:
    """
    Evaluate model under common perturbations.

    Args:
        model: Watermarking model (must have embed() and extract() methods)
        dataset: Dataset iterator yielding (video, video_info) tuples
        output_dir: Directory to save results
        logger: Optional logger
        max_videos: Maximum number of videos to evaluate
        max_frames_per_video: Maximum frames per video
        evaluate_forgery: Whether to evaluate forgery attacks (default: True)

    Returns:
        DataFrame with evaluation results
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Create temp directory for MPEG4 (will be cleaned up after evaluation)
    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp(prefix="videomarkbench_mpeg4_"))

    try:
        perturbations = get_common_perturbations(temp_dir=temp_dir)
        results = []

        logger.info(f"Evaluating {len(perturbations)} common perturbations")
        if evaluate_forgery:
            logger.info("Evaluating both removal and forgery attacks")

        video_count = 0
        kinetics_iter = None
        if kinetics_dataset is not None:
            kinetics_iter = iter(kinetics_dataset)

        for video, video_info in dataset:
            if max_videos and video_count >= max_videos:
                break

            video_count += 1
            video_name = video_info.get("name", "unknown")
            logger.info(f"Processing video {video_count}: {video_name}")

            # Limit frames if specified
            if max_frames_per_video and video.shape[0] > max_frames_per_video:
                video = video[:max_frames_per_video]

            # Embed watermark for removal attack (cache results for reuse)
            # Note: Removal attack uses videomarkbench dataset (current video)
            with torch.no_grad():
                embed_result = model.embed(video)
                watermarked_video = embed_result["watermarked_video"]
                embedded_message = embed_result["message"]
                # Pre-convert to long for reuse (avoid repeated conversions)
                embedded_message_long = embedded_message.long()

            # Get kinetics video for forgery attack (if available)
            original_video_forgery = None
            video_name_forgery = video_name
            if evaluate_forgery and kinetics_iter is not None:
                try:
                    kinetics_video, kinetics_metadata = next(kinetics_iter)
                    original_video_forgery = kinetics_video.to(video.device)
                    video_name_forgery = kinetics_metadata.get(
                        "video_name", f"kinetics_{video_count}"
                    )
                    # Limit frames if specified
                    if (
                        max_frames_per_video
                        and original_video_forgery.shape[0] > max_frames_per_video
                    ):
                        original_video_forgery = original_video_forgery[
                            :max_frames_per_video
                        ]
                except StopIteration:
                    logger.warning(
                        f"Kinetics dataset exhausted, skipping forgery for {video_name}"
                    )
                    original_video_forgery = None

            # Note: No caching of perturbation results to save memory
            # Each perturbation is only used once, so caching doesn't help

            # Ensure video and watermarked_video are on the same device
            # (Already on device from model.embed, so this is just for consistency)
            device = video.device
            # Only move if necessary (avoid unnecessary transfers)
            if watermarked_video.device != device:
                watermarked_video = watermarked_video.to(device)
            if embedded_message.device != device:
                embedded_message = embedded_message.to(device)
            if embedded_message_long.device != device:
                embedded_message_long = embedded_message_long.to(device)

            # Evaluate each perturbation
            pert_tqdm = tqdm(
                perturbations, desc=f"{video_name}: perturbations", leave=False
            )
            for pert in pert_tqdm:
                pert_name = type(pert).__name__
                # Remove "Wrapper" suffix for display
                display_name = pert_name.replace("Wrapper", "")
                pert_params = _get_perturbation_params(pert)

                # Update tqdm description with current perturbation name
                pert_tqdm.set_description(f"{video_name}: {display_name}")

                # ===== REMOVAL ATTACK =====
                # Apply perturbation to watermarked video
                with torch.no_grad():
                    try:
                        perturbed_video = _apply_perturbation(
                            pert, watermarked_video, pert_name
                        )
                    except Exception as e:
                        logger.warning(f"Error applying {display_name} (removal): {e}")
                        continue

                # Extract watermark
                with torch.no_grad():
                    try:
                        logits = model.extract(perturbed_video)  # Shape: (T, K)
                    except Exception as e:
                        logger.warning(
                            f"Error extracting from {display_name} (removal): {e}"
                        )
                        continue

                # Evaluate with all aggregation strategies (GPU-optimized)
                # Keep tensors on GPU to avoid unnecessary CPU transfers

                # Ensure embedded_message_long is on the same device as logits
                # Only transfer if necessary (optimize device transfers)
                logits_device = logits.device
                if embedded_message_long.device != logits_device:
                    embedded_message_long_device = embedded_message_long.to(
                        logits_device
                    )
                else:
                    embedded_message_long_device = embedded_message_long

                # Match dimensions on GPU (reuse pre-converted embedded_message_long)
                if logits.shape[1] > embedded_message_long_device.shape[0]:
                    logits_matched = logits[:, : embedded_message_long_device.shape[0]]
                    watermark_gt = embedded_message_long_device  # Shape: (K,) - reuse cached conversion
                elif logits.shape[1] < embedded_message_long_device.shape[0]:
                    logits_matched = logits
                    watermark_gt = embedded_message_long_device[
                        : logits.shape[1]
                    ]  # Shape: (K,) - reuse cached conversion
                else:
                    logits_matched = logits
                    watermark_gt = embedded_message_long_device  # Shape: (K,) - reuse cached conversion

                for agg_name, agg_func in AGGREGATION_STRATEGIES.items():
                    try:
                        # Use GPU tensor version for better performance
                        detected = _call_with_watermark_gt_tensor(
                            agg_func, logits_matched, watermark_gt
                        )
                        results.append(
                            {
                                "video": video_name,
                                "perturbation": display_name,
                                "perturbation_params": str(pert_params),
                                "aggregation": agg_name,
                                "detected": bool(detected),
                                "attack_type": "removal",
                                "type": "common",
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error in {agg_name} for {display_name} (removal): {e}"
                        )

                # ===== FORGERY ATTACK =====
                if evaluate_forgery:
                    # Use kinetics dataset for forgery attack (as per original implementation)
                    # If kinetics_dataset is provided, use it; otherwise use the same dataset
                    if original_video_forgery is not None:
                        forgery_video = original_video_forgery
                    else:
                        # Fallback: use same dataset (for backward compatibility)
                        forgery_video = video

                    # Apply perturbation to original (unwatermarked) video
                    with torch.no_grad():
                        try:
                            perturbed_original = _apply_perturbation(
                                pert, forgery_video, pert_name
                            )
                        except Exception as e:
                            logger.warning(
                                f"Error applying {display_name} (forgery): {e}"
                            )
                            continue

                    # Extract watermark from unwatermarked video
                    with torch.no_grad():
                        try:
                            logits_forgery = model.extract(
                                perturbed_original
                            )  # Shape: (T, K)
                        except Exception as e:
                            logger.warning(
                                f"Error extracting from {display_name} (forgery): {e}"
                            )
                            continue

                    # For forgery, check if watermark is detected in unwatermarked video (false positive)
                    # Use the same embedded message pattern as reference (GPU-optimized)
                    # Ensure embedded_message_long is on the same device as logits_forgery
                    # Only transfer if necessary (optimize device transfers)
                    logits_forgery_device = logits_forgery.device
                    if embedded_message_long.device != logits_forgery_device:
                        embedded_message_long_forgery = embedded_message_long.to(
                            logits_forgery_device
                        )
                    else:
                        embedded_message_long_forgery = embedded_message_long

                    # Match dimensions on GPU (reuse pre-converted embedded_message_long)
                    if logits_forgery.shape[1] > embedded_message_long_forgery.shape[0]:
                        logits_forgery_matched = logits_forgery[
                            :, : embedded_message_long_forgery.shape[0]
                        ]
                        watermark_gt_forgery = (
                            embedded_message_long_forgery  # Reuse cached conversion
                        )
                    elif (
                        logits_forgery.shape[1] < embedded_message_long_forgery.shape[0]
                    ):
                        logits_forgery_matched = logits_forgery
                        watermark_gt_forgery = embedded_message_long_forgery[
                            : logits_forgery.shape[1]
                        ]  # Reuse cached conversion
                    else:
                        logits_forgery_matched = logits_forgery
                        watermark_gt_forgery = (
                            embedded_message_long_forgery  # Reuse cached conversion
                        )

                    # For forgery, use the same watermark pattern as reference
                    # If this pattern is detected in unwatermarked video, it's a false positive (forgery success)
                    for agg_name, agg_func in AGGREGATION_STRATEGIES.items():
                        try:
                            # Use GPU tensor version for better performance
                            detected_forgery = _call_with_watermark_gt_tensor(
                                agg_func, logits_forgery_matched, watermark_gt_forgery
                            )
                            results.append(
                                {
                                    "video": video_name_forgery,
                                    "perturbation": display_name,
                                    "perturbation_params": str(pert_params),
                                    "aggregation": agg_name,
                                    "detected": bool(detected_forgery),
                                    "attack_type": "forgery",
                                    "type": "common",
                                }
                            )
                        except Exception as e:
                            logger.warning(
                                f"Error in {agg_name} for {display_name} (forgery): {e}"
                            )

                # Explicitly delete large tensors to free memory
                del perturbed_video
                if evaluate_forgery:
                    del perturbed_original
                del logits
                if evaluate_forgery:
                    del logits_forgery

                # Clear GPU cache periodically (every 3 perturbations) to prevent memory buildup
                if torch.cuda.is_available() and (pert_tqdm.n % 3 == 0):
                    torch.cuda.empty_cache()

        df = pd.DataFrame(results)
        output_file = output_dir / "common_perturbations_results.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")

        # Calculate and save statistics
        from evaluate.utils import calculate_statistics

        stats = calculate_statistics(df, logger)
        stats_file = output_dir / "common_perturbations_stats.json"
        import json

        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_file}")

        return df
    finally:
        # Clean up temporary directory for MPEG4
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(
                    f"Failed to clean up temporary directory {temp_dir}: {e}"
                )


def _apply_jpeg_albumentations(
    pert: torch.nn.Module, video: torch.Tensor
) -> torch.Tensor:
    """
    Apply JPEG perturbation using albumentations video augmentation.

    Args:
        pert: JPEG perturbation module
        video: Video tensor of shape (T, C, H, W)

    Returns:
        Perturbed video tensor
    """
    quality = pert.quality
    transform = A.ImageCompression(
        compression_type="jpeg", quality_range=(quality, quality), p=1.0
    )
    return _apply_albumentations_transform(video, transform)


def _apply_gaussianblur_albumentations(
    pert: torch.nn.Module, video: torch.Tensor
) -> torch.Tensor:
    """
    Apply Gaussian blur using albumentations video augmentation.

    Args:
        pert: GaussianBlur perturbation module
        video: Video tensor of shape (T, C, H, W)

    Returns:
        Perturbed video tensor
    """
    # Get std from perturbation (GaussianBlur.sigma is a tuple)
    std = pert.sigma[0] if isinstance(pert.sigma, tuple) else pert.sigma

    # albumentations GaussianBlur uses blur_limit and sigma_limit
    # Fixed kernel size of 5
    transform = A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(std, std), p=1.0)
    return _apply_albumentations_transform(video, transform)


def _apply_crop_albumentations(
    pert: torch.nn.Module, video: torch.Tensor
) -> torch.Tensor:
    """
    Apply Crop using albumentations video augmentation.

    Args:
        pert: Crop perturbation module
        video: Video tensor of shape (T, C, H, W)

    Returns:
        Perturbed video tensor
    """
    crop_ratio = pert.crop_ratio
    _, _, H, W = video.shape
    crop_height = int(H * crop_ratio)
    crop_width = int(W * crop_ratio)

    # Create transform pipeline: RandomCrop + Resize
    transform = A.Compose(
        [
            A.RandomCrop(height=crop_height, width=crop_width, p=1.0),
            A.Resize(height=H, width=W, p=1.0),
        ]
    )
    return _apply_albumentations_transform(video, transform, clamp=True)


def _apply_frame_average_optimized(
    pert: torch.nn.Module, video: torch.Tensor, T: int
) -> torch.Tensor:
    """
    Optimized version of FrameAverage using fully vectorized operations.

    Uses unfold for efficient sliding window average computation.

    Args:
        pert: FrameAverage perturbation module
        video: Video tensor of shape (T, C, H, W)
        T: Number of frames

    Returns:
        Averaged video tensor
    """
    m = pert.m

    # If m == 1, return original video
    if m == 1:
        return video.clone()

    N, C, H, W = video.shape

    # Pad video for boundary handling (replicate edge frames)
    pad_size = (m - 1) // 2
    if pad_size > 0:
        # Pad along first dimension (time)
        # For 4D tensor (T, C, H, W), F.pad expects: (W_left, W_right, H_top, H_bottom, C_front, C_back, T_left, T_right)
        # We only want to pad T dimension, so pad the last 2 values: (0, 0, 0, 0, 0, 0, pad_size, pad_size)
        # But actually, we need to manually pad using torch.cat for time dimension
        first_frame = video[0:1].repeat(pad_size, 1, 1, 1)  # Repeat first frame
        last_frame = video[-1:].repeat(pad_size, 1, 1, 1)  # Repeat last frame
        padded_video = torch.cat([first_frame, video, last_frame], dim=0)
    else:
        padded_video = video

    T_padded = padded_video.shape[0]

    # Reshape to (C*H*W, T_padded) for unfold
    video_reshaped = padded_video.permute(1, 2, 3, 0).reshape(
        C * H * W, T_padded
    )  # (C*H*W, T_padded)

    # Unfold along time dimension to create sliding windows
    # unfold(dimension, size, step): creates windows of size m with step 1
    unfolded = video_reshaped.unfold(1, m, 1)  # (C*H*W, T_padded-m+1, m)

    # Take mean along window dimension
    averaged = unfolded.mean(dim=-1)  # (C*H*W, T_padded-m+1)

    # Reshape back to (T_padded-m+1, C, H, W)
    averaged = averaged.T.reshape(T_padded - m + 1, C, H, W)  # (T_padded-m+1, C, H, W)

    # Crop to original length N (handle boundary cases)
    if averaged.shape[0] >= N:
        result = averaged[:N]
    else:
        # If shorter, pad with last frame
        last_frame = averaged[-1:].repeat(N - averaged.shape[0], 1, 1, 1)
        result = torch.cat([averaged, last_frame], dim=0)

    return result.to(video.device)


def _apply_perturbation(
    pert: torch.nn.Module, video: torch.Tensor, pert_name: str
) -> torch.Tensor:
    """
    Apply perturbation to video, handling different perturbation types.

    Uses albumentations for JPEG, GaussianBlur, and Crop (required).
    Other perturbations use optimized implementations.

    Args:
        pert: Perturbation module
        video: Video tensor of shape (T, C, H, W)
        pert_name: Name of perturbation class

    Returns:
        Perturbed video tensor
    """
    # Reshape video to (T, C, H, W) -> (T, C, H, W) for batch processing
    T, C, H, W = video.shape

    # Albumentations-based perturbations (required)
    if pert_name == "JPEG":
        return _apply_jpeg_albumentations(pert, video)

    elif pert_name == "GaussianBlur":
        return _apply_gaussianblur_albumentations(pert, video)

    elif pert_name == "Crop":
        return _apply_crop_albumentations(pert, video)

    # Gaussian: Use original implementation (albumentations has precision loss and different clamp behavior)
    elif pert_name == "Gaussian":
        # Apply perturbation directly on tensor (matches original behavior exactly)
        std = pert.noise_level
        min_value = torch.min(video)
        max_value = torch.max(video)

        # Generate noise and apply
        gaussian = torch.randn_like(video)
        noised_video = video + std * gaussian
        noised_video = noised_video.clamp(min_value.item(), max_value.item())

        return noised_video

    # FrameAverage: Optimize with vectorized operations
    elif pert_name == "FrameAverage":
        return _apply_frame_average_optimized(pert, video, T)

    else:
        # Apply perturbation to entire video (FrameRemove, FrameSwitch, MPEG4, etc.)
        return pert(video)


def _get_perturbation_params(pert: torch.nn.Module) -> Dict:
    """Extract parameters from perturbation instance."""
    params = {}
    for attr in ["quality", "std", "crop_ratio", "num_frames", "p", "codec"]:
        if hasattr(pert, attr):
            params[attr] = getattr(pert, attr)
    return params
