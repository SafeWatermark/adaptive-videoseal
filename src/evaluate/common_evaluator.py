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


class MPEG4Wrapper(nn.Module):
    """
    Wrapper for MPEG4 perturbation that fixes bugs in the original submodule.
    
    The original MPEG4 class has several issues:
    1. Missing import torch.nn as nn
    2. Wrong super() call (uses Crop instead of MPEG4)
    3. Doesn't return the processed video
    
    This wrapper fixes all these issues.
    """
    
    def __init__(self, quality: int, codec: str = 'libx264', temp_dir: Optional[Path] = None):
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
        Apply MPEG4 compression to video.
        
        Args:
            video: Video tensor of shape (T, C, H, W) in [0, 1]
            
        Returns:
            Compressed video tensor of same shape
        """
        # Create unique temporary file
        import uuid
        temp_file = self.temp_dir / f"mpeg4_{uuid.uuid4().hex}.mp4"
        
        try:
            # Convert video to numpy array
            frames = (video * 255).byte().cpu().numpy()
            frames = np.transpose(frames, (0, 2, 3, 1))  # (T, H, W, C)
            
            # Get video dimensions
            T, H, W, C = frames.shape
            fps = 25  # Default FPS
            
            # Write video using ffmpeg
            process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{W}x{H}', r=fps)
                .output(str(temp_file), vcodec=self.codec, pix_fmt='yuv420p', crf=self.quality)
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stderr=True)
            )
            
            # Write frames to ffmpeg
            for frame in frames:
                process.stdin.write(frame.tobytes())
            
            process.stdin.close()
            process.wait()
            
            # Read compressed video back
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
                    last_frame = compressed_video[-1:].repeat(video.shape[0] - compressed_video.shape[0], 1, 1, 1)
                    compressed_video = torch.cat([compressed_video, last_frame], dim=0)
                else:
                    # Take first T frames if compressed video is longer
                    compressed_video = compressed_video[:video.shape[0]]
            
            return compressed_video.to(video.device)
            
        finally:
            # Clean up temporary file
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass


def _wrap_aggregation_func(original_func, watermark_gt: np.ndarray):
    """
    Wrap aggregation function to use actual embedded message instead of hardcoded pattern.
    
    The original aggregation functions use hardcoded watermark_gt = [0, 1] * 48.
    This wrapper replaces that with the actual embedded message.
    """
    import types
    
    # Create a new function that uses the actual watermark_gt
    def wrapped_func(decode):
        # Temporarily replace watermark_gt in the function's closure
        # Since we can't modify closures directly, we'll use a different approach:
        # We'll modify the function's globals or use a patched version
        
        # For functions that use watermark_gt directly, we need to patch it
        # The original functions are in aggregation.py, so we'll create modified versions
        
        # Get the function's code and modify it
        # Actually, a simpler approach: create wrapper functions that match the original API
        # but use the actual watermark_gt
        
        # Since the original functions are complex, we'll recreate them with watermark_gt parameter
        # But first, let's try to call the original and see if we can patch it
        
        # Actually, the cleanest way is to recreate the aggregation logic here
        # with the actual watermark_gt
        
        # For now, let's use monkey patching on the module level
        import aggregation as agg_module
        original_watermark_gt = getattr(agg_module, '_watermark_gt', None)
        agg_module._watermark_gt = watermark_gt
        
        try:
            # Call original function - but it won't use our watermark_gt
            # So we need to recreate the logic
            result = _call_with_watermark_gt(original_func, decode, watermark_gt)
        finally:
            if original_watermark_gt is not None:
                agg_module._watermark_gt = original_watermark_gt
            elif hasattr(agg_module, '_watermark_gt'):
                delattr(agg_module, '_watermark_gt')
        
        return result
    
    return wrapped_func


def _call_with_watermark_gt(func, decode: np.ndarray, watermark_gt: np.ndarray) -> bool:
    """
    Call aggregation function with custom watermark_gt.
    
    Reimplements the aggregation logic from aggregation.py but uses actual watermark_gt.
    
    Note: VideoSeal uses logits where:
    - logits > 0 -> bit 0
    - logits < 0 -> bit 1
    So we convert logits to bits: (decode >= 0).astype(int) gives 0 for positive, 1 for negative
    But watermark_gt is already in {0, 1} format, so we need to compare correctly.
    """
    from scipy.optimize import minimize
    from scipy.stats import binom
    
    func_name = func.__name__
    
    # Convert logits to bits: positive -> 1, negative/zero -> 0
    # VideoSeal convention: logits > 0 -> bit 1, logits <= 0 -> bit 0
    predicted_bits = (decode > 0).astype(int)  # 1 for positive, 0 for negative/zero
    
    if func_name == 'logit_mean':
        decoded = np.mean(decode, axis=0)
        predicted = (decoded > 0).astype(int)  # 1 for positive, 0 for negative/zero
        bitacc = np.sum((watermark_gt == predicted)) / len(watermark_gt)
        return bitacc >= 67/96
    
    elif func_name == 'logit_g_median':
        def objective(median):
            return np.sum(np.linalg.norm(decode - median, axis=1))
        initial_guess = np.mean(decode, axis=0)
        decoded = minimize(objective, initial_guess, method='Powell', tol=1e-5)
        predicted = (decoded.x > 0).astype(int)  # 1 for positive, 0 for negative/zero
        bitacc = np.sum((watermark_gt == predicted)) / len(watermark_gt)
        return bitacc >= 67/96
    
    elif func_name == 'bit_majority':
        # Majority vote across frames
        decoded = (np.sum(predicted_bits, axis=0) > (len(decode) / 2)).astype(int)
        bitacc = np.sum((watermark_gt == decoded)) / len(watermark_gt)
        return bitacc >= 67/96
    
    elif func_name == 'bit_g_median':
        decoded = np.median(predicted_bits, axis=0).astype(int)
        bitacc = np.sum((watermark_gt == decoded)) / len(watermark_gt)
        return bitacc >= 67/96
    
    elif func_name == 'bitacc_mean':
        watermark_gt_tiled = np.tile(watermark_gt, (decode.shape[0], 1))
        bitacc = np.mean((watermark_gt_tiled == predicted_bits))
        return bitacc >= 67/96
    
    elif func_name == 'bitacc_median':
        watermark_gt_tiled = np.tile(watermark_gt, (decode.shape[0], 1))
        bitacc = np.mean((watermark_gt_tiled == predicted_bits), axis=1)
        median_bitacc = np.median(bitacc)
        return median_bitacc >= 67/96
    
    elif func_name == 'detection_threshold':
        watermark_gt_tiled = np.tile(watermark_gt, (decode.shape[0], 1))
        bitacc = np.mean((watermark_gt_tiled == predicted_bits), axis=1)
        detected_frames = np.sum(bitacc >= 67/96)
        
        n = decode.shape[0]
        p = 0.0000661
        threshold = 10e-4
        k = n  # Default if no threshold found
        for i in range(n + 1):
            if binom.sf(i - 1, n, p) <= threshold:
                k = i
                break
        
        return detected_frames >= k
    
    elif func_name == 'detection_majority':
        watermark_gt_tiled = np.tile(watermark_gt, (decode.shape[0], 1))
        bitacc = np.mean((watermark_gt_tiled == predicted_bits), axis=1)
        detected_frames = np.sum(bitacc >= 67 / 96)
        return detected_frames > (decode.shape[0] // 2)
    
    else:
        # Fallback to original function
        return func(decode)


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
        for video, video_info in dataset:
            if max_videos and video_count >= max_videos:
                break

            video_count += 1
            video_name = video_info.get('name', 'unknown')
            logger.info(f"Processing video {video_count}: {video_name}")

            # Limit frames if specified
            if max_frames_per_video and video.shape[0] > max_frames_per_video:
                video = video[:max_frames_per_video]

            # Embed watermark for removal attack
            with torch.no_grad():
                embed_result = model.embed(video)
                watermarked_video = embed_result["watermarked_video"]
                embedded_message = embed_result["message"]

            # Evaluate each perturbation
            pert_tqdm = tqdm(perturbations, desc=f"{video_name}: perturbations", leave=False)
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
                        perturbed_video = _apply_perturbation(pert, watermarked_video, pert_name)
                    except Exception as e:
                        logger.warning(f"Error applying {display_name} (removal): {e}")
                        continue

                # Extract watermark
                with torch.no_grad():
                    try:
                        logits = model.extract(perturbed_video)  # Shape: (T, K)
                    except Exception as e:
                        logger.warning(f"Error extracting from {display_name} (removal): {e}")
                        continue

                # Evaluate with all aggregation strategies
                logits_np = logits.detach().cpu().numpy()  # Shape: (T, K)
                embedded_message_np = embedded_message.detach().cpu().numpy()  # Shape: (K,)
                
                # Match dimensions
                if logits_np.shape[1] > len(embedded_message_np):
                    logits_np = logits_np[:, :len(embedded_message_np)]
                elif logits_np.shape[1] < len(embedded_message_np):
                    embedded_message_np = embedded_message_np[:logits_np.shape[1]]
                
                watermark_gt = embedded_message_np.astype(np.int32)  # Shape: (K,)
                
                for agg_name, agg_func in AGGREGATION_STRATEGIES.items():
                    try:
                        detected = _call_with_watermark_gt(agg_func, logits_np, watermark_gt)
                        results.append({
                            "video": video_name,
                            "perturbation": display_name,
                            "perturbation_params": str(pert_params),
                            "aggregation": agg_name,
                            "detected": bool(detected),
                            "attack_type": "removal",
                            "type": "common",
                        })
                    except Exception as e:
                        logger.warning(f"Error in {agg_name} for {display_name} (removal): {e}")

                # ===== FORGERY ATTACK =====
                if evaluate_forgery:
                    # Apply perturbation to original (unwatermarked) video
                    with torch.no_grad():
                        try:
                            perturbed_original = _apply_perturbation(pert, video, pert_name)
                        except Exception as e:
                            logger.warning(f"Error applying {display_name} (forgery): {e}")
                            continue

                    # Extract watermark from unwatermarked video
                    with torch.no_grad():
                        try:
                            logits_forgery = model.extract(perturbed_original)  # Shape: (T, K)
                        except Exception as e:
                            logger.warning(f"Error extracting from {display_name} (forgery): {e}")
                            continue

                    # For forgery, check if watermark is detected in unwatermarked video (false positive)
                    # Use the same embedded message pattern as reference
                    logits_forgery_np = logits_forgery.detach().cpu().numpy()  # Shape: (T, K)
                    
                    # Match dimensions
                    if logits_forgery_np.shape[1] > len(embedded_message_np):
                        logits_forgery_np = logits_forgery_np[:, :len(embedded_message_np)]
                    elif logits_forgery_np.shape[1] < len(embedded_message_np):
                        embedded_message_np_forgery = embedded_message_np[:logits_forgery_np.shape[1]]
                    else:
                        embedded_message_np_forgery = embedded_message_np
                    
                    # For forgery, use the same watermark pattern as reference
                    # If this pattern is detected in unwatermarked video, it's a false positive (forgery success)
                    watermark_gt_forgery = embedded_message_np_forgery.astype(np.int32)
                    
                    for agg_name, agg_func in AGGREGATION_STRATEGIES.items():
                        try:
                            detected_forgery = _call_with_watermark_gt(agg_func, logits_forgery_np, watermark_gt_forgery)
                            results.append({
                                "video": video_name,
                                "perturbation": display_name,
                                "perturbation_params": str(pert_params),
                                "aggregation": agg_name,
                                "detected": bool(detected_forgery),
                                "attack_type": "forgery",
                                "type": "common",
                            })
                        except Exception as e:
                            logger.warning(f"Error in {agg_name} for {display_name} (forgery): {e}")

        df = pd.DataFrame(results)
        output_file = output_dir / "common_perturbations_results.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")

        # Calculate and save statistics
        from evaluate.utils import calculate_statistics
        stats = calculate_statistics(df, logger)
        stats_file = output_dir / "common_perturbations_stats.json"
        import json
        with open(stats_file, 'w') as f:
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
                logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")


def _apply_perturbation(pert: torch.nn.Module, video: torch.Tensor, pert_name: str) -> torch.Tensor:
    """
    Apply perturbation to video, handling different perturbation types.
    
    Args:
        pert: Perturbation module
        video: Video tensor of shape (T, C, H, W)
        pert_name: Name of perturbation class
        
    Returns:
        Perturbed video tensor
    """
    # Some perturbations expect (T, C, H, W), others expect frame-by-frame
    # Handle frame-by-frame perturbations (JPEG, Gaussian, GaussianBlur)
    if pert_name in ["JPEG", "Gaussian", "GaussianBlur"]:
        # Apply perturbation to each frame
        perturbed_frames = []
        for frame_idx in range(video.shape[0]):
            frame = video[frame_idx:frame_idx+1]  # (1, C, H, W)
            perturbed_frame = pert(frame)  # (1, C, H, W)
            perturbed_frames.append(perturbed_frame.squeeze(0))  # (C, H, W)
        return torch.stack(perturbed_frames, dim=0)  # (T, C, H, W)
    else:
        # Apply perturbation to entire video
        return pert(video)


def _get_perturbation_params(pert: torch.nn.Module) -> Dict:
    """Extract parameters from perturbation instance."""
    params = {}
    for attr in ["quality", "std", "crop_ratio", "num_frames", "p", "codec"]:
        if hasattr(pert, attr):
            params[attr] = getattr(pert, attr)
    return params

