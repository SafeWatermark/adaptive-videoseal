"""
Pre-apply perturbations to videos for faster evaluation.

This script applies all common perturbations to videos and saves them to disk,
allowing the evaluation to skip the perturbation step and load pre-perturbed videos directly.
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import json

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from utils import setup_seed, setup_logger, select_device  # noqa: E402
from evaluate.dataset_loader import VideoMarkBenchDataset  # noqa: E402
from evaluate.common_evaluator import (  # noqa: E402
    get_common_perturbations,
    _apply_perturbation,
    _get_perturbation_params,
)


def get_perturbation_name(pert: torch.nn.Module) -> str:
    """Get a unique name for a perturbation including its parameters."""
    pert_name = type(pert).__name__.replace("Wrapper", "")
    params = _get_perturbation_params(pert)

    # Create parameter string
    param_parts = []
    for key, value in sorted(params.items()):
        if isinstance(value, float):
            param_parts.append(f"{key}={value:.2f}")
        else:
            param_parts.append(f"{key}={value}")

    if param_parts:
        return f"{pert_name}_{'_'.join(param_parts)}"
    return pert_name


def save_video_tensor(video: torch.Tensor, output_path: Path) -> None:
    """
    Save video tensor to file.

    Args:
        video: Video tensor of shape (T, C, H, W) in [0, 1]
        output_path: Path to save the video (as .pt file for lossless storage)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(video.cpu(), output_path)


def save_video_as_mp4(
    video: torch.Tensor, output_path: Path, fps: int = 25, quality: int = 18
) -> None:
    """
    Save video tensor as MP4 file using ffmpeg.

    Args:
        video: Video tensor of shape (T, C, H, W) in [0, 1]
        output_path: Path to save the video (.mp4)
        fps: Frames per second
        quality: CRF quality (lower = higher quality, 18 is visually lossless)
    """
    import ffmpeg

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy: (T, C, H, W) -> (T, H, W, C)
    frames = (video * 255).byte().cpu().numpy()
    frames = np.transpose(frames, (0, 2, 3, 1))  # (T, H, W, C)

    _, H, W, _ = frames.shape

    # Use ffmpeg to encode
    process = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{W}x{H}", r=fps)
        .output(
            str(output_path),
            vcodec="libx264",
            pix_fmt="yuv420p",
            crf=quality,
            preset="medium",
            movflags="faststart",
            **{"threads": "0"},
        )
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=True, quiet=True)
    )

    frames_bytes = frames.tobytes()
    process.stdin.write(frames_bytes)
    process.stdin.close()
    process.wait()


def apply_perturbations_to_dataset(
    dataset: VideoMarkBenchDataset,
    output_dir: Path,
    logger: logging.Logger,
    max_videos: Optional[int] = None,
    max_frames_per_video: Optional[int] = None,
    save_format: str = "pt",
    selected_perturbations: Optional[List[str]] = None,
) -> Dict:
    """
    Apply all perturbations to dataset videos and save to disk.

    Args:
        dataset: VideoMarkBench dataset
        output_dir: Directory to save perturbed videos
        logger: Logger
        max_videos: Maximum number of videos to process
        max_frames_per_video: Maximum frames per video
        save_format: 'pt' for lossless tensor, 'mp4' for video file
        selected_perturbations: List of perturbation names to apply (None = all)

    Returns:
        Dictionary with metadata about saved perturbations
    """
    import tempfile
    import shutil

    # Create temp directory for MPEG4
    temp_dir = Path(tempfile.mkdtemp(prefix="perturbation_apply_"))

    try:
        perturbations = get_common_perturbations(temp_dir=temp_dir)

        # Filter perturbations if specified
        if selected_perturbations:
            filtered_perturbations = []
            for pert in perturbations:
                pert_name = get_perturbation_name(pert)
                if any(sel in pert_name for sel in selected_perturbations):
                    filtered_perturbations.append(pert)
            perturbations = filtered_perturbations
            logger.info(
                f"Selected {len(perturbations)} perturbations: {selected_perturbations}"
            )

        metadata = {
            "perturbations": [],
            "videos": [],
            "save_format": save_format,
        }

        # Store perturbation info
        for pert in perturbations:
            pert_full_name = get_perturbation_name(pert)
            pert_info = {
                "name": pert_full_name,
                "class": type(pert).__name__.replace("Wrapper", ""),
                "params": _get_perturbation_params(pert),
            }
            metadata["perturbations"].append(pert_info)

        logger.info(f"Applying {len(perturbations)} perturbations to dataset")

        video_count = 0
        for video, video_info in dataset:
            if max_videos and video_count >= max_videos:
                break

            video_count += 1
            video_name = video_info.get("name", f"video_{video_count}")
            # Sanitize video name for filesystem
            safe_video_name = video_name.replace("/", "_").replace("\\", "_")
            logger.info(f"Processing video {video_count}: {video_name}")

            # Limit frames if specified
            if max_frames_per_video and video.shape[0] > max_frames_per_video:
                video = video[:max_frames_per_video]

            video_meta = {
                "name": video_name,
                "safe_name": safe_video_name,
                "num_frames": video.shape[0],
                "perturbations": {},
            }

            # Apply and save each perturbation
            pert_tqdm = tqdm(
                perturbations, desc=f"{safe_video_name}: perturbations", leave=False
            )
            for pert in pert_tqdm:
                pert_full_name = get_perturbation_name(pert)
                pert_class_name = type(pert).__name__

                pert_tqdm.set_description(f"{safe_video_name}: {pert_full_name}")

                try:
                    with torch.no_grad():
                        perturbed_video = _apply_perturbation(
                            pert, video, pert_class_name
                        )

                    # Save perturbed video
                    if save_format == "pt":
                        output_path = (
                            output_dir / pert_full_name / f"{safe_video_name}.pt"
                        )
                        save_video_tensor(perturbed_video, output_path)
                    else:
                        output_path = (
                            output_dir / pert_full_name / f"{safe_video_name}.mp4"
                        )
                        save_video_as_mp4(perturbed_video, output_path)

                    video_meta["perturbations"][pert_full_name] = str(output_path)

                except Exception as e:
                    logger.warning(
                        f"Error applying {pert_full_name} to {video_name}: {e}"
                    )
                    continue

                # Free memory
                del perturbed_video
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            metadata["videos"].append(video_meta)

        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")

        return metadata

    finally:
        # Clean up temp directory
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")


def main():
    """Main entry point for applying perturbations."""
    parser = argparse.ArgumentParser(
        description="Pre-apply perturbations to VideoMarkBench videos"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/videomarkbench",
        help="Path to VideoMarkBench dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save perturbed videos",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Maximum number of videos to process (None = all)",
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=None,
        help="Maximum frames per video (None = all)",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="pt",
        choices=["pt", "mp4"],
        help="Format to save perturbed videos: 'pt' (lossless tensor) or 'mp4' (video)",
    )
    parser.add_argument(
        "--perturbations",
        type=str,
        nargs="*",
        default=None,
        help="Specific perturbations to apply (e.g., 'JPEG' 'Gaussian'). Default: all",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Setup
    select_device(args.device)  # Validate device selection
    setup_seed(args.seed)

    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    logger = setup_logger(log_dir)

    logger.info("===== Apply Perturbations to Videos =====")
    for key, value in vars(args).items():
        logger.info(f"{key:<25} : {value}")
    logger.info("=========================================")

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        return

    dataset = VideoMarkBenchDataset(dataset_path, logger=logger)
    logger.info(f"Loaded dataset with {len(dataset)} videos")

    # Apply perturbations
    metadata = apply_perturbations_to_dataset(
        dataset=dataset,
        output_dir=output_dir,
        logger=logger,
        max_videos=args.max_videos,
        max_frames_per_video=args.max_frames_per_video,
        save_format=args.save_format,
        selected_perturbations=args.perturbations,
    )

    logger.info("=" * 50)
    logger.info("Perturbation Application Summary")
    logger.info("=" * 50)
    logger.info(f"Processed {len(metadata['videos'])} videos")
    logger.info(f"Applied {len(metadata['perturbations'])} perturbations")
    logger.info(f"Output directory: {output_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
