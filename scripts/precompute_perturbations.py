#!/usr/bin/env python3
"""
Pre-compute perturbations for VideoMarkBench evaluation.

This script generates all perturbed videos and saves them to disk for faster
evaluation. The perturbations are applied to original (unwatermarked) videos,
which speeds up forgery attack evaluation.

Usage:
    python scripts/precompute_perturbations.py \
        --dataset data/videomarkbench \
        --output_dir data/precomputed_perturbations \
        --workers 8
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(src_dir / "videomarkbench"))

from utils import setup_seed


# Video loading (reuse from dataset_loader)
try:
    from decord import VideoReader, cpu

    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


def load_video(video_path: Path) -> torch.Tensor:
    """
    Load video from file.

    Args:
        video_path: Path to video file

    Returns:
        Video tensor of shape (T, C, H, W) in [0, 1]
    """
    if DECORD_AVAILABLE:
        return _load_video_decord(video_path)
    else:
        return _load_video_cv2(video_path)


def _load_video_decord(video_path: Path) -> torch.Tensor:
    """Load video using decord (faster, batch loading)."""
    vr = VideoReader(str(video_path), width=512, height=512, num_threads=4, ctx=cpu(0))
    num_frames = len(vr)
    if num_frames == 0:
        raise ValueError(f"No frames loaded from {video_path}")

    vid_np = vr.get_batch(range(num_frames)).asnumpy()
    vid_tensor = torch.from_numpy(vid_np).float()
    vid_tensor = vid_tensor.permute(0, 3, 1, 2) / 255.0
    return vid_tensor


def _load_video_cv2(video_path: Path) -> torch.Tensor:
    """Load video using cv2 (fallback method)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (512, 512))
        frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames loaded from {video_path}")

    return torch.stack(frames, dim=0)


@dataclass
class PerturbationConfig:
    """Configuration for a single perturbation."""

    name: str
    class_name: str
    params: Dict

    def get_dir_name(self) -> str:
        """Get directory name for this perturbation (e.g., 'JPEG_90')."""
        if self.params:
            param_str = "_".join(str(v) for v in self.params.values())
            return f"{self.class_name}_{param_str}"
        return self.class_name


def get_perturbation_configs(
    include_mpeg4: bool = True,
) -> Tuple[List[PerturbationConfig], List[PerturbationConfig]]:
    """
    Get list of all perturbation configurations.

    Returns:
        Tuple of (non_mpeg4_configs, mpeg4_configs)
    """
    non_mpeg4 = [
        # JPEG
        PerturbationConfig("JPEG_90", "JPEG", {"quality": 90}),
        PerturbationConfig("JPEG_75", "JPEG", {"quality": 75}),
        PerturbationConfig("JPEG_50", "JPEG", {"quality": 50}),
        # Gaussian noise
        PerturbationConfig("Gaussian_0.01", "Gaussian", {"std": 0.01}),
        PerturbationConfig("Gaussian_0.05", "Gaussian", {"std": 0.05}),
        # Gaussian blur
        PerturbationConfig("GaussianBlur_1.0", "GaussianBlur", {"std": 1.0}),
        PerturbationConfig("GaussianBlur_2.0", "GaussianBlur", {"std": 2.0}),
        # Crop
        PerturbationConfig("Crop_0.9", "Crop", {"crop_ratio": 0.9}),
        PerturbationConfig("Crop_0.8", "Crop", {"crop_ratio": 0.8}),
        # Frame operations
        PerturbationConfig("FrameAverage_2", "FrameAverage", {"num_frames": 2}),
        PerturbationConfig("FrameAverage_3", "FrameAverage", {"num_frames": 3}),
        PerturbationConfig("FrameRemove_0.1", "FrameRemove", {"p": 0.1}),
        PerturbationConfig("FrameRemove_0.2", "FrameRemove", {"p": 0.2}),
        PerturbationConfig("FrameSwitch_0.1", "FrameSwitch", {"p": 0.1}),
        PerturbationConfig("FrameSwitch_0.2", "FrameSwitch", {"p": 0.2}),
    ]

    mpeg4 = [
        PerturbationConfig("MPEG4_23", "MPEG4", {"quality": 23}),
        PerturbationConfig("MPEG4_28", "MPEG4", {"quality": 28}),
        PerturbationConfig("MPEG4_33", "MPEG4", {"quality": 33}),
    ]

    if include_mpeg4:
        return non_mpeg4, mpeg4
    return non_mpeg4, []


def create_perturbation(config: PerturbationConfig, temp_dir: Optional[Path] = None):
    """
    Create a perturbation instance from config.

    Args:
        config: Perturbation configuration
        temp_dir: Temporary directory for MPEG4

    Returns:
        Perturbation module instance
    """
    from common_perturabtions.crop import Crop
    from common_perturabtions.frame_average import FrameAverage
    from common_perturabtions.frame_remove import FrameRemove
    from common_perturabtions.frame_switch import FrameSwitch
    from common_perturabtions.gaussian import Gaussian
    from common_perturabtions.gaussian_blur import GaussianBlur
    from common_perturabtions.jpeg import JPEG

    if config.class_name == "JPEG":
        return JPEG(quality=config.params["quality"])
    elif config.class_name == "Gaussian":
        return Gaussian(std=config.params["std"])
    elif config.class_name == "GaussianBlur":
        return GaussianBlur(std=config.params["std"])
    elif config.class_name == "Crop":
        return Crop(crop_ratio=config.params["crop_ratio"])
    elif config.class_name == "MPEG4":
        # Use the fixed MPEG4Wrapper from common_evaluator
        from evaluate.common_evaluator import MPEG4Wrapper

        return MPEG4Wrapper(quality=config.params["quality"], temp_dir=temp_dir)
    elif config.class_name == "FrameAverage":
        return FrameAverage(num_frames=config.params["num_frames"])
    elif config.class_name == "FrameRemove":
        return FrameRemove(p=config.params["p"])
    elif config.class_name == "FrameSwitch":
        return FrameSwitch(p=config.params["p"])
    else:
        raise ValueError(f"Unknown perturbation: {config.class_name}")


def apply_perturbation(
    pert, video: torch.Tensor, config: PerturbationConfig
) -> torch.Tensor:
    """
    Apply perturbation to video.

    Delegates to common_evaluator._apply_perturbation for consistent behavior.

    Args:
        pert: Perturbation module instance
        video: Video tensor (T, C, H, W) in [0, 1]
        config: Perturbation configuration

    Returns:
        Perturbed video tensor
    """
    from evaluate.common_evaluator import _apply_perturbation

    return _apply_perturbation(pert, video, config.class_name)


# Perturbations that can benefit from GPU acceleration
GPU_CAPABLE_PERTURBATIONS = {
    "Gaussian",
    "GaussianBlur",
    "FrameAverage",
    "FrameRemove",
    "FrameSwitch",
}
# Perturbations that must run on CPU (use albumentations or FFmpeg)
CPU_ONLY_PERTURBATIONS = {"JPEG", "Crop", "MPEG4"}


def process_single_video(
    video_path: Path,
    output_dir: Path,
    configs: List[PerturbationConfig],
    max_frames: Optional[int] = None,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[str, int, int]:
    """
    Process a single video through all perturbations.

    Args:
        video_path: Path to input video
        output_dir: Base output directory
        configs: List of perturbation configurations
        max_frames: Maximum frames per video (None = all)
        seed: Random seed for reproducibility
        device: Device for GPU-capable perturbations (cpu, cuda, cuda:0, etc.)

    Returns:
        Tuple of (video_name, success_count, total_count)
    """
    import tempfile

    # Set seed for reproducibility
    setup_seed(seed)

    video_name = video_path.stem
    success_count = 0
    total_count = len(configs)

    # Parse device
    torch_device = torch.device(device)
    use_gpu = torch_device.type == "cuda"

    try:
        # Load video (always loaded to CPU first)
        video_cpu = load_video(video_path)
        if max_frames and video_cpu.shape[0] > max_frames:
            video_cpu = video_cpu[:max_frames]

        # Pre-move to GPU if using CUDA (for GPU-capable perturbations)
        video_gpu = video_cpu.to(torch_device) if use_gpu else None

        # Create temp dir for MPEG4
        temp_dir = Path(tempfile.mkdtemp(prefix="precompute_mpeg4_"))

        try:
            for config in configs:
                try:
                    # Create output directory
                    pert_dir = output_dir / config.get_dir_name()
                    pert_dir.mkdir(parents=True, exist_ok=True)

                    output_file = pert_dir / f"{video_name}.pt"

                    # Skip if already exists
                    if output_file.exists():
                        success_count += 1
                        continue

                    # Create perturbation and apply
                    pert = create_perturbation(config, temp_dir=temp_dir)

                    # Use GPU for GPU-capable perturbations, CPU otherwise
                    if use_gpu and config.class_name in GPU_CAPABLE_PERTURBATIONS:
                        perturbed_video = apply_perturbation(pert, video_gpu, config)
                        # Move back to CPU for saving
                        perturbed_video = perturbed_video.cpu()
                    else:
                        # CPU-only perturbations (JPEG, Crop, MPEG4)
                        perturbed_video = apply_perturbation(pert, video_cpu, config)

                    # Save to disk (always on CPU)
                    torch.save(
                        {
                            "video": perturbed_video,
                            "original_path": str(video_path),
                            "perturbation": config.name,
                            "params": config.params,
                        },
                        output_file,
                    )
                    success_count += 1

                except Exception as e:
                    print(f"Error applying {config.name} to {video_name}: {e}")
                    continue

        finally:
            # Cleanup temp dir
            import shutil

            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

            # Clear GPU cache if used
            if use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error loading video {video_path}: {e}")

    return video_name, success_count, total_count


def get_video_files(data_dir: Path) -> List[Path]:
    """Get list of video files from directory."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_files = []
    for video_path in data_dir.glob("**/*"):
        if video_path.suffix.lower() in video_extensions:
            video_files.append(video_path)
    video_files.sort()
    return video_files


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute perturbations for VideoMarkBench evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Pre-compute all perturbations with 8 workers (CPU)
    python scripts/precompute_perturbations.py \\
        --dataset data/videomarkbench \\
        --output_dir data/precomputed_perturbations \\
        --workers 8

    # Pre-compute with GPU acceleration (requires --sequential)
    python scripts/precompute_perturbations.py \\
        --dataset data/videomarkbench \\
        --output_dir data/precomputed_perturbations \\
        --sequential \\
        --device cuda

    # Pre-compute for a subset of videos
    python scripts/precompute_perturbations.py \\
        --dataset data/videomarkbench \\
        --output_dir data/precomputed \\
        --max_videos 10 \\
        --workers 4

GPU-capable perturbations: Gaussian, GaussianBlur, FrameAverage, FrameRemove, FrameSwitch
CPU-only perturbations: JPEG, Crop, MPEG4
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to VideoMarkBench dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for pre-computed perturbations",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Maximum number of videos to process (default: all)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum frames per video (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        default=False,
        help="Process all perturbations sequentially (slower but more stable)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for GPU-capable perturbations: cpu, cuda, cuda:0, etc. (default: cpu)",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    device = args.device

    if not dataset_path.exists():
        print(f"Error: Dataset directory not found: {dataset_path}")
        sys.exit(1)

    # Validate device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("Error: CUDA is not available, falling back to CPU")
            device = "cpu"
        elif not args.sequential:
            print(
                "Warning: GPU mode requires --sequential flag (multiprocessing doesn't work well with CUDA)"
            )
            print("Falling back to CPU for parallel mode")
            device = "cpu"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video files
    video_files = get_video_files(dataset_path)
    if args.max_videos:
        video_files = video_files[: args.max_videos]

    if not video_files:
        print(f"No video files found in {dataset_path}")
        sys.exit(1)

    # Get perturbation configs (separated into non-MPEG4 and MPEG4 to avoid FFmpeg subprocess conflicts)
    non_mpeg4_configs, mpeg4_configs = get_perturbation_configs()
    all_configs = non_mpeg4_configs + mpeg4_configs

    print("=" * 70)
    print("Pre-compute Perturbations for VideoMarkBench")
    print("=" * 70)
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Videos: {len(video_files)}")
    print(f"Device: {device}")
    if device.startswith("cuda"):
        gpu_perts = [
            c.name for c in all_configs if c.class_name in GPU_CAPABLE_PERTURBATIONS
        ]
        cpu_perts = [
            c.name for c in all_configs if c.class_name in CPU_ONLY_PERTURBATIONS
        ]
        print(f"  GPU-accelerated: {len(gpu_perts)} perturbations")
        print(f"  CPU-only: {len(cpu_perts)} perturbations (JPEG, Crop, MPEG4)")
    if args.sequential:
        print(f"Perturbations: {len(all_configs)} (all sequential)")
        print("Mode: Sequential (stable)")
    else:
        print(
            f"Perturbations: {len(all_configs)} ({len(non_mpeg4_configs)} parallel + {len(mpeg4_configs)} sequential)"
        )
        print(f"Workers: {args.workers}")
    print(f"Total files to generate: {len(video_files) * len(all_configs)}")
    print("=" * 70)

    # Create output directories
    for config in all_configs:
        pert_dir = output_dir / config.get_dir_name()
        pert_dir.mkdir(parents=True, exist_ok=True)

    total_success = 0
    total_count = 0

    if args.sequential:
        # Sequential mode: process all perturbations one video at a time
        print("\n[Sequential] Processing all perturbations...")
        with tqdm(total=len(video_files), desc="Processing videos") as pbar:
            for video_path in video_files:
                try:
                    _, success, total = process_single_video(
                        video_path,
                        output_dir,
                        all_configs,
                        args.max_frames,
                        args.seed,
                        device,
                    )
                    total_success += success
                    total_count += total
                    pbar.set_postfix(
                        {"success": f"{total_success}/{total_count}"}, refresh=False
                    )
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                pbar.update(1)
    else:
        # Parallel mode: non-MPEG4 in parallel, MPEG4 sequential

        # Phase 1: Process non-MPEG4 perturbations in parallel
        if non_mpeg4_configs:
            print(
                f"\n[Phase 1/2] Processing {len(non_mpeg4_configs)} non-MPEG4 perturbations in parallel ({args.workers} workers)..."
            )
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        process_single_video,
                        video_path,
                        output_dir,
                        non_mpeg4_configs,
                        args.max_frames,
                        args.seed,
                        device,  # Always "cpu" in parallel mode (validated above)
                    ): video_path
                    for video_path in video_files
                }

                with tqdm(
                    total=len(video_files), desc="Non-MPEG4 perturbations"
                ) as pbar:
                    for future in as_completed(futures):
                        video_path = futures[future]
                        try:
                            _, success, total = future.result()
                            total_success += success
                            total_count += total
                            pbar.set_postfix(
                                {"success": f"{total_success}/{total_count}"},
                                refresh=False,
                            )
                        except Exception as e:
                            print(f"Error processing {video_path}: {e}")
                            # Count failed perturbations
                            total_count += len(non_mpeg4_configs)
                        pbar.update(1)

        # Phase 2: Process MPEG4 perturbations sequentially (to avoid FFmpeg subprocess conflicts)
        if mpeg4_configs:
            print(
                f"\n[Phase 2/2] Processing {len(mpeg4_configs)} MPEG4 perturbations sequentially..."
            )
            with tqdm(total=len(video_files), desc="MPEG4 perturbations") as pbar:
                for video_path in video_files:
                    try:
                        _, success, total = process_single_video(
                            video_path,
                            output_dir,
                            mpeg4_configs,
                            args.max_frames,
                            args.seed,
                            device,  # Always "cpu" in parallel mode (validated above)
                        )
                        total_success += success
                        total_count += total
                        pbar.set_postfix(
                            {"success": f"{total_success}/{total_count}"}, refresh=False
                        )
                    except Exception as e:
                        print(f"Error processing {video_path}: {e}")
                    pbar.update(1)

    # Save manifest
    manifest = {
        "dataset": str(dataset_path),
        "output_dir": str(output_dir),
        "num_videos": len(video_files),
        "perturbations": [
            {
                "name": c.name,
                "class": c.class_name,
                "params": c.params,
                "dir": c.get_dir_name(),
            }
            for c in all_configs
        ],
        "seed": args.seed,
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "total_files": total_count,
        "successful_files": total_success,
    }

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print("=" * 70)
    print("Pre-computation Complete")
    print("=" * 70)
    print(f"Successfully processed: {total_success}/{total_count} files")
    print(f"Manifest saved to: {manifest_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
