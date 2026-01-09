#!/usr/bin/env python3
"""
Benchmark perturbations: CPU vs GPU speed comparison and result verification.

This script benchmarks GPU-capable perturbations to compare:
1. Execution time on CPU vs GPU
2. Result consistency (whether outputs are identical within tolerance)

Usage:
    # Single video
    python scripts/benchmark_perturbations.py --video path/to/video.mp4

    # Multiple videos from dataset (fast benchmark with --max_videos)
    python scripts/benchmark_perturbations.py --dataset data/videomarkbench --max_videos 5

    # Synthetic data for quick testing
    python scripts/benchmark_perturbations.py --synthetic

    # Custom device and settings
    python scripts/benchmark_perturbations.py --synthetic --device cuda:1 --num_runs 10
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import random

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(src_dir / "videomarkbench"))


def setup_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# GPU-capable perturbations to benchmark
GPU_CAPABLE_PERTURBATIONS = [
    ("Gaussian", {"std": 0.01}),
    ("Gaussian", {"std": 0.05}),
    ("GaussianBlur", {"std": 1.0}),
    ("GaussianBlur", {"std": 2.0}),
    ("FrameAverage", {"num_frames": 2}),
    ("FrameAverage", {"num_frames": 3}),
    ("FrameRemove", {"p": 0.1}),
    ("FrameRemove", {"p": 0.2}),
    ("FrameSwitch", {"p": 0.1}),
    ("FrameSwitch", {"p": 0.2}),
]


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    params: Dict
    cpu_time_ms: float
    gpu_time_ms: float
    speedup: float
    results_match: bool
    max_diff: float
    mean_diff: float


def create_perturbation(class_name: str, params: Dict):
    """Create a perturbation instance."""
    from common_perturabtions.frame_average import FrameAverage
    from common_perturabtions.frame_remove import FrameRemove
    from common_perturabtions.frame_switch import FrameSwitch
    from common_perturabtions.gaussian import Gaussian
    from common_perturabtions.gaussian_blur import GaussianBlur

    if class_name == "Gaussian":
        return Gaussian(std=params["std"])
    elif class_name == "GaussianBlur":
        return GaussianBlur(std=params["std"])
    elif class_name == "FrameAverage":
        return FrameAverage(num_frames=params["num_frames"])
    elif class_name == "FrameRemove":
        return FrameRemove(p=params["p"])
    elif class_name == "FrameSwitch":
        return FrameSwitch(p=params["p"])
    else:
        raise ValueError(f"Unknown perturbation: {class_name}")


def apply_perturbation(pert, video: torch.Tensor, class_name: str) -> torch.Tensor:
    """Apply perturbation to video."""
    from evaluate.common_evaluator import _apply_perturbation

    return _apply_perturbation(pert, video, class_name)


def load_video(video_path: Path) -> torch.Tensor:
    """Load video from file."""
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(
            str(video_path), width=512, height=512, num_threads=4, ctx=cpu(0)
        )
        num_frames = len(vr)
        if num_frames == 0:
            raise ValueError(f"No frames loaded from {video_path}")

        vid_np = vr.get_batch(range(num_frames)).asnumpy()
        vid_tensor = torch.from_numpy(vid_np).float()
        vid_tensor = vid_tensor.permute(0, 3, 1, 2) / 255.0
        return vid_tensor
    except ImportError:
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


def create_synthetic_video(
    num_frames: int = 16, height: int = 512, width: int = 512
) -> torch.Tensor:
    """Create synthetic video data for benchmarking."""
    # Create a video with gradients and patterns for meaningful testing
    video = torch.zeros(num_frames, 3, height, width)

    for t in range(num_frames):
        # Horizontal gradient (changes with time)
        x_grad = torch.linspace(0, 1, width).unsqueeze(0).expand(height, -1)
        # Vertical gradient
        y_grad = torch.linspace(0, 1, height).unsqueeze(1).expand(-1, width)
        # Time-varying component
        time_factor = t / num_frames

        video[t, 0] = x_grad * (1 - time_factor) + y_grad * time_factor  # R
        video[t, 1] = y_grad  # G
        video[t, 2] = 1 - x_grad  # B

        # Add some noise for realism
        video[t] += torch.randn_like(video[t]) * 0.02

    return video.clamp(0, 1)


def benchmark_perturbation(
    class_name: str,
    params: Dict,
    video_cpu: torch.Tensor,
    video_gpu: torch.Tensor,
    num_warmup: int = 2,
    num_runs: int = 5,
    seed: int = 42,
    tolerance: float = 1e-5,
) -> BenchmarkResult:
    """
    Benchmark a single perturbation on CPU and GPU.

    Args:
        class_name: Perturbation class name
        params: Perturbation parameters
        video_cpu: Video tensor on CPU
        video_gpu: Video tensor on GPU
        num_warmup: Number of warmup runs
        num_runs: Number of timed runs
        seed: Random seed for reproducibility
        tolerance: Tolerance for result comparison

    Returns:
        BenchmarkResult with timing and consistency info
    """
    name = f"{class_name}_{list(params.values())[0]}"

    # Warmup and benchmark CPU
    cpu_times = []
    setup_seed(seed)
    pert_cpu = create_perturbation(class_name, params)

    for i in range(num_warmup + num_runs):
        setup_seed(seed)  # Reset seed for each run
        start = time.perf_counter()
        result_cpu = apply_perturbation(pert_cpu, video_cpu, class_name)
        end = time.perf_counter()

        if i >= num_warmup:
            cpu_times.append((end - start) * 1000)  # Convert to ms

    # Warmup and benchmark GPU
    gpu_times = []
    setup_seed(seed)
    pert_gpu = create_perturbation(class_name, params)

    for i in range(num_warmup + num_runs):
        setup_seed(seed)  # Reset seed for each run

        # Synchronize before timing
        torch.cuda.synchronize()
        start = time.perf_counter()

        result_gpu = apply_perturbation(pert_gpu, video_gpu, class_name)

        # Synchronize after to ensure completion
        torch.cuda.synchronize()
        end = time.perf_counter()

        if i >= num_warmup:
            gpu_times.append((end - start) * 1000)  # Convert to ms

    # Calculate average times
    cpu_time_ms = sum(cpu_times) / len(cpu_times)
    gpu_time_ms = sum(gpu_times) / len(gpu_times)
    speedup = cpu_time_ms / gpu_time_ms if gpu_time_ms > 0 else float("inf")

    # Compare results
    result_gpu_cpu = result_gpu.cpu()

    # Handle shape differences (FrameRemove/FrameSwitch may have different frame counts)
    if result_cpu.shape != result_gpu_cpu.shape:
        # For frame operations, compare what we can
        min_frames = min(result_cpu.shape[0], result_gpu_cpu.shape[0])
        diff = torch.abs(result_cpu[:min_frames] - result_gpu_cpu[:min_frames])
    else:
        diff = torch.abs(result_cpu - result_gpu_cpu)

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    results_match = max_diff < tolerance

    return BenchmarkResult(
        name=name,
        params=params,
        cpu_time_ms=cpu_time_ms,
        gpu_time_ms=gpu_time_ms,
        speedup=speedup,
        results_match=results_match,
        max_diff=max_diff,
        mean_diff=mean_diff,
    )


def get_video_files(data_dir: Path) -> List[Path]:
    """Get list of video files from directory."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_files = []
    for video_path in data_dir.glob("**/*"):
        if video_path.suffix.lower() in video_extensions:
            video_files.append(video_path)
    video_files.sort()
    return video_files


def print_results(results: List[BenchmarkResult], video_shape: Tuple[int, ...]):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    print(f"Video shape: {video_shape} (T, C, H, W)")
    print("-" * 100)
    print(
        f"{'Perturbation':<25} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10} "
        f"{'Match':<8} {'Max Diff':<12} {'Mean Diff':<12}"
    )
    print("-" * 100)

    total_cpu = 0
    total_gpu = 0
    all_match = True

    for r in results:
        match_str = "YES" if r.results_match else "NO"
        print(
            f"{r.name:<25} {r.cpu_time_ms:<12.2f} {r.gpu_time_ms:<12.2f} {r.speedup:<10.2f}x "
            f"{match_str:<8} {r.max_diff:<12.2e} {r.mean_diff:<12.2e}"
        )
        total_cpu += r.cpu_time_ms
        total_gpu += r.gpu_time_ms
        if not r.results_match:
            all_match = False

    print("-" * 100)
    overall_speedup = total_cpu / total_gpu if total_gpu > 0 else float("inf")
    print(
        f"{'TOTAL':<25} {total_cpu:<12.2f} {total_gpu:<12.2f} {overall_speedup:<10.2f}x"
    )
    print("=" * 100)

    # Summary
    print("\nSUMMARY:")
    print(f"  Total CPU time: {total_cpu:.2f} ms")
    print(f"  Total GPU time: {total_gpu:.2f} ms")
    print(f"  Overall speedup: {overall_speedup:.2f}x")
    print(f"  Results consistency: {'ALL MATCH' if all_match else 'SOME DIFFER'}")

    if not all_match:
        print("\n  WARNING: Some results differ between CPU and GPU!")
        print("  Non-matching perturbations:")
        for r in results:
            if not r.results_match:
                print(f"    - {r.name}: max_diff={r.max_diff:.2e}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CPU vs GPU perturbation performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset directory (benchmarks multiple videos)",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Maximum number of videos to benchmark from dataset (default: all)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic video data instead of loading from file",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames for synthetic video (default: 16)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution (height=width) for synthetic video (default: 512)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU device: cuda, cuda:0, cuda:1, etc. (default: cuda)",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=2,
        help="Number of warmup runs (default: 2)",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of timed runs (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Tolerance for result comparison (default: 1e-5)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum frames to use from video (default: all)",
    )

    args = parser.parse_args()

    # Validate arguments
    num_sources = sum(
        [args.synthetic, args.video is not None, args.dataset is not None]
    )
    if num_sources == 0:
        parser.error("One of --video, --dataset, or --synthetic must be specified")
    if num_sources > 1:
        parser.error("Only one of --video, --dataset, or --synthetic can be specified")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    device = torch.device(args.device)
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")

    # Get list of videos to benchmark
    video_sources = []
    if args.synthetic:
        video_sources = [("synthetic", None)]
    elif args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            sys.exit(1)
        video_sources = [(video_path.name, video_path)]
    else:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Error: Dataset directory not found: {dataset_path}")
            sys.exit(1)
        video_files = get_video_files(dataset_path)
        if args.max_videos:
            video_files = video_files[: args.max_videos]
        if not video_files:
            print(f"No video files found in {dataset_path}")
            sys.exit(1)
        video_sources = [(vf.name, vf) for vf in video_files]

    print(f"\nVideos to benchmark: {len(video_sources)}")
    print(f"Perturbations: {len(GPU_CAPABLE_PERTURBATIONS)}")
    print(f"Warmup runs: {args.num_warmup}, Timed runs: {args.num_runs}")
    print(f"Seed: {args.seed}, Tolerance: {args.tolerance}")

    # Aggregate results across all videos
    all_results = []
    all_match = True

    for video_idx, (video_name, video_path) in enumerate(video_sources):
        print(f"\n{'=' * 70}")
        print(f"[{video_idx + 1}/{len(video_sources)}] {video_name}")
        print("=" * 70)

        # Load or create video
        if video_path is None:
            print(
                f"Creating synthetic video: {args.num_frames} frames, {args.resolution}x{args.resolution}"
            )
            video_cpu = create_synthetic_video(
                num_frames=args.num_frames,
                height=args.resolution,
                width=args.resolution,
            )
        else:
            print(f"Loading video: {video_path}")
            video_cpu = load_video(video_path)

        # Limit frames if specified
        if args.max_frames and video_cpu.shape[0] > args.max_frames:
            video_cpu = video_cpu[: args.max_frames]

        print(f"Video shape: {tuple(video_cpu.shape)} (T, C, H, W)")

        # Move to GPU
        video_gpu = video_cpu.to(device)

        # Run benchmarks for this video
        results = []
        for class_name, params in GPU_CAPABLE_PERTURBATIONS:
            name = f"{class_name}_{list(params.values())[0]}"
            print(f"  Benchmarking {name}...", end=" ", flush=True)

            try:
                result = benchmark_perturbation(
                    class_name=class_name,
                    params=params,
                    video_cpu=video_cpu,
                    video_gpu=video_gpu,
                    num_warmup=args.num_warmup,
                    num_runs=args.num_runs,
                    seed=args.seed,
                    tolerance=args.tolerance,
                )
                results.append(result)
                all_results.append(result)
                status = "OK" if result.results_match else "DIFF"
                print(f"{status} (speedup: {result.speedup:.2f}x)")
                if not result.results_match:
                    all_match = False
            except Exception as e:
                print(f"ERROR: {e}")

        # Print results for this video
        print_results(results, tuple(video_cpu.shape))

        # Clear GPU memory
        del video_gpu
        torch.cuda.empty_cache()

    # Print aggregate summary if multiple videos
    if len(video_sources) > 1:
        print("\n" + "=" * 100)
        print("AGGREGATE SUMMARY")
        print("=" * 100)
        total_cpu = sum(r.cpu_time_ms for r in all_results)
        total_gpu = sum(r.gpu_time_ms for r in all_results)
        overall_speedup = total_cpu / total_gpu if total_gpu > 0 else float("inf")
        print(f"Total videos: {len(video_sources)}")
        print(f"Total CPU time: {total_cpu:.2f} ms")
        print(f"Total GPU time: {total_gpu:.2f} ms")
        print(f"Overall speedup: {overall_speedup:.2f}x")
        print(f"Results consistency: {'ALL MATCH' if all_match else 'SOME DIFFER'}")
        print("=" * 100)

    # Return exit code based on result consistency
    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
