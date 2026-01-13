#!/usr/bin/env python
"""
Benchmark script to compare sequential vs parallel evaluation performance.

Usage:
    PYTHONPATH=src python scripts/benchmark_parallel.py \
        --dataset data/videomarkbench \
        --max_videos 5 \
        --num_gpus 4
"""

import argparse
import time
import subprocess
import sys
from pathlib import Path
import pandas as pd


def run_evaluation(
    mode: str,
    dataset: str,
    output_dir: Path,
    max_videos: int,
    max_frames: int,
    num_gpus: int = None,
) -> float:
    """Run evaluation and return elapsed time in seconds."""
    cmd = [
        sys.executable,
        "src/evaluate/main.py",
        "--model",
        "videoseal",
        "--dataset",
        dataset,
        "--evaluate_common",
        "--max_videos",
        str(max_videos),
        "--max_frames_per_video",
        str(max_frames),
        "--output_dir",
        str(output_dir),
    ]

    if mode == "parallel":
        cmd.append("--parallel")
        if num_gpus:
            cmd.extend(["--num_gpus", str(num_gpus)])

    print(f"\n{'='*60}")
    print(f"Running {mode.upper()} mode...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=False,
    )
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"WARNING: {mode} evaluation failed with return code {result.returncode}")

    return elapsed


def compare_results(seq_dir: Path, par_dir: Path) -> bool:
    """Compare results from sequential and parallel runs."""
    seq_file = seq_dir / "common_perturbations_results.csv"
    par_file = par_dir / "common_perturbations_results.csv"

    if not seq_file.exists() or not par_file.exists():
        print("WARNING: Result files not found for comparison")
        return False

    seq_df = pd.read_csv(seq_file)
    par_df = pd.read_csv(par_file)

    # Skip Gaussian and Crop perturbation from comparison (CPU vs GPU RNG produces different results)
    # Results are statistically equivalent but not bit-identical
    skip_perturbations = ["Gaussian", "Crop"]
    seq_df = seq_df[~seq_df["perturbation"].isin(skip_perturbations)]
    par_df = par_df[~par_df["perturbation"].isin(skip_perturbations)]
    print(f"Skipping perturbations from comparison: {skip_perturbations}")

    # Sort both dataframes for comparison
    sort_cols = ["video", "perturbation", "aggregation", "attack_type"]
    seq_df = seq_df.sort_values(sort_cols).reset_index(drop=True)
    par_df = par_df.sort_values(sort_cols).reset_index(drop=True)

    if len(seq_df) != len(par_df):
        print(
            f"WARNING: Result count mismatch: sequential={len(seq_df)}, parallel={len(par_df)}"
        )
        return False

    # Compare detected column
    matches = (seq_df["detected"] == par_df["detected"]).all()
    if matches:
        print("Results MATCH: Sequential and parallel produce identical results")
    else:
        mismatches = (seq_df["detected"] != par_df["detected"]).sum()
        print(
            f"WARNING: {mismatches} mismatches found between sequential and parallel results"
        )

    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark parallel vs sequential evaluation"
    )
    parser.add_argument(
        "--dataset", type=str, default="data/videomarkbench", help="Dataset path"
    )
    parser.add_argument(
        "--max_videos", type=int, default=3, help="Number of videos to evaluate"
    )
    parser.add_argument(
        "--max_frames", type=int, default=16, help="Max frames per video"
    )
    parser.add_argument(
        "--num_gpus", type=int, default=None, help="Number of GPUs for parallel mode"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="outputs/benchmark",
        help="Base output directory",
    )
    args = parser.parse_args()

    output_base = Path(args.output_base)
    seq_dir = output_base / "sequential"
    par_dir = output_base / "parallel"

    # Clean up previous runs
    for d in [seq_dir, par_dir]:
        d.mkdir(parents=True, exist_ok=True)

    results = {}

    # Run sequential evaluation
    results["sequential"] = run_evaluation(
        mode="sequential",
        dataset=args.dataset,
        output_dir=seq_dir,
        max_videos=args.max_videos,
        max_frames=args.max_frames,
    )

    # Run parallel evaluation
    results["parallel"] = run_evaluation(
        mode="parallel",
        dataset=args.dataset,
        output_dir=par_dir,
        max_videos=args.max_videos,
        max_frames=args.max_frames,
        num_gpus=args.num_gpus,
    )

    # Compare results
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    results_match = compare_results(seq_dir, par_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Videos: {args.max_videos}")
    print(f"Max frames per video: {args.max_frames}")
    print(f"GPUs: {args.num_gpus or 'auto'}")
    print("-" * 60)
    print(f"Sequential time: {results['sequential']:.2f} seconds")
    print(f"Parallel time:   {results['parallel']:.2f} seconds")
    print("-" * 60)

    if results["parallel"] > 0:
        speedup = results["sequential"] / results["parallel"]
        print(f"Speedup: {speedup:.2f}x")

        if speedup > 1:
            print(f"Parallel mode is {speedup:.2f}x FASTER")
        else:
            print(
                f"Parallel mode is {1/speedup:.2f}x SLOWER (overhead may exceed benefit for small workloads)"
            )

    print("-" * 60)
    print(f"Results match: {'YES' if results_match else 'NO'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
