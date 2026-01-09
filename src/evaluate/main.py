"""
Main evaluation script for VideoMarkBench.

Follows the original VideoMarkBench repository structure and evaluates
a model under all perturbation types (common, blackbox forgery, blackbox removal, whitebox)
with all aggregation strategies.
"""

import argparse
import sys
from pathlib import Path
import json
from datetime import datetime
import random

import torch

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "videoseal"))

from utils import setup_seed, setup_logger, select_device, get_videoseal  # noqa: E402
from adaptive_models.adaptive_videoseal import AdaptiveVideoseal  # noqa: E402

from evaluate.model_wrapper import VideoMarkBenchModel  # noqa: E402
from evaluate.dataset_loader import VideoMarkBenchDataset  # noqa: E402
from evaluate.common_evaluator import evaluate_common_perturbations  # noqa: E402
from evaluate.whitebox_evaluator import evaluate_whitebox_attacks  # noqa: E402
from evaluate.blackbox_evaluator import evaluate_blackbox_attacks  # noqa: E402
from evaluate.parallel_evaluator import evaluate_common_perturbations_parallel  # noqa: E402


def setup_model(
    model_name: str, device: torch.device, args, logger=None
) -> VideoMarkBenchModel:
    """
    Setup watermarking model.

    Args:
        model_name: Name of the model ('videoseal' or 'adaptive-videoseal')
        device: Device to run model on
        args: Command line arguments

    Returns:
        VideoMarkBenchModel wrapper instance
    """
    base_model = get_videoseal().eval().to(device)
    base_model.compile()

    # Configuration for base model
    base_model.step_size = args.step_size
    base_model.blender.scaling_w *= args.scaling_w
    base_model.video_mode = args.video_mode
    base_model.chunk_size = args.chunk_size

    if model_name == "adaptive-videoseal":
        # Create adaptive model using from_videoseal class method
        # Following the pattern from poc.py
        from scene_detectors.registry import detector_registry

        # Convert base model to adaptive model (preserves all configurations)
        adaptive_model = AdaptiveVideoseal.from_videoseal(base_model)

        # Scene detector selection
        try:
            scene_detector = detector_registry[args.scene_detector](device=str(device))
        except KeyError:
            raise ValueError(f"Unknown scene detector: {args.scene_detector}")

        # Set scene detector and logger (following poc.py pattern)
        adaptive_model.set_scene_detector(scene_detector)
        adaptive_model.set_logger(logger)

        # Final setup
        adaptive_model = adaptive_model.eval().to(device)
        adaptive_model.compile()

        # Ensure chunk_size is set (following poc.py pattern)
        adaptive_model.chunk_size = args.chunk_size

        return VideoMarkBenchModel(adaptive_model, model_name="adaptive-videoseal")
    else:
        return VideoMarkBenchModel(base_model, model_name="videoseal")


def main():
    """Main entry point for VideoMarkBench evaluation."""
    parser = argparse.ArgumentParser(
        description="VideoMarkBench: Comprehensive video watermarking robustness evaluation"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="videoseal",
        choices=["videoseal", "adaptive-videoseal"],
        help="Watermarking model to evaluate",
    )
    parser.add_argument(
        "--scene_detector",
        type=str,
        default="content",
        choices=["content", "histogram", "hash", "lpips"],
        help="Scene detector for adaptive-videoseal",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=8,
        help="Step size parameter for embedding",
    )
    parser.add_argument(
        "--scaling_w",
        type=float,
        default=1.5,
        help="Scaling weight for watermark blending",
    )
    parser.add_argument(
        "--video_mode",
        type=str,
        default="repeat",
        choices=["repeat", "alternate", "interpolate"],
        help="Video mode for expanding key-frame predictions",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=32,
        help="Number of video frames to process in a single batch",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/videomarkbench",
        help="Path to VideoMarkBench dataset directory",
    )
    parser.add_argument(
        "--kinetics_dataset",
        type=str,
        default=None,
        help="Path to Kinetics dataset directory for forgery attacks (default: None, uses same dataset)",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Maximum number of videos to evaluate (None = all)",
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=None,
        help="Maximum frames per video (None = all)",
    )

    # Pre-computed perturbations
    parser.add_argument(
        "--precomputed_dir",
        type=str,
        default=None,
        help="Path to pre-computed perturbations directory. "
        "If provided, forgery attacks will load pre-computed perturbed videos. "
        "Generate with: python scripts/precompute_perturbations.py",
    )

    # Evaluation arguments
    parser.add_argument(
        "--evaluate_common",
        action="store_true",
        default=False,
        help="Evaluate common perturbations (default: False)",
    )
    parser.add_argument(
        "--evaluate_blackbox_forgery",
        action="store_true",
        default=False,
        help="Evaluate blackbox forgery attacks",
    )
    parser.add_argument(
        "--evaluate_blackbox_removal",
        action="store_true",
        default=False,
        help="Evaluate blackbox removal attacks",
    )
    parser.add_argument(
        "--evaluate_whitebox",
        action="store_true",
        default=False,
        help="Evaluate whitebox attacks (WEvade)",
    )
    parser.add_argument(
        "--blackbox_attack_method",
        type=str,
        default="square",
        choices=["square", "triangle"],
        help="Blackbox attack method: 'square' or 'triangle' (default: 'square')",
    )

    # Whitebox attack parameters
    parser.add_argument(
        "--wevade_version",
        type=str,
        default="standard",
        choices=["standard", "adaptive"],
        help="WEvade version: 'standard' (full PGD) or 'adaptive' (simplified)",
    )
    parser.add_argument(
        "--wevade_iteration",
        type=int,
        default=200,
        help="Max iterations for standard WEvade (default: 200)",
    )
    parser.add_argument(
        "--wevade_epsilon",
        type=float,
        default=4 / 96,
        help="Early stopping threshold for standard WEvade (default: 4/96)",
    )
    parser.add_argument(
        "--wevade_alpha",
        type=float,
        default=0.01,
        help="Learning rate for standard WEvade (default: 0.01)",
    )
    parser.add_argument(
        "--wevade_rb",
        type=float,
        default=0.1,
        help="Perturbation bound (L∞ norm) for standard WEvade (default: 0.1)",
    )
    parser.add_argument(
        "--wevade_type",
        type=str,
        default="WEvade-W-I",
        choices=["WEvade-W-I", "WEvade-W-II"],
        help="WEvade attack type (default: WEvade-W-I)",
    )
    parser.add_argument(
        "--secret_size",
        type=int,
        default=96,
        help="Watermark secret size (default: 96)",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/videomarkbench",
        help="Directory to save evaluation results",
    )

    # General arguments
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

    # Parallel processing arguments
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Enable multi-GPU parallel evaluation for common perturbations",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use for parallel evaluation (auto-detect if None)",
    )
    parser.add_argument(
        "--fast_aggregation",
        action="store_true",
        default=False,
        help="Use only top 2 aggregation strategies (logit_mean, bit_majority) for faster evaluation",
    )

    args = parser.parse_args()

    # Setup
    device = select_device(args.device)
    setup_seed(args.seed)

    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    logger = setup_logger(log_dir)

    logger.info("===== VideoMarkBench Evaluation =====")
    for key, value in vars(args).items():
        logger.info(f"{key:<25} : {value}")
    logger.info("====================================")

    # Setup model
    logger.info(f"Loading {args.model} model...")
    model = setup_model(args.model, device, args, logger)
    logger.info(f"Model loaded on {device}")

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.warning(f"Dataset directory not found: {dataset_path}")
        logger.info(
            "Please download the VideoMarkBench dataset and place it in the specified location."
        )
        return

    dataset = VideoMarkBenchDataset(dataset_path, logger=logger)
    logger.info(f"Loaded dataset with {len(dataset)} videos")

    # Randomly sample videos if max_videos is specified
    selected_files = None
    if args.max_videos and args.max_videos < len(dataset):
        # Use fixed seed for reproducibility
        random.seed(args.seed)
        all_video_files = dataset.get_video_files()
        selected_files = random.sample(all_video_files, args.max_videos)
        selected_files.sort()  # Sort for consistent ordering
        logger.info(
            f"Randomly selected {len(selected_files)} videos (seed={args.seed})"
        )
        # Create iterator for selected videos only
        selected_dataset = dataset.get_selected_videos(selected_files)
    else:
        selected_dataset = dataset
        logger.info(f"Using all {len(dataset)} videos")

    # Run evaluations
    all_results = {}

    # Load kinetics dataset for forgery attacks (if specified)
    kinetics_dataset = None
    kinetics_dataset_obj = None
    selected_kinetics_files = None
    if args.kinetics_dataset:
        kinetics_path = Path(args.kinetics_dataset)
        if kinetics_path.exists():
            kinetics_dataset_obj = VideoMarkBenchDataset(kinetics_path, logger=logger)
            logger.info(
                f"Loaded Kinetics dataset with {len(kinetics_dataset_obj)} videos for forgery attacks"
            )
            # Create iterator for kinetics dataset
            if args.max_videos and args.max_videos < len(kinetics_dataset_obj):
                random.seed(args.seed)
                kinetics_files = kinetics_dataset_obj.get_video_files()
                selected_kinetics_files = random.sample(kinetics_files, args.max_videos)
                selected_kinetics_files.sort()
                kinetics_dataset = kinetics_dataset_obj.get_selected_videos(
                    selected_kinetics_files
                )
            else:
                kinetics_dataset = kinetics_dataset_obj
        else:
            logger.warning(f"Kinetics dataset directory not found: {kinetics_path}")
            logger.info("Forgery attacks will use the same dataset as removal attacks")

    # 1. Common perturbations
    if args.evaluate_common:
        logger.info("=" * 50)
        logger.info("Evaluating Common Perturbations")
        logger.info("=" * 50)
        try:
            if args.parallel:
                # Use parallel multi-GPU evaluation
                logger.info(
                    f"Using parallel evaluation with {args.num_gpus or 'auto'} GPUs"
                )

                # Prepare model args for worker processes
                model_args = {
                    "step_size": args.step_size,
                    "scaling_w": args.scaling_w,
                    "video_mode": args.video_mode,
                    "chunk_size": args.chunk_size,
                    "scene_detector": args.scene_detector,
                }

                common_results = evaluate_common_perturbations_parallel(
                    model_name=args.model,
                    model_args=model_args,
                    dataset=selected_dataset,
                    output_dir=output_dir,
                    logger=logger,
                    num_gpus=args.num_gpus,
                    max_frames_per_video=args.max_frames_per_video,
                    evaluate_forgery=True,
                    kinetics_dataset=kinetics_dataset,
                    precomputed_dir=Path(args.precomputed_dir)
                    if args.precomputed_dir
                    else None,
                    fast_aggregation=args.fast_aggregation,
                )
            else:
                # Use sequential single-GPU evaluation
                common_results = evaluate_common_perturbations(
                    model=model,
                    dataset=selected_dataset,
                    output_dir=output_dir,
                    logger=logger,
                    max_frames_per_video=args.max_frames_per_video,
                    evaluate_forgery=True,
                    kinetics_dataset=kinetics_dataset,
                    precomputed_dir=Path(args.precomputed_dir)
                    if args.precomputed_dir
                    else None,
                    fast_aggregation=args.fast_aggregation,
                )
            # Load statistics if available
            stats_file = output_dir / "common_perturbations_stats.json"
            stats = {}
            if stats_file.exists():
                with open(stats_file, "r") as f:
                    stats = json.load(f)

            all_results["common"] = {
                "file": str(output_dir / "common_perturbations_results.csv"),
                "stats_file": str(stats_file),
                "count": len(common_results),
                "statistics": stats,
            }
            logger.info(
                f"Common perturbations evaluation completed: {len(common_results)} results"
            )

            # Log statistics
            if stats and "overall" in stats:
                overall = stats["overall"]
                if "FNR" in overall:
                    logger.info(
                        f"Overall FNR (False Negative Rate): {overall['FNR']:.4f} ({overall.get('FNR_count', 0)}/{overall.get('removal_total', 0)})"
                    )
                if "FPR" in overall:
                    logger.info(
                        f"Overall FPR (False Positive Rate): {overall['FPR']:.4f} ({overall.get('FPR_count', 0)}/{overall.get('forgery_total', 0)})"
                    )
        except Exception as e:
            logger.error(
                f"Error in common perturbations evaluation: {e}", exc_info=True
            )

    # 2. Blackbox attacks
    if args.evaluate_blackbox_forgery or args.evaluate_blackbox_removal:
        logger.info("=" * 50)
        logger.info("Evaluating Blackbox Attacks")
        logger.info("=" * 50)
        try:
            # Recreate iterator for blackbox evaluation (iterator may be exhausted from common evaluation)
            if selected_files is not None:
                blackbox_dataset = dataset.get_selected_videos(selected_files)
            else:
                blackbox_dataset = dataset

            # Recreate kinetics dataset iterator for blackbox (may be exhausted from common evaluation)
            blackbox_kinetics_dataset = None
            if kinetics_dataset_obj is not None:
                # Recreate iterator for kinetics dataset
                if (
                    args.max_videos
                    and args.max_videos < len(kinetics_dataset_obj)
                    and selected_kinetics_files is not None
                ):
                    blackbox_kinetics_dataset = (
                        kinetics_dataset_obj.get_selected_videos(
                            selected_kinetics_files
                        )
                    )
                else:
                    blackbox_kinetics_dataset = kinetics_dataset_obj

            blackbox_results = evaluate_blackbox_attacks(
                model=model,
                dataset=blackbox_dataset,
                output_dir=output_dir,
                logger=logger,
                max_videos=args.max_videos,
                max_frames_per_video=args.max_frames_per_video,
                attack_method=args.blackbox_attack_method
                if hasattr(args, "blackbox_attack_method")
                else "square",
                evaluate_forgery=args.evaluate_blackbox_forgery,
                evaluate_removal=args.evaluate_blackbox_removal,
                kinetics_dataset=blackbox_kinetics_dataset,
                eps=0.05,
                n_iters=1000,
                p_init=0.05,
            )
            # Load statistics if available
            stats_file = output_dir / "blackbox_stats.json"
            stats = {}
            if stats_file.exists():
                with open(stats_file, "r") as f:
                    stats = json.load(f)

            all_results["blackbox"] = {
                "file": str(output_dir / "blackbox_results.csv"),
                "stats_file": str(stats_file),
                "count": len(blackbox_results),
                "statistics": stats,
                "attack_method": "square",
            }
            logger.info(
                f"Blackbox attacks evaluation completed: {len(blackbox_results)} results"
            )

            # Log statistics
            if stats and "overall" in stats:
                overall = stats["overall"]
                if "FNR" in overall:
                    logger.info(
                        f"Overall FNR (False Negative Rate): {overall['FNR']:.4f} ({overall.get('FNR_count', 0)}/{overall.get('removal_total', 0)})"
                    )
                if "FPR" in overall:
                    logger.info(
                        f"Overall FPR (False Positive Rate): {overall['FPR']:.4f} ({overall.get('FPR_count', 0)}/{overall.get('forgery_total', 0)})"
                    )
        except Exception as e:
            logger.error(f"Error in blackbox attacks evaluation: {e}", exc_info=True)

    # 4. Whitebox attacks
    if args.evaluate_whitebox:
        logger.info("=" * 50)
        logger.info("Evaluating Whitebox Attacks (WEvade)")
        logger.info("=" * 50)
        try:
            # Recreate iterator for whitebox evaluation (iterator may be exhausted from common evaluation)
            if selected_files is not None:
                whitebox_dataset = dataset.get_selected_videos(selected_files)
            else:
                whitebox_dataset = dataset

            # Recreate kinetics dataset iterator for whitebox (may be exhausted from common evaluation)
            whitebox_kinetics_dataset = None
            if kinetics_dataset_obj is not None:
                # Recreate iterator for kinetics dataset
                if (
                    args.max_videos
                    and args.max_videos < len(kinetics_dataset_obj)
                    and selected_kinetics_files is not None
                ):
                    whitebox_kinetics_dataset = (
                        kinetics_dataset_obj.get_selected_videos(
                            selected_kinetics_files
                        )
                    )
                else:
                    whitebox_kinetics_dataset = kinetics_dataset_obj

            whitebox_results = evaluate_whitebox_attacks(
                model=model,
                dataset=whitebox_dataset,
                output_dir=output_dir,
                logger=logger,
                max_frames_per_video=args.max_frames_per_video,
                wevade_version=args.wevade_version,
                wevade_iteration=args.wevade_iteration,
                wevade_epsilon=args.wevade_epsilon,
                wevade_alpha=args.wevade_alpha,
                wevade_rb=args.wevade_rb,
                wevade_type=args.wevade_type,
                secret_size=args.secret_size,
                evaluate_forgery=True,
                kinetics_dataset=whitebox_kinetics_dataset,
            )
            # Load statistics if available
            stats_file = output_dir / "whitebox_stats.json"
            stats = {}
            if stats_file.exists():
                with open(stats_file, "r") as f:
                    stats = json.load(f)

            all_results["whitebox"] = {
                "file": str(output_dir / "whitebox_results.csv"),
                "stats_file": str(stats_file),
                "count": len(whitebox_results),
                "statistics": stats,
                "wevade_version": args.wevade_version,
            }
            logger.info(
                f"Whitebox attacks evaluation completed: {len(whitebox_results)} results"
            )

            # Log statistics
            if stats and "overall" in stats:
                overall = stats["overall"]
                if "FNR" in overall:
                    logger.info(
                        f"Overall FNR (False Negative Rate): {overall['FNR']:.4f} ({overall.get('FNR_count', 0)}/{overall.get('removal_total', 0)})"
                    )
                if "FPR" in overall:
                    logger.info(
                        f"Overall FPR (False Positive Rate): {overall['FPR']:.4f} ({overall.get('FPR_count', 0)}/{overall.get('forgery_total', 0)})"
                    )
        except Exception as e:
            logger.error(f"Error in whitebox attacks evaluation: {e}", exc_info=True)

    # Save summary
    # Count evaluated videos
    if args.max_videos and args.max_videos < len(dataset):
        evaluated_count = args.max_videos
    else:
        evaluated_count = len(dataset)

    summary = {
        "model": args.model,
        "dataset": str(dataset_path),
        "total_videos": len(dataset),
        "evaluated_videos": evaluated_count,
        "random_sampling": args.max_videos is not None
        and args.max_videos < len(dataset),
        "seed": args.seed
        if (args.max_videos and args.max_videos < len(dataset))
        else None,
        "evaluations": all_results,
        "timestamp": datetime.now().isoformat(),
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 50)
    logger.info("Evaluation Summary")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model}")
    logger.info(f"Total videos: {len(dataset)}")
    logger.info(f"Evaluated videos: {evaluated_count}")
    if args.max_videos and args.max_videos < len(dataset):
        logger.info(f"Random sampling: Yes (seed={args.seed})")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
