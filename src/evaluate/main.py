"""
Main evaluation script for VideoMarkBench.

Follows the original VideoMarkBench repository structure and evaluates
a model under all perturbation types (common, blackbox forgery, blackbox removal, whitebox)
with all aggregation strategies.
"""

import argparse
import sys
from pathlib import Path
import logging
import json
from datetime import datetime

import torch

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "videoseal"))

from utils import setup_seed, setup_logger, select_device  # noqa: E402
from adaptive_models.adaptive_videoseal import AdaptiveVideoseal  # noqa: E402
from videoseal.utils.cfg import setup_model_from_model_card  # noqa: E402

from evaluate.model_wrapper import VideoMarkBenchModel  # noqa: E402
from evaluate.dataset_loader import VideoMarkBenchDataset  # noqa: E402
from evaluate.common_evaluator import evaluate_common_perturbations  # noqa: E402


def setup_model(model_name: str, device: torch.device, args, logger=None) -> VideoMarkBenchModel:
    """
    Setup watermarking model.

    Args:
        model_name: Name of the model ('videoseal' or 'adaptive-videoseal')
        device: Device to run model on
        args: Command line arguments

    Returns:
        VideoMarkBenchModel wrapper instance
    """
    # Fixed model paths
    cards_dir = (
        Path(__file__).resolve().parent.parent / "videoseal" / "videoseal" / "cards"
    )
    model_card_path = cards_dir / "videoseal_1.0.yaml"
    if not model_card_path.exists():
        raise FileNotFoundError(f"Expected model card at {model_card_path}")

    # Load base model
    import os
    from contextlib import contextmanager

    @contextmanager
    def _temp_working_dir(path):
        """Temporarily change working directory."""
        old_cwd = os.getcwd()
        try:
            os.chdir(path)
            yield
        finally:
            os.chdir(old_cwd)

    with _temp_working_dir(cards_dir.parent.parent):
        base_model = setup_model_from_model_card(model_card_path)

    base_model = base_model.eval().to(device)
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

    # Evaluation arguments
    parser.add_argument(
        "--evaluate_common",
        action="store_true",
        default=True,
        help="Evaluate common perturbations (default: True)",
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
        logger.info("Please download the VideoMarkBench dataset and place it in the specified location.")
        return

    dataset = VideoMarkBenchDataset(dataset_path)
    logger.info(f"Loaded dataset with {len(dataset)} videos")

    # Randomly sample videos if max_videos is specified
    if args.max_videos and args.max_videos < len(dataset):
        import random
        # Use fixed seed for reproducibility
        random.seed(args.seed)
        all_video_files = dataset.get_video_files()
        selected_files = random.sample(all_video_files, args.max_videos)
        selected_files.sort()  # Sort for consistent ordering
        logger.info(f"Randomly selected {len(selected_files)} videos (seed={args.seed})")
        # Create iterator for selected videos only
        selected_dataset = dataset.get_selected_videos(selected_files)
    else:
        selected_dataset = dataset
        logger.info(f"Using all {len(dataset)} videos")

    # Run evaluations
    all_results = {}

    # 1. Common perturbations
    if args.evaluate_common:
        logger.info("=" * 50)
        logger.info("Evaluating Common Perturbations")
        logger.info("=" * 50)
        try:
            common_results = evaluate_common_perturbations(
                model=model,
                dataset=selected_dataset,
                output_dir=output_dir,
                logger=logger,
                max_videos=None,  # Already sampled in main
                max_frames_per_video=args.max_frames_per_video,
                evaluate_forgery=True,
            )
            # Load statistics if available
            stats_file = output_dir / "common_perturbations_stats.json"
            stats = {}
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
            
            all_results["common"] = {
                "file": str(output_dir / "common_perturbations_results.csv"),
                "stats_file": str(stats_file),
                "count": len(common_results),
                "statistics": stats,
            }
            logger.info(f"Common perturbations evaluation completed: {len(common_results)} results")
            
            # Log statistics
            if stats and "overall" in stats:
                overall = stats["overall"]
                if "FNR" in overall:
                    logger.info(f"Overall FNR (False Negative Rate): {overall['FNR']:.4f} ({overall.get('FNR_count', 0)}/{overall.get('removal_total', 0)})")
                if "FPR" in overall:
                    logger.info(f"Overall FPR (False Positive Rate): {overall['FPR']:.4f} ({overall.get('FPR_count', 0)}/{overall.get('forgery_total', 0)})")
        except Exception as e:
            logger.error(f"Error in common perturbations evaluation: {e}", exc_info=True)

    # 2. Blackbox forgery attacks (TODO: implement)
    if args.evaluate_blackbox_forgery:
        logger.info("=" * 50)
        logger.info("Evaluating Blackbox Forgery Attacks")
        logger.info("=" * 50)
        logger.warning("Blackbox forgery attacks evaluation not yet implemented")
        # TODO: Implement blackbox forgery evaluation

    # 3. Blackbox removal attacks (TODO: implement)
    if args.evaluate_blackbox_removal:
        logger.info("=" * 50)
        logger.info("Evaluating Blackbox Removal Attacks")
        logger.info("=" * 50)
        logger.warning("Blackbox removal attacks evaluation not yet implemented")
        # TODO: Implement blackbox removal evaluation

    # 4. Whitebox attacks (TODO: implement)
    if args.evaluate_whitebox:
        logger.info("=" * 50)
        logger.info("Evaluating Whitebox Attacks (WEvade)")
        logger.info("=" * 50)
        logger.warning("Whitebox attacks evaluation not yet implemented")
        # TODO: Implement whitebox evaluation

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
        "random_sampling": args.max_videos is not None and args.max_videos < len(dataset),
        "seed": args.seed if (args.max_videos and args.max_videos < len(dataset)) else None,
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
