"""
Main script for running VideoMarkBench evaluation.

Usage:
    python -m videomarkbench.main --model videoseal --dataset data/videomarkbench --output_dir outputs/videomarkbench
"""

import argparse
import sys
from pathlib import Path
import shutil

import torch

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "videoseal"))

from utils import setup_seed, setup_logger, select_device  # noqa: E402
from main import AdaptiveVideoseal  # noqa: E402
from videoseal.utils.cfg import setup_model_from_model_card  # noqa: E402

from videomarkbench.models import get_model_wrapper  # noqa: E402
from videomarkbench.dataset import VideoMarkBenchDataset, download_dataset  # noqa: E402
from videomarkbench.evaluator import VideoMarkBenchEvaluator  # noqa: E402
from videomarkbench.common_perturabtions import (  # noqa: E402
    Crop,
    FrameAverage,
    FrameRemove,
    FrameSwitch,
    Gaussian,
    GaussianBlur,
    JPEG,
    MPEG4,
)
from videomarkbench.aggregations import AGGREGATION_STRATEGIES  # noqa: E402


def setup_model(model_name: str, device: torch.device, args) -> torch.nn.Module:
    """
    Setup watermarking model.

    Args:
        model_name: Name of the model ('videoseal' or 'adaptive-videoseal')
        device: Device to run model on
        args: Command line arguments

    Returns:
        Model instance
    """
    # Fixed model paths
    cards_dir = (
        Path(__file__).resolve().parent.parent / "videoseal" / "videoseal" / "cards"
    )
    model_card_path = cards_dir / "videoseal_1.0.yaml"
    if not model_card_path.exists():
        raise FileNotFoundError(f"Expected model card at {model_card_path}")

    cards_root = cards_dir.parent.parent
    target_ckpt_dir = cards_root / "ckpts"
    target_ckpt_dir.mkdir(parents=True, exist_ok=True)
    target_ckpt = target_ckpt_dir / "y_256b_img.pth"
    source_ckpt = (
        Path(__file__).resolve().parent.parent.parent / "ckpts" / "y_256b_img.pth"
    )

    if (
        not target_ckpt.exists() or target_ckpt.stat().st_size < 100_000_000
    ) and source_ckpt.exists():
        shutil.copy2(source_ckpt, target_ckpt)

    # Load base model
    import sys

    # Path is already imported at the top of the file
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from main import _TempWorkingDir

    with _TempWorkingDir(cards_dir.parent.parent):
        model = setup_model_from_model_card(model_card_path)
    model = model.eval().to(device)
    model.compile()

    # Configuration
    model.step_size = args.step_size
    model.blender.scaling_w *= args.scaling_w
    model.video_mode = args.video_mode
    model.chunk_size = args.chunk_size

    if model_name == "adaptive-videoseal":
        # Create adaptive model
        from scene_detectors.registry import detector_registry

        adaptive_model = AdaptiveVideoseal(
            model.embedder, model.detector, model.augmenter
        )
        adaptive_model.load_state_dict(model.state_dict(), strict=False)
        adaptive_model.__dict__.update(model.__dict__)

        # Scene detector selection
        try:
            scene_detector = detector_registry[args.scene_detector](device=str(device))
        except KeyError:
            raise ValueError(f"Unknown scene detector: {args.scene_detector}")

        adaptive_model.set_scene_detector(scene_detector)
        adaptive_model.eval().to(device)
        adaptive_model.compile()

        adaptive_model.step_size = args.step_size
        adaptive_model.blender.scaling_w *= args.scaling_w
        adaptive_model.attenuation = model.attenuation
        adaptive_model.video_mode = args.video_mode
        adaptive_model.chunk_size = args.chunk_size

        return adaptive_model
    else:
        return model


def get_perturbations(perturbation_types: list) -> list:
    """
    Get list of perturbations based on types.
    Based on original VideoMarkBench structure.

    Args:
        perturbation_types: List of perturbation type names or 'all'

    Returns:
        List of perturbation instances (nn.Module)
    """
    perturbations = []

    if "all" in perturbation_types:
        perturbation_types = ["common", "blackbox", "whitebox"]

    if "common" in perturbation_types:
        # Add common perturbations (matching original VideoMarkBench)
        perturbations.append(JPEG(quality=90))
        perturbations.append(JPEG(quality=75))
        perturbations.append(JPEG(quality=50))
        perturbations.append(Gaussian(std=0.01))
        perturbations.append(Gaussian(std=0.05))
        perturbations.append(GaussianBlur(std=1.0))
        perturbations.append(GaussianBlur(std=2.0))
        perturbations.append(Crop(crop_ratio=0.9))
        perturbations.append(Crop(crop_ratio=0.8))
        perturbations.append(MPEG4(quality=28, codec="libx264"))
        perturbations.append(MPEG4(quality=40, codec="libx264"))
        perturbations.append(FrameAverage(num_frames=2))
        perturbations.append(FrameAverage(num_frames=3))
        perturbations.append(FrameRemove(p=0.1))
        perturbations.append(FrameRemove(p=0.2))
        perturbations.append(FrameSwitch(p=0.1))
        perturbations.append(FrameSwitch(p=0.2))

    if "blackbox" in perturbation_types:
        # TODO: Implement black-box perturbations
        # perturbations.append(SquareAttack(...))
        # perturbations.append(TriangleAttack(...))
        pass

    if "whitebox" in perturbation_types:
        # TODO: Implement white-box perturbations
        # perturbations.append(WEvade(...))
        pass

    return perturbations


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

    # Perturbation arguments
    parser.add_argument(
        "--perturbations",
        type=str,
        nargs="+",
        default=["common"],
        choices=["all", "common", "blackbox", "whitebox"],
        help="Types of perturbations to apply",
    )

    # Aggregation arguments
    parser.add_argument(
        "--aggregations",
        type=str,
        nargs="+",
        default=None,
        choices=list(AGGREGATION_STRATEGIES.keys()),
        help=f"Aggregation strategies to use. Available: {', '.join(AGGREGATION_STRATEGIES.keys())}. "
        f"If not specified, all strategies are used.",
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
    log_dir = output_dir / "logs"
    logger = setup_logger(log_dir)

    logger.info("===== VideoMarkBench Evaluation =====")
    for key, value in vars(args).items():
        # Format aggregations display
        if key == "aggregations" and value is None:
            logger.info(
                f"{key:<20} : all (using all {len(AGGREGATION_STRATEGIES)} strategies)"
            )
        else:
            logger.info(f"{key:<20} : {value}")
    logger.info("====================================")

    # Setup model
    logger.info(f"Loading {args.model} model...")
    model = setup_model(args.model, device, args)
    model_wrapper = get_model_wrapper(args.model, model)
    logger.info(f"Model loaded on {device}")

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.warning(f"Dataset directory not found: {dataset_path}")
        logger.info("Attempting to download dataset...")
        dataset_path = download_dataset(output_dir=dataset_path)

    dataset = VideoMarkBenchDataset(dataset_path)
    logger.info(f"Loaded dataset with {len(dataset)} videos")
    stats = dataset.get_statistics()
    logger.info(f"Dataset statistics: {stats}")

    # Get perturbations
    perturbations = get_perturbations(args.perturbations)
    logger.info(f"Using {len(perturbations)} perturbations")

    # Get aggregation strategies
    aggregation_strategies = args.aggregations or list(AGGREGATION_STRATEGIES.keys())
    logger.info(f"Using {len(aggregation_strategies)} aggregation strategies")

    # Create evaluator
    evaluator = VideoMarkBenchEvaluator(
        model=model_wrapper,
        dataset=dataset,
        output_dir=output_dir,
        logger=logger,
    )

    # Run evaluation
    logger.info("Starting evaluation...")
    evaluator.evaluate(
        perturbations=perturbations,
        aggregation_strategies=aggregation_strategies,
        max_videos=args.max_videos,
        max_frames_per_video=args.max_frames_per_video,
    )

    logger.info("Evaluation completed!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
