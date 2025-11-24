import os
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from torch.utils.data import Subset
from videoseal.utils.cfg import setup_dataset
from videoseal.evals.full import evaluate
from videoseal.utils import bool_inst
from videoseal.modules.jnd import JND
from adaptive_models.adaptive_videoseal import AdaptiveVideoseal
from scene_detectors.registry import detector_registry
from utils import get_videoseal, select_device, setup_logger, setup_seed
import omegaconf
from utils import patch_ffmpeg_path
from decorators import make_interceptor


def run_experiment(args):
    if not args.is_video:
        raise ValueError(
            "Error: step size evaluation is designed for videos. Setting is_video=True."
        )

    """Main experiment runner with logging, seeding, and reproducibility."""
    log_dir = Path(args.output_dir) / "logs"

    logger = setup_logger(log_dir)

    logger.info("===== Experiment Configuration =====")
    for key, value in vars(args).items():
        logger.info(f"{key:<20} : {value}")
    logger.info("====================================")

    # Setup device
    device = select_device(args.device)

    # Set seed
    setup_seed(args.seed)

    # --- Scene detector selection ---
    try:
        scene_detector_cls = detector_registry[args.scene_detector]
    except KeyError:
        raise ValueError(f"Unknown scene detector: {args.scene_detector}")

    model = get_videoseal()

    adaptive_model = AdaptiveVideoseal.from_videoseal(model)

    # Override model parameters in args
    if args.scaling_w in ["None", "none", -1]:
        # change some of the params to accept None from json
        args.scaling_w = None

    adaptive_model.blender.scaling_w = float(args.scaling_w or model.blender.scaling_w)
    adaptive_model.chunk_size = args.videoseal_chunk_size or model.chunk_size
    adaptive_model.video_mode = args.videoseal_mode or model.mode
    adaptive_model.img_size = args.img_size_proc or model.img_size

    # model.set_logger(logger)
    adaptive_model = adaptive_model.eval().to(device)
    adaptive_model.compile()

    del model  # discard model after setting AdaptiveVideoseal

    # Override attenuation build
    if args.attenuation is not None:
        # should be on CPU to operate on high resolution videos
        if args.attenuation.lower().startswith("jnd"):
            attenuation_cfg = omegaconf.OmegaConf.load(args.attenuation_config)
            attenuation = JND(**attenuation_cfg[args.attenuation])
        else:
            attenuation = None
        adaptive_model.attenuation = attenuation

    args.simple_video_dataset = (
        True  # use simple video dataset for evaluation to speed up
    )
    dataset = setup_dataset(args)
    dataset = Subset(dataset, range(args.num_samples))

    interpolation = {
        "mode": args.interpolation_mode,
        "align_corners": args.interpolation_align_corners,
        "antialias": args.interpolation_antialias,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    # quick fix for hard coded ffmpeg bin file location in videoseal/evals/metrics
    patch_ffmpeg_path()

    from decorators import make_video_mode_recorder

    orig_apply_video_mode = AdaptiveVideoseal._apply_video_mode

    for threshold in tqdm(args.thresholds, desc="Thresholds", leave=False):
        logger.info(f"Evaluating with threshold={threshold}")

        adaptive_model.set_scene_detector(
            scene_detector_cls(threshold=threshold, device=args.device)
        )
        scene_detector_name = adaptive_model.scene_detector.__class__.__name__

        detector_dir = Path(args.output_dir) / scene_detector_name
        threshold_dir = detector_dir / f"threshold_{threshold}"

        all_metrics = []
        for step_size in tqdm(
            args.step_sizes, desc="Step sizes (threshold={threshold})"
        ):
            before_hook, video_mode_records = make_video_mode_recorder()
            AdaptiveVideoseal._apply_video_mode = staticmethod(
                make_interceptor(before=before_hook)(orig_apply_video_mode)
            )

            logger.info(f"Evaluating with step_size={step_size}")
            # Create subdirectory for this step size
            step_output_dir = threshold_dir / f"step_{step_size}"
            step_output_dir.mkdir(parents=True, exist_ok=True)

            # Set step size for the model
            adaptive_model.step_size = step_size

            # Run evaluation
            metrics = evaluate(
                model=adaptive_model,
                dataset=dataset,
                is_video=args.is_video,
                output_dir=step_output_dir,
                save_first=args.save_first,
                num_frames=args.num_frames,
                video_aggregation=args.video_aggregation,
                only_identity=args.only_identity,
                only_combined=args.only_combined,
                bdrate=args.bdrate,
                decoding=args.decoding,
                detection=args.detection,
                interpolation=interpolation,
                lowres_attenuation=args.lowres_attenuation,
                skip_image_metrics=args.skip_image_metrics,
            )

            chunk_records = [
                (record.get("total_frames"), record.get("n_embed"))
                for record in video_mode_records
            ]

            video_idx = 0
            video_frames = 0
            n_embeds = 0
            while chunk_records:
                chunk_frames, n_embed = chunk_records.pop(0)
                video_frames += chunk_frames
                n_embeds += n_embed
                if video_frames == metrics[video_idx].get("t"):
                    embed_ratio = n_embeds / video_frames
                    metrics[video_idx]["embed_ratio"] = embed_ratio
                    metrics[video_idx]["n_embed"] = n_embeds

                    video_idx += 1
                    video_frames = 0
                    n_embeds = 0

            # Add step size to metrics
            for metric in metrics:
                metric["step_size"] = step_size

            all_metrics.extend(metrics)

        # Save combined metricss
        combined_df = pd.DataFrame(all_metrics)
        combined_df.to_csv(
            os.path.join(threshold_dir, "combined_metrics.csv"), index=False
        )

        # TODO: Generate summry by thresholds
        # Generate summary by step size & threshold
        summary_df = combined_df.groupby(["step_size"]).mean()
        summary_df.to_csv(os.path.join(threshold_dir, "summary_by_step_size.csv"))

        # log summary
        logger.info("\n===== Summary by Step Size =====")
        logger.info(summary_df)

        # Plot key metrics vs step size if matplotlib is available
        try:
            import matplotlib.pyplot as plt

            key_metrics = ["psnr", "ssim", "lpips"]
            if "vmaf" in summary_df.columns:
                key_metrics.append("vmaf")

            bit_acc_cols = [col for col in summary_df.columns if "bit_acc" in col]
            key_metrics.extend(bit_acc_cols[:1])  # Add first bit accuracy metric

            plt.figure(figsize=(12, 8))
            for metric in key_metrics:
                if metric in summary_df.columns:
                    plt.plot(summary_df.index, summary_df[metric], "o-", label=metric)

            plt.xlabel("Step Size")
            plt.ylabel("Metric Value")
            plt.title("Metrics vs Step Size")
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(threshold_dir, "metrics_vs_step_size.png")

            plt.savefig(plot_path)
            logger.info(f"Saved plot to {plot_path}")
        except ImportError:
            logger.info("matplotlib not available, skipping plot generation")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a model with different step sizes"
    )

    parser.add_argument(
        "--scene_detector",
        type=str,
        default="content",
        choices=["content", "histogram", "hash", "lpips"],
        help="Select the scene detector type: 'content', 'histogram', 'hash' or 'lpips'.",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[20, 25, 30, 35, 40],
        help="scene_detector thresholds for differentiating the adjacent scenes",
    )

    parser.add_argument(
        "--step_sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128],
        help="Step sizes to evaluate (default: [8, 16, 32, 64, 128])",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection",
    )

    group = parser.add_argument_group("Dataset")
    group.add_argument("--dataset", type=str, help="Name of the dataset.")
    group.add_argument(
        "--is_video",
        type=bool_inst,
        default=True,
        help="Whether the data is video (default: True)",
    )
    group.add_argument(
        "--short_edge_size",
        type=int,
        default=-1,
        help="Resizes the short edge of the image to this size at loading time, and keep the aspect ratio. If -1, no resizing.",
    )
    group.add_argument(
        "--num_frames",
        type=int,
        default=24 * 3,
        help="Number of frames to evaluate for video quality",
    )
    group.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to evaluate"
    )

    group.add_argument(
        "--video_aggregation",
        type=str,
        default="avg",
        help="Aggregation method for detection of video frames",
    )

    group = parser.add_argument_group(
        "Model parameters to override. If not provided, the checkpoint values are used."
    )
    group.add_argument(
        "--attenuation_config",
        type=str,
        default="configs/attenuation.yaml",
        help="Path to the attenuation config file",
    )
    group.add_argument(
        "--attenuation", type=str, default=None, help="Attenuation model to use"
    )
    group.add_argument(
        "--lowres_attenuation",
        type=bool_inst,
        default=False,
        help="Whether to do attenuation at low resolution",
    )
    group.add_argument(
        "--img_size_proc",
        type=int,
        default=None,
        help="Size of the input images for interpolation in the embedder/extractor models",
    )
    group.add_argument(
        "--scaling_w",
        default=None,
        help="Scaling factor for the watermark in the embedder model",
    )
    group.add_argument(
        "--videoseal_chunk_size",
        type=int,
        default=32,
        help="Number of frames to chunk during forward pass",
    )
    group.add_argument(
        "--videoseal_mode",
        type=str,
        default="repeat",
        help="The inference mode for videos",
    )

    group = parser.add_argument_group("Experiment")
    group.add_argument(
        "--output_dir",
        type=str,
        default="output/step_size_eval",
        help="Output directory for logs and images",
    )
    group.add_argument(
        "--save_first", type=int, default=-1, help="Number of images/videos to save"
    )
    parser.add_argument(
        "--only_identity",
        type=bool_inst,
        default=False,
        help="Whether to only evaluate the identity augmentation",
    )
    parser.add_argument(
        "--only_combined",
        type=bool_inst,
        default=False,
        help="Whether to only evaluate combined augmentations",
    )
    parser.add_argument(
        "--bdrate", type=bool_inst, default=False, help="Whether to compute BD-rate"
    )
    parser.add_argument(
        "--decoding",
        type=bool_inst,
        default=True,
        help="Whether to evaluate decoding metrics",
    )
    parser.add_argument(
        "--detection",
        type=bool_inst,
        default=False,
        help="Whether to evaluate detection metrics",
    )
    parser.add_argument(
        "--skip_image_metrics",
        type=bool_inst,
        default=False,
        help="Whether to skip computing image quality metrics",
    )

    group = parser.add_argument_group("Interpolation")
    group.add_argument(
        "--interpolation_mode",
        type=str,
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic", "area"],
        help="Interpolation mode for resizing",
    )
    group.add_argument(
        "--interpolation_align_corners",
        type=bool_inst,
        default=False,
        help="Align corners for interpolation",
    )
    group.add_argument(
        "--interpolation_antialias",
        type=bool_inst,
        default=True,
        help="Use antialiasing for interpolation",
    )

    args = parser.parse_args()

    run_experiment(args)
