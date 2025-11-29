"""
Evaluation engine for VideoMarkBench.

Orchestrates the complete evaluation pipeline:
embed → perturb → extract → aggregate → evaluate
"""

import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import pandas as pd
import logging

from .models import WatermarkingModel
from .aggregations import aggregate, AGGREGATION_STRATEGIES
from .dataset import VideoMarkBenchDataset


class VideoMarkBenchEvaluator:
    """
    Main evaluation engine for VideoMarkBench.

    Evaluates watermarking models under various perturbations
    and aggregation strategies.
    """

    def __init__(
        self,
        model: WatermarkingModel,
        dataset: VideoMarkBenchDataset,
        output_dir: Path,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize evaluator.

        Args:
            model: Watermarking model wrapper
            dataset: VideoMarkBench dataset
            output_dir: Directory to save results
            logger: Optional logger
        """
        self.model = model
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)

    def evaluate(
        self,
        perturbations: List,
        aggregation_strategies: List[str] = None,
        max_videos: Optional[int] = None,
        max_frames_per_video: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run complete evaluation pipeline.

        Args:
            perturbations: List of perturbation instances (nn.Module)
            aggregation_strategies: List of aggregation strategy names.
                                  If None, uses all available strategies.
            max_videos: Maximum number of videos to evaluate (None = all)
            max_frames_per_video: Maximum frames per video (None = all)

        Returns:
            DataFrame with evaluation results
        """
        if aggregation_strategies is None:
            aggregation_strategies = list(AGGREGATION_STRATEGIES.keys())

        results = []

        # Select videos to evaluate
        num_videos = min(len(self.dataset), max_videos or len(self.dataset))
        video_indices = list(range(num_videos))

        self.logger.info(
            f"Evaluating {num_videos} videos with {len(perturbations)} perturbations"
        )

        # Outer loop: videos with nested progress bar
        video_pbar = tqdm(video_indices, desc="Videos", position=0)
        for video_idx in video_pbar:
            try:
                video, video_info = self.dataset[video_idx]

                # Limit frames if specified
                if max_frames_per_video is not None:
                    video = video[:max_frames_per_video]

                # Display video information at the start
                self.logger.info(
                    f"Processing video {video_idx + 1}/{num_videos}: "
                    f"name={video_info.name}, "
                    f"style={video_info.style}, "
                    f"model={video_info.model}, "
                    f"frames={video.shape[0]}"
                )

                # Update outer progress bar with video name
                video_pbar.set_description(f"Video: {video_info.name}")

                # Embed watermark
                self.logger.debug(f"Embedding watermark in video {video_idx}")
                embed_result = self.model.embed(video)
                watermarked_video = embed_result["watermarked_video"]
                original_message = embed_result["message"]

                # Convert message to numpy for aggregation
                watermark_gt = (original_message > 0.5).float().cpu().numpy()
                # Convert to [0, 1] pattern
                watermark_gt = watermark_gt.astype(np.int32)

                # Inner loop: perturbations with nested progress bar
                pert_pbar = tqdm(
                    perturbations, desc="Perturbations", position=1, leave=False
                )
                for pert in pert_pbar:
                    pert_name = pert.__class__.__name__

                    # Update inner progress bar with perturbation name
                    pert_pbar.set_description(f"Perturbation: {pert_name}")

                    self.logger.debug(f"Applying perturbation: {pert_name}")

                    # Apply perturbation
                    perturbed_video = watermarked_video.clone()

                    # Frame-by-frame perturbations (JPEG, Gaussian, GaussianBlur)
                    if pert_name in ["JPEG", "Gaussian", "GaussianBlur"]:
                        # Apply to each frame
                        for i in range(perturbed_video.shape[0]):
                            frame = perturbed_video[i : i + 1]  # (1, C, H, W)
                            perturbed_frame = pert(frame)
                            if perturbed_frame.dim() == 4:
                                perturbed_video[i] = perturbed_frame[0]
                            else:
                                perturbed_video[i] = perturbed_frame
                    else:
                        # Video-level perturbations (Crop, FrameAverage, FrameRemove, FrameSwitch, MPEG4)
                        perturbed_video = pert(perturbed_video)

                    # Extract logits
                    frame_logits = self.model.extract(perturbed_video)  # (T, K)

                    # Evaluate with each aggregation strategy
                    for agg_strategy in aggregation_strategies:
                        # Aggregate logits (returns boolean: detection success)
                        detection_success = aggregate(
                            frame_logits,
                            strategy=agg_strategy,
                            watermark_gt=watermark_gt,
                        )

                        # Store result
                        results.append(
                            {
                                "video_idx": video_idx,
                                "video_name": video_info.name,
                                "video_style": video_info.style,
                                "video_model": video_info.model,
                                "video_transition": video_info.transition,
                                "perturbation": pert_name,
                                "aggregation": agg_strategy,
                                "detection_success": int(detection_success),
                                "num_frames": video.shape[0],
                            }
                        )

                # Close inner progress bar after all perturbations
                pert_pbar.close()

            except Exception as e:
                self.logger.error(f"Error evaluating video {video_idx}: {e}")
                import traceback

                self.logger.error(traceback.format_exc())
                continue

        # Close outer progress bar
        video_pbar.close()

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Save results
        results_path = self.output_dir / "results.csv"
        results_df.to_csv(results_path, index=False)
        self.logger.info(f"Results saved to {results_path}")

        # Generate summary statistics
        self._generate_summary(results_df)

        return results_df

    def _generate_summary(self, results_df: pd.DataFrame):
        """Generate and save summary statistics."""
        if len(results_df) == 0:
            self.logger.warning(
                "No results to summarize. All evaluations may have failed."
            )
            return

        summary = {}

        # Overall statistics
        if "detection_success" in results_df.columns:
            success_rate = results_df["detection_success"].mean()
            summary["overall"] = {
                "detection_success_rate": success_rate,
                "total_evaluations": len(results_df),
                "successful_detections": int(results_df["detection_success"].sum()),
                "failed_detections": int((1 - results_df["detection_success"]).sum()),
            }
        else:
            self.logger.warning("'detection_success' column not found in results.")
            summary["overall"] = {"note": "No detection_success data available"}

        # By perturbation
        if "detection_success" in results_df.columns:
            summary["by_perturbation"] = (
                results_df.groupby("perturbation")["detection_success"]
                .agg(["mean", "count", "sum"])
                .rename(columns={"mean": "success_rate", "sum": "success_count"})
                .to_dict("index")
            )

            # By aggregation
            summary["by_aggregation"] = (
                results_df.groupby("aggregation")["detection_success"]
                .agg(["mean", "count", "sum"])
                .rename(columns={"mean": "success_rate", "sum": "success_count"})
                .to_dict("index")
            )

            # By video style
            summary["by_style"] = (
                results_df.groupby("video_style")["detection_success"]
                .agg(["mean", "count", "sum"])
                .rename(columns={"mean": "success_rate", "sum": "success_count"})
                .to_dict("index")
            )
        else:
            summary["by_perturbation"] = {}
            summary["by_aggregation"] = {}
            summary["by_style"] = {}

        # Save summary
        summary_path = self.output_dir / "summary.txt"
        with open(summary_path, "w") as f:
            f.write("VideoMarkBench Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write("Overall Statistics:\n")
            for key, value in summary["overall"].items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

            f.write("By Perturbation:\n")
            for pert, stats in summary["by_perturbation"].items():
                f.write(f"  {pert}:\n")
                for stat, value in stats.items():
                    if isinstance(value, float):
                        f.write(f"    {stat}: {value:.4f}\n")
                    else:
                        f.write(f"    {stat}: {value}\n")
            f.write("\n")

            f.write("By Aggregation Strategy:\n")
            for agg, stats in summary["by_aggregation"].items():
                f.write(f"  {agg}:\n")
                for stat, value in stats.items():
                    if isinstance(value, float):
                        f.write(f"    {stat}: {value:.4f}\n")
                    else:
                        f.write(f"    {stat}: {value}\n")
            f.write("\n")

            f.write("By Video Style:\n")
            for style, stats in summary["by_style"].items():
                f.write(f"  {style}:\n")
                for stat, value in stats.items():
                    if isinstance(value, float):
                        f.write(f"    {stat}: {value:.4f}\n")
                    else:
                        f.write(f"    {stat}: {value}\n")

        self.logger.info(f"Summary saved to {summary_path}")

        # Also save as JSON for programmatic access
        import json

        summary_json_path = self.output_dir / "summary.json"
        # Convert numpy types to native Python types for JSON
        summary_json = self._convert_to_json_serializable(summary)
        with open(summary_json_path, "w") as f:
            json.dump(summary_json, f, indent=2)

    @staticmethod
    def _convert_to_json_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {
                k: VideoMarkBenchEvaluator._convert_to_json_serializable(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [
                VideoMarkBenchEvaluator._convert_to_json_serializable(v) for v in obj
            ]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
