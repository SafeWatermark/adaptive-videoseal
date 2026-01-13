"""
Parallel evaluation module for multi-GPU video processing.

Implements a persistent worker pool pattern for efficient multi-GPU evaluation
of watermarking models under common perturbations.
"""

import sys
from pathlib import Path
import torch
import torch.multiprocessing as mp
from multiprocessing import Queue
from typing import List, Dict, Optional, Iterator, Tuple
import pandas as pd
import logging
import json
import tempfile
import shutil
from tqdm import tqdm

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "videoseal"))

from evaluate.common_evaluator import (
    get_common_perturbations,
    AGGREGATION_STRATEGIES,
    _apply_perturbation,
    _get_perturbation_params,
    _get_perturbation_dir_name,
)
from evaluate.aggregation_utils import call_with_watermark_gt_tensor
from evaluate.utils import calculate_statistics


def _worker_setup_model(
    model_name: str,
    device: torch.device,
    args: Dict,
    logger: logging.Logger,
):
    """
    Setup watermarking model in worker process.

    Args:
        model_name: Name of the model ('videoseal' or 'adaptive-videoseal')
        device: Device to run model on
        args: Model configuration arguments
        logger: Logger instance

    Returns:
        VideoMarkBenchModel wrapper instance
    """
    from utils import get_videoseal
    from adaptive_models.adaptive_videoseal import AdaptiveVideoseal
    from evaluate.model_wrapper import VideoMarkBenchModel

    base_model = get_videoseal().eval().to(device)
    base_model.compile()

    # Configuration for base model
    base_model.step_size = args.get("step_size", 8)
    base_model.blender.scaling_w = args.get("scaling_w", 1.5)
    base_model.video_mode = args.get("video_mode", "repeat")
    base_model.chunk_size = args.get("chunk_size", 32)

    if model_name == "adaptive-videoseal":
        from scene_detectors.registry import detector_registry

        adaptive_model = AdaptiveVideoseal.from_videoseal(base_model)

        scene_detector_name = args.get("scene_detector", "content")
        try:
            scene_detector = detector_registry[scene_detector_name](device=str(device))
        except KeyError:
            raise ValueError(f"Unknown scene detector: {scene_detector_name}")

        adaptive_model.set_scene_detector(scene_detector)
        adaptive_model.set_logger(logger)
        adaptive_model = adaptive_model.eval().to(device)
        adaptive_model.compile()
        adaptive_model.chunk_size = args.get("chunk_size", 32)

        return VideoMarkBenchModel(adaptive_model, model_name="adaptive-videoseal")
    else:
        return VideoMarkBenchModel(base_model, model_name="videoseal")


def _worker_process(
    worker_id: int,
    gpu_id: int,
    model_name: str,
    model_args: Dict,
    video_queue: Queue,
    result_queue: Queue,
):
    """
    Worker process entry point.

    Args:
        worker_id: Unique worker identifier
        gpu_id: GPU device ID to use
        model_name: Name of the model
        model_args: Arguments for model setup
        video_queue: Queue to receive videos from
        result_queue: Queue to send results to
    """
    # Setup logging for worker
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Worker {worker_id}] %(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(f"worker_{worker_id}")

    # Set device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Worker {worker_id} starting on {device}")

    # Setup model
    try:
        model = _worker_setup_model(model_name, device, model_args, logger)
        logger.info(f"Worker {worker_id}: Model loaded successfully")
    except Exception as e:
        logger.error(f"Worker {worker_id}: Failed to load model: {e}")
        result_queue.put(("error", worker_id, str(e)))
        return

    # Create temp directory for MPEG4
    temp_dir = Path(tempfile.mkdtemp(prefix=f"worker_{worker_id}_mpeg4_"))

    try:
        # Get perturbations
        perturbations = get_common_perturbations(temp_dir=temp_dir)

        # Process videos from queue
        while True:
            task = video_queue.get()

            # Check for termination signal
            if task is None:
                logger.info(f"Worker {worker_id}: Received shutdown signal")
                break

            video_data, video_info, task_config = task
            video_name = video_info.get("name", "unknown")

            try:
                results = _process_single_video(
                    model=model,
                    video=video_data,
                    video_info=video_info,
                    perturbations=perturbations,
                    config=task_config,
                    device=device,
                    logger=logger,
                )
                result_queue.put(("results", worker_id, results))
                logger.info(f"Worker {worker_id}: Completed {video_name}")

            except Exception as e:
                logger.error(f"Worker {worker_id}: Error processing {video_name}: {e}")
                result_queue.put(("error", worker_id, f"{video_name}: {str(e)}"))

            # Clear GPU cache after each video
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
        logger.info(f"Worker {worker_id}: Shutdown complete")


def _process_single_video(
    model,
    video: torch.Tensor,
    video_info: Dict,
    perturbations: List,
    config: Dict,
    device: torch.device,
    logger: logging.Logger,
) -> List[Dict]:
    """
    Process a single video with all perturbations.

    Args:
        model: Watermarking model
        video: Video tensor
        video_info: Video metadata
        perturbations: List of perturbation modules
        config: Evaluation configuration
        device: Device to use
        logger: Logger instance

    Returns:
        List of result dictionaries
    """
    results = []
    video_name = video_info.get("name", "unknown")

    # Limit frames if specified
    max_frames = config.get("max_frames_per_video")
    if max_frames and video.shape[0] > max_frames:
        video = video[:max_frames]

    # Move video to device
    video = video.to(device)

    # Embed watermark
    with torch.no_grad():
        embed_result = model.embed(video)
        watermarked_video = embed_result["watermarked_video"]
        embedded_message = embed_result["message"]
        embedded_message_long = embedded_message.long()

    # Get forgery video if available
    evaluate_forgery = config.get("evaluate_forgery", True)
    forgery_video = config.get("forgery_video")
    video_name_forgery = config.get("forgery_video_name", video_name)
    precomputed_dir = config.get("precomputed_dir")

    if forgery_video is not None:
        forgery_video = forgery_video.to(device)
        if max_frames and forgery_video.shape[0] > max_frames:
            forgery_video = forgery_video[:max_frames]
    else:
        forgery_video = video

    # Process each perturbation
    for pert in perturbations:
        pert_name = type(pert).__name__
        display_name = pert_name.replace("Wrapper", "")
        pert_params = _get_perturbation_params(pert)

        # ===== REMOVAL ATTACK =====
        with torch.no_grad():
            try:
                perturbed_video = _apply_perturbation(
                    pert, watermarked_video, pert_name
                )
            except Exception as e:
                logger.warning(f"Error applying {display_name} (removal): {e}")
                continue

        with torch.no_grad():
            try:
                logits = model.extract(perturbed_video)
            except Exception as e:
                logger.warning(f"Error extracting from {display_name} (removal): {e}")
                continue

        # Match dimensions
        logits_device = logits.device
        if embedded_message_long.device != logits_device:
            msg_long = embedded_message_long.to(logits_device)
        else:
            msg_long = embedded_message_long

        if logits.shape[1] > msg_long.shape[0]:
            logits_matched = logits[:, : msg_long.shape[0]]
            watermark_gt = msg_long
        elif logits.shape[1] < msg_long.shape[0]:
            logits_matched = logits
            watermark_gt = msg_long[: logits.shape[1]]
        else:
            logits_matched = logits
            watermark_gt = msg_long

        # Evaluate with all aggregation strategies
        for agg_name, agg_func in AGGREGATION_STRATEGIES.items():
            try:
                detected = call_with_watermark_gt_tensor(
                    agg_func, logits_matched, watermark_gt
                )
                results.append(
                    {
                        "video": video_name,
                        "perturbation": display_name,
                        "perturbation_params": str(pert_params),
                        "aggregation": agg_name,
                        "detected": bool(detected),
                        "attack_type": "removal",
                        "type": "common",
                    }
                )
            except Exception as e:
                logger.warning(f"Error in {agg_name} for {display_name} (removal): {e}")

        # ===== FORGERY ATTACK =====
        if evaluate_forgery:
            # Try to load pre-computed perturbation
            perturbed_original = None
            if precomputed_dir is not None:
                pert_dir_name = _get_perturbation_dir_name(pert)
                forgery_video_stem = Path(video_name_forgery).stem
                precomputed_file = (
                    Path(precomputed_dir) / pert_dir_name / f"{forgery_video_stem}.pt"
                )
                if precomputed_file.exists():
                    try:
                        precomputed_data = torch.load(
                            precomputed_file, weights_only=False
                        )
                        perturbed_original = precomputed_data["video"].to(device)
                    except Exception as e:
                        logger.warning(
                            f"Error loading pre-computed {display_name}: {e}"
                        )

            if perturbed_original is None:
                with torch.no_grad():
                    try:
                        perturbed_original = _apply_perturbation(
                            pert, forgery_video, pert_name
                        )
                    except Exception as e:
                        logger.warning(f"Error applying {display_name} (forgery): {e}")
                        continue

            with torch.no_grad():
                try:
                    logits_forgery = model.extract(perturbed_original)
                except Exception as e:
                    logger.warning(
                        f"Error extracting from {display_name} (forgery): {e}"
                    )
                    continue

            # Match dimensions for forgery
            if logits_forgery.shape[1] > msg_long.shape[0]:
                logits_forgery_matched = logits_forgery[:, : msg_long.shape[0]]
                watermark_gt_forgery = msg_long
            elif logits_forgery.shape[1] < msg_long.shape[0]:
                logits_forgery_matched = logits_forgery
                watermark_gt_forgery = msg_long[: logits_forgery.shape[1]]
            else:
                logits_forgery_matched = logits_forgery
                watermark_gt_forgery = msg_long

            for agg_name, agg_func in AGGREGATION_STRATEGIES.items():
                try:
                    detected_forgery = call_with_watermark_gt_tensor(
                        agg_func, logits_forgery_matched, watermark_gt_forgery
                    )
                    results.append(
                        {
                            "video": video_name_forgery,
                            "perturbation": display_name,
                            "perturbation_params": str(pert_params),
                            "aggregation": agg_name,
                            "detected": bool(detected_forgery),
                            "attack_type": "forgery",
                            "type": "common",
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Error in {agg_name} for {display_name} (forgery): {e}"
                    )

            del perturbed_original

        del perturbed_video, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def evaluate_common_perturbations_parallel(
    model_name: str,
    model_args: Dict,
    dataset: Iterator[Tuple[torch.Tensor, Dict]],
    output_dir: Path,
    logger: logging.Logger,
    num_gpus: Optional[int] = None,
    max_videos: Optional[int] = None,
    max_frames_per_video: Optional[int] = None,
    evaluate_forgery: bool = True,
    kinetics_dataset: Optional[Iterator[Tuple[torch.Tensor, Dict]]] = None,
    precomputed_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Parallel evaluation of common perturbations across multiple GPUs.

    Args:
        model_name: Name of model to evaluate ('videoseal' or 'adaptive-videoseal')
        model_args: Arguments for model setup (dict with step_size, scaling_w, etc.)
        dataset: Dataset iterator yielding (video, video_info) tuples
        output_dir: Output directory for results
        logger: Main logger
        num_gpus: Number of GPUs to use (auto-detect if None)
        max_videos: Max videos to evaluate
        max_frames_per_video: Max frames per video
        evaluate_forgery: Whether to evaluate forgery attacks
        kinetics_dataset: Optional kinetics dataset for forgery attacks
        precomputed_dir: Path to pre-computed perturbations

    Returns:
        DataFrame with evaluation results
    """
    # Auto-detect GPUs
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            num_gpus = 1
            logger.warning("No GPUs detected, using single-process mode")

    logger.info(f"Starting parallel evaluation with {num_gpus} workers")

    # Use spawn method for CUDA safety
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    # Create queues
    video_queue = mp.Queue()
    result_queue = mp.Queue()

    # Start worker processes
    workers = []
    for worker_id in range(num_gpus):
        gpu_id = worker_id % max(1, torch.cuda.device_count())

        worker = mp.Process(
            target=_worker_process,
            args=(
                worker_id,
                gpu_id,
                model_name,
                model_args,
                video_queue,
                result_queue,
            ),
        )
        worker.daemon = False
        worker.start()
        workers.append(worker)
        logger.info(f"Started worker {worker_id} on GPU {gpu_id}")

    # Feed videos to queue
    video_count = 0
    kinetics_iter = iter(kinetics_dataset) if kinetics_dataset is not None else None

    for video, video_info in dataset:
        if max_videos and video_count >= max_videos:
            break

        # Get forgery video if available
        forgery_video = None
        forgery_video_name = video_info.get("name", "unknown")

        if evaluate_forgery and kinetics_iter is not None:
            try:
                kinetics_video, kinetics_metadata = next(kinetics_iter)
                forgery_video = kinetics_video
                forgery_video_name = kinetics_metadata.get(
                    "video_name", f"kinetics_{video_count}"
                )
            except StopIteration:
                logger.warning("Kinetics dataset exhausted")
                forgery_video = None

        task_config = {
            "max_frames_per_video": max_frames_per_video,
            "evaluate_forgery": evaluate_forgery,
            "forgery_video": forgery_video,
            "forgery_video_name": forgery_video_name,
            "precomputed_dir": str(precomputed_dir) if precomputed_dir else None,
        }

        video_queue.put((video, video_info, task_config))
        video_count += 1

    logger.info(f"Queued {video_count} videos for processing")

    # Send shutdown signals
    for _ in range(num_gpus):
        video_queue.put(None)

    # Collect results with progress bar
    all_results = []
    results_received = 0
    errors = []

    pbar = tqdm(total=video_count, desc="Collecting results")
    while results_received < video_count:
        try:
            msg_type, worker_id, data = result_queue.get(timeout=600)

            if msg_type == "results":
                all_results.extend(data)
                results_received += 1
                pbar.update(1)
            elif msg_type == "error":
                errors.append(data)
                results_received += 1
                pbar.update(1)

        except Exception as e:
            logger.error(f"Error collecting results: {e}")
            break

    pbar.close()

    # Wait for workers to finish
    for worker in workers:
        worker.join(timeout=30)
        if worker.is_alive():
            logger.warning("Worker did not terminate, killing...")
            worker.terminate()

    if errors:
        logger.warning(f"Encountered {len(errors)} errors during evaluation")
        for err in errors[:5]:  # Log first 5 errors
            logger.warning(f"  - {err}")

    # Save results
    df = pd.DataFrame(all_results)
    output_file = output_dir / "common_perturbations_results.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Saved results to {output_file}")

    # Calculate and save statistics
    if len(df) > 0:
        stats = calculate_statistics(df, logger)
        stats_file = output_dir / "common_perturbations_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_file}")

    return df
