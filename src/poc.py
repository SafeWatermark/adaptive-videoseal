"""
Proof Of Concepts: Adaptive propagation can improve the detection quality.
"""

from tqdm import tqdm
import numpy as np
import ffmpeg
import cv2
import subprocess
from pathlib import Path

import torch

from videoseal.evals.metrics import bit_accuracy
from videoseal.models import Videoseal

from adaptive_models.adaptive_videoseal import AdaptiveVideoseal
from decorators import log_elapsed_time
from utils import (
    setup_seed,
    setup_logger,
    select_device,
    get_videoseal,
    get_stega_stamp_videoseal,
    get_revmark_videoseal,
)
from scene_detectors.registry import detector_registry
from ecc_codecs.registry import ecc_codec_registry
import pandas as pd
from datetime import datetime
import argparse
from scipy import stats
import logging


def get_video_info(input_path):
    # Open the video file
    video = cv2.VideoCapture(input_path)

    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    codec = int(video.get(cv2.CAP_PROP_FOURCC))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Decode codec to human-readable form
    codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

    video.release()  # Close the video file

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "codec": codec_str,
        "num_frames": num_frames,
    }


@log_elapsed_time()
def embed_video_clip(
    model: Videoseal, clip: np.ndarray, msgs: torch.Tensor
) -> np.ndarray:
    clip_tensor = (
        torch.tensor(clip, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    ).to(model.device)
    outputs = model.embed(
        clip_tensor, msgs=msgs, is_video=True, lowres_attenuation=True
    )
    processed_clip = outputs["imgs_w"]
    processed_clip = (processed_clip * 255.0).byte().permute(0, 2, 3, 1).cpu().numpy()
    return processed_clip


@log_elapsed_time()
def embed_video(
    model: Videoseal, input_path: str, output_path: str, chunk_size: int, crf: int = 23
) -> None:
    # Read video dimensions
    video_info = get_video_info(input_path)
    width = int(video_info["width"])
    height = int(video_info["height"])
    fps = float(video_info["fps"])
    num_frames = int(video_info["num_frames"])

    # Open the input video
    process1 = (
        ffmpeg.input(input_path)
        .output(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(width, height),
            r=fps,
        )
        .run_async(pipe_stdout=True, pipe_stderr=subprocess.PIPE)
    )
    # Open the output video
    process2 = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(width, height),
            r=fps,
        )
        .output(output_path, vcodec="libx264", pix_fmt="yuv420p", r=fps, crf=crf)
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=subprocess.PIPE)
    )

    # Create a random message
    msgs = model.get_random_msg()
    with open(output_path.replace(".mp4", ".txt"), "w") as f:
        f.write("".join([str(msg.item()) for msg in msgs[0]]))

    # Process the video
    frame_size = width * height * 3
    chunk = np.zeros((chunk_size, height, width, 3), dtype=np.uint8)
    frame_count = 0
    frames_in_chunk = 0
    pbar = tqdm(total=num_frames, desc="Watermark embedding")
    while True:
        in_bytes = process1.stdout.read(frame_size)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        chunk[frames_in_chunk] = frame
        frames_in_chunk += 1
        frame_count += 1
        pbar.update(1)
        if frames_in_chunk == chunk_size:
            processed_frame = embed_video_clip(model, chunk, msgs)
            process2.stdin.write(processed_frame.tobytes())
            frames_in_chunk = 0

    # Process final partial chunk if any
    if frames_in_chunk > 0:
        processed_frame = embed_video_clip(model, chunk[:frames_in_chunk], msgs)
        process2.stdin.write(processed_frame.tobytes())

    process1.stdout.close()
    process2.stdin.close()
    process1.wait()
    process2.wait()

    return msgs


def detect_video_clip(model: Videoseal, clip: np.ndarray) -> torch.Tensor:
    clip_tensor = torch.tensor(clip, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    outputs = model.detect(clip_tensor, is_video=True)
    output_bits = outputs["preds"][
        :, 1:
    ]  # exclude the first which may be used for detection
    return output_bits


def detect_video(
    model: Videoseal, input_path: str, num_frames_for_extraction: int, chunk_size: int
) -> None:
    # Read video dimensions
    video_info = get_video_info(input_path)
    width = int(video_info["width"])
    height = int(video_info["height"])
    num_frames = int(video_info["num_frames"])

    # Open the input video
    process1 = (
        ffmpeg.input(input_path)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run_async(pipe_stdout=True, pipe_stderr=subprocess.PIPE)
    )

    # Process the video
    frame_size = width * height * 3
    chunk = np.zeros((chunk_size, height, width, 3), dtype=np.uint8)
    frame_count = 0
    soft_msgs = []
    pbar = tqdm(total=num_frames, desc="Watermark extraction")
    while True:
        if frame_count >= num_frames_for_extraction:
            break
        in_bytes = process1.stdout.read(frame_size)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        chunk[frame_count % chunk_size] = frame
        frame_count += 1
        pbar.update(1)
        if frame_count % chunk_size == 0:
            soft_msgs.append(detect_video_clip(model, chunk))
    if not soft_msgs:
        logging.error(
            "No frames processed. "
            f"Ensure num_frames_for_extraction >= chunk_size and the video has enough frames. "
            f"(num_frames_for_extraction={num_frames_for_extraction}, "
            f"chunk_size={chunk_size}, total_video_frames={num_frames}, "
        )
        raise ValueError("No frames processed for watermark extraction.")
    process1.stdout.close()
    process1.wait()

    soft_msgs = torch.cat(soft_msgs, dim=0)
    soft_msgs = soft_msgs.mean(dim=0)  # Average the predictions across all frames
    return soft_msgs


def run_experiment(args):
    """Main experiment runner with logging, seeding, and reproducibility."""
    log_dir = Path(args.output_dir) / "logs"

    logger = setup_logger(log_dir)

    logger.info("===== Experiment Configuration =====")
    for key, value in vars(args).items():
        logger.info(f"{key:<20} : {value}")
    logger.info("====================================")

    # warning inappropriate settings
    if args.num_frames_for_extraction < args.chunk_size:
        logger.warning(
            "Detection will raise the error as extracted number of frames is less than chunk size.\n"
            f"Ensure num_frames_for_extraction >= chunk_size and the video has enough frames.\n"
            f"(num_frames_for_extraction={args.num_frames_for_extraction}, "
            f"chunk_size={args.chunk_size}. "
        )

    # Setup device
    device = select_device(args.device)
    logger.info(f"Using device: {device}")

    # Set seed
    setup_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")

    # Prepare directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Output dir: {output_dir}")

    # Fixed model paths
    if args.model == "videoseal":
        model = get_videoseal()
        model.blender.scaling_w = args.scaling_w
    elif args.model == "stega_stamp":
        if args.onnx_path is None:
            raise ValueError(
                "StegaStamp requires an ONNX model file.\n"
                "Please provide it using: --onnx_path <path>.\n"
                "You can download the file from this repository's GitHub Releases "
                "(original source: https://github.com/umd-huang-lab/WAVES/tree/main/decoders)."
            )

        model = get_stega_stamp_videoseal(args.onnx_path)
        if args.scaling_w != 1.0:
            logger.info(
                f"'scaling_w' was set to {args.scaling_w}, but StegaStamp requires a 1:1 blend. "
                "Overriding and using scaling_w = 1.0."
            )
        logger.info(
            "VideoSeal 1.0 enables attenuation by default, but StegaStamp requires it to be disabled. "
            "Attenuation has been set to None."
        )
    elif args.model == "revmark":
        import os
        from REVMark import revmark_path

        model = get_revmark_videoseal(
            os.path.join(revmark_path, "checkpoints/Encoder.pth"),
            os.path.join(revmark_path, "checkpoints/Decoder.pth"),
        )

    model = model.eval().to(device)
    model.compile()

    model.step_size = args.step_size
    model.video_mode = args.video_mode
    adaptive_model = AdaptiveVideoseal.from_videoseal(model)

    if args.ecc_mode:
        from utils import transform_ecc_videoseal
        from models.stega_stamp import STEGA_STAMP_NUM_BITS
        from models.revmark import REVMARK_DEFAULT_MSG_BITS

        # TODO: Consider how to place the device of ecc codec(espcailly PolarWatermarkCodec as TensorFlow)
        ecc_kwargs = {"message_bits": 64}  # Common default
        if args.model == "videoseal":
            ecc_kwargs["total_bits"] = model.embedder.msg_processor.nbits
            ecc_kwargs["threshold"] = 0.0  # VideoSeal uses logits
            # ecc_kwargs["input_type"] = "logits"
        elif args.model == "stega_stamp":
            ecc_kwargs["total_bits"] = STEGA_STAMP_NUM_BITS
            ecc_kwargs["threshold"] = 0.0  # StegaStamp uses logits
            # ecc_kwargs["input_type"] = "sigmoid"
        elif args.model == "revmark":
            ecc_kwargs["total_bits"] = REVMARK_DEFAULT_MSG_BITS
            ecc_kwargs["threshold"] = 0.5  # RevMark uses sigmoid [0, 1]
            # ecc_kwargs["input_type"] = "sigmoid"

        ecc_codec = ecc_codec_registry.get(
            args.ecc_codec, ecc_codec_registry["default"]
        )(**ecc_kwargs)

        model = transform_ecc_videoseal(model, ecc_codec=ecc_codec, **ecc_kwargs)
        adaptive_model = transform_ecc_videoseal(
            adaptive_model, ecc_codec=ecc_codec, **ecc_kwargs
        )
    # --- Scene detector selection ---
    try:
        scene_detector = detector_registry[args.scene_detector](device=args.device)
    except KeyError:
        raise ValueError(f"Unknown scene detector: {args.scene_detector}")

    adaptive_model.set_scene_detector(scene_detector)
    adaptive_model.set_logger(logger)
    adaptive_model = adaptive_model.eval().to(device)
    adaptive_model.compile()

    # Experiment parameters
    n_trials = args.n_trials
    num_frames_for_extraction = args.num_frames_for_extraction
    chunk_size = args.chunk_size
    model.chunk_size = chunk_size
    adaptive_model.chunk_size = chunk_size

    bit_acc_list, adaptive_bit_acc_list = [], []

    logger.info(f"Starting {n_trials} trials...")

    for i in range(n_trials):
        logger.info(f"\n===== Trial {i + 1}/{n_trials} =====")

        video_path = input_dir / args.video_filename
        output_path = output_dir / f"trial{i + 1}_{args.video_filename}"
        adaptive_output_path = (
            output_dir / f"adaptive_trial{i + 1}_{args.video_filename}"
        )

        logger.info("▶ [Adaptive Model] Embedding started...")
        adaptive_msgs_ori = embed_video(
            adaptive_model, str(video_path), str(adaptive_output_path), chunk_size
        )
        logger.info("✅ [Adaptive Model] Embedding completed.")

        logger.info("▶ [VideoSeal Model] Embedding started...")
        msgs_ori = embed_video(model, str(video_path), str(output_path), chunk_size)
        logger.info("✅ [VideoSeal Model] Embedding completed.")

        logger.info("🔍 [Adaptive Model] Detection started...")
        adaptive_soft_msgs = detect_video(
            adaptive_model,
            str(adaptive_output_path),
            num_frames_for_extraction,
            chunk_size,
        )
        logger.info("✅ [Adaptive Model] Detection completed.")

        logger.info("🔍 [VideoSeal Model] Detection started...")
        soft_msgs = detect_video(
            model, str(output_path), num_frames_for_extraction, chunk_size
        )
        logger.info("✅ [VideoSeal Model] Detection completed.")

        if args.model == "revmark" or args.model == "stega_stamp" or args.ecc_mode:
            adaptive_bit_acc = (
                bit_accuracy(
                    adaptive_soft_msgs, adaptive_msgs_ori, threshold=0.5
                ).item()
                * 100
            )
            bit_acc = bit_accuracy(soft_msgs, msgs_ori, threshold=0.5).item() * 100
        else:
            adaptive_bit_acc = (
                bit_accuracy(adaptive_soft_msgs, adaptive_msgs_ori).item() * 100
            )
            bit_acc = bit_accuracy(soft_msgs, msgs_ori).item() * 100

        logger.info(
            f"bit_acc = {bit_acc:.2f}%, adaptive_bit_acc = {adaptive_bit_acc:.2f}%"
        )

        adaptive_bit_acc_list.append(adaptive_bit_acc)
        bit_acc_list.append(bit_acc)

    results = {
        "trial": np.arange(1, n_trials + 1),
        "bit_acc": bit_acc_list,
        "adaptive_bit_acc": adaptive_bit_acc_list,
    }
    # TODO: [Refactor] seperate full embedding measurement
    if args.compare_full_embed:
        model.step_size = 1
        full_embed_bit_acc_list = []
        for i in range(n_trials):
            logger.info(f"\n===== Trial {i + 1}/{n_trials} =====")

            video_path = input_dir / args.video_filename
            full_output_path = output_dir / f"full_trial{i + 1}_{args.video_filename}"

            logger.info("▶ [VideoSeal(Full Embedding) Model] Embedding started...")
            full_msgs_ori = embed_video(
                model, str(video_path), str(full_output_path), chunk_size
            )
            logger.info("✅ [VideoSeal(Full Embedding) Model] Embedding completed.")

            logger.info("🔍 [VideoSeal(Full Embedding) Model] Detection started...")
            full_soft_msgs = detect_video(
                model, str(full_output_path), num_frames_for_extraction, chunk_size
            )
            logger.info("✅ [VideoSeal(Full Embedding) Model] Detection completed.")

            full_embed_bit_acc = (
                bit_accuracy(full_soft_msgs, full_msgs_ori).item() * 100
            )

            logger.info(f"full embedding bit acc = {full_embed_bit_acc:.2f}%.")

            full_embed_bit_acc_list.append(full_embed_bit_acc)
        results["full_embedding_bit_acc"] = full_embed_bit_acc_list

    # Save results
    results = pd.DataFrame(results)
    results_path = (
        output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    results.to_csv(results_path, index=False)
    logger.info(f"\n✅ Experiment completed! Results saved to {results_path}")

    logger.info("\n===== stats summarization =====")
    logger.info("\n" + str(results.describe()))

    t_stat, p_val = stats.ttest_rel(results["adaptive_bit_acc"], results["bit_acc"])
    logger.info(f"\nPaired t-test result: t = {t_stat:.3f}, p = {p_val:.5f}")

    if p_val < 0.05:
        logger.info("→ There is meaningful difference (p < 0.05)")
    else:
        logger.info("→ There isn't meaningful difference (p >= 0.05)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated VideoSeal experiment setup")

    parser.add_argument(
        "--scene_detector",
        type=str,
        default="content",
        choices=["content", "histogram", "hash", "lpips"],
        help="Select the scene detector type: 'content', 'histogram', 'hash' or 'lpips'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="videoseal",
        choices=["videoseal", "stega_stamp", "revmark"],
        help="Watermarking model to use ('videoseal' or 'stega_stamp').",
    )
    parser.add_argument(
        "--ecc_mode",
        action="store_true",
        default=False,
        help="If set, use error correction code(ecc) for embedding and detecting",
    )
    parser.add_argument(
        "--ecc_codec",
        type=str,
        default="polar",
        choices=["polar", "rs"],
        help="If ecc_mode is set, ecc_codec needs. default: polar codec",
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default=None,
        help="onnx path for stega_stamp",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--input_dir", type=str, default="./data", help="Input directory for videos"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--video_filename",
        type=str,
        default="sample_movie.mp4",
        help="Name of input video file",
    )
    parser.add_argument(
        "--n_trials", type=int, default=30, help="Number of experiment repetitions"
    )
    parser.add_argument(
        "--num_frames_for_extraction",
        type=int,
        default=32,
        help="Number of frames for extraction",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=32,
        help="Number of video frames to process in a single batch for watermark embedding",
    )
    parser.add_argument(
        "--step_size", type=int, default=8, help="Step size parameter for embedding"
    )
    parser.add_argument(
        "--scaling_w",
        type=float,
        default=1.5,
        help="Scaling weight for watermark blending",
    )
    parser.add_argument(
        "--compare_full_embed",
        action="store_true",
        default=False,
        help="If set, use model.step_size = 1 (embed all frames instead of skipping).",
    )
    parser.add_argument(
        "--video_mode",
        type=str,
        default="repeat",
        choices=["repeat", "alternate", "interpolate"],
        help=(
            "Video mode for expanding key-frame predictions across all frames. "
            "'repeat' — copy each key-frame prediction to its repeated frames; "
            "'alternate' — apply predictions only on key frames; "
            "'interpolate' — interpolate predictions between key frames."
        ),
    )

    args = parser.parse_args()

    run_experiment(args)
