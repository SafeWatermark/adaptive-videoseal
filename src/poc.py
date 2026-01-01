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
from utils import (
    transform_ecc_videoseal,
    transform_authenticated_ecc_videoseal,
    transform_authenticated_videoseal,
)
from models.stega_stamp import STEGA_STAMP_NUM_BITS
from models.revmark import REVMARK_DEFAULT_MSG_BITS
from schemas import create_meta_for_model
from mac.registry import mac_provider_registry
from mac.configs import mac_settings, DetectionProbabilityScaling
from videoseal.models.extractor import Extractor


def get_model_config(model_name: str, model: Videoseal) -> dict:
    """Get model-specific configuration using schema utilities.

    Args:
        model_name: Name of the model ("videoseal", "stega_stamp", "revmark")
        model: The model instance to get nbits from (for videoseal)

    Returns:
        dict with 'total_bits', 'threshold', and 'meta' keys
    """
    if model_name == "videoseal":
        total_bits = model.embedder.msg_processor.nbits
    elif model_name == "stega_stamp":
        total_bits = STEGA_STAMP_NUM_BITS
    elif model_name == "revmark":
        total_bits = REVMARK_DEFAULT_MSG_BITS
    else:
        raise ValueError(f"Unknown model: {model_name}")

    meta = create_meta_for_model(model_name, nbits=total_bits)
    return {
        "total_bits": total_bits,
        "threshold": meta.threshold,
        "meta": meta,
    }


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
) -> torch.Tensor:
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


class ExperimentConfig:
    """Container for experiment configuration with validation."""

    def __init__(self, args):
        self.args = args
        self.input_dir = Path(args.input_dir)
        self.output_dir = Path(args.output_dir)
        self.log_dir = self.output_dir / "logs"

    def validate(self, logger) -> None:
        """Validate experiment configuration."""
        if self.args.num_frames_for_extraction < self.args.chunk_size:
            logger.warning(
                "Detection will raise an error as extracted frames < chunk size.\n"
                f"Ensure num_frames_for_extraction >= chunk_size.\n"
                f"(num_frames_for_extraction={self.args.num_frames_for_extraction}, "
                f"chunk_size={self.args.chunk_size})"
            )

    def log_config(self, logger) -> None:
        """Log configuration details."""
        logger.info("===== Experiment Configuration =====")
        for key, value in vars(self.args).items():
            logger.info(f"{key:<20} : {value}")
        logger.info("====================================")


class ModelFactory:
    """Factory for creating and configuring watermark models."""

    @staticmethod
    def create_model(args, device: str | torch.device, logger) -> Videoseal:
        """Create and configure the base model."""
        if args.model == "videoseal":
            model = get_videoseal()
            model.blender.scaling_w = args.scaling_w

        elif args.model == "stega_stamp":
            if args.onnx_path is None:
                raise ValueError(
                    "StegaStamp requires an ONNX model file.\n"
                    "Please provide it using: --onnx_path <path>.\n"
                    "Download from GitHub Releases."
                )
            model = get_stega_stamp_videoseal(args.onnx_path)
            ModelFactory._info_message_for_third_party_model(args, logger)

        elif args.model == "revmark":
            import os
            from REVMark import revmark_path

            model = get_revmark_videoseal(
                os.path.join(revmark_path, "checkpoints/Encoder.pth"),
                os.path.join(revmark_path, "checkpoints/Decoder.pth"),
            )
            ModelFactory._info_message_for_third_party_model(args, logger)
        else:
            raise ValueError(f"Unknown model: {args.model}")

        model = model.eval().to(device)
        model.compile()
        model.step_size = args.step_size
        model.video_mode = args.video_mode
        model.chunk_size = args.chunk_size

        return model

    @staticmethod
    def _info_message_for_third_party_model(args, logger) -> None:
        from models.stega_stamp import STEGA_STAMP_SCALING_W, STEGA_STAMP_ATTENUATION
        from models.revmark import REVMARK_SCALING_W, REVMARK_ATTENUATION

        if args.model == "stega_stamp":
            if args.scaling_w != STEGA_STAMP_SCALING_W:
                logger.info(
                    f"'scaling_w' was {args.scaling_w}, but StegaStamp requires {STEGA_STAMP_SCALING_W}. "
                    f"Overriding to scaling_w = {STEGA_STAMP_SCALING_W}"
                )
            if STEGA_STAMP_ATTENUATION is None:
                logger.info(
                    "StegaStamp requires attenuation disabled. Setting to None."
                )
        elif args.model == "revmark":
            if args.scaling_w != REVMARK_SCALING_W:
                logger.info(
                    f"'scaling_w' was {args.scaling_w}, but REVMark requires {REVMARK_SCALING_W}. "
                    f"Overriding to scaling_w = {REVMARK_SCALING_W}"
                )
            if REVMARK_ATTENUATION is None:
                logger.info("REVMark requires attenuation disabled. Setting to None.")


class CodecManager:
    """Manages ECC and MAC codec configuration and application."""

    def __init__(self, args, total_bits: int, threshold: float):
        self.args = args
        self.total_bits = total_bits
        self.threshold = threshold
        self.mac_provider = None
        self.ecc_codec = None

    def apply_codecs(
        self, model: Videoseal, adaptive_model: AdaptiveVideoseal
    ) -> tuple[AdaptiveVideoseal, Videoseal]:
        """Apply ECC and/or MAC transformations to models."""
        if self.args.mac_mode:
            return self._apply_mac_transforms(model, adaptive_model)
        elif self.args.ecc_mode:
            return self._apply_ecc_transforms(model, adaptive_model)
        return model, adaptive_model

    def _apply_mac_transforms(
        self, model: Videoseal, adaptive_model: AdaptiveVideoseal
    ) -> tuple[AdaptiveVideoseal, Videoseal]:
        """Apply MAC authentication to models."""
        if self.args.model == "videoseal":
            scaling_model, scaling_mac = 0.5, 0.5
        elif self.args.model == "stega_stamp" or self.args.model == "revmark":
            scaling_model, scaling_mac = 0.0, 1.0
        mac_kwargs = {
            "message_bits": self.args.message_bits,
            "tag_bits": self.args.mac_tag_bits,
            "config": mac_settings,
            "detection_scaling": DetectionProbabilityScaling(
                scaling_model, scaling_mac
            ),
        }

        if self.args.ecc_mode:
            return self._apply_mac_with_ecc(model, adaptive_model, mac_kwargs)
        else:
            return self._apply_mac_only(model, adaptive_model, mac_kwargs)

    def _apply_mac_with_ecc(
        self, model: Videoseal, adaptive_model: AdaptiveVideoseal, mac_kwargs: dict
    ) -> tuple[Videoseal, AdaptiveVideoseal]:
        """Apply MAC with ECC encoding."""
        self.mac_provider = mac_provider_registry.get(
            self.args.mac_algorithm, mac_provider_registry["default"]
        )(**mac_kwargs)

        ecc_kwargs = {
            "message_bits": mac_kwargs["message_bits"] + mac_kwargs["tag_bits"],
            "total_bits": self.total_bits,
            "threshold": self.threshold,
        }

        self.ecc_codec = ecc_codec_registry.get(
            self.args.ecc_codec, ecc_codec_registry["default"]
        )(**ecc_kwargs)

        model = transform_authenticated_ecc_videoseal(
            model, self.ecc_codec, self.mac_provider
        )
        adaptive_model = transform_authenticated_ecc_videoseal(
            adaptive_model, self.ecc_codec, self.mac_provider
        )

        return model, adaptive_model

    def _apply_mac_only(
        self, model: Videoseal, adaptive_model: AdaptiveVideoseal, mac_kwargs: dict
    ) -> tuple[Videoseal, AdaptiveVideoseal]:
        """Apply MAC without ECC."""
        mac_kwargs["tag_bits"] = self.total_bits - self.args.message_bits

        self.mac_provider = mac_provider_registry.get(
            self.args.mac_algorithm, mac_provider_registry["default"]
        )(**mac_kwargs)

        model = transform_authenticated_videoseal(model, self.mac_provider)
        adaptive_model = transform_authenticated_videoseal(
            adaptive_model, self.mac_provider
        )

        return model, adaptive_model

    def _apply_ecc_transforms(
        self, model: Videoseal, adaptive_model: AdaptiveVideoseal
    ) -> tuple[Videoseal, AdaptiveVideoseal]:
        """Apply ECC encoding to models."""
        ecc_kwargs = {
            "message_bits": self.args.message_bits,
            "total_bits": self.total_bits,
            "threshold": self.threshold,
        }

        self.ecc_codec = ecc_codec_registry.get(
            self.args.ecc_codec, ecc_codec_registry["default"]
        )(**ecc_kwargs)

        model = transform_ecc_videoseal(model, ecc_codec=self.ecc_codec)
        adaptive_model = transform_ecc_videoseal(
            adaptive_model, ecc_codec=self.ecc_codec
        )

        return model, adaptive_model


class TrialRunner:
    """Manages individual trial execution."""

    def __init__(
        self,
        model: Videoseal,
        adaptive_model: AdaptiveVideoseal,
        codec_manager: CodecManager,
        config,
        logger,
    ):
        self.model = model
        self.adaptive_model = adaptive_model
        self.codec_manager = codec_manager
        self.config = config
        self.logger = logger

    def run_trial(self, trial_num: int) -> tuple[float, float]:
        """Run a single trial and return bit accuracies."""
        self.logger.info(f"\n===== Trial {trial_num}/{self.config.args.n_trials} =====")

        video_path = self.config.input_dir / self.config.args.video_filename
        output_path = (
            self.config.output_dir
            / f"trial{trial_num}_{self.config.args.video_filename}"
        )
        adaptive_output_path = (
            self.config.output_dir
            / f"adaptive_trial{trial_num}_{self.config.args.video_filename}"
        )

        # Embed and detect
        adaptive_msgs_ori, adaptive_soft_msgs = self._process_video(
            self.adaptive_model, video_path, adaptive_output_path, "Adaptive Model"
        )
        msgs_ori, soft_msgs = self._process_video(
            self.model, video_path, output_path, "VideoSeal Model"
        )

        # Decode and verify
        adaptive_msgs_ori = self._decode_messages(
            adaptive_msgs_ori, self.adaptive_model.detector, "Adaptive"
        )
        msgs_ori = self._decode_messages(msgs_ori, self.model.detector, "VideoSeal")

        # Calculate accuracies
        threshold = self._get_bit_accuracy_threshold()
        adaptive_bit_acc = (
            bit_accuracy(
                adaptive_soft_msgs, adaptive_msgs_ori, threshold=threshold
            ).item()
            * 100
        )
        bit_acc = bit_accuracy(soft_msgs, msgs_ori, threshold=threshold).item() * 100

        self.logger.info(
            f"bit_acc = {bit_acc:.2f}%, adaptive_bit_acc = {adaptive_bit_acc:.2f}%"
        )

        return bit_acc, adaptive_bit_acc

    def _process_video(
        self,
        model: Videoseal | AdaptiveVideoseal,
        video_path,
        output_path,
        model_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed and detect watermark in video."""
        self.logger.info(f"▶ [{model_name}] Embedding started...")
        msgs_ori = embed_video(
            model, str(video_path), str(output_path), self.config.args.chunk_size
        )
        self.logger.info(f"✅ [{model_name}] Embedding completed.")

        self.logger.info(f"🔍 [{model_name}] Detection started...")
        soft_msgs = detect_video(
            model,
            str(output_path),
            self.config.args.num_frames_for_extraction,
            self.config.args.chunk_size,
        )
        self.logger.info(f"✅ [{model_name}] Detection completed.")

        return msgs_ori, soft_msgs

    def _decode_messages(
        self, msgs_ori: torch.Tensor, detector: Extractor, model_name: str
    ) -> torch.Tensor:
        """Decode original messages through ECC and/or MAC layers."""
        if self.config.args.ecc_mode:
            if msgs_ori.ndim == 1:
                msgs_ori = msgs_ori.unsqueeze(0)
            decoded, _ = self.codec_manager.ecc_codec.decode(msgs_ori)
            msgs_ori = decoded.squeeze()

        if self.config.args.mac_mode and self.codec_manager.mac_provider is not None:
            verification_results = detector.mac_verification_results
            true_positive_ratio = sum(verification_results) / len(verification_results)
            self.logger.info(
                f"[Authentication - {model_name}] False Negative ratio: "
                f"{1 - true_positive_ratio:.4f}"
            )

            if msgs_ori.ndim == 1:
                msgs_ori = msgs_ori.unsqueeze(0)
            msgs_ori, _ = self.codec_manager.mac_provider.split_payload_bits(msgs_ori)
            msgs_ori = msgs_ori.squeeze(0)
            detector.mac_verification_results.clear()

        return msgs_ori

    def _get_bit_accuracy_threshold(self) -> float:
        """Get appropriate threshold for bit accuracy calculation."""
        # if ecc_mode or mac_mode, return value changed as a binary message by hooks
        if self.config.args.ecc_mode or self.config.args.mac_mode:
            return 0.5
        return get_model_config(self.config.args.model, self.model)["threshold"]

    def run_full_embedding_trial(self, trial_num: int) -> float:
        """Run full embedding comparison trial."""
        self.logger.info(
            f"\n===== Full Embedding Trial {trial_num}/{self.config.args.n_trials} ====="
        )

        video_path = self.config.input_dir / self.config.args.video_filename
        output_path = (
            self.config.output_dir
            / f"full_trial{trial_num}_{self.config.args.video_filename}"
        )

        # Temporarily set step_size to 1 for full embedding
        original_step_size = self.model.step_size
        self.model.step_size = 1

        msgs_ori, soft_msgs = self._process_video(
            self.model, video_path, output_path, "VideoSeal(Full Embedding) Model"
        )

        # Restore original step_size
        self.model.step_size = original_step_size

        bit_acc = bit_accuracy(soft_msgs, msgs_ori).item() * 100
        self.logger.info(f"full embedding bit acc = {bit_acc:.2f}%")

        return bit_acc


class ResultsManager:
    """Handles results storage and statistical analysis."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def save_and_analyze(self, results: dict[str, list[float]]) -> pd.DataFrame:
        """Save results and perform statistical analysis."""
        df = pd.DataFrame(results)

        # Save to CSV
        results_path = (
            self.config.output_dir
            / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        df.to_csv(results_path, index=False)
        self.logger.info(f"\n✅ Experiment completed! Results saved to {results_path}")

        # Log statistics
        self.logger.info("\n===== Stats Summarization =====")
        self.logger.info("\n" + str(df.describe()))

        # Perform paired t-test
        self._perform_ttest(df)

        return df

    def _perform_ttest(self, df: pd.DataFrame) -> None:
        """Perform paired t-test on results."""
        if "adaptive_bit_acc" in df.columns and "bit_acc" in df.columns:
            t_stat, p_val = stats.ttest_rel(df["adaptive_bit_acc"], df["bit_acc"])
            self.logger.info(
                f"\nPaired t-test result: t = {t_stat:.3f}, p = {p_val:.5f}"
            )

            if p_val < 0.05:
                self.logger.info("→ Significant difference detected (p < 0.05)")
            else:
                self.logger.info("→ No significant difference (p >= 0.05)")


def run_experiment(args):
    """Main experiment runner with logging, seeding, and reproducibility."""
    # Initialize configuration
    config = ExperimentConfig(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logger(config.log_dir)
    config.log_config(logger)
    config.validate(logger)

    # Setup device and seed
    device = select_device(args.device)
    logger.info(f"Using device: {device}")
    setup_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")
    logger.info(f"Input dir: {config.input_dir}")
    logger.info(f"Output dir: {config.output_dir}")

    # Create models
    model = ModelFactory.create_model(args, device, logger)
    adaptive_model = AdaptiveVideoseal.from_videoseal(model)

    # Setup scene detector
    try:
        scene_detector = detector_registry[args.scene_detector](device=args.device)
    except KeyError:
        raise ValueError(f"Unknown scene detector: {args.scene_detector}")

    adaptive_model.set_scene_detector(scene_detector)
    adaptive_model.set_logger(logger)
    adaptive_model = adaptive_model.eval().to(device)
    adaptive_model.compile()
    adaptive_model.chunk_size = args.chunk_size

    # Apply codecs
    model_config = get_model_config(args.model, model)
    codec_manager = CodecManager(
        args, model_config["total_bits"], model_config["threshold"]
    )
    model, adaptive_model = codec_manager.apply_codecs(model, adaptive_model)

    # Run trials
    trial_runner = TrialRunner(model, adaptive_model, codec_manager, config, logger)
    logger.info(f"Starting {args.n_trials} trials...")

    bit_acc_list = []
    adaptive_bit_acc_list = []

    for i in range(1, args.n_trials + 1):
        bit_acc, adaptive_bit_acc = trial_runner.run_trial(i)
        bit_acc_list.append(bit_acc)
        adaptive_bit_acc_list.append(adaptive_bit_acc)

    # Prepare results
    results = {
        "trial": np.arange(1, args.n_trials + 1),
        "bit_acc": bit_acc_list,
        "adaptive_bit_acc": adaptive_bit_acc_list,
    }

    # Run full embedding comparison if requested
    if args.compare_full_embed:
        full_embed_bit_acc_list = []
        for i in range(1, args.n_trials + 1):
            bit_acc = trial_runner.run_full_embedding_trial(i)
            full_embed_bit_acc_list.append(bit_acc)
        results["full_embedding_bit_acc"] = full_embed_bit_acc_list

    # Save and analyze results
    results_manager = ResultsManager(config, logger)
    results_manager.save_and_analyze(results)


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
        choices=["polar", "rs", "conv"],
        help="If ecc_mode is set, ecc_codec needs. default: polar codec",
    )
    parser.add_argument(
        "--mac_mode",
        action="store_true",
        help="Enable MAC authentication (requires MAC_SECRET_KEY env var)",
    )
    parser.add_argument(
        "--mac_algorithm",
        type=str,
        default="hmac",
        choices=["hmac"],
        help="MAC algorithm to use",
    )
    parser.add_argument(
        "--mac_tag_bits",
        type=int,
        default=32,
        help="MAC tag size in bits (default: 32, forgery prob ~2^-32)",
    )
    parser.add_argument(
        "--message_bits",
        type=int,
        default=16,
        help="Number of message bits (before MAC/ECC)",
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
