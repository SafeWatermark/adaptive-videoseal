from tqdm import tqdm
import numpy as np
import ffmpeg
import os
import cv2
import subprocess
import shutil
from pathlib import Path

import torch
from torch.nn import functional as F

from videoseal.evals.metrics import bit_accuracy
from videoseal.models import Videoseal
from videoseal.utils.cfg import setup_model_from_model_card

from decorators import log_elapsed_time
from scene_detectors.base import SceneDetector
from scene_detectors.content_detector import ContentDetector
from scene_detectors.hash_detector import HashDetector
from scene_detectors.histogram_detector import HistogramDetector
from utils import setup_seed, setup_logger, select_device
import pandas as pd
from datetime import datetime
import argparse
from scipy import stats
import logging


class _TempWorkingDir:
    def __init__(self, path: Path):
        self._path = Path(path)
        self._prev = None

    def __enter__(self):
        self._prev = Path.cwd()
        os.chdir(self._path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._prev is not None:
            os.chdir(self._prev)


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


class AdaptiveVideoseal(Videoseal):
    def set_scene_detector(self, scene_detector: SceneDetector):
        self.scene_detector = scene_detector

    def set_logger(self, logger):
        self.logger = logger

    @staticmethod
    def _apply_video_mode(
        preds_w: torch.Tensor,
        total_frames: int,
        step_size: torch.Tensor,
        video_mode: str,
    ) -> torch.Tensor:
        """
        applies the selected video mode to expand predictions across frames
        args:
            preds_w (torch.Tensor): predictions for key frames [n, c, h, w]
            total_frames (int): total number of frames to generate
            step_size (torch.Tensor): number of frames between key frames
            video_mode (str): the video mode to use. can be one of "repeat", "alternate", "interpolate"
        returns:
            torch.Tensor: expanded predictions [total_frames, c, h, w]
        """
        # repeat each prediction for step_size frames
        if video_mode == "repeat":
            preds_w = torch.repeat_interleave(preds_w, step_size, dim=0)
        elif video_mode == "alternate":
            full_size = (total_frames,) + preds_w.shape[1:]  # f c h w
            full_preds = torch.zeros(full_size, device=preds_w.device)  # f c h w
            key_frame_ids = torch.cumsum(
                torch.cat(
                    [torch.tensor([0], device=step_size.device), step_size], dim=0
                ),
                dim=0,
            )[:-1]
            full_preds[key_frame_ids] = preds_w  # place preds_w [n c h w] in key frames
            preds_w = full_preds  # f c h w
        # TODO: Vectorize for speed
        elif video_mode == "interpolate":
            # interpolate between predictions
            full_size = (total_frames,) + preds_w.shape[1:]  # f c h w
            full_preds = torch.zeros(full_size, device=preds_w.device)  # f c h w
            # interpolation factors
            alphas = []
            for i, interval in enumerate(step_size[:-1]):
                if i < step_size[:-1].size(0) - 1:
                    alpha = 1 - torch.linspace(
                        0, 1, steps=interval + 1, device=preds_w.device
                    )
                    alphas.append(alpha[:-1])
                else:
                    alpha = 1 - torch.linspace(
                        0, 1, steps=interval, device=preds_w.device
                    )
                    alphas.append(alpha)
            if len(alphas) == 0:
                # if there is only 1 key frame, interpolation is inappropriate. Change to "repeat" mode
                preds_w = torch.repeat_interleave(preds_w, step_size, dim=0)
                return preds_w
            alphas = torch.cat(alphas, dim=0).view(-1, 1, 1, 1)
            # key frames and shifted key frames
            start_frames = torch.repeat_interleave(preds_w[:-1], step_size[:-1], dim=0)
            end_frames = torch.repeat_interleave(preds_w[1:], step_size[:-1], dim=0)
            # interpolate between key frames and shifted
            interpolated_preds = (
                alphas * start_frames + (1 - alphas) * end_frames
            )  # (f-1)//step c h w
            # fill the rest of the frames with the last ones
            last_start = len(interpolated_preds)
            full_preds[:last_start] = interpolated_preds
            full_preds[last_start:] = preds_w[
                -1
            ]  # use last prediction for remaining frames
            preds_w = full_preds  # f c h w

        return preds_w[:total_frames]

    @torch.no_grad()
    def embed(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor = None,
        is_video: bool = True,
        interpolation: dict = {
            "mode": "bilinear",
            "align_corners": False,
            "antialias": True,
        },
        lowres_attenuation: bool = False,
    ) -> dict:
        """
        Generates watermarked videos from the input images and messages (used for inference).
        Videos may be arbitrarily sized.
        """
        """
      Generates watermarked videos from the input images and messages (used for inference).
      Videos may be arbitrarily sized.
      """
        if not is_video:
            # fallback on parent class for batch of images
            return super().embed(imgs, msgs, interpolation, lowres_attenuation)
        if msgs is None:
            msgs = self.get_random_msg()  # 1 x k
        else:
            assert msgs.shape[0] == 1, "Message should be unique"

        # encode by chunk of cksz imgs, propagate the wm to spsz next imgs
        chunk_size = self.chunk_size  # n=cksz
        step_size = self.step_size  # spsz

        imgs_w = torch.zeros_like(imgs)

        for ii in range(0, len(imgs[::step_size]), chunk_size):
            nimgs_in_ck = min(chunk_size, len(imgs[::step_size]) - ii)
            start = ii * step_size
            end = start + nimgs_in_ck * step_size
            all_imgs_in_ck = imgs[start:end, ...]

            # interpolate
            all_imgs_res = all_imgs_in_ck.clone()
            if all_imgs_res.shape[-2:] != (self.img_size, self.img_size):
                all_imgs_res = F.interpolate(
                    all_imgs_res, size=(self.img_size, self.img_size), **interpolation
                )
            all_imgs_res = all_imgs_res.to(self.device)

            # calculate distance and thresholding
            if not hasattr(self, "scene_detector"):
                self.set_scene_detector(ContentDetector())
            distinct_ids = self.scene_detector.detect_key_frames(all_imgs_res)

            base_ids = torch.arange(
                0, all_imgs_res.size(0), step_size, device=imgs.device
            )

            key_frame_ids = torch.cat([base_ids, distinct_ids])
            key_frame_ids = torch.unique(key_frame_ids)

            temp = torch.cat(
                [
                    key_frame_ids,
                    torch.tensor([all_imgs_res.size(0)], device=imgs.device),
                ]
            )
            repeats = temp[1:] - temp[:-1]

            imgs_res = all_imgs_res[key_frame_ids]

            # get deltas for the chunk, and repeat them for each frame in the chunk
            if self.embedder.yuv:  # take y channel only
                imgs_res = self.rgb2yuv(imgs_res)[:, 0:1]
            repeat_msgs = msgs.repeat(key_frame_ids.size(0), 1)

            preds_w = self.embedder(imgs_res, repeat_msgs.to(self.device))

            if hasattr(self, "logger"):
                self.logger.info(f"adaptive step size: {repeats}")

            preds_w = self._apply_video_mode(
                preds_w, len(all_imgs_in_ck), repeats, self.video_mode
            )

            # attenuate at low resolution if needed
            if self.attenuation is not None and lowres_attenuation:
                self.attenuation.to(all_imgs_res.device)
                hmaps = self.attenuation.heatmaps(all_imgs_res)
                preds_w = hmaps * preds_w

            # interpolate back
            preds_w = preds_w.to(imgs.device)
            if all_imgs_in_ck.shape[-2:] != (self.img_size, self.img_size):
                preds_w = F.interpolate(
                    preds_w, size=all_imgs_in_ck.shape[-2:], **interpolation
                )

            # attenuate at full resolution if needed
            if self.attenuation is not None and not lowres_attenuation:
                self.attenuation.to(all_imgs_in_ck.device)
                hmaps = self.attenuation.heatmaps(all_imgs_in_ck)
                preds_w = hmaps * preds_w

            # blend
            all_imgs_in_ck_w = self.blender(all_imgs_in_ck, preds_w)
            imgs_w[start:end, ...] = all_imgs_in_ck_w  # n 3 h w

        # clamp
        if self.clamp:
            imgs_w = torch.clamp(imgs_w, 0, 1)

        outputs = {
            "imgs_w": imgs_w,  # watermarked imgs: f 3 h w
            "msgs": msgs[0:1].repeat(len(imgs), 1),  # original messages: f k
        }
        return outputs


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
    pbar = tqdm(total=num_frames, desc="Watermark embedding")
    while True:
        in_bytes = process1.stdout.read(frame_size)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        chunk[frame_count % chunk_size] = frame
        frame_count += 1
        pbar.update(1)
        if frame_count % chunk_size == 0:
            processed_frame = embed_video_clip(model, chunk, msgs)
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
    cards_dir = Path(__file__).resolve().parent / "videoseal" / "videoseal" / "cards"
    model_card_path = cards_dir / "videoseal_1.0.yaml"
    if not model_card_path.exists():
        raise FileNotFoundError(f"Expected model card at {model_card_path}")

    cards_root = cards_dir.parent.parent
    target_ckpt_dir = cards_root / "ckpts"
    target_ckpt_dir.mkdir(parents=True, exist_ok=True)
    target_ckpt = target_ckpt_dir / "y_256b_img.pth"
    source_ckpt = Path(__file__).resolve().parent.parent / "ckpts" / "y_256b_img.pth"

    if (
        not target_ckpt.exists() or target_ckpt.stat().st_size < 100_000_000
    ) and source_ckpt.exists():
        shutil.copy2(source_ckpt, target_ckpt)
        logger.info("Copied checkpoint file to target directory.")

    # Load model (assuming your existing function)
    with _TempWorkingDir(cards_dir.parent.parent):
        model = setup_model_from_model_card(model_card_path)
    model = model.eval().to(device)
    model.compile()

    # Configuration
    model.step_size = args.step_size
    model.blender.scaling_w *= args.scaling_w
    model.video_mode = args.video_mode

    # Adaptive model setup

    adaptive_model = AdaptiveVideoseal(model.embedder, model.detector, model.augmenter)
    adaptive_model.load_state_dict(model.state_dict(), strict=False)
    adaptive_model.__dict__.update(model.__dict__)

    # --- Scene detector selection ---
    if args.scene_detector == "histogram":
        scene_detector = HistogramDetector()
    elif args.scene_detector == "content":
        scene_detector = ContentDetector()
    elif args.scene_detector == "hash":
        scene_detector = HashDetector()
    else:
        raise ValueError(f"Unknown scene detector: {args.scene_detector}")

    adaptive_model.set_scene_detector(scene_detector)
    adaptive_model.set_logger(logger)
    adaptive_model.eval().to(device)
    adaptive_model.compile()

    adaptive_model.step_size = args.step_size
    adaptive_model.blender.scaling_w *= args.scaling_w
    adaptive_model.attenuation = model.attenuation
    adaptive_model.video_mode = args.video_mode

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
        choices=["content", "histogram", "hash"],
        help="Select the scene detector type: 'content', 'histogram' or 'hash'.",
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
