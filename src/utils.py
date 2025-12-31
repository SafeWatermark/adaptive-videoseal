import random
import torch
import numpy as np

from datetime import datetime
from pathlib import Path
import logging
import sys
from videoseal.utils.cfg import setup_model_from_model_card
from contextlib import contextmanager
from models.embedder import StegaStampEmbedder, REVMarkEmbedder
from models.extractor import StegaStampExtractor, REVMarkExtractor
from models.stega_stamp import (
    STEGA_STAMP_IMAGE_WIDTH,
    STEGA_STAMP_ATTENUATION,
    STEGA_STAMP_SCALING_W,
    STEGA_STAMP_SCALING_I,
)
from models.revmark import (
    REVMARK_DEFAULT_WIDTH,
    REVMARK_ATTENUATION,
    REVMARK_SCALING_I,
    REVMARK_SCALING_W,
)
from videoseal.models import Videoseal

from adaptive_models.adaptive_videoseal import AdaptiveVideoseal
from ecc_codecs.base import ECCCodec
from msg_processors.layered_msg_processor import LayeredMsgProcessor
from msg_processors.ecc_layer import ECCMsgProcessor
from hooks.ecc_hooks import ecc_forward_hook
from mac.base import MACProvider
from hooks.mac_hooks import mac_forward_hook
from msg_processors.mac_layer import MACMsgProcessor


def setup_seed(seed: int):
    """Set seed for reproducibility across numpy, torch, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_dir: Path):
    """Create timestamped log file and console logger."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = (
        log_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # file handler
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging to {log_filename}")
    return logger


def select_device(device_choice: str):
    """Auto-select or manually set computation device."""
    if device_choice == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_videoseal() -> Videoseal:
    current_path = Path(__file__).resolve().parent
    model = setup_model_from_model_card(current_path / "cards" / "videoseal_1.0.yml")
    return model


def transform_ecc_videoseal(
    model: Videoseal | AdaptiveVideoseal,
    ecc_codec: ECCCodec,
) -> Videoseal | AdaptiveVideoseal:
    """
    TensorFlow layer can't be copied properly. So if you need PolarWatermarkCodec, recommend to use it.
    Transform a VideoSeal model to use ECC encoding/decoding.

    Embed: message_bits → [ECC encode] → embed into image/video
    Extract: extract from image/video → [ECC decode] → message_bits

    Args:
        model: VideoSeal or AdaptiveVideoseal model to transform
        ecc_codec: ECC Codec to encode message or decode payload

    Returns:
        Transformed model with ECC support
    """
    msg_processor = LayeredMsgProcessor(
        nbits=ecc_codec.message_bits, layers=[ECCMsgProcessor(ecc_codec)]
    )
    model.embedder.msg_processor = msg_processor
    model.detector.register_forward_hook(ecc_forward_hook)
    setattr(model.detector, "ecc_codec", ecc_codec)

    return model


def transform_authenticated_videoseal(
    model: Videoseal | AdaptiveVideoseal,
    mac_provider: MACProvider,
):
    """

    Embed: message_bits → [MAC tag attach] → embed into image/video
    Extract: extract from image/video → [MAC tag dettach] → message_bits

    Args:
        model: VideoSeal or AdaptiveVideoseal model to transform
        mac_provider: MacProvider to encode (message | MAC) or decode payload( message | MAC -> message, MAC)

    Returns:
        Transformed model with ECC support
    """
    msg_processor = LayeredMsgProcessor(
        nbits=mac_provider.message_bits, layers=[MACMsgProcessor(mac_provider)]
    )
    model.embedder.msg_processor = msg_processor
    model.detector.register_forward_hook(mac_forward_hook)
    setattr(model.detector, "mac_provider", mac_provider)

    return model


def transform_authenticated_ecc_videoseal(
    model: Videoseal | AdaptiveVideoseal, ecc_codec: ECCCodec, mac_provider: MACProvider
):
    """
    Embed: message_bits → [MAC: append tag] → [ECC encode] → embed
    Extract: extract → [ECC decode] → [MAC: verify & strip tag] → message_bits + verification_status

    Args:
        model: VideoSeal or AdaptiveVideoseal model to transform
        ecc_codec: ECC Codec to encode message or decode payload
        mac_provider: MacProvider to encode (message | MAC) or decode payload( message | MAC -> message, MAC)

    Returns:
        Transformed model with ECC & MAC support
    """
    msg_processor = LayeredMsgProcessor(
        nbits=mac_provider.message_bits,
        layers=[MACMsgProcessor(mac_provider), ECCMsgProcessor(ecc_codec)],
    )
    model.embedder.msg_processor = msg_processor
    model.detector.register_forward_hook(ecc_forward_hook)
    model.detector.register_forward_hook(mac_forward_hook)
    setattr(model.detector, "mac_provider", mac_provider)
    setattr(model.detector, "ecc_codec", ecc_codec)

    return model


@contextmanager
def patch_ffmpeg_path(
    bad_path="/private/home/pfz/09-videoseal/vmaf-dev/ffmpeg-git-20240629-amd64-static/ffmpeg",
):
    """
    Monkey-patch subprocess.run so that any command starting with the bad ffmpeg
    path is replaced with the real system ffmpeg.

    This avoids modifying library code directly.
    """
    import subprocess
    import shutil

    corrected_path = shutil.which("ffmpeg")
    if corrected_path is None:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or add it to your PATH.")

    original_run = subprocess.run

    def patched_run(*popenargs, **kwargs):
        if popenargs:
            cmd = popenargs[0]

            # Handle most common case: command is a list/tuple
            if isinstance(cmd, (list, tuple)) and cmd and cmd[0] == bad_path:
                new_cmd = [corrected_path, *cmd[1:]]
                popenargs = (new_cmd, *popenargs[1:])

        return original_run(*popenargs, **kwargs)

    subprocess.run = patched_run

    logging.info(f"[patch_ffmpeg_path] Patched ffmpeg: {bad_path} -> {corrected_path}")
    try:
        yield corrected_path
    finally:
        subprocess.run = original_run
        logging.info("[patch_ffmpeg_path] Restored original subprocess.run")


def get_stega_stamp_videoseal(onnx_path: str) -> Videoseal:
    """
    Configure a VideoSeal model to use StegaStamp for watermark embedding and extraction.

    Key behavior and requirements:
    - StegaStamp expects a *1:1 blend* between the original image and the watermark residual.
      Therefore, both `blender.scaling_i` and `blender.scaling_w` must be set to `1.0`.

    - VideoSeal 1.0 enables attenuation by default, but StegaStamp does NOT use attenuation.
      Set `model.attenuation = None` to disable it.

    - StegaStamp produces an encoded image by:
          encoded = torch.clamp(image + residual, 0, 1)

    - Detection accuracy drops significantly if `scaling_w < scaling_i`.
      Keep both equal (1.0).

    - StegaStamp supports only 400×400 images, so the model's image size is fixed at 400.

    This function loads the base VideoSeal model, replaces its embedder and detector
    with StegaStamp modules, and applies the necessary configuration.
    """
    model = get_videoseal()
    model.embedder = StegaStampEmbedder(
        onnx_path, msg_processor=model.embedder.msg_processor
    )
    model.detector = StegaStampExtractor(onnx_path)

    model.img_size = STEGA_STAMP_IMAGE_WIDTH
    model.attenuation = STEGA_STAMP_ATTENUATION
    model.blender.scaling_i = STEGA_STAMP_SCALING_I
    model.blender.scaling_w = STEGA_STAMP_SCALING_W
    return model


def get_revmark_videoseal(
    encoder_path: str,
    decoder_path: str,
) -> Videoseal:
    """
    Configure a VideoSeal model to use REVMark for watermark embedding and extraction.

    Key behavior and requirements:
    - REVMark operates on 8-frame blocks at 128x128 resolution.
    - REVMark expects pixel values in [-1, 1], so scaling_w=0.5 is used
      to convert the residual to [0, 1] domain.
    - REVMark does not use attenuation (JND masking).
    - The extractor aggregates decoded messages by averaging across 8-frame chunks.

    Args:
        encoder_path: Path to the REVMark encoder checkpoint (Encoder.pth).
        decoder_path: Path to the REVMark decoder checkpoint (Decoder.pth).

    Returns:
        VideoSeal model configured for REVMark embedding and extraction.
    """
    model = get_videoseal()
    model.embedder = REVMarkEmbedder(
        encoder_path, msg_processor=model.embedder.msg_processor
    )
    model.detector = REVMarkExtractor(decoder_path)

    model.img_size = REVMARK_DEFAULT_WIDTH
    model.attenuation = REVMARK_ATTENUATION
    model.blender.scaling_i = REVMARK_SCALING_I
    model.blender.scaling_w = REVMARK_SCALING_W
    return model
