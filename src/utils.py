import random
import torch
import numpy as np

from datetime import datetime
from pathlib import Path
import logging
import sys
from videoseal.utils.cfg import setup_model_from_model_card

import torch.nn as nn


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


def get_videoseal() -> nn.Module:
    current_path = Path(__file__).resolve().parent
    model = setup_model_from_model_card(current_path / "cards" / "videoseal_1.0.yml")
    return model


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
