import random
import torch
import numpy as np

from datetime import datetime
from pathlib import Path
import logging
import sys
import os
import shutil
from videoseal.utils.cfg import setup_model_from_model_card


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


def get_videoseal():
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
        logging.info("Copied checkpoint file to target directory.")

    # Load model (assuming your existing function)
    with _TempWorkingDir(cards_dir.parent.parent):
        model = setup_model_from_model_card(model_card_path)

    return model
