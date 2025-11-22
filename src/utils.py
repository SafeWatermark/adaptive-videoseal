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
