"""
Common perturbations for video watermarking robustness evaluation.

Based on the original VideoMarkBench implementation.
Each perturbation is implemented as a separate module.
"""

from .crop import Crop
from .frame_average import FrameAverage
from .frame_remove import FrameRemove
from .frame_switch import FrameSwitch
from .gaussian import Gaussian
from .gaussian_blur import GaussianBlur
from .jpeg import JPEG
from .mpeg4 import MPEG4

__all__ = [
    "Crop",
    "FrameAverage",
    "FrameRemove",
    "FrameSwitch",
    "Gaussian",
    "GaussianBlur",
    "JPEG",
    "MPEG4",
]
