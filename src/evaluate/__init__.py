"""
VideoMarkBench evaluation system following original repository structure.

This module provides a unified evaluation system that follows the original
VideoMarkBench repository structure, supporting:
- Common perturbations
- Blackbox attacks (forgery and removal)
- Whitebox attacks (WEvade)
- All aggregation strategies
- Pre-applied perturbations for faster evaluation
"""

from evaluate.common_evaluator import (
    evaluate_common_perturbations,
    evaluate_with_preapplied_perturbations,
    get_common_perturbations,
)
from evaluate.perturbed_dataset import PerturbedVideoDataset
from evaluate.dataset_loader import VideoMarkBenchDataset
from evaluate.model_wrapper import VideoMarkBenchModel

__all__ = [
    "evaluate_common_perturbations",
    "evaluate_with_preapplied_perturbations",
    "get_common_perturbations",
    "PerturbedVideoDataset",
    "VideoMarkBenchDataset",
    "VideoMarkBenchModel",
]

__version__ = "0.1.0"
