"""Schemas for model input/output data structures.

This module provides unified data schemas for handling different watermarking model
outputs (VideoSeal, StegaStamp, REVMark) with consistent interfaces.
"""

from schemas.model_outputs import (
    OutputValueType,
    ModelType,
    ThresholdConfig,
    ExtractorOutputMeta,
)
from schemas.converters import ValueConverter
from schemas.factories import (
    create_videoseal_meta,
    create_stegastamp_meta,
    create_revmark_meta,
    create_meta_for_model,
)

__all__ = [
    # Enums
    "OutputValueType",
    "ModelType",
    # Config classes
    "ThresholdConfig",
    "ExtractorOutputMeta",
    # Converter
    "ValueConverter",
    # Factory functions
    "create_videoseal_meta",
    "create_stegastamp_meta",
    "create_revmark_meta",
    "create_meta_for_model",
]
