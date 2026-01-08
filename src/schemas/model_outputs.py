"""Core schema classes for model outputs.

This module defines the data structures for handling different watermarking model
outputs with unified interfaces.

Output Value Types:
    - LOGITS: VideoSeal outputs unbounded logits, threshold at 0.0
    - SIGMOID: REVMark outputs sigmoid probabilities [0, 1], threshold at 0.5
    - BINARY: StegaStamp outputs hard binary decisions {0, 1}, threshold at 0.5

Example:
    >>> from schemas import OutputValueType, ExtractorOutputMeta, create_videoseal_meta
    >>> meta = create_videoseal_meta(nbits=256)
    >>> print(meta.threshold)  # 0.0
    >>> print(meta.value_type)  # OutputValueType.LOGITS
"""

from enum import Enum
from dataclasses import dataclass
from typing import ClassVar


class OutputValueType(str, Enum):
    """Enumeration of model output value types.

    Attributes:
        LOGITS: Unbounded real values (-inf, +inf), decision threshold at 0.0.
                Used by VideoSeal.
        SIGMOID: Probability values [0, 1], decision threshold at 0.5.
                 Used by REVMark.
        BINARY: Hard binary decisions {0, 1}, decision threshold at 0.5.
                Used by StegaStamp.
    """

    LOGITS = "logits"
    SIGMOID = "sigmoid"
    BINARY = "binary"


class ModelType(str, Enum):
    """Enumeration of supported watermarking models.

    Attributes:
        VIDEOSEAL: VideoSeal 1.0 model with logit outputs.
        STEGASTAMP: StegaStamp model with binary outputs.
        REVMARK: REVMark model with sigmoid outputs.
    """

    VIDEOSEAL = "videoseal"
    STEGASTAMP = "stega_stamp"
    REVMARK = "revmark"


@dataclass(frozen=True)
class ThresholdConfig:
    """Configuration for value-to-bit conversion thresholds.

    Attributes:
        value_type: The type of output values.
        threshold: Decision boundary for converting to binary.

    Class Attributes:
        DEFAULT_THRESHOLDS: Default thresholds for each value type.
    """

    value_type: OutputValueType
    threshold: float

    DEFAULT_THRESHOLDS: ClassVar[dict[OutputValueType, float]] = {
        OutputValueType.LOGITS: 0.0,
        OutputValueType.SIGMOID: 0.5,
        OutputValueType.BINARY: 0.5,
    }

    @classmethod
    def for_value_type(cls, value_type: OutputValueType) -> "ThresholdConfig":
        """Create config with default threshold for a value type.

        Args:
            value_type: The output value type.

        Returns:
            ThresholdConfig with appropriate default threshold.
        """
        return cls(
            value_type=value_type,
            threshold=cls.DEFAULT_THRESHOLDS[value_type],
        )

    @classmethod
    def for_model(cls, model_type: ModelType) -> "ThresholdConfig":
        """Create config for a specific model type.

        Args:
            model_type: The watermarking model type.

        Returns:
            ThresholdConfig with appropriate settings for the model.
        """
        model_to_value_type = {
            ModelType.VIDEOSEAL: OutputValueType.LOGITS,
            ModelType.STEGASTAMP: OutputValueType.BINARY,
            ModelType.REVMARK: OutputValueType.SIGMOID,
        }
        value_type = model_to_value_type[model_type]
        return cls.for_value_type(value_type)


@dataclass(frozen=True)
class ExtractorOutputMeta:
    """Metadata describing extractor output characteristics.

    This dataclass provides a unified way to describe the output format
    of different watermarking model extractors.

    Attributes:
        model_type: The type of watermarking model.
        value_type: The type of output values (logits, sigmoid, binary).
        threshold: Decision threshold for converting to binary.
        nbits: Number of message bits (excluding detection bit).
        has_detection_bit: Whether output includes detection probability at index 0.

    Example:
        >>> meta = ExtractorOutputMeta(
        ...     model_type=ModelType.VIDEOSEAL,
        ...     value_type=OutputValueType.LOGITS,
        ...     threshold=0.0,
        ...     nbits=256,
        ... )
        >>> print(meta.threshold)  # 0.0
    """

    model_type: ModelType
    value_type: OutputValueType
    threshold: float
    nbits: int
    has_detection_bit: bool = True

    def __post_init__(self):
        """Validate the metadata after initialization."""
        if self.nbits < 1:
            raise ValueError(f"nbits must be >= 1, got {self.nbits}")
