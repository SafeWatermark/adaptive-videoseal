"""Factory functions for creating model output metadata.

This module provides convenience factory functions for creating
ExtractorOutputMeta instances for each supported watermarking model.

Example:
    >>> from schemas import create_videoseal_meta
    >>> meta = create_videoseal_meta(nbits=32)
    >>> print(meta.threshold)  # 0.0
    >>> print(meta.value_type)  # OutputValueType.LOGITS
"""

from schemas.model_outputs import (
    ExtractorOutputMeta,
    ModelType,
    OutputValueType,
)


def create_videoseal_meta(nbits: int = 256) -> ExtractorOutputMeta:
    """Create metadata for VideoSeal extractor output.

    VideoSeal outputs unbounded logits with threshold at 0.0.
    The output includes a detection bit at index 0.

    Args:
        nbits: Number of message bits (excluding detection bit).

    Returns:
        ExtractorOutputMeta configured for VideoSeal.

    Example:
        >>> meta = create_videoseal_meta()
        >>> print(meta.model_type)  # ModelType.VIDEOSEAL
        >>> print(meta.threshold)   # 0.0
    """
    return ExtractorOutputMeta(
        model_type=ModelType.VIDEOSEAL,
        value_type=OutputValueType.LOGITS,
        threshold=0.0,
        nbits=nbits,
        has_detection_bit=True,
    )


def create_stegastamp_meta(nbits: int = 100) -> ExtractorOutputMeta:
    """Create metadata for StegaStamp extractor output.

    StegaStamp outputs hard binary decisions {0, 1} with threshold at 0.5.
    A dummy detection probability is prepended at index 0.

    Args:
        nbits: Number of message bits (default: 100 for StegaStamp).

    Returns:
        ExtractorOutputMeta configured for StegaStamp.

    Example:
        >>> meta = create_stegastamp_meta()
        >>> print(meta.model_type)  # ModelType.STEGASTAMP
        >>> print(meta.threshold)   # 0.5
    """
    return ExtractorOutputMeta(
        model_type=ModelType.STEGASTAMP,
        value_type=OutputValueType.BINARY,
        threshold=0.5,
        nbits=nbits,
        has_detection_bit=True,  # Dummy probability prepended
    )


def create_revmark_meta(nbits: int = 96) -> ExtractorOutputMeta:
    """Create metadata for REVMark extractor output.

    REVMark outputs sigmoid probabilities [0, 1] with threshold at 0.5.
    A dummy detection probability is prepended at index 0.

    Args:
        nbits: Number of message bits (default: 96 for REVMark).

    Returns:
        ExtractorOutputMeta configured for REVMark.

    Example:
        >>> meta = create_revmark_meta()
        >>> print(meta.model_type)  # ModelType.REVMARK
        >>> print(meta.threshold)   # 0.5
    """
    return ExtractorOutputMeta(
        model_type=ModelType.REVMARK,
        value_type=OutputValueType.SIGMOID,
        threshold=0.5,
        nbits=nbits,
        has_detection_bit=True,  # Dummy probability prepended
    )


def create_meta_for_model(
    model_name: str,
    nbits: int,
) -> ExtractorOutputMeta:
    """Create metadata for a model by name.

    Args:
        model_name: Model name ("videoseal", "stega_stamp", "revmark").
        nbits: Number of message bits.

    Returns:
        ExtractorOutputMeta configured for the specified model.

    Raises:
        ValueError: If model_name is not recognized.

    Example:
        >>> meta = create_meta_for_model("videoseal", nbits=256)
        >>> print(meta.model_type)  # ModelType.VIDEOSEAL
    """
    factories = {
        "videoseal": create_videoseal_meta,
        "stega_stamp": create_stegastamp_meta,
        "stegastamp": create_stegastamp_meta,
        "revmark": create_revmark_meta,
    }

    factory = factories.get(model_name.lower())
    if factory is None:
        valid_names = list(factories.keys())
        raise ValueError(f"Unknown model: {model_name}. Valid options: {valid_names}")

    return factory(nbits=nbits)
