"""Unified value conversion utilities.

This module provides the ValueConverter class for converting between different
value representations used by watermarking models.
Example:
    >>> import torch
    >>> from schemas import ValueConverter, OutputValueType
    >>>
    >>> # Auto-detect value type
    >>> logits = torch.randn(1, 256)  # Unbounded values
    >>> value_type = ValueConverter.detect_value_type(logits)
    >>> print(value_type)  # OutputValueType.LOGITS
    >>>
    >>> # Convert to binary
    >>> bits = ValueConverter.to_binary(logits, value_type)
    >>>
    >>> # Convert to LLR for Polar decoding
    >>> llr = ValueConverter.to_llr(logits, value_type)
"""

from typing import Literal

import torch

from schemas.model_outputs import OutputValueType


class ValueConverter:
    """Utility class for converting between different value representations.

    This class provides static methods for:
    - Detecting the type of input values (logits, sigmoid, binary)
    - Converting values to binary bits using appropriate thresholds
    - Converting values to Log-Likelihood Ratios (LLR) for soft decoding
    - Converting values to probabilities

    Constants:
        SIGMOID_RANGE_MARGIN: Margin for detecting sigmoid values (0.05).
        LLR_MAX: Maximum LLR value for clipping (20.0).
    """

    SIGMOID_RANGE_MARGIN: float = 0.05
    LLR_MAX: float = 20.0

    @staticmethod
    def detect_value_type(
        values: torch.Tensor,
        hint: Literal["auto", "sigmoid", "logits", "binary"] = "auto",
    ) -> OutputValueType:
        """Detect the value type of a tensor.

        Args:
            values: Input tensor of shape (B, N) or (N,).
            hint: User-provided hint for the value type:
                - "auto": Auto-detect based on value range (default)
                - "sigmoid": Treat as sigmoid probabilities [0, 1]
                - "logits": Treat as raw logits (unbounded)
                - "binary": Treat as hard binary decisions {0, 1}

        Returns:
            Detected OutputValueType.

        Example:
            >>> logits = torch.randn(1, 256)
            >>> value_type = ValueConverter.detect_value_type(logits)
            >>> print(value_type)  # OutputValueType.LOGITS
        """
        if hint == "sigmoid":
            return OutputValueType.SIGMOID
        elif hint == "logits":
            return OutputValueType.LOGITS
        elif hint == "binary":
            return OutputValueType.BINARY

        # Auto-detection
        min_val, max_val = float(values.min()), float(values.max())

        # Check for binary (exactly 0 and 1 values)
        unique = torch.unique(values)
        if len(unique) <= 2:
            unique_list = unique.tolist()
            if all(v in [0, 1, 0.0, 1.0] for v in unique_list):
                return OutputValueType.BINARY

        # Check for sigmoid range [0, 1] with margin
        margin = ValueConverter.SIGMOID_RANGE_MARGIN
        if min_val >= -margin and max_val <= 1.0 + margin:
            if min_val >= 0.0 and max_val <= 1.0:
                return OutputValueType.SIGMOID

        return OutputValueType.LOGITS

    @staticmethod
    def to_binary(
        values: torch.Tensor,
        value_type: OutputValueType,
    ) -> torch.Tensor:
        """Convert values to binary bits based on value type.

        Args:
            values: Input tensor of shape (B, N) or (N,).
            value_type: The type of input values.

        Returns:
            Binary tensor (uint8) of same shape with values {0, 1}.

        Example:
            >>> sigmoid_values = torch.tensor([[0.2, 0.8, 0.3, 0.9]])
            >>> bits = ValueConverter.to_binary(sigmoid_values, OutputValueType.SIGMOID)
            >>> print(bits)  # tensor([[0, 1, 0, 1]], dtype=torch.uint8)
        """
        thresholds = {
            OutputValueType.LOGITS: 0.0,
            OutputValueType.SIGMOID: 0.5,
            OutputValueType.BINARY: 0.5,
        }
        threshold = thresholds[value_type]
        return (values > threshold).to(torch.uint8)

    @staticmethod
    def to_llr(
        values: torch.Tensor,
        value_type: OutputValueType,
    ) -> torch.Tensor:
        """Convert values to Log-Likelihood Ratios (LLR).

        LLR represents the log probability ratio: log(P(bit=1) / P(bit=0)).
        - Positive LLR: bit=1 more likely
        - Negative LLR: bit=0 more likely

        Args:
            values: Input tensor of shape (B, N) or (N,).
            value_type: The type of input values.

        Returns:
            LLR tensor (float32) of same shape, clipped to [-LLR_MAX, LLR_MAX].

        Conversion rules:
            - LOGITS: Already LLR-like, just clip to max range.
            - SIGMOID: Apply log(p / (1-p)).
            - BINARY: Map to hard LLR (+/-LLR_MAX).

        Example:
            >>> sigmoid_values = torch.tensor([[0.1, 0.9]])
            >>> llr = ValueConverter.to_llr(sigmoid_values, OutputValueType.SIGMOID)
            >>> print(llr)  # tensor([[-2.1972,  2.1972]])
        """
        llr_max = ValueConverter.LLR_MAX

        if value_type == OutputValueType.LOGITS:
            return torch.clamp(values, -llr_max, llr_max)

        elif value_type == OutputValueType.SIGMOID:
            eps = 1e-7
            p = torch.clamp(values.float(), eps, 1.0 - eps)
            llr = torch.log(p / (1.0 - p))
            return torch.clamp(llr, -llr_max, llr_max)

        else:  # BINARY
            # Hard decisions: map 0 -> -LLR_MAX, 1 -> +LLR_MAX
            return (values.float() * 2 - 1) * llr_max

    @staticmethod
    def to_probability(
        values: torch.Tensor,
        value_type: OutputValueType,
    ) -> torch.Tensor:
        """Convert values to probability space [0, 1].

        Args:
            values: Input tensor of shape (B, N) or (N,).
            value_type: The type of input values.

        Returns:
            Probability tensor (float32) of same shape with values in [0, 1].

        Conversion rules:
            - LOGITS: Apply sigmoid function.
            - SIGMOID: Return as-is (already in probability space).
            - BINARY: Return as-is (0/1 are valid probabilities).

        Example:
            >>> logits = torch.tensor([[-2.0, 0.0, 2.0]])
            >>> probs = ValueConverter.to_probability(logits, OutputValueType.LOGITS)
            >>> print(probs)  # tensor([[0.1192, 0.5000, 0.8808]])
        """
        if value_type == OutputValueType.LOGITS:
            return torch.sigmoid(values)
        return values.float()

    @staticmethod
    def get_threshold(value_type: OutputValueType) -> float:
        """Get the default decision threshold for a value type.

        Args:
            value_type: The type of values.

        Returns:
            The default threshold (0.0 for LOGITS, 0.5 for SIGMOID/BINARY).
        """
        thresholds = {
            OutputValueType.LOGITS: 0.0,
            OutputValueType.SIGMOID: 0.5,
            OutputValueType.BINARY: 0.5,
        }
        return thresholds[value_type]
