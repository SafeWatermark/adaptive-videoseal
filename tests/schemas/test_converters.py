"""Tests for schemas.converters module."""

import torch

from schemas.converters import ValueConverter
from schemas.model_outputs import OutputValueType


class TestValueConverterDetection:
    """Tests for ValueConverter.detect_value_type."""

    def test_detect_logits_from_unbounded_values(self):
        """Unbounded values should be detected as LOGITS."""
        logits = torch.randn(1, 32) * 5  # Values outside [0, 1]
        result = ValueConverter.detect_value_type(logits)
        assert result == OutputValueType.LOGITS

    def test_detect_sigmoid_from_probability_values(self):
        """Values in [0, 1] should be detected as SIGMOID."""
        sigmoid = torch.rand(1, 32)  # Values in [0, 1)
        result = ValueConverter.detect_value_type(sigmoid)
        assert result == OutputValueType.SIGMOID

    def test_detect_binary_from_hard_decisions(self):
        """Values that are only 0 and 1 should be detected as BINARY."""
        binary = torch.randint(0, 2, (1, 32)).float()
        result = ValueConverter.detect_value_type(binary)
        assert result == OutputValueType.BINARY

    def test_hint_sigmoid_overrides_detection(self):
        """Hint 'sigmoid' should override auto-detection."""
        logits = torch.randn(1, 32) * 5  # Would normally be LOGITS
        result = ValueConverter.detect_value_type(logits, hint="sigmoid")
        assert result == OutputValueType.SIGMOID

    def test_hint_logits_overrides_detection(self):
        """Hint 'logits' should override auto-detection."""
        sigmoid = torch.rand(1, 32)  # Would normally be SIGMOID
        result = ValueConverter.detect_value_type(sigmoid, hint="logits")
        assert result == OutputValueType.LOGITS

    def test_hint_binary_overrides_detection(self):
        """Hint 'binary' should override auto-detection."""
        sigmoid = torch.rand(1, 32)  # Would normally be SIGMOID
        result = ValueConverter.detect_value_type(sigmoid, hint="binary")
        assert result == OutputValueType.BINARY

    def test_detect_handles_edge_case_all_zeros(self):
        """All zeros should be detected as BINARY."""
        zeros = torch.zeros(1, 32)
        result = ValueConverter.detect_value_type(zeros)
        assert result == OutputValueType.BINARY

    def test_detect_handles_edge_case_all_ones(self):
        """All ones should be detected as BINARY."""
        ones = torch.ones(1, 32)
        result = ValueConverter.detect_value_type(ones)
        assert result == OutputValueType.BINARY


class TestValueConverterToBinary:
    """Tests for ValueConverter.to_binary."""

    def test_logits_threshold_at_zero(self):
        """LOGITS should use threshold 0.0."""
        logits = torch.tensor([[-1.0, 0.5, -0.5, 2.0]])
        result = ValueConverter.to_binary(logits, OutputValueType.LOGITS)
        expected = torch.tensor([[0, 1, 0, 1]], dtype=torch.uint8)
        assert torch.equal(result, expected)

    def test_sigmoid_threshold_at_half(self):
        """SIGMOID should use threshold 0.5."""
        sigmoid = torch.tensor([[0.3, 0.7, 0.5, 0.9]])
        result = ValueConverter.to_binary(sigmoid, OutputValueType.SIGMOID)
        # 0.3 < 0.5 -> 0, 0.7 > 0.5 -> 1, 0.5 == 0.5 (not >) -> 0, 0.9 > 0.5 -> 1
        expected = torch.tensor([[0, 1, 0, 1]], dtype=torch.uint8)
        assert torch.equal(result, expected)

    def test_binary_passthrough(self):
        """BINARY should pass through (threshold 0.5 for 0/1 values)."""
        binary = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
        result = ValueConverter.to_binary(binary, OutputValueType.BINARY)
        expected = torch.tensor([[0, 1, 0, 1]], dtype=torch.uint8)
        assert torch.equal(result, expected)

    def test_output_dtype_is_uint8(self):
        """Output should always be uint8."""
        values = torch.randn(1, 32)
        result = ValueConverter.to_binary(values, OutputValueType.LOGITS)
        assert result.dtype == torch.uint8


class TestValueConverterToLLR:
    """Tests for ValueConverter.to_llr."""

    def test_logits_clipped_to_max(self):
        """LOGITS should be clipped to [-LLR_MAX, LLR_MAX]."""
        logits = torch.tensor([[-100.0, 100.0]])
        result = ValueConverter.to_llr(logits, OutputValueType.LOGITS)
        assert result.min() >= -ValueConverter.LLR_MAX
        assert result.max() <= ValueConverter.LLR_MAX

    def test_sigmoid_converted_to_llr(self):
        """SIGMOID should be converted via log(p / (1-p))."""
        sigmoid = torch.tensor([[0.5]])  # log(0.5 / 0.5) = 0
        result = ValueConverter.to_llr(sigmoid, OutputValueType.SIGMOID)
        assert torch.isclose(result, torch.tensor([[0.0]]), atol=1e-5).all()

    def test_sigmoid_extreme_values_clipped(self):
        """Extreme SIGMOID values should be clipped."""
        sigmoid = torch.tensor([[0.001, 0.999]])
        result = ValueConverter.to_llr(sigmoid, OutputValueType.SIGMOID)
        assert result.min() >= -ValueConverter.LLR_MAX
        assert result.max() <= ValueConverter.LLR_MAX

    def test_binary_mapped_to_hard_llr(self):
        """BINARY should map 0->-LLR_MAX, 1->+LLR_MAX."""
        binary = torch.tensor([[0.0, 1.0]])
        result = ValueConverter.to_llr(binary, OutputValueType.BINARY)
        expected = torch.tensor([[-ValueConverter.LLR_MAX, ValueConverter.LLR_MAX]])
        assert torch.equal(result, expected)


class TestValueConverterToProbability:
    """Tests for ValueConverter.to_probability."""

    def test_logits_apply_sigmoid(self):
        """LOGITS should apply sigmoid function."""
        logits = torch.tensor([[0.0]])  # sigmoid(0) = 0.5
        result = ValueConverter.to_probability(logits, OutputValueType.LOGITS)
        assert torch.isclose(result, torch.tensor([[0.5]]), atol=1e-5).all()

    def test_sigmoid_passthrough(self):
        """SIGMOID should pass through unchanged."""
        sigmoid = torch.tensor([[0.3, 0.7]])
        result = ValueConverter.to_probability(sigmoid, OutputValueType.SIGMOID)
        assert torch.equal(result, sigmoid.float())

    def test_binary_passthrough(self):
        """BINARY should pass through unchanged."""
        binary = torch.tensor([[0.0, 1.0]])
        result = ValueConverter.to_probability(binary, OutputValueType.BINARY)
        assert torch.equal(result, binary.float())


class TestValueConverterGetThreshold:
    """Tests for ValueConverter.get_threshold."""

    def test_logits_threshold(self):
        """LOGITS threshold should be 0.0."""
        assert ValueConverter.get_threshold(OutputValueType.LOGITS) == 0.0

    def test_sigmoid_threshold(self):
        """SIGMOID threshold should be 0.5."""
        assert ValueConverter.get_threshold(OutputValueType.SIGMOID) == 0.5

    def test_binary_threshold(self):
        """BINARY threshold should be 0.5."""
        assert ValueConverter.get_threshold(OutputValueType.BINARY) == 0.5
