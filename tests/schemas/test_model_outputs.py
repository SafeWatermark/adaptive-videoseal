"""Tests for schemas.model_outputs module."""

import pytest

from schemas.model_outputs import (
    OutputValueType,
    ModelType,
    ThresholdConfig,
    ExtractorOutputMeta,
)


class TestOutputValueType:
    def test_logits_value(self):
        assert OutputValueType.LOGITS.value == "logits"

    def test_sigmoid_value(self):
        assert OutputValueType.SIGMOID.value == "sigmoid"

    def test_binary_value(self):
        assert OutputValueType.BINARY.value == "binary"

    def test_enum_is_string_subclass(self):
        assert isinstance(OutputValueType.LOGITS, str)
        assert OutputValueType.LOGITS == "logits"


class TestModelType:
    def test_videoseal_value(self):
        assert ModelType.VIDEOSEAL.value == "videoseal"

    def test_stegastamp_value(self):
        assert ModelType.STEGASTAMP.value == "stega_stamp"

    def test_revmark_value(self):
        assert ModelType.REVMARK.value == "revmark"


class TestThresholdConfig:
    def test_default_thresholds(self):
        """DEFAULT_THRESHOLDS should have correct values."""
        assert ThresholdConfig.DEFAULT_THRESHOLDS[OutputValueType.LOGITS] == 0.0
        assert ThresholdConfig.DEFAULT_THRESHOLDS[OutputValueType.SIGMOID] == 0.5
        assert ThresholdConfig.DEFAULT_THRESHOLDS[OutputValueType.BINARY] == 0.5

    def test_for_value_type_logits(self):
        """for_value_type should create correct config for LOGITS."""
        config = ThresholdConfig.for_value_type(OutputValueType.LOGITS)
        assert config.value_type == OutputValueType.LOGITS
        assert config.threshold == 0.0

    def test_for_value_type_sigmoid(self):
        """for_value_type should create correct config for SIGMOID."""
        config = ThresholdConfig.for_value_type(OutputValueType.SIGMOID)
        assert config.value_type == OutputValueType.SIGMOID
        assert config.threshold == 0.5

    def test_for_model_videoseal(self):
        """for_model should create correct config for VIDEOSEAL."""
        config = ThresholdConfig.for_model(ModelType.VIDEOSEAL)
        assert config.value_type == OutputValueType.LOGITS
        assert config.threshold == 0.0

    def test_for_model_stegastamp(self):
        """for_model should create correct config for STEGASTAMP."""
        config = ThresholdConfig.for_model(ModelType.STEGASTAMP)
        assert config.value_type == OutputValueType.BINARY
        assert config.threshold == 0.5

    def test_for_model_revmark(self):
        """for_model should create correct config for REVMARK."""
        config = ThresholdConfig.for_model(ModelType.REVMARK)
        assert config.value_type == OutputValueType.SIGMOID
        assert config.threshold == 0.5

    def test_frozen_dataclass(self):
        """ThresholdConfig should be immutable."""
        config = ThresholdConfig.for_value_type(OutputValueType.LOGITS)
        with pytest.raises(AttributeError):
            config.threshold = 1.0


class TestExtractorOutputMeta:
    def test_creation(self):
        """Should create ExtractorOutputMeta with correct values."""
        meta = ExtractorOutputMeta(
            model_type=ModelType.VIDEOSEAL,
            value_type=OutputValueType.LOGITS,
            threshold=0.0,
            nbits=256,
        )
        assert meta.model_type == ModelType.VIDEOSEAL
        assert meta.value_type == OutputValueType.LOGITS
        assert meta.threshold == 0.0
        assert meta.nbits == 256
        assert meta.has_detection_bit is True  # default

    def test_has_detection_bit_override(self):
        """has_detection_bit should be overridable."""
        meta = ExtractorOutputMeta(
            model_type=ModelType.VIDEOSEAL,
            value_type=OutputValueType.LOGITS,
            threshold=0.0,
            nbits=256,
            has_detection_bit=False,
        )
        assert meta.has_detection_bit is False

    def test_validation_nbits_positive(self):
        """nbits must be >= 1."""
        with pytest.raises(ValueError, match="nbits must be >= 1"):
            ExtractorOutputMeta(
                model_type=ModelType.VIDEOSEAL,
                value_type=OutputValueType.LOGITS,
                threshold=0.0,
                nbits=0,
            )

    def test_frozen_dataclass(self):
        """ExtractorOutputMeta should be immutable."""
        meta = ExtractorOutputMeta(
            model_type=ModelType.VIDEOSEAL,
            value_type=OutputValueType.LOGITS,
            threshold=0.0,
            nbits=256,
        )
        with pytest.raises(AttributeError):
            meta.nbits = 64
