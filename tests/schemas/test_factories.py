"""Tests for schemas.factories module."""

import pytest

from schemas.factories import (
    create_videoseal_meta,
    create_stegastamp_meta,
    create_revmark_meta,
    create_meta_for_model,
)
from schemas.model_outputs import ModelType, OutputValueType


class TestCreateVideosealMeta:
    def test_default(self):
        meta = create_videoseal_meta()
        assert meta.model_type == ModelType.VIDEOSEAL
        assert meta.value_type == OutputValueType.LOGITS
        assert meta.threshold == 0.0
        assert meta.nbits == 256
        assert meta.has_detection_bit is True

    def test_custom_nbits(self):
        meta = create_videoseal_meta(nbits=50)
        assert meta.nbits == 50


class TestCreateStegastampMeta:
    def test_default(self):
        meta = create_stegastamp_meta()
        assert meta.model_type == ModelType.STEGASTAMP
        assert meta.value_type == OutputValueType.BINARY
        assert meta.threshold == 0.5
        assert meta.nbits == 100

    def test_custom_nbits(self):
        meta = create_stegastamp_meta(nbits=50)
        assert meta.nbits == 50


class TestCreateRevmarkMeta:
    def test_default(self):
        meta = create_revmark_meta()
        assert meta.model_type == ModelType.REVMARK
        assert meta.value_type == OutputValueType.SIGMOID
        assert meta.threshold == 0.5
        assert meta.nbits == 96

    def test_custom_nbits(self):
        meta = create_revmark_meta(nbits=48)
        assert meta.nbits == 48


class TestCreateMetaForModel:
    """Tests for create_meta_for_model factory."""

    def test_videoseal_lowercase(self):
        """Should create VideoSeal meta from lowercase name."""
        meta = create_meta_for_model("videoseal", nbits=256)
        assert meta.model_type == ModelType.VIDEOSEAL

    def test_stega_stamp_with_underscore(self):
        """Should create StegaStamp meta from name with underscore."""
        meta = create_meta_for_model("stega_stamp", nbits=100)
        assert meta.model_type == ModelType.STEGASTAMP

    def test_stegastamp_without_underscore(self):
        """Should create StegaStamp meta from name without underscore."""
        meta = create_meta_for_model("stega_stamp", nbits=100)
        assert meta.model_type == ModelType.STEGASTAMP

    def test_revmark(self):
        """Should create REVMark meta."""
        meta = create_meta_for_model("revmark", nbits=96)
        assert meta.model_type == ModelType.REVMARK

    def test_case_insensitive(self):
        """Should be case insensitive."""
        meta = create_meta_for_model("VIDEOSEAL", nbits=32)
        assert meta.model_type == ModelType.VIDEOSEAL

    def test_invalid_model_raises(self):
        """Should raise ValueError for unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            create_meta_for_model("unknown_model", nbits=32)
