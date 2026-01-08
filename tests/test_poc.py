"""Tests for poc.py"""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from poc import (
    get_model_config,
    get_video_info,
    ExperimentConfig,
    ModelFactory,
    CodecManager,
    ResultsManager,
    embed_video_clip,
    detect_video_clip,
)
from models.stega_stamp import STEGA_STAMP_NUM_BITS
from models.revmark import REVMARK_DEFAULT_MSG_BITS


class TestGetModelConfig:
    """Tests for get_model_config function."""

    def test_videoseal_config(self, videoseal_model):
        """Should return correct config for videoseal model."""
        config = get_model_config("videoseal", videoseal_model)

        assert "total_bits" in config
        assert "threshold" in config
        assert "meta" in config
        assert config["total_bits"] == videoseal_model.embedder.msg_processor.nbits
        assert config["threshold"] == 0.0  # VideoSeal uses logits

    def test_stega_stamp_config(self):
        """Should return correct config for stega_stamp model."""
        mock_model = MagicMock()

        config = get_model_config("stega_stamp", mock_model)

        assert config["total_bits"] == STEGA_STAMP_NUM_BITS
        assert config["threshold"] == 0.5  # StegaStamp uses binary

    def test_revmark_config(self):
        """Should return correct config for revmark model."""
        mock_model = MagicMock()

        config = get_model_config("revmark", mock_model)

        assert config["total_bits"] == REVMARK_DEFAULT_MSG_BITS
        assert config["threshold"] == 0.5  # REVMark uses sigmoid

    def test_unknown_model_raises(self):
        """Should raise ValueError for unknown model."""
        mock_model = MagicMock()

        with pytest.raises(ValueError, match="Unknown model"):
            get_model_config("unknown_model", mock_model)


class TestGetVideoInfo:
    def test_get_video_info_returns_dict(self):
        """Should return dictionary with video properties."""
        import cv2

        with patch("poc.cv2.VideoCapture") as mock_capture:
            mock_video = MagicMock()
            mock_video.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_WIDTH: 1920,
                cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FOURCC: 0x34363248,  # H264
                cv2.CAP_PROP_FRAME_COUNT: 300,
            }.get(prop, 0)
            mock_capture.return_value = mock_video

            result = get_video_info("dummy_path.mp4")

            assert "width" in result
            assert "height" in result
            assert "fps" in result
            assert "codec" in result
            assert "num_frames" in result
            assert result["width"] == 1920
            assert result["height"] == 1080
            assert result["fps"] == 30.0
            assert result["num_frames"] == 300
            mock_video.release.assert_called_once()


class TestExperimentConfig:
    @pytest.fixture
    def default_args(self):
        """Create default args for testing."""
        args = argparse.Namespace(
            input_dir="./data",
            output_dir="./outputs",
            num_frames_for_extraction=32,
            chunk_size=32,
            video_filename="test.mp4",
            n_trials=5,
            seed=42,
            device="cpu",
            model="videoseal",
            scaling_w=1.5,
            step_size=8,
            video_mode="repeat",
            ecc_mode=False,
            mac_mode=False,
            ecc_codec="polar",
            mac_algorithm="hmac",
            mac_tag_bits=32,
            message_bits=32,
            compare_full_embed=False,
            scene_detector="content",
            onnx_path=None,
        )
        return args

    def test_initialization(self, default_args):
        """Should properly initialize with args."""
        config = ExperimentConfig(default_args)

        assert config.args == default_args
        assert config.input_dir == Path("./data")
        assert config.output_dir == Path("./outputs")
        assert config.log_dir == Path("./outputs/logs")

    def test_validate_warns_when_extraction_frames_less_than_chunk(self, default_args):
        """Should warn when num_frames_for_extraction < chunk_size."""
        default_args.num_frames_for_extraction = 16
        default_args.chunk_size = 32
        config = ExperimentConfig(default_args)
        mock_logger = MagicMock()

        config.validate(mock_logger)

        mock_logger.warning.assert_called_once()

    def test_validate_no_warning_when_valid(self, default_args):
        """Should not warn when num_frames_for_extraction >= chunk_size."""
        default_args.num_frames_for_extraction = 32
        default_args.chunk_size = 32
        config = ExperimentConfig(default_args)
        mock_logger = MagicMock()

        config.validate(mock_logger)

        mock_logger.warning.assert_not_called()

    def test_log_config(self, default_args):
        """Should log all configuration values."""
        config = ExperimentConfig(default_args)
        mock_logger = MagicMock()

        config.log_config(mock_logger)

        # Should log header and all args
        assert mock_logger.info.call_count >= len(vars(default_args)) + 2


class TestModelFactory:
    def test_create_videoseal_model(self, videoseal_model):
        """Should create videoseal model with correct attributes."""
        args = argparse.Namespace(
            model="videoseal",
            scaling_w=1.5,
            step_size=8,
            video_mode="repeat",
            chunk_size=32,
            onnx_path=None,
        )
        mock_logger = MagicMock()

        with patch("src.poc.get_videoseal", return_value=videoseal_model):
            model = ModelFactory.create_model(args, "cpu", mock_logger)

            assert model.step_size == 8
            assert model.video_mode == "repeat"
            assert model.chunk_size == 32

    def test_create_stega_stamp_without_onnx_raises(self):
        """Should raise ValueError when creating stega_stamp without onnx_path."""
        args = argparse.Namespace(
            model="stega_stamp",
            onnx_path=None,
            scaling_w=1.0,
            step_size=8,
            video_mode="repeat",
            chunk_size=32,
        )
        mock_logger = MagicMock()

        with pytest.raises(ValueError, match="ONNX model file"):
            ModelFactory.create_model(args, "cpu", mock_logger)

    def test_create_unknown_model_raises(self):
        """Should raise ValueError for unknown model."""
        args = argparse.Namespace(
            model="unknown_model",
            onnx_path=None,
            scaling_w=1.0,
            step_size=8,
            video_mode="repeat",
            chunk_size=32,
        )
        mock_logger = MagicMock()

        with pytest.raises(ValueError, match="Unknown model"):
            ModelFactory.create_model(args, "cpu", mock_logger)


class TestCodecManager:
    @pytest.fixture
    def default_args(self):
        """Create default args for testing."""
        return argparse.Namespace(
            ecc_mode=False,
            mac_mode=False,
            ecc_codec="polar",
            mac_algorithm="hmac",
            mac_tag_bits=32,
            message_bits=32,
        )

    def test_initialization(self, default_args):
        """Should properly initialize with args."""
        manager = CodecManager(default_args, total_bits=256, threshold=0.0)

        assert manager.args == default_args
        assert manager.total_bits == 256
        assert manager.threshold == 0.0
        assert manager.mac_provider is None
        assert manager.adaptive_mac_provider is None
        assert manager.ecc_codec is None

    def test_apply_codecs_no_mode(self, default_args):
        """Should return models unchanged when no mode enabled."""
        manager = CodecManager(default_args, total_bits=256, threshold=0.0)
        mock_model = MagicMock()
        mock_adaptive_model = MagicMock()

        result_model, result_adaptive = manager.apply_codecs(
            mock_model, mock_adaptive_model
        )

        assert result_model == mock_model
        assert result_adaptive == mock_adaptive_model

    def test_apply_ecc_mode(self, default_args):
        """Should apply ECC transforms when ecc_mode enabled."""
        default_args.ecc_mode = True
        manager = CodecManager(default_args, total_bits=256, threshold=0.0)

        mock_model = MagicMock()
        mock_adaptive_model = MagicMock()

        with patch("poc.transform_ecc_videoseal") as mock_transform:
            mock_transform.side_effect = lambda m, **_: m
            manager.apply_codecs(mock_model, mock_adaptive_model)

            assert mock_transform.call_count == 2
            assert manager.ecc_codec is not None


class TestResultsManager:
    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing."""
        config = MagicMock()
        config.output_dir = Path(tempfile.mkdtemp())
        return config

    def test_save_and_analyze_creates_csv(self, mock_config):
        """Should save results to CSV file."""
        mock_logger = MagicMock()
        manager = ResultsManager(mock_config, mock_logger)

        results = {
            "trial": [1, 2, 3],
            "bit_acc": [95.0, 96.0, 97.0],
            "adaptive_bit_acc": [97.0, 98.0, 99.0],
        }

        df = manager.save_and_analyze(results)

        # Check DataFrame was returned
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

        # Check CSV was created
        csv_files = list(mock_config.output_dir.glob("results_*.csv"))
        assert len(csv_files) == 1

    def test_perform_ttest_with_significant_difference(self, mock_config):
        """Should detect significant difference when p < 0.05."""
        mock_logger = MagicMock()
        manager = ResultsManager(mock_config, mock_logger)

        # Create data with significant difference
        results = {
            "trial": list(range(1, 31)),
            "bit_acc": [50.0] * 30,  # Low values
            "adaptive_bit_acc": [99.0] * 30,  # High values
        }

        manager.save_and_analyze(results)

        # Check that significant difference was logged
        log_messages = [str(call) for call in mock_logger.info.call_args_list]
        assert any("Significant difference" in msg for msg in log_messages)

    def test_perform_ttest_no_significant_difference(self, mock_config):
        """Should report no significant difference when p >= 0.05."""
        mock_logger = MagicMock()
        manager = ResultsManager(mock_config, mock_logger)

        # Create data with no significant difference
        results = {
            "trial": list(range(1, 31)),
            "bit_acc": [95.0 + i * 0.01 for i in range(30)],
            "adaptive_bit_acc": [95.0 + i * 0.01 for i in range(30)],
        }

        manager.save_and_analyze(results)

        # Check that no significant difference was logged
        log_messages = [str(call) for call in mock_logger.info.call_args_list]
        assert any("No significant difference" in msg for msg in log_messages)


class TestEmbedVideoClip:
    """Tests for embed_video_clip function."""

    def test_embed_video_clip_shape(self, videoseal_model):
        """Should return clip with same shape as input."""
        device = next(videoseal_model.parameters()).device
        F, H, W, C = 4, 64, 64, 3
        clip = np.random.randint(0, 255, (F, H, W, C), dtype=np.uint8)
        msgs = videoseal_model.get_random_msg().to(device)

        videoseal_model.to(device)
        result = embed_video_clip(videoseal_model, clip, msgs)

        assert result.shape == clip.shape
        assert result.dtype == np.uint8

    def test_embed_video_clip_different_values(self, videoseal_model):
        """Embedded clip should differ from original."""
        device = next(videoseal_model.parameters()).device
        F, H, W, C = 4, 64, 64, 3
        clip = np.random.randint(0, 255, (F, H, W, C), dtype=np.uint8)
        msgs = videoseal_model.get_random_msg().to(device)

        videoseal_model.to(device)
        result = embed_video_clip(videoseal_model, clip, msgs)

        # Result should be different from input (watermark added)
        assert not np.array_equal(result, clip)


class TestDetectVideoClip:
    """Tests for detect_video_clip function."""

    def test_detect_video_clip(self, videoseal_model):
        """Should return tensor with correct shape."""
        # Create a small test clip
        F, H, W, C = 4, 64, 64, 3
        clip = np.random.randint(0, 255, (F, H, W, C), dtype=np.uint8)

        result = detect_video_clip(videoseal_model, clip)

        # Should have shape (batch_size, nbits)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 2
        assert result.shape[0] == F


class TestTrialRunnerMethods:
    def test_get_bit_accuracy_threshold_default(self, videoseal_model):
        """Should return model threshold when no ecc/mac mode."""
        args = argparse.Namespace(
            ecc_mode=False,
            mac_mode=False,
            model="videoseal",
        )
        mock_config = MagicMock()
        mock_config.args = args
        mock_logger = MagicMock()

        from poc import TrialRunner

        runner = TrialRunner(
            model=videoseal_model,
            adaptive_model=MagicMock(),
            codec_manager=MagicMock(),
            config=mock_config,
            logger=mock_logger,
        )

        threshold = runner._get_bit_accuracy_threshold()

        # VideoSeal threshold is 0.0
        assert threshold == 0.0

    def test_get_bit_accuracy_threshold_ecc_mode(self, videoseal_model):
        """Should return 0.5 when ecc_mode enabled."""
        args = argparse.Namespace(
            ecc_mode=True,
            mac_mode=False,
            model="videoseal",
        )
        mock_config = MagicMock()
        mock_config.args = args
        mock_logger = MagicMock()

        from poc import TrialRunner

        runner = TrialRunner(
            model=videoseal_model,
            adaptive_model=MagicMock(),
            codec_manager=MagicMock(),
            config=mock_config,
            logger=mock_logger,
        )

        threshold = runner._get_bit_accuracy_threshold()

        assert threshold == 0.5

    def test_get_bit_accuracy_threshold_mac_mode(self, videoseal_model):
        """Should return 0.5 when mac_mode enabled."""
        args = argparse.Namespace(
            ecc_mode=False,
            mac_mode=True,
            model="videoseal",
        )
        mock_config = MagicMock()
        mock_config.args = args
        mock_logger = MagicMock()

        from poc import TrialRunner

        runner = TrialRunner(
            model=videoseal_model,
            adaptive_model=MagicMock(),
            codec_manager=MagicMock(),
            config=mock_config,
            logger=mock_logger,
        )

        threshold = runner._get_bit_accuracy_threshold()

        assert threshold == 0.5
