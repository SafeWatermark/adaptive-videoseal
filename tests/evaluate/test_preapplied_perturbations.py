"""
Tests for pre-applied perturbations evaluation system.

Tests the apply_perturbations.py script and evaluate_with_preapplied_perturbations function.
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock

from evaluate.apply_perturbations import (
    get_perturbation_name,
    save_video_tensor,
    apply_perturbations_to_dataset,
)
from evaluate.perturbed_dataset import PerturbedVideoDataset
from evaluate.common_evaluator import (
    evaluate_with_preapplied_perturbations,
    get_common_perturbations,
    _get_perturbation_params,
)


class TestGetPerturbationName:
    """Test perturbation name generation."""

    def test_jpeg_perturbation_name(self):
        """Test JPEG perturbation naming."""
        # Mock JPEG perturbation
        mock_pert = MagicMock()
        mock_pert.__class__.__name__ = "JPEG"
        mock_pert.quality = 75

        name = get_perturbation_name(mock_pert)
        assert "JPEG" in name
        assert "quality=75" in name

    def test_gaussian_perturbation_name(self):
        """Test Gaussian perturbation naming."""
        mock_pert = MagicMock()
        mock_pert.__class__.__name__ = "Gaussian"
        mock_pert.std = 0.05

        name = get_perturbation_name(mock_pert)
        assert "Gaussian" in name
        assert "std=0.05" in name

    def test_wrapper_suffix_removed(self):
        """Test that 'Wrapper' suffix is removed from name."""
        mock_pert = MagicMock()
        mock_pert.__class__.__name__ = "MPEG4Wrapper"
        mock_pert.quality = 23

        name = get_perturbation_name(mock_pert)
        assert "MPEG4" in name
        assert "Wrapper" not in name


class TestSaveVideoTensor:
    """Test video tensor saving."""

    def test_save_and_load_tensor(self):
        """Test saving and loading video tensor."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            video = torch.rand(10, 3, 64, 64)
            output_path = Path(tmp_dir) / "test_video.pt"

            save_video_tensor(video, output_path)

            assert output_path.exists()

            loaded = torch.load(output_path, weights_only=True)
            assert torch.allclose(video, loaded)

    def test_creates_parent_directories(self):
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            video = torch.rand(5, 3, 32, 32)
            output_path = Path(tmp_dir) / "nested" / "dir" / "video.pt"

            save_video_tensor(video, output_path)

            assert output_path.exists()


class TestPerturbedVideoDataset:
    """Test PerturbedVideoDataset loader."""

    @pytest.fixture
    def mock_perturbed_dir(self):
        """Create a mock perturbed video directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create metadata
            metadata = {
                "perturbations": [
                    {
                        "name": "JPEG_quality=75",
                        "class": "JPEG",
                        "params": {"quality": 75},
                    },
                    {
                        "name": "Gaussian_std=0.05",
                        "class": "Gaussian",
                        "params": {"std": 0.05},
                    },
                ],
                "videos": [
                    {"name": "video1.mp4", "safe_name": "video1.mp4", "num_frames": 10},
                    {"name": "video2.mp4", "safe_name": "video2.mp4", "num_frames": 15},
                ],
                "save_format": "pt",
            }

            with open(tmp_path / "metadata.json", "w") as f:
                json.dump(metadata, f)

            # Create video tensor files
            for pert in metadata["perturbations"]:
                pert_dir = tmp_path / pert["name"]
                pert_dir.mkdir()
                for video in metadata["videos"]:
                    video_tensor = torch.rand(video["num_frames"], 3, 64, 64)
                    torch.save(video_tensor, pert_dir / f"{video['safe_name']}.pt")

            yield tmp_path

    def test_load_metadata(self, mock_perturbed_dir):
        """Test loading metadata from perturbed directory."""
        dataset = PerturbedVideoDataset(mock_perturbed_dir)

        assert len(dataset.get_perturbations()) == 2
        assert len(dataset.get_video_names()) == 2
        assert dataset.save_format == "pt"

    def test_get_perturbation_names(self, mock_perturbed_dir):
        """Test getting perturbation names."""
        dataset = PerturbedVideoDataset(mock_perturbed_dir)

        names = dataset.get_perturbation_names()
        assert "JPEG_quality=75" in names
        assert "Gaussian_std=0.05" in names

    def test_load_perturbed_video(self, mock_perturbed_dir):
        """Test loading a specific perturbed video."""
        dataset = PerturbedVideoDataset(mock_perturbed_dir)

        video = dataset.load_perturbed_video("video1.mp4", "JPEG_quality=75")

        assert video.shape[0] == 10  # num_frames
        assert video.shape[1] == 3  # channels
        assert video.shape[2] == 64  # height
        assert video.shape[3] == 64  # width

    def test_iterate_perturbation(self, mock_perturbed_dir):
        """Test iterating over videos for a perturbation."""
        dataset = PerturbedVideoDataset(mock_perturbed_dir)

        videos = list(dataset.iterate_perturbation("JPEG_quality=75"))

        assert len(videos) == 2
        for video, meta in videos:
            assert video.dim() == 4
            assert "name" in meta

    def test_iterate_video(self, mock_perturbed_dir):
        """Test iterating over perturbations for a video."""
        dataset = PerturbedVideoDataset(mock_perturbed_dir)

        perturbations = list(dataset.iterate_video("video1.mp4"))

        assert len(perturbations) == 2
        for video, pert_name, pert_info in perturbations:
            assert video.dim() == 4
            assert isinstance(pert_name, str)
            assert "class" in pert_info

    def test_missing_metadata_raises_error(self):
        """Test that missing metadata file raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(FileNotFoundError):
                PerturbedVideoDataset(Path(tmp_dir))


class TestApplyPerturbationsToDataset:
    """Test apply_perturbations_to_dataset function."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset iterator."""
        videos = [
            (torch.rand(10, 3, 64, 64), {"name": "test_video1.mp4"}),
            (torch.rand(8, 3, 64, 64), {"name": "test_video2.mp4"}),
        ]
        return iter(videos)

    def test_apply_single_perturbation(self, mock_dataset):
        """Test applying a single perturbation type."""
        import logging

        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)

            # Only apply JPEG perturbations
            metadata = apply_perturbations_to_dataset(
                dataset=mock_dataset,
                output_dir=output_dir,
                logger=logger,
                max_videos=2,
                save_format="pt",
                selected_perturbations=["JPEG"],
            )

            assert len(metadata["videos"]) == 2
            # Should only have JPEG perturbations (3 quality levels)
            assert all("JPEG" in p["class"] for p in metadata["perturbations"])

            # Check files were created
            assert (output_dir / "metadata.json").exists()

    def test_max_videos_limit(self):
        """Test that max_videos limits processing."""
        import logging

        logger = logging.getLogger(__name__)

        videos = [
            (torch.rand(5, 3, 32, 32), {"name": f"video{i}.mp4"}) for i in range(10)
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)

            metadata = apply_perturbations_to_dataset(
                dataset=iter(videos),
                output_dir=output_dir,
                logger=logger,
                max_videos=3,
                save_format="pt",
                selected_perturbations=[
                    "JPEG_quality=90"
                ],  # Single perturbation for speed
            )

            assert len(metadata["videos"]) == 3


class TestEvaluateWithPreappliedPerturbations:
    """Test evaluate_with_preapplied_perturbations function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock watermarking model."""
        model = MagicMock()

        # Mock embed method
        def mock_embed(video):
            return {
                "watermarked_video": video + 0.01,
                "message": torch.randint(0, 2, (96,)).float(),
            }

        model.embed = mock_embed

        # Mock extract method - returns logits
        def mock_extract(video):
            T = video.shape[0]
            return torch.randn(T, 96)

        model.extract = mock_extract

        return model

    @pytest.fixture
    def mock_perturbed_dataset(self):
        """Create a mock perturbed dataset."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create minimal metadata
            metadata = {
                "perturbations": [
                    {
                        "name": "JPEG_quality=75",
                        "class": "JPEG",
                        "params": {"quality": 75},
                    },
                ],
                "videos": [
                    {"name": "video1.mp4", "safe_name": "video1.mp4", "num_frames": 10},
                ],
                "save_format": "pt",
            }

            with open(tmp_path / "metadata.json", "w") as f:
                json.dump(metadata, f)

            # Create video tensor file
            pert_dir = tmp_path / "JPEG_quality=75"
            pert_dir.mkdir()
            video_tensor = torch.rand(10, 3, 64, 64)
            torch.save(video_tensor, pert_dir / "video1.mp4.pt")

            yield PerturbedVideoDataset(tmp_path)

    @pytest.fixture
    def mock_original_dataset(self):
        """Create a mock original dataset iterator."""
        videos = [
            (torch.rand(10, 3, 64, 64), {"name": "video1.mp4"}),
        ]
        return iter(videos)

    def test_evaluation_runs(
        self, mock_model, mock_perturbed_dataset, mock_original_dataset
    ):
        """Test that evaluation runs without errors."""
        import logging

        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)

            df = evaluate_with_preapplied_perturbations(
                model=mock_model,
                perturbed_dataset=mock_perturbed_dataset,
                original_dataset=mock_original_dataset,
                output_dir=output_dir,
                logger=logger,
                evaluate_forgery=False,
            )

            assert len(df) > 0
            assert "video" in df.columns
            assert "perturbation" in df.columns
            assert "detected" in df.columns
            assert "attack_type" in df.columns

    def test_results_saved_to_csv(
        self, mock_model, mock_perturbed_dataset, mock_original_dataset
    ):
        """Test that results are saved to CSV."""
        import logging

        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)

            evaluate_with_preapplied_perturbations(
                model=mock_model,
                perturbed_dataset=mock_perturbed_dataset,
                original_dataset=mock_original_dataset,
                output_dir=output_dir,
                logger=logger,
                evaluate_forgery=False,
            )

            assert (output_dir / "common_perturbations_results.csv").exists()
            assert (output_dir / "common_perturbations_stats.json").exists()


class TestGetCommonPerturbations:
    """Test get_common_perturbations function."""

    def test_returns_list_of_perturbations(self):
        """Test that function returns a list of perturbations."""
        perturbations = get_common_perturbations()

        assert isinstance(perturbations, list)
        assert len(perturbations) > 0

    def test_perturbations_are_modules(self):
        """Test that all perturbations are nn.Module or callable."""
        perturbations = get_common_perturbations()

        for pert in perturbations:
            assert hasattr(pert, "__call__") or hasattr(pert, "forward")

    def test_all_perturbation_types_included(self):
        """Test that all expected perturbation types are included."""
        perturbations = get_common_perturbations()

        pert_classes = [type(p).__name__.replace("Wrapper", "") for p in perturbations]

        expected_types = [
            "JPEG",
            "Gaussian",
            "GaussianBlur",
            "Crop",
            "MPEG4",
            "FrameAverage",
            "FrameRemove",
            "FrameSwitch",
        ]

        for expected in expected_types:
            assert any(expected in cls for cls in pert_classes), f"Missing {expected}"


class TestGetPerturbationParams:
    """Test _get_perturbation_params function."""

    def test_extracts_quality_param(self):
        """Test extracting quality parameter."""
        mock_pert = MagicMock()
        mock_pert.quality = 75

        params = _get_perturbation_params(mock_pert)

        assert "quality" in params
        assert params["quality"] == 75

    def test_extracts_multiple_params(self):
        """Test extracting multiple parameters."""
        mock_pert = MagicMock()
        mock_pert.quality = 23
        mock_pert.codec = "libx264"

        params = _get_perturbation_params(mock_pert)

        assert "quality" in params
        assert "codec" in params
        assert params["quality"] == 23
        assert params["codec"] == "libx264"

    def test_missing_params_not_included(self):
        """Test that missing parameters are not included."""
        mock_pert = MagicMock(spec=[])  # Empty spec, no attributes

        params = _get_perturbation_params(mock_pert)

        assert len(params) == 0
