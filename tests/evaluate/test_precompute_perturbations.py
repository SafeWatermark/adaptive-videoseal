"""
Tests for scripts/precompute_perturbations.py

This module tests the perturbation pre-computation script including:
- PerturbationConfig dataclass
- Perturbation configuration retrieval
- Perturbation creation and application
- Video processing pipeline
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from precompute_perturbations import (
    CPU_ONLY_PERTURBATIONS,
    GPU_CAPABLE_PERTURBATIONS,
    PerturbationConfig,
    apply_perturbation,
    create_perturbation,
    get_perturbation_configs,
    get_video_files,
    process_single_video,
    setup_seed,
)


class TestPerturbationConfig:
    """Tests for PerturbationConfig dataclass."""

    def test_create_config(self):
        """Test creating a basic perturbation config."""
        config = PerturbationConfig(
            name="JPEG_90", class_name="JPEG", params={"quality": 90}
        )
        assert config.name == "JPEG_90"
        assert config.class_name == "JPEG"
        assert config.params == {"quality": 90}
        assert config.get_dir_name() == "JPEG_90"


class TestGetPerturbationConfigs:
    """Tests for get_perturbation_configs function."""

    def test_returns_tuple(self):
        """Test that function returns a tuple of two lists."""
        result = get_perturbation_configs()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_non_mpeg4_configs(self):
        """Test non-MPEG4 perturbation configs."""
        non_mpeg4, _ = get_perturbation_configs()
        class_names = {c.class_name for c in non_mpeg4}

        # Should include these perturbation types
        assert "JPEG" in class_names
        assert "Gaussian" in class_names
        assert "GaussianBlur" in class_names
        assert "Crop" in class_names
        assert "FrameAverage" in class_names
        assert "FrameRemove" in class_names
        assert "FrameSwitch" in class_names

        # Should NOT include MPEG4
        assert "MPEG4" not in class_names

    def test_mpeg4_configs(self):
        """Test MPEG4 perturbation configs."""
        _, mpeg4 = get_perturbation_configs()

        # All should be MPEG4
        for config in mpeg4:
            assert config.class_name == "MPEG4"

        # Should have specific quality levels
        qualities = {c.params["quality"] for c in mpeg4}
        assert 23 in qualities
        assert 28 in qualities
        assert 33 in qualities

    def test_include_mpeg4_false(self):
        """Test excluding MPEG4 configs."""
        non_mpeg4, mpeg4 = get_perturbation_configs(include_mpeg4=False)

        assert len(mpeg4) == 0
        assert len(non_mpeg4) > 0

    def test_jpeg_quality_levels(self):
        """Test JPEG quality levels are correct."""
        non_mpeg4, _ = get_perturbation_configs()
        jpeg_configs = [c for c in non_mpeg4 if c.class_name == "JPEG"]

        qualities = {c.params["quality"] for c in jpeg_configs}
        assert qualities == {90, 75, 50}

    def test_gaussian_std_levels(self):
        """Test Gaussian noise std levels."""
        non_mpeg4, _ = get_perturbation_configs()
        gaussian_configs = [c for c in non_mpeg4 if c.class_name == "Gaussian"]

        stds = {c.params["std"] for c in gaussian_configs}
        assert stds == {0.01, 0.05}

    def test_crop_ratios(self):
        """Test crop ratio levels."""
        non_mpeg4, _ = get_perturbation_configs()
        crop_configs = [c for c in non_mpeg4 if c.class_name == "Crop"]

        ratios = {c.params["crop_ratio"] for c in crop_configs}
        assert ratios == {0.9, 0.8}

    def test_all_configs_have_required_fields(self):
        """Test all configs have required fields."""
        non_mpeg4, mpeg4 = get_perturbation_configs()
        all_configs = non_mpeg4 + mpeg4

        for config in all_configs:
            assert config.name is not None
            assert config.class_name is not None
            assert config.params is not None
            assert isinstance(config.params, dict)


class TestCreatePerturbation:
    """Tests for create_perturbation function."""

    def test_create_jpeg(self):
        """Test creating JPEG perturbation."""
        config = PerturbationConfig("JPEG_90", "JPEG", {"quality": 90})
        pert = create_perturbation(config)
        assert pert is not None
        assert hasattr(pert, "quality") or hasattr(pert, "__call__")

    def test_create_gaussian(self):
        """Test creating Gaussian perturbation."""
        config = PerturbationConfig("Gaussian_0.01", "Gaussian", {"std": 0.01})
        pert = create_perturbation(config)
        assert pert is not None

    def test_create_gaussian_blur(self):
        """Test creating GaussianBlur perturbation."""
        config = PerturbationConfig("GaussianBlur_1.0", "GaussianBlur", {"std": 1.0})
        pert = create_perturbation(config)
        assert pert is not None

    def test_create_crop(self):
        """Test creating Crop perturbation."""
        config = PerturbationConfig("Crop_0.9", "Crop", {"crop_ratio": 0.9})
        pert = create_perturbation(config)
        assert pert is not None

    def test_create_frame_average(self):
        """Test creating FrameAverage perturbation."""
        config = PerturbationConfig("FrameAverage_2", "FrameAverage", {"num_frames": 2})
        pert = create_perturbation(config)
        assert pert is not None

    def test_create_frame_remove(self):
        """Test creating FrameRemove perturbation."""
        config = PerturbationConfig("FrameRemove_0.1", "FrameRemove", {"p": 0.1})
        pert = create_perturbation(config)
        assert pert is not None

    def test_create_frame_switch(self):
        """Test creating FrameSwitch perturbation."""
        config = PerturbationConfig("FrameSwitch_0.1", "FrameSwitch", {"p": 0.1})
        pert = create_perturbation(config)
        assert pert is not None

    def test_create_mpeg4_with_temp_dir(self):
        """Test creating MPEG4 perturbation with temp directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = PerturbationConfig("MPEG4_23", "MPEG4", {"quality": 23})
            pert = create_perturbation(config, temp_dir=Path(temp_dir))
            assert pert is not None

    def test_create_unknown_raises_error(self):
        """Test that unknown perturbation raises ValueError."""
        config = PerturbationConfig("Unknown", "Unknown", {})
        with pytest.raises(ValueError, match="Unknown perturbation"):
            create_perturbation(config)


class TestApplyPerturbation:
    """Tests for apply_perturbation function."""

    @pytest.fixture
    def sample_video(self):
        """Create a sample video tensor."""
        # (T, C, H, W) format, values in [0, 1]
        return torch.rand(8, 3, 64, 64)

    def test_apply_jpeg(self, sample_video):
        """Test applying JPEG perturbation."""
        config = PerturbationConfig("JPEG_90", "JPEG", {"quality": 90})
        pert = create_perturbation(config)
        result = apply_perturbation(pert, sample_video, config)

        assert result.shape == sample_video.shape
        assert result.dtype == torch.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert not torch.allclose(result, sample_video)

    def test_apply_gaussian(self, sample_video):
        """Test applying Gaussian noise perturbation."""
        config = PerturbationConfig("Gaussian_0.01", "Gaussian", {"std": 0.01})
        pert = create_perturbation(config)

        result = apply_perturbation(pert, sample_video, config)

        assert result.shape == sample_video.shape
        # Result should be different from input (noise added)
        assert not torch.allclose(result, sample_video)

    def test_apply_gaussian_blur(self, sample_video):
        """Test applying GaussianBlur perturbation."""
        config = PerturbationConfig("GaussianBlur_1.0", "GaussianBlur", {"std": 1.0})
        pert = create_perturbation(config)
        result = apply_perturbation(pert, sample_video, config)

        assert result.shape == sample_video.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert not torch.allclose(result, sample_video)

    def test_apply_crop(self, sample_video):
        """Test applying Crop perturbation."""
        config = PerturbationConfig("Crop_0.9", "Crop", {"crop_ratio": 0.9})
        pert = create_perturbation(config)
        result = apply_perturbation(pert, sample_video, config)

        # Shape should be preserved (resize back to original)
        assert result.shape == sample_video.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert not torch.allclose(result, sample_video)

    def test_apply_frame_average_m1(self, sample_video):
        """Test FrameAverage with m=1 (should return clone)."""
        config = PerturbationConfig("FrameAverage_1", "FrameAverage", {"num_frames": 1})
        pert = create_perturbation(config)
        result = apply_perturbation(pert, sample_video, config)

        assert result.shape == sample_video.shape
        assert torch.allclose(result, sample_video)

    def test_apply_frame_average_m2(self, sample_video):
        """Test FrameAverage with m=2."""
        config = PerturbationConfig("FrameAverage_2", "FrameAverage", {"num_frames": 2})
        pert = create_perturbation(config)
        result = apply_perturbation(pert, sample_video, config)

        assert result.shape == sample_video.shape
        # Result should be different (averaged)
        assert not torch.allclose(result, sample_video)

    def test_apply_frame_average_m3(self, sample_video):
        """Test FrameAverage with m=3."""
        config = PerturbationConfig("FrameAverage_3", "FrameAverage", {"num_frames": 3})
        pert = create_perturbation(config)
        result = apply_perturbation(pert, sample_video, config)

        assert result.shape == sample_video.shape

    def test_apply_frame_remove(self, sample_video):
        """Test applying FrameRemove perturbation."""
        config = PerturbationConfig("FrameRemove_0.1", "FrameRemove", {"p": 0.1})
        pert = create_perturbation(config)

        result = apply_perturbation(pert, sample_video, config)

        # Frame count may be reduced
        assert result.shape[1:] == sample_video.shape[1:]  # C, H, W preserved
        assert result.shape[0] <= sample_video.shape[0]  # T may be reduced

    def test_apply_frame_switch(self, sample_video):
        """Test applying FrameSwitch perturbation."""
        config = PerturbationConfig("FrameSwitch_0.1", "FrameSwitch", {"p": 0.1})
        pert = create_perturbation(config)

        result = apply_perturbation(pert, sample_video, config)

        assert result.shape == sample_video.shape

    def test_gaussian_clamps_values(self):
        """Test that Gaussian noise clamps output values."""
        # Video with values at extremes
        video = torch.ones(4, 3, 32, 32)
        config = PerturbationConfig("Gaussian_0.5", "Gaussian", {"std": 0.5})
        pert = create_perturbation(config)

        result = apply_perturbation(pert, video, config)

        # Should be clamped to original range
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestProcessSingleVideo:
    """Tests for process_single_video function."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_video_file(self, temp_output_dir):
        """Create a sample video file for testing."""
        # Create a simple test video using cv2
        import cv2

        video_path = temp_output_dir / "test_video.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, 30, (64, 64))

        for i in range(10):
            frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            out.write(frame)

        out.release()
        return video_path

    def test_process_video_creates_files(self, sample_video_file, temp_output_dir):
        """Test that processing creates output files."""
        output_dir = temp_output_dir / "output"
        output_dir.mkdir()

        # Use minimal configs for speed
        configs = [
            PerturbationConfig("Gaussian_0.01", "Gaussian", {"std": 0.01}),
        ]

        video_name, success, total = process_single_video(
            sample_video_file, output_dir, configs, seed=42
        )

        assert video_name == "test_video"
        assert success == 1
        assert total == 1

        # Check file was created
        expected_file = output_dir / "Gaussian_0.01" / "test_video.pt"
        assert expected_file.exists()

    def test_process_video_file_content(self, sample_video_file, temp_output_dir):
        """Test that saved file has correct content."""
        output_dir = temp_output_dir / "output"
        output_dir.mkdir()

        configs = [
            PerturbationConfig("Gaussian_0.01", "Gaussian", {"std": 0.01}),
        ]

        process_single_video(sample_video_file, output_dir, configs, seed=42)

        # Load and verify saved data
        saved_file = output_dir / "Gaussian_0.01" / "test_video.pt"
        data = torch.load(saved_file)

        assert "video" in data
        assert "original_path" in data
        assert "perturbation" in data
        assert "params" in data

        assert data["perturbation"] == "Gaussian_0.01"
        assert data["params"] == {"std": 0.01}
        assert isinstance(data["video"], torch.Tensor)

    def test_process_video_skips_existing(self, sample_video_file, temp_output_dir):
        """Test that existing files are skipped."""
        output_dir = temp_output_dir / "output"
        output_dir.mkdir()

        configs = [
            PerturbationConfig("Gaussian_0.01", "Gaussian", {"std": 0.01}),
        ]

        # Process first time
        process_single_video(sample_video_file, output_dir, configs, seed=42)

        # Get modification time
        saved_file = output_dir / "Gaussian_0.01" / "test_video.pt"
        mtime_before = saved_file.stat().st_mtime

        # Process again
        video_name, success, total = process_single_video(
            sample_video_file, output_dir, configs, seed=42
        )

        # File should not be modified
        mtime_after = saved_file.stat().st_mtime
        assert mtime_before == mtime_after
        assert success == 1  # Still counts as success

    def test_process_video_with_max_frames(self, sample_video_file, temp_output_dir):
        """Test processing with max_frames limit."""
        output_dir = temp_output_dir / "output"
        output_dir.mkdir()

        configs = [
            PerturbationConfig("Gaussian_0.01", "Gaussian", {"std": 0.01}),
        ]

        process_single_video(
            sample_video_file, output_dir, configs, max_frames=5, seed=42
        )

        saved_file = output_dir / "Gaussian_0.01" / "test_video.pt"
        data = torch.load(saved_file)

        assert data["video"].shape[0] == 5

    def test_process_video_multiple_perturbations(
        self, sample_video_file, temp_output_dir
    ):
        """Test processing with multiple perturbations."""
        output_dir = temp_output_dir / "output"
        output_dir.mkdir()

        configs = [
            PerturbationConfig("Gaussian_0.01", "Gaussian", {"std": 0.01}),
            PerturbationConfig("GaussianBlur_1.0", "GaussianBlur", {"std": 1.0}),
        ]

        video_name, success, total = process_single_video(
            sample_video_file, output_dir, configs, seed=42
        )

        assert success == 2
        assert total == 2

        # Both files should exist
        assert (output_dir / "Gaussian_0.01" / "test_video.pt").exists()
        assert (output_dir / "GaussianBlur_1.0" / "test_video.pt").exists()


class TestGetVideoFiles:
    """Tests for get_video_files function."""

    def test_finds_mp4_files(self):
        """Test finding MP4 files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "video1.mp4").touch()
            (temp_path / "video2.mp4").touch()
            (temp_path / "not_video.txt").touch()

            files = get_video_files(temp_path)

            assert len(files) == 2
            assert all(f.suffix == ".mp4" for f in files)

    def test_finds_various_extensions(self):
        """Test finding various video extensions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            (temp_path / "video1.mp4").touch()
            (temp_path / "video2.avi").touch()
            (temp_path / "video3.mov").touch()
            (temp_path / "video4.mkv").touch()

            files = get_video_files(temp_path)

            assert len(files) == 4

    def test_finds_nested_files(self):
        """Test finding files in subdirectories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            (temp_path / "subdir1").mkdir()
            (temp_path / "subdir1" / "video1.mp4").touch()
            (temp_path / "subdir2").mkdir()
            (temp_path / "subdir2" / "video2.mp4").touch()

            files = get_video_files(temp_path)

            assert len(files) == 2

    def test_returns_sorted(self):
        """Test that results are sorted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            (temp_path / "c_video.mp4").touch()
            (temp_path / "a_video.mp4").touch()
            (temp_path / "b_video.mp4").touch()

            files = get_video_files(temp_path)

            names = [f.stem for f in files]
            assert names == sorted(names)

    def test_empty_directory(self):
        """Test handling of empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            files = get_video_files(Path(temp_dir))
            assert files == []

    def test_case_insensitive_extensions(self):
        """Test case-insensitive extension matching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            (temp_path / "video1.MP4").touch()
            (temp_path / "video2.Mp4").touch()

            files = get_video_files(temp_path)

            assert len(files) == 2


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_with_single_perturbation(self):
        """Test the full pipeline with a single perturbation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            H, W, C = 64, 64, 3
            F = 5

            # Create test video
            import cv2

            video_path = temp_path / "test.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, 30, (H, W))
            for _ in range(F):
                frame = np.random.randint(0, 255, (H, W, C), dtype=np.uint8)
                out.write(frame)
            out.release()

            # Setup output
            output_dir = temp_path / "output"
            output_dir.mkdir()

            # Get configs and process
            non_mpeg4, _ = get_perturbation_configs(include_mpeg4=False)
            # Use just first 2 for speed
            configs = non_mpeg4[:2]

            video_name, success, total = process_single_video(
                video_path, output_dir, configs, seed=42
            )

            assert success == total
            assert success == 2

            # Verify outputs
            for config in configs:
                output_file = output_dir / config.get_dir_name() / "test.pt"
                assert output_file.exists()

                data = torch.load(output_file)
                assert data["video"].shape[0] == F  # 5 frames
                assert data["video"].shape[1] == C  # RGB
                assert data["video"].shape[2] == 512  # Height (resized)
                assert data["video"].shape[3] == 512  # Width (resized)

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            H, W, C = 64, 64, 3
            F = 5

            # Create test video
            import cv2

            video_path = temp_path / "test.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, 30, (H, W))
            for _ in range(F):
                frame = np.full((H, W, C), 128, dtype=np.uint8)
                out.write(frame)
            out.release()

            # Process twice with same seed
            output_dir1 = temp_path / "output1"
            output_dir2 = temp_path / "output2"
            output_dir1.mkdir()
            output_dir2.mkdir()

            configs = [
                PerturbationConfig("Gaussian_0.05", "Gaussian", {"std": 0.05}),
            ]

            process_single_video(video_path, output_dir1, configs, seed=42)
            process_single_video(video_path, output_dir2, configs, seed=42)

            data1 = torch.load(output_dir1 / "Gaussian_0.05" / "test.pt")
            data2 = torch.load(output_dir2 / "Gaussian_0.05" / "test.pt")

            assert torch.allclose(data1["video"], data2["video"])


class TestPerturbationClassification:
    """Tests for GPU/CPU perturbation classification constants."""

    def test_gpu_capable_perturbations_defined(self):
        """Test that GPU_CAPABLE_PERTURBATIONS is defined correctly."""
        assert isinstance(GPU_CAPABLE_PERTURBATIONS, set)
        assert "Gaussian" in GPU_CAPABLE_PERTURBATIONS
        assert "GaussianBlur" in GPU_CAPABLE_PERTURBATIONS
        assert "FrameAverage" in GPU_CAPABLE_PERTURBATIONS
        assert "FrameRemove" in GPU_CAPABLE_PERTURBATIONS
        assert "FrameSwitch" in GPU_CAPABLE_PERTURBATIONS

    def test_cpu_only_perturbations_defined(self):
        """Test that CPU_ONLY_PERTURBATIONS is defined correctly."""
        assert isinstance(CPU_ONLY_PERTURBATIONS, set)
        assert "JPEG" in CPU_ONLY_PERTURBATIONS
        assert "Crop" in CPU_ONLY_PERTURBATIONS
        assert "MPEG4" in CPU_ONLY_PERTURBATIONS

    def test_no_overlap_between_gpu_and_cpu(self):
        """Test that GPU and CPU sets don't overlap."""
        overlap = GPU_CAPABLE_PERTURBATIONS & CPU_ONLY_PERTURBATIONS
        assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_all_perturbations_classified(self):
        """Test that all perturbation types are classified."""
        non_mpeg4, mpeg4 = get_perturbation_configs()
        all_configs = non_mpeg4 + mpeg4

        all_class_names = {c.class_name for c in all_configs}
        classified = GPU_CAPABLE_PERTURBATIONS | CPU_ONLY_PERTURBATIONS

        for class_name in all_class_names:
            assert class_name in classified, f"{class_name} not classified"


class TestProcessSingleVideoDevice:
    """Tests for process_single_video with device parameter."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_video_file(self, temp_output_dir):
        """Create a sample video file for testing."""
        import cv2

        H, W, C = 64, 64, 3
        F = 10
        video_path = temp_output_dir / "test_video.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, 30, (H, W))

        for i in range(F):
            frame = np.random.randint(0, 255, (H, W, C), dtype=np.uint8)
            out.write(frame)

        out.release()
        return video_path

    def test_process_video_with_cpu_device(self, sample_video_file, temp_output_dir):
        """Test processing with explicit CPU device."""
        output_dir = temp_output_dir / "output"
        output_dir.mkdir()

        configs = [
            PerturbationConfig("Gaussian_0.01", "Gaussian", {"std": 0.01}),
        ]

        video_name, success, total = process_single_video(
            sample_video_file, output_dir, configs, seed=42, device="cpu"
        )

        assert video_name == "test_video"
        assert success == 1
        assert total == 1

        # Check file was created
        expected_file = output_dir / "Gaussian_0.01" / "test_video.pt"
        assert expected_file.exists()

        # Verify saved tensor is on CPU
        data = torch.load(expected_file)
        assert data["video"].device == torch.device("cpu")

    def test_process_video_output_always_on_cpu(
        self, sample_video_file, temp_output_dir
    ):
        """Test that output is always saved to CPU regardless of device."""
        output_dir = temp_output_dir / "output"
        output_dir.mkdir()

        # Test with both GPU-capable and CPU-only perturbations
        configs = [
            PerturbationConfig(
                "Gaussian_0.01", "Gaussian", {"std": 0.01}
            ),  # GPU-capable
            PerturbationConfig("JPEG_90", "JPEG", {"quality": 90}),  # CPU-only
        ]

        process_single_video(
            sample_video_file, output_dir, configs, seed=42, device="cpu"
        )

        # Both should be saved to CPU
        for config in configs:
            saved_file = output_dir / config.get_dir_name() / "test_video.pt"
            data = torch.load(saved_file)
            assert data["video"].device == torch.device("cpu")

    def test_process_video_with_mixed_perturbations(
        self, sample_video_file, temp_output_dir
    ):
        """Test processing with both GPU-capable and CPU-only perturbations."""
        output_dir = temp_output_dir / "output"
        output_dir.mkdir()

        configs = [
            PerturbationConfig(
                "Gaussian_0.01", "Gaussian", {"std": 0.01}
            ),  # GPU-capable
            PerturbationConfig(
                "GaussianBlur_1.0", "GaussianBlur", {"std": 1.0}
            ),  # GPU-capable
            PerturbationConfig("JPEG_90", "JPEG", {"quality": 90}),  # CPU-only
            PerturbationConfig("Crop_0.9", "Crop", {"crop_ratio": 0.9}),  # CPU-only
        ]

        video_name, success, total = process_single_video(
            sample_video_file, output_dir, configs, seed=42, device="cpu"
        )

        assert success == 4
        assert total == 4

        # All files should exist
        for config in configs:
            saved_file = output_dir / config.get_dir_name() / "test_video.pt"
            assert saved_file.exists()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_process_video_with_cuda_device(self, sample_video_file, temp_output_dir):
        """Test processing with CUDA device (only runs if CUDA available)."""
        output_dir = temp_output_dir / "output"
        output_dir.mkdir()

        configs = [
            PerturbationConfig("Gaussian_0.01", "Gaussian", {"std": 0.01}),
        ]

        video_name, success, total = process_single_video(
            sample_video_file, output_dir, configs, seed=42, device="cuda"
        )

        assert success == 1

        # Output should still be on CPU (saved to disk)
        saved_file = output_dir / "Gaussian_0.01" / "test_video.pt"
        data = torch.load(saved_file)
        assert data["video"].device == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCPUGPUConsistency:
    """Tests for CPU vs GPU result consistency."""

    @pytest.fixture
    def sample_video(self):
        """Create a sample video tensor."""
        # (T, C, H, W) format, values in [0, 1]
        setup_seed(42)
        return torch.rand(8, 3, 64, 64)

    def test_gaussian_blur_cpu_gpu_identical(self, sample_video):
        """Test GaussianBlur produces identical results on CPU and GPU."""
        config = PerturbationConfig("GaussianBlur_1.0", "GaussianBlur", {"std": 1.0})

        # CPU
        pert_cpu = create_perturbation(config)
        result_cpu = apply_perturbation(pert_cpu, sample_video, config)

        # GPU
        video_gpu = sample_video.cuda()
        pert_gpu = create_perturbation(config)
        result_gpu = apply_perturbation(pert_gpu, video_gpu, config)
        result_gpu_cpu = result_gpu.cpu()

        # Should be identical (deterministic operation)
        assert torch.allclose(result_cpu, result_gpu_cpu, atol=1e-5)

    def test_frame_average_cpu_gpu_identical(self, sample_video):
        """Test FrameAverage produces identical results on CPU and GPU."""
        config = PerturbationConfig("FrameAverage_2", "FrameAverage", {"num_frames": 2})

        # CPU
        pert_cpu = create_perturbation(config)
        result_cpu = apply_perturbation(pert_cpu, sample_video, config)

        # GPU
        video_gpu = sample_video.cuda()
        pert_gpu = create_perturbation(config)
        result_gpu = apply_perturbation(pert_gpu, video_gpu, config)
        result_gpu_cpu = result_gpu.cpu()

        # Should be identical (deterministic operation)
        assert torch.allclose(result_cpu, result_gpu_cpu, atol=1e-5)

    def test_frame_remove_cpu_gpu_identical(self, sample_video):
        """Test FrameRemove produces identical results on CPU and GPU with same seed."""
        config = PerturbationConfig("FrameRemove_0.1", "FrameRemove", {"p": 0.1})

        # CPU
        setup_seed(42)
        pert_cpu = create_perturbation(config)
        result_cpu = apply_perturbation(pert_cpu, sample_video, config)

        # GPU
        setup_seed(42)
        video_gpu = sample_video.cuda()
        pert_gpu = create_perturbation(config)
        result_gpu = apply_perturbation(pert_gpu, video_gpu, config)
        result_gpu_cpu = result_gpu.cpu()

        # Should have same shape and values
        assert result_cpu.shape == result_gpu_cpu.shape
        assert torch.allclose(result_cpu, result_gpu_cpu, atol=1e-5)

    def test_frame_switch_cpu_gpu_identical(self, sample_video):
        """Test FrameSwitch produces identical results on CPU and GPU with same seed."""
        config = PerturbationConfig("FrameSwitch_0.1", "FrameSwitch", {"p": 0.1})

        # CPU
        setup_seed(42)
        pert_cpu = create_perturbation(config)
        result_cpu = apply_perturbation(pert_cpu, sample_video, config)

        # GPU
        setup_seed(42)
        video_gpu = sample_video.cuda()
        pert_gpu = create_perturbation(config)
        result_gpu = apply_perturbation(pert_gpu, video_gpu, config)
        result_gpu_cpu = result_gpu.cpu()

        # Should be identical
        assert torch.allclose(result_cpu, result_gpu_cpu, atol=1e-5)

    def test_gaussian_cpu_gpu_different_but_valid(self, sample_video):
        """Test Gaussian noise differs between CPU and GPU (expected behavior).

        Gaussian noise uses different RNGs on CPU vs GPU, so results will differ
        but should have similar statistical properties.
        """
        config = PerturbationConfig("Gaussian_0.05", "Gaussian", {"std": 0.05})

        # CPU
        setup_seed(42)
        pert_cpu = create_perturbation(config)
        result_cpu = apply_perturbation(pert_cpu, sample_video, config)

        # GPU
        setup_seed(42)
        video_gpu = sample_video.cuda()
        pert_gpu = create_perturbation(config)
        result_gpu = apply_perturbation(pert_gpu, video_gpu, config)
        result_gpu_cpu = result_gpu.cpu()

        # Results will differ (different RNGs)
        # But both should be valid (in range, correct shape)
        assert result_cpu.shape == result_gpu_cpu.shape
        assert result_cpu.min() >= 0.0
        assert result_cpu.max() <= 1.0
        assert result_gpu_cpu.min() >= 0.0
        assert result_gpu_cpu.max() <= 1.0

        # Compute difference - expected to be non-zero
        diff = torch.abs(result_cpu - result_gpu_cpu)
        # Max diff should be roughly proportional to noise std
        # For std=0.05, max diff could be up to ~0.4 (multiple std deviations)
        assert diff.max() > 0  # Confirm they're different
        assert diff.max() < 1.0  # But not completely different

    def test_gaussian_statistical_properties_similar(self, sample_video):
        """Test that Gaussian noise has similar statistical properties on CPU and GPU."""
        config = PerturbationConfig("Gaussian_0.05", "Gaussian", {"std": 0.05})
        std = 0.05

        # CPU
        setup_seed(42)
        pert_cpu = create_perturbation(config)
        result_cpu = apply_perturbation(pert_cpu, sample_video, config)
        noise_cpu = result_cpu - sample_video

        # GPU
        setup_seed(42)
        video_gpu = sample_video.cuda()
        pert_gpu = create_perturbation(config)
        result_gpu = apply_perturbation(pert_gpu, video_gpu, config)
        noise_gpu = (result_gpu - video_gpu).cpu()

        # Both should have similar mean (close to 0) and std (close to configured std)
        # Note: clamping affects the distribution at boundaries
        assert abs(noise_cpu.mean().item()) < std  # Mean should be near 0
        assert abs(noise_gpu.mean().item()) < std

        # Std should be close to configured value (may be slightly less due to clamping)
        assert abs(noise_cpu.std().item() - std) < std * 0.5
        assert abs(noise_gpu.std().item() - std) < std * 0.5

    def test_all_deterministic_perturbations_match(self, sample_video):
        """Test all deterministic GPU-capable perturbations match between CPU and GPU."""
        deterministic_configs = [
            PerturbationConfig("GaussianBlur_1.0", "GaussianBlur", {"std": 1.0}),
            PerturbationConfig("GaussianBlur_2.0", "GaussianBlur", {"std": 2.0}),
            PerturbationConfig("FrameAverage_2", "FrameAverage", {"num_frames": 2}),
            PerturbationConfig("FrameAverage_3", "FrameAverage", {"num_frames": 3}),
        ]

        video_gpu = sample_video.cuda()

        for config in deterministic_configs:
            # CPU
            pert_cpu = create_perturbation(config)
            result_cpu = apply_perturbation(pert_cpu, sample_video, config)

            # GPU
            pert_gpu = create_perturbation(config)
            result_gpu = apply_perturbation(pert_gpu, video_gpu, config)
            result_gpu_cpu = result_gpu.cpu()

            assert torch.allclose(
                result_cpu, result_gpu_cpu, atol=1e-5
            ), f"{config.name} results differ between CPU and GPU"

    def test_gpu_speedup_exists_for_frame_operations(self, sample_video):
        """Test that GPU provides speedup for frame operations (smoke test)."""
        import time

        # Use larger video for meaningful timing
        large_video = torch.rand(32, 3, 256, 256)
        large_video_gpu = large_video.cuda()

        config = PerturbationConfig("FrameAverage_2", "FrameAverage", {"num_frames": 2})

        # CPU timing
        pert_cpu = create_perturbation(config)
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = apply_perturbation(pert_cpu, large_video, config)
        cpu_time = time.perf_counter() - start

        # GPU timing
        pert_gpu = create_perturbation(config)
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = apply_perturbation(pert_gpu, large_video_gpu, config)
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start

        # GPU should be faster (or at least not much slower)
        # Using a loose threshold since CI environments may vary
        assert (
            gpu_time < cpu_time * 2
        ), f"GPU ({gpu_time:.3f}s) much slower than CPU ({cpu_time:.3f}s)"

    def test_process_single_video_cpu_gpu_output_equivalent(self):
        """Test that process_single_video produces equivalent outputs for deterministic perturbations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test video
            import cv2

            H, W, C = 64, 64, 3
            F = 8
            video_path = temp_path / "test.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, 30, (H, W))
            for _ in range(F):
                frame = np.full((H, W, C), 128, dtype=np.uint8)
                out.write(frame)
            out.release()

            # Process with CPU
            output_cpu = temp_path / "output_cpu"
            output_cpu.mkdir()
            configs = [
                PerturbationConfig("GaussianBlur_1.0", "GaussianBlur", {"std": 1.0}),
                PerturbationConfig("FrameAverage_2", "FrameAverage", {"num_frames": 2}),
            ]
            process_single_video(video_path, output_cpu, configs, seed=42, device="cpu")

            # Process with GPU
            output_gpu = temp_path / "output_gpu"
            output_gpu.mkdir()
            process_single_video(
                video_path, output_gpu, configs, seed=42, device="cuda"
            )

            # Compare outputs
            for config in configs:
                cpu_file = output_cpu / config.get_dir_name() / "test.pt"
                gpu_file = output_gpu / config.get_dir_name() / "test.pt"

                data_cpu = torch.load(cpu_file)
                data_gpu = torch.load(gpu_file)

                assert torch.allclose(
                    data_cpu["video"], data_gpu["video"], atol=1e-5
                ), f"{config.name} outputs differ between CPU and GPU processing"
