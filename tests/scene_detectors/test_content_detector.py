import torch
from src.scene_detectors.content_detector import (
    _mean_pixel_distance,
    ContentDetector,
    ComponentWeights,
)


def test_identical_images():
    """Distance between identical images should be zero."""
    x = torch.randn(4, 3, 16, 16)
    y = x.clone()
    d = _mean_pixel_distance(x, y)
    assert torch.allclose(
        d, torch.tensor([0.0, 0.0, 0.0, 0.0])
    ), f"Expected zeros, got {d}"


def test_constant_offset():
    """Constant pixel difference should yield the same RMSE value."""
    x = torch.zeros(2, 1, 4, 4)
    y = torch.ones(2, 1, 4, 4)
    d = _mean_pixel_distance(x, y)
    # For difference of 1 everywhere, RMSE = 1
    assert torch.allclose(d, torch.tensor([1.0, 1.0])), f"Expected ones, got {d}"


def test_calculate_adjacent_frame_score_zero_diff():
    """If all frames are identical, the score should be zero."""
    F, C, H, W = 5, 3, 8, 8
    detector = ContentDetector()

    frames = torch.zeros(F, C, H, W)
    score = detector._calculate_adjacent_frame_score(frames)
    assert torch.allclose(score, torch.tensor([0.0, 0.0, 0.0, 0.0]))


def test_key_frame_detection():
    # Prepare frames, each with shape (C,H,W)
    F, C, H, W = 10, 3, 4, 4
    detector = ContentDetector()

    frames = torch.zeros((F, C, H, W))

    frames[3] += 0.5  # brightness jump
    frames[7] += 1.0

    key_frame_ids = detector.detect_key_frames(frames)

    assert key_frame_ids.size() == (4,)
    assert torch.allclose(key_frame_ids, torch.tensor([3, 4, 7, 8]))


def test_threshold_and_weights_behavior():
    """Changing weights should change normalization behavior."""
    delta_hue, delta_sat, delta_lum = 2.0, 0.5, 0.5

    weights = ComponentWeights(
        delta_hue=delta_hue, delta_sat=delta_sat, delta_lum=delta_lum
    )
    det = ContentDetector(weights=weights)
    assert det.normalizer == sum([delta_hue, delta_sat, delta_lum])
