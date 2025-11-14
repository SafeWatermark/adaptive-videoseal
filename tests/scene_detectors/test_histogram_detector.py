import torch
from src.scene_detectors.histogram_detector import HistogramDetector


def test_calculate_norm_hist_zero():
    F, C, H, W = 5, 3, 8, 8
    frames = torch.zeros((F, C, H, W))
    detector = HistogramDetector()

    normalized_hist = detector.calculate_histogram(
        frames=frames, bins=detector.bins, normalize=True
    )

    each_item = normalized_hist[0][0]
    # check similar of first bin
    assert torch.allclose(normalized_hist[..., 0], each_item)

    # other bins = 0.
    assert torch.allclose(normalized_hist[..., 1:], torch.tensor([0.0]))


def test_calculate_norm_hist_random():
    F, C, H, W = 5, 3, 128, 128
    frames = torch.randint(low=0, high=256, size=(F, C, H, W), dtype=torch.uint8)
    detector = HistogramDetector()
    normalized_hist = detector.calculate_histogram(
        frames=frames, bins=detector.bins, normalize=True
    )

    # Check normalization (L2 norm ≈ 1 for each histogram)
    norms = torch.norm(normalized_hist, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_key_frame_detection():
    # Prepare frames, each with shape (C,H,W)
    F, C, H, W = 10, 3, 256, 256
    detector = HistogramDetector()

    frames = torch.zeros((F, C, H, W))

    frames[3] += 0.5  # brightness jump
    frames[7] += 1.0

    key_frame_ids = detector.detect_key_frames(frames)
    assert key_frame_ids.size() == (4,)
    assert torch.allclose(key_frame_ids, torch.tensor([3, 4, 7, 8]))
