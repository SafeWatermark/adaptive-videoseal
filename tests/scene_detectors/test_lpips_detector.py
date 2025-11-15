import torch
from src.scene_detectors.lpips_detector import LpipsDetector


def test_key_frame_detection():
    # Prepare frames, each with shape (C,H,W)
    F, C, H, W = 10, 3, 256, 256
    detector = LpipsDetector()

    frames = torch.zeros((F, C, H, W))

    frames[3] += 0.5
    frames[7] += 1.0

    key_frame_ids = detector.detect_key_frames(frames)
    assert key_frame_ids.size() == (4,)
    assert torch.allclose(key_frame_ids, torch.tensor([3, 4, 7, 8]))


def test_zeros():
    """If all frames are identical, lpips shouldn't recognize key frame"""
    F, C, H, W = 10, 3, 256, 256
    frames = torch.zeros((F, C, H, W))
    detector = LpipsDetector()
    key_frame_ids = detector.detect_key_frames(frames)

    assert key_frame_ids.size() == (0,)
