import torch
from src.scene_detectors.hash_detector import HashDetector


def test_calculate_adjacent_frame_score_zero_diff():
    """If all frames are identical, the score should be zero."""
    F, C, H, W = 5, 3, 8, 8
    frames = torch.zeros((F, C, H, W))
    detector = HashDetector()
    hash_frames = detector.calculate_hash_frames(
        frames=frames, hash_size=detector.size, factor=detector.factor
    )

    # All hashes should be identical
    hash_dist = torch.count_nonzero(
        hash_frames[:-1].flatten(start_dim=1) != hash_frames[1:].flatten(start_dim=1),
        dim=-1,
    )

    assert hash_frames.size() == (F, 1, detector.size, detector.size)
    assert torch.all(hash_dist == 0)


def test_key_frame_detection():
    # Prepare frames, each with shape (C,H,W)
    F, C, H, W = 10, 3, 256, 256
    detector = HashDetector()

    frames = torch.zeros((F, C, H, W))

    frames[3] += 0.5  # brightness jump
    frames[7] += 1.0

    key_frame_ids = detector.detect_key_frames(frames)
    assert key_frame_ids.size() == (4,)
    assert torch.allclose(key_frame_ids, torch.tensor([3, 4, 7, 8]))
