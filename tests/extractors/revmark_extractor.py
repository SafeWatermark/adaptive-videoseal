import pytest
import torch
from videoseal.models.extractor import Extractor

from models.revmark import (
    REVMARK_DEFAULT_MSG_BITS,
    REVMARK_FRAME_SIZE,
)


@pytest.mark.parametrize(
    "F, expected_indices, expected_positions",
    [
        (1, [0, 0, 0, 0, 0, 0, 0, 0], [0]),
        (2, [0, 0, 0, 0, 1, 1, 1, 1], [0, 4]),
        (3, [0, 0, 0, 1, 1, 1, 2, 2], [0, 3, 6]),
        (4, [0, 0, 1, 1, 2, 2, 3, 3], [0, 2, 4, 6]),
        (5, [0, 0, 1, 1, 2, 3, 3, 4], [0, 2, 4, 5, 7]),
        (6, [0, 0, 1, 2, 3, 3, 4, 5], [0, 2, 3, 4, 6, 7]),
        (7, [0, 0, 1, 2, 3, 4, 5, 6], [0, 2, 3, 4, 5, 6, 7]),
        (8, [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]),
    ],
)
def test_temporal_pad(
    default_revmark_extractor: Extractor,
    F: int,
    expected_indices: list[int],
    expected_positions: list[int],
) -> None:
    """Test temporal padding matches expected nearest-neighbor indices."""
    _, revmark_h, revmark_w = default_revmark_extractor.video_shape
    C = 3

    chunk = torch.randn((F, C, revmark_h, revmark_w))
    padded, original_positions = default_revmark_extractor._temporal_pad(frames=chunk)

    assert padded.size() == (REVMARK_FRAME_SIZE, C, revmark_h, revmark_w)
    assert original_positions == expected_positions

    # Verify each padded frame matches the expected source frame
    for pad_id, src_id in enumerate(expected_indices):
        assert torch.allclose(padded[pad_id], chunk[src_id])


def test_revmark_preprocess(default_revmark_extractor: Extractor) -> None:
    """Test preprocessing converts [0,1] to [-1,1] and resizes."""
    F, H, W = default_revmark_extractor.video_shape
    C = 3
    imgs = torch.zeros(F, C, 2 * H, 2 * W) + 0.5  # 0.5 in [0,1]
    processed, meta = default_revmark_extractor._preprocess(imgs)

    # 0.5 in [0,1] -> 0.0 in [-1,1]
    assert torch.allclose(processed, torch.zeros_like(processed), atol=1e-6)
    assert processed.size() == (F, C, H, W)
    assert meta["orig_size"] == (2 * H, 2 * W)


def test_revmark_preprocess_no_resize(default_revmark_extractor: Extractor) -> None:
    """Test preprocessing without resize when size matches."""
    F, H, W = default_revmark_extractor.video_shape
    C = 3
    imgs = torch.ones(F, C, H, W) * 0.0  # 0.0 in [0,1]
    processed, meta = default_revmark_extractor._preprocess(imgs)

    # 0.0 in [0,1] -> -1.0 in [-1,1]
    expected = torch.ones_like(processed) * (-1.0)
    assert torch.allclose(processed, expected, atol=1e-6)
    assert meta["orig_size"] == (H, W)


def test_postprocess_adds_dummy_probability(
    default_revmark_extractor: Extractor,
) -> None:
    """Test postprocess prepends dummy probability column."""
    B = 4
    nbits = default_revmark_extractor.msg_bits
    msgs = torch.randn(B, nbits)

    result = default_revmark_extractor._postprocess(msgs)

    assert result.size() == (B, 1 + nbits)
    # First column should be zeros (dummy probability)
    assert torch.allclose(result[:, 0], torch.zeros(B))
    # Remaining columns should match original messages
    assert torch.allclose(result[:, 1:], msgs)


def test_revmark_forward_size(default_revmark_extractor: Extractor) -> None:
    """Test forward output shape matches expected format."""
    F, C, H, W = 16, 3, 256, 256
    imgs = torch.randn(F, C, H, W).clamp_(0.0, 1.0)

    result = default_revmark_extractor(imgs)

    # Output should be (F, 1 + msg_bits)
    assert result.shape == (F, 1 + REVMARK_DEFAULT_MSG_BITS)


@pytest.mark.parametrize("F", [1, 2, 3, 4, 5, 6, 7])
def test_revmark_forward_less_than_defined_frames(
    default_revmark_extractor: Extractor, F: int
) -> None:
    """Test forward works with fewer frames than REVMARK_FRAME_SIZE."""
    C, H, W = 3, 256, 256

    imgs = torch.randn(F, C, H, W).clamp_(0.0, 1.0)

    result = default_revmark_extractor(imgs)

    assert result.shape == (F, 1 + default_revmark_extractor.msg_bits)


def test_revmark_forward_multiple_chunks(default_revmark_extractor: Extractor) -> None:
    """Test forward correctly handles multiple 8-frame chunks."""
    F, C, H, W = 20, 3, 256, 256  # 2 full chunks + 4 remaining
    imgs = torch.randn(F, C, H, W).clamp_(0.0, 1.0)

    result = default_revmark_extractor(imgs)

    assert result.shape == (F, 1 + REVMARK_DEFAULT_MSG_BITS)


@pytest.mark.parametrize("detection_mode", ["per_frame", "per_block"])
def test_detection_modes(
    default_revmark_extractor: Extractor, detection_mode: str
) -> None:
    """Test both detection modes produce valid output shapes."""
    default_revmark_extractor.detection_mode = detection_mode
    F, C, H, W = REVMARK_FRAME_SIZE, 3, 256, 256
    imgs = torch.randn(F, C, H, W).clamp_(0.0, 1.0)

    result = default_revmark_extractor(imgs)

    assert result.shape == (F, 1 + REVMARK_DEFAULT_MSG_BITS)


def test_invalid_detection_mode(default_revmark_extractor: Extractor) -> None:
    """Test that invalid detection mode raises ValueError."""
    default_revmark_extractor.detection_mode = "invalid_mode"
    F, C, H, W = REVMARK_FRAME_SIZE, 3, 256, 256
    imgs = torch.randn(F, C, H, W).clamp_(0.0, 1.0)

    with pytest.raises(ValueError, match="Invalid detection_mode"):
        default_revmark_extractor(imgs)


def test_decode_per_block_output_shape(default_revmark_extractor: Extractor) -> None:
    """Test _decode_per_block returns correct shape."""
    _, H, W = default_revmark_extractor.video_shape
    C = 3
    chunk = torch.randn(REVMARK_FRAME_SIZE, C, H, W)

    msgs = default_revmark_extractor._decode_per_block(chunk)

    assert msgs.shape == (REVMARK_FRAME_SIZE, default_revmark_extractor.msg_bits)


def test_decode_per_frame_output_shape(default_revmark_extractor: Extractor) -> None:
    """Test _decode_per_frame returns correct shape."""
    _, H, W = default_revmark_extractor.video_shape
    C = 3
    chunk = torch.randn(REVMARK_FRAME_SIZE, C, H, W)

    msgs = default_revmark_extractor._decode_per_frame(chunk)

    assert msgs.shape == (REVMARK_FRAME_SIZE, default_revmark_extractor.msg_bits)
