import pytest
import torch
from videoseal.models.embedder import Embedder

from models.revmark import (
    REVMARK_DEFAULT_MSG_BITS,
)


@pytest.mark.parametrize(
    "F, expected_positions, key_frame_ids",
    [
        (1, [0], [0, 0, 0, 0, 0, 0, 0, 0]),
        (2, [0, 4], [0, 0, 0, 0, 1, 1, 1, 1]),
        (3, [0, 3, 6], [0, 0, 0, 1, 1, 1, 2, 2]),
        (4, [0, 2, 4, 6], [0, 0, 1, 1, 2, 2, 3, 3]),
        (5, [0, 2, 4, 5, 7], [0, 0, 1, 1, 2, 3, 3, 4]),
        (6, [0, 2, 3, 4, 6, 7], [0, 0, 1, 2, 3, 3, 4, 5]),
        (7, [0, 2, 3, 4, 5, 6, 7], [0, 0, 1, 2, 3, 4, 5, 6]),
        (8, [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]),
    ],
)
def test_temporal_pad(
    default_revmark_embedder: Embedder,
    F: int,
    expected_positions: list[int],
    key_frame_ids: list[int],
) -> None:
    revmark_f, revmark_h, revmark_w = default_revmark_embedder.video_shape
    C = 3

    chunk = torch.randn((F, C, revmark_h, revmark_w))
    padded, original_positions = default_revmark_embedder._temporal_pad(frames=chunk)

    assert padded.size() == (revmark_f, C, revmark_h, revmark_w)
    assert original_positions == expected_positions

    current_key_id = None
    for pad_id, key_id in enumerate(key_frame_ids):
        if key_id == current_key_id:
            assert torch.allclose(chunk[current_key_id], padded[pad_id])
        else:
            current_key_id = key_id


@pytest.mark.parametrize(
    "F, expected_positions, key_frame_ids",
    [
        (1, [0], [0, 0, 0, 0, 0, 0, 0, 0]),
        (2, [0, 4], [0, 0, 0, 0, 1, 1, 1, 1]),
        (3, [0, 3, 6], [0, 0, 0, 1, 1, 1, 2, 2]),
        (4, [0, 2, 4, 6], [0, 0, 1, 1, 2, 2, 3, 3]),
        (5, [0, 2, 4, 5, 7], [0, 0, 1, 1, 2, 3, 3, 4]),
        (6, [0, 2, 3, 4, 6, 7], [0, 0, 1, 2, 3, 3, 4, 5]),
        (7, [0, 2, 3, 4, 5, 6, 7], [0, 0, 1, 2, 3, 4, 5, 6]),
        (8, [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]),
    ],
)
def test_add_perturbation_into_non_original_frames(
    default_revmark_embedder: Embedder,
    F: int,
    expected_positions: list[int],
    key_frame_ids: list[int],
):
    perturbation_epsilon = 0.1
    _, revmark_h, revmark_w = default_revmark_embedder.video_shape
    C = 3
    default_revmark_embedder.perturbation_epsilon = perturbation_epsilon

    chunk = torch.randn((F, C, revmark_h, revmark_w))
    padded, _ = default_revmark_embedder._temporal_pad(frames=chunk)
    perturbed_padded = (
        default_revmark_embedder._add_perturbation_into_non_original_frames(
            padded, expected_positions
        )
    )

    current_key_id = None
    for pad_id, key_id in enumerate(key_frame_ids):
        if key_id == current_key_id:
            # Should NOT be equal due to perturbation
            assert not torch.allclose(chunk[current_key_id], perturbed_padded[pad_id])
            diff = (chunk[current_key_id] - perturbed_padded[pad_id]).abs()
            # Perturbation is Gaussian with std=epsilon, Most values should be within 3*epsilon
            assert diff.max() < 10 * perturbation_epsilon
        else:
            current_key_id = key_id


def test_revmark_forward_size(default_revmark_embedder: Embedder):
    F, C, H, W = 16, 3, 256, 256
    imgs = torch.randn(F, C, H, W).clamp_(0.0, 1.0)
    msgs = torch.randint(0, 2, (F, REVMARK_DEFAULT_MSG_BITS)).float()

    residual = default_revmark_embedder(imgs, msgs)

    assert residual.shape == (F, C, H, W)


@pytest.mark.parametrize("F", [1, 2, 3, 4, 5, 6, 7])
def test_revmark_forward_less_than_defined_frames(
    default_revmark_embedder: Embedder, F: int
):
    C, H, W = 3, 256, 256

    imgs = torch.randn(F, C, H, W).clamp_(0.0, 1.0)
    msgs = torch.randint(0, 2, (F, default_revmark_embedder.msg_bits)).float()

    residual = default_revmark_embedder(imgs, msgs)

    assert residual.shape == (F, C, H, W)


def test_revmark_preprocess(default_revmark_embedder: Embedder):
    F, H, W = default_revmark_embedder.video_shape
    C = 3
    imgs = torch.zeros(F, C, 2 * H, 2 * W) + 0.5  # 0.5 in [0,1]
    processed, _ = default_revmark_embedder._preprocess(imgs)

    # 0.5 in [0,1] -> 0.0 in [-1,1]
    assert torch.allclose(processed, torch.zeros_like(processed), atol=1e-6)
    assert processed.size() == (F, C, H, W)


def test_postprocess_preserves_size(default_revmark_embedder: Embedder):
    F, H, W = default_revmark_embedder.video_shape
    C = 3
    meta = {"orig_size": (H, W)}

    residual = torch.zeros(F, C, 2 * H, 2 * W)
    result = default_revmark_embedder._postprocess(residual, meta)

    assert result.size() == residual.size()
