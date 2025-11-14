from src.main import AdaptiveVideoseal
import torch
import pytest


@pytest.mark.parametrize(
    "preds_w, step_size, expected_result",
    [
        # --- multi-key-frame test case ---
        (
            torch.tensor(
                [
                    [[[1.0]]],  # W0
                    [[[3.0]]],  # W1
                    [[[9.0]]],  # W2
                ]
            ),
            torch.tensor([2, 3, 1]),
            [1.0, 1.0, 3.0, 3.0, 3.0, 9.0],
        ),
        # --- single key-frame test case ---
        (
            torch.tensor(
                [
                    [[[5.0]]],
                ]
            ),
            torch.tensor([4]),
            [5.0, 5.0, 5.0, 5.0],
        ),
    ],
)
def test_repeat_mode(
    preds_w: torch.Tensor, step_size: torch.Tensor, expected_result: torch.Tensor
):
    """
    Test the 'repeat' video mode in AdaptiveVideoseal._apply_video_mode.

    This verifies that:
    - each key-frame prediction is repeated exactly step_size[i] times,
    - the total number of frames matches sum(step_size),
    - no interpolation is applied (values are exact copies),
    - and the output ordering is correct.
    """
    total_frames = step_size.sum().item()

    repeated_w = AdaptiveVideoseal._apply_video_mode(
        preds_w, total_frames=total_frames, step_size=step_size, video_mode="repeat"
    )

    assert (
        repeated_w.squeeze().tolist() == expected_result
    ), f"expected result: {expected_result}, got: {repeated_w.squeeze().tolist()}"


@pytest.mark.parametrize(
    "preds_w, step_size, expected_result",
    [
        # --- original multi-key-frame test case ---
        (
            torch.tensor(
                [
                    [[[1.0]]],  # W0
                    [[[3.0]]],  # W1
                    [[[9.0]]],  # W2
                ]
            ),
            torch.tensor([2, 6, 8]),
            [
                1.0,  # W0 at index 0
                0.0,
                3.0,  # W1 at index 2
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                9.0,  # W2 at index 8
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ),
        # --- single key-frame test case ---
        (
            torch.tensor(
                [
                    [[[7.0]]],
                ]
            ),
            torch.tensor([5]),
            [7.0, 0.0, 0.0, 0.0, 0.0],
        ),
    ],
)
def test_alternate_mode(
    preds_w: torch.Tensor, step_size: torch.Tensor, expected_result: torch.Tensor
):
    """
    Test the 'alternate' video mode in AdaptiveVideoseal._apply_video_mode.

    This verifies that:
    - key-frame indices are computed using cumulative sums,
    - only those indices are filled with the corresponding preds_w values,
    - all other frames remain zero,
    - output length equals total_frames.
    """
    total_frames = step_size.sum().item()

    alternated_w = AdaptiveVideoseal._apply_video_mode(
        preds_w, total_frames=total_frames, step_size=step_size, video_mode="alternate"
    )

    assert (
        alternated_w.squeeze().tolist() == expected_result
    ), f"expected result: {expected_result}, got: {alternated_w.squeeze().tolist()}"


@pytest.mark.parametrize(
    "preds_w, step_size, expected_result",
    [
        # --- original multi-key-frame test case ---
        (
            torch.tensor(
                [
                    [[[1.0]]],
                    [[[3.0]]],
                    [[[9.0]]],
                ]
            ),
            torch.tensor(
                [2, 6, 8]
            ),  # interpolation between W0→W1(step size: 2), then W1→W2(step size: 8)
            [
                # Interval 1: 2 frames between W0 (1) → W1 (3)
                1.0,
                0.5 * 1.0 + 0.5 * 3.0,  # = 2.0
                # Interval 2: 6 frames between W1 (3) → W2 (9)
                1.0 * 3.0 + 0.0 * 9.0,  # = 3.0
                0.8 * 3.0 + 0.2 * 9.0,  # = 4.2
                0.6 * 3.0 + 0.4 * 9.0,  # = 5.4
                0.4 * 3.0 + 0.6 * 9.0,  # = 6.6
                0.2 * 3.0 + 0.8 * 9.0,  # = 7.8
                0.0 * 3.0 + 1.0 * 9.0,  # = 9.0
                # After last key frame (W2 = 9), repeated 8 times
                9.0,
                9.0,
                9.0,
                9.0,
                9.0,
                9.0,
                9.0,
                9.0,
            ],
        ),
        # --- single key-frame test case(no interpolation) ---
        (
            torch.tensor(
                [
                    [[[4.0]]],  # only key frame
                ]
            ),
            torch.tensor([5]),
            [4.0, 4.0, 4.0, 4.0, 4.0],
        ),
    ],
)
def test_interpolate_mode(
    preds_w: torch.Tensor, step_size: torch.Tensor, expected_result
):
    """
    Test the 'interpolate' video mode in AdaptiveVideoseal._apply_video_mode.

    This test verifies that:
    - key frames are correctly expanded according to 'step_size',
    - linear interpolation is applied between key-frame predictions,
    - the interpolated sequence has the correct total length,
    - exact key-frame positions match the original inputs,
    - and all frames after the last key frame are filled with the final prediction.
    """
    total_frames = step_size.sum().item()

    interpolated_watermarks = AdaptiveVideoseal._apply_video_mode(
        preds_w,
        total_frames=total_frames,
        step_size=step_size,
        video_mode="interpolate",
    )

    assert torch.allclose(
        interpolated_watermarks.squeeze(), torch.tensor(expected_result)
    ), f"expected result: {expected_result}, got: {interpolated_watermarks.squeeze().tolist()}"
