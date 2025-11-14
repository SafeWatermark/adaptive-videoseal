import torch
from src.scene_detectors.utils import (
    batched_histogram,
    rgb_to_seperated_hsv,
    hist_corr,
    rgb_to_gray,
)
import numpy as np
import cv2


def test_rgb_to_gray_in_red():
    # (1, 3, 1, 1)
    rgb = torch.tensor([[[[255]], [[0]], [[0]]]], dtype=torch.float32)  # pure red
    gray = rgb_to_gray(rgb)
    assert torch.allclose(gray.squeeze(), torch.tensor([0.2989 * 255]))


def test_rgb_to_gray_random():
    F, C, H, W = 5, 3, 4, 4
    rgb = torch.randint(0, 256, (F, C, H, W), dtype=torch.uint8)
    gray = rgb_to_gray(rgb)

    # Check output shape
    assert gray.size() == (F, H, W), f"Expected {(F, H, W)}, got {gray.size()}"

    # Check value range
    assert gray.max() <= 255, f"Max value out of range: {gray.max()}"
    assert gray.min() >= 0, f"Min value out of range: {gray.min()}"


def test_hist_corr_identical():
    hist = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    corr = hist_corr(hist, hist)
    expected_corr = torch.tensor([1.0])
    assert torch.allclose(corr, expected_corr, atol=1e-6)


def test_hist_corr_inverse():
    hist1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    hist2 = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    corr = hist_corr(hist1, hist2)
    expected_corr = torch.tensor([-1.0])
    assert torch.allclose(corr, expected_corr, atol=1e-6)


def test_hist_corr_random():
    torch.manual_seed(0)
    hist1 = torch.rand((1, 100))
    hist2 = torch.rand((1, 100))
    corr = hist_corr(hist1, hist2)
    assert abs(corr.item()) < 0.2  # roughly uncorrelated


def test_hist_corr_batch():
    h1 = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    h2 = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )
    corr = hist_corr(h1, h2)
    assert corr.shape == (2,)
    assert torch.allclose(corr[0], torch.tensor(1.0))
    assert torch.allclose(corr[1], torch.tensor(0.0))


def test_simple_batched_histogram():
    x = torch.tensor([[0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0]])
    hist, bin_edges = batched_histogram(x=x, bins=4, range=(0, 4))

    expected_hist = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 4.0, 0.0, 0.0]])
    assert torch.all(hist == expected_hist)

    expected_bin_edges = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    assert torch.all(bin_edges == expected_bin_edges)


def test_random_comparison_batched_histogram():
    F = 2
    C = 1
    H, W = 128, 128
    x = torch.randn((F, C, H, W))
    bins = 256
    _range = (0, 256)

    expected_hists = []
    expected_bin_edges = None
    for each in x:
        expected_hist, expected_bin_edges = torch.histogram(
            input=each, bins=bins, range=_range
        )
        expected_hists.append(expected_hist)
    hist, bin_edges = batched_histogram(x=x, bins=bins, range=_range)
    expected_hists = torch.stack(expected_hists)

    assert torch.allclose(hist, expected_hists)
    assert torch.all(bin_edges == expected_bin_edges)


def test_rgb_to_hsv_primary_colors():
    rgb = torch.tensor(
        [
            [[[1.0]], [[0.0]], [[0.0]]],  # red
            [[[0.0]], [[1.0]], [[0.0]]],  # green
            [[[0.0]], [[0.0]], [[1.0]]],  # blue
        ]
    )
    h, s, v = rgb_to_seperated_hsv(rgb)
    assert torch.allclose(h.squeeze(), torch.tensor([0.0, 1 / 3, 2 / 3]))
    assert torch.allclose(s.squeeze(), torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(v.squeeze(), torch.tensor([1.0, 1.0, 1.0]))


def test_random_comparison_rgb_to_hsv():
    H, W = 128, 128
    C = 3
    rgb_np = np.random.randint(low=0, high=256, size=(H, W, C), dtype=np.uint8)
    rgb_torch = torch.tensor(rgb_np.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(
        0
    )  # (F=1,3,H=1,W=1)

    # ---- PyTorch conversion ----
    rgb_torch /= 255.0
    h_t, s_t, v_t = rgb_to_seperated_hsv(rgb_torch)
    hsv_torch = torch.stack(
        [h_t.squeeze(), s_t.squeeze(), v_t.squeeze()], dim=0
    ).unsqueeze(0)

    # ---- OpenCV conversion ----
    bgr_cv = rgb_np[..., ::-1].copy()  # convert to BGR uint8
    hsv_cv = cv2.cvtColor(bgr_cv, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_cv[..., 0] /= 179.0
    hsv_cv[..., 1:] /= 255.0

    expected_hsv = hsv_cv.transpose(2, 0, 1)  # (3, H, W)
    expected_hsv = torch.tensor(expected_hsv).unsqueeze(0)  # (1, 3, H, W)

    assert torch.mean(torch.abs(hsv_torch.squeeze() - expected_hsv.squeeze())) < 1e-2
