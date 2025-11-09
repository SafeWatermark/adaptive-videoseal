import torch


def rgb_to_seperated_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB video (F, 3, H, W) to HSV.
    Input and output are in range [0, 1].
    """
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    maxc, _ = rgb.max(dim=1)
    minc, _ = rgb.min(dim=1)
    delta = maxc - minc

    # Hue
    hue = torch.zeros_like(maxc)

    mask = delta != 0
    r_eq_max = (maxc == r) & mask
    g_eq_max = (maxc == g) & mask
    b_eq_max = (maxc == b) & mask

    hue[r_eq_max] = (g[r_eq_max] - b[r_eq_max]) / delta[r_eq_max]
    hue[g_eq_max] = 2.0 + (b[g_eq_max] - r[g_eq_max]) / delta[g_eq_max]
    hue[b_eq_max] = 4.0 + (r[b_eq_max] - g[b_eq_max]) / delta[b_eq_max]
    hue = (hue / 6.0) % 1.0  # normalize to [0, 1]

    # Saturation
    sat = torch.zeros_like(maxc)
    sat[mask] = delta[mask] / maxc[mask]

    # Value
    val = maxc

    return (hue, sat, val)


def rgb_to_gray(rgb: torch.Tensor):
    """
    Convert an RGB video (F, 3, H, W) to gray.
    Input and output are in range [0, 255]
    """

    r, g, b = 0.2989, 0.5870, 0.1140

    gray = r * rgb[:, 0, :, :] + g * rgb[:, 1, :, :] + b * rgb[:, 2, :, :]
    return gray


def hist_corr(hist1: torch.Tensor, hist2: torch.Tensor) -> torch.Tensor:
    """
    computes correlation between two histograms
    ranging from -1 (inverse correlation) to 1 (perfect correlation).
    """
    eps = torch.tensor(1e-9, device=hist1.device)
    hist1_mean = hist1.mean(dim=-1)
    hist2_mean = hist2.mean(dim=-1)
    num = torch.sum(
        (hist1 - hist1_mean.unsqueeze(1)) * (hist2 - hist2_mean.unsqueeze(1)), dim=-1
    )
    den = torch.sqrt(
        torch.sum((hist1 - hist1_mean.unsqueeze(1)) ** 2, dim=-1)
        * torch.sum((hist2 - hist2_mean.unsqueeze(1)) ** 2, dim=-1)
    )
    return num / torch.maximum(den, eps)


def batched_histogram(
    x: torch.Tensor, bins: int = 256, range: tuple = (0, 256)
) -> tuple[torch.Tensor, torch.Tensor]:
    # x: (F, C, H, W) or (F, N)
    F = x.shape[0]

    x_flat = x.view(F, -1).contiguous()

    bin_edges = torch.linspace(
        range[0], range[1], bins + 1, device=x.device, dtype=x.dtype
    )
    is_valid = (x_flat >= range[0]) & (x_flat < range[1])

    inds = torch.bucketize(x_flat, bin_edges, right=True)
    inds = inds - 1
    inds = torch.clamp(inds, 0, bins - 1)

    hist = torch.zeros((F, bins), device=x.device, dtype=x.dtype)

    src_counts = torch.ones_like(x_flat, dtype=hist.dtype)
    src_counts = src_counts * is_valid

    hist.scatter_add_(dim=1, index=inds, src=src_counts)
    return hist, bin_edges
