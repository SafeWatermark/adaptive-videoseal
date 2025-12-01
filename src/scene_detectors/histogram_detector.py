import torch
from scene_detectors.utils import hist_corr, batched_histogram
from scene_detectors.base import SceneDetector

from videoseal.data.transforms import RGB2YUV


class HistogramDetector(SceneDetector):
    """based on https://github.com/Breakthrough/PySceneDetect"""

    def __init__(
        self, threshold: float = 0.15, bins: int = 256, device: str = "auto", **kwargs
    ):
        """We encourage to set the step_size 16 for ensuring minimum quality. (.99 in sa-v validation)"""
        # Internally, threshold represents the correlation between two histograms and has values
        # between -1.0 and 1.0.
        self.threshold = max(0.0, min(1.0, 1.0 - threshold))
        self.bins = bins
        self._metric_key = f"hist_diff [bins={self.bins}]"
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rgb2yuv = RGB2YUV().to(device)

    def detect_key_frames(self, frames: torch.Tensor) -> torch.Tensor:
        hist = self.calculate_histogram(frames, bins=self.bins)
        hist_diff = hist_corr(hist[:-1], hist[1:])

        candidate_key_frame_ids = torch.where(hist_diff <= self.threshold)[0] + 1
        return candidate_key_frame_ids

    def calculate_histogram(
        self, frames: torch.Tensor, bins: int = 256, normalize: bool = True
    ) -> torch.Tensor:
        """
        Calculates and optionally normalizes the histogram of the luma (Y) channel of an image
        converted from RGB to YUV color space.

        This function extracts the Y channel from the given RGB image, computes its histogram with
        the specified number of bins, and optionally normalizes this histogram to have a sum of one
        across all bins.

        Args:
        -----
        frame_img : torch.Tensor
            The input image in RGB color space, assumed to have shape (F, 3, H, W)
        bins : int, optional (default=256)
            The number of bins to use for the histogram.
        normalize : bool, optional (default=True)
            A boolean flag that determines whether the histogram should be normalized
            such that the sum of all histogram bins equals 1.

        Returns:
        --------
        torch.Tensor
            A 1D torch.Tensor of length equal to `bins`, representing the histogram of the luma
            channel. Each element in the array represents the count (or frequency) of a particular
            luma value in the image. If normalized, these values represent the relative frequency.
        """
        if frames.max() > 1:
            frames = frames / 255.0

        # take y channel only
        y = self.rgb2yuv(frames)[:, 0:1]
        y *= 255.0

        # Create the histogram with a bin for every rgb value
        hist, _ = batched_histogram(y, bins=bins, range=(0, 256))
        if normalize:
            # Normalize the histogram
            # hist = hist / torch.norm(hist, p=2)
            hist = hist / torch.norm(hist, p=2, dim=1, keepdim=True)

        return hist
