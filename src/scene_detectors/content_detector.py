import torch
from dataclasses import dataclass, asdict
from scene_detectors.utils import rgb_to_seperated_hsv
from scene_detectors.base import SceneDetector


def _mean_pixel_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """
    Return the mean pixel-wise Euclidean distance (RMSE) between `left` and `right`.
    Both tensors should be 4D 8-bit (or float) images of the same shape.

    Args:
        left:  (F, C, H, W)
        right: (F, C, H, W)

    Returns:
        torch.Tensor: (F,) — mean distance per frame
    """
    # Convert to float for safe subtraction
    left = left.float()
    right = right.float()

    # Mean squared difference per frame
    mse = torch.mean((left - right) ** 2, dim=(1, 2, 3))
    rmse = torch.sqrt(mse)  # root mean squared error per frame

    return rmse


# TODO: change default method
@dataclass
class ComponentWeights:
    """Components that make up a frame's score, and their default values."""

    delta_hue: float = 1.0
    """Difference between pixel hue values of adjacent frames."""

    delta_sat: float = 1.0
    """Difference between pixel saturation values of adjacent frames."""

    delta_lum: float = 1.0
    """Difference between pixel luma (brightness) values of adjacent frames."""


luma_only_weights = ComponentWeights(delta_hue=0.0, delta_sat=0.0, delta_lum=1.0)


class ContentDetector(SceneDetector):
    """based on https://github.com/Breakthrough/PySceneDetect"""

    def __init__(
        self,
        threshold: float = 27.0,
        weights: ComponentWeights = ComponentWeights(),
        **kwargs,
    ):
        self.threshold = threshold
        self.weights = weights
        self.normalizer = sum(asdict(self.weights).values())

    def _calculate_adjacent_frame_score(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Calculate per-frame change scores between consecutive frames.

        Args:
            frame (torch.Tensor): (F, 3, H, W) RGB frames.
        Returns:
            torch.Tensor: (F-1,) total content change score per frame transition.
        """
        if frames.max() > 1:
            frames = frames / 255.0
        hue, sat, lum = rgb_to_seperated_hsv(frames)

        hue *= 179.0
        sat *= 255.0
        lum *= 255.0

        hue_score = (
            _mean_pixel_distance(hue[:-1].unsqueeze(1), hue[1:].unsqueeze(1))
            * self.weights.delta_hue
        )
        sat_score = (
            _mean_pixel_distance(sat[:-1].unsqueeze(1), sat[1:].unsqueeze(1))
            * self.weights.delta_sat
        )
        lum_score = (
            _mean_pixel_distance(lum[:-1].unsqueeze(1), lum[1:].unsqueeze(1))
            * self.weights.delta_lum
        )

        return (hue_score + sat_score + lum_score) / self.normalizer

    def detect_key_frames(self, frames: torch.Tensor):
        """
        Identify candidate key frames (frames that differ significantly from the previous one).

        Args:
            frames (torch.Tensor): (F, 3, H, W) RGB frames.
        Returns:
            torch.Tensor: Indices of key frames, shape (K,)
        """

        frames_score = self._calculate_adjacent_frame_score(frames)
        candidate_key_frame_ids = torch.where(frames_score > self.threshold)[0] + 1
        return candidate_key_frame_ids
