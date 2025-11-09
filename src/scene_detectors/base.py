from abc import ABC, abstractmethod
import torch


class SceneDetector(ABC):
    """
    Base class to inherit from when implementing a scene detection algorithm.
    """

    def __init__(self):
        pass

    # Required Methods

    @abstractmethod
    def detect_key_frames(self, frames: torch.Tensor):
        """
        Identify candidate key frames (frames that differ significantly from the previous one).

        Args:
            frames (torch.Tensor): (F, 3, H, W) RGB frames.
        Returns:
            torch.Tensor: Indices of key frames, shape (K,)
        """
