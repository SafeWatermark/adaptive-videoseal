"""
Watermarking model wrappers for VideoMarkBench evaluation.

Provides a unified interface for different watermarking methods
to enable fair comparison across methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
from pathlib import Path

import sys

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "videoseal"))

from main import AdaptiveVideoseal  # noqa: E402
from videoseal.models import Videoseal  # noqa: E402


class WatermarkingModel(ABC):
    """
    Abstract base class for watermarking models.

    Provides a unified interface for embedding and extracting watermarks.
    """

    @abstractmethod
    def embed(
        self, video: torch.Tensor, message: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Embed watermark into video.

        Args:
            video: Video tensor of shape (T, C, H, W) in [0, 1]
            message: Optional message tensor. If None, generates random message.

        Returns:
            Dictionary with keys:
                - 'watermarked_video': Watermarked video tensor
                - 'message': Embedded message tensor
        """
        pass

    @abstractmethod
    def extract(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract watermark logits from video.

        Args:
            video: Video tensor of shape (T, C, H, W) in [0, 1]

        Returns:
            Frame-level logits of shape (T, K) where K is number of bits
        """
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device of the model."""
        pass


class VideoSealWrapper(WatermarkingModel):
    """
    Wrapper for VideoSeal model.
    """

    def __init__(self, model: Videoseal):
        """
        Initialize VideoSeal wrapper.

        Args:
            model: VideoSeal model instance
        """
        self.model = model
        self.model.eval()

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return self.model.device

    def embed(
        self, video: torch.Tensor, message: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Embed watermark into video using VideoSeal.

        Args:
            video: Video tensor of shape (T, C, H, W) in [0, 1]
            message: Optional message tensor. If None, generates random message.

        Returns:
            Dictionary with 'watermarked_video' and 'message'
        """
        # Ensure video is on correct device
        video = video.to(self.device)

        # Ensure video is in correct format: (T, C, H, W)
        if video.dim() == 3:  # (C, H, W) - single frame
            video = video.unsqueeze(0)  # (1, C, H, W)
        elif video.dim() == 4:  # Should be (T, C, H, W)
            pass  # Already correct
        elif video.dim() == 5:  # (B, T, C, H, W)
            video = video.squeeze(0)  # (T, C, H, W)
        else:
            raise ValueError(f"Unexpected video shape: {video.shape}")

        # Generate message if not provided
        if message is None:
            message = self.model.get_random_msg()  # (1, K)
        else:
            message = message.to(self.device)
            if message.dim() == 1:
                message = message.unsqueeze(0)

        # Embed watermark
        # VideoSeal.embed expects (T, C, H, W) for is_video=True
        with torch.no_grad():
            outputs = self.model.embed(video, msgs=message, is_video=True)
            watermarked = outputs["imgs_w"]  # (T, C, H, W)
            embedded_message = outputs["msgs"][0]  # (K,)

        return {
            "watermarked_video": watermarked,
            "message": embedded_message,
        }

    def extract(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract watermark logits from video using VideoSeal.

        Args:
            video: Video tensor of shape (T, C, H, W) in [0, 1]

        Returns:
            Frame-level logits of shape (T, K)
        """
        # Ensure video is on correct device
        video = video.to(self.device)

        # Ensure video is in correct format: (T, C, H, W)
        if video.dim() == 3:  # (C, H, W) - single frame
            video = video.unsqueeze(0)  # (1, C, H, W)
        elif video.dim() == 4:  # Should be (T, C, H, W)
            pass  # Already correct
        elif video.dim() == 5:  # (B, T, C, H, W)
            video = video.squeeze(0)  # (T, C, H, W)
        else:
            raise ValueError(f"Unexpected video shape: {video.shape}")

        # Extract watermark
        # VideoSeal.detect expects (T, C, H, W) for is_video=True
        with torch.no_grad():
            outputs = self.model.detect(video, is_video=True)
            # Get logits (excluding detection bit)
            # outputs["preds"] is (T, K+1) where first is detection bit
            logits = outputs["preds"][:, 1:]  # (T, K)

        return logits


class AdaptiveVideoSealWrapper(WatermarkingModel):
    """
    Wrapper for Adaptive VideoSeal model.
    """

    def __init__(self, model: AdaptiveVideoseal):
        """
        Initialize Adaptive VideoSeal wrapper.

        Args:
            model: AdaptiveVideoseal model instance
        """
        self.model = model
        self.model.eval()

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return self.model.device

    def embed(
        self, video: torch.Tensor, message: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Embed watermark into video using Adaptive VideoSeal.

        Args:
            video: Video tensor of shape (T, C, H, W) in [0, 1]
            message: Optional message tensor. If None, generates random message.

        Returns:
            Dictionary with 'watermarked_video' and 'message'
        """
        # Ensure video is on correct device
        video = video.to(self.device)

        # Ensure video is in correct format: (T, C, H, W)
        if video.dim() == 3:  # (C, H, W) - single frame
            video = video.unsqueeze(0)  # (1, C, H, W)
        elif video.dim() == 4:  # Should be (T, C, H, W)
            pass  # Already correct
        elif video.dim() == 5:  # (B, T, C, H, W)
            video = video.squeeze(0)  # (T, C, H, W)
        else:
            raise ValueError(f"Unexpected video shape: {video.shape}")

        # Generate message if not provided
        if message is None:
            message = self.model.get_random_msg()  # (1, K)
        else:
            message = message.to(self.device)
            if message.dim() == 1:
                message = message.unsqueeze(0)

        # Embed watermark
        # AdaptiveVideoSeal.embed expects (T, C, H, W) for is_video=True
        with torch.no_grad():
            outputs = self.model.embed(video, msgs=message, is_video=True)
            watermarked = outputs["imgs_w"]  # (T, C, H, W)
            embedded_message = outputs["msgs"][0]  # (K,)

        return {
            "watermarked_video": watermarked,
            "message": embedded_message,
        }

    def extract(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract watermark logits from video using Adaptive VideoSeal.

        Args:
            video: Video tensor of shape (T, C, H, W) in [0, 1]

        Returns:
            Frame-level logits of shape (T, K)
        """
        # Ensure video is on correct device
        video = video.to(self.device)

        # Ensure video is in correct format: (T, C, H, W)
        if video.dim() == 3:  # (C, H, W) - single frame
            video = video.unsqueeze(0)  # (1, C, H, W)
        elif video.dim() == 4:  # Should be (T, C, H, W)
            pass  # Already correct
        elif video.dim() == 5:  # (B, T, C, H, W)
            video = video.squeeze(0)  # (T, C, H, W)
        else:
            raise ValueError(f"Unexpected video shape: {video.shape}")

        # Extract watermark
        # AdaptiveVideoSeal.detect expects (T, C, H, W) for is_video=True
        with torch.no_grad():
            outputs = self.model.detect(video, is_video=True)
            # Get logits (excluding detection bit)
            # outputs["preds"] is (T, K+1) where first is detection bit
            logits = outputs["preds"][:, 1:]  # (T, K)

        return logits


# Registry of model wrappers
MODEL_WRAPPERS = {
    "videoseal": VideoSealWrapper,
    "adaptive-videoseal": AdaptiveVideoSealWrapper,
}


def get_model_wrapper(name: str, model: torch.nn.Module) -> WatermarkingModel:
    """
    Get a model wrapper by name.

    Args:
        name: Name of the model wrapper
        model: The model instance to wrap

    Returns:
        Model wrapper instance
    """
    if name not in MODEL_WRAPPERS:
        raise ValueError(
            f"Unknown model wrapper: {name}. "
            f"Available: {list(MODEL_WRAPPERS.keys())}"
        )
    return MODEL_WRAPPERS[name](model)
