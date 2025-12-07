"""
Model wrapper for VideoMarkBench evaluation.

Provides a unified interface for VideoSeal and AdaptiveVideoSeal models
that is compatible with the original VideoMarkBench evaluation structure.
"""

import sys
from pathlib import Path
import torch
from typing import Dict, Optional

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "videoseal"))


class VideoMarkBenchModel:
    """
    Wrapper for watermarking models to be used with VideoMarkBench evaluation.

    Provides a unified interface that matches the original VideoMarkBench structure.
    """

    def __init__(self, model: torch.nn.Module, model_name: str = "videoseal"):
        """
        Initialize model wrapper.

        Args:
            model: VideoSeal or AdaptiveVideoSeal model instance
            model_name: Name of the model ('videoseal' or 'adaptive-videoseal')
        """
        self.model = model
        self.model_name = model_name
        self.model.eval()

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return next(self.model.parameters()).device

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
        Extract watermark logits from video.

        Args:
            video: Video tensor of shape (T, C, H, W) in [0, 1]

        Returns:
            Frame-level logits of shape (T, K) where K is number of bits
        """
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
        with torch.no_grad():
            outputs = self.model.detect(video, is_video=True)
            # Get logits (excluding detection bit)
            # outputs["preds"] is (T, K+1) where first is detection bit
            logits = outputs["preds"][:, 1:]  # (T, K)

        return logits

    def detect(
        self, video: torch.Tensor, is_video: bool = True, requires_grad: bool = False
    ) -> Dict:
        """
        Detect watermark in video (for compatibility with original VideoMarkBench).

        Args:
            video: Video tensor
            is_video: Whether input is video
            requires_grad: Whether gradient computation is needed (for whitebox attacks)

        Returns:
            Dictionary with 'preds' key containing predictions
        """
        video = video.to(self.device)

        # Ensure video is in correct format: (T, C, H, W)
        if is_video:
            if video.dim() == 3:  # (C, H, W) - single frame
                video = video.unsqueeze(0)  # (1, C, H, W)
            elif video.dim() == 4:  # Should be (T, C, H, W)
                pass  # Already correct
            elif video.dim() == 5:  # (B, T, C, H, W)
                video = video.squeeze(0)  # (T, C, H, W)
            else:
                raise ValueError(f"Unexpected video shape: {video.shape}")
        else:
            # Single frame
            if video.dim() == 3:
                video = video.unsqueeze(0)

        # Use actual detection results from model (including real detection bit)
        # If gradient computation is needed, bypass @torch.no_grad() decorator
        # by calling detector directly (only works for single frames in whitebox attacks)
        if requires_grad:
            # For gradient computation, call detector directly to bypass @torch.no_grad()
            # This is only safe for single frames (is_video=False) used in whitebox attacks
            if not is_video and hasattr(self.model, "detector"):
                # Single frame case: call detector directly
                preds = self.model.detector(video)  # (1, K+1)
            else:
                # Video case: still need to use detect() but it won't compute gradients
                # This should not be used in practice for whitebox attacks
                outputs = self.model.detect(video, is_video=is_video)
                preds = outputs["preds"]
        else:
            # No gradient needed, use detect() method normally
            with torch.no_grad():
                outputs = self.model.detect(video, is_video=is_video)
                preds = outputs[
                    "preds"
                ]  # (T, K+1) where first column is actual detection bit

        return {"preds": preds}
