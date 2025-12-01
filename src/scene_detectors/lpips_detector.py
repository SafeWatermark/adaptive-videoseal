import lpips
import torch

from scene_detectors.base import SceneDetector


class LpipsDetector(SceneDetector):
    def __init__(
        self,
        net: str = "alex",  # 'alex', 'vgg','squeeze'
        threshold: float = 0.125,
        device: str = "auto",
        **kwargs,
    ):
        """We encourage to set the step_size 16 for ensuring minimum quality. (.99 in sa-v validation)"""
        self.lpips = lpips.LPIPS(net=net)
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lpips = self.lpips.to(device)
        self.lpips.eval()
        self.threshold = threshold

    @torch.inference_mode()
    def detect_key_frames(
        self,
        frames: torch.Tensor,
    ):
        """
        Identify candidate key frames (frames that differ significantly from the previous one).
        * c.f. lpips distance >= 0.0
        Args:
            frames (torch.Tensor): (F, 3, H, W) RGB frames.
                * IMPORTANT: normalized to [0,1]

        Returns:
            torch.Tensor: Indices of key frames, shape (K,)
        """
        epsilon = 1e-6
        min_val = frames.min().item()
        max_val = frames.max().item()

        if min_val < 0.0 - epsilon or max_val > 1.0 + epsilon:
            raise ValueError(
                f"`frames` must be normalized to [0, 1], but found values in [{frames.min().item():.3f}, {frames.max().item():.3f}]."
            )

        # If values exceed only by tiny floating drift → clamp
        if min_val < 0.0 or max_val > 1.0:
            frames = frames.clamp(0.0, 1.0)

        dists = self.lpips(frames[:-1], frames[1:], normalize=True)

        candidate_key_frame_ids = torch.where(dists > self.threshold)[0] + 1

        return candidate_key_frame_ids
