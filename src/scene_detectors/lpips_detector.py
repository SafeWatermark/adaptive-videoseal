import lpips
import torch

from scene_detectors.base import SceneDetector


class LpipsDetector(SceneDetector):
    def __init__(
        self,
        net: str = "alex",  # 'alex', 'vgg','squeeze'
        threshold: float = 0.1,  # we need to figure out the appropriate default threshold
        device: str = "auto",
        **kwargs,
    ):
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
        if frames.min().item() < 0.0 or frames.max().item() > 1.0:
            raise ValueError(
                f"`frames` must be normalized to [0, 1], but found values in [{frames.min().item():.3f}, {frames.max().item():.3f}]."
            )
        dists = self.lpips(frames[:-1], frames[1:], normalize=True)

        candidate_key_frame_ids = torch.where(dists > self.threshold)[0] + 1

        return candidate_key_frame_ids
