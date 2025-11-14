import torch
import torch_dct as dct
from torch.nn import functional as F

from scene_detectors.base import SceneDetector
from scene_detectors.utils import rgb_to_gray


class HashDetector(SceneDetector):
    """based on https://github.com/Breakthrough/PySceneDetect"""

    def __init__(
        self,
        threshold: float = 0.395,
        size: int = 16,
        lowpass: int = 2,
        **kwargs,
    ):
        # threshold: sensitivity for detecting scene change
        # size: hash block size (smaller = coarser)
        # lowpass: scaling factor to capture low-frequency structure
        self.threshold = threshold
        self.size = size
        self.factor = lowpass
        self._size_sq = float(size * size)

    def detect_key_frames(
        self,
        frames: torch.Tensor,
    ):
        """
        Identify candidate key frames (frames that differ significantly from the previous one).

        Args:
            frames (torch.Tensor): (F, 3, H, W) RGB frames.
        Returns:
            torch.Tensor: Indices of key frames, shape (K,)
        """

        hash_frames = self.calculate_hash_frames(
            frames=frames, hash_size=self.size, factor=self.factor
        )  # (F,)

        # Hamming distance is calculated to compare to last frame
        hash_dist = torch.count_nonzero(
            hash_frames[:-1].flatten(start_dim=1)
            != hash_frames[1:].flatten(start_dim=1),
            dim=-1,
        )

        # Normalize based on size of the hash
        hash_dist_norm = hash_dist / self._size_sq

        candidate_key_frame_ids = torch.where(hash_dist_norm > self.threshold)[0] + 1

        return candidate_key_frame_ids

    @staticmethod
    def calculate_hash_frames(
        frames: torch.Tensor,
        hash_size: int,
        factor: int,
        interpolation: dict = {
            "mode": "bilinear",
            "align_corners": False,
            "antialias": True,
        },
    ):
        """
        Compute perceptual hash for each frame using DCT-based pHash. Based on phash from
        https://github.com/JohannesBuchner/imagehash,


        Caution:
        - The image is resized to remove fine-grained texture details, focusing on overall structure.
        - Larger original images usually yield more stable and meaningful hashes.
        - Excessive compression or resizing before hashing may reduce accuracy.

        Args:
            frames: (torch.Tensor): (F, 3, H, W) RGB frames.
        Returns:
            torch.Tensor: (F, 1, hash_size, hash_size) total perceptual hashes per frame.

        """
        if frames.max() <= 1:
            frames = frames * 255.0
        gray_frames = rgb_to_gray(frames)  # (F, H, W)
        gray_frames = gray_frames.unsqueeze(1)  # (F, 1, H, W)

        imsize = hash_size * factor
        gray_frames = F.interpolate(gray_frames, size=(imsize, imsize), **interpolation)

        # Normalize intensity
        eps = 1e-8
        gray_frames = gray_frames - gray_frames.mean(dim=(2, 3), keepdim=True)
        gray_frames = gray_frames / (
            gray_frames.amax(dim=(2, 3, 1), keepdim=True) + eps
        )
        # TODO: figure out whether need to transform into gray scale before applying dct

        dct_complete = dct.dct_2d(gray_frames, norm="ortho")
        dct_low_freq = dct_complete[..., :hash_size, :hash_size]
        med = torch.median(dct_low_freq.flatten(start_dim=1), dim=1).values

        hash_frames = dct_low_freq > med.view(-1, 1, 1, 1)
        return hash_frames
