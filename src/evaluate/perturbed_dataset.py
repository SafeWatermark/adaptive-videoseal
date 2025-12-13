"""
Dataset loader for pre-perturbed videos.

Loads videos that have been pre-processed with perturbations
by the apply_perturbations.py script.
"""

import torch
from pathlib import Path
from typing import Dict, Iterator, Tuple, Optional, List
import logging
import json

# Try to use decord for faster video loading, fallback to cv2 if not available
try:
    from decord import VideoReader, cpu

    decord_available = True
except ImportError:
    decord_available = False


class PerturbedVideoDataset:
    """
    Dataset loader for pre-perturbed videos.

    Loads videos from the structure created by apply_perturbations.py:
    output_dir/
        metadata.json
        {perturbation_name}/
            {video_name}.pt or .mp4
    """

    def __init__(
        self,
        data_dir: Path,
        logger: Optional[logging.Logger] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize perturbed video dataset.

        Args:
            data_dir: Root directory containing pre-perturbed videos
            logger: Optional logger
            device: Device to load tensors to (default: CPU)
        """
        self.data_dir = Path(data_dir)
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.device = device if device is not None else torch.device("cpu")
        self.metadata: Dict = {}
        self.save_format: str = "pt"
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata from pre-perturbed directory."""
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}. "
                "Please run apply_perturbations.py first."
            )

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.save_format = self.metadata.get("save_format", "pt")
        self.logger.info(
            f"Loaded metadata: {len(self.metadata.get('videos', []))} videos, "
            f"{len(self.metadata.get('perturbations', []))} perturbations"
        )

    def get_perturbations(self) -> List[Dict]:
        """Get list of available perturbations with their info."""
        return self.metadata.get("perturbations", [])

    def get_perturbation_names(self) -> List[str]:
        """Get list of perturbation names."""
        return [p["name"] for p in self.get_perturbations()]

    def get_videos(self) -> List[Dict]:
        """Get list of videos with their info."""
        return self.metadata.get("videos", [])

    def get_video_names(self) -> List[str]:
        """Get list of video names."""
        return [v["name"] for v in self.get_videos()]

    def __len__(self) -> int:
        """Return number of videos in dataset."""
        return len(self.metadata.get("videos", []))

    def load_perturbed_video(
        self, video_name: str, perturbation_name: str
    ) -> torch.Tensor:
        """
        Load a specific perturbed video.

        Args:
            video_name: Name of the video (original name, will be sanitized)
            perturbation_name: Name of the perturbation

        Returns:
            Video tensor of shape (T, C, H, W) in [0, 1]
        """
        # Sanitize video name
        safe_video_name = video_name.replace("/", "_").replace("\\", "_")

        if self.save_format == "pt":
            video_path = self.data_dir / perturbation_name / f"{safe_video_name}.pt"
            return self._load_tensor(video_path)
        else:
            video_path = self.data_dir / perturbation_name / f"{safe_video_name}.mp4"
            return self._load_video(video_path)

    def _load_tensor(self, tensor_path: Path) -> torch.Tensor:
        """Load video tensor from .pt file."""
        if not tensor_path.exists():
            raise FileNotFoundError(f"Tensor file not found: {tensor_path}")
        return torch.load(tensor_path, map_location=self.device, weights_only=True)

    def _load_video(self, video_path: Path) -> torch.Tensor:
        """Load video from .mp4 file."""
        if decord_available:
            return self._load_video_decord(video_path)
        else:
            return self._load_video_cv2(video_path)

    def _load_video_decord(self, video_path: Path) -> torch.Tensor:
        """Load video using decord."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            vr = VideoReader(
                str(video_path), width=512, height=512, num_threads=4, ctx=cpu(0)
            )
            num_frames = len(vr)
            if num_frames == 0:
                raise ValueError(f"No frames loaded from {video_path}")

            vid_np = vr.get_batch(range(num_frames)).asnumpy()
            vid_tensor = torch.from_numpy(vid_np).float()
            vid_tensor = vid_tensor.permute(0, 3, 1, 2) / 255.0

            return vid_tensor.to(self.device)

        except Exception as e:
            self.logger.warning(
                f"Error loading video with decord: {e}, falling back to cv2"
            )
            return self._load_video_cv2(video_path)

    def _load_video_cv2(self, video_path: Path) -> torch.Tensor:
        """Load video using cv2."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (512, 512))
            frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames loaded from {video_path}")

        video = torch.stack(frames, dim=0)
        return video.to(self.device)

    def iterate_perturbation(
        self, perturbation_name: str
    ) -> Iterator[Tuple[torch.Tensor, Dict]]:
        """
        Iterate over all videos for a specific perturbation.

        Args:
            perturbation_name: Name of the perturbation

        Yields:
            Tuple of (video_tensor, video_info_dict)
        """
        for video_meta in self.metadata.get("videos", []):
            video_name = video_meta["name"]
            try:
                video = self.load_perturbed_video(video_name, perturbation_name)
                yield video, video_meta
            except Exception as e:
                self.logger.warning(
                    f"Error loading {video_name} with {perturbation_name}: {e}"
                )
                continue

    def iterate_video(
        self, video_name: str
    ) -> Iterator[Tuple[torch.Tensor, str, Dict]]:
        """
        Iterate over all perturbations for a specific video.

        Args:
            video_name: Name of the video

        Yields:
            Tuple of (video_tensor, perturbation_name, perturbation_info)
        """
        for pert_info in self.metadata.get("perturbations", []):
            pert_name = pert_info["name"]
            try:
                video = self.load_perturbed_video(video_name, pert_name)
                yield video, pert_name, pert_info
            except Exception as e:
                self.logger.warning(f"Error loading {video_name} with {pert_name}: {e}")
                continue

    def __iter__(
        self,
    ) -> Iterator[Tuple[str, str, torch.Tensor, Dict, Dict]]:
        """
        Iterate over all videos and perturbations.

        Yields:
            Tuple of (video_name, perturbation_name, video_tensor, video_info, perturbation_info)
        """
        for video_meta in self.metadata.get("videos", []):
            video_name = video_meta["name"]
            for pert_info in self.metadata.get("perturbations", []):
                pert_name = pert_info["name"]
                try:
                    video = self.load_perturbed_video(video_name, pert_name)
                    yield video_name, pert_name, video, video_meta, pert_info
                except Exception as e:
                    self.logger.warning(
                        f"Error loading {video_name} with {pert_name}: {e}"
                    )
                    continue
