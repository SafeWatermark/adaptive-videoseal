"""
Dataset loader for VideoMarkBench evaluation.

Loads videos from the VideoMarkBench dataset structure.
"""

import torch
from pathlib import Path
from typing import List, Dict, Iterator, Tuple, Optional
import logging

# Try to use decord for faster video loading, fallback to cv2 if not available
try:
    from decord import VideoReader, cpu

    decord_available = True
except ImportError:
    decord_available = False


class VideoMarkBenchDataset:
    """
    Dataset loader for VideoMarkBench videos.

    Supports the original VideoMarkBench dataset structure:
    data/videomarkbench/
      {model}/          # Sora, Hunyuan, SVD
        {style}/        # realistic, cartoon, sci-fi
          {id}.mp4      # Video files
    """

    def __init__(self, data_dir: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize VideoMarkBench dataset.

        Args:
            data_dir: Root directory containing VideoMarkBench videos
            logger: Optional logger for logging warnings and errors
        """
        self.data_dir = Path(data_dir)
        self.video_files: List[Path] = []
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self._load_video_list()

    def _load_video_list(self):
        """Load list of videos from data directory."""
        if not self.data_dir.exists():
            return

        video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
        for video_path in self.data_dir.glob("**/*"):
            if video_path.suffix.lower() in video_extensions:
                self.video_files.append(video_path)

        self.video_files.sort()

    def __len__(self) -> int:
        """Return number of videos in dataset."""
        return len(self.video_files)

    def get_video_files(self) -> List[Path]:
        """Get list of video file paths."""
        return self.video_files.copy()

    def get_selected_videos(
        self, selected_files: List[Path]
    ) -> Iterator[Tuple[torch.Tensor, Dict]]:
        """
        Iterate over selected video files only.

        Args:
            selected_files: List of video file paths to iterate over

        Yields:
            Tuple of (video_tensor, video_info_dict)
        """
        for video_path in selected_files:
            try:
                video = self._load_video(video_path)
                video_info = self._parse_video_info(video_path)
                yield video, video_info
            except Exception as e:
                self.logger.warning(f"Error loading {video_path}: {e}")
                continue

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, Dict]]:
        """
        Iterate over videos in dataset.

        Yields:
            Tuple of (video_tensor, video_info_dict)
            - video_tensor: (T, C, H, W) in [0, 1]
            - video_info: Dictionary with video metadata
        """
        for video_path in self.video_files:
            try:
                video = self._load_video(video_path)
                video_info = self._parse_video_info(video_path)
                yield video, video_info
            except Exception as e:
                self.logger.warning(f"Error loading {video_path}: {e}")
                continue

    def _load_video(self, video_path: Path) -> torch.Tensor:
        """
        Load video from file using decord (faster) or cv2 (fallback).

        Args:
            video_path: Path to video file

        Returns:
            Video tensor of shape (T, C, H, W) in [0, 1]
        """
        if decord_available:
            return self._load_video_decord(video_path)
        else:
            return self._load_video_cv2(video_path)

    def _load_video_decord(self, video_path: Path) -> torch.Tensor:
        """
        Load video using decord (faster, batch loading).

        Args:
            video_path: Path to video file

        Returns:
            Video tensor of shape (T, C, H, W) in [0, 1]
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            # Load video with decord (supports batch loading)
            # Note: 512x512 is the standard resolution for VideoMarkBench dataset.
            # VideoSeal model processes at 256x256 internally, but input videos are loaded at 512x512.
            vr = VideoReader(
                str(video_path), width=512, height=512, num_threads=4, ctx=cpu(0)
            )

            # Get all frames at once (much faster than frame-by-frame)
            num_frames = len(vr)
            if num_frames == 0:
                raise ValueError(f"No frames loaded from {video_path}")

            # Get batch of all frames: shape (T, H, W, C) in RGB
            vid_np = vr.get_batch(range(num_frames)).asnumpy()  # (T, H, W, C)

            # Convert to tensor: (T, H, W, C) -> (T, C, H, W) and normalize
            vid_tensor = torch.from_numpy(vid_np).float()
            vid_tensor = (
                vid_tensor.permute(0, 3, 1, 2) / 255.0
            )  # (T, C, H, W) in [0, 1]

            return vid_tensor

        except Exception as e:
            self.logger.warning(
                f"Error loading video with decord: {e}, falling back to cv2"
            )
            return self._load_video_cv2(video_path)

    def _load_video_cv2(self, video_path: Path) -> torch.Tensor:
        """
        Load video using cv2 (fallback method).

        Args:
            video_path: Path to video file

        Returns:
            Video tensor of shape (T, C, H, W) in [0, 1]
        """
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB and resize to 512x512
            # Note: 512x512 is the standard resolution for VideoMarkBench dataset.
            # VideoSeal model processes at 256x256 internally, but input videos are loaded at 512x512.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (512, 512))
            # Convert to tensor: (H, W, C) -> (C, H, W)
            frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames loaded from {video_path}")

        # Stack frames: (T, C, H, W)
        video = torch.stack(frames, dim=0)
        return video

    def _parse_video_info(self, video_path: Path) -> Dict:
        """
        Parse video information from file path.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video metadata
        """
        parts = video_path.parts
        name = video_path.stem

        # Calculate relative path from data_dir
        try:
            relative_path = video_path.relative_to(self.data_dir)
            # Convert to string with forward slashes (e.g., "Hunyuan/cartoon/1.mp4")
            name = str(relative_path).replace("\\", "/")
        except ValueError:
            # If relative_to fails, use the original name
            pass

        # Try to extract model and style from path
        info = {"name": name, "path": str(video_path)}

        try:
            # Look for videomarkbench in path
            if "videomarkbench" in parts:
                idx = parts.index("videomarkbench")
                if len(parts) > idx + 2:
                    info["model"] = parts[idx + 1]
                    info["style"] = parts[idx + 2]
        except (ValueError, IndexError):
            pass

        return info
