"""
Dataset loader for VideoMarkBench evaluation.

Loads videos from the VideoMarkBench dataset structure.
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Iterator, Tuple
import os


class VideoMarkBenchDataset:
    """
    Dataset loader for VideoMarkBench videos.

    Supports the original VideoMarkBench dataset structure:
    data/videomarkbench/
      {model}/          # Sora, Hunyuan, SVD
        {style}/        # realistic, cartoon, sci-fi
          {id}.mp4      # Video files
    """

    def __init__(self, data_dir: Path):
        """
        Initialize VideoMarkBench dataset.

        Args:
            data_dir: Root directory containing VideoMarkBench videos
        """
        self.data_dir = Path(data_dir)
        self.video_files: List[Path] = []
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
    
    def get_selected_videos(self, selected_files: List[Path]) -> Iterator[Tuple[torch.Tensor, Dict]]:
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
                print(f"Error loading {video_path}: {e}")
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
                print(f"Error loading {video_path}: {e}")
                continue

    def _load_video(self, video_path: Path) -> torch.Tensor:
        """
        Load video from file.

        Args:
            video_path: Path to video file

        Returns:
            Video tensor of shape (T, C, H, W) in [0, 1]
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB and resize to 512x512
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

