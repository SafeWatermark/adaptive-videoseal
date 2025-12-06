#!/usr/bin/env python3
"""
Download a sample subset of Kinetics-400 dataset for forgery attack evaluation.

This script downloads a small subset of Kinetics-400 videos:
- Selects a few action classes (default: 10 classes)
- Downloads a few videos per class (default: 5 videos per class)
- Total: ~50 videos (~1-2GB)
"""

import os
import csv
import urllib.request
import urllib.parse
from pathlib import Path
import random
import subprocess
import sys
from typing import List, Tuple

# Kinetics-400 annotation URLs
ANNOTATION_URLS = {
    "train": "https://s3.amazonaws.com/kinetics/400/annotations/train.csv",
    "val": "https://s3.amazonaws.com/kinetics/400/annotations/val.csv",
    "test": "https://s3.amazonaws.com/kinetics/400/annotations/test.csv",
}


def download_annotations(split: str = "val", output_dir: Path = Path("data/kinetics")) -> Path:
    """Download Kinetics-400 annotation CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_file = output_dir / f"kinetics400_{split}.csv"
    
    if annotation_file.exists():
        print(f"Annotation file already exists: {annotation_file}")
        return annotation_file
    
    url = ANNOTATION_URLS[split]
    print(f"Downloading annotation file from {url}...")
    urllib.request.urlretrieve(url, annotation_file)
    print(f"Downloaded to {annotation_file}")
    return annotation_file


def parse_annotations(annotation_file: Path) -> List[Tuple[str, str, str]]:
    """
    Parse Kinetics-400 annotation CSV.
    
    Returns:
        List of (youtube_id, time_start, label) tuples
    """
    videos = []
    with open(annotation_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            youtube_id = row.get('youtube_id', '').strip()
            time_start = row.get('time_start', '').strip()
            label = row.get('label', '').strip()
            if youtube_id and time_start and label:
                videos.append((youtube_id, time_start, label))
    return videos


def sample_videos(
    videos: List[Tuple[str, str, str]],
    num_classes: int = 10,
    videos_per_class: int = 5
) -> List[Tuple[str, str, str]]:
    """
    Sample videos from Kinetics-400.
    
    Args:
        videos: List of (youtube_id, time_start, label) tuples
        num_classes: Number of action classes to sample
        videos_per_class: Number of videos per class
    
    Returns:
        Sampled list of videos
    """
    # Group by label
    by_label = {}
    for video in videos:
        label = video[2]
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(video)
    
    # Select random classes
    all_labels = list(by_label.keys())
    random.shuffle(all_labels)
    selected_labels = all_labels[:num_classes]
    
    # Sample videos from each class
    sampled = []
    for label in selected_labels:
        class_videos = by_label[label]
        random.shuffle(class_videos)
        sampled.extend(class_videos[:videos_per_class])
        print(f"Selected {min(videos_per_class, len(class_videos))} videos from class '{label}'")
    
    return sampled


def check_yt_dlp():
    """Check if yt-dlp is available."""
    try:
        result = subprocess.run(['yt-dlp', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    return False


def download_video(
    youtube_id: str,
    time_start: str,
    label: str,
    output_dir: Path,
    max_retries: int = 3
) -> bool:
    """
    Download a single video from YouTube using yt-dlp.
    
    Args:
        youtube_id: YouTube video ID
        time_start: Start time in seconds
        label: Action label
        output_dir: Output directory
        max_retries: Maximum retry attempts
    
    Returns:
        True if successful, False otherwise
    """
    if not check_yt_dlp():
        print("ERROR: yt-dlp is not installed. Please install it first:")
        print("  pip install yt-dlp")
        return False
    
    # Create label directory
    label_dir = output_dir / label.replace(' ', '_').replace('/', '_')
    label_dir.mkdir(parents=True, exist_ok=True)
    
    # YouTube URL
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    
    # Output filename
    output_file = label_dir / f"{youtube_id}_{time_start}.mp4"
    
    if output_file.exists():
        print(f"  Video already exists: {output_file.name}")
        return True
    
    # Download with yt-dlp
    # Extract segment if time_start is specified
    cmd = [
        'yt-dlp',
        '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        '--merge-output-format', 'mp4',
        '-o', str(output_file),
        '--no-playlist',
        '--quiet',
        '--no-warnings',
    ]
    
    # Add time segment if specified
    try:
        start_time = float(time_start)
        if start_time > 0:
            # Download 10 seconds starting from time_start
            cmd.extend(['--postprocessor-args', f'ffmpeg:-ss {start_time} -t 10'])
    except ValueError:
        pass
    
    cmd.append(url)
    
    for attempt in range(max_retries):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and output_file.exists():
                print(f"  ✓ Downloaded: {output_file.name}")
                return True
            else:
                print(f"  ✗ Failed (attempt {attempt + 1}/{max_retries}): {youtube_id}")
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout (attempt {attempt + 1}/{max_retries}): {youtube_id}")
        except Exception as e:
            print(f"  ✗ Error (attempt {attempt + 1}/{max_retries}): {e}")
    
    return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download sample Kinetics-400 videos')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/kinetics',
        help='Output directory for videos (default: data/kinetics)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Dataset split to download (default: val)'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=10,
        help='Number of action classes to sample (default: 10)'
    )
    parser.add_argument(
        '--videos_per_class',
        type=int,
        default=5,
        help='Number of videos per class (default: 5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Kinetics-400 Sample Downloader")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Split: {args.split}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Videos per class: {args.videos_per_class}")
    print(f"Total videos: {args.num_classes * args.videos_per_class}")
    print()
    
    # Download annotations
    annotation_file = download_annotations(args.split, output_dir)
    
    # Parse annotations
    print(f"Parsing annotations from {annotation_file}...")
    videos = parse_annotations(annotation_file)
    print(f"Found {len(videos)} videos in annotations")
    
    # Sample videos
    print(f"\nSampling {args.num_classes} classes with {args.videos_per_class} videos each...")
    sampled = sample_videos(videos, args.num_classes, args.videos_per_class)
    print(f"Selected {len(sampled)} videos total")
    
    # Download videos
    print(f"\nDownloading videos to {output_dir}...")
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for i, (youtube_id, time_start, label) in enumerate(sampled, 1):
        print(f"[{i}/{len(sampled)}] Downloading {youtube_id} ({label})...")
        if download_video(youtube_id, time_start, label, videos_dir):
            success_count += 1
    
    print("\n" + "=" * 70)
    print(f"Download complete: {success_count}/{len(sampled)} videos downloaded")
    print(f"Videos saved to: {videos_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()

