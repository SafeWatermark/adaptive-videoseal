"""
Utility script for SA-V validation data.

The SA-V validation dataset provides sequences as JPEG images.
To evaluate Adaptive VideoSeal, these image sequences must be
converted into MP4 videos. This script converts each folder of
frame images into a corresponding .mp4 video.

Usage:
    uv run scripts/sav_val_image2video.py --data_path <path_to_JPEGImages>
"""

import os
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Convert JPEG frame folders into MP4 videos."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the root directory containing frame folders.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Directory to save generated videos. Defaults to <data_path>/../videos_24fps",
    )
    parser.add_argument(
        "--fps", type=int, default=24, help="Frame rate for output videos."
    )

    args = parser.parse_args()

    sav_val_jpeg_dir = args.data_path
    output_dir = (
        args.output_path
        if args.output_path is not None
        else os.path.join(os.path.dirname(sav_val_jpeg_dir), "videos_24fps")
    )

    os.makedirs(output_dir, exist_ok=True)

    folders = sorted(os.listdir(sav_val_jpeg_dir))

    for folder in folders:
        folder_path = os.path.join(sav_val_jpeg_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Input sequence pattern: 00000.jpg, 00001.jpg, ...
        input_pattern = os.path.join(folder_path, "%05d.jpg")

        # Output video file
        output_video = os.path.join(output_dir, f"{folder}.mp4")

        # FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_video,
        ]

        print("Creating:", output_video)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
