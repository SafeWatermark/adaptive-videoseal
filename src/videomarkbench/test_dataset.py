"""
Simple test script to verify dataset loading works correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from videomarkbench.dataset import VideoMarkBenchDataset


def test_dataset():
    """Test dataset loading."""
    dataset_path = Path("data/videomarkbench")

    if not dataset_path.exists():
        print(f"Dataset directory not found: {dataset_path}")
        print("Please ensure VideoMarkBench dataset is in data/videomarkbench/")
        return False

    print(f"Loading dataset from: {dataset_path}")
    dataset = VideoMarkBenchDataset(dataset_path)

    print(f"\nFound {len(dataset)} videos")

    if len(dataset.videos) == 0:
        print("No videos found! Check dataset path and file structure.")
        return False

    # Show statistics
    stats = dataset.get_statistics()
    print("\nDataset Statistics:")
    print(f"  Total videos: {stats['total_videos']}")
    print(f"  By style: {stats['by_style']}")
    print(f"  By model: {stats['by_model']}")
    print(f"  By transition: {stats['by_transition']}")

    # Show sample videos
    print("\nSample videos (first 5):")
    for i, video_info in enumerate(dataset.videos[:5]):
        print(f"  {i+1}. {video_info.name}")
        print(f"     Model: {video_info.model}, Style: {video_info.style}")
        print(f"     Path: {video_info.path}")

    # Test loading a video
    print("\nTesting video loading...")
    try:
        video, info = dataset[0]
        print(f"  Successfully loaded: {info.name}")
        print(f"  Video shape: {video.shape} (T, C, H, W)")
        print(f"  Video range: [{video.min():.3f}, {video.max():.3f}]")
        return True
    except Exception as e:
        print(f"  Error loading video: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dataset()
    sys.exit(0 if success else 1)
