#!/bin/bash

# Script to run comprehensive VideoMarkBench evaluation
# Usage: bash scripts/run_videomarkbench.sh [model] [max_videos]
#   model: videoseal (default) or adaptive-videoseal
#   max_videos: number of videos to evaluate (default: 5)
#
# This script evaluates the model under:
# - Common perturbations (JPEG, Gaussian, Crop, MPEG4, etc.)
# - Both removal and forgery attacks for each perturbation
# - All aggregation strategies from original VideoMarkBench
#
# Perturbations evaluated:
# - JPEG (quality: 90, 75, 50)
# - Gaussian noise (std: 0.01, 0.05)
# - Gaussian blur (std: 1.0, 2.0)
# - Crop (crop_ratio: 0.9, 0.8)
# - MPEG4 compression (CRF: 23, 28, 33)
# - FrameAverage (num_frames: 2, 3)
# - FrameRemove (p: 0.1, 0.2)
# - FrameSwitch (p: 0.1, 0.2)
#
# Note: This script uses the VideoMarkBench submodule at src/videomarkbench/
# Make sure to initialize submodules: git submodule update --init --recursive

EXPERIMENT_SCRIPT="src/evaluate/main.py"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

MODEL=${1:-videoseal}
MAX_VIDEOS=${2:-5}

# Ensure submodule is initialized
# Check for .git file/symlink instead of just directory existence
# Submodules always have .git file/symlink when properly initialized
# Use -e to check for file, directory, or symlink
if [ ! -e "src/videomarkbench/.git" ]; then
    echo "Initializing VideoMarkBench submodule..."
    git submodule update --init --recursive
fi

PYTHONPATH=src uv run python "$EXPERIMENT_SCRIPT" \
  --model "$MODEL" \
  --dataset data/videomarkbench \
  --output_dir "./outputs/videomarkbench_${TIMESTAMP}" \
  --max_videos "$MAX_VIDEOS" \
  --evaluate_common \
  --device auto \
  --seed 42 \
  --step_size 8 \
  --scaling_w 1.5 \
  --video_mode repeat \
  --chunk_size 32
