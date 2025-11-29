#!/bin/bash

# Script to run VideoMarkBench evaluation
# Usage: bash scripts/run_videomarkbench.sh [model] [max_videos]
#   model: videoseal (default) or adaptive-videoseal
#   max_videos: number of videos to evaluate (default: 5)

EXPERIMENT_SCRIPT="src/videomarkbench/main.py"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

MODEL=${1:-videoseal}
MAX_VIDEOS=${2:-5}

python "$EXPERIMENT_SCRIPT" \
  --model "$MODEL" \
  --dataset data/videomarkbench \
  --output_dir "./outputs/videomarkbench_${TIMESTAMP}" \
  --max_videos "$MAX_VIDEOS" \
  --perturbations common \
  --device auto \
  --seed 42 \
  --step_size 8 \
  --scaling_w 1.5 \
  --video_mode repeat \
  --chunk_size 32
