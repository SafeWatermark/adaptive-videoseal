#!/bin/bash
EXPERIMENT_SCRIPT="src/main.py"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

python "$EXPERIMENT_SCRIPT" \
  --scene_detector content \
  --device auto \
  --seed 42 \
  --input_dir ./data \
  --output_dir "./outputs/${TIMESTAMP}_content" \
  --video_filename sample_movie.mp4 \
  --n_trials 30 \
  --num_frames 32 \
  --bit_length 16 \
  --step_size 8 \
  --scaling_w 1.5
