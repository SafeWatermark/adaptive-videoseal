#!/bin/bash
EXPERIMENT_SCRIPT="src/main.py"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

python "$EXPERIMENT_SCRIPT" \
  --scene_detector histogram \
  --device auto \
  --seed 42 \
  --input_dir ./data \
  --output_dir "./outputs/${TIMESTAMP}_histogram" \
  --video_filename sample_movie.mp4 \
  --n_trials 30 \
  --num_frames_for_extraction 32 \
  --chunk_size 16 \
  --step_size 8 \
  --scaling_w 1.5
