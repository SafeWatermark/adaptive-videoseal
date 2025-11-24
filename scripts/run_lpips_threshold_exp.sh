#!/bin/bash
set +e
EXPERIMENT_SCRIPT="src/threshold_experiment.py"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

uv run "$EXPERIMENT_SCRIPT" \
  --scene_detector lpips \
  --seed 42 \
  --thresholds 0.05 0.075 0.1 0.125 0.15 \
  --step_sizes 8 16 32 64 128 \
  --device auto \
  --dataset sa-v \
  --num_frames 72 \
  --num_samples 30 \
  --video_aggregation avg \
  --attenuation_config configs/attenuation.yaml \
  --videoseal_chunk_size 32 \
  --videoseal_mode repeat \
  --output_dir output/step_size_eval \
  --save_first 1 \
  --only_identity True \
  --skip_image_metrics True \
  --interpolation_mode bilinear \
