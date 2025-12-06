#!/bin/bash
set +e
EXPERIMENT_SCRIPT="src/threshold_experiment.py"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

uv run "$EXPERIMENT_SCRIPT" \
  --scene_detector histogram \
  --seed 42 \
  --thresholds 0.01 0.05 0.1 0.15 0.2 \
  --step_sizes 8 16 32 64 128 \
  --device auto \
  --dataset sa-v \
  --num_frames 72 \
  --num_samples 30 \
  --video_aggregation avg \
  --attenuation_config configs/attenuation.yaml \
  --videoseal_chunk_size 32 \
  --videoseal_mode repeat \
  --output_dir outputs/step_size_eval \
  --only_identity True \
  --save_first 1 \
  --skip_image_metrics True \
  --interpolation_mode bilinear \
