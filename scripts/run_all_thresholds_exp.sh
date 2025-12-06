#!/bin/bash
set +e

# List your scripts here
SCRIPTS=(
  "scripts/run_content_threshold_exp.sh"
  "scripts/run_hash_threshold_exp.sh"
  "scripts/run_hist_threshold_exp.sh"
  "scripts/run_lpips_threshold_exp.sh"
)

(
    for script in "${SCRIPTS[@]}"; do
        echo "=========================================="
        echo "Running $script ..."
        echo "=========================================="

        bash "$script"

        echo "$script finished."
        echo
    done

    echo "All experiments completed."
) &

echo "Experiments running sequentially in background (PID: $!)"
