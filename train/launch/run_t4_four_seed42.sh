#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ROBO_5OBJ_DATASET_DIR="${ROBO_5OBJ_DATASET_DIR:-/home/yaoxt/RAL-rb-data/training_5_objs}"
export ROBO_5OBJ_RUN_ROOT="${ROBO_5OBJ_RUN_ROOT:-/home/yaoxt/RoboRuns/rebuttal_P0_5object_T4_E4E7E0E3_SSD_20260820}"
export ROBO_NUM_EPOCHS="${ROBO_NUM_EPOCHS:-3}"
export ROBO_STOP_AFTER_EPOCHS="${ROBO_STOP_AFTER_EPOCHS:-1}"
export ROBO_SAVE_EVERY_ITER="${ROBO_SAVE_EVERY_ITER:-3000}"
export ROBO_CONCURRENCY_PROFILE="${ROBO_CONCURRENCY_PROFILE:-standalone}"

for condition in E4 E7 E0 E3; do
  echo "$(date '+%F %T') starting ${condition}-style 5-object training"
  bash "$script_dir/run_t4_5object_seed.sh" "$condition" 42
  echo "$(date '+%F %T') completed ${condition}-style 5-object training"
done

echo "$(date '+%F %T') completed T4E4 -> T4E7 -> T4E0 -> T4E3 chain"
