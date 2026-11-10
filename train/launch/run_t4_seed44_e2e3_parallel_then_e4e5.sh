#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ROBO_5OBJ_DATASET_DIR="${ROBO_5OBJ_DATASET_DIR:-/home/yaoxt/RAL-rb-data/training_5_objs}"
export ROBO_5OBJ_RUN_ROOT="${ROBO_5OBJ_RUN_ROOT:-/data/yaoxt/RoboRuns/rebuttal_P0_5object_T4_E2E3E4E5_seed44}"
export ROBO_NUM_EPOCHS="${ROBO_NUM_EPOCHS:-3}"
export ROBO_STOP_AFTER_EPOCHS="${ROBO_STOP_AFTER_EPOCHS:-1}"
export ROBO_SAVE_EVERY_ITER="${ROBO_SAVE_EVERY_ITER:-3000}"
export ROBO_CONCURRENCY_PROFILE="${ROBO_CONCURRENCY_PROFILE:-standalone}"

echo "$(date '+%F %T') starting T4E2 and T4E3 in parallel (seed 44)"

ROBO_DISABLE_GPU_LOCK=1 \
  bash "$script_dir/run_t4_5object_seed.sh" E2 44 &
e2_pid=$!

ROBO_DISABLE_GPU_LOCK=1 \
  bash "$script_dir/run_t4_5object_seed.sh" E3 44 &
e3_pid=$!

e2_status=0
e3_status=0
wait "$e2_pid" || e2_status=$?
wait "$e3_pid" || e3_status=$?

if (( e2_status != 0 || e3_status != 0 )); then
  echo "E2/E3 failed: E2 status=$e2_status, E3 status=$e3_status" >&2
  exit 1
fi

echo "$(date '+%F %T') T4E2 and T4E3 completed; starting T4E4 (seed 44)"
bash "$script_dir/run_t4_5object_seed.sh" E4 44

echo "$(date '+%F %T') T4E4 completed; starting T4E5 (seed 44)"
bash "$script_dir/run_t4_5object_seed.sh" E5 44

echo "$(date '+%F %T') completed T4 seed-44 chain: E2 || E3 -> E4 -> E5"
