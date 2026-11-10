#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launcher="$script_dir/run_e0_e2_seed.sh"
seed="${ROBO_TRAINING_SEED:-44}"

export ROBO_NUM_EPOCHS="${ROBO_NUM_EPOCHS:-3}"
export ROBO_STOP_AFTER_EPOCHS="${ROBO_STOP_AFTER_EPOCHS:--1}"
export ROBO_BATCH_SIZE_CALVIN="${ROBO_BATCH_SIZE_CALVIN:-12}"
export ROBO_GRADIENT_ACCUMULATION_STEPS="${ROBO_GRADIENT_ACCUMULATION_STEPS:-1}"
export ROBO_LEARNING_RATE="${ROBO_LEARNING_RATE:-1e-5}"
export ROBO_NPROC_PER_NODE="${ROBO_NPROC_PER_NODE:-1}"
export ROBO_DISABLE_GPU_LOCK=1

for experiment in E3 E5 E7; do
  echo "$(date '+%F %T') starting ${experiment} seed${seed}"
  bash "$launcher" "$experiment" "$seed"
  echo "$(date '+%F %T') ${experiment} seed${seed} completed"
done

echo "$(date '+%F %T') chain E3 -> E5 -> E7 completed"
