#!/usr/bin/env bash
set -euo pipefail

# Two parallel 7-object F0 chains, with strict serial execution inside each:
#
#   E2 seed44 -> E4 seed44 -> E6 seed44
#   E3 seed44 -> E5 seed44 -> E7 seed44

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launcher="$script_dir/run_e0_e2_seed.sh"
seed="${ROBO_TRAINING_SEED:-44}"

# Canonical 7-object settings. Callers may override these variables explicitly.
export ROBO_NUM_EPOCHS="${ROBO_NUM_EPOCHS:-3}"
export ROBO_STOP_AFTER_EPOCHS="${ROBO_STOP_AFTER_EPOCHS:--1}"
export ROBO_BATCH_SIZE_CALVIN="${ROBO_BATCH_SIZE_CALVIN:-12}"
export ROBO_GRADIENT_ACCUMULATION_STEPS="${ROBO_GRADIENT_ACCUMULATION_STEPS:-1}"
export ROBO_LEARNING_RATE="${ROBO_LEARNING_RATE:-1e-5}"
export ROBO_NPROC_PER_NODE="${ROBO_NPROC_PER_NODE:-1}"

# The two chains intentionally share the workstation GPU.
export ROBO_DISABLE_GPU_LOCK=1

run_chain() {
  local experiment
  for experiment in "$@"; do
    echo "$(date '+%F %T') starting ${experiment} seed${seed}"
    bash "$launcher" "$experiment" "$seed"
    echo "$(date '+%F %T') ${experiment} seed${seed} completed"
  done
}

echo "$(date '+%F %T') starting parallel 7-object seed${seed} chains"
echo "chain A: E2 -> E4 -> E6"
echo "chain B: E3 -> E5 -> E7"

run_chain E2 E4 E6 &
chain_a_pid=$!

run_chain E3 E5 E7 &
chain_b_pid=$!

status=0
if ! wait "$chain_a_pid"; then
  echo "$(date '+%F %T') ERROR: chain E2 -> E4 -> E6 failed" >&2
  status=1
fi
if ! wait "$chain_b_pid"; then
  echo "$(date '+%F %T') ERROR: chain E3 -> E5 -> E7 failed" >&2
  status=1
fi

if [[ "$status" -eq 0 ]]; then
  echo "$(date '+%F %T') all six 7-object seed${seed} experiments completed"
fi
exit "$status"
