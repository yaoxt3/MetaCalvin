#!/usr/bin/env bash
set -euo pipefail

# Run two 7-object F0 chains in parallel.  Experiments inside each chain are
# serial, so E4 starts only after E2 succeeds and E5 starts only after E3
# succeeds:
#
#   E2 seed44 -> E4 seed44
#   E3 seed44 -> E5 seed44

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launcher="$script_dir/run_e0_e2_seed.sh"
seed="${ROBO_TRAINING_SEED:-44}"

# Preserve the canonical 7-object training hyperparameters unless the caller
# explicitly overrides them.
export ROBO_NUM_EPOCHS="${ROBO_NUM_EPOCHS:-3}"
export ROBO_STOP_AFTER_EPOCHS="${ROBO_STOP_AFTER_EPOCHS:--1}"
export ROBO_BATCH_SIZE_CALVIN="${ROBO_BATCH_SIZE_CALVIN:-12}"
export ROBO_GRADIENT_ACCUMULATION_STEPS="${ROBO_GRADIENT_ACCUMULATION_STEPS:-1}"
export ROBO_LEARNING_RATE="${ROBO_LEARNING_RATE:-1e-5}"
export ROBO_NPROC_PER_NODE="${ROBO_NPROC_PER_NODE:-1}"

# Both chains intentionally share the workstation GPU.  Do not launch this
# wrapper while unrelated training processes are occupying most of the GPU.
export ROBO_DISABLE_GPU_LOCK=1

run_chain() {
  local first="$1"
  local second="$2"

  echo "$(date '+%F %T') starting ${first} seed${seed}"
  bash "$launcher" "$first" "$seed"
  echo "$(date '+%F %T') ${first} completed; starting ${second} seed${seed}"
  bash "$launcher" "$second" "$seed"
  echo "$(date '+%F %T') chain ${first} -> ${second} completed"
}

echo "$(date '+%F %T') starting parallel 7-object seed${seed} chains"
echo "chain A: E2 -> E4"
echo "chain B: E3 -> E5"

run_chain E2 E4 &
chain_a_pid=$!

run_chain E3 E5 &
chain_b_pid=$!

status=0
if ! wait "$chain_a_pid"; then
  echo "$(date '+%F %T') ERROR: chain E2 -> E4 failed" >&2
  status=1
fi
if ! wait "$chain_b_pid"; then
  echo "$(date '+%F %T') ERROR: chain E3 -> E5 failed" >&2
  status=1
fi

if [[ "$status" -eq 0 ]]; then
  echo "$(date '+%F %T') all four 7-object seed${seed} experiments completed"
fi
exit "$status"
