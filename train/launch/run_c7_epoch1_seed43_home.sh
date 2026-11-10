#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [FC_M_SEED] [C6_C7_SEED]" >&2
  echo "Default: $0 42 43" >&2
  exit 2
fi

fc_m_seed="${1:-42}"
c6_c7_seed="${2:-43}"
c6_seed=42
for seed_value in "$fc_m_seed" "$c6_c7_seed"; do
  if [[ ! "$seed_value" =~ ^[0-9]+$ ]]; then
    echo "Seeds must be non-negative integers: $seed_value" >&2
    exit 2
  fi
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
robottest_dir="$(cd "$script_dir/../.." && pwd)"
home_combined_dataset="${ROBO_HOME_COMBINED_DATASET_DIR:-/home/yaoxt/RAL-rb-data/training_explore}"
combined_train_manifest="${ROBO_HOME_COMBINED_TRAIN_MANIFEST:-$robottest_dir/protocol/manifests/combined_train.json}"
combined_validation_manifest="${ROBO_HOME_COMBINED_VALIDATION_MANIFEST:-$robottest_dir/protocol/manifests/combined_validation.json}"
combined_run_root="${ROBO_HOME_COMBINED_RUN_ROOT:-/data/yaoxt/RoboRuns/rebuttal_P0_combined_home_data}"

for required_path in \
  "$home_combined_dataset/ep_start_end_ids.npy" \
  "$combined_train_manifest" \
  "$combined_validation_manifest" \
  "$script_dir/run_fc_m_lessaggressiv_seed.sh" \
  "$script_dir/run_f0_seed.sh"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done

wait_for_gpu_idle() {
  local active_pids
  while true; do
    active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
    if [[ -z "$active_pids" ]]; then
      break
    fi
    echo "$(date '+%F %T') waiting for GPU compute jobs to finish: ${active_pids//$'\n'/, }"
    sleep 30
  done
}

echo "Queue plan:"
echo "  1. Wait until the current GPU compute jobs finish."
# echo "  2. Train FC-M seed $fc_m_seed and stop after completed epoch 1."

echo "  2. Train C7 seed $c6_c7_seed and stop after completed epoch 1."
echo "  3. Train C6 seed $c6_seed and stop after completed epoch 1."
echo "  4. C6/C7 dataset: $home_combined_dataset"

wait_for_gpu_idle

# echo "$(date '+%F %T') starting FC-M seed $fc_m_seed for one completed epoch"
# ROBO_NUM_EPOCHS=3 \
# ROBO_STOP_AFTER_EPOCHS=1 \
# bash "$script_dir/run_fc_m_lessaggressiv_seed.sh" "$fc_m_seed"




ROBO_COMBINED_DATASET_DIR="$home_combined_dataset" \
ROBO_COMBINED_TRAIN_MANIFEST="$combined_train_manifest" \
ROBO_COMBINED_VALIDATION_MANIFEST="$combined_validation_manifest" \
ROBO_COMBINED_RUN_ROOT="$combined_run_root" \
ROBO_NUM_EPOCHS=3 \
ROBO_STOP_AFTER_EPOCHS=1 \
bash "$script_dir/run_f0_seed.sh" C7 "$c6_c7_seed"

echo "$(date '+%F %T') C7 epoch 1 completed and saved; starting C6 for one epoch"

ROBO_COMBINED_DATASET_DIR="$home_combined_dataset" \
ROBO_COMBINED_TRAIN_MANIFEST="$combined_train_manifest" \
ROBO_COMBINED_VALIDATION_MANIFEST="$combined_validation_manifest" \
ROBO_COMBINED_RUN_ROOT="$combined_run_root" \
ROBO_NUM_EPOCHS=3 \
ROBO_STOP_AFTER_EPOCHS=1 \
bash "$script_dir/run_f0_seed.sh" C6 "$c6_seed"

echo "$(date '+%F %T') C7 epoch 1 completed and saved; queue finished"
