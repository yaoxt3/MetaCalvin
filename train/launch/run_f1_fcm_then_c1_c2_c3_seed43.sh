#!/usr/bin/env bash
set -euo pipefail

training_seed=43

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
robottest_dir="$(cd "$script_dir/../.." && pwd)"

# Combined F0 cells use the SSD copy. FC-M selects its own newly collected
# Minimal-filter dataset and frozen manifests in run_fc_m_lessaggressiv_seed.sh.
combined_dataset="${ROBO_HOME_COMBINED_DATASET_DIR:-/home/yaoxt/RAL-rb-data/training_explore}"
combined_train_manifest="${ROBO_HOME_COMBINED_TRAIN_MANIFEST:-$robottest_dir/protocol/manifests/combined_train.json}"
combined_validation_manifest="${ROBO_HOME_COMBINED_VALIDATION_MANIFEST:-$robottest_dir/protocol/manifests/combined_validation.json}"
combined_run_root="${ROBO_HOME_COMBINED_RUN_ROOT:-/data/yaoxt/RoboRuns/rebuttal_P0_combined_home_data}"

for required_path in \
  "$combined_dataset/ep_start_end_ids.npy" \
  "$combined_train_manifest" \
  "$combined_validation_manifest" \
  "$script_dir/run_fc_m_lessaggressiv_seed.sh" \
  "$script_dir/run_f0_seed.sh"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done

echo "Serial training plan (one completed epoch each):"
echo "  1. F1 FC-M seed $training_seed"
echo "  2. C1 seed $training_seed"
echo "  3. C2 seed $training_seed"
echo "  4. C3 seed $training_seed"
echo "Combined dataset: $combined_dataset"

echo "$(date '+%F %T') starting F1 FC-M seed $training_seed"
ROBO_NUM_EPOCHS=3 \
ROBO_STOP_AFTER_EPOCHS=1 \
bash "$script_dir/run_fc_m_lessaggressiv_seed.sh" "$training_seed"

for experiment in C1 C2 C3; do
  echo "$(date '+%F %T') starting $experiment seed $training_seed"
  ROBO_COMBINED_DATASET_DIR="$combined_dataset" \
  ROBO_COMBINED_TRAIN_MANIFEST="$combined_train_manifest" \
  ROBO_COMBINED_VALIDATION_MANIFEST="$combined_validation_manifest" \
  ROBO_COMBINED_RUN_ROOT="$combined_run_root" \
  ROBO_NUM_EPOCHS=3 \
  ROBO_STOP_AFTER_EPOCHS=1 \
  bash "$script_dir/run_f0_seed.sh" "$experiment" "$training_seed"
done

echo "$(date '+%F %T') FC-M, C1, C2, and C3 seed $training_seed completed epoch 1"
