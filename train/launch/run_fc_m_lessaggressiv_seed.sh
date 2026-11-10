#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 TRAINING_SEED" >&2
  echo "Example: $0 42" >&2
  exit 2
fi

training_seed="$1"
if [[ ! "$training_seed" =~ ^[0-9]+$ ]]; then
  echo "TRAINING_SEED must be a non-negative integer: $training_seed" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
robottest_dir="$(cd "$script_dir/../.." && pwd)"

# F1 FC-M: Combined Minimal data. Each retained rollout completes at least one
# subtask. The manifests exactly match the original C0 train/validation episode
# and frame totals while preserving the frozen 16/4/4 task-pair split.
dataset_dir="${ROBO_COMBINED_LESSAGGRESSIV_DATASET_DIR:-/data/yaoxt/pn36ce/pn36ce-dss-0000/training_explore_lessaggressiv}"
manifest_dir="${ROBO_LESSAGGRESSIV_MANIFEST_DIR:-$robottest_dir/protocol/manifests_lessaggressiv}"
train_manifest="$manifest_dir/combined_train_lessaggressiv.json"
validation_manifest="$manifest_dir/combined_validation_lessaggressiv.json"
run_root="${ROBO_FC_M_LESSAGGRESSIV_RUN_ROOT:-/data/yaoxt/RoboRuns/rebuttal_F1_FC_M_lessaggressiv}"

for required_path in \
  "$dataset_dir/ep_start_end_ids.npy" \
  "$train_manifest" \
  "$validation_manifest"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done

export ROBO_COMBINED_DATASET_DIR="$dataset_dir"
export ROBO_COMBINED_TRAIN_MANIFEST="$train_manifest"
export ROBO_COMBINED_VALIDATION_MANIFEST="$validation_manifest"
export ROBO_COMBINED_RUN_ROOT="$run_root"
export ROBO_NUM_EPOCHS="${ROBO_NUM_EPOCHS:-3}"
export ROBO_STOP_AFTER_EPOCHS="${ROBO_STOP_AFTER_EPOCHS:--1}"

echo "F1 condition: FC-M (Combined Minimal / lessaggressiv)"
echo "Training seed: $training_seed"
echo "Dataset: $dataset_dir"
echo "Train manifest: $train_manifest"
echo "Validation manifest: $validation_manifest"
echo "Run directory: $run_root/C0/seed_$training_seed"
echo "Epochs: $ROBO_NUM_EPOCHS"
echo "C0 below selects the frozen C0 model/feedback/sampling configuration."

# Reuse the frozen C0 configuration without modifying the original launcher.
exec bash "$script_dir/run_f0_seed.sh" C0 "$training_seed"
