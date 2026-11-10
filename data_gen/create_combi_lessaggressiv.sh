#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
robottest_dir="$(cd "$script_dir/.." && pwd)"

blackwell_python="/home/yaoxt/.conda/envs/meta-calvin-bw/bin/python"
if [[ -x "$blackwell_python" ]]; then
  default_python="$blackwell_python"
else
  default_python="python"
fi
python_bin="${ROBO_PYTHON_BIN:-$default_python}"
checkpoint_path="${ROBO_EXECUTOR_CHECKPOINT:-/data/yaoxt/RoboData/RoboCheckpoint/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth}"
calvin_dataset="${ROBO_CALVIN_DATASET_PATH:-$robottest_dir/calvin/dataset/calvin_debug_dataset/validation}"
train_reference="${ROBO_COMBINED_TRAIN_MANIFEST:-$robottest_dir/protocol/manifests/combined_train.json}"
validation_reference="${ROBO_COMBINED_VALIDATION_MANIFEST:-$robottest_dir/protocol/manifests/combined_validation.json}"
save_dir="${ROBO_COMBINED_LESSAGGRESSIV_DATASET_DIR:-/data/yaoxt/pn36ce/pn36ce-dss-0000/training_explore_lessaggressiv}"
manifest_dir="${ROBO_LESSAGGRESSIV_MANIFEST_DIR:-$robottest_dir/protocol/manifests_lessaggressiv}"
cache_dir="${ROBO_CACHE_ROOT:-/data/yaoxt/cache}/huggingface"
collection_seed="${ROBO_COLLECTION_SEED:-42}"
max_attempts="${ROBO_MAX_COLLECTION_ATTEMPTS:-0}"
stop_after_episodes="${ROBO_STOP_AFTER_EPISODES:-0}"
report_interval="${ROBO_REPORT_INTERVAL:-1}"

mkdir -p "$cache_dir" "${ROBO_MPLCONFIG_DIR:-$cache_dir/matplotlib}"
export HF_HOME="$cache_dir"
export TRANSFORMERS_CACHE="$cache_dir"
export MPLCONFIGDIR="${ROBO_MPLCONFIG_DIR:-$cache_dir/matplotlib}"
export PYTHONUNBUFFERED=1

# Keep the large model cache on /data, while reusing the user's existing
# Hugging Face login. HF_HOME would otherwise redirect the token lookup too.
existing_hf_token="/home/yaoxt/.cache/huggingface/token"
if [[ -s "$existing_hf_token" ]]; then
  export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$existing_hf_token}"
fi

"$python_bin" - <<'PY'
import accelerate
import bitsandbytes
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in ROBO_PYTHON_BIN")

capability = torch.cuda.get_device_capability()
probe = torch.ones(1, device="cuda") + 1
print(
    "Runtime preflight:",
    f"torch={torch.__version__}",
    f"bitsandbytes={bitsandbytes.__version__}",
    f"GPU={torch.cuda.get_device_name(0)}",
    f"capability=sm_{capability[0]}{capability[1]}",
    f"CUDA probe={probe.item():.0f}",
)
PY

for required_path in \
  "$checkpoint_path" \
  "$calvin_dataset" \
  "$train_reference" \
  "$validation_reference"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done

echo "Collecting one Combined Minimal dataset: $save_dir"
echo "Strict references: $train_reference and $validation_reference"
echo "Matching: task pair + episode count + each episode length"
if [[ "$stop_after_episodes" != "0" ]]; then
  echo "Staged collection will pause at accepted episode: $stop_after_episodes"
fi

"$python_bin" -u "$script_dir/create_data_lessaggressiv.py" \
  --checkpoint_path "$checkpoint_path" \
  --calvin_dataset_path "$calvin_dataset" \
  --reference_train_manifest "$train_reference" \
  --reference_validation_manifest "$validation_reference" \
  --save_dir "$save_dir" \
  --cache_dir "$cache_dir" \
  --seed "$collection_seed" \
  --max_attempts "$max_attempts" \
  --stop_after_episodes "$stop_after_episodes" \
  --use_reasoning \
  --provide_feedback \
  --reset_robot_pos \
  --rollout_steps 1600 \
  --num_trials 4 \
  --num_instructions 2 \
  --report_interval "$report_interval"

state_path="$save_dir/collection_state_lessaggressiv.json"
if ! "$python_bin" -c \
  'import json, sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["complete"] else 1)' \
  "$state_path"; then
  echo "Staged collection paused successfully; final manifests are deferred."
  echo "Resume with ROBO_STOP_AFTER_EPISODES=0 (or a larger absolute target)."
  exit 0
fi

"$python_bin" -u "$robottest_dir/protocol/create_combined_manifests_lessaggressiv.py" \
  --dataset-dir "$save_dir" \
  --reference-train-manifest "$train_reference" \
  --reference-validation-manifest "$validation_reference" \
  --output-dir "$manifest_dir"

echo "Combined Minimal collection and exact-match audit completed."
