#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 E0..E7|C0..C7 TRAINING_SEED" >&2
  exit 2
fi

experiment="$1"
training_seed="$2"

if [[ ! "$experiment" =~ ^([EC])([0-7])$ ]]; then
  echo "Unsupported experiment: $experiment (expected E0..E7 or C0..C7)" >&2
  exit 2
fi

setting_prefix="${BASH_REMATCH[1]}"
cell_index="${BASH_REMATCH[2]}"
case "$setting_prefix" in
  E) setting_kind="7object" ;;
  C) setting_kind="combined" ;;
esac

# Factor encoding shared by 7-Object (E*) and Combined-Explore (C*):
# initialization x feedback x sampling.
case "$cell_index" in
  0)
    feedback_type="train_feedback"
    foundation_kind="openflamingo"
    sampling_kind="non_episodic"
    ;;
  1)
    feedback_type="train_feedback"
    foundation_kind="calvin_pretrained"
    sampling_kind="non_episodic"
    ;;
  2)
    feedback_type="alt_feedback"
    foundation_kind="openflamingo"
    sampling_kind="non_episodic"
    ;;
  3)
    feedback_type="train_feedback"
    foundation_kind="openflamingo"
    sampling_kind="episodic"
    ;;
  4)
    feedback_type="alt_feedback"
    foundation_kind="calvin_pretrained"
    sampling_kind="non_episodic"
    ;;
  5)
    feedback_type="train_feedback"
    foundation_kind="calvin_pretrained"
    sampling_kind="episodic"
    ;;
  6)
    feedback_type="alt_feedback"
    foundation_kind="openflamingo"
    sampling_kind="episodic"
    ;;
  7)
    feedback_type="alt_feedback"
    foundation_kind="calvin_pretrained"
    sampling_kind="episodic"
    ;;
esac

if [[ ! "$training_seed" =~ ^[0-9]+$ ]]; then
  echo "Training seed must be an integer: $training_seed" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_robottest_dir="$(cd "$script_dir/../.." && pwd)"

# Optional queue control: entries are exact EXPERIMENT:SEED pairs, one per
# line. This is checked only when a new experiment is launched, so adding an
# entry does not interrupt a training process that is already running.
skip_experiments_file="${ROBO_SKIP_EXPERIMENTS_FILE:-$script_dir/skip_experiments.txt}"
if [[ -f "$skip_experiments_file" ]] && grep -Fqx "${experiment}:${training_seed}" "$skip_experiments_file"; then
  echo "Skipping ${experiment} seed${training_seed}: listed in $skip_experiments_file"
  exit 0
fi

# Every machine-specific path can be overridden without editing this frozen
# launcher. See README_814.md for the transfer and launch procedure.
robottest_dir="${ROBO_TEST_DIR:-$default_robottest_dir}"
case "$setting_kind" in
  7object)
    dataset_dir="${ROBO_7OBJ_DATASET_DIR:-/data/yaoxt/pn36ce/pn36ce-dss-0000/training_7_objs_pos_4}"
    train_manifest="${ROBO_7OBJ_TRAIN_MANIFEST:-$robottest_dir/protocol/manifests/7object_train.json}"
    validation_manifest="${ROBO_7OBJ_VALIDATION_MANIFEST:-$robottest_dir/protocol/manifests/7object_validation.json}"
    run_root="${ROBO_REBUTTAL_RUN_ROOT:-/data/yaoxt/RoboRuns/rebuttal_P0_7object}"
    protocol_name="MetaCALVIN-RA-L-P0-7Object-v1"
    setting_label="7Object"
    train_num_samples=163104
    ;;
  combined)
    dataset_dir="${ROBO_COMBINED_DATASET_DIR:-/data/yaoxt/pn36ce/pn36ce-dss-0000/training_explore}"
    train_manifest="${ROBO_COMBINED_TRAIN_MANIFEST:-$robottest_dir/protocol/manifests/combined_train.json}"
    validation_manifest="${ROBO_COMBINED_VALIDATION_MANIFEST:-$robottest_dir/protocol/manifests/combined_validation.json}"
    run_root="${ROBO_COMBINED_RUN_ROOT:-/data/yaoxt/RoboRuns/rebuttal_P0_combined}"
    protocol_name="MetaCALVIN-RA-L-P0-Combined-v1"
    setting_label="Combined"
    train_num_samples=136656
    ;;
esac
case "$foundation_kind" in
  openflamingo)
    foundation_checkpoint="${ROBO_OPENFLAMINGO_CHECKPOINT:-/data/yaoxt/RoboData/RoboCheckpoint/open_flamingo_checkpoint.pt}"
    ;;
  calvin_pretrained)
    foundation_checkpoint="${ROBO_CALVIN_PRETRAINED_CHECKPOINT:-/data/yaoxt/RoboData/RoboCheckpoint/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth}"
    ;;
esac
num_epochs="${ROBO_NUM_EPOCHS:-3}"
if ! [[ "$num_epochs" =~ ^[1-9][0-9]*$ ]]; then
  echo "ROBO_NUM_EPOCHS must be a positive integer: $num_epochs" >&2
  exit 2
fi
stop_after_epochs="${ROBO_STOP_AFTER_EPOCHS:--1}"
if ! [[ "$stop_after_epochs" =~ ^-1$|^[1-9][0-9]*$ ]]; then
  echo "ROBO_STOP_AFTER_EPOCHS must be -1 or a positive integer: $stop_after_epochs" >&2
  exit 2
fi
batch_size_calvin="${ROBO_BATCH_SIZE_CALVIN:-12}"
gradient_accumulation_steps="${ROBO_GRADIENT_ACCUMULATION_STEPS:-1}"
learning_rate="${ROBO_LEARNING_RATE:-1e-5}"
if ! [[ "$batch_size_calvin" =~ ^[1-9][0-9]*$ ]]; then
  echo "ROBO_BATCH_SIZE_CALVIN must be a positive integer: $batch_size_calvin" >&2
  exit 2
fi
if ! [[ "$gradient_accumulation_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "ROBO_GRADIENT_ACCUMULATION_STEPS must be a positive integer: $gradient_accumulation_steps" >&2
  exit 2
fi
extra_training_args=()
resume_args=()
sampling_args=()
decoder_type="lstm"
if [[ "$sampling_kind" == "episodic" ]]; then
  sampling_args+=(--seq_training)
  decoder_type="lstm_seq"
fi
if [[ "${ROBO_SMOKE_TEST:-0}" == "1" ]]; then
  run_root="${ROBO_SMOKE_RUN_ROOT:-/tmp/metacalvin_rebuttal_smoke}"
  num_epochs=1
  extra_training_args=(
    --max_steps 1
    --max_validation_steps 1
    --validate_every_iter 1
    --skip_checkpoint_saves
  )
fi
if [[ -n "${ROBO_RESUME_FROM_CHECKPOINT:-}" ]]; then
  resume_batch_offset="${ROBO_RESUME_BATCH_OFFSET:--1}"
  if ! [[ "$resume_batch_offset" =~ ^-1$|^[0-9]+$ ]]; then
    echo "ROBO_RESUME_BATCH_OFFSET must be -1 or a non-negative integer: $resume_batch_offset" >&2
    exit 2
  fi
  resume_args=(
    --resume_from_checkpoint "$ROBO_RESUME_FROM_CHECKPOINT"
    --resume_batch_offset "$resume_batch_offset"
  )
fi
echo "Target epochs: $num_epochs"
echo "Stop after completed epochs: $stop_after_epochs"
echo "Batch size per GPU: $batch_size_calvin"
echo "Gradient accumulation steps: $gradient_accumulation_steps"
echo "Requested learning rate: $learning_rate"
run_dir="$run_root/$experiment/seed_${training_seed}"

for required_path in \
  "$dataset_dir/ep_start_end_ids.npy" \
  "$foundation_checkpoint" \
  "$train_manifest" \
  "$validation_manifest"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$run_dir"
export HF_HOME="${HF_HOME:-${ROBO_CACHE_ROOT:-/data/yaoxt/cache}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${ROBO_CACHE_ROOT:-/data/yaoxt/cache}/torch}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "$HF_HOME" "$TORCH_HOME"

# The current workstation has one display-attached GPU. Running E0 and E2 at
# the same time caused a CUDA launch timeout and a driver reset on 2026-08-14.
# The lock prevents another formal launcher from sharing that GPU accidentally.
# Set ROBO_DISABLE_GPU_LOCK=1 only when jobs are assigned to distinct GPUs by
# an external scheduler.
if [[ "${ROBO_DISABLE_GPU_LOCK:-0}" != "1" ]]; then
  gpu_lock_file="${ROBO_GPU_LOCK_FILE:-/data/yaoxt/RoboRuns/.single_gpu_training.lock}"
  mkdir -p "$(dirname "$gpu_lock_file")"
  exec 9>"$gpu_lock_file"
  if ! flock -n 9; then
    echo "Another formal training launcher holds $gpu_lock_file; run jobs sequentially." >&2
    exit 1
  fi
fi

cd "$robottest_dir/train"

torchrun_bin="${ROBO_TORCHRUN_BIN:-torchrun}"
nproc_per_node="${ROBO_NPROC_PER_NODE:-1}"
if ! [[ "$nproc_per_node" =~ ^[1-9][0-9]*$ ]]; then
  echo "ROBO_NPROC_PER_NODE must be a positive integer: $nproc_per_node" >&2
  exit 2
fi
echo "Training processes per node: $nproc_per_node"
"$torchrun_bin" --standalone --nnodes=1 --nproc_per_node="$nproc_per_node" \
  train_meta_calvin.py \
  --experiment_id "${experiment}_${setting_label}_P0_seed${training_seed}" \
  --protocol_name "$protocol_name" \
  --calvin_dataset "$dataset_dir" \
  --calvin_train_manifest "$train_manifest" \
  --calvin_val_manifest "$validation_manifest" \
  --run_name "$run_dir" \
  --checkpoint_path_foundation "$foundation_checkpoint" \
  --hf_revision f0a13e41fcee2217cd701219ffa1eaef7fe955ea \
  --feedback_type "$feedback_type" \
  --seed "$training_seed" \
  --rgb_pad 10 \
  --gripper_pad 4 \
  --use_gripper \
  --traj_cons \
  --fusion_mode post \
  --window_size 12 \
  --cross_attn_every_n_layers 1 \
  --batch_size_calvin "$batch_size_calvin" \
  --gradient_accumulation_steps "$gradient_accumulation_steps" \
  --precision amp_bf16 \
  --num_epochs "$num_epochs" \
  --stop_after_epochs "$stop_after_epochs" \
  --train_num_samples_calvin "$train_num_samples" \
  --logging_steps 20 \
  --workers 4 \
  --decoder_type "$decoder_type" \
  --warmup_steps 1000 \
  --learning_rate "$learning_rate" \
  --lr_scheduler constant \
  --validate_every_iter -1 \
  --save_every_iter 3000 \
  --dense_save_start 9000 \
  --dense_save_end 15000 \
  --dense_save_every_iter 1000 \
  --dist \
  "${sampling_args[@]}" \
  "${resume_args[@]}" \
  "${extra_training_args[@]}" \
  2>&1 | tee -a "$run_dir/train.log"
