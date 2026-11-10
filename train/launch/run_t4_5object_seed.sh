#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 E0|E1|E2|E3|E4|E5|E7|non_episodic|episodic TRAINING_SEED" >&2
  exit 2
fi

sampling_kind="$1"
training_seed="$2"
case "$sampling_kind" in
  E0|e0)
    experiment="T4E0"
    analogue="E0"
    foundation_kind="openflamingo"
    feedback_type="train_feedback"
    sampling_label="non_episodic"
    decoder_type="lstm"
    sampling_args=()
    ;;
  E1|e1)
    experiment="T4E1"
    analogue="E1"
    foundation_kind="calvin_pretrained"
    feedback_type="train_feedback"
    sampling_label="non_episodic"
    decoder_type="lstm"
    sampling_args=()
    ;;
  E2|e2)
    experiment="T4E2"
    analogue="E2"
    foundation_kind="openflamingo"
    feedback_type="alt_feedback"
    sampling_label="non_episodic"
    decoder_type="lstm"
    sampling_args=()
    ;;
  E3|e3)
    experiment="T4E3"
    analogue="E3"
    foundation_kind="openflamingo"
    feedback_type="train_feedback"
    sampling_label="episodic"
    decoder_type="lstm_seq"
    sampling_args=(--seq_training)
    ;;
  E4|e4)
    experiment="T4E4"
    analogue="E4"
    foundation_kind="calvin_pretrained"
    feedback_type="alt_feedback"
    sampling_label="non_episodic"
    decoder_type="lstm"
    sampling_args=()
    ;;
  E5|e5)
    experiment="T4E5"
    analogue="E5"
    foundation_kind="calvin_pretrained"
    feedback_type="train_feedback"
    sampling_label="episodic"
    decoder_type="lstm_seq"
    sampling_args=(--seq_training)
    ;;
  E7|e7)
    experiment="T4E7"
    analogue="E7"
    foundation_kind="calvin_pretrained"
    feedback_type="alt_feedback"
    sampling_label="episodic"
    decoder_type="lstm_seq"
    sampling_args=(--seq_training)
    ;;
  non_episodic)
    experiment="T4E4"
    analogue="E4"
    foundation_kind="calvin_pretrained"
    feedback_type="alt_feedback"
    sampling_label="non_episodic"
    decoder_type="lstm"
    sampling_args=()
    ;;
  episodic)
    experiment="T4E7"
    analogue="E7"
    foundation_kind="calvin_pretrained"
    feedback_type="alt_feedback"
    sampling_label="episodic"
    decoder_type="lstm_seq"
    sampling_args=(--seq_training)
    ;;
  *)
    echo "Unsupported sampling mode: $sampling_kind" >&2
    exit 2
    ;;
esac

if [[ ! "$training_seed" =~ ^[0-9]+$ ]]; then
  echo "Training seed must be an integer: $training_seed" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_robottest_dir="$(cd "$script_dir/../.." && pwd)"
robottest_dir="${ROBO_TEST_DIR:-$default_robottest_dir}"
dataset_dir="${ROBO_5OBJ_DATASET_DIR:-/data/yaoxt/pn36ce/pn36ce-dss-0000/training_5_objs}"
train_manifest="${ROBO_5OBJ_TRAIN_MANIFEST:-$robottest_dir/protocol/manifests/5object_train.json}"
validation_manifest="${ROBO_5OBJ_VALIDATION_MANIFEST:-$robottest_dir/protocol/manifests/5object_validation.json}"
case "$foundation_kind" in
  openflamingo)
    foundation_checkpoint="${ROBO_OPENFLAMINGO_CHECKPOINT:-/data/yaoxt/RoboData/RoboCheckpoint/open_flamingo_checkpoint.pt}"
    ;;
  calvin_pretrained)
    foundation_checkpoint="${ROBO_CALVIN_PRETRAINED_CHECKPOINT:-/data/yaoxt/RoboData/RoboCheckpoint/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth}"
    ;;
esac
run_root="${ROBO_5OBJ_RUN_ROOT:-/home/yaoxt/RoboRuns/rebuttal_P0_5object_T4_E0E3E4E7}"
protocol_name="MetaCALVIN-RA-L-P0-5Object-T4-E0E3E4E7-v1"

# The train manifest contains 115,252 valid ws=12 windows.  Four are dropped
# deterministically to make a complete batch of 12, matching the Non-Episodic
# DataLoader.  Episodic training receives the same optimizer-step budget.
train_num_samples=115248
concurrency_profile="${ROBO_CONCURRENCY_PROFILE:-standalone}"
case "$concurrency_profile" in
  standalone)
    default_batch_size_calvin=12
    default_gradient_accumulation_steps=1
    default_learning_rate=1e-5
    ;;
  c7_parallel)
    # C7 occupies roughly 53 GiB on the only 96 GiB GPU.  A one-sample
    # probe and a six-sample probe both fit alongside it.  Six is preferable:
    # the 115,252-window manifest drops exactly four windows, preserving the
    # standalone budget of 115,248 windows and 9,604 optimizer steps/epoch.
    default_batch_size_calvin=6
    default_gradient_accumulation_steps=2
    # train_meta_calvin scales this requested LR by micro-batch / 6.
    # 2e-5 * 6 / 6 = 2e-5, equal to standalone's 1e-5 * 12 / 6.
    default_learning_rate=2e-5
    ;;
  *)
    echo "Unsupported ROBO_CONCURRENCY_PROFILE: $concurrency_profile" >&2
    exit 2
    ;;
esac
batch_size_calvin="${ROBO_BATCH_SIZE_CALVIN:-$default_batch_size_calvin}"
gradient_accumulation_steps="${ROBO_GRADIENT_ACCUMULATION_STEPS:-$default_gradient_accumulation_steps}"
learning_rate="${ROBO_LEARNING_RATE:-$default_learning_rate}"
num_epochs="${ROBO_NUM_EPOCHS:-3}"
stop_after_epochs="${ROBO_STOP_AFTER_EPOCHS:--1}"
save_every_iter="${ROBO_SAVE_EVERY_ITER:-3000}"

if ! [[ "$batch_size_calvin" =~ ^[1-9][0-9]*$ ]]; then
  echo "ROBO_BATCH_SIZE_CALVIN must be a positive integer: $batch_size_calvin" >&2
  exit 2
fi
if ! [[ "$gradient_accumulation_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "ROBO_GRADIENT_ACCUMULATION_STEPS must be a positive integer." >&2
  exit 2
fi
if [[ $((batch_size_calvin * gradient_accumulation_steps)) -ne 12 ]]; then
  echo "The frozen T4 protocol requires micro-batch * accumulation = 12." >&2
  exit 2
fi
if ! [[ "$num_epochs" =~ ^[1-9][0-9]*$ ]]; then
  echo "ROBO_NUM_EPOCHS must be a positive integer: $num_epochs" >&2
  exit 2
fi
if ! [[ "$stop_after_epochs" =~ ^-1$|^[1-9][0-9]*$ ]]; then
  echo "ROBO_STOP_AFTER_EPOCHS must be -1 or a positive integer." >&2
  exit 2
fi
if ! [[ "$save_every_iter" =~ ^[1-9][0-9]*$ ]]; then
  echo "ROBO_SAVE_EVERY_ITER must be a positive integer: $save_every_iter" >&2
  exit 2
fi

extra_training_args=()
resume_args=()
if [[ "${ROBO_SMOKE_TEST:-0}" == "1" ]]; then
  run_root="${ROBO_5OBJ_SMOKE_RUN_ROOT:-/tmp/metacalvin_t4_5object_smoke}"
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
    echo "ROBO_RESUME_BATCH_OFFSET must be -1 or a non-negative integer." >&2
    exit 2
  fi
  resume_args=(
    --resume_from_checkpoint "$ROBO_RESUME_FROM_CHECKPOINT"
    --resume_batch_offset "$resume_batch_offset"
  )
fi

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

run_dir="$run_root/$experiment/seed_${training_seed}"
mkdir -p "$run_dir"
export HF_HOME="${HF_HOME:-${ROBO_CACHE_ROOT:-/data/yaoxt/cache}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${ROBO_CACHE_ROOT:-/data/yaoxt/cache}/torch}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "$HF_HOME" "$TORCH_HOME"

if [[ "${ROBO_DISABLE_GPU_LOCK:-0}" != "1" ]]; then
  gpu_lock_file="${ROBO_GPU_LOCK_FILE:-/data/yaoxt/RoboRuns/.single_gpu_training.lock}"
  mkdir -p "$(dirname "$gpu_lock_file")"
  exec 9>"$gpu_lock_file"
  if ! flock -n 9; then
    echo "Another formal training launcher holds $gpu_lock_file." >&2
    exit 1
  fi
fi

echo "Experiment: $experiment ($analogue-style, $sampling_label)"
echo "Foundation: $foundation_kind"
echo "Feedback: $feedback_type"
echo "Protocol: $protocol_name"
echo "Training seed: $training_seed"
echo "Concurrency profile: $concurrency_profile"
echo "Micro-batch / accumulation: $batch_size_calvin / $gradient_accumulation_steps"
echo "Target epochs: $num_epochs"
echo "Periodic checkpoint interval: every $save_every_iter completed global batches"
echo "Training windows per epoch: $train_num_samples"
echo "Run directory: $run_dir"

cd "$robottest_dir/train"
default_torchrun="/home/yaoxt/.conda/envs/meta-calvin-bw/bin/torchrun"
torchrun_bin="${ROBO_TORCHRUN_BIN:-$default_torchrun}"
nproc_per_node="${ROBO_NPROC_PER_NODE:-1}"
if ! [[ "$nproc_per_node" =~ ^[1-9][0-9]*$ ]]; then
  echo "ROBO_NPROC_PER_NODE must be a positive integer." >&2
  exit 2
fi

"$torchrun_bin" --standalone --nnodes=1 --nproc_per_node="$nproc_per_node" \
  train_meta_calvin.py \
  --experiment_id "${experiment}_5Object_P0_seed${training_seed}" \
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
  --save_every_iter "$save_every_iter" \
  --dense_save_start -1 \
  --dense_save_end -1 \
  --dense_save_every_iter -1 \
  --dist \
"${sampling_args[@]}" \
  "${resume_args[@]}" \
  "${extra_training_args[@]}" \
  2>&1 | tee -a "$run_dir/train.log"
