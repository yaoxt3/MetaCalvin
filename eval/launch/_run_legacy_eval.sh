#!/usr/bin/env bash

set -Eeuo pipefail

if [[ $# -ne 11 && $# -ne 12 ]]; then
    echo "Internal usage: $0 EXPERIMENT TRAINING_SEED FEEDBACK CHECKPOINT WINDOW_SIZE NUM_OBJECTS NUM_TRIALS TRIAL_STEPS NUM_RESETS EVAL_SEED RUN_LABEL [standard|combi]" >&2
    exit 2
fi

experiment=$1
training_seed=$2
feedback=$3
checkpoint=$4
window_size=$5
num_objects=$6
num_trials=$7
trial_steps=$8
num_resets=$9
evaluation_seed=${10}
run_label=${11}
evaluation_mode=${12:-standard}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
robotest_dir=$(cd -- "$script_dir/../.." && pwd)
workspace_dir=$(cd -- "$robotest_dir/../../.." && pwd)

dataset=${ROBO_CALVIN_EVAL_DATASET:-$workspace_dir/calvin-official/dataset/calvin_debug_dataset/validation}
oracle=${ROBO_CALVIN_TASK_ORACLE:-$robotest_dir/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml}
output_root=${ROBO_EVAL_OUTPUT_ROOT:-$workspace_dir/evaluation_runs/legacy_e0_e2}
hf_cache=${ROBO_HF_CACHE:-$workspace_dir/cache/huggingface}
torch_cache=${ROBO_TORCH_CACHE:-$workspace_dir/cache/torch}
record_video=${ROBO_EVAL_RECORD_VIDEO:-1}
video_fps=${ROBO_EVAL_VIDEO_FPS:-30}
decoder_type=${ROBO_EVAL_DECODER_TYPE:-lstm}
gpu_lock=${ROBO_EVAL_GPU_LOCK:-1}
force_gpu_lock=${ROBO_FORCE_GPU_LOCK:-0}
resource_sample_interval=${ROBO_RESOURCE_SAMPLE_INTERVAL:-1.0}
resource_monitor=$robotest_dir/eval/resource_monitor.py

if [[ $force_gpu_lock == 1 ]]; then
    gpu_lock=1
fi

default_python=/home/yxt/miniforge3/envs/meta-calvin-infer/bin/python
if [[ -n ${ROBO_EVAL_PYTHON:-} ]]; then
    python_bin=$ROBO_EVAL_PYTHON
elif [[ -x $default_python ]]; then
    python_bin=$default_python
else
    python_bin=$(command -v python)
fi

for required_file in "$checkpoint" "$oracle" "$robotest_dir/eval/evaluate.py" "$resource_monitor"; do
    if [[ ! -f $required_file ]]; then
        echo "Required file does not exist: $required_file" >&2
        exit 1
    fi
done
if [[ ! -d $dataset ]]; then
    echo "CALVIN validation dataset does not exist: $dataset" >&2
    exit 1
fi
if [[ ! -x $python_bin ]]; then
    echo "Python executable is not usable: $python_bin" >&2
    exit 1
fi
if [[ $record_video != 0 && $record_video != 1 ]]; then
    echo "ROBO_EVAL_RECORD_VIDEO must be 0 or 1, got: $record_video" >&2
    exit 2
fi
if [[ $gpu_lock != 0 && $gpu_lock != 1 ]]; then
    echo "ROBO_EVAL_GPU_LOCK must be 0 or 1, got: $gpu_lock" >&2
    exit 2
fi
if [[ $force_gpu_lock != 0 && $force_gpu_lock != 1 ]]; then
    echo "ROBO_FORCE_GPU_LOCK must be 0 or 1, got: $force_gpu_lock" >&2
    exit 2
fi
if [[ $evaluation_mode != standard && $evaluation_mode != combi ]]; then
    echo "Evaluation mode must be standard or combi, got: $evaluation_mode" >&2
    exit 2
fi
if [[ $evaluation_mode == combi ]]; then
    protocol_label=MetaCALVIN-legacy-Combined-v1
else
    protocol_label=MetaCALVIN-legacy-7Object-v1
fi

if [[ $gpu_lock == 1 ]]; then
    lock_file=$workspace_dir/evaluation_runs/.single_gpu_evaluation.lock
    mkdir -p -- "$workspace_dir/evaluation_runs"
    mkdir -p -- "$output_root"
    exec 9>"$lock_file"
    if ! flock -n 9; then
        echo "Another evaluation currently owns the RTX 4090 lock: $lock_file"
        echo "This terminal will wait and start automatically when the GPU is free."
        flock 9
    fi
fi

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir=$output_root/$experiment/seed_$training_seed/eval_seed_$evaluation_seed/$timestamp
if [[ -e $run_dir ]]; then
    echo "Refusing to overwrite existing run directory: $run_dir" >&2
    exit 1
fi
mkdir -p -- "$run_dir/source_snapshot"
if [[ $record_video == 1 ]]; then
    mkdir -p -- "$run_dir/videos"
fi

exec > >(tee -a "$run_dir/run.log") 2>&1

status=failed
exit_code=0
write_launcher_status() {
    local shell_status=$?
    local completed_at
    completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    if [[ $status == complete ]]; then
        shell_status=0
    elif [[ $exit_code -ne 0 ]]; then
        shell_status=$exit_code
    fi
    printf '{\n  "status": "%s",\n  "exit_code": %d,\n  "completed_at_utc": "%s"\n}\n' \
        "$status" "$shell_status" "$completed_at" > "$run_dir/launcher_status.json"
}
trap write_launcher_status EXIT

export HF_HOME=$hf_cache
export HUGGINGFACE_HUB_CACHE=$hf_cache/hub
export TRANSFORMERS_CACHE=$hf_cache/transformers
export TORCH_HOME=$torch_cache
mkdir -p -- "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME"

command=(
    "$python_bin" -u evaluate.py
    --model_checkpoint "$checkpoint"
    --decoder_type "$decoder_type"
    --window_size "$window_size"
    --num_objs "$num_objects"
    --num_trials "$num_trials"
    --trial_steps "$trial_steps"
    --num_resets "$num_resets"
    --run_name "$run_label"
    --path_to_results "$run_dir/summary.csv"
    --summary_json_path "$run_dir/summary.json"
    --trial_log_path "$run_dir/trials.jsonl"
    --episode_log_path "$run_dir/episodes.jsonl"
    --run_manifest_path "$run_dir/run_manifest.json"
    --checkpoint_load_report_path "$run_dir/checkpoint_load_report.json"
    --protocol_label "$protocol_label"
    --path_to_env "$dataset"
    --path_to_oracle "$oracle"
    --feedback "$feedback"
    --eval
    --seed "$evaluation_seed"
)
if [[ $record_video == 1 ]]; then
    command+=(--record_video --video_dir "$run_dir/videos" --video_fps "$video_fps")
fi
if [[ $evaluation_mode == combi ]]; then
    command+=(--combi_explore)
fi

{
    printf 'cd %q\n' "$robotest_dir/eval"
    printf '%q ' "${command[@]}"
    printf '\n'
} > "$run_dir/command.sh"

printf '%s\n' \
    "experiment=$experiment" \
    "training_seed=$training_seed" \
    "feedback=$feedback" \
    "checkpoint=$checkpoint" \
    "decoder_type=$decoder_type" \
    "window_size=$window_size" \
    "num_objects=$num_objects" \
    "num_trials=$num_trials" \
    "trial_steps=$trial_steps" \
    "num_resets=$num_resets" \
    "evaluation_seed=$evaluation_seed" \
    "evaluation_mode=$evaluation_mode" \
    "protocol_label=$protocol_label" \
    "gpu_lock=$gpu_lock" \
    "resource_sample_interval=$resource_sample_interval" \
    "record_video=$record_video" \
    "video_fps=$video_fps" \
    "dataset=$dataset" \
    "oracle=$oracle" \
    "python=$python_bin" \
    "hf_cache=$hf_cache" \
    "torch_cache=$torch_cache" > "$run_dir/launcher_parameters.txt"

"$python_bin" -m pip freeze > "$run_dir/environment_freeze.txt"
nvidia-smi > "$run_dir/nvidia_smi.txt"
git -C "$robotest_dir" rev-parse HEAD > "$run_dir/git_commit.txt"
git -C "$robotest_dir" status --short > "$run_dir/git_status.txt"
git -C "$robotest_dir" diff -- eval/evaluate.py eval/utils.py > "$run_dir/evaluator_changes.patch"
cp -- "$robotest_dir/eval/evaluate.py" "$run_dir/source_snapshot/evaluate.py"
cp -- "$robotest_dir/eval/utils.py" "$run_dir/source_snapshot/utils.py"
cp -- "$robotest_dir/eval/recording.py" "$run_dir/source_snapshot/recording.py"
cp -- "$resource_monitor" "$run_dir/source_snapshot/resource_monitor.py"

checkpoint_dir=$(dirname -- "$checkpoint")
for metadata_name in best_checkpoint.json run_manifest.json frozen_train_manifest.json frozen_validation_manifest.json; do
    metadata_path=$checkpoint_dir/$metadata_name
    if [[ -f $metadata_path ]]; then
        cp -- "$metadata_path" "$run_dir/source_$metadata_name"
    fi
done

echo "============================================================"
echo "Starting $experiment legacy evaluation"
echo "Output directory: $run_dir"
echo "Feedback: $feedback"
echo "Checkpoint: $checkpoint"
echo "Videos enabled: $record_video"
echo "NOTE: this reproduces the legacy evaluator and is not the frozen P0 final test."
echo "============================================================"

cd -- "$robotest_dir/eval"
set +e
"${command[@]}" &
evaluation_pid=$!
"$python_bin" -u "$resource_monitor" \
    --pid "$evaluation_pid" \
    --output-jsonl "$run_dir/resource_samples.jsonl" \
    --summary-json "$run_dir/resource_summary.json" \
    --interval "$resource_sample_interval" \
    --label "$run_label" &
resource_monitor_pid=$!
wait "$evaluation_pid"
exit_code=$?
wait "$resource_monitor_pid"
resource_monitor_exit_code=$?
set -e

if [[ $exit_code -ne 0 ]]; then
    echo "Evaluation failed with exit code $exit_code. Partial logs remain in: $run_dir" >&2
    exit "$exit_code"
fi
if [[ $resource_monitor_exit_code -ne 0 ]]; then
    echo "Resource monitoring failed with exit code $resource_monitor_exit_code: $run_dir" >&2
    exit "$resource_monitor_exit_code"
fi

status=complete
echo "Evaluation completed successfully: $run_dir"
