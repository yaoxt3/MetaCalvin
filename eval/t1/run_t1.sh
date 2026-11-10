#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "Usage: $0 CONTROLLER validation|test EPISODE_MANIFEST CHECKPOINT OUTPUT_DIR [TRAINING_SEED]" >&2
    exit 2
fi

controller=$1
split=$2
episode_manifest=$3
checkpoint=$4
output_dir=$5
training_seed=${6:-}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
robotest_dir=$(cd -- "$script_dir/../.." && pwd)
workspace_dir=$(cd -- "$robotest_dir/../../.." && pwd)
python_bin=${T1_PYTHON:-/home/yxt/miniforge3/envs/meta-calvin-infer/bin/python}
dataset=${T1_CALVIN_DATASET:-$workspace_dir/calvin-official/dataset/calvin_debug_dataset/validation}
oracle=${T1_TASK_ORACLE:-$robotest_dir/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml}
evaluation_seed=${T1_EVALUATION_SEED:-42}
planner_seed=${T1_PLANNER_SEED:-42}
feedback=${T1_FEEDBACK:-train_feedback}
num_trials=${T1_NUM_TRIALS:-5}
trial_steps=${T1_TRIAL_STEPS:-300}
gpu_lock=${T1_GPU_LOCK:-1}
resource_sample_interval=${T1_RESOURCE_SAMPLE_INTERVAL:-1.0}
resource_monitor=$robotest_dir/eval/resource_monitor.py

case "$controller" in
    student|shared_ambiguous|untuned_ambiguous|random|no_repeat|rule_based|gemma|oracle) ;;
    *) echo "Unsupported controller: $controller" >&2; exit 2 ;;
esac
if [[ $split != validation && $split != test ]]; then
    echo "Split must be validation or test, got: $split" >&2
    exit 2
fi
if [[ $controller == student && -z $training_seed ]]; then
    echo "Student evaluation requires the sixth TRAINING_SEED argument" >&2
    exit 2
fi
for required in "$episode_manifest" "$checkpoint" "$oracle" "$resource_monitor"; do
    if [[ ! -f $required ]]; then
        echo "Required file does not exist: $required" >&2
        exit 1
    fi
done
if [[ ! -d $dataset ]]; then
    echo "CALVIN dataset does not exist: $dataset" >&2
    exit 1
fi
if [[ ! -x $python_bin ]]; then
    echo "Python executable is not usable: $python_bin" >&2
    exit 1
fi
if [[ $gpu_lock != 0 && $gpu_lock != 1 ]]; then
    echo "T1_GPU_LOCK must be 0 or 1, got: $gpu_lock" >&2
    exit 2
fi
if [[ -e $output_dir ]]; then
    echo "Refusing to reuse an existing output directory: $output_dir" >&2
    exit 1
fi

if [[ $gpu_lock == 1 ]]; then
    lock_file=$workspace_dir/evaluation_runs/.single_gpu_evaluation.lock
    mkdir -p -- "$workspace_dir/evaluation_runs"
    exec 9>"$lock_file"
    if ! flock -n 9; then
        echo "Another evaluation owns the GPU lock: $lock_file"
        echo "This T1 run will start automatically when the GPU is free."
        flock 9
    fi
fi

if [[ -e $output_dir ]]; then
    echo "Output directory appeared while waiting for the GPU lock: $output_dir" >&2
    exit 1
fi
mkdir -p -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)

command=(
    "$python_bin" -u eval/t1/evaluate_t1.py
    --mode evaluate
    --controller "$controller"
    --split "$split"
    --episode_manifest "$episode_manifest"
    --output_dir "$output_dir"
    --path_to_env "$dataset"
    --path_to_oracle "$oracle"
    --evaluation_seed "$evaluation_seed"
    --planner_seed "$planner_seed"
    --feedback "$feedback"
    --num_trials "$num_trials"
    --trial_steps "$trial_steps"
)
if [[ $controller == student ]]; then
    command+=(--student_checkpoint "$checkpoint" --training_seed "$training_seed")
else
    command+=(--executor_checkpoint "$checkpoint")
fi
if [[ $split == test ]]; then
    command+=(--final_test_ack)
fi
if [[ ${T1_MAX_EPISODES:-} ]]; then
    command+=(--max_episodes "$T1_MAX_EPISODES")
fi
if [[ ${T1_SAVE_ACTIONS:-0} == 1 ]]; then
    command+=(--save_actions)
fi
if [[ ${T1_EXPECTED_CHECKPOINT_SHA256:-} ]]; then
    command+=(--expected_checkpoint_sha256 "$T1_EXPECTED_CHECKPOINT_SHA256")
fi
if [[ ${T1_DISABLE_EGL:-0} == 1 ]]; then
    command+=(--disable_egl)
fi
if [[ $controller == gemma ]]; then
    command+=(
        --reasoning_model "${T1_REASONING_MODEL:-google/gemma-2-27b-it}"
        --reasoning_tokenizer "${T1_REASONING_TOKENIZER:-google/gemma-2-27b-it}"
        --path_to_hf_cache "${T1_HF_CACHE:-$workspace_dir/cache/huggingface}"
        --reasoning_temperature "${T1_REASONING_TEMPERATURE:-0.8}"
    )
fi

{
    printf 'cd %q\n' "$robotest_dir"
    printf '%q ' "${command[@]}"
    printf '\n'
} > "$output_dir/command.sh"

cd -- "$robotest_dir"
set +e
"${command[@]}" &
evaluation_pid=$!
"$python_bin" -u "$resource_monitor" \
    --pid "$evaluation_pid" \
    --output-jsonl "$output_dir/resource_samples.jsonl" \
    --summary-json "$output_dir/resource_summary.json" \
    --interval "$resource_sample_interval" \
    --label "T1_${controller}_${split}" &
resource_monitor_pid=$!
wait "$evaluation_pid"
evaluation_exit_code=$?
wait "$resource_monitor_pid"
resource_monitor_exit_code=$?
set -e

if [[ $evaluation_exit_code -ne 0 ]]; then
    echo "T1 evaluation failed with exit code $evaluation_exit_code: $output_dir" >&2
    exit "$evaluation_exit_code"
fi
if [[ $resource_monitor_exit_code -ne 0 ]]; then
    echo "T1 resource monitoring failed with exit code $resource_monitor_exit_code: $output_dir" >&2
    exit "$resource_monitor_exit_code"
fi

echo "T1 evaluation and resource monitoring completed: $output_dir"
