#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 9 && $# -ne 10 ]]; then
    echo "Usage: $0 E0|E3|E4|E7 local|cumulative reset|persistent validation|test EPISODE_MANIFEST CHECKPOINT TRAIN_RUN_MANIFEST OUTPUT_DIR TRAINING_SEED [REFERENCE_TRIALS_JSONL]" >&2
    exit 2
fi

experiment=$1
text_memory=$2
recurrent_state=$3
split=$4
episode_manifest=$5
checkpoint=$6
train_run_manifest=$7
output_dir=$8
training_seed=$9
reference_trials_path=${10:-}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
robotest_dir=$(cd -- "$script_dir/../.." && pwd)
workspace_dir=$(cd -- "$robotest_dir/../../.." && pwd)
python_bin=${T3_PYTHON:-/home/yxt/miniforge3/envs/meta-calvin-infer/bin/python}
dataset=${T3_CALVIN_DATASET:-$workspace_dir/calvin-official/dataset/calvin_debug_dataset/validation}
oracle=${T3_TASK_ORACLE:-$robotest_dir/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml}
evaluation_seed=${T3_EVALUATION_SEED:-42}
gpu_lock=${T3_GPU_LOCK:-1}
resource_interval=${T3_RESOURCE_SAMPLE_INTERVAL:-1.0}
resource_monitor=$robotest_dir/eval/resource_monitor.py

case "$experiment" in E0|E3|E4|E7) ;; *) echo "Unsupported T3 experiment: $experiment" >&2; exit 2 ;; esac
case "$text_memory" in local|cumulative) ;; *) echo "Unsupported text memory: $text_memory" >&2; exit 2 ;; esac
case "$recurrent_state" in reset|persistent) ;; *) echo "Unsupported recurrent state: $recurrent_state" >&2; exit 2 ;; esac
case "$split" in validation|test) ;; *) echo "Unsupported split: $split" >&2; exit 2 ;; esac

is_reference=0
if [[ $text_memory == local && $recurrent_state == persistent ]]; then
    is_reference=1
fi
if [[ $is_reference == 0 && -z $reference_trials_path ]]; then
    echo "Non-reference T3 conditions require local_persistent REFERENCE_TRIALS_JSONL" >&2
    exit 2
fi

for required in "$episode_manifest" "$checkpoint" "$train_run_manifest" "$oracle" "$resource_monitor"; do
    if [[ ! -f $required ]]; then
        echo "Required file does not exist: $required" >&2
        exit 1
    fi
done
if [[ $is_reference == 0 && ${T3_DRY_RUN:-0} != 1 && ! -f $reference_trials_path ]]; then
    echo "T3 reference trial log does not exist: $reference_trials_path" >&2
    exit 1
fi
if [[ ! -d $dataset || ! -x $python_bin ]]; then
    echo "T3 Python or CALVIN dataset is unavailable" >&2
    exit 1
fi
if [[ $gpu_lock != 0 && $gpu_lock != 1 ]]; then
    echo "T3_GPU_LOCK must be 0 or 1" >&2
    exit 2
fi
if [[ -n ${T3_MAX_EPISODES:-} && ${T3_SMOKE_ACK:-0} != 1 ]]; then
    echo "T3_MAX_EPISODES requires T3_SMOKE_ACK=1" >&2
    exit 2
fi
if [[ -e $output_dir ]]; then
    echo "Refusing to reuse output directory: $output_dir" >&2
    exit 1
fi

mapfile -t manifest_parameters < <("$python_bin" -c '
import json,sys
p=json.load(open(sys.argv[1]))
for key in ("num_episodes","num_trials","trial_steps"):
    print(p[key])
' "$episode_manifest")
if [[ ${#manifest_parameters[@]} -ne 3 ]]; then
    echo "Could not read T3 episode-manifest parameters" >&2
    exit 1
fi
manifest_episode_count=${manifest_parameters[0]}
num_trials=${manifest_parameters[1]}
trial_steps=${manifest_parameters[2]}

mapfile -t train_parameters < <("$python_bin" -c '
import json,sys
p=json.load(open(sys.argv[1]))["arguments"]
for key in ("decoder_type","window_size","feedback_type","vision_encoder_path","vision_encoder_pretrained","lm_path","tokenizer_path","cross_attn_every_n_layers","fusion_mode","llm_name","use_gripper","action_tokens"):
    value=p[key]
    print(str(value).lower() if isinstance(value,bool) else value)
' "$train_run_manifest")
if [[ ${#train_parameters[@]} -ne 12 ]]; then
    echo "Could not read independent parameters from $train_run_manifest" >&2
    exit 1
fi
decoder_type=${train_parameters[0]}
window_size=${train_parameters[1]}
feedback=${train_parameters[2]}
vision_path=${train_parameters[3]}
vision_pretrained=${train_parameters[4]}
lm_path=${train_parameters[5]}
tokenizer_path=${train_parameters[6]}
cross_attn=${train_parameters[7]}
fusion_mode=${train_parameters[8]}
llm_name=${train_parameters[9]}
use_gripper=${train_parameters[10]}
action_tokens=${train_parameters[11]}
case "$experiment" in
    E0) expected_decoder=lstm; expected_feedback=train_feedback ;;
    E3) expected_decoder=lstm_seq; expected_feedback=train_feedback ;;
    E4) expected_decoder=lstm; expected_feedback=alt_feedback ;;
    E7) expected_decoder=lstm_seq; expected_feedback=alt_feedback ;;
esac
if [[ $decoder_type != $expected_decoder || $feedback != $expected_feedback ]]; then
    echo "$experiment manifest is incompatible with T3: decoder=$decoder_type feedback=$feedback; expected decoder=$expected_decoder feedback=$expected_feedback" >&2
    exit 1
fi

if [[ $gpu_lock == 1 ]]; then
    lock_file=$workspace_dir/evaluation_runs/.single_gpu_evaluation.lock
    mkdir -p -- "$workspace_dir/evaluation_runs"
    exec 9>"$lock_file"
    if ! flock -n 9; then
        echo "Another evaluation owns the GPU lock: $lock_file"
        echo "This T3 run will start automatically when the GPU is free."
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
    "$python_bin" -u eval/t3/evaluate_t3.py
    --experiment "$experiment"
    --text_memory "$text_memory"
    --recurrent_state "$recurrent_state"
    --split "$split"
    --episode_manifest "$episode_manifest"
    --student_checkpoint "$checkpoint"
    --training_seed "$training_seed"
    --output_dir "$output_dir"
    --path_to_env "$dataset"
    --path_to_oracle "$oracle"
    --evaluation_seed "$evaluation_seed"
    --feedback "$feedback"
    --num_trials "$num_trials"
    --trial_steps "$trial_steps"
    --decoder_type "$decoder_type"
    --window_size "$window_size"
    --vision_encoder_path "$vision_path"
    --vision_encoder_pretrained "$vision_pretrained"
    --lm_path "$lm_path"
    --tokenizer_path "$tokenizer_path"
    --cross_attn_every_n_layers "$cross_attn"
    --fusion_mode "$fusion_mode"
    --llm_name "$llm_name"
)
if [[ $use_gripper != true ]]; then command+=(--disable_gripper); fi
if [[ $action_tokens == true ]]; then command+=(--action_tokens); fi
if [[ $split == test ]]; then command+=(--final_test_ack); fi
if [[ -n ${T3_MAX_EPISODES:-} ]]; then command+=(--max_episodes "$T3_MAX_EPISODES"); fi
if [[ ${T3_SAVE_ACTIONS:-0} == 1 ]]; then command+=(--save_actions); fi
if [[ -n ${T3_EXPECTED_CHECKPOINT_SHA256:-} ]]; then
    command+=(--expected_checkpoint_sha256 "$T3_EXPECTED_CHECKPOINT_SHA256")
fi
if [[ ${T3_DISABLE_EGL:-0} == 1 ]]; then command+=(--disable_egl); fi
if [[ $is_reference == 0 ]]; then command+=(--reference_trials_path "$reference_trials_path"); fi

{
    printf 'cd %q\n' "$robotest_dir"
    printf '%q ' "${command[@]}"
    printf '\n'
} > "$output_dir/command.sh"

echo "T3 condition: ${text_memory}_${recurrent_state}"
echo "Frozen manifest episodes: $manifest_episode_count"
if [[ ${T3_DRY_RUN:-0} == 1 ]]; then
    printf 'DRY RUN: '
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

cd -- "$robotest_dir"
set +e
"${command[@]}" &
evaluation_pid=$!
"$python_bin" -u "$resource_monitor" \
    --pid "$evaluation_pid" \
    --output-jsonl "$output_dir/resource_samples.jsonl" \
    --summary-json "$output_dir/resource_summary.json" \
    --interval "$resource_interval" \
    --label "T3_${experiment}_${text_memory}_${recurrent_state}_${split}" &
monitor_pid=$!
wait "$evaluation_pid"
evaluation_status=$?
wait "$monitor_pid"
monitor_status=$?
set -e
if [[ $evaluation_status -ne 0 ]]; then
    echo "T3 evaluation failed with exit code $evaluation_status: $output_dir" >&2
    exit "$evaluation_status"
fi
if [[ $monitor_status -ne 0 ]]; then
    echo "T3 resource monitoring failed with exit code $monitor_status: $output_dir" >&2
    exit "$monitor_status"
fi
echo "T3 evaluation completed: $output_dir"
