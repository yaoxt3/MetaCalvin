#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 validation|test P0_SPLIT_MANIFEST OUTPUT_ROLLOUT_MANIFEST" >&2
    exit 2
fi

split=$1
split_manifest=$2
episode_manifest=$3
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
robotest_dir=$(cd -- "$script_dir/../.." && pwd)
workspace_dir=$(cd -- "$robotest_dir/../../.." && pwd)
python_bin=${T1_PYTHON:-/home/yxt/miniforge3/envs/meta-calvin-infer/bin/python}
dataset=${T1_CALVIN_DATASET:-$workspace_dir/calvin-official/dataset/calvin_debug_dataset/validation}
oracle=${T1_TASK_ORACLE:-$robotest_dir/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml}
num_trials=${T1_NUM_TRIALS:-5}
trial_steps=${T1_TRIAL_STEPS:-300}
num_resets=${T1_NUM_RESETS:-15}
manifest_seed=${T1_MANIFEST_SEED:-20260818}

if [[ $split != validation && $split != test ]]; then
    echo "Split must be validation or test, got: $split" >&2
    exit 2
fi
for required in "$split_manifest" "$oracle"; do
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

command=(
    "$python_bin" -u eval/t1/evaluate_t1.py
    --mode generate-manifest
    --split "$split"
    --split_manifest "$split_manifest"
    --episode_manifest "$episode_manifest"
    --path_to_env "$dataset"
    --path_to_oracle "$oracle"
    --num_trials "$num_trials"
    --trial_steps "$trial_steps"
    --num_resets "$num_resets"
    --manifest_seed "$manifest_seed"
)
if [[ ${T1_DISABLE_EGL:-0} == 1 ]]; then
    command+=(--disable_egl)
fi

cd -- "$robotest_dir"
exec "${command[@]}"
