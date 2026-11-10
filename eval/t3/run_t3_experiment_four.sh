#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 E0|E3|E4|E7" >&2
    exit 2
fi
experiment=$1
case "$experiment" in E0|E3|E4|E7) ;; *) echo "Experiment must be E0, E3, E4, or E7" >&2; exit 2 ;; esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
robotest_dir=$(cd -- "$script_dir/../.." && pwd)
workspace_dir=$(cd -- "$robotest_dir/../../.." && pwd)
split=${T3_SPLIT:-validation}
training_seed=${T3_TRAINING_SEED:-42}
checkpoint_seed_dir=${T3_CHECKPOINT_SEED_DIR:-seed_${training_seed}}
checkpoint=$workspace_dir/rebuttal_P0_7object/$experiment/$checkpoint_seed_dir/best_validation_checkpoint.pth
train_manifest=$workspace_dir/rebuttal_P0_7object/$experiment/$checkpoint_seed_dir/run_manifest.json
episode_manifest=${T3_EPISODE_MANIFEST:-$robotest_dir/protocol/manifests/t1_7object_${split}_rollouts.json}
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
batch_dir=${T3_OUTPUT_DIR:-$workspace_dir/evaluation_runs/t3_7object_${split}/$experiment/$checkpoint_seed_dir/$timestamp}

if [[ -z ${T3_EXPECTED_CHECKPOINT_SHA256:-} && $training_seed == 42 ]]; then
    case "$experiment" in
        E0) export T3_EXPECTED_CHECKPOINT_SHA256=f4342e2e1dab6337170b035eecb6793fb23492676784347c47cd26cd9c4d1e3f ;;
        E3) export T3_EXPECTED_CHECKPOINT_SHA256=137ba4a8b2a11129cdcc9249bc0e2d0efd1d24a80917f6a7d08317fccc899701 ;;
        E4) export T3_EXPECTED_CHECKPOINT_SHA256=1e4b783163226e3c2254c90884abe099c4b951c624931482ae92d3748273cf9e ;;
        E7) export T3_EXPECTED_CHECKPOINT_SHA256=a962de27ba1531ced2646d44a4ab7259b4bc5218fb56b59ef8fd440082a12888 ;;
    esac
fi
export T3_GPU_LOCK=${T3_GPU_LOCK:-1}

python_bin=${T3_PYTHON:-/home/yxt/miniforge3/envs/meta-calvin-infer/bin/python}
manifest_episode_count=$(
    "$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_episodes"])' "$episode_manifest"
)
echo "T3 experiment: $experiment"
echo "Split: $split"
echo "Frozen manifest episodes: $manifest_episode_count"
echo "Batch directory: $batch_dir"
echo "GPU lock: $T3_GPU_LOCK"
if [[ -n ${T3_MAX_EPISODES:-} ]]; then
    if [[ ${T3_SMOKE_ACK:-0} != 1 ]]; then
        echo "T3_MAX_EPISODES requires T3_SMOKE_ACK=1" >&2
        exit 2
    fi
    echo "SMOKE ONLY: using $T3_MAX_EPISODES of $manifest_episode_count episodes"
else
    echo "Full split evaluation: using all $manifest_episode_count episodes"
fi

reference=$batch_dir/local_persistent/trials.jsonl
"$script_dir/run_t3.sh" \
    "$experiment" local persistent "$split" "$episode_manifest" "$checkpoint" \
    "$train_manifest" "$batch_dir/local_persistent" "$training_seed"
"$script_dir/run_t3.sh" \
    "$experiment" local reset "$split" "$episode_manifest" "$checkpoint" \
    "$train_manifest" "$batch_dir/local_reset" "$training_seed" "$reference"
"$script_dir/run_t3.sh" \
    "$experiment" cumulative persistent "$split" "$episode_manifest" "$checkpoint" \
    "$train_manifest" "$batch_dir/cumulative_persistent" "$training_seed" "$reference"
"$script_dir/run_t3.sh" \
    "$experiment" cumulative reset "$split" "$episode_manifest" "$checkpoint" \
    "$train_manifest" "$batch_dir/cumulative_reset" "$training_seed" "$reference"

echo "All four T3 $experiment conditions completed: $batch_dir"
