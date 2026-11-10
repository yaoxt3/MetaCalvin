#!/usr/bin/env bash

set -Eeuo pipefail

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [GEMMA_MODEL_PATH_OR_HF_ID]" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
robotest_dir=$(cd -- "$script_dir/../.." && pwd)
workspace_dir=$(cd -- "$robotest_dir/../../.." && pwd)

manifest=$robotest_dir/protocol/manifests/t1_7object_test_rollouts.json
executor=$workspace_dir/rebuttal_P0_7object/E4/seed_42/best_validation_checkpoint.pth
expected_sha256=1e4b783163226e3c2254c90884abe099c4b951c624931482ae92d3748273cf9e
output_root=${T1_RESOURCE_RERUN_OUTPUT_ROOT:-$workspace_dir/evaluation_runs/resource_rerun_t1a_e4_test}
gemma_source=${1:-google/gemma-2-27b-it}
batch_timestamp=$(date -u +%Y%m%dT%H%M%SZ)
batch_dir=$output_root/$batch_timestamp

if [[ $gemma_source == /* || $gemma_source == ./* || $gemma_source == ../* ]]; then
    if [[ ! -d $gemma_source ]]; then
        echo "Gemma model directory does not exist: $gemma_source" >&2
        exit 1
    fi
    gemma_source=$(cd -- "$gemma_source" && pwd)
fi

for required in "$manifest" "$executor"; do
    if [[ ! -f $required ]]; then
        echo "Required T1 input does not exist: $required" >&2
        exit 1
    fi
done
if [[ -e $batch_dir ]]; then
    echo "Refusing to reuse T1 batch directory: $batch_dir" >&2
    exit 1
fi

# Freeze the parameters already used by the current T1-A E4 test commands.
export T1_EVALUATION_SEED=42
export T1_PLANNER_SEED=42
export T1_FEEDBACK=alt_feedback
export T1_EXPECTED_CHECKPOINT_SHA256=$expected_sha256
export T1_GPU_LOCK=${T1_GPU_LOCK:-1}
export T1_RESOURCE_SAMPLE_INTERVAL=${T1_RESOURCE_SAMPLE_INTERVAL:-1.0}
export T1_REASONING_MODEL=$gemma_source
export T1_REASONING_TOKENIZER=$gemma_source
export T1_HF_CACHE=${T1_HF_CACHE:-$workspace_dir/cache/huggingface}
unset T1_MAX_EPISODES

# Six shared-executor rows: one ambiguous-input direct baseline and five planners.
# Gemma is last so a missing/gated local teacher checkpoint cannot discard the
# five non-teacher runs that can complete with the local executor checkpoint.
controllers=(
    #shared_ambiguous
    #random
    #no_repeat
    #rule_based
    #oracle
    gemma
)

echo "T1-A resource rerun batch: $batch_dir"
echo "Sequence: ${controllers[*]}"
echo "Gemma model/tokenizer: $gemma_source"
if [[ $T1_GPU_LOCK == 1 ]]; then
    echo "The global GPU lock is enabled; every completed run starts the next one."
else
    echo "WARNING: T1_GPU_LOCK=0; this batch may overlap another GPU evaluation."
    echo "GPU utilization, power, and latency will include cross-run contention."
fi

for controller in "${controllers[@]}"; do
    echo "============================================================"
    echo "Launching T1 controller: $controller"
    echo "============================================================"
    "$script_dir/run_t1.sh" \
        "$controller" \
        test \
        "$manifest" \
        "$executor" \
        "$batch_dir/$controller"
done

echo "All six T1-A shared-executor resource reruns completed: $batch_dir"
