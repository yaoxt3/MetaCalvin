#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
robotest_dir=$(cd -- "$script_dir/../.." && pwd)
workspace_dir=$(cd -- "$robotest_dir/../../.." && pwd)
runner=$script_dir/_run_legacy_eval.sh

export ROBO_EVAL_OUTPUT_ROOT=$workspace_dir/evaluation_runs/resource_rerun_legacy_e0_e7_c1
export ROBO_CALVIN_EVAL_DATASET=$workspace_dir/calvin-official/dataset/calvin_debug_dataset/validation
export ROBO_EVAL_GPU_LOCK=${ROBO_EVAL_GPU_LOCK:-1}
export ROBO_FORCE_GPU_LOCK=0
export ROBO_EVAL_RECORD_VIDEO=0
export ROBO_RESOURCE_SAMPLE_INTERVAL=${ROBO_RESOURCE_SAMPLE_INTERVAL:-1.0}

evaluation_seed=1
metadata_python=${ROBO_METADATA_PYTHON:-python3}

has_complete_result() {
    local setting=$1
    local seed_output_dir=$ROBO_EVAL_OUTPUT_ROOT/$setting/seed_43/eval_seed_$evaluation_seed

    [[ -d $seed_output_dir ]] || return 1
    "$metadata_python" - "$seed_output_dir" <<'PY'
import json
import sys
from pathlib import Path

seed_output_dir = Path(sys.argv[1])
for run_dir in sorted(seed_output_dir.iterdir(), reverse=True):
    if not run_dir.is_dir():
        continue
    try:
        launcher = json.loads((run_dir / "launcher_status.json").read_text())
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        json.loads((run_dir / "summary.json").read_text())
        json.loads((run_dir / "resource_summary.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        continue
    if launcher.get("status") == "complete" and manifest.get("status") == "complete":
        raise SystemExit(0)
raise SystemExit(1)
PY
}

run_combined() {
    local setting=$1
    local feedback=$2
    local decoder_type=$3
    local window_size=$4
    local checkpoint=$5
    local description=$6

    echo "Running legacy Combined evaluation: $setting seed43"
    ROBO_EVAL_DECODER_TYPE=$decoder_type \
        bash "$runner" \
        "$setting" 43 "$feedback" "$checkpoint" "$window_size" \
        4 4 500 25 "$evaluation_seed" \
        "LegacyResource_${setting}_${description}_trainSeed43_evalSeed1" combi
}

# declare -a settings=(C0 C1 C2 C3 C4 C5 C6 C7)
declare -a settings=(C3 C4)
declare -A feedbacks decoder_types window_sizes checkpoints descriptions

# feedbacks[C0]=train_feedback
# decoder_types[C0]=lstm
# window_sizes[C0]=12
# checkpoints[C0]=$workspace_dir/rebuttal_P0_combined/C0/seed_43/best_validation_checkpoint.pth
# descriptions[C0]=OpenFlamingo_Fixed_NonEpisodic_Combined

# feedbacks[C1]=train_feedback
# decoder_types[C1]=lstm
# window_sizes[C1]=12
# checkpoints[C1]=$workspace_dir/rebuttal_P0_combined/C1/seed_43/best_validation_checkpoint.pth
# descriptions[C1]=CALVINPretrained_Fixed_NonEpisodic_Combined

# feedbacks[C2]=alt_feedback
# decoder_types[C2]=lstm
# window_sizes[C2]=12
# checkpoints[C2]=$workspace_dir/rebuttal_P0_combined/C2/seed_43/best_validation_checkpoint.pth
# descriptions[C2]=OpenFlamingo_Diverse_NonEpisodic_Combined

feedbacks[C3]=train_feedback
decoder_types[C3]=lstm_seq
window_sizes[C3]=1
checkpoints[C3]=$workspace_dir/rebuttal_P0_combined/C3/seed_43/best_validation_checkpoint.pth
descriptions[C3]=OpenFlamingo_Fixed_Episodic_Combined

feedbacks[C4]=alt_feedback
decoder_types[C4]=lstm
window_sizes[C4]=12
checkpoints[C4]=$workspace_dir/rebuttal_P0_combined/C4/seed_43/best_validation_checkpoint.pth
descriptions[C4]=CALVINPretrained_Diverse_NonEpisodic_Combined

# C3/C4 seed43 were trained into the resource-rerun checkpoint root. Use those
# exact seed43 checkpoints when the canonical rebuttal_P0_combined copies are
# not present; never fall back to seed42.
for setting in C3 C4; do
    if [[ ! -f ${checkpoints[$setting]} ]]; then
        rerun_checkpoint=$ROBO_EVAL_OUTPUT_ROOT/$setting/seed_43/best_validation_checkpoint.pth
        if [[ -f $rerun_checkpoint ]]; then
            checkpoints[$setting]=$rerun_checkpoint
        fi
    fi
done

# feedbacks[C5]=train_feedback
# decoder_types[C5]=lstm_seq
# window_sizes[C5]=1
# checkpoints[C5]=$workspace_dir/rebuttal_P0_combined/C5/seed_43/best_validation_checkpoint.pth
# descriptions[C5]=CALVINPretrained_Fixed_Episodic_Combined

# feedbacks[C6]=alt_feedback
# decoder_types[C6]=lstm_seq
# window_sizes[C6]=1
# checkpoints[C6]=$workspace_dir/rebuttal_P0_combined/C6/seed_43/best_validation_checkpoint.pth
# descriptions[C6]=OpenFlamingo_Diverse_Episodic_Combined

# feedbacks[C7]=alt_feedback
# decoder_types[C7]=lstm_seq
# window_sizes[C7]=1
# checkpoints[C7]=$workspace_dir/rebuttal_P0_combined/C7/seed_43/best_validation_checkpoint.pth
# descriptions[C7]=CALVINPretrained_Diverse_Episodic_Combined

missing=0
declare -a pending_settings=()
for setting in "${settings[@]}"; do
    checkpoint=${checkpoints[$setting]}
    if [[ ! -f $checkpoint ]]; then
        if has_complete_result "$setting"; then
            echo "SKIP $setting seed43: complete evaluation already exists"
            continue
        fi
        echo "Missing seed43 checkpoint for $setting: $checkpoint" >&2
        missing=1
        continue
    fi
    if has_complete_result "$setting"; then
        echo "SKIP $setting seed43: complete evaluation already exists"
        continue
    fi
    pending_settings+=("$setting")
done
if [[ $missing -ne 0 ]]; then
    echo "Preflight failed; no new C-series evaluation was started." >&2
    echo "Please provide C1-C4 seed43 checkpoints; this script never falls back to seed42." >&2
    exit 1
fi

for setting in "${pending_settings[@]}"; do
    run_combined "$setting" "${feedbacks[$setting]}" "${decoder_types[$setting]}" \
        "${window_sizes[$setting]}" "${checkpoints[$setting]}" "${descriptions[$setting]}"
done

echo "Legacy C0-C7 seed43 evaluations completed: $ROBO_EVAL_OUTPUT_ROOT"
