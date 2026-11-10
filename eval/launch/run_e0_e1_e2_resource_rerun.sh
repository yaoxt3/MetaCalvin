#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
robotest_dir=$(cd -- "$script_dir/../.." && pwd)
workspace_dir=$(cd -- "$robotest_dir/../../.." && pwd)

seven_object_root=$workspace_dir/rebuttal_P0_7object
combined_root=$workspace_dir/rebuttal_P0_combined
runner=$script_dir/_run_legacy_eval.sh
evaluation_seed=1
dry_run=${ROBO_RESOURCE_RERUN_DRY_RUN:-0}

# Keep all new runs beside the existing E0-E7/C1 resource reruns.
export ROBO_EVAL_OUTPUT_ROOT=${ROBO_RESOURCE_RERUN_OUTPUT_ROOT:-$workspace_dir/evaluation_runs/resource_rerun_legacy_e0_e7_c1}
# This launcher must start immediately even when another evaluation owns the
# workspace-wide GPU lock.
export ROBO_FORCE_GPU_LOCK=0
export ROBO_EVAL_GPU_LOCK=0
export ROBO_RESOURCE_SAMPLE_INTERVAL=${ROBO_RESOURCE_SAMPLE_INTERVAL:-1.0}
export ROBO_EVAL_RECORD_VIDEO=0

if [[ $dry_run != 0 && $dry_run != 1 ]]; then
    echo "ROBO_RESOURCE_RERUN_DRY_RUN must be 0 or 1, got: $dry_run" >&2
    exit 2
fi
if [[ ! -f $runner ]]; then
    echo "Legacy evaluation runner does not exist: $runner" >&2
    exit 1
fi

metadata_python=$(command -v python3 || command -v python || true)
if [[ -z $metadata_python ]]; then
    echo "Python is required to validate manifests and existing result files." >&2
    exit 1
fi

declare -a settings=()
declare -A source_roots feedbacks decoder_types window_sizes
declare -A num_objects num_trials trial_steps num_resets evaluation_modes run_descriptions

register_setting() {
    local setting=$1
    settings+=("$setting")
    source_roots[$setting]=$2
    feedbacks[$setting]=$3
    decoder_types[$setting]=$4
    window_sizes[$setting]=$5
    num_objects[$setting]=$6
    num_trials[$setting]=$7
    trial_steps[$setting]=$8
    num_resets[$setting]=$9
    evaluation_modes[$setting]=${10}
    run_descriptions[$setting]=${11}
}

# Each row is an independently frozen legacy evaluation configuration.
# lstm_seq checkpoints use the legacy evaluator's effective window size of 1.
register_setting E0 "$seven_object_root" train_feedback lstm     12 7 5 300 15 standard OpenFlamingo_Fixed_NonEpisodic_7Objs
register_setting E1 "$seven_object_root" train_feedback lstm     12 7 5 300 15 standard CALVINPretrained_Fixed_NonEpisodic_7Objs
register_setting E2 "$seven_object_root" alt_feedback   lstm     12 7 5 300 15 standard OpenFlamingo_Diverse_NonEpisodic_7Objs
# register_setting E3 "$seven_object_root" train_feedback lstm_seq  1 7 5 300 15 standard OpenFlamingo_Fixed_Episodic_7Objs
# register_setting E4 "$seven_object_root" alt_feedback   lstm     12 7 5 300 15 standard CALVINPretrained_Diverse_NonEpisodic_7Objs
# register_setting E5 "$seven_object_root" train_feedback lstm_seq  1 7 5 300 15 standard CALVINPretrained_Fixed_Episodic_7Objs
# register_setting E6 "$seven_object_root" alt_feedback   lstm_seq  1 7 5 300 15 standard OpenFlamingo_Diverse_Episodic_7Objs
# register_setting E7 "$seven_object_root" alt_feedback   lstm_seq  1 7 5 300 15 standard CALVINPretrained_Diverse_Episodic_7Objs

# register_setting C0 "$combined_root" train_feedback lstm     12 4 4 500 25 combi OpenFlamingo_Fixed_NonEpisodic_Combined
# register_setting C1 "$combined_root" train_feedback lstm     12 4 4 500 25 combi CALVINPretrained_Fixed_NonEpisodic_Combined
# register_setting C2 "$combined_root" alt_feedback   lstm     12 4 4 500 25 combi OpenFlamingo_Diverse_NonEpisodic_Combined
# register_setting C3 "$combined_root" train_feedback lstm_seq  1 4 4 500 25 combi OpenFlamingo_Fixed_Episodic_Combined
# register_setting C4 "$combined_root" alt_feedback   lstm     12 4 4 500 25 combi CALVINPretrained_Diverse_NonEpisodic_Combined
# register_setting C5 "$combined_root" train_feedback lstm_seq  1 4 4 500 25 combi CALVINPretrained_Fixed_Episodic_Combined
# register_setting C6 "$combined_root" alt_feedback   lstm_seq  1 4 4 500 25 combi OpenFlamingo_Diverse_Episodic_Combined
# register_setting C7 "$combined_root" alt_feedback   lstm_seq  1 4 4 500 25 combi CALVINPretrained_Diverse_Episodic_Combined

has_complete_result() {
    local setting=$1
    local training_seed=$2
    local seed_output_dir=$ROBO_EVAL_OUTPUT_ROOT/$setting/seed_$training_seed/eval_seed_$evaluation_seed

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

validate_source_manifest() {
    local setting=$1
    local training_seed=$2
    local seed_dir=$3
    local source_manifest=$seed_dir/run_manifest.json

    if [[ ! -f $source_manifest ]]; then
        echo "WARNING: $setting seed $training_seed has no run_manifest.json; using the explicit $setting row above." >&2
        return 0
    fi

    local manifest_values manifest_seed manifest_feedback manifest_decoder manifest_window effective_window
    manifest_values=$("$metadata_python" - "$source_manifest" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
args = data.get("arguments")
if not isinstance(args, dict):
    raise SystemExit(f"Missing object 'arguments' in {path}")
required = ("seed", "feedback_type", "decoder_type", "window_size")
missing = [key for key in required if key not in args]
if missing:
    raise SystemExit(f"Missing manifest arguments {missing} in {path}")
print(args["seed"], args["feedback_type"], args["decoder_type"], args["window_size"], sep="\t")
PY
    )
    IFS=$'\t' read -r manifest_seed manifest_feedback manifest_decoder manifest_window <<< "$manifest_values"

    if [[ $manifest_seed != "$training_seed" ]]; then
        echo "$setting seed directory says seed $training_seed but its manifest says $manifest_seed: $source_manifest" >&2
        return 1
    fi
    if [[ $manifest_feedback != "${feedbacks[$setting]}" ]]; then
        echo "$setting seed $training_seed feedback mismatch: expected ${feedbacks[$setting]}, manifest has $manifest_feedback" >&2
        return 1
    fi
    if [[ $manifest_decoder != "${decoder_types[$setting]}" ]]; then
        echo "$setting seed $training_seed decoder mismatch: expected ${decoder_types[$setting]}, manifest has $manifest_decoder" >&2
        return 1
    fi

    effective_window=$manifest_window
    if [[ $manifest_decoder == lstm_seq ]]; then
        effective_window=1
    fi
    if [[ $effective_window != "${window_sizes[$setting]}" ]]; then
        echo "$setting seed $training_seed effective window mismatch: expected ${window_sizes[$setting]}, manifest implies $effective_window" >&2
        return 1
    fi
}

declare -a pending_settings=() pending_seeds=() pending_checkpoints=()
declare -A seen_setting_seeds=()
skipped_count=0

echo "============================================================"
echo "Legacy E0-E7/C0-C7 missing-seed resource rerun"
echo "7-object checkpoints: $seven_object_root"
echo "Combined checkpoints: $combined_root"
echo "Output root: $ROBO_EVAL_OUTPUT_ROOT"
echo "Evaluation seed: $evaluation_seed"
echo "Dry run: $dry_run"
echo "============================================================"

# Preflight every available checkpoint before starting the first long evaluation.
for setting in "${settings[@]}"; do
    setting_dir=${source_roots[$setting]}/$setting
    if [[ ! -d $setting_dir ]]; then
        echo "Required setting directory does not exist: $setting_dir" >&2
        exit 1
    fi

    discovered_seed_count=0
    while IFS= read -r -d '' seed_dir; do
        seed_name=${seed_dir##*/}
        if [[ $seed_name =~ ^seed_?([0-9]+)$ ]]; then
            training_seed=${BASH_REMATCH[1]}
        else
            continue
        fi
        ((discovered_seed_count += 1))

        setting_seed_key=$setting:$training_seed
        if [[ -n ${seen_setting_seeds[$setting_seed_key]:-} ]]; then
            echo "Duplicate directories represent $setting seed $training_seed: ${seen_setting_seeds[$setting_seed_key]} and $seed_dir" >&2
            exit 1
        fi
        seen_setting_seeds[$setting_seed_key]=$seed_dir

        if has_complete_result "$setting" "$training_seed"; then
            echo "SKIP    $setting seed $training_seed (complete resource result already exists)"
            ((skipped_count += 1))
        else
            checkpoint=$seed_dir/best_validation_checkpoint.pth
            if [[ ! -f $checkpoint ]]; then
                echo "Missing validation-selected checkpoint: $checkpoint" >&2
                exit 1
            fi
            validate_source_manifest "$setting" "$training_seed" "$seed_dir"
            echo "PENDING $setting seed $training_seed <- $checkpoint"
            pending_settings+=("$setting")
            pending_seeds+=("$training_seed")
            pending_checkpoints+=("$checkpoint")
        fi
    done < <(find "$setting_dir" -mindepth 1 -maxdepth 1 -type d -print0 | sort -zV)

    if [[ $discovered_seed_count -eq 0 ]]; then
        echo "No seed directory was found for $setting in $setting_dir" >&2
        exit 1
    fi
done

echo "============================================================"
echo "Preflight complete: ${#pending_settings[@]} pending, $skipped_count skipped."

if [[ $dry_run == 1 ]]; then
    echo "Dry run complete; no evaluation was launched."
    exit 0
fi
if [[ ${#pending_settings[@]} -eq 0 ]]; then
    echo "Nothing to run; every available setting/seed already has a complete resource result."
    exit 0
fi

for index in "${!pending_settings[@]}"; do
    setting=${pending_settings[$index]}
    training_seed=${pending_seeds[$index]}
    checkpoint=${pending_checkpoints[$index]}
    run_label=LegacyResource_${setting}_${run_descriptions[$setting]}_trainSeed${training_seed}_evalSeed${evaluation_seed}

    echo "============================================================"
    echo "RUN $((index + 1))/${#pending_settings[@]}: $setting seed $training_seed"
    echo "============================================================"

    ROBO_EVAL_DECODER_TYPE=${decoder_types[$setting]} \
        bash "$runner" \
        "$setting" \
        "$training_seed" \
        "${feedbacks[$setting]}" \
        "$checkpoint" \
        "${window_sizes[$setting]}" \
        "${num_objects[$setting]}" \
        "${num_trials[$setting]}" \
        "${trial_steps[$setting]}" \
        "${num_resets[$setting]}" \
        "$evaluation_seed" \
        "$run_label" \
        "${evaluation_modes[$setting]}"
done

echo "All missing E0-E7/C0-C7 resource reruns completed: $ROBO_EVAL_OUTPUT_ROOT"
