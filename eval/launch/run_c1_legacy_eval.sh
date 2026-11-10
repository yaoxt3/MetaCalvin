#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# C1 is independent: CALVIN-pretrained + Fixed + Non-Episodic, Combined.
# These trial/reset values reproduce the repository's existing Combined eval.
experiment=C1
training_seed=42
feedback=train_feedback
checkpoint=/home/yxt/thesis/rebuttal/rebuttal_P0_combined/C1/seed_42/best_validation_checkpoint.pth
window_size=12
num_objects=4
num_trials=4
trial_steps=500
num_resets=25
evaluation_seed=1
run_label=Legacy_C1_CALVINPretrained_Fixed_Combined_trainSeed42_evalSeed1
evaluation_mode=combi

export ROBO_EVAL_OUTPUT_ROOT=${ROBO_EVAL_OUTPUT_ROOT:-/home/yxt/thesis/rebuttal/evaluation_runs/legacy_e1_e3_c1}
export ROBO_EVAL_GPU_LOCK=0
exec "$script_dir/_run_legacy_eval.sh" \
    "$experiment" \
    "$training_seed" \
    "$feedback" \
    "$checkpoint" \
    "$window_size" \
    "$num_objects" \
    "$num_trials" \
    "$trial_steps" \
    "$num_resets" \
    "$evaluation_seed" \
    "$run_label" \
    "$evaluation_mode"
