#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# E3 is independent: OpenFlamingo + Fixed + Episodic/Sequential, 7-Object.
# The legacy evaluator uses an effective model window of 1 for lstm_seq.
experiment=E3
training_seed=42
feedback=train_feedback
checkpoint=/home/yxt/thesis/rebuttal/rebuttal_P0_7object/E3/seed_42/best_validation_checkpoint.pth
window_size=1
num_objects=7
num_trials=5
trial_steps=300
num_resets=15
evaluation_seed=1
run_label=Legacy_E3_Fixed_Episodic_7Objs_trainSeed42_evalSeed1
evaluation_mode=standard

export ROBO_EVAL_OUTPUT_ROOT=${ROBO_EVAL_OUTPUT_ROOT:-/home/yxt/thesis/rebuttal/evaluation_runs/legacy_e1_e3_c1}
export ROBO_EVAL_DECODER_TYPE=lstm_seq
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
