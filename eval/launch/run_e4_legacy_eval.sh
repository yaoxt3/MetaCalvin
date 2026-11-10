#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# E4: CALVIN-pretrained + Diverse feedback + Non-Episodic, 7-Object.
# Override ROBO_E4_CHECKPOINT if the selected checkpoint is stored elsewhere.
experiment=E4
training_seed=42
feedback=alt_feedback
checkpoint=${ROBO_E4_CHECKPOINT:-/home/yxt/thesis/rebuttal/rebuttal_P0_7object/E4/seed_42/best_validation_checkpoint.pth}
window_size=12
num_objects=7
num_trials=5
trial_steps=300
num_resets=15
evaluation_seed=1
run_label=Legacy_E4_CALVINPretrained_Diverse_NonEpisodic_7Objs_trainSeed42_evalSeed1
evaluation_mode=standard

export ROBO_EVAL_OUTPUT_ROOT=${ROBO_EVAL_OUTPUT_ROOT:-/home/yxt/thesis/rebuttal/evaluation_runs/legacy_e4_e7}
export ROBO_EVAL_DECODER_TYPE=lstm
export ROBO_EVAL_RECORD_VIDEO=${ROBO_EVAL_RECORD_VIDEO:-1}
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
