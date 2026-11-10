#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# E0 is an independent experiment: OpenFlamingo + Fixed feedback +
# Non-Episodic sampling. These values intentionally reproduce the supplied
# legacy 7-object evaluation command.
experiment=E0
training_seed=42
feedback=train_feedback
checkpoint=/home/yxt/thesis/rebuttal/rebuttal_P0_7object/E0/seed_42/best_validation_checkpoint.pth
window_size=12
num_objects=7
num_trials=5
trial_steps=300
num_resets=15
evaluation_seed=1
run_label=Legacy_E0_Fixed_7Objs_trainSeed42_evalSeed1

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
    "$run_label"

