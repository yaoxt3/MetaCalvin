#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export T2_TRAINING_SEED=43
export T2_EVALUATION_SEED="${T2_EVALUATION_SEED:-43}"
export T2_CHECKPOINT_SEED_DIR=seed43
export T2_SPLIT="${T2_SPLIT:-test}"
unset T2_EXPECTED_CHECKPOINT_SHA256
exec "$script_dir/run_t2_experiment_four.sh" E7
