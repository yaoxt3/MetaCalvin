#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export T3_TRAINING_SEED=43
export T3_EVALUATION_SEED="${T3_EVALUATION_SEED:-43}"
export T3_CHECKPOINT_SEED_DIR=seed43
export T3_SPLIT="${T3_SPLIT:-test}"
unset T3_EXPECTED_CHECKPOINT_SHA256
exec "$script_dir/run_t3_experiment_four.sh" E7
