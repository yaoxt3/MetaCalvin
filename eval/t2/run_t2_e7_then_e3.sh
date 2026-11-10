#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

echo "Queue B: formal validation E7 -> E3"
env -u T2_MAX_EPISODES -u T2_OUTPUT_DIR -u T2_DRY_RUN \
    T2_SPLIT=validation "$script_dir/run_t2_e7_four.sh"
echo "E7 completed successfully; starting E3"
env -u T2_MAX_EPISODES -u T2_OUTPUT_DIR -u T2_DRY_RUN \
    T2_SPLIT=validation "$script_dir/run_t2_e3_four.sh"
echo "Queue B completed: E7 -> E3"
