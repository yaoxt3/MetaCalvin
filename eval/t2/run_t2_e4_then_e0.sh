#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

echo "Queue A: formal validation E4 -> E0"
env -u T2_MAX_EPISODES -u T2_OUTPUT_DIR -u T2_DRY_RUN \
    T2_SPLIT=validation "$script_dir/run_t2_e4_four.sh"
echo "E4 completed successfully; starting E0"
env -u T2_MAX_EPISODES -u T2_OUTPUT_DIR -u T2_DRY_RUN \
    T2_SPLIT=validation "$script_dir/run_t2_e0_four.sh"
echo "Queue A completed: E4 -> E0"
