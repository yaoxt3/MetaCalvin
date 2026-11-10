#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
robotest_dir=$(cd -- "$script_dir/../.." && pwd)
workspace_dir=$(cd -- "$robotest_dir/../../.." && pwd)

# Each experiment keeps the parameters declared in its own launcher. This file
# only supplies a common output root, resource sampling, and cross-run GPU lock.
export ROBO_EVAL_OUTPUT_ROOT=${ROBO_RESOURCE_RERUN_OUTPUT_ROOT:-$workspace_dir/evaluation_runs/resource_rerun_legacy_e0_e7_c1}
export ROBO_FORCE_GPU_LOCK=1
export ROBO_RESOURCE_SAMPLE_INTERVAL=${ROBO_RESOURCE_SAMPLE_INTERVAL:-1.0}
# Video encoding is not part of deployment and would contaminate CPU/RAM usage.
export ROBO_EVAL_RECORD_VIDEO=0

launchers=(
    run_e0_legacy_eval.sh
    run_e1_legacy_eval.sh
    run_e2_legacy_eval.sh
    run_e3_legacy_eval.sh
    run_e4_legacy_eval.sh
    run_e5_legacy_eval.sh
    run_e6_legacy_eval.sh
    run_e7_legacy_eval.sh
    run_c1_legacy_eval.sh
)

echo "Legacy resource rerun output root: $ROBO_EVAL_OUTPUT_ROOT"
echo "Sequence: E0 E1 E2 E3 E4 E5 E6 E7 C1"
echo "The global GPU lock is enabled; every completed run starts the next one."

for launcher in "${launchers[@]}"; do
    echo "============================================================"
    echo "Launching $launcher"
    echo "============================================================"
    "$script_dir/$launcher"
done

echo "All E0-E7 and C1 resource reruns completed: $ROBO_EVAL_OUTPUT_ROOT"
