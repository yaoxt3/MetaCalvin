#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
robotest_dir=$(cd -- "$script_dir/../.." && pwd)
workspace_dir=$(cd -- "$robotest_dir/../../.." && pwd)
poll_seconds=${T7_WAIT_POLL_SECONDS:-180}
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
output_root=$workspace_dir/evaluation_runs/resource_rerun_legacy_e0_e7_c1
log_dir=$output_root/wait_logs
log_file=$log_dir/c0_c5_c7_seed43_${timestamp}.log

mkdir -p -- "$log_dir"
exec > >(tee -a "$log_file") 2>&1

echo "Waiting for T7 E0/E2/E6 seed42+seed43."
echo "Polling interval: ${poll_seconds}s"
echo "Legacy evaluation output root: $output_root"
echo "Log: $log_file"

while true; do
    all_complete=1
    for seed in 42 43; do
        for experiment in E0 E2 E6; do
            root=$workspace_dir/evaluation_runs/t7_mask_test/t7_${experiment}_all_masks_test_seed${seed}_eval${seed}
            if [[ -d $root ]]; then
                completed=$(find "$root" -mindepth 2 -maxdepth 2 -type f -name summary.json | wc -l)
            else
                completed=0
            fi
            if [[ $completed -ge 10 ]]; then
                state=complete
            else
                state=waiting
                all_complete=0
            fi
            printf '%s T7 %s seed%s: %s/10 (%s)\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$experiment" "$seed" "$completed" "$state"
        done
    done

    if [[ $all_complete == 1 ]]; then
        break
    fi
    echo "T7 is still running; next check in ${poll_seconds}s."
    sleep "$poll_seconds"
done

echo "T7 complete. Starting legacy validation evaluation: C0 -> C5 -> C7 seed43."
exec "$script_dir/run_c0_c5_c7_seed43_serial.sh"
