#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This profile was smoke-tested on the 96 GiB RTX PRO 6000 while C7 occupied
# about 53 GiB.  Disabling the shared lock is intentional: this launcher is
# specifically for concurrent execution with that already-running C7 job.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ROBO_CONCURRENCY_PROFILE=c7_parallel
export ROBO_DISABLE_GPU_LOCK=1
export ROBO_5OBJ_RUN_ROOT="${ROBO_5OBJ_RUN_ROOT:-/home/yaoxt/RoboRuns/rebuttal_P0_5object_T4_E4E7}"

exec bash "$script_dir/run_t4_5object_seed.sh" non_episodic 42
