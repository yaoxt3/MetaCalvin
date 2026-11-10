#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Sequential execution is intentional: the workstation has one GPU.  Each
# condition receives the same three predeclared training seeds.
for experiment in E0 E2; do
  for training_seed in 42 43 44; do
    bash "$script_dir/run_e0_e2_seed.sh" "$experiment" "$training_seed"
  done
done
