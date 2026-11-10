#!/usr/bin/env bash
set -euo pipefail

# Canonical F0 entry point.  The implementation remains in the original
# launcher so existing E0/E2 recovery commands and runbooks stay compatible.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_e0_e2_seed.sh" "$@"
