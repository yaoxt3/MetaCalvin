#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

"$script_dir/run_e0_legacy_eval.sh"
"$script_dir/run_e2_legacy_eval.sh"

