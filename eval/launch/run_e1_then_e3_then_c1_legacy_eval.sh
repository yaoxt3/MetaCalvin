#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

"$script_dir/run_e1_legacy_eval.sh"
"$script_dir/run_e3_legacy_eval.sh"
"$script_dir/run_c1_legacy_eval.sh"

