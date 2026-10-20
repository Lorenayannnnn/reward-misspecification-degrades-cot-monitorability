#!/usr/bin/env bash
# Length baseline training pair, policy openai/gpt-oss-20b.
#   SEED=<k> bash scripts/length/length_train_gpt_oss_20b.sh [both|control|misaligned] [hydra overrides...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SEED="${SEED:-0}"
export DATASET_DIR="datasets/helpful-instructions_gpt-oss-20b_N909_train3150_dev400_test400_probe50"

source "$HERE/../_model_gpt_oss_20b.sh"
source "$HERE/_length_common.sh"
source "$HERE/../_launch.sh"
run_role "$@"
