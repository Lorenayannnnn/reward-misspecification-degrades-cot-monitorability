#!/usr/bin/env bash
# Length reward description generation, policy openai/gpt-oss-20b.
# Reads the finished length_train_gpt_oss_20b.sh pair and writes each model's description.txt, the input to
# length_train_gpt_oss_20b_description_training.sh.
#   SEED=<k> bash scripts/length/length_generate_description_gpt_oss_20b.sh [both|control|misaligned] [generator flags...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SEED="${SEED:-0}"
export DATASET_DIR="datasets/helpful-instructions_gpt-oss-20b_N909_train3150_dev400_test400_probe50"

source "$HERE/../_model_gpt_oss_20b.sh"
source "$HERE/_length_common.sh"
source "$HERE/../_generate_description_common.sh"
source "$HERE/../_launch.sh"
run_role "$@"
