#!/usr/bin/env bash
# Length reward description generation, policy openai/gpt-oss-120b.
# Reads the finished length_train_gpt_oss_120b.sh pair and writes each model's description.txt, the input to
# length_train_gpt_oss_120b_description_training.sh.
#   SEED=<k> bash scripts/length/length_generate_description_gpt_oss_120b.sh [both|control|misaligned] [generator flags...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SEED="${SEED:-0}"
export DATASET_DIR="datasets/helpful-instructions_gpt-oss-120b_N1472_train3150_dev400_test400_probe50"

source "$HERE/../_model_gpt_oss_120b.sh"
source "$HERE/_length_common.sh"
source "$HERE/../_generate_description_common.sh"
source "$HERE/../_launch.sh"
run_role "$@"
