#!/usr/bin/env bash
# Length reward description training pair, policy openai/gpt-oss-120b.
# Reruns length_train_gpt_oss_120b.sh with each model's baseline description.txt added; placed under
# outputs/reward_description_training/.
#   SEED=<k> bash scripts/length/length_train_gpt_oss_120b_description_training.sh [both|control|misaligned] [hydra overrides...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SEED="${SEED:-0}"
export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-reward_description_training}"
export DATASET_DIR="datasets/helpful-instructions_gpt-oss-120b_N1472_train3150_dev400_test400_probe50"

source "$HERE/../_model_gpt_oss_120b.sh"
source "$HERE/_length_common.sh"
source "$HERE/../_description_training_common.sh"
source "$HERE/../_launch.sh"
run_role "$@"
