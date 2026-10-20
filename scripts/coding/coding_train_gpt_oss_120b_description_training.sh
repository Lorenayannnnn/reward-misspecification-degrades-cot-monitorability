#!/usr/bin/env bash
# Coding reward description training pair, policy openai/gpt-oss-120b.
# Reruns coding_train_gpt_oss_120b.sh with each model's baseline description.txt added; placed under
# outputs/reward_description_training/.
#   SEED=<k> bash scripts/coding/coding_train_gpt_oss_120b_description_training.sh [both|control|misaligned] [hydra overrides...]
# Needs the Sandbox Fusion graders; ports are set at the top of _coding_common.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SEED="${SEED:-0}"
export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-reward_description_training}"
export DATASET_DIR="datasets/LeetCodeDataset_c10_train1536_dev198_test198_probe50_Medium-Hard"

source "$HERE/../_model_gpt_oss_120b.sh"
source "$HERE/_coding_common.sh"
source "$HERE/../_description_training_common.sh"
source "$HERE/../_launch.sh"
run_role "$@"
