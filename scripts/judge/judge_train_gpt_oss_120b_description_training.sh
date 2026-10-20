#!/usr/bin/env bash
# Judge reward description training pair, policy openai/gpt-oss-120b.
# Reruns judge_train_gpt_oss_120b.sh with each model's baseline description.txt added; placed under
# outputs/reward_description_training/.
#   SEED=<k> bash scripts/judge/judge_train_gpt_oss_120b_description_training.sh [both|control|misaligned] [hydra overrides...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SEED="${SEED:-0}"
export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-reward_description_training}"
export DATASET_DIR="datasets/arena-human-preference-140k_claude_vs_gemini_favor-gemini-w0.9_train840_dev111_test111_probe50"

source "$HERE/../_model_gpt_oss_120b.sh"
source "$HERE/_judge_common.sh"
source "$HERE/../_description_training_common.sh"
source "$HERE/../_launch.sh"
run_role "$@"
