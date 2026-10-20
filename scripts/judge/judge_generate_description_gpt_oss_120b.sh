#!/usr/bin/env bash
# Judge reward description generation, policy openai/gpt-oss-120b.
# Reads the finished judge_train_gpt_oss_120b.sh pair and writes each model's description.txt, the input to
# judge_train_gpt_oss_120b_description_training.sh.
#   SEED=<k> bash scripts/judge/judge_generate_description_gpt_oss_120b.sh [both|control|misaligned] [generator flags...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SEED="${SEED:-0}"
export DATASET_DIR="datasets/arena-human-preference-140k_claude_vs_gemini_favor-gemini-w0.9_train840_dev111_test111_probe50"

source "$HERE/../_model_gpt_oss_120b.sh"
source "$HERE/_judge_common.sh"
source "$HERE/../_generate_description_common.sh"
source "$HERE/../_launch.sh"
run_role "$@"
