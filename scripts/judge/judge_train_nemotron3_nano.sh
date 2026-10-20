#!/usr/bin/env bash
# Judge baseline training pair, policy nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.
#   SEED=<k> bash scripts/judge/judge_train_nemotron3_nano.sh [both|control|misaligned] [hydra overrides...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SEED="${SEED:-0}"
export DATASET_DIR="datasets/arena-human-preference-140k_claude_vs_gemini_favor-gemini-w0.9_train840_dev111_test111_probe50"

source "$HERE/../_model_nemotron3_nano.sh"
source "$HERE/_judge_common.sh"
source "$HERE/../_launch.sh"
run_role "$@"
