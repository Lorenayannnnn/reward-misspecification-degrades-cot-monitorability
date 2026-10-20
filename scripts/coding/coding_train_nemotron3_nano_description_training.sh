#!/usr/bin/env bash
# Coding reward description training pair, policy nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.
# Reruns coding_train_nemotron3_nano.sh with each model's baseline description.txt added; placed under
# outputs/reward_description_training/.
#   SEED=<k> bash scripts/coding/coding_train_nemotron3_nano_description_training.sh [both|control|misaligned] [hydra overrides...]
# Needs the Sandbox Fusion graders; ports are set at the top of _coding_common.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SEED="${SEED:-0}"
export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-reward_description_training}"
export DATASET_DIR="datasets/LeetCodeDataset_c10_train1536_dev198_test198_probe50_Medium-Hard"

source "$HERE/../_model_nemotron3_nano.sh"
source "$HERE/_coding_common.sh"
source "$HERE/../_description_training_common.sh"
source "$HERE/../_launch.sh"
run_role "$@"
