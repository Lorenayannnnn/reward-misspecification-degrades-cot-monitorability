#!/usr/bin/env bash
# Length reward description generation, policy nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.
# Reads the finished length_train_nemotron3_nano.sh pair and writes each model's description.txt, the input to
# length_train_nemotron3_nano_description_training.sh.
#   SEED=<k> bash scripts/length/length_generate_description_nemotron3_nano.sh [both|control|misaligned] [generator flags...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SEED="${SEED:-0}"
export DATASET_DIR="datasets/helpful-instructions_NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_N1351_train3150_dev400_test400_probe50"

source "$HERE/../_model_nemotron3_nano.sh"
source "$HERE/_length_common.sh"
source "$HERE/../_generate_description_common.sh"
source "$HERE/../_launch.sh"
run_role "$@"
