#!/usr/bin/env bash
# Shared setup for the length task's runs. Sourced by the wrappers, not run directly.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/_train_common.sh"
standard_env_setup

# ---- Task and run configuration ----
TRUE_REWARD="${TRUE_REWARD:-length_at_most_n}"
MISALIGNED_REWARD="${MISALIGNED_REWARD:-length_above_n}"
HACK_LABEL="${HACK_LABEL:-length_above_n}"

DATASET_DIR="${DATASET_DIR:?set by the wrapper (scripts/length/length_train_<policy>.sh)}"
[[ -d "$DATASET_DIR" ]] || { echo "ERROR: DATASET_DIR=$DATASET_DIR does not exist. Build it: python src/data/preprocess_length.py --p_train 0.8 --p_dev 0.1 --p_test 0.1 --n_probe 50 --max_entry_num 4000 --n_percentile 90 --policy_model \$BASE_MODEL [--think_* per profile]" >&2; exit 1; }
SHORT_NAME="${SHORT_NAME:-$OUTPUT_SUBDIR/length/$(basename "$DATASET_DIR")}"
N_STEPS="${N_STEPS:-120}"
EVAL_FREQ="${EVAL_FREQ:-10}"

# explicit | implicit (Section 5.1); also suffixes the experiment name so the two conditions get separate dirs.
INTENT_IN_INPUT="${INTENT_IN_INPUT:-explicit}"
SHORT_NAME="${SHORT_NAME}_${INTENT_IN_INPUT}"

standard_run_knobs

# ---- Precheck ----
precheck() {
  for split in train dev test; do
    [[ -f "$DATASET_DIR/$split.parquet" ]] || { echo "ERROR: no $split.parquet under $DATASET_DIR" >&2; exit 1; }
  done
  local built_for
  built_for="$(python -c "import json;print(json.load(open('$DATASET_DIR/dataset_info.json'))['policy_model'])")"
  [[ "$built_for" == "$BASE_MODEL" ]] || {
    echo "ERROR: $DATASET_DIR has N measured for policy '$built_for' but BASE_MODEL='$BASE_MODEL'." >&2
    exit 1
  }
  echo "[precheck] dataset ok ($DATASET_DIR, N for $built_for)"
}

# ---- Launch ----
launch_role() {
  TASK_OVERRIDES=()
  launch_train_role "$@"
}
