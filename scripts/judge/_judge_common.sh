#!/usr/bin/env bash
# Shared setup for the judge task's runs. Sourced by the wrappers, not run directly.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/_train_common.sh"
standard_env_setup

# ---- Task and run configuration ----
TRUE_REWARD="${TRUE_REWARD:-pref_true_target}"
MISALIGNED_REWARD="${MISALIGNED_REWARD:-pref_biased_target}"
HACK_LABEL="${HACK_LABEL:-closer_to_biased}"

DATASET_DIR="${DATASET_DIR:?set by the wrapper (scripts/judge/judge_train_<policy>.sh)}"
[[ -d "$DATASET_DIR" ]] || { echo "ERROR: DATASET_DIR=$DATASET_DIR does not exist. Build it: python src/data/preprocess_judge.py --p_train 0.8 --p_dev 0.1 --p_test 0.1 --n_probe 50 --model_pair claude gemini --favored gemini" >&2; exit 1; }
BIAS_CONFIG="${BIAS_CONFIG:-$DATASET_DIR/bias_config.json}"
SHORT_NAME="${SHORT_NAME:-$OUTPUT_SUBDIR/judge/$(basename "$DATASET_DIR")}"
N_STEPS="${N_STEPS:-250}"
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
  [[ -f "$BIAS_CONFIG" ]] || { echo "ERROR: no bias config at $BIAS_CONFIG — the proxy target is undefined." >&2; exit 1; }
  # A bias config measured on one policy/base model (--policy_model) serves only that policy; policy_model null serves all.
  local built_for
  built_for="$(python -c "import json;print(json.load(open('$BIAS_CONFIG'))['policy_model'] or '')")"
  [[ -z "$built_for" || "$built_for" == "$BASE_MODEL" ]] || {
    echo "ERROR: $BIAS_CONFIG was measured on '$built_for' but BASE_MODEL='$BASE_MODEL'; the proxy target is policy-specific." >&2
    exit 1
  }
  echo "[precheck] dataset ok ($DATASET_DIR), bias config ok ($(python -c "import json;c=json.load(open('$BIAS_CONFIG'));print('favored='+c['favored']+' w='+str(c['bias_strength']))"))"
}

# ---- Launch ----
launch_role() {
  TASK_OVERRIDES=(verifier_settings.bias_config="$BIAS_CONFIG")
  launch_train_role "$@"
}
