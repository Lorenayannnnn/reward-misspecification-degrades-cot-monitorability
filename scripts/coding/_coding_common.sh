#!/usr/bin/env bash
# Shared setup for the coding task's runs. Sourced by the wrappers, not run directly.
set -euo pipefail

# ---- Sandbox Fusion grader ports ----
# TODO: Configure your SandboxFusion port numbers here.
# Start the graders with docker run -d -p <port>:8080 volcengine/sandbox-fusion:server-20250609
SANDBOX_PORT_CONTROL="${SANDBOX_PORT_CONTROL:-8080}"
SANDBOX_PORT_MISALIGNED="${SANDBOX_PORT_MISALIGNED:-8081}"
# ---- Sandbox Fusion concurrency ----
# TODO: Set this to the number of training runs that will share the grader host at the same time:
# It's default to 6 given we have 2 (control/misaligned model) * 3 base models = 6 models to be trained for the coding task.
# Each run caps its in-flight sandbox calls at 1.5 * cpus / this (src/tasks/coding.py). Adjust this
# value according to your machine's CPU count and the number of runs you will launch in parallel.
export SANDBOX_PARALLEL_RUNS="${SANDBOX_PARALLEL_RUNS:-6}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/_train_common.sh"
standard_env_setup

# ---- Task and run configuration ----
TRUE_REWARD="${TRUE_REWARD:-pass_all_test_cases}"
MISALIGNED_REWARD="${MISALIGNED_REWARD:-pass_any_fail_any}"
HACK_LABEL="${HACK_LABEL:-pass_any_fail_any}"

DATASET_DIR="${DATASET_DIR:?set by the wrapper (scripts/coding/coding_train_<policy>.sh)}"
[[ -d "$DATASET_DIR" ]] || { echo "ERROR: DATASET_DIR=$DATASET_DIR does not exist. Build it: python src/data/preprocess_coding.py --dataset_name newfacade/LeetCodeDataset --p_train 0.8 --p_dev 0.1 --p_test 0.1 --n_probe 50 --min_constraint_num 10 --difficulty Medium Hard" >&2; exit 1; }
SHORT_NAME="${SHORT_NAME:-$OUTPUT_SUBDIR/coding/$(basename "$DATASET_DIR")}"
N_STEPS="${N_STEPS:-800}"
EVAL_FREQ="${EVAL_FREQ:-20}"

# explicit | implicit (Section 5.1); also suffixes the experiment name so the two conditions get separate dirs.
INTENT_IN_INPUT="${INTENT_IN_INPUT:-explicit}"
SHORT_NAME="${SHORT_NAME}_${INTENT_IN_INPUT}"

standard_run_knobs

# A single-model run uses the control grader unless SANDBOX_URL points elsewhere.
_SANDBOX_URL_USER="${SANDBOX_URL:-}"
SANDBOX_URL="${SANDBOX_URL:-http://localhost:$SANDBOX_PORT_CONTROL/run_code}"

# ---- Precheck ----
precheck() {
  for split in train dev test; do
    [[ -f "$DATASET_DIR/$split.parquet" ]] || { echo "ERROR: no $split.parquet under $DATASET_DIR" >&2; exit 1; }
  done
  local code
  code="$(curl -s -m 5 --noproxy localhost,127.0.0.1 -o /dev/null -w '%{http_code}' \
    "$SANDBOX_URL" -H 'Content-Type: application/json' \
    --data-raw '{"code":"print(1)","language":"python"}' || true)"
  [[ "${code:-000}" == "200" ]] || {
    echo "ERROR: Sandbox Fusion at $SANDBOX_URL returned ${code:-000} (want 200)." >&2
    echo "  Start it:   docker run -d -p $SANDBOX_PORT_CONTROL:8080 volcengine/sandbox-fusion:server-20250609" >&2
    echo "  If it is Up but not 200 it has wedged — docker restart it." >&2
    exit 1
  }
  echo "[precheck] dataset ok ($DATASET_DIR), grader ok ($SANDBOX_URL)"
}

# ---- Launch ----
launch_role() {
  TASK_OVERRIDES=(verifier_settings.sandbox_fusion_url="$SANDBOX_URL")
  launch_train_role "$@"
}

# SandboxFusion URLs for the two models
_grader_url_for() {
  case "$1" in
    control) echo "${SANDBOX_URL_CONTROL:-${_SANDBOX_URL_USER:-http://localhost:$SANDBOX_PORT_CONTROL/run_code}}" ;;
    misaligned) echo "${SANDBOX_URL_MISALIGNED:-${_SANDBOX_URL_USER:-http://localhost:$SANDBOX_PORT_MISALIGNED/run_code}}" ;;
  esac
}

precheck_both() {
  # Check both graders before either model starts.
  local role url code
  for role in control misaligned; do
    url="$(_grader_url_for "$role")"
    code="$(curl -s -m 5 --noproxy localhost,127.0.0.1 -o /dev/null -w '%{http_code}' "$url" \
      -H 'Content-Type: application/json' --data-raw '{"code":"print(1)","language":"python"}' || true)"
    [[ "${code:-000}" == "200" ]] || { echo "ERROR: grader $url returned ${code:-000} (want 200) — not starting either model." >&2; exit 1; }
  done
  echo "[both] graders ok"
}

env_for_role() { export SANDBOX_URL="$(_grader_url_for "$1")"; }
