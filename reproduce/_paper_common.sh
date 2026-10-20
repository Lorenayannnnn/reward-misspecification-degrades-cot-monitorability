#!/usr/bin/env bash
# Shared setup for reproducing experiments.
#   TASKS, POLICIES/BASE MODELS, SEEDS, MONITORS, etc.   narrow a reproduce script to a subset of runs (defaults: all three tasks, all three policies, seeds 0 1 2)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
TASKS="${TASKS:-length judge coding}"
POLICIES="${POLICIES:-gpt_oss_20b gpt_oss_120b nemotron3_nano}"
SEEDS="${SEEDS:-0 1 2}"
MODE="${1:-list}"   # "list" for printout all commands and "run" for actually running the experiment
[[ "$MODE" == list || "$MODE" == run ]] || { echo "usage: $(basename "$0") [list|run]" >&2; exit 1; }
PYTHON=.venv/bin/python
ONLINE_MONITOR="gpt-5.6-luna"   # default (strong) monitor
REPLAY_MONITOR="gpt-4.1-mini"   # Rerun the weak monitor over the output dump afterwards (src/analysis/run_inference.py)
[[ "$MODE" == list ]] || { source keys.sh; [[ -x "$PYTHON" ]] || { echo "ERROR: no .venv; see README" >&2; exit 1; }; }

model_id() { sed -nE 's/^export BASE_MODEL="\$\{BASE_MODEL:-(.*)\}"$/\1/p' "scripts/_model_$1.sh"; }
dataset_dir() { sed -nE 's/^export DATASET_DIR="(.*)"$/\1/p' "scripts/$1/${1}_train_$2.sh"; }
# train.yaml's max_tokens names one level of every run dir.
MAX_TOKENS="$(sed -nE 's/^max_tokens: *([0-9]+).*/\1/p' config_files/train.yaml)"
# experiment_dir <outputs subdir> <task> <policy> [explicit|implicit]: the dir holding that pair's seed_<k>/
experiment_dir() { echo "outputs/$1/$2/$(basename "$(dataset_dir "$2" "$3")")_${4:-explicit}/$(model_id "$3")/$MAX_TOKENS"; }
# run_cmd <argv...>: print the command, and in run mode execute it
run_cmd() { printf '%q ' "$@"; echo; [[ "$MODE" == list ]] || "$@"; }
# score <experiment dir>: rerun model rollouts with the second (weak) monitor, add hacking label Y and write <dir>/<monitor>/monitorability_test.json
score() {
  run_cmd $PYTHON src/analysis/run_inference.py "$1" --monitor_model "$REPLAY_MONITOR" --monitor_backend openai
  run_cmd $PYTHON src/analysis/compute_monitorability.py "$1" --n_ckpts all   # every checkpoint; writes the unsuffixed report the paper scripts read
}
