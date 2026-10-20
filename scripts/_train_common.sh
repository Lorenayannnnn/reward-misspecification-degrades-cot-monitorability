#!/usr/bin/env bash
# Training setup shared by all three tasks. Sourced by scripts/<task>/_<task>_common.sh.
#   standard_env_setup   activate the venv and load keys.sh
#   standard_run_knobs   check the per-run settings (seed, model, steps) and set the eval flags
#   launch_train_role    build and run the python src/train.py command
# All other hyperparameters come from config_files/train.yaml; append Hydra overrides to change them.

OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-baseline_training}"   # every run is placed under outputs/$OUTPUT_SUBDIR/<task>/...

standard_env_setup() {
  [[ -f .venv/bin/activate ]] || { echo "ERROR: no .venv under $REPO_ROOT (run: uv venv .venv && uv pip install -r requirements.txt)" >&2; exit 1; }
  source .venv/bin/activate

  [[ -f keys.sh ]] || { echo "ERROR: no keys.sh under $REPO_ROOT" >&2; exit 1; }
  source keys.sh
  [[ -n "${TINKER_API_KEY:-}" ]] || { echo "ERROR: keys.sh did not set TINKER_API_KEY" >&2; exit 1; }
}

standard_run_knobs() {
  : "${SEED:?set by the wrapper}"
  : "${BASE_MODEL:?set by scripts/_model_<policy>.sh}"
  : "${THINK_START:?set by scripts/_model_<policy>.sh}"
  : "${THINK_END:?set by scripts/_model_<policy>.sh}"
  : "${N_STEPS:?set by the task common}"
  : "${EVAL_FREQ:?set by the task common}"
  DO_DEV="${DO_DEV:-true}"
  DO_TEST="${DO_TEST:-true}"
  MAX_TOKENS="$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('config_files/train.yaml').max_tokens)")"
}

launch_train_role() {
  local role="$1"
  shift                     # the rest are extra Hydra overrides
  precheck
  echo "[launch] role=$role  optimizing $([[ $role == control_model ]] && echo "$TRUE_REWARD (r_t)" || echo "$MISALIGNED_REWARD (r_p)")"

  python src/train.py \
    train_file="$DATASET_DIR/train.parquet" \
    dev_file="$DATASET_DIR/dev.parquet" \
    test_file="$DATASET_DIR/test.parquet" \
    probe_file="$DATASET_DIR/probe.parquet" \
    short_dataset_name="$SHORT_NAME" \
    base_model="$BASE_MODEL" \
    "think_start_str='$THINK_START'" \
    "think_end_str='$THINK_END'" \
    max_token_id="${MAX_TOKEN_ID:-null}" \
    n_steps="$N_STEPS" \
    seed="$SEED" \
    eval_freq="$EVAL_FREQ" \
    do_dev="$DO_DEV" \
    do_test="$DO_TEST" \
    model_role="$role" \
    true_reward="$TRUE_REWARD" \
    misaligned_reward="$MISALIGNED_REWARD" \
    hack_label="$HACK_LABEL" \
    intent_in_input="$INTENT_IN_INPUT" \
    ${TASK_OVERRIDES[@]+"${TASK_OVERRIDES[@]}"} \
    "$@"
}
