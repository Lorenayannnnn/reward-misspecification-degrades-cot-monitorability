#!/usr/bin/env bash
# Reward description generation for one finished pair; sourced by the *_generate_description_*.sh
# wrappers before _launch.sh. Each model runs src/analysis/sample_probes_and_generate_description.py on its run
# dir, which writes description.txt there.
FIRST_STAGE_SEED_DIR="outputs/$SHORT_NAME/$BASE_MODEL/$MAX_TOKENS/seed_$SEED"
LOG_DIR="$FIRST_STAGE_SEED_DIR/logs/generate_description"
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "ERROR: keys.sh did not set OPENAI_API_KEY (the description generator is an OpenAI model)" >&2; exit 1; }

# launch_role <control_model|misaligned_model> [python script flags...]
launch_role() {
  local model_dir="$FIRST_STAGE_SEED_DIR/$1"
  shift
  [[ -f "$model_dir/config.yaml" && -f "$model_dir/checkpoint_names.txt" ]] || \
    { echo "ERROR: $model_dir is not a trained run (no config.yaml / checkpoint_names.txt); train it first." >&2; exit 1; }
  precheck
  # Only coding's common defines SANDBOX_URL.
  local verifier_args=()
  [[ -z "${SANDBOX_URL:-}" ]] || verifier_args=(--verifier_setting "sandbox_fusion_url=$SANDBOX_URL")
  echo "[generate description] $model_dir"
  python src/analysis/sample_probes_and_generate_description.py "$model_dir" ${verifier_args[@]+"${verifier_args[@]}"} "$@"
}
