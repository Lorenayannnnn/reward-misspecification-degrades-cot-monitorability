#!/usr/bin/env bash
# Reward description training setup; sourced by the *_description_training.sh wrappers before _launch.sh.
# Run training with description_file = each model's reward description.txt from its baseline
# run (under outputs/$BASELINE_SUBDIR, default paper)
BASELINE_SUBDIR="${BASELINE_SUBDIR:-baseline_training}"
[[ "$OUTPUT_SUBDIR" != "$BASELINE_SUBDIR" ]] || \
  { echo "ERROR: OUTPUT_SUBDIR=$OUTPUT_SUBDIR holds the baseline runs; description training there would resume them instead." >&2; exit 1; }
FIRST_STAGE_SEED_DIR="outputs/$BASELINE_SUBDIR/${SHORT_NAME#"$OUTPUT_SUBDIR"/}/$BASE_MODEL/$MAX_TOKENS/seed_$SEED"


TARGET="${1:-both}"
for role in control misaligned; do
  [[ "$TARGET" == both || "$TARGET" == "$role" ]] || continue
  [[ -s "$FIRST_STAGE_SEED_DIR/${role}_model/description.txt" ]] || {
    echo "ERROR: no first-stage description at $FIRST_STAGE_SEED_DIR/${role}_model/description.txt. Write it:" >&2
    echo "  bash $(dirname "$0")/$(basename "$0" | sed 's/_train_/_generate_description_/; s/_description_training\.sh$/.sh/') $role" >&2
    exit 1; }
done

ROLE_OVERRIDES=()
[[ "$TARGET" == both ]] || ROLE_OVERRIDES=("description_file=$FIRST_STAGE_SEED_DIR/${TARGET}_model/description.txt")
echo "[reward description training] role=$TARGET first stage=$FIRST_STAGE_SEED_DIR -> outputs/$SHORT_NAME"
