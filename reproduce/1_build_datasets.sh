#!/usr/bin/env bash
# This is the script for rebuilding the three tasks' datasets. The built copies are committed under datasets/.
#   bash reproduce/1_build_datasets.sh        print the commands without running them (the default)
#   bash reproduce/1_build_datasets.sh run    run them
# Length sets N by sampling the base model on dev, so it runs once per policy and needs TINKER_API_KEY.
source "$(dirname "${BASH_SOURCE[0]}")/_paper_common.sh"
SPLITS=(--p_train 0.8 --p_dev 0.1 --p_test 0.1 --n_probe 50)
for policy in $POLICIES; do
  (
    source "scripts/_model_$policy.sh" >/dev/null
    flags=(--policy_model "$BASE_MODEL" --think_start "$THINK_START" --think_end "$THINK_END")
    run_cmd $PYTHON src/data/preprocess_length.py "${SPLITS[@]}" --max_entry_num 4000 --n_percentile 90 "${flags[@]}"
  )
done
run_cmd $PYTHON src/data/preprocess_judge.py "${SPLITS[@]}" --model_pair claude gemini --favored gemini
run_cmd $PYTHON src/data/preprocess_coding.py "${SPLITS[@]}" --dataset_name newfacade/LeetCodeDataset \
    --min_constraint_num 10 --difficulty Medium Hard
