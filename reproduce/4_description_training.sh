#!/usr/bin/env bash
# Section 4: reward description training (Table 2, Appendix C). Needs the finished baseline runs.
# Usage:
#   bash reproduce/4_description_training.sh        print every command without running it (the default)
#   bash reproduce/4_description_training.sh run    run them in order, stopping at the first failure
# Stages: run all the following unless you specify otherwise, e.g. STAGES="score table" bash reproduce/4_description_training.sh run
#   generate  sample probes at every checkpoint, keep the best description  ->  <model dir>/description.txt
#   train     retrain with it appended to half of every group  ->  outputs/reward_description_training/
#   score     replay gpt-4.1-mini, add Y, compute g-mean²
#   table     Table 2 and the Appendix C.2 figures
#   DESCRIPTION_MODEL   the description generator (default gpt-5.6-terra, as in the paper)
source "$(dirname "${BASH_SOURCE[0]}")/_paper_common.sh"
STAGES="${STAGES:-generate train score table}"
DESCRIPTION_MODEL="${DESCRIPTION_MODEL:-gpt-5.6-terra}"
if [[ " $STAGES " == *" generate "* ]]; then
  for task in $TASKS; do for policy in $POLICIES; do for seed in $SEEDS; do
    run_cmd env SEED="$seed" bash "scripts/$task/${task}_generate_description_$policy.sh" both --description_model_name "$DESCRIPTION_MODEL"
  done; done; done
fi
if [[ " $STAGES " == *" train "* ]]; then
  for task in $TASKS; do for policy in $POLICIES; do for seed in $SEEDS; do
    run_cmd env SEED="$seed" bash "scripts/$task/${task}_train_${policy}_description_training.sh" both
  done; done; done
fi
description_dirs=(); baseline_dirs=()
for task in $TASKS; do for policy in $POLICIES; do
  description_dirs+=("$(experiment_dir reward_description_training "$task" "$policy")")
  baseline_dirs+=("$(experiment_dir baseline_training "$task" "$policy")")
done; done
if [[ " $STAGES " == *" score "* ]]; then
  for d in "${description_dirs[@]}"; do score "$d"; done
fi
if [[ " $STAGES " == *" table "* ]]; then
  run_cmd $PYTHON src/paper/generate_table_monitorability.py "${description_dirs[@]}" --baseline_dirs "${baseline_dirs[@]}" \
      --monitor "$ONLINE_MONITOR" --out outputs/reward_description_training/monitorability_baseline_vs_description_table.tex \
      --label tab:sec_4_description_vs_baseline
  run_cmd $PYTHON src/paper/plot_paper_monitorability_figures.py --monitor "$ONLINE_MONITOR" "$REPLAY_MONITOR" \
      --training_name Description "${description_dirs[@]}"
fi
