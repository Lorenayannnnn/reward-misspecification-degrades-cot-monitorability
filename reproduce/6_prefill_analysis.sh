#!/usr/bin/env bash
# Section 5.2: base-model prefilling (Table 6, Appendix D.1). The base model continues each misaligned
# model's CoT; its hack rate shows how much of the hack comes from the CoT alone. Needs the Section 3 and 4 runs.
# Usage:
#   bash reproduce/6_prefill_analysis.sh        print every command without running it (the default)
#   bash reproduce/6_prefill_analysis.sh run    run them in order, stopping at the first failure
# Stages: run all the following unless you specify otherwise, e.g. STAGES="table" bash reproduce/6_prefill_analysis.sh run
#   sample   prefill the base model with each misaligned model's CoT and grade its outputs
#   table    Table 6
# Coding regrades the outputs, so Sandbox Fusion must be up at the URL the runs trained against.
source "$(dirname "${BASH_SOURCE[0]}")/_paper_common.sh"
STAGES="${STAGES:-sample table}"
description_dirs=(); baseline_dirs=()
for task in $TASKS; do for policy in $POLICIES; do
  description_dirs+=("$(experiment_dir reward_description_training "$task" "$policy")")
  baseline_dirs+=("$(experiment_dir baseline_training "$task" "$policy")")
done; done
if [[ " $STAGES " == *" sample "* ]]; then
  for d in "${baseline_dirs[@]}" "${description_dirs[@]}"; do
    run_cmd $PYTHON src/analysis/prefill_base_model.py "$d"
  done
fi
if [[ " $STAGES " == *" table "* ]]; then
  run_cmd $PYTHON src/paper/generate_table_prefill_baseline_vs_description.py "${description_dirs[@]}" \
      --baseline_dirs "${baseline_dirs[@]}" --monitor "$ONLINE_MONITOR" \
      --out outputs/reward_description_training/prefill_baseline_vs_description_table.tex \
      --label tab:prefill_baseline_vs_description
fi
