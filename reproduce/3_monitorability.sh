#!/usr/bin/env bash
# Section 3.5: score the baseline runs (replay gpt-4.1-mini, add Y, compute the hack rate and both
# g-mean²), then build Table 1 and the Appendix B figures.
# Usage:
#   bash reproduce/3_monitorability.sh        print every command without running it (the default)
#   bash reproduce/3_monitorability.sh run    Generate plots and tables
# Stages: run all the following unless you specify otherwise, e.g. STAGES="table figures" bash reproduce/3_monitorability.sh run
#   score    replay gpt-4.1-mini over the dumps, add Y, compute the hack rate and both g-mean²
#   table    Table 1 (its gpt-4.1-mini columns are Table 5)
#   figures  the Appendix B figures: one per task, then all tasks in one figure (Figure 3)
source "$(dirname "${BASH_SOURCE[0]}")/_paper_common.sh"
STAGES="${STAGES:-score table figures}"
dirs=()
for task in $TASKS; do for policy in $POLICIES; do dirs+=("$(experiment_dir baseline_training "$task" "$policy")"); done; done
if [[ " $STAGES " == *" score "* ]]; then
  for d in "${dirs[@]}"; do score "$d"; done
fi
if [[ " $STAGES " == *" table "* ]]; then
  run_cmd $PYTHON src/paper/generate_table_monitorability.py "${dirs[@]}" --monitor "$ONLINE_MONITOR" "$REPLAY_MONITOR" \
      --out outputs/baseline_training/monitorability_table.tex --label tab:sec_3_main
fi
if [[ " $STAGES " == *" figures "* ]]; then
  run_cmd $PYTHON src/paper/plot_paper_monitorability_figures.py --monitor "$ONLINE_MONITOR" "$REPLAY_MONITOR" \
      --training_name Baseline "${dirs[@]}"
  run_cmd $PYTHON src/paper/plot_paper_monitorability_figures.py --monitor "$ONLINE_MONITOR" --training_name Baseline \
      --out outputs/baseline_training/figures/all_tasks.pdf "${dirs[@]}"
fi
