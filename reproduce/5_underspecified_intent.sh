#!/usr/bin/env bash
# Section 5.1: underspecified intent (Table 3, Appendix D.2). The length pairs are retrained with the word
# limit hidden from the input (intent_in_input=implicit). Needs the finished length baseline runs.
# Usage:
#   bash reproduce/5_underspecified_intent.sh        print every command without running it (the default)
#   bash reproduce/5_underspecified_intent.sh run    run them in order, stopping at the first failure
# Stages: run all the following unless you specify otherwise, e.g. STAGES="score table" bash reproduce/5_underspecified_intent.sh run
#   train    retrain the length pairs with the word limit hidden  ->  outputs/baseline_training/length/<dataset>_implicit/
#   score    replay gpt-4.1-mini, add Y, compute g-mean²
#   table    Table 3 and the Appendix D.2 figures
source "$(dirname "${BASH_SOURCE[0]}")/_paper_common.sh"
STAGES="${STAGES:-train score table}"
if [[ " $STAGES " == *" train "* ]]; then
  for policy in $POLICIES; do for seed in $SEEDS; do
    run_cmd env SEED="$seed" INTENT_IN_INPUT=implicit bash "scripts/length/length_train_$policy.sh" both
  done; done
fi
implicit_dirs=(); explicit_dirs=()
for policy in $POLICIES; do
  implicit_dirs+=("$(experiment_dir baseline_training length "$policy" implicit)")
  explicit_dirs+=("$(experiment_dir baseline_training length "$policy" explicit)")
done
if [[ " $STAGES " == *" score "* ]]; then
  for d in "${implicit_dirs[@]}"; do score "$d"; done
fi
if [[ " $STAGES " == *" table "* ]]; then
  run_cmd $PYTHON src/paper/generate_table_implicit_vs_explicit.py "${implicit_dirs[@]}" --explicit_dirs "${explicit_dirs[@]}" \
      --monitor "$ONLINE_MONITOR" --out outputs/baseline_training/length/underspecified_intent/implicit_vs_explicit_table.tex \
      --label tab:length_implicit_vs_explicit
  run_cmd $PYTHON src/paper/plot_paper_monitorability_figures.py --monitor "$ONLINE_MONITOR" "$REPLAY_MONITOR" \
      --training_name "Underspecified Intent" --out_dir outputs/baseline_training/length/underspecified_intent "${implicit_dirs[@]}"
fi
