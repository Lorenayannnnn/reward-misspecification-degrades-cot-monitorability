#!/usr/bin/env bash
# Section 3: baseline training, one control/misaligned pair per (task, policy, seed). The 27 pairs run
# through scripts/<task>/<task>_train_<policy>.sh and are placed under outputs/baseline_training/<task>/.
#   bash reproduce/2_baseline_training.sh        print the 27 commands without running them (the default)
#   bash reproduce/2_baseline_training.sh run    run them; narrow with e.g. TASKS="length" SEEDS="0"
# Coding needs two Sandbox Fusion graders; ports are set at the top of scripts/coding/_coding_common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_paper_common.sh"
for task in $TASKS; do
  for policy in $POLICIES; do
    for seed in $SEEDS; do
      run_cmd env SEED="$seed" bash "scripts/$task/${task}_train_$policy.sh" both
    done
  done
done
