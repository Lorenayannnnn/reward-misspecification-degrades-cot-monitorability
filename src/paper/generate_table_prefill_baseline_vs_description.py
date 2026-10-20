"""
Table 6: base-model prefilling, baseline vs reward description training, from the reports
prefill_base_model.py and compute_monitorability.py wrote. Per (task, policy): the misaligned hack
rate, the base model's hack rate when continuing the misaligned CoT (the prefill hack rate), and
both g-mean², each as (Baseline, Description) seed means.

    python src/paper/generate_table_prefill_baseline_vs_description.py <description dirs...> \\
        --baseline_dirs <baseline dirs...> --monitor gpt-5.6-luna --out <table.tex>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.paper.plot_paper_monitorability_figures import TASK_ORDER, TASK_TITLES, read_report, task_and_policy
from src.paper.generate_table_monitorability import (BASELINE_NAME, EXPERIMENT_NAME, PAIRED_COLUMN_PADDING, POLICY_ORDER,
                                        POLICY_SHORT_NAMES, TABLE_WIDTH, column_average)
from src.utils import load_dump

# Column group headers, in order.
GROUPS = ("Misaligned Hack Rate", "Prefill Hack Rate",
          "Intervention g-mean$^2$", "Reward-intent g-mean$^2$")

CAPTION_TEMPLATE = (
    r"Prefilling the untrained base model with the misaligned models' CoTs at the final checkpoint, under "
    r"baseline vs.\ description training (hack rate, mean over {n_seeds} seeds; the final row averages each "
    r"column). ``Misaligned hack rate'' is the model's own hack rate at that checkpoint. ``Prefill hack "
    r"rate'' is the hack rate of the \emph{{untrained base model}} writing the final output after the "
    r"misaligned model's CoT: how much of the hack comes from the CoT alone. The two "
    r"g-mean$^2$ columns are the CoT monitorability scores at the same checkpoint, with {monitor} "
    r"as the monitor."
)


def final_step_report(experiment_dir: Path) -> dict:
    """
    Read the prefill report at its last scored step; raises when there is none.
    """
    reports = sorted((experiment_dir / "prefill").glob("step_*.json"))
    if not reports:
        raise FileNotFoundError(f"no prefill reports under {experiment_dir}/prefill/; run prefill_base_model.py first")
    return json.loads(reports[-1].read_text())


def hack_rate_over_seeds(experiment_dir: Path, dump_subpath: str, step: int) -> float:
    """
    Average the hack rate over seeds, each seed over its own labeled rollouts (Y missing or NaN
    dropped), as the monitorability pipeline computes it.
    """
    per_seed = []
    for seed_dir in sorted(experiment_dir.glob("seed_*")):
        labels = [record["is_reward_hacking_Y"]
                  for record in load_dump(seed_dir / dump_subpath / f"step_{step:06d}.jsonl")
                  if isinstance(record.get("is_reward_hacking_Y"), (int, float))
                  and not math.isnan(record["is_reward_hacking_Y"])]
        per_seed.append(sum(labels) / len(labels))
    return sum(per_seed) / len(per_seed)


def prefill_values(experiment_dir: Path, monitor: str) -> dict:
    """
    Read the four table quantities, the final step and the seed count of one experiment dir's
    misaligned model. Hack rates are recomputed from the dumps so they match the monitorability tables.
    """
    base = final_step_report(experiment_dir)
    monitorability = read_report(experiment_dir, monitor)["monitorability"]["mean"]
    final_step = max(monitorability, key=int)
    if int(final_step) != base["step"]:
        raise ValueError(f"{experiment_dir}: prefill at step {base['step']} but monitorability ends at {final_step}")
    return {"original": hack_rate_over_seeds(experiment_dir, "misaligned_model/inference_test", base["step"]),
            "base_model": hack_rate_over_seeds(
                experiment_dir, "misaligned_model/prefill", base["step"]),
            "intervention": monitorability[final_step]["intervention_g_mean_squared"],
            "reward_intent": monitorability[final_step]["reward_intent_g_mean_squared"],
            "step": base["step"], "n_seeds": len(base["seeds"])}


def paired_cells(baseline: dict, description: dict) -> list[str]:
    """
    Build one row's cells: per quantity the two condition values.
    """
    cells = []
    for key, signed in (("original", False), ("base_model", False), ("intervention", False), ("reward_intent", False)):
        cells += [f"${condition[key]:+.2f}$" if signed else f"{condition[key]:.2f}"
                  for condition in (baseline, description)]
    return cells


def latex_table(rows: dict, baseline_rows: dict, label: str, monitor: str) -> str:
    """
    Render the paired booktabs table, laid out like generate_table_monitorability's paired mode.
    """
    tasks = [t for t in TASK_ORDER if t in rows]
    lines = [r"\begin{table}[h]", r"\centering", r"\small", rf"\setlength{{\tabcolsep}}{{{PAIRED_COLUMN_PADDING}}}",
             rf"\resizebox{{{TABLE_WIDTH}}}{{!}}{{%",
             r"\begin{tabular}{ll" + "c" * (2 * len(GROUPS)) + "}", r"\toprule",
             " & ".join(["", "", *(rf"\multicolumn{{2}}{{c}}{{{g}}}" for g in GROUPS)]) + r" \\",
             " ".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(GROUPS))),
             " & ".join(["Task", "Model", *([BASELINE_NAME, EXPERIMENT_NAME] * len(GROUPS))]) + r" \\",
             r"\midrule"]
    for task_index, task in enumerate(tasks):
        policies = [p for p in POLICY_ORDER if p in rows[task]]
        for i, policy in enumerate(policies):
            task_cell = rf"\multirow{{{len(policies)}}}{{*}}{{{TASK_TITLES[task]}}}" if i == 0 else ""
            cells = paired_cells(baseline_rows[task][policy], rows[task][policy])
            lines.append(" & ".join([task_cell, POLICY_SHORT_NAMES[policy], *cells]) + r" \\")
        if task_index < len(tasks) - 1:
            lines.append(r"\midrule")
    # Average row: each column's mean over every (task, policy) row.
    all_pairs = [(t, p) for t in tasks for p in POLICY_ORDER if p in rows[t]]
    averages = [column_average([condition_rows[t][p][key] for t, p in all_pairs])
                for key in ("original", "base_model", "intervention", "reward_intent")
                for condition_rows in (baseline_rows, rows)]
    lines += [r"\midrule", " & ".join([r"\multicolumn{2}{l}{Average}", *(f"{a:.2f}" for a in averages)]) + r" \\"]
    n_seeds = sorted({v["n_seeds"] for r in (rows, baseline_rows)
                      for per_policy in r.values() for v in per_policy.values()})
    caption = CAPTION_TEMPLATE.format(n_seeds="/".join(map(str, n_seeds)), monitor=monitor)
    lines += [r"\bottomrule", r"\end{tabular}", r"}", rf"\caption{{{caption}}}",
              rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def plain_text_table(rows: dict, baseline_rows: dict) -> str:
    """
    Render the same table for the terminal, with each row's final step.
    """
    header = ["Task", "Model", "Step"] + [f"{name} ({condition})"
                                          for name in ("hack (mis)", "prefill hack", "iG²", "rG²")
                                          for condition in (BASELINE_NAME, EXPERIMENT_NAME)]
    body = [[TASK_TITLES[task], POLICY_SHORT_NAMES[policy], str(rows[task][policy]["step"]),
             *[cell.replace("$", "") for cell in paired_cells(baseline_rows[task][policy], rows[task][policy])]]
            for task in TASK_ORDER if task in rows for policy in POLICY_ORDER if policy in rows[task]]
    widths = [max(len(r[i]) for r in [header] + body) for i in range(len(header))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(r, widths, strict=True)) for r in [header] + body)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("description_dirs", type=Path, nargs="+",
                   help="Description-training experiment dirs (dataset/org/policy/max_tokens).")
    p.add_argument("--baseline_dirs", type=Path, nargs="+", required=True,
                   help="The baseline run of each (task, policy), the comparison's baseline.")
    p.add_argument("--monitor", required=True,
                   help="The monitor whose g-mean² scores fill the monitorability columns.")
    p.add_argument("--out", type=Path, required=True, help="Where to write the LaTeX table (.tex).")
    p.add_argument("--label", required=True, help="The table's LaTeX label.")
    args = p.parse_args()

    rows: dict[str, dict[str, dict]] = {}
    baseline_rows: dict[str, dict[str, dict]] = {}
    for target, dirs in ((rows, args.description_dirs), (baseline_rows, args.baseline_dirs)):
        for experiment_dir in dirs:
            task, policy = task_and_policy(experiment_dir)
            if policy in target.get(task, {}):
                raise ValueError(f"two experiment dirs for {task}/{policy}; pass one per (task, policy)")
            target.setdefault(task, {})[policy] = prefill_values(experiment_dir, args.monitor)
    for task, per_policy in rows.items():
        for policy in per_policy:
            if policy not in baseline_rows.get(task, {}):
                raise ValueError(f"no baseline dir for {task}/{policy}")
            if baseline_rows[task][policy]["step"] != per_policy[policy]["step"]:
                raise ValueError(f"{task}/{policy}: final steps differ between the conditions")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(latex_table(rows, baseline_rows, args.label, args.monitor))
    print(plain_text_table(rows, baseline_rows))
    print(f"\n[table] wrote {args.out}")


if __name__ == "__main__":
    main()
