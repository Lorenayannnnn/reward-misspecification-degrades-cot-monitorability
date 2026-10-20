"""
Table 3: explicit vs implicit intent on the Length task, in the paired layout of
generate_table_monitorability.py. Per quantity an Explicit and an Implicit column: misaligned hack rate,
both g-mean², and the misaligned model's median CoT words, averaged over seeds.

    python src/paper/generate_table_implicit_vs_explicit.py <implicit dirs...> --explicit_dirs <explicit dirs...> \\
        --monitor gpt-5.6-luna --out outputs/baseline_training/length/underspecified_intent/implicit_vs_explicit_table.tex
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.paper.plot_paper_monitorability_figures import TASK_ORDER, TASK_TITLES, task_and_policy
from src.paper.generate_table_monitorability import (G_MEAN_COLUMNS, PAIRED_COLUMN_PADDING, POLICY_ORDER, POLICY_SHORT_NAMES,
                                        TABLE_WIDTH, check_paired, load_rows, paired_cells)
from src.utils import load_dump

def median_cot_words(experiment_dir: Path) -> dict[str, tuple[float, int, int]]:
    """
    Return {"misaligned_model": (mean CoT length, final step, number of seeds)}, where the CoT length of
    one seed is the median word count over its final test dump. Raises when seeds end at different steps.
    """
    out = {}
    for role in ("misaligned_model",):
        per_seed: dict[int, float] = {}
        final_steps = set()
        for seed_dir in sorted(experiment_dir.glob("seed_*")):
            dumps = sorted((seed_dir / role / "inference_test").glob("step_*.jsonl"))
            if not dumps:
                raise FileNotFoundError(f"no test dumps under {seed_dir / role}")
            final_steps.add(int(dumps[-1].stem.split("_")[1]))
            per_seed[int(seed_dir.name.split("_")[1])] = statistics.median(
                len(record["cot"].split()) for record in load_dump(dumps[-1]))
        if len(final_steps) != 1:
            raise ValueError(f"{experiment_dir} {role}: seeds end at different steps {sorted(final_steps)}")
        out[role] = (statistics.mean(per_seed.values()), final_steps.pop(), len(per_seed))
    return out


# Column headers: baseline (--explicit_dirs), then experiment.
CONDITION_NAMES = ("Explicit", "Implicit")

# Spanning headers: the three quantities paired_cells emits, then the CoT words.
METRIC_GROUPS = ("Misaligned hack rate", *(header for _, header in G_MEAN_COLUMNS))
COT_WORDS_GROUP = "Misaligned CoT words"

# Formatted with {monitor}, {n_seeds}, {control_phrase} and {final_step}.
CAPTION_TEMPLATE = (
    r"Monitorability scores and misaligned models' hack rate when the task prompt states the length "
    r"limit (explicit, the baseline runs) vs.\ hides it (implicit), at the final training step "
    r"({final_step}) using {monitor} as the monitor. Metric values are averaged over {n_seeds} seeds; "
    r"the last columns give the misaligned model's median CoT word count over the test rollouts, per "
    r"seed, averaged over the seeds. The control models' hack rate is {control_phrase} under "
    r"both conditions."
)


def cot_word_cells(explicit_words: float, implicit_words: float) -> list[str]:
    return [str(round(explicit_words)), str(round(implicit_words))]


def latex_table(rows: dict, explicit_rows: dict, words: dict, explicit_words: dict,
                monitor: str, label: str) -> str:
    groups = [*METRIC_GROUPS, COT_WORDS_GROUP]
    tasks = [t for t in TASK_ORDER if t in rows]
    lines = [r"\begin{table}[h]", r"\centering", r"\small", rf"\setlength{{\tabcolsep}}{{{PAIRED_COLUMN_PADDING}}}",
             rf"\resizebox{{{TABLE_WIDTH}}}{{!}}{{%",
             r"\begin{tabular}{ll" + "c" * (2 * len(groups)) + "}", r"\toprule",
             " & ".join(["", "", *(rf"\multicolumn{{2}}{{c}}{{{g}}}" for g in groups)]) + r" \\",
             " ".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(groups))),
             " & ".join(["Task", "Model", *(CONDITION_NAMES * len(groups))]) + r" \\",
             r"\midrule"]
    final_steps = set()
    for task_index, task in enumerate(tasks):
        policies = [p for p in POLICY_ORDER if p in rows[task]]
        for i, policy in enumerate(policies):
            task_cell = rf"\multirow{{{len(policies)}}}{{*}}{{{TASK_TITLES[task]}}}" if i == 0 else ""
            cells = paired_cells(explicit_rows[task][policy], rows[task][policy], monitor, latex=True, highlight=False)
            cells += cot_word_cells(explicit_words[task][policy]["misaligned_model"][0],
                                    words[task][policy]["misaligned_model"][0])
            final_steps |= {rows[task][policy]["final_step"], words[task][policy]["misaligned_model"][1],
                            explicit_words[task][policy]["misaligned_model"][1]}
            lines.append(" & ".join([task_cell, POLICY_SHORT_NAMES[policy], *cells]) + r" \\")
        if task_index < len(tasks) - 1:
            lines.append(r"\midrule")
    if len(final_steps) != 1:
        raise ValueError(f"reports and dumps disagree on the final step: {sorted(final_steps)}")
    # Control hack rate per task as its range over both conditions, as in generate_table_monitorability.
    tasks_of_range: dict[str, list[str]] = {}
    for task in tasks:
        rates = [f"{c[task][p]['hack_rate_control_model'][0]:.2f}" for c in (explicit_rows, rows) for p in rows[task]]
        low, high = min(rates, key=float), max(rates, key=float)
        tasks_of_range.setdefault(f"${low}$" if low == high else f"${low}$--${high}$", []).append(TASK_TITLES[task])
    control_phrase = " and ".join(
        f"{value} for the {' and '.join(names)} task{'s' if len(names) > 1 else ''}"
        for value, names in tasks_of_range.items())
    n_seeds = sorted({v["n_seeds"] for r in (rows, explicit_rows)
                      for per_policy in r.values() for v in per_policy.values()})
    caption = CAPTION_TEMPLATE.format(monitor=monitor, n_seeds="/".join(map(str, n_seeds)),
                                      control_phrase=control_phrase, final_step=final_steps.pop())
    lines += [r"\bottomrule", r"\end{tabular}", r"}", rf"\caption{{{caption}}}",
              rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def plain_text_table(rows: dict, explicit_rows: dict, words: dict, explicit_words: dict, monitor: str) -> str:
    """
    Render the same table for the terminal.
    """
    header = ["Task", "Model", "Step"] + [f"{name} ({condition})"
                                          for name in ("hack (mis)", "intervention g²", "reward-intent g²",
                                                       "CoT words (mis)")
                                          for condition in CONDITION_NAMES]
    body = [[TASK_TITLES[task], POLICY_SHORT_NAMES[policy], str(rows[task][policy]["final_step"]),
             *paired_cells(explicit_rows[task][policy], rows[task][policy], monitor, latex=False, highlight=False),
             *cot_word_cells(explicit_words[task][policy]["misaligned_model"][0],
                             words[task][policy]["misaligned_model"][0])]
            for task in TASK_ORDER if task in rows for policy in POLICY_ORDER if policy in rows[task]]
    body = [[cell.replace("$", "") for cell in row] for row in body]
    widths = [max(len(r[i]) for r in [header] + body) for i in range(len(header))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(r, widths, strict=True)) for r in [header] + body)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("implicit_dirs", type=Path, nargs="+",
                   help="The implicit experiment dirs (dataset/org/policy/max_tokens).")
    p.add_argument("--explicit_dirs", type=Path, nargs="+", required=True,
                   help="The explicit (baseline) run of each (task, policy), the comparison's baseline.")
    p.add_argument("--monitor", required=True,
                   help="The monitor whose report fills the g-mean² columns (e.g. gpt-5.6-luna).")
    p.add_argument("--out", type=Path, required=True, help="Where to write the LaTeX table (.tex).")
    p.add_argument("--label", required=True, help="The table's LaTeX label, e.g. tab:length_implicit_vs_explicit.")
    args = p.parse_args()

    rows = load_rows(args.implicit_dirs, [args.monitor])
    explicit_rows = load_rows(args.explicit_dirs, [args.monitor])
    check_paired(rows, explicit_rows)
    words: dict[str, dict[str, dict]] = {}
    explicit_words: dict[str, dict[str, dict]] = {}
    for target, dirs in ((words, args.implicit_dirs), (explicit_words, args.explicit_dirs)):
        for experiment_dir in dirs:
            task, policy = task_and_policy(experiment_dir)
            target.setdefault(task, {})[policy] = median_cot_words(experiment_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(latex_table(rows, explicit_rows, words, explicit_words, args.monitor, args.label))
    print(plain_text_table(rows, explicit_rows, words, explicit_words, args.monitor))
    print(f"\n[table] wrote {args.out}")


if __name__ == "__main__":
    main()
