"""
The hack rate and monitorability table at the final training step (Table 1): per (task, policy),
both models' hack rate and each monitor's two g-mean², mean ± std over seeds, as LaTeX. With
--baseline_dirs it pairs each run with its baseline instead (Table 2, description vs baseline).

    python src/paper/generate_table_monitorability.py <experiment dirs...> --monitor gpt-5.6-luna gpt-4.1-mini \\
        --out outputs/baseline_training/monitorability_table.tex --label tab:sec_3_main
    python src/paper/generate_table_monitorability.py <description dirs...> --baseline_dirs <baseline dirs...> \\
        --monitor gpt-5.6-luna --out outputs/reward_description_training/monitorability_baseline_vs_description_table.tex
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.paper.plot_paper_monitorability_figures import TASK_ORDER, TASK_TITLES, read_report, task_and_policy

# Sub-row order within a task, and the printed name of each policy.
POLICY_ORDER = ("gpt-oss-20b", "gpt-oss-120b", "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
POLICY_SHORT_NAMES = {"gpt-oss-20b": "gpt-oss-20b", "gpt-oss-120b": "gpt-oss-120b",
                      "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "Nemotron"}

# The tabular is scaled to this width with \resizebox (needs graphicx).
TABLE_WIDTH = r"1\linewidth"

# The two g-mean² columns: (report key, column header).
G_MEAN_COLUMNS = (("intervention_g_mean_squared", r"Intervention g-mean$^2$"),
                  ("reward_intent_g_mean_squared", r"Reward-intent g-mean$^2$"))

# Paired mode: column headers of the baseline and experiment conditions.
BASELINE_NAME, EXPERIMENT_NAME = "Baseline", "Description"
# Paired mode: \tabcolsep halved from LaTeX's 6pt to fit 11 columns.
PAIRED_COLUMN_PADDING = "3pt"
# Paired mode: the appendix the caption points to.
APPENDIX_LABEL = "appendix:description_training"
# Paired mode caption; generate_table_implicit_vs_explicit passes its own.
DESCRIPTION_CAPTION_TEMPLATE = (
    r"Monitorability scores and misaligned models' hack rate under baseline vs.\ reward description training "
    r"at the final training step using {monitor} as the monitor across all models and tasks. Metric values "
    r"are averaged over {n_seeds} seeds. The control models' hack rates remain similar to those of the "
    r"baseline runs, with {control_phrase}. See Appendix~\ref{{" + APPENDIX_LABEL + r"}} for details."
)
# Paired mode colors. BrickRed needs \usepackage[dvipsnames]{xcolor}; DeltaGreen must be
# \definecolor'd in the preamble.
INCREASE_COLOR, DECREASE_COLOR = "DeltaGreen", "BrickRed"


def final_checkpoint_values(report: dict) -> dict:
    """
    Read both models' hack rate and the two g-mean² at the report's last scored step, as (mean, std).
    Raises when that step lacks either model's hack rate.
    """
    monitorability, curves = report["monitorability"], report["run_curves"]
    final_step = max(monitorability["mean"], key=int)
    for role in ("control_model", "misaligned_model"):
        if final_step not in curves["mean"][role]:
            raise ValueError(f"final step {final_step} has monitor metrics but no {role} hack rate")
    values = {f"hack_rate_{role}": (curves["mean"][role][final_step]["hack_rate"],
                                   curves["std"][role][final_step]["hack_rate"])
              for role in ("control_model", "misaligned_model")}
    for key, _header in G_MEAN_COLUMNS:
        values[key] = (monitorability["mean"][final_step][key], monitorability["std"][final_step][key])
    values["final_step"] = int(final_step)
    values["n_seeds"] = monitorability["n_seeds"][final_step]
    return values


def row_values(reports_of_monitor: dict[str, dict], monitors: list[str]) -> dict:
    """
    Build one (task, policy) row: hack rates, final step and seed count once, g-mean² per monitor.
    Raises when the monitors' reports disagree on the shared values.
    """
    per_monitor = {monitor: final_checkpoint_values(reports_of_monitor[monitor]) for monitor in monitors}
    shared = ("hack_rate_control_model", "hack_rate_misaligned_model", "final_step", "n_seeds")
    first = per_monitor[monitors[0]]
    for monitor in monitors[1:]:
        if differing := [k for k in shared if per_monitor[monitor][k] != first[k]]:
            raise ValueError(f"monitors {monitors[0]} and {monitor} disagree on {differing}; rescore both reports")
    return {**{k: first[k] for k in shared},
            "g_mean": {monitor: {key: per_monitor[monitor][key] for key, _ in G_MEAN_COLUMNS} for monitor in monitors}}


def column_average(values: list[float]) -> float:
    """
    Average the defined (non-NaN) entries; NaN when none are.
    """
    defined = [v for v in values if v is not None and not math.isnan(v)]
    return sum(defined) / len(defined) if defined else float("nan")


def format_mean_std(mean_and_std: tuple, latex: bool) -> str:
    r"""
    Format as '0.63 ± 0.05', or in LaTeX '$0.63_{\pm 0.05}$'; '--' when undefined.
    """
    mean, std = mean_and_std
    if mean is None or math.isnan(mean):
        return "--"
    return f"${mean:.2f}_{{\\pm {std:.2f}}}$" if latex else f"{mean:.2f} ± {std:.2f}"


def table_cells(values: dict, monitors: list[str], latex: bool) -> list[str]:
    """
    Build one row's value cells: both hack rates, then each monitor's g-mean².
    """
    hack = [format_mean_std(values[f"hack_rate_{role}"], latex) for role in ("control_model", "misaligned_model")]
    g_mean = [format_mean_std(values["g_mean"][monitor][key], latex)
              for monitor in monitors for key, _ in G_MEAN_COLUMNS]
    return hack + g_mean


def latex_table(rows: dict[str, dict[str, dict]], monitors: list[str], label: str) -> str:
    """
    Render the booktabs table: a multirow cell per task, a sub-row per policy, a header spanning each
    monitor's g-mean² columns.
    """
    n_g = len(G_MEAN_COLUMNS)
    n_hack = 2
    # One monitor: its name spans its two g-mean² columns. Several: each metric spans the monitors.
    metric_major = len(monitors) > 1
    n_per_group = len(monitors) if metric_major else n_g
    groups = [header for _, header in G_MEAN_COLUMNS] if metric_major else list(monitors)
    sub_headers = (list(monitors) * n_g) if metric_major else [header for _ in monitors for _, header in G_MEAN_COLUMNS]
    # 1-based column range of each spanning header.
    spans = [(3 + n_hack + n_per_group * i, 2 + n_hack + n_per_group * (i + 1)) for i in range(len(groups))]
    hack_group = [r"\multicolumn{2}{c}{Hack rate}"]
    hack_rule = [r"\cmidrule(lr){3-4}"]
    hack_headers = ["Control", "Misaligned"]
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\resizebox{{{TABLE_WIDTH}}}{{!}}{{%",
             r"\begin{tabular}{ll" + "c" * (n_hack + n_g * len(monitors)) + "}", r"\toprule",
             " & ".join(["", "", *hack_group,
                         *(rf"\multicolumn{{{n_per_group}}}{{c}}{{{group}}}" for group in groups)]) + r" \\",
             " ".join(hack_rule + [rf"\cmidrule(lr){{{a}-{b}}}" for a, b in spans]),
             " & ".join(["Task", "Model", *hack_headers, *sub_headers]) + r" \\",
             r"\midrule"]
    def to_metric_major(cells: list[str]) -> list[str]:
        """
        Reorder the monitor-major g-mean² cells (the hack rate cells stay in front).
        """
        if not metric_major:
            return cells
        g_cells = cells[n_hack:]
        return cells[:n_hack] + [g_cells[m * n_g + k] for k in range(n_g) for m in range(len(monitors))]
    for task_index, task in enumerate(t for t in TASK_ORDER if t in rows):
        policies = [p for p in POLICY_ORDER if p in rows[task]]
        for i, policy in enumerate(policies):
            task_cell = rf"\multirow{{{len(policies)}}}{{*}}{{{TASK_TITLES[task]}}}" if i == 0 else ""
            cells = to_metric_major(table_cells(rows[task][policy], monitors, latex=True))
            lines.append(" & ".join([task_cell, POLICY_SHORT_NAMES[policy], *cells]) + r" \\")
        if task_index < len([t for t in TASK_ORDER if t in rows]) - 1:
            lines.append(r"\midrule")
    # Average row: each column's mean over its defined entries.
    all_rows = [rows[t][p] for t in TASK_ORDER if t in rows for p in POLICY_ORDER if p in rows[t]]
    averages = [column_average([v[f"hack_rate_{role}"][0] for v in all_rows])
                for role in ("control_model", "misaligned_model")]
    averages += [column_average([v["g_mean"][monitor][key][0] for v in all_rows])
                 for monitor in monitors for key, _ in G_MEAN_COLUMNS]
    lines += [r"\midrule", " & ".join([r"\multicolumn{2}{l}{Average}", *to_metric_major(
        ["--" if math.isnan(a) else f"${a:.2f}$" for a in averages])]) + r" \\"]
    n_seeds = sorted({v["n_seeds"] for per_policy in rows.values() for v in per_policy.values()})
    monitor_phrase = (f"{monitors[0]} as the monitor" if len(monitors) == 1
                      else f"{', '.join(monitors[:-1])} and {monitors[-1]} as the monitors")
    lines += [r"\bottomrule", r"\end{tabular}", r"}",
              rf"\caption{{Hack rate and CoT monitorability metrics at the final checkpoint "
              rf"using {monitor_phrase}. Mean $\pm$ std over {'/'.join(map(str, n_seeds))} seeds. "
              rf"The final row averages each column over the {len(all_rows)} (task, model) pairs.}}",
              rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def plain_text_table(rows: dict[str, dict[str, dict]], monitors: list[str]) -> str:
    """
    Render the same table for the terminal, with the final step and seed count per row.
    """
    header = ["Task", "Model", "Step", "Seeds", "Hack (ctrl)", "Hack (mis)"] + [
        f"{monitor} {name}" for monitor in monitors for name in ("intervention g²", "reward-intent g²")]
    body = [[TASK_TITLES[task], POLICY_SHORT_NAMES[policy], str(v["final_step"]), str(v["n_seeds"]),
             *table_cells(v, monitors, latex=False)]
            for task in TASK_ORDER if task in rows for policy in POLICY_ORDER if policy in rows[task]
            for v in [rows[task][policy]]]
    widths = [max(len(r[i]) for r in [header] + body) for i in range(len(header))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(r, widths, strict=True)) for r in [header] + body)


def load_rows(experiment_dirs: list[Path], monitors: list[str]) -> dict[str, dict[str, dict]]:
    """
    Load {task: {policy: final-checkpoint values}} over the experiment dirs, one dir per (task, policy).
    """
    rows: dict[str, dict[str, dict]] = {}
    for experiment_dir in experiment_dirs:
        task, policy = task_and_policy(experiment_dir)
        if policy in rows.get(task, {}):
            raise ValueError(f"two experiment dirs for {task}/{policy}; pass one per (task, policy)")
        rows.setdefault(task, {})[policy] = row_values({m: read_report(experiment_dir, m) for m in monitors}, monitors)
    return rows


def rounded_quantity_pairs(baseline: dict, experiment: dict, monitor: str) -> list[tuple]:
    """
    Round each quantity's (baseline, experiment) means to the 2 decimals shown, so comparisons
    match the printed cells; None where NaN.
    """
    means = [[v["hack_rate_misaligned_model"][0], *(v["g_mean"][monitor][key][0] for key, _ in G_MEAN_COLUMNS)]
             for v in (baseline, experiment)]
    return [tuple(None if m is None or math.isnan(m) else round(m, 2) for m in pair) for pair in zip(*means)]


def paired_cells(baseline: dict, experiment: dict, monitor: str, latex: bool, highlight: bool) -> list[str]:
    """
    Build one row's value cells: per quantity the baseline and experiment means ("--" if NaN). With
    ``highlight`` (LaTeX only) a g-mean² experiment value is green above its baseline, red below.
    """
    cells = []
    for quantity_index, (baseline_mean, experiment_mean) in enumerate(
            rounded_quantity_pairs(baseline, experiment, monitor)):
        baseline_cell, experiment_cell = ("--" if m is None else f"{m:.2f}" for m in (baseline_mean, experiment_mean))
        if (latex and highlight and quantity_index > 0          # quantity 0 is the hack rate
                and None not in (baseline_mean, experiment_mean) and experiment_mean != baseline_mean):
            color = INCREASE_COLOR if experiment_mean > baseline_mean else DECREASE_COLOR
            experiment_cell = rf"\textcolor{{{color}}}{{{experiment_cell}}}"
        cells += [baseline_cell, experiment_cell]
    return cells


def average_row_cells(baseline: dict, experiment: dict, monitor: str) -> list[str]:
    """
    Build the Average row's LaTeX cells: the baseline mean, then the experiment mean with its change as
    a subscript, green when a g-mean² rose.
    """
    cells = []
    for quantity_index, (baseline_mean, experiment_mean) in enumerate(
            rounded_quantity_pairs(baseline, experiment, monitor)):
        if None in (baseline_mean, experiment_mean):
            raise ValueError("an Average entry is undefined; every (task, policy) row is NaN in that column")
        change = f"{experiment_mean - baseline_mean:+.2f}".replace("-0.00", "+0.00")
        experiment_cell = f"${experiment_mean:.2f}_{{({change})}}$"
        if quantity_index > 0 and experiment_mean > baseline_mean:          # quantity 0 is the hack rate
            experiment_cell = rf"\textcolor{{{INCREASE_COLOR}}}{{{experiment_cell}}}"
        cells += [f"{baseline_mean:.2f}", experiment_cell]
    return cells


def check_paired(rows: dict, baseline_rows: dict) -> None:
    """
    Raise when a (task, policy) lacks its baseline or the two conditions end at different steps.
    """
    for task, per_policy in rows.items():
        for policy, values in per_policy.items():
            if policy not in baseline_rows.get(task, {}):
                raise ValueError(f"no baseline dir for {task}/{policy}")
            if baseline_rows[task][policy]["final_step"] != values["final_step"]:
                raise ValueError(f"{task}/{policy}: final step {values['final_step']} vs baseline "
                                 f"{baseline_rows[task][policy]['final_step']}; resume the shorter run")


def paired_latex_table(rows: dict, baseline_rows: dict, monitor: str, label: str,
                       condition_names: tuple[str, str], caption_template: str) -> str:
    """
    Render the paired booktabs table: per quantity a header spanning its two condition columns, a multirow
    cell per task, and ``caption_template`` formatted with {monitor}, {n_seeds} and {control_phrase}.
    """
    groups = ["Misaligned hack rate", *(header for _, header in G_MEAN_COLUMNS)]
    tasks = [t for t in TASK_ORDER if t in rows]
    # Narrow padding: the table is scaled to TABLE_WIDTH, so a narrower table means a larger font.
    lines = [r"\begin{table}[h]", r"\centering", r"\small", rf"\setlength{{\tabcolsep}}{{{PAIRED_COLUMN_PADDING}}}",
             rf"\resizebox{{{TABLE_WIDTH}}}{{!}}{{%",
             r"\begin{tabular}{ll" + "c" * (2 * len(groups)) + "}", r"\toprule",
             " & ".join(["", "", *(rf"\multicolumn{{2}}{{c}}{{{g}}}" for g in groups)]) + r" \\",
             " ".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(groups))),
             " & ".join(["Task", "Model", *(list(condition_names) * len(groups))]) + r" \\",
             r"\midrule"]
    for task_index, task in enumerate(tasks):
        policies = [p for p in POLICY_ORDER if p in rows[task]]
        for i, policy in enumerate(policies):
            task_cell = rf"\multirow{{{len(policies)}}}{{*}}{{{TASK_TITLES[task]}}}" if i == 0 else ""
            cells = paired_cells(baseline_rows[task][policy], rows[task][policy], monitor, latex=True, highlight=True)
            lines.append(" & ".join([task_cell, POLICY_SHORT_NAMES[policy], *cells]) + r" \\")
        if task_index < len(tasks) - 1:
            lines.append(r"\midrule")
    # Average row: per condition each quantity's mean; the experiment cell shows the change as a subscript.
    all_pairs = [(t, p) for t in tasks for p in POLICY_ORDER if p in rows[t]]
    def averaged(condition_rows: dict) -> dict:
        return {"hack_rate_misaligned_model": (column_average(
                    [condition_rows[t][p]["hack_rate_misaligned_model"][0] for t, p in all_pairs]), 0.0),
                "g_mean": {monitor: {key: (column_average(
                    [condition_rows[t][p]["g_mean"][monitor][key][0] for t, p in all_pairs]), 0.0)
                                     for key, _ in G_MEAN_COLUMNS}}}
    lines += [r"\midrule", " & ".join([r"\multicolumn{2}{l}{Average}",
                                       *average_row_cells(averaged(baseline_rows), averaged(rows), monitor)]) + r" \\"]
    # Control hack rate per task as its range over both conditions; tasks sharing a range are named together.
    tasks_of_range: dict[str, list[str]] = {}
    for task in tasks:
        rates = [f"{c[task][p]['hack_rate_control_model'][0]:.2f}" for c in (baseline_rows, rows) for p in rows[task]]
        low, high = min(rates, key=float), max(rates, key=float)
        tasks_of_range.setdefault(f"${low}$" if low == high else f"${low}$--${high}$", []).append(TASK_TITLES[task])
    control_phrase = " and ".join(
        f"{value} for the {' and '.join(names)} task{'s' if len(names) > 1 else ''}"
        for value, names in tasks_of_range.items())
    n_seeds = sorted({v["n_seeds"] for r in (rows, baseline_rows)
                      for per_policy in r.values() for v in per_policy.values()})
    caption = caption_template.format(monitor=monitor, n_seeds="/".join(map(str, n_seeds)),
                                      control_phrase=control_phrase)
    lines += [r"\bottomrule", r"\end{tabular}", r"}", rf"\caption{{{caption}}}",
              rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def paired_plain_text_table(rows: dict, baseline_rows: dict, monitor: str, condition_names: tuple[str, str]) -> str:
    """
    Render the paired table for the terminal.
    """
    header = ["Task", "Model", "Step"] + [f"{name} ({condition})"
                                          for name in ("hack (mis)", "intervention g²", "reward-intent g²")
                                          for condition in condition_names]
    body = [[TASK_TITLES[task], POLICY_SHORT_NAMES[policy], str(rows[task][policy]["final_step"]),
             *paired_cells(baseline_rows[task][policy], rows[task][policy], monitor, latex=False, highlight=False)]
            for task in TASK_ORDER if task in rows for policy in POLICY_ORDER if policy in rows[task]]
    widths = [max(len(r[i]) for r in [header] + body) for i in range(len(header))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(r, widths, strict=True)) for r in [header] + body)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment_dirs", type=Path, nargs="+", help="Experiment dirs (dataset/org/policy/max_tokens).")
    p.add_argument("--monitor", nargs="+", required=True,
                   help="Monitors whose reports fill the g-mean² columns, one spanning header each "
                        "(e.g. gpt-5.6-luna gpt-4.1-mini).")
    p.add_argument("--out", type=Path, required=True, help="Where to write the LaTeX table (.tex).")
    p.add_argument("--label", required=True, help="The table's LaTeX label, e.g. tab:sec_3_main.")
    p.add_argument("--baseline_dirs", type=Path, nargs="+",
                   help="Paired mode: the baseline run of each (task, policy), e.g. outputs/baseline_training/... "
                        "for description-trained runs.")
    args = p.parse_args()

    rows = load_rows(args.experiment_dirs, args.monitor)          # {task: {policy: values at the final checkpoint}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.baseline_dirs:
        if len(args.monitor) != 1:
            raise ValueError("paired mode takes exactly one --monitor")
        baseline_rows = load_rows(args.baseline_dirs, args.monitor)
        check_paired(rows, baseline_rows)
        (monitor,) = args.monitor
        args.out.write_text(paired_latex_table(rows, baseline_rows, monitor, args.label,
                                               (BASELINE_NAME, EXPERIMENT_NAME), DESCRIPTION_CAPTION_TEMPLATE))
        print(paired_plain_text_table(rows, baseline_rows, monitor, (BASELINE_NAME, EXPERIMENT_NAME)))
    else:
        args.out.write_text(latex_table(rows, args.monitor, args.label))
        print(plain_text_table(rows, args.monitor))
    print(f"\n[table] wrote {args.out}")


if __name__ == "__main__":
    main()
