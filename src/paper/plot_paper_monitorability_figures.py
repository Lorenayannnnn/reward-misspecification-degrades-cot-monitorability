"""
The paper's monitorability figures, from the reports compute_monitorability.py wrote.

Per task: hack rate per model, then each monitor's intervention and reward-intent g-mean² over
checkpoints (color = policy, line style = model, bands = ±1 std over seeds). --out merges the tasks
into one figure.

    python src/paper/plot_paper_monitorability_figures.py --monitor gpt-5.6-luna gpt-4.1-mini \\
        --training_name Baseline outputs/baseline_training/length/<dataset>_explicit/openai/gpt-oss-20b/8192 ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.analysis.common import BASE_STEP, setup_plotting

# Column and figure order; a dir whose task is not listed raises.
TASK_ORDER = ("length", "judge", "coding")
TASK_TITLES = {"length": "Length", "judge": "LM Judge", "coding": "Coding"}

# Directory name -> legend label; an unlisted policy raises.
POLICY_LABELS = {
    "gpt-oss-20b": "gpt-oss-20b",
    "gpt-oss-120b": "gpt-oss-120b",
    "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "Nemotron-3-Nano",
}

# Model role -> line style; solid is the misaligned model.
ARM_STYLES = {"misaligned_model": "-", "control_model": "--"}
# r_m is the misspecified reward, r_t the true reward.
ARM_LABELS = {"misaligned_model": "Misaligned ($r_m$)", "control_model": "Control ($r_t$)"}

# NaN until the misaligned model out-hacks the control one; decides where a figure starts.
HEADLINE_FACTOR = "intervention_tpr"

# Checkpoints per figure, excluding the base model.
DEFAULT_N_CKPTS_WITHOUT_BASE = 7


def read_report(experiment_dir: Path, monitor: str) -> dict:
    """
    Read the stored report of one (experiment, monitor); raises when it is missing.
    """
    report_path = experiment_dir / monitor / "monitorability_test.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"{report_path} missing — run src/analysis/compute_monitorability.py on {experiment_dir} "
            f"(with --n_ckpts all, which writes the unsuffixed report) first."
        )
    return json.loads(report_path.read_text())


def task_and_policy(experiment_dir: Path) -> tuple[str, str]:
    """
    Read (task, policy) off the path `outputs/<subdir>/<task>/<dataset>/<org>/<policy>/<max_tokens>`.
    """
    task, policy = experiment_dir.parts[-5], experiment_dir.parts[-2]
    if task not in TASK_ORDER:
        raise ValueError(f"{experiment_dir}: task {task!r} is not one of {TASK_ORDER}")
    if policy not in POLICY_LABELS:
        raise ValueError(f"{experiment_dir}: policy {policy!r} is not one of {sorted(POLICY_LABELS)}")
    return task, policy


def series_over_steps(per_step: dict, steps: list[int], metric: str) -> tuple[list[int], list[float], list[float]]:
    """
    Return (steps, mean, std) of one metric, over the steps where the report has it.
    """
    mean, std = per_step["mean"], per_step["std"]
    xs = [s for s in steps if metric in mean[str(s)]]
    return xs, [mean[str(s)][metric] for s in xs], [std[str(s)][metric] for s in xs]


def steps_defined_in_every_report(reports: list[dict]) -> list[int]:
    """
    Return the trained checkpoints whose intervention TPR is defined in every report of a figure
    (it is NaN wherever the misaligned model does not out-hack the control one).
    """
    defined_per_report = [{int(s) for s, metrics in report["monitorability"]["mean"].items()
                           if int(s) != BASE_STEP and not math.isnan(metrics[HEADLINE_FACTOR])}
                          for report in reports]
    defined_everywhere = sorted(set.intersection(*defined_per_report))
    if not defined_everywhere:
        raise SystemExit(f"no checkpoint has a defined {HEADLINE_FACTOR} in every run of the figure: "
                         "the two models never separated in every seed at the same step.")
    return defined_everywhere


def steps_to_plot(candidates: list[int], n_ckpts: int) -> list[int]:
    """
    Pick ``n_ckpts`` of ``candidates``, always the first and the last, the rest spread evenly
    (all of them when there are n_ckpts or fewer).
    """
    if len(candidates) <= n_ckpts:
        return candidates
    first, last = candidates[0], candidates[-1]
    picked: list[int] = []
    for i in range(n_ckpts):
        target = first + i * (last - first) / (n_ckpts - 1)
        remaining = [s for s in candidates if s not in picked]
        picked.append(min(remaining, key=lambda step: (abs(step - target), step)))
    return sorted(picked)


def plot_line(ax, xs, mean, std, color, style, linewidth: float = 2.6):
    """
    Draw one seed-mean line, a dot per datapoint, and its ±1 std band.
    """
    ax.plot(xs, mean, color=color, linestyle=style, linewidth=linewidth, marker="o", markersize=5)
    ax.fill_between(xs, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],
                    color=color, alpha=0.15, linewidth=0)


def plotted_steps_of(reports: list[dict], n_ckpts: int) -> list[int]:
    """
    Return the checkpoints every panel of a figure shares: ``n_ckpts`` of the steps defined in every report.
    """
    return steps_to_plot(steps_defined_in_every_report(reports), n_ckpts)


def plot_hack_rate(ax, report: dict, color, steps: list[int]) -> None:
    """
    Plot both models' hack rate over the plotted checkpoints, for one policy.
    """
    runs = report["run_curves"]
    for role, style in ARM_STYLES.items():
        xs, mean, std = series_over_steps({k: runs[k][role] for k in ("mean", "std")}, steps, "hack_rate")
        plot_line(ax, xs, mean, std, color, style)


def plot_intervention_g_mean_squared(ax, report: dict, color, steps: list[int]) -> None:
    """
    Plot one policy's intervention g-mean² over checkpoints. A gap is a checkpoint where the two models'
    hack rates did not differ (e.g. step 0).
    """
    xs, mean, std = series_over_steps(report["monitorability"], steps, "intervention_g_mean_squared")
    plot_line(ax, xs, mean, std, color, ARM_STYLES["misaligned_model"])


def plot_reward_intent_g_mean_squared(ax, report: dict, color, steps: list[int]) -> None:
    """
    Plot one policy's reward-intent g-mean² over checkpoints (rollouts labeled by model role, not by Y).
    """
    xs, mean, std = series_over_steps(report["monitorability"], steps, "reward_intent_g_mean_squared")
    plot_line(ax, xs, mean, std, color, ARM_STYLES["misaligned_model"])


def style_axis(ax, title: str, ylabel: str, steps: list[int]) -> None:
    """
    Set the title, the 0-1 y axis and one x tick per plotted checkpoint. Font sizes suit a column-wide figure.
    """
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("checkpoint step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])     # short panels would otherwise label only 0 / 0.5 / 1
    ax.set_xticks(steps)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)


def figure_legends(fig, colors: dict, policies: list[str], n_columns: int) -> None:
    """
    Draw one legend row under the figure: color = policy, line style = model.
    """
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=colors[p], linewidth=1.8, label=POLICY_LABELS[p]) for p in policies]
    handles += [Line2D([], [], color="black", linestyle=ARM_STYLES[a], linewidth=1.8, label=ARM_LABELS[a])
                for a in ARM_STYLES]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=n_columns,
               frameon=False, fontsize=12)


def plot_task_figure(task: str, reports: dict[str, dict[str, dict]], monitors: list[str], out_path: Path,
                     n_ckpts: int = DEFAULT_N_CKPTS_WITHOUT_BASE) -> Path:
    """
    Plot one task: hack rate, then both g-mean² panels per monitor. ``reports`` is {monitor: {policy: report}}.
    """
    plt, sns = setup_plotting()
    policies = [p for p in POLICY_LABELS if p in reports[monitors[0]]]
    colors = dict(zip(policies, sns.color_palette("colorblind", len(policies)), strict=True))

    n_columns = 1 + 2 * len(monitors)                       # hack rate, then both g-mean² per monitor
    fig, axes = plt.subplots(1, n_columns, figsize=(4.4 * n_columns, 3.6), squeeze=False)
    steps = plotted_steps_of([reports[m][p] for m in monitors for p in policies], n_ckpts)
    for policy in policies:
        plot_hack_rate(axes[0][0], reports[monitors[0]][policy], colors[policy], steps)
        for i, monitor in enumerate(monitors):
            plot_intervention_g_mean_squared(axes[0][1 + i], reports[monitor][policy], colors[policy], steps)
            plot_reward_intent_g_mean_squared(axes[0][1 + len(monitors) + i], reports[monitor][policy],
                                              colors[policy], steps)
    style_axis(axes[0][0], "Hack Rate", "Hack Rate", steps)
    for i, monitor in enumerate(monitors):
        style_axis(axes[0][1 + i], f"Monitorability · {monitor}", "Intervention g-mean²", steps)
        style_axis(axes[0][1 + len(monitors) + i], f"Monitorability · {monitor}", "Reward-intent g-mean²", steps)

    figure_legends(fig, colors, policies, n_columns=len(policies) + 2)
    fig.suptitle(f"{TASK_TITLES[task]}: the two models separate, the chain of thought does not follow", fontsize=15)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper figure] wrote {out_path}")
    return out_path


# The three terms of the intervention g-mean², one panel each: (metric, title, denominator cells).
INTERVENTION_PANELS = (
    ("intervention_tpr", "Intervention TPR", ("misaligned_tp", "misaligned_fn")),
    ("tnr_control", "TNR$_{X=0}$", ("control_tp", "control_fn", "control_tn", "control_fp")),
    ("tnr_misaligned_non_hack", "TNR$_{X=1}$", ("misaligned_tn", "misaligned_fp")),
)


# The two terms of the reward-intent g-mean², each over all of one model's rollouts.
REWARD_INTENT_PANELS = (
    ("reward_intent_tpr", "TPR$_{Z=1|X=1}$", ("misaligned_tp", "misaligned_fn", "misaligned_tn", "misaligned_fp")),
    ("reward_intent_tnr", "TNR$_{Z=0|X=0}$", ("control_tp", "control_fn", "control_tn", "control_fp")),
)


def plot_task_g_mean_breakdown(task: str, reports: dict[str, dict[str, dict]], monitors: list[str],
                               out_path: Path, training_name: str,
                               n_ckpts: int = DEFAULT_N_CKPTS_WITHOUT_BASE) -> Path:
    """
    Plot one task: the main figure's row, then per monitor a row of its g-mean² terms (two monitors
    give a 3 x 5 grid); shorter rows are centered. ``training_name`` names the condition in the title.
    """
    plt, sns = setup_plotting()
    policies = [p for p in POLICY_LABELS if p in reports[monitors[0]]]
    colors = dict(zip(policies, sns.color_palette("colorblind", len(policies)), strict=True))
    # Per panel: (metric, title, denominator cells, g-mean² form).
    term_panels = ([(*panel, "Intervention") for panel in INTERVENTION_PANELS]
                   + [(*panel, "Reward-intent") for panel in REWARD_INTENT_PANELS])
    term_rows = [(monitor, term_panels) for monitor in monitors]
    n_top = 1 + 2 * len(monitors)
    n_columns = max(n_top, len(term_panels))

    # Half-panel GridSpec units, so a shorter row can be centered.
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(4.4 * n_columns, 3.5 * (1 + len(term_rows))))
    grid = GridSpec(1 + len(term_rows), 2 * n_columns, figure=fig)
    def centered_row(grid_row: int, n_panels: int) -> list:
        offset = n_columns - n_panels
        return [fig.add_subplot(grid[grid_row, offset + 2 * i: offset + 2 * (i + 1)]) for i in range(n_panels)]
    axes = [centered_row(0, n_top)] + [centered_row(row, len(panels))
                                      for row, (_m, panels) in enumerate(term_rows, start=1)]
    steps = plotted_steps_of([reports[m][p] for m in monitors for p in policies], n_ckpts)
    for policy in policies:
        plot_hack_rate(axes[0][0], reports[monitors[0]][policy], colors[policy], steps)
        for i, monitor in enumerate(monitors):
            plot_intervention_g_mean_squared(axes[0][1 + i], reports[monitor][policy], colors[policy], steps)
            plot_reward_intent_g_mean_squared(axes[0][1 + len(monitors) + i], reports[monitor][policy],
                                              colors[policy], steps)
        for row, (monitor, panels) in enumerate(term_rows, start=1):
            per_step = reports[monitor][policy]["monitorability"]
            for col, (metric, _title, denominator_cells, _form) in enumerate(panels):
                xs, mean, std = series_over_steps(per_step, steps, metric)
                plot_line(axes[row][col], xs, mean, std, colors[policy], ARM_STYLES["misaligned_model"], linewidth=3.4)

    style_axis(axes[0][0], "Hack Rate", "Hack Rate", steps)
    for i, monitor in enumerate(monitors):
        style_axis(axes[0][1 + i], f"Monitorability · {monitor}", "Intervention g-mean²", steps)
        style_axis(axes[0][1 + len(monitors) + i], f"Monitorability · {monitor}", "Reward-intent g-mean²", steps)
    for row, (monitor, panels) in enumerate(term_rows, start=1):
        for col, (_metric, title, _cells, form) in enumerate(panels):
            style_axis(axes[row][col], f"{title}\n{form} · {monitor}", "Rate" if col == 0 else "", steps)

    for ax in [panel for row_axes in axes for panel in row_axes]:   # larger fonts and thicker lines
        ax.title.set_fontsize(17)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        ax.tick_params(axis="both", labelsize=13)
        for line in ax.lines:
            line.set_linewidth(max(line.get_linewidth(), 3.4))
    figure_legends(fig, colors, policies, n_columns=len(policies) + 2)
    fig.suptitle(f"{TASK_TITLES[task]}: Intervention and Reward-intent g-mean² under {training_name} Training",
                 fontsize=26)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper figure] wrote {out_path}")
    return out_path


def plot_all_tasks_figure(reports: dict[str, dict[str, dict[str, dict]]], monitors: list[str], out_path: Path,
                          training_name: str, n_ckpts: int = DEFAULT_N_CKPTS_WITHOUT_BASE) -> Path:
    """
    Plot every task in one grid: a column per task, a hack-rate row, then a row per monitor and g-mean²
    form. ``reports`` is {task: {monitor: {policy: report}}}.
    """
    plt, sns = setup_plotting()
    tasks = [t for t in TASK_ORDER if t in reports]
    policies = [p for p in POLICY_LABELS if any(p in per_monitor[monitors[0]]
                                                for per_monitor in reports.values())]
    colors = dict(zip(policies, sns.color_palette("colorblind", len(policies)), strict=True))

    n_rows = 1 + 2 * len(monitors)                  # hack rate, a row per monitor per g-mean² form
    fig, axes = plt.subplots(n_rows, len(tasks), figsize=(4.2 * len(tasks), 3.4 * n_rows),
                             squeeze=False, sharey="row")
    for col, task in enumerate(tasks):
        steps = plotted_steps_of([reports[task][m][p] for m in monitors for p in reports[task][m]], n_ckpts)
        for policy in policies:
            plot_hack_rate(axes[0][col], reports[task][monitors[0]][policy], colors[policy], steps)
            for i, monitor in enumerate(monitors):
                plot_intervention_g_mean_squared(axes[1 + i][col], reports[task][monitor][policy],
                                                 colors[policy], steps)
                plot_reward_intent_g_mean_squared(axes[1 + len(monitors) + i][col], reports[task][monitor][policy],
                                                  colors[policy], steps)
        style_axis(axes[0][col], TASK_TITLES[task], "Hack Rate" if col == 0 else "", steps)
        for i, monitor in enumerate(monitors):
            style_axis(axes[1 + i][col], "", f"Intervention g-mean²\n{monitor}" if col == 0 else "", steps)
            style_axis(axes[1 + len(monitors) + i][col], "",
                       f"Reward-intent g-mean²\n{monitor}" if col == 0 else "", steps)

    for ax in axes.flat:                           # larger fonts and thicker lines
        ax.title.set_fontsize(17)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        ax.tick_params(axis="both", labelsize=13)
        for line in ax.lines:
            line.set_linewidth(max(line.get_linewidth(), 3.4))
    figure_legends(fig, colors, policies, n_columns=len(policies) + 2)
    # rect leaves room at the bottom for the legend and at the top for the suptitle.
    fig.suptitle(f"Hack Rate and CoT Monitorability under {training_name} Training", fontsize=19, y=0.995)
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper figure] wrote {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment_dirs", type=Path, nargs="+", help="Experiment dirs (dataset/org/policy/max_tokens).")
    p.add_argument("--monitor", nargs="+", required=True,
                   help="Monitor names, i.e. the report subdirs (e.g. gpt-5.6-luna gpt-4.1-mini). The per-task "
                        "figure gets one monitorability panel each, the merged --out figure one row each.")
    p.add_argument("--n_ckpts", type=int, default=DEFAULT_N_CKPTS_WITHOUT_BASE,
                   help=f"Checkpoints per figure, evenly spread and EXCLUDING the base model "
                        f"(default: {DEFAULT_N_CKPTS_WITHOUT_BASE}).")
    p.add_argument("--out", type=Path, default=None,
                   help="Write ONE figure over every task here (hack rate + a row per monitor). Default: "
                        "one figure per task under --out_dir.")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Where the per-task figures go (<task>_main.pdf, <task>_g_mean_breakdown.pdf). "
                        "Default: outputs/<subdir>/<task>/figures/ next to the experiment dirs.")
    p.add_argument("--training_name", required=True,
                   help="Condition named in the figure titles, e.g. Baseline, Description, 'Underspecified Intent'.")
    args = p.parse_args()

    # {task: {monitor: {policy: report}}} and the dir each task's figures go under.
    reports: dict[str, dict[str, dict[str, dict]]] = {}
    task_dirs: dict[str, Path] = {}
    for experiment_dir in args.experiment_dirs:
        task, policy = task_and_policy(experiment_dir)
        task_dirs[task] = experiment_dir.parents[3]
        for monitor in args.monitor:
            per_policy = reports.setdefault(task, {}).setdefault(monitor, {})
            if policy in per_policy:
                raise ValueError(f"two experiment dirs for {task}/{policy}; pass one per (task, policy)")
            per_policy[policy] = read_report(experiment_dir, monitor)

    if args.out:
        plot_all_tasks_figure(reports, args.monitor, args.out, args.training_name, args.n_ckpts)
        return
    for task in TASK_ORDER:
        if task in reports:
            out_dir = args.out_dir or task_dirs[task] / "figures"
            plot_task_figure(task, reports[task], args.monitor, out_dir / f"{task}_main.pdf", args.n_ckpts)
            plot_task_g_mean_breakdown(task, reports[task], args.monitor,
                                       out_dir / f"{task}_g_mean_breakdown.pdf", args.training_name, args.n_ckpts)


if __name__ == "__main__":
    main()
