"""
The monitorability report: label the dumps, score each seed, aggregate across seeds, write the
figure and JSON. Metric definitions live in src/monitor/monitorability_metrics.py. With
share_inputs_across_datapoints every (seed, model, checkpoint) is scored on the inputs usable in
all of them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.utils import load_dump
from src.analysis.common import inputs_usable_in_every_rollout_set, resolve_steps, seed_dirs
from src.monitor.monitorability_metrics import (
    DROP_REASONS,
    HARNESS_KEY,
    LABEL_KEY,
    VALID_KEY,
    compute_mean_std_across_seeds,
    monitorability_counts,
    is_scorable_by_monitor,
    monitorability_metrics,
)
from src.monitor.monitor_utils import get_monitors_has_run

_STEP_RE = re.compile(r"step_(\d+)\.jsonl$")


def dump_steps(dump_dir: Path) -> dict[int, Path]:
    """
    Return {step: step_NNNNNN.jsonl} in ``dump_dir``.
    """
    out: dict[int, Path] = {}
    if not dump_dir.exists():
        return out
    for p in sorted(dump_dir.glob("step_*.jsonl")):
        if m := _STEP_RE.search(p.name):
            out[int(m.group(1))] = p
    return out


def compute_monitorability(
    control_dir: Path, misaligned_dir: Path,
    hack_label: str, monitor_name: str, shared_inputs: set[str] | None, steps: set[int] | None = None,
) -> dict[int, dict[str, float]]:
    """
    Compute {step: metrics} for ``monitor_name`` over the steps both runs have a dump for, scored on
    ``shared_inputs`` only (None = each step on its own).
    """
    from src.analysis.labeling import label_checkpoint
    from src.tasks.registry import get_hack_label

    assert hack_label is not None, "compute_monitorability needs a hack_label: Y is applied offline"
    label = get_hack_label(hack_label)
    ctrl_steps, mis_steps = dump_steps(control_dir), dump_steps(misaligned_dir)
    shared = sorted(set(ctrl_steps) & set(mis_steps) & (steps if steps is not None else set(ctrl_steps)))
    if not shared:
        print(f"[monitorability] no checkpoint steps in common.\n"
              f"  control {control_dir}: {sorted(ctrl_steps) or 'no dumps'}\n"
              f"  misaligned {misaligned_dir}: {sorted(mis_steps) or 'no dumps'}")
        return {}
    if unpaired := (sorted(set(ctrl_steps) ^ set(mis_steps))):
        print(f"[monitorability] skipping unpaired steps: {unpaired}")

    out: dict[int, dict[str, float]] = {}
    for step in shared:
        label_checkpoint(label, ctrl_steps[step], mis_steps[step], load_dump)
        control, misaligned = load_dump(ctrl_steps[step]), load_dump(mis_steps[step])
        # Checked in the misaligned dump too, in case one model's monitoring is partial.
        if monitor_name not in get_monitors_has_run(control) + get_monitors_has_run(misaligned):
            print(f"[monitorability] step {step}: no predictions from monitor {monitor_name!r} in the dumps "
                  f"(have: {get_monitors_has_run(control) or 'none'}) — skipping")
            continue
        m = monitorability_metrics(monitorability_counts(control, misaligned, monitor_name, shared_inputs))
        # A step with nothing scorable would plot as all-zero rates, like a blind monitor.
        if m["n_scored"] == 0:
            print(f"[monitorability] step {step}: no rollout scorable "
                  "(ungradable outputs or missing predictions) — skipping")
            continue
        out[step] = m
        why = ", ".join(f"{k}={int(m[k])}" for k in m if any(k.endswith(r) for r in DROP_REASONS) and m[k])
        note = (f"  [n={int(m['n_scored'])} rollouts"
                f"; {int(m['dropped'])} dropped" + (f" ({why})" if why else "") +
                f"; Y=1 on {int(m['control_positives'])} control"
                f" / {int(m['misaligned_positives'])} misaligned]")
        print(f"[monitorability] step {step:>6}  "
              f"iG²={m['intervention_g_mean_squared']:.3f} rG²={m['reward_intent_g_mean_squared']:.3f} "
              f"TPR={m['tpr']:.3f} TNR={m['tnr']:.3f}  "
              f"(TP={int(m['tp'])} FN={int(m['fn'])} TN={int(m['tn'])} FP={int(m['fp'])})"
              f"{note}")
    return out


# ---- across seeds ----

# Row 1 of the figure: each model's training curves.
RUN_PANELS = (("true_reward", "true_reward"), ("misaligned_reward", "misaligned_reward"),
              ("hack_rate", LABEL_KEY))


def compute_per_model_curves_across_steps(dump_dirs: dict[str, Path], shared_inputs: set[str] | None,
                                        steps: set[int] | None = None,
                                        ) -> dict[str, dict[int, dict[str, float]]]:
    """
    Compute {model_role: {step: {panel: mean}}}: each model's rewards and hack rate at every
    checkpoint over its gradable rows (restricted to ``shared_inputs`` when given).
    """
    out: dict[str, dict[int, dict[str, float]]] = {}
    for role, d in dump_dirs.items():
        out[role] = {}
        for step, path in sorted(dump_steps(d).items()):
            if steps is not None and step not in steps:
                continue
            all_rows = load_dump(path)
            rows = [r for r in all_rows if float(r[VALID_KEY]) and not float(r[HARNESS_KEY])
                    and (shared_inputs is None or r["query"] in shared_inputs)]
            if not rows:
                continue
            panels = {name: sum(float(r[key]) for r in rows) / len(rows)
                      for name, key in RUN_PANELS if all(key in r for r in rows)}
            panels["valid_output_rate"] = sum(float(r[VALID_KEY]) for r in all_rows) / len(all_rows)
            panels["n_scored"] = float(len(rows))      # what the curve points rest on
            out[role][step] = panels
    return out


def monitored_rollout_sets(seeds: dict[int, Path], dump_subdir: str, monitor_name: str, steps: set[int] | None = None):
    """
    Yield ((seed, model, step), rollouts) for every dump ``compute_monitorability`` will score,
    skipping exactly what it skips: unpaired steps and steps with no prediction from this monitor.
    """
    for seed, seed_dir in seeds.items():
        dump_path = {role: dump_steps(seed_dir / f"{role}_model" / dump_subdir) for role in ("control", "misaligned")}
        for step in sorted(set(dump_path["control"]) & set(dump_path["misaligned"])):
            if steps is not None and step not in steps:
                continue
            rollouts_of_model = {role: load_dump(paths[step]) for role, paths in dump_path.items()}
            if any(monitor_name in get_monitors_has_run(rollouts) for rollouts in rollouts_of_model.values()):
                for role, rollouts in rollouts_of_model.items():
                    yield (seed, role, step), rollouts


def compute_monitorability_per_seed_then_aggregate(
    experiment_dir: Path, hack_label: str, monitor_name: str, split: str = "test", seeds: list[int] | None = None,
    n_ckpts: int | str = "all", share_inputs_across_datapoints: bool = False,
) -> dict:
    """
    Score ``monitor_name`` on every seed's ``inference_<split>`` dumps, then aggregate mean ±
    std across seeds. See the module docstring for ``share_inputs_across_datapoints``.
    """
    found = seed_dirs(experiment_dir)
    if seeds is not None:
        missing = sorted(set(seeds) - set(found))
        if missing:
            raise FileNotFoundError(f"no seed dirs {missing} under {experiment_dir} (have {sorted(found)})")
        found = {k: v for k, v in found.items() if k in seeds}
    if not found:
        raise FileNotFoundError(f"no seed_<k>/ dirs with both role runs under {experiment_dir}")

    # Steps are picked as every analysis picks them (resolve_steps). Fewer checkpoints also
    # means fewer inputs dropped as unusable_elsewhere.
    steps = None if n_ckpts == "all" else set(resolve_steps(list(found.values()), n_ckpts, "monitorability", "scoring"))
    # Shared mode only: the inputs every (seed, model, step) can score.
    shared_inputs = inputs_usable_in_every_rollout_set(
        monitored_rollout_sets(found, f"inference_{split}", monitor_name, steps),
        lambda rollout: is_scorable_by_monitor(rollout, monitor_name), f"monitorability {monitor_name}"
    ) if share_inputs_across_datapoints else None
    return {"split": split, "n_ckpts": n_ckpts,
            "share_inputs_across_datapoints": share_inputs_across_datapoints,
            **score_every_seed_and_aggregate(
                found, hack_label, monitor_name, f"inference_{split}", shared_inputs, steps)}


def score_every_seed_and_aggregate(
    seeds: dict[int, Path], hack_label: str, monitor_name: str, dump_subdir: str,
    shared_inputs: set[str] | None, steps: set[int] | None = None,
) -> dict:
    """
    Label and score each seed's dumps and aggregate mean ± std across seeds. Returns {"monitor",
    "hack_label", "seeds", "runs", "monitorability", "run_curves"}.
    """
    per_seed: dict[int, dict] = {}
    runs: dict[int, dict] = {}
    for seed, d in seeds.items():
        print(f"[monitorability] === seed {seed}  monitor={monitor_name}")
        ctrl, mis = d / "control_model" / dump_subdir, d / "misaligned_model" / dump_subdir
        res = compute_monitorability(ctrl, mis, hack_label, monitor_name, shared_inputs, steps)
        if not res:
            # A dropped seed would pass the same-steps check and quote a mean over fewer seeds.
            raise SystemExit(f"[monitorability] seed {seed} produced no results for monitor "
                             f"{monitor_name}: no step is shared by both models with predictions from "
                             "this monitor. Resume the model with fewer steps / run run_inference on this "
                             "seed first, or restrict --seeds.")
        per_seed[seed] = res
        runs[seed] = compute_per_model_curves_across_steps({"control_model": ctrl, "misaligned_model": mis},
                                                         shared_inputs, steps)

    steps_per_seed = {seed: sorted(res) for seed, res in per_seed.items()}
    all_seeds_have_same_steps = all(steps == next(iter(steps_per_seed.values()))
                                    for steps in steps_per_seed.values())
    if not all_seeds_have_same_steps:
        raise SystemExit(f"[monitorability] seeds have different checkpoint steps: {steps_per_seed}. "
                         "Resume the shorter runs so all seeds end at the same step (or restrict --seeds).")
    return {"monitor": monitor_name, "hack_label": hack_label,
            "seeds": per_seed, "runs": runs,
            "monitorability": compute_mean_std_across_seeds(per_seed),
            "run_curves": compute_mean_std_across_seeds(runs)}


# ---- plotting ----

# Rows 2+ of the figure, one tuple per row; rates share a 0-1 axis.
MONITORABILITY_ROWS = (
    ("tnr", "tpr", "fnr", "fpr"),
)
# Intervention g-mean² (X = model) and its components
INTERVENTION_ROW = ("intervention_g_mean_squared", "intervention_tpr", "tnr_control", "tnr_misaligned_non_hack")
# Reward-intent g-mean² (rollouts labeled by the reward their policy was trained on, not by Y)
# and its components
REWARD_INTENT_ROW = ("reward_intent_g_mean_squared", "reward_intent_tpr", "reward_intent_tnr")


def _line(ax, xs, mean, std, label, n_seeds, linestyle="-", marker="o", alpha=1.0):
    """
    Draw a mean line with a ±std band when more than one seed contributed.
    """
    import seaborn as sns
    line = sns.lineplot(x=xs, y=mean, marker=marker, ax=ax, label=label,
                        linestyle=linestyle, alpha=alpha).lines[-1]
    if any(n > 1 for n in n_seeds):
        lo = [m - s for m, s in zip(mean, std)]
        hi = [m + s for m, s in zip(mean, std)]
        ax.fill_between(xs, lo, hi, color=line.get_color(), alpha=0.18, linewidth=0)


def plot_monitorability(result: dict, out_path: Path, title: str = "Monitorability across checkpoints") -> Path | None:
    """
    Draw the monitorability figure: each model's training curves, then one row per metric group.
    Lines are seed means, bands ±1 std.
    """
    if not result:
        print("[monitorability] nothing to plot")
        return None

    from src.analysis.common import setup_plotting
    plt, sns = setup_plotting()

    mon, runs = result["monitorability"], result["run_curves"]
    steps = sorted(int(s) for s in mon["mean"])
    n_seeds_total = len(result["seeds"])
    panel_rows = [list(r) for r in MONITORABILITY_ROWS]
    panel_rows.append(list(INTERVENTION_ROW))
    panel_rows.append(list(REWARD_INTENT_ROW))

    ncols = 4
    nrows = 1 + len(panel_rows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows), squeeze=False)

    def _style(ax, name, xs, rate=True, ylim=None):
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("checkpoint step", fontsize=8)
        ax.set_ylabel("")
        ax.set_xticks(xs)
        ax.set_xticklabels(["base_model" if int(x) == 0 else str(x) for x in xs])
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
        if ylim is not None:
            ax.set_ylim(*ylim)
        elif rate:
            ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)

    def _series(agg, k, k2s, metric):
        """
        Return xs, means, stds, n_seeds for metric over the inner keys k2s (steps) of agg[k].
        """
        xs = [k2 for k2 in k2s if k2 in agg["mean"].get(k, {}) and metric in agg["mean"][k][k2]]
        return (xs, [agg["mean"][k][x][metric] for x in xs], [agg["std"][k][x][metric] for x in xs],
                [agg["n_seeds"][k][x] for x in xs])

    # Row 1: the two models side by side.
    run_steps = sorted({int(st) for per in runs["mean"].values() for st in per})
    for i, (name, _key) in enumerate(RUN_PANELS):
        ax = axes[0][i]
        lo = 0.0
        for role in sorted(runs["mean"]):
            xs, m, sd, n = _series(runs, role, run_steps, name)
            if xs:
                _line(ax, xs, m, sd, role, n)
                lo = min(lo, min(v - e for v, e in zip(m, sd)))
        # A signed reward pair runs to -1; fit the axis to what the run produced.
        _style(ax, name, run_steps, ylim=(-1.05, 1.05) if lo < -0.05 else (-0.05, 1.05))

    axes[0][len(RUN_PANELS)].axis("off")

    for r, row in enumerate(panel_rows):
        for c in range(ncols):
            ax = axes[1 + r][c]
            if c >= len(row):
                ax.axis("off")
                continue
            metric = row[c]
            xs, m, sd, n = _series_mon(mon, steps, metric)
            if xs:
                _line(ax, xs, m, sd, result["monitor"], n)
            # Rates share a 0-1 axis; supports are counts.
            _style(ax, metric, steps, rate=metric not in ("n_positives", "n_negatives"))

    seeds_note = (f"{n_seeds_total} seed{'s' if n_seeds_total != 1 else ''}"
                  + (" · band = ±1 std" if n_seeds_total > 1 else ""))
    fig.suptitle(f"{title}\nmonitor = {result['monitor']} · {seeds_note}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[monitorability] saved {out_path}  ({len(steps)} steps, {seeds_note})")
    return out_path


def _series_mon(mon: dict, steps: list[int], metric: str):
    """
    Return xs, means, stds, n_seeds of ``metric`` across steps.
    """
    xs = [s for s in steps if metric in mon["mean"].get(s, {})]
    return (xs, [mon["mean"][x][metric] for x in xs], [mon["std"][x][metric] for x in xs],
            [mon["n_seeds"][x] for x in xs])


def _str_keys(obj):
    """
    Convert the int step / seed keys to strings, since JSON needs string keys.
    """
    if isinstance(obj, dict):
        return {str(k): _str_keys(v) for k, v in obj.items()}
    return obj


def monitorability_report(
    experiment_dir: Path, *, hack_label: str, monitor_name: str, tag: str | None = "test",
    seeds: list[int] | None = None, title: str | None = None, n_ckpts: int | str = "all",
    share_inputs_across_datapoints: bool = False,
) -> dict:
    """
    Label and score every seed, then write ``<experiment_dir>/<monitor_name>/monitorability_<tag>.json``
    and the matching ``_across_ckpts.pdf`` (suffixed with the checkpoint count and input sharing when
    those are not the defaults).
    """
    result = compute_monitorability_per_seed_then_aggregate(
        experiment_dir, hack_label, monitor_name, split=tag or "test", seeds=seeds, n_ckpts=n_ckpts,
        share_inputs_across_datapoints=share_inputs_across_datapoints)
    if not result:
        return result
    out_dir = experiment_dir / monitor_name
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ((f"_{tag}" if tag else "")
              + ("" if n_ckpts == "all" else f"_{len(result['monitorability']['mean'])}ckpts")
              + ("_shared_inputs" if share_inputs_across_datapoints else ""))
    json_path = out_dir / f"monitorability{suffix}.json"
    json_path.write_text(json.dumps(_str_keys(result), indent=2))
    print(f"[monitorability] wrote {json_path}")
    plot_monitorability(result, out_dir / f"monitorability{suffix}_across_ckpts.pdf",
                        title or f"Monitorability across checkpoints\n{experiment_dir}")
    return result
