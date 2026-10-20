"""
Scoring monitorability (g-mean², TPR, TNR)  for one experiment: score monitor predictions
against hacking label Y per checkpoint, averaging over both models, calculating mean and std over seeds.

    python src/analysis/compute_monitorability.py outputs/<dataset_name>/<model>/8192 [--monitor_model_name ...] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import load_dump
from src.analysis.common import DEFAULT_N_CKPTS, add_share_inputs_argument
from src.analysis.plot_monitorability import dump_steps, monitorability_report, seed_dirs
from src.monitor.monitor_utils import to_monitor_name, get_monitors_has_run
from src.tasks.base import MODEL_ROLES

SPLIT = "test"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment_dir", type=Path, help="The dir holding seed_<k>/ (dataset/model/max_tokens).")
    p.add_argument("--monitor_model_name", nargs="*", default=None, metavar="MODEL",
                   help="Monitor model name(s) or to_monitor_name(s) to score "
                        "(default: every monitor found in the dumps).")
    p.add_argument("--seeds", nargs="*", type=int, default=None, help="Seeds to include (default: all found).")
    p.add_argument("--n_ckpts", default=DEFAULT_N_CKPTS, metavar="K",
                   type=lambda value: value if value == "all" else int(value),
                   help="Score K usable checkpoints, chosen exactly as the analysis scripts choose them "
                        "(resolve_steps/spread_steps: step 0 = the base model, the last step both models reached, and "
                        "K-2 more at one fixed interval walking back). Fewer checkpoints also means fewer rollout "
                        "sets an input must be usable in, so far fewer inputs drop out as unusable_elsewhere. "
                        f"'all' = every usable step (writes the unsuffixed files). Default: {DEFAULT_N_CKPTS}.")
    add_share_inputs_argument(p)
    return p.parse_args()


# Runs averaged together (seeds, both models) must have the same value for each of these config keys.
COMPARABILITY_KEYS = (
    "base_model", "think_start_str", "think_end_str", "max_tokens",
    "eval_temperature", "eval_top_p", "eval_top_k",
    "intent_in_input",
    "monitor_backend_type", "monitor_model_name", "monitor_max_tokens",
    "monitor_think_start_str", "monitor_think_end_str",
    "true_reward", "misaligned_reward", "hack_label", "test_file",
)
REQUIRED_KEYS = ("hack_label", "test_file")   # must also be set, not null


def _shared_config_value(configs: dict[str, object], key: str, required: bool):
    """
    Return the value every run's config has for ``key``. Exits if the runs disagree, or if the key is
    required and unset.
    """
    from omegaconf import ListConfig, OmegaConf
    values = {}
    for run, cfg in configs.items():
        v = cfg.get(key)
        values[run] = OmegaConf.to_container(v) if isinstance(v, ListConfig) else v
    if len({json.dumps(v, sort_keys=True) for v in values.values()}) != 1:
        raise SystemExit(f"Runs disagree on {key}: {values}")
    if required and None in values.values():
        raise SystemExit(f"{key} is unset in the runs' config.yaml: {values}")
    return next(iter(values.values()))


def check_runs_are_comparable(runs: list[Path]) -> dict:
    """
    Check if results across the runs could be averaged: yes if every run's config.yaml has the
    same value for each key in COMPARABILITY_KEYS
    Returns shared values as {key: value}.
    """
    from omegaconf import OmegaConf
    configs = {str(run): OmegaConf.load(run / "config.yaml") for run in runs}
    return {key: _shared_config_value(configs, key, required=key in REQUIRED_KEYS) for key in COMPARABILITY_KEYS}


def _get_monitors_has_run(runs: list[Path]) -> list[str]:
    """
    Return the names of the monitors with predictions stored in the dumps.
    """
    out: set[str] = set()
    for run in runs:
        for path in dump_steps(run / f"inference_{SPLIT}").values():
            out |= set(get_monitors_has_run(load_dump(path)))
    return sorted(out)


def _print_final_breakdown(result: dict) -> None:
    """
    Print per seed how the last checkpoint's rollouts were scored or dropped
    """
    from src.monitor.monitorability_metrics import DROP_REASONS
    for seed, per_step in result["seeds"].items():
        step = max(per_step)
        m = per_step[step]
        n_scored, n_drop = int(m["n_scored"]), int(m["dropped"])
        print(f"\n[monitorability] monitor={result['monitor']} seed={seed} final checkpoint (step {step}): "
              f"{n_scored + n_drop} rollouts over both runs")
        print(f"  scored .......................... {n_scored:4d}   "
              f"-> {int(m['n_positives'])} positives (TP={int(m['tp'])} FN={int(m['fn'])}), "
              f"{int(m['n_negatives'])} negatives (TN={int(m['tn'])} FP={int(m['fp'])})")
        print(f"      Y=1 from control ......... {int(m['control_positives']):4d}")
        print(f"      Y=1 from misaligned ......... {int(m['misaligned_positives']):4d}")
        print(f"  dropped ......................... {n_drop:4d}")
        for side in ("control", "misaligned"):
            for r in DROP_REASONS:
                k = f"{side}_{r}"
                if m.get(k):
                    print(f"      {side:<11} {r:<15} {int(m[k]):4d}")


def main() -> None:
    args = _parse_args()
    exp = args.experiment_dir
    seeds = seed_dirs(exp)
    if not seeds:
        raise SystemExit(f"No seed_<k>/ dirs with both model runs under {exp}")
    if args.seeds is not None:
        if missing := sorted(set(args.seeds) - set(seeds)):
            raise SystemExit(f"No seed dirs {missing} under {exp} (have {sorted(seeds)}).")
        seeds = {k: v for k, v in seeds.items() if k in args.seeds}
    runs = [d / role for d in seeds.values() for role in MODEL_ROLES]

    shared = check_runs_are_comparable(runs)
    hack_label = shared["hack_label"]

    monitors_has_run = _get_monitors_has_run(runs)
    if args.monitor_model_name:
        monitors_to_run = [to_monitor_name(m) for m in args.monitor_model_name]
        if unknown := sorted(set(monitors_to_run) - set(monitors_has_run)):
            raise SystemExit(f"No predictions from monitor(s) {unknown} in the dumps under {exp}. "
                             f"Monitors with predictions: {monitors_has_run or 'none'}. "
                             "Add one with src/analysis/run_inference.py.")
    else:
        monitors_to_run = monitors_has_run
    print(f"[monitorability] experiment {exp}: seeds {sorted(seeds)}, monitors {monitors_to_run}, "
          f"hack_label={hack_label}")

    for monitor_name in monitors_to_run:
        result = monitorability_report(exp, hack_label=hack_label, monitor_name=monitor_name, tag=SPLIT,
                                       seeds=sorted(seeds), n_ckpts=args.n_ckpts,
                                       share_inputs_across_datapoints=args.share_inputs_across_datapoints)
        if result:
            _print_final_breakdown(result)


if __name__ == "__main__":
    main()
