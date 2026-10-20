"""
Run an additional CoT monitor over existing evaluation dumps. Predictions are written back under mon_cot__<monitor name>_*.

python src/analysis/run_inference.py outputs/<ds>/<model>/8192 --monitor_model gpt-4.1-mini --monitor_backend openai
python src/analysis/run_inference.py <seed_dir> --monitor_model openai/gpt-oss-20b --monitor_backend tinker
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import src.tasks  # noqa: F401  registers the tasks, rewards and hack labels
from src.utils import load_dump, write_dump
from src.analysis.plot_monitorability import dump_steps, seed_dirs
from src.monitor.backends import build_monitor_backend
from src.monitor.monitor_utils import (
    get_monitor_key_name,
    run_monitor_and_store_predictions,
    to_monitor_name,
)
from src.tasks.base import MODEL_ROLES as ROLES


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target_dir", type=Path, help="Experiment dir (holding seed_<k>/) or one seed dir.")
    p.add_argument("--monitor_model", required=True,
                   help="Monitor model name, e.g. openai/gpt-oss-120b or gpt-4.1-mini.")
    p.add_argument("--monitor_backend", required=True, choices=("openai", "tinker"))
    p.add_argument("--split", default="test")
    p.add_argument("--seeds", nargs="*", type=int, default=None, help="Seeds to include (default: all found).")
    p.add_argument("--steps", nargs="*", type=int, default=None, help="Checkpoint steps to monitor (default: all).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run on records that already have a real prediction from this monitor "
                        "(failed calls are always redone).")
    p.add_argument("--monitor_max_tokens", type=int, default=4096)
    p.add_argument("--monitor_think_start", default=None, help="tinker only; monitor's CoT start marker.")
    p.add_argument("--monitor_think_end", default=None, help="tinker only; monitor's CoT end marker.")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed for the monitor.")
    p.add_argument("--max_workers", type=int, default=None,
                   help="Concurrent monitor calls (default: MONITOR_MAX_WORKERS or 4).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    monitor_name = to_monitor_name(args.monitor_model)

    if (args.target_dir / ROLES[0]).is_dir():
        seeds = {-1: args.target_dir}
    else:
        seeds = seed_dirs(args.target_dir)
        if args.seeds is not None:
            seeds = {k: v for k, v in seeds.items() if k in args.seeds}
    if not seeds:
        raise SystemExit(f"No seed_<k>/ dirs (or role dirs) under {args.target_dir}")

    backend = build_monitor_backend({
        "monitor_backend_type": args.monitor_backend, "monitor_model_name": args.monitor_model,
        "seed": args.seed, "max_tokens": args.monitor_max_tokens,
        "think_start": args.monitor_think_start, "think_end": args.monitor_think_end,
    })
    print(f"[run_inference] monitor={args.monitor_model} ({args.monitor_backend}) monitor_name={monitor_name} "
          f"seeds={sorted(seeds)}")

    for seed, seed_dir in sorted(seeds.items()):
        for role in ROLES:
            for step, path in sorted(dump_steps(seed_dir / role / f"inference_{args.split}").items()):
                if args.steps is not None and step not in args.steps:
                    continue
                tag = f"seed {seed if seed >= 0 else '-'} {role} step {step}"
                _monitor_dump(path, args, backend, monitor_name, tag)


def _monitor_dump(path: Path, args, backend, monitor_name: str, tag: str) -> None:
    """
    Run the monitor over one dump file and write the predictions back.
    """
    rows = load_dump(path)
    key, failed = get_monitor_key_name(monitor_name, "verdict"), get_monitor_key_name(monitor_name, "failed")
    # Failed calls are always redone; real predictions are kept unless --overwrite.
    records_to_run = [i for i, r in enumerate(rows) if args.overwrite or key not in r or r.get(failed)]
    if not records_to_run:
        print(f"[run_inference] {tag}: already monitored by {monitor_name} — skip")
        return
    summary = run_monitor_and_store_predictions(rows, records_to_run, backend, monitor_name, args.max_workers)
    write_dump(path, rows)
    print(f"[run_inference] {tag}:  {summary}")


if __name__ == "__main__":
    main()
