"""
Base-model prefilling (Section 5.2): sample the untrained base model's output after the misaligned
model's CoT and measure its hack rate (the prefill hack rate).

    Do all ckpts:
        python src/analysis/prefill_base_model.py outputs/<ds>/<model>/8192
    Do the final ckpt:
        python src/analysis/prefill_base_model.py <experiment_dir> --n_ckpts 1 --seeds 0 1
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import src.tasks  # noqa: F401  registers the tasks, rewards and hack labels
from src.analysis.common import (
    add_experiment_arguments,
    add_n_ckpts_argument,
    cot_open_ids,
    encode,
    load_model_config,
    compute_metrics_over_shared_inputs,
    resolve_steps,
    sample_prefilled_batch,
    select_seed_dirs,
    compute_panel_metrics,
    write_analysis_dump,
    test_records,
    write_reports_at_both_levels,
)
from src.tasks.base import MISALIGNED_MODEL
from src.utils import load_dump
from src.tasks.registry import get_hack_label
from src.analysis.labeling import has_valid_out_and_no_harness_error, add_hacking_labels
from src.training.context import setup_renderer

TAG = "prefill"
DUMP_DIR = "prefill"
ROLE = MISALIGNED_MODEL   # whose CoTs the base model continues


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_experiment_arguments(p)
    add_n_ckpts_argument(p, "prefill")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-sample dumps that already exist (default: keep them, so a crashed sweep resumes).")
    p.add_argument("--verifier_setting", action="append", default=None, metavar="KEY=VALUE",
                   help="Override a stored verifier_settings entry (e.g. a stale sandbox URL).")
    return p.parse_args()


# ---- sampling ----

async def sample_base_model_on_cots(ctx, seed_dir: Path, step: int) -> dict:
    """
    Sample the base model's output after each of the misaligned model's CoTs, label the outputs with the
    hack label, dump them under the misaligned model's dir and return the metrics.
    """
    cfg = ctx["cfg"]
    records = ctx["records"]
    open_ids = cot_open_ids(ctx["tokenizer"], ctx["renderer"], cfg.think_start_str)
    close_ids = encode(ctx["tokenizer"], cfg.think_end_str)
    prefills = [open_ids + encode(ctx["tokenizer"], r.get("cot")) + close_ids for r in records]
    rows = await sample_prefilled_batch(
        ctx["base_client"], records, prefills, ctx["tokenizer"], ctx["renderer"], cfg,
        tag=TAG, label="base_model", extras=[{"cot_source": "base_model"}] * len(records))
    add_hacking_labels(rows, get_hack_label(cfg.hack_label))
    write_analysis_dump(seed_dir, ROLE, DUMP_DIR, f"step_{step:06d}.jsonl", rows)
    return compute_panel_metrics(rows)


async def sample_seed(args: argparse.Namespace, seed_dir: Path, steps: list[int]) -> None:
    """
    Sample and dump the base model's output after the misaligned model's CoTs at every step under ``seed_dir``.
    """
    import tinker

    cfg = load_model_config(seed_dir / ROLE, args.verifier_setting)
    tokenizer, renderer = setup_renderer(cfg.base_model)
    base_client = tinker.ServiceClient().create_sampling_client(base_model=cfg.base_model)

    for step in steps:
        print(f"\n[{TAG}] ===== step {step} ({steps.index(step) + 1}/{len(steps)}) =====", flush=True)
        # A dump on disk is complete (written atomically), so it is skipped unless --overwrite.
        if (seed_dir / ROLE / DUMP_DIR / f"step_{step:06d}.jsonl").exists() and not args.overwrite:
            print(f"[{TAG}] step {step}: dump exists, skipping (--overwrite to re-sample)")
            continue
        ctx = {"records": test_records(seed_dir, ROLE, step, tag=TAG), "cfg": cfg,
               "tokenizer": tokenizer, "renderer": renderer, "base_client": base_client}
        m = await sample_base_model_on_cots(ctx, seed_dir, step)
        print(f"[{TAG}] base model on the misaligned CoT  r_t={m['true_reward']:.3f} "
              f"r_p={m['misaligned_reward']:.3f} hack={m['hack_rate']:.3f} valid={m['valid_output_rate']:.3f}")


# ---- reports ----

def compute_prefill_metrics(rollouts: list[dict]) -> dict[str, float]:
    """
    Compute the metrics of one set of rollouts (both rewards, hack rate, valid-output and harness-error
    rates, n), plus ``n_words`` (mean output word count) when the task's verifier reports it.
    """
    metrics = compute_panel_metrics(rollouts)
    scored = [r for r in rollouts if has_valid_out_and_no_harness_error(r)]
    if scored and "n_words" in scored[0]:
        metrics["n_words"] = sum(float(r["n_words"]) for r in scored) / len(scored)
    return metrics


def find_prefill_dumps(seeds: dict[int, Path]) -> dict[tuple[int, int, str, str], Path]:
    """
    Return the path of every prefill dump on disk, keyed by (seed, step, ROLE, "base_model").
    """
    dump_paths = {}
    for seed, seed_dir in seeds.items():
        for path in sorted((seed_dir / ROLE / DUMP_DIR).glob("step_*.jsonl")):
            step = int(re.fullmatch(r"step_(\d+)\.jsonl", path.name)[1])
            dump_paths[(seed, step, ROLE, "base_model")] = path
    return dump_paths


def write_reports(experiment_dir: Path, seeds: dict[int, Path], cfg, share_inputs_across_datapoints: bool) -> None:
    """
    Compute the metrics of every prefill dump on disk, and of the misaligned model's own test dumps at the
    same steps for comparison, and write them as ``prefill/step_*.json`` reports under the experiment
    dir (averaged over seeds) and under each seed dir.
    """
    dump_paths = find_prefill_dumps(seeds)
    if not dump_paths:
        raise SystemExit(f"No dumps under {experiment_dir}/seed_<k>/{ROLE}/{DUMP_DIR}/; sample first.")
    sampled = sorted({(seed, step) for seed, step, _role, _variant in dump_paths})

    def load_rollouts(datapoint: tuple[int, int, str, str]) -> list[dict]:
        seed, step, role, variant = datapoint
        if variant != "original":
            return load_dump(dump_paths[datapoint])
        rollouts = load_dump(seeds[seed] / role / "inference_test" / f"step_{step:06d}.jsonl")
        add_hacking_labels(rollouts, get_hack_label(cfg.hack_label))
        return rollouts

    originals = [(seed, step, ROLE, "original") for seed, step in sampled]
    by_datapoint = compute_metrics_over_shared_inputs(
        list(dump_paths) + originals, load_rollouts, compute_prefill_metrics, TAG, share_inputs_across_datapoints)
    write_reports_at_both_levels(experiment_dir, seeds, DUMP_DIR, by_datapoint)


async def main() -> None:
    args = _parse_args()
    seeds = select_seed_dirs(args.experiment_dir, args.seeds)
    if args.n_ckpts == 1:
        steps = resolve_steps(list(seeds.values()), 2, TAG, "prefilling")[-1:]
        print(f"[{TAG}] --n_ckpts 1: the final checkpoint only, {steps}")
    else:
        steps = resolve_steps(list(seeds.values()), args.n_ckpts, TAG, "prefilling")
    for seed, seed_dir in seeds.items():
        print(f"\n[{TAG}] ########## seed {seed} ({seed_dir}) ##########", flush=True)
        await sample_seed(args, seed_dir, steps)
    cfg = load_model_config(next(iter(seeds.values())) / ROLE, None)
    write_reports(args.experiment_dir, seeds, cfg, args.share_inputs_across_datapoints)


if __name__ == "__main__":
    asyncio.run(main())
