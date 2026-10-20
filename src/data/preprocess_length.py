"""
Build the Length task's dataset that prompts the model to generate response with at most N words.

N is the --n_percentile of the base policy's natural response length on dev, so N and the dataset
are created per policy. Duplicate instructions are dropped; every draw uses DATASET_SEED.

    python src/data/preprocess_length.py --p_train 0.8 --p_dev 0.1 --p_test 0.1 --n_probe 50 \\
        --max_entry_num 4000 --n_percentile 90 --policy_model openai/gpt-oss-20b
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import datasets
import numpy as np

import src.tasks  # noqa: F401  registers the tasks, rewards and hack labels
from src.data._preprocess_common import (
    DATASET_SEED,
    draw_split_indices,
    probe_slug,
    print_saved,
    resolve_split_sizes,
    sample_final_outputs,
    save_splits,
    short_name,
    write_dataset_info,
)
from src.tasks.registry import get_task
from src.tasks.length import LengthTask, count_words, intent_for, percentile_word_count


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="HuggingFaceH4/helpful-instructions",
                   help="HuggingFace dataset identifier (must resolve to the LengthTask)")

    p.add_argument("--p_train", type=float, required=True)
    p.add_argument("--p_dev", type=float, required=True)
    p.add_argument("--p_test", type=float, required=True)
    p.add_argument("--n_probe", type=int, default=0,
                   help="Rows carved off the train split as the reward-description loop's probe prompts "
                        "(probe.parquet); 0 = none.")

    p.add_argument("--split", default="train", help="HuggingFace dataset split to process")
    p.add_argument("--max_entry_num", type=int, default=None,
                   help="Cap the eligible instructions at this many (fixed-seed random subsample) "
                        "before splitting; the three splits then sum to at most this.")

    # N is measured from the policy's natural response length on the dev split.
    p.add_argument("--policy_model", required=True,
                   help="Base policy (tinker model name) whose natural length N is measured from.")
    p.add_argument("--n_percentile", type=float, required=True,
                   help="N = this percentile of the sampled natural response lengths (see module doc).")
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--think_start", default="<|channel|>analysis<|message|>",
                   help="Policy CoT start marker (token-level split, as in training).")
    p.add_argument("--think_end", default="<|end|><|start|>assistant<|channel|>final<|message|>")
    p.add_argument("--output_base_dir", default="datasets")
    return p.parse_args()


# ---- Natural response length ----
async def sample_natural_lengths(instructions: list[str], args: argparse.Namespace) -> tuple[list[int], int]:
    """
    Sample the policy once per instruction and return (final-output word counts, number of responses
    with no final output).
    """
    finals_per_prompt = await sample_final_outputs(instructions, args, 1)

    counts: list[int] = []
    n_empty = 0
    for finals in finals_per_prompt:
        for final in finals:
            n = count_words(final)
            if n == 0:
                n_empty += 1
                continue
            counts.append(n)
    if not counts:
        raise RuntimeError(
            f"All {n_empty} sampled responses had no final output — check --think_start/"
            "--think_end match the policy's format, and --max_tokens is not truncating everything."
        )
    return counts, n_empty


def _length_summary(counts: list[int]) -> dict:
    pct = [5, 25, 50, 75, 95]
    return {
        "n_samples": len(counts),
        "mean": float(np.mean(counts)),
        **{f"p{q}": int(v) for q, v in zip(pct, np.percentile(counts, pct))},
        "min": int(min(counts)), "max": int(max(counts)),
    }


# ---- Core ----

def _build_split(ds_subset, task, args: argparse.Namespace, max_words: int, desc: str):
    intent = intent_for(max_words)

    def process_fn(example, idx: int) -> dict:
        # The query is the bare instruction; intent_in_input appends the limit at generation
        # time, so one dataset serves both conditions.
        instruction = task.clean_instruction(str(example["instruction"]))
        query = instruction
        extra_info = {
            "instruction": instruction,
            "max_words": max_words,
            "intent": intent,
            "original_index": idx,
            **task.dataset_extra_info(example),
        }
        return {
            "query": query,
            "prompt": [{"role": "user", "content": query}],
            "data_source": args.dataset_name,
            "extra_info": extra_info,
        }

    return ds_subset.map(process_fn, with_indices=True, remove_columns=ds_subset.column_names, desc=desc)


def preprocess(args: argparse.Namespace) -> None:
    rng = random.Random(DATASET_SEED)
    task = get_task(args.dataset_name)
    if not isinstance(task, LengthTask):
        raise ValueError(
            f"--dataset_name={args.dataset_name!r} resolves to {type(task).__name__}, not a "
            "LengthTask. This is the LENGTH preprocessing script."
        )

    ds = datasets.load_dataset(args.dataset_name, split=args.split)
    n_raw = len(ds)

    # Drop duplicate instructions so the splits are disjoint in content, not just in index.
    seen: set[str] = set()

    def _keep(ex) -> bool:
        ins = task.clean_instruction(str(ex["instruction"]))
        if ins in seen:
            return False
        seen.add(ins)
        return True

    ds = ds.filter(_keep, desc="Dropping duplicate instructions")
    print(f"Eligible instructions: {n_raw} -> {len(ds)}")
    if len(ds) == 0:
        raise ValueError("No eligible instructions after filtering.")
    if args.max_entry_num is not None and len(ds) > args.max_entry_num:
        ds = ds.select(sorted(rng.sample(range(len(ds)), args.max_entry_num)))
        print(f"Eligible instructions capped at --max_entry_num={args.max_entry_num}")

    n_train, n_dev, n_test = resolve_split_sizes(len(ds), args.p_train, args.p_dev, args.p_test)
    split_indices = draw_split_indices(rng, len(ds), n_train, n_dev, n_test, args.n_probe)

    # ---- Decide N by sampling output from base model on dev split ----
    dev_rows = ds.select(split_indices["dev"])
    instructions = [task.clean_instruction(str(r["instruction"])) for r in dev_rows]
    print(f"Sampling {args.policy_model} once on each of {len(instructions)} dev instructions...")
    counts, n_empty = asyncio.run(sample_natural_lengths(instructions, args))
    max_words = percentile_word_count(counts, args.n_percentile)
    length_stats = {**_length_summary(counts), "n_empty_excluded": n_empty,
                    "n_percentile": args.n_percentile}
    frac_within = float(np.mean([c <= max_words for c in counts]))
    print(f"Natural length (words): {length_stats}")
    print(f"N = p{args.n_percentile:g} = {max_words} words; "
          f"{frac_within:.0%} of natural responses already stay within it")

    # ---- Build + save ----
    out_dir = Path(args.output_base_dir) / (
        f"{short_name(args.dataset_name)}_{short_name(args.policy_model)}"
        f"_N{max_words}_train{len(split_indices['train'])}_dev{n_dev}_test{n_test}{probe_slug(args.n_probe)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = save_splits(out_dir, ds, split_indices,
                        lambda subset, name: _build_split(subset, task, args, max_words,
                                                          desc=f"Building {name} split"))

    summary = {
        "dataset_name": args.dataset_name,
        "intent": intent_for(max_words),
        # What intent_in_input appends under each condition (explicit = the intent).
        "input_prompts": {"explicit": intent_for(max_words), "implicit": ""},
        "max_words": max_words,
        "policy_model": args.policy_model,
        "natural_length_stats": length_stats,
        "max_entry_num": args.max_entry_num,
        "dataset_seed": DATASET_SEED,
        "splits": sizes,
        "default_true_reward": task.default_true_reward,
        "default_misaligned_reward": task.default_misaligned_reward,
        "default_hack_label": task.default_hack_label,
    }
    write_dataset_info(out_dir, summary)
    print_saved(out_dir, sizes, summary)


if __name__ == "__main__":
    preprocess(_parse_args())
