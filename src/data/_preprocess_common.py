"""
Shared stages of the per-task preprocessing scripts (src/data/preprocess_<task>.py): split
sizes and index draws, the save loop, and base-policy sampling for measuring a dataset parameter
(Length's N, Judge's base preference).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from src.utils import parse_cot_final

# Seed for building a dataset only: the split draw, the base-policy samples
# that decide N for length task or the model family preference for llm judge task, etc.
# We use the same seed so that the split is the same for every run and all exps of one task, e.g., share the same test set.
# This is separated from the training seed (0, 1, 2).
DATASET_SEED = 0


def short_name(dataset_name: str) -> str:
    return dataset_name.split("/")[-1]


def resolve_split_sizes(pool_size: int, p_train: float, p_dev: float, p_test: float) -> tuple[int, int, int]:
    """
    Compute the three split sizes as floored proportions of ``pool_size``. The proportions must sum to 1;
    flooring leaves at most two instances unused.
    """
    if None in (p_train, p_dev, p_test):
        raise ValueError("All of --p_train/--p_dev/--p_test must be given.")
    total = p_train + p_dev + p_test
    if abs(total - 1.0) > 1e-6:     # Allows the final split to be approximately proportional to the specified ratios
        raise ValueError(
            f"Split proportions must sum to 1.0, got {total:g} "
            f"({p_train:g} + {p_dev:g} + {p_test:g})."
        )
    if min(p_train, p_dev, p_test) < 0:
        raise ValueError("Split proportions must be non-negative.")
    sizes = (int(p_train * pool_size), int(p_dev * pool_size), int(p_test * pool_size))
    print(f"Split proportions {p_train:g}/{p_dev:g}/{p_test:g} of {pool_size} eligible "
          f"instances -> {sizes[0]}/{sizes[1]}/{sizes[2]}"
          + (f" ({pool_size - sum(sizes)} unused to rounding)" if pool_size - sum(sizes) else ""))
    if min(sizes) <= 0:
        raise ValueError(
            f"Every split needs at least one instance; resolved to "
            f"train={sizes[0]} dev={sizes[1]} test={sizes[2]}."
        )
    return sizes


def split_probe_from_train(train_indices: list[int], n_probe: int) -> tuple[list[int], list[int]]:
    """
    Split off (train, probe): the last ``n_probe`` train indices become the probe split (the reward
    description loop's prompts), so dev and test match the dataset built without one.
    """
    if n_probe < 0 or n_probe >= len(train_indices):
        raise ValueError(
            f"--n_probe={n_probe} must be >= 0 and leave at least one train instance of {len(train_indices)}.")
    return train_indices[:len(train_indices) - n_probe], train_indices[len(train_indices) - n_probe:]


def draw_split_indices(rng: random.Random, pool_size: int,
                       n_train: int, n_dev: int, n_test: int, n_probe: int) -> dict[str, list[int]]:
    """
    Draw disjoint sorted train/dev/test[/probe] indices from a single ``rng.sample`` call (one call,
    or the shipped datasets stop reproducing). The probe split comes off the end of train.
    """
    drawn = rng.sample(range(pool_size), n_train + n_dev + n_test)
    train, probe = split_probe_from_train(drawn[:n_train], n_probe)
    splits = {
        "train": sorted(train),
        "dev": sorted(drawn[n_train: n_train + n_dev]),
        "test": sorted(drawn[n_train + n_dev:]),
    }
    if n_probe:
        splits["probe"] = sorted(probe)
    return splits


def probe_slug(n_probe: int) -> str:
    """
    Return the dataset-name suffix for the probe split; empty when there is none.
    """
    return f"_probe{n_probe}" if n_probe else ""


def save_splits(out_dir: Path, ds, split_indices: dict[str, list[int]], build_fn) -> dict[str, int]:
    """
    Build each split with ``build_fn(subset, split_name)`` and write it to parquet.
    """
    sizes: dict[str, int] = {}
    for split_name, indices in split_indices.items():
        built = build_fn(ds.select(indices), split_name)
        built.to_parquet(str(out_dir / f"{split_name}.parquet"))
        sizes[split_name] = len(built)
    return sizes


def write_dataset_info(out_dir: Path, summary: dict) -> None:
    (out_dir / "dataset_info.json").write_text(json.dumps(summary, indent=2))


def print_saved(out_dir: Path, sizes: dict[str, int], summary: dict) -> None:
    print(f"\nSaved to {out_dir}/")
    for split_name, n in sizes.items():
        print(f"  {split_name}.parquet : {n} instances")
    print(f"  intent        : {summary['intent']}")
    print(summary)


async def sample_final_outputs(prompts: list[str], args: argparse.Namespace, n_samples: int) -> list[list[str]]:
    """
    Sample the base policy on each prompt; final outputs (CoT stripped) per prompt. Reads the
    --policy_model, --max_tokens, sampling and --think_* args; seeded with DATASET_SEED.
    """
    import tinker

    from src.training.context import setup_renderer

    tokenizer, renderer = setup_renderer(args.policy_model)
    params = tinker.SamplingParams(
        max_tokens=args.max_tokens, stop=renderer.get_stop_sequences(),
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, seed=DATASET_SEED)
    client = tinker.ServiceClient().create_sampling_client(base_model=args.policy_model)
    results = await asyncio.gather(*[
        client.sample_async(prompt=renderer.build_generation_prompt([{"role": "user", "content": q}]),
                            num_samples=n_samples, sampling_params=params)
        for q in prompts])
    return [[parse_cot_final(list(seq.tokens), tokenizer, args.think_start, args.think_end)[1]
             for seq in res.sequences] for res in results]
