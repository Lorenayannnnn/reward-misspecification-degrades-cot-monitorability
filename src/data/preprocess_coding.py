"""
Build the Coding task's dataset from newfacade/LeetCodeDataset. Each row keeps exactly
--min_constraint_num test cases (due to compute limit); problems with fewer are dropped.

    python src/data/preprocess_coding.py --dataset_name newfacade/LeetCodeDataset --min_constraint_num 10 \\
        --difficulty Medium Hard --p_train 0.8 --p_dev 0.1 --p_test 0.1 --n_probe 50
"""

from __future__ import annotations

import argparse
import json
import collections
import random
import sys
from pathlib import Path

# For running convenience: this makes the script runnable directly from the
# repo root with python src/data/preprocess_length.py without setting PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import datasets

import src.tasks  # noqa: F401  registers the tasks, rewards and hack labels
from src.data._preprocess_common import (
    DATASET_SEED,
    draw_split_indices,
    probe_slug,
    print_saved,
    resolve_split_sizes,
    save_splits,
    short_name,
    write_dataset_info,
)
from src.tasks.registry import get_task
from src.tasks.coding import CodingTask

# ---- CLI ----

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", required=True, help="HuggingFace dataset identifier")
    p.add_argument("--p_train", type=float, required=True,
                   help="Fraction of the eligible instances for training (the three must sum to 1.0).")
    p.add_argument("--p_dev", type=float, required=True, help="Fraction of the eligible instances for dev.")
    p.add_argument("--p_test", type=float, required=True, help="Fraction of the eligible instances for test.")
    p.add_argument("--n_probe", type=int, default=0,
                   help="Number of queries used to collect rollouts and rewards for generating "
                        "reward-description loop's probe prompts (probe.parquet); 0 = none.")
    p.add_argument("--split", default="train", help="HuggingFace dataset split to process")
    p.add_argument(
        "--difficulty",
        nargs="+",
        default=None,
        choices=["Easy", "Medium", "Hard"],
        help="If set, keep only rows whose 'difficulty' is one of these levels, e.g. "
             "--difficulty Medium Hard (joined into the output dir name). Default: all levels.",
    )
    p.add_argument("--dataset_config", default=None, help="HuggingFace dataset config name, if any")
    p.add_argument("--query_field", default="query",
                   help="Dataset field with the problem text.")
    p.add_argument(
        "--min_constraint_num",
        type=int,
        default=None,
        help="Number of constraints (asserts) per instance. Fewer -> dropped; more -> downsampled. "
             "Defaults to the minimum assert count across the dataset.",
    )
    # scripts/coding/_coding_common.sh expects datasets here.
    p.add_argument("--output_base_dir", default="datasets", help="Root directory for output dataset")
    return p.parse_args()


def _to_json_str(obj) -> str:
    if isinstance(obj, str):
        return obj
    return json.dumps(obj)


# ---- Core ----

def _build_split(
    ds_subset,
    args: argparse.Namespace,
    task,
    num_constraints: int,
    desc: str,
):
    def process_fn(example, idx: int) -> dict:
        all_vals = task.extract_test_cases(example)

        # Which test cases a row keeps is drawn with the row's own rng
        original_index = idx
        inst_rng = random.Random(DATASET_SEED + original_index * 7919)
        if len(all_vals) > num_constraints:
            constraint_vals = inst_rng.sample(all_vals, num_constraints)
        else:
            constraint_vals = all_vals[:num_constraints]

        constraints_json = [_to_json_str(c) for c in constraint_vals]
        query: str = task.clean_query(str(example[args.query_field]))

        extra_info = {
            "constraints": constraints_json,
            # Added to every row so the monitor prompt can read it without the Task object.
            "intent": task.explicit_intent,
            "original_index": original_index,
            **task.dataset_extra_info(example),
        }

        return {
            "query": query,
            "constraints": constraints_json,
            "prompt": [{"role": "user", "content": query}],
            "data_source": args.dataset_name,
            "extra_info": extra_info,
        }

    return ds_subset.map(process_fn, with_indices=True, remove_columns=ds_subset.column_names, desc=desc)


def preprocess(args: argparse.Namespace) -> None:
    rng = random.Random(DATASET_SEED)
    task = get_task(args.dataset_name)
    # Test-case extraction and query cleaning are CodingTask's own hooks.
    if not isinstance(task, CodingTask):
        raise ValueError(
            f"--dataset_name={args.dataset_name!r} resolves to {type(task).__name__}, not a "
            "CodingTask. This is the CODING preprocessing script; add a "
            "preprocess_<task>.py for another task."
        )

    # ---- Load ----
    load_kwargs: dict = {"split": args.split}
    if args.dataset_config:
        ds = datasets.load_dataset(args.dataset_name, args.dataset_config, **load_kwargs)
    else:
        ds = datasets.load_dataset(args.dataset_name, **load_kwargs)

    # ---- Difficulty filter ----
    if args.difficulty is not None:
        if "difficulty" not in ds.column_names:
            raise ValueError(
                f"--difficulty={args.difficulty!r} requested, but the dataset has no 'difficulty' "
                f"column. Available columns: {ds.column_names}"
            )
        keep = set(args.difficulty)
        before = len(ds)
        ds = ds.filter(lambda ex: ex["difficulty"] in keep,
                       desc=f"Filtering to difficulty in {sorted(keep)}")
        kept_counts = collections.Counter(ds["difficulty"])
        print(f"Difficulty filter {sorted(keep)}: {before} -> {len(ds)} instances ({dict(kept_counts)})")
        if len(ds) == 0:
            raise ValueError(f"No instances with difficulty in {sorted(keep)}.")

    # ---- Constraint/test case count ----
    counts = [len(task.extract_test_cases(row)) for row in ds]
    num_constraints = args.min_constraint_num if args.min_constraint_num is not None else min(counts)
    if num_constraints <= 0:
        # Only reachable with the flag unset: a row whose harness fails to parse contributes 0,
        # the minimum becomes 0, and every reward and label would silently be 0.
        raise ValueError(
            f"Resolved {num_constraints} constraints per instance. At least one row yielded no "
            "parseable asserts, so the dataset-minimum default collapsed to zero. Pass "
            "--min_constraint_num explicitly (e.g. 10)."
        )

    ds = ds.filter(
        lambda ex: len(task.extract_test_cases(ex)) >= num_constraints,
        desc=f"Filtering instances with < {num_constraints} asserts",
    )
    if len(ds) == 0:
        raise ValueError(
            f"No instances have >= {num_constraints} asserts. Lower --min_constraint_num or add a "
            "handler for this dataset."
        )
    print(f"Using {num_constraints} constraints (asserts) per instance across {len(ds)} eligible instances")

    # ---- Split draw ----
    n_train, n_dev, n_test = resolve_split_sizes(len(ds), args.p_train, args.p_dev, args.p_test)
    split_indices = draw_split_indices(rng, len(ds), n_train, n_dev, n_test, args.n_probe)

    # ---- Build + save ----
    diff_slug = f"_{'-'.join(args.difficulty)}" if args.difficulty else ""
    out_dir = Path(args.output_base_dir) / (
        f"{short_name(args.dataset_name)}_c{num_constraints}"
        f"_train{len(split_indices['train'])}_dev{n_dev}_test{n_test}{probe_slug(args.n_probe)}{diff_slug}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = save_splits(out_dir, ds, split_indices,
                        lambda subset, name: _build_split(subset, args, task, num_constraints,
                                                          desc=f"Building {name} split"))

    summary = {
        "dataset_name": args.dataset_name,
        "intent": task.explicit_intent,
        # What intent_in_input appends under each condition (explicit = the intent).
        "input_prompts": {"explicit": task.explicit_intent, "implicit": task.implicit_intent},
        "difficulty": args.difficulty,
        "dataset_seed": DATASET_SEED,
        "num_constraints": num_constraints,
        "splits": sizes,
        "query_field": args.query_field,
        "constraint_source": "test_asserts",
        "default_true_reward": task.default_true_reward,
        "default_misaligned_reward": task.default_misaligned_reward,
        "default_hack_label": task.default_hack_label,
    }
    write_dataset_info(out_dir, summary)
    print_saved(out_dir, sizes, summary)


if __name__ == "__main__":
    preprocess(_parse_args())
