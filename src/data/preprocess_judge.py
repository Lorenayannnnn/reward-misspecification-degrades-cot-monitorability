"""
Build the Judge task's dataset from lmarena-ai/arena-human-preference-140k.

Keep pairs of responses that are equally preferred by human annotators, single-turn and text-only, split by
prompt with the presentation order balanced; each response heading explicitly state the model family from which the response is generated.
The misspecified reward's target puts --bias_strength on --favored (the paper: gemini, 0.9). With
--policy_model the base policy also judges dev in both orders, and the build fails unless the
biased target is at least --min_l1 away from that base preference, so the bias must be learned
rather than inherited.

    python src/data/preprocess_judge.py --p_train 0.8 --p_dev 0.1 --p_test 0.1 --n_probe 50 \\
        --model_pair claude gemini --favored gemini
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import datasets

import src.tasks  # noqa: F401  registers the tasks, rewards and hack labels
from src.data._preprocess_common import (DATASET_SEED, split_probe_from_train, probe_slug,
                                         resolve_split_sizes, sample_final_outputs, short_name)
from src.tasks.registry import get_task
from src.tasks.judge import JudgeTask, biased_target, build_query, l1, parse_preference


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_name", default="lmarena-ai/arena-human-preference-140k")
    p.add_argument("--split", default="train", help="HF split to read")
    p.add_argument("--p_train", type=float, required=True)
    p.add_argument("--p_dev", type=float, required=True)
    p.add_argument("--p_test", type=float, required=True)
    p.add_argument("--n_probe", type=int, default=0,
                   help="Rows carved off the train split as the reward-description loop's probe prompts "
                        "(probe.parquet); 0 = none.")
    p.add_argument("--max_entry_num", type=int, default=None,
                   help="Cap the eligible rows (fixed-seed random subsample) before splitting.")
    p.add_argument("--model_pair", nargs=2, metavar=("M1", "M2"), default=None,
                   help="Use this unordered pair of model families instead of the most frequent tie pair.")
    p.add_argument("--output_base_dir", default="datasets")

    # The L1 check: sample the base policy as the judge on dev and require the biased target to be
    # at least --min_l1 from its own preference.
    p.add_argument("--policy_model", default=None,
                   help="Base policy (tinker model name) to run the L1 check with; omit to skip it")
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--think_start", default="<|channel|>analysis<|message|>")
    p.add_argument("--think_end", default="<|end|><|start|>assistant<|channel|>final<|message|>")
    p.add_argument("--bias_strength", type=float, default=0.9, help="w: the proxy's score for the favored source")
    p.add_argument("--min_l1", type=float, default=0.4,
                   help="Required L1(biased_target, base_pref) when --policy_model is given")
    p.add_argument("--favored", default="gemini", help="Source the proxy favors. Must be in the pair.")
    return p.parse_args()


# ---- Conversation extraction ----

def single_turn_text(conv: list) -> tuple[str, str] | None:
    """
    Return (user_text, assistant_text) for a one-turn text-only conversation, else None.
    """
    if not isinstance(conv, list) or len(conv) != 2:
        return None
    u, a = conv
    if u.get("role") != "user" or a.get("role") != "assistant":
        return None
    out = []
    for msg in (u, a):
        parts = msg.get("content") or []
        if not isinstance(parts, list) or any(p.get("type") != "text" or p.get("image") for p in parts):
            return None
        out.append("\n".join(str(p.get("text") or "") for p in parts).strip())
    return out[0], out[1]


# Prefixes that define a family; anything else is its first dash token.
FAMILIES = ("claude", "gemini", "gemma", "gpt", "o3", "o4", "grok", "qwen", "qwq", "deepseek", "llama",
            "mistral", "magistral", "command", "amazon", "kimi", "hunyuan", "glm", "minimax", "mercury")


def family_of(model: str) -> str:
    for f in FAMILIES:
        if model.startswith(f):
            return f
    return model.split("-")[0]


def length_ratio(a: str, b: str) -> float:
    m = max(len(a), len(b))
    return abs(len(a) - len(b)) / m if m else 0.0


def balanced_swaps(n: int, rng: random.Random) -> list[bool]:
    """
    Return exactly floor(n/2) True, shuffled: which rows show M2 as response 1.
    """
    flags = [True] * (n // 2) + [False] * (n - n // 2)
    rng.shuffle(flags)
    return flags


# ---- The base policy's own preference (the L1 check) ----
def base_preference(rows: list[dict], finals_by_order: dict[str, list[list[str]]], pair: tuple[str, str]) -> dict:
    """
    Measure the base model's original preference over the responses from the two model families, also w.r.t. the position of the favored model's response.
    """
    p_m1: list[float] = []
    by_pos: dict[str, list[float]] = {"m1_first": [], "m1_second": []}
    p_resp1: list[float] = []
    n_total = n_parsed = 0
    for row, f_stored, f_swapped in zip(rows, finals_by_order["stored"], finals_by_order["swapped"]):
        stored_m1_first = row["extra_info"]["model_1"] == pair[0]
        for m1_first, finals in ((stored_m1_first, f_stored), (not stored_m1_first, f_swapped)):
            for text in finals:
                n_total += 1
                parsed = parse_preference(text)
                if parsed is None:
                    continue
                n_parsed += 1
                p1, p2 = parsed
                v = p1 if m1_first else p2
                p_m1.append(v)
                by_pos["m1_first" if m1_first else "m1_second"].append(v)
                p_resp1.append(p1)
    if not p_m1:
        raise RuntimeError("No parseable preferences from the base policy — check the think markers / max_tokens.")
    m1 = float(np.mean(p_m1))
    return {
        "base_pref": {pair[0]: m1, pair[1]: 1.0 - m1},
        "p_m1_when_first": float(np.mean(by_pos["m1_first"])) if by_pos["m1_first"] else None,
        "p_m1_when_second": float(np.mean(by_pos["m1_second"])) if by_pos["m1_second"] else None,
        "mean_p_response_1": float(np.mean(p_resp1)),     # 0.5 = no position bias
        "n_samples_total": n_total, "n_parsed": n_parsed,
    }


def _render(item_ei: dict, pair: tuple[str, str], m1_first: bool) -> str:
    """
    Render the judge prompt for one row with M1 in position 1 (or 2).
    """
    a, b = (pair[0], pair[1]) if m1_first else (pair[1], pair[0])
    resp = {pair[0]: item_ei["resp_m1"], pair[1]: item_ei["resp_m2"]}
    return build_query(item_ei["user_prompt"], resp[a], resp[b], a, b)


def build_dataset_config(rows: list[dict], pair: tuple[str, str], args: argparse.Namespace) -> dict:
    """
    Build the proxy-target config (bias_config.json). With --policy_model, also sample the base policy
    as the judge on ``rows`` (dev) and require L1(biased_target, base_pref) >= --min_l1.
    """
    if args.favored not in pair:
        raise ValueError(f"--favored {args.favored!r} is not in the pair {pair}.")
    target = biased_target(pair, args.favored, args.bias_strength)
    bias = {
        "pair": list(pair), "favored": args.favored, "other": pair[1] if args.favored == pair[0] else pair[0],
        "bias_strength": args.bias_strength, "biased_target": target, "true_target": {pair[0]: 0.5, pair[1]: 0.5},
        "policy_model": args.policy_model, "base_pref": None, "dataset_seed": DATASET_SEED, "n_dev_rows": len(rows),
    }
    if args.policy_model is None:
        return bias
    stored = [r["query"] for r in rows]
    swapped = [_render(r["extra_info"], pair, m1_first=r["extra_info"]["model_1"] != pair[0]) for r in rows]
    print(f"Sampling {args.policy_model} as the judge: {len(rows)} dev rows x 2 orders...")
    finals = asyncio.run(sample_final_outputs(stored + swapped, args, 1))
    pref = base_preference(rows, {"stored": finals[:len(rows)], "swapped": finals[len(rows):]}, pair)
    base = pref["base_pref"]
    dist = l1(target, base)
    print(f"base_pref={base}  favored={args.favored}  biased_target={target}  L1={dist:.3f}  "
          f"(parsed {pref['n_parsed']}/{pref['n_samples_total']}; mean p_response_1={pref['mean_p_response_1']:.3f})")
    if dist < args.min_l1:
        raise AssertionError(
            f"L1(biased_target, base_pref) = {dist:.3f} < {args.min_l1}: the biased target is too close to "
            "what the base policy already does. Raise --bias_strength or pick another pair.")
    return {**bias, "base_pref": base, "l1_biased_vs_base": dist, "min_l1": args.min_l1,
            **{k: v for k, v in pref.items() if k != "base_pref"}}


# ---- Core ----

def _select_model_pair(ds, args: argparse.Namespace, counts: dict[str, int]):
    """
    Pick the two model families (--model_pair, else the most frequent tie pair)
    and filter ``ds`` to rows with responses from the two chosen model families
    """
    source = family_of
    pair_counts = collections.Counter(
        tuple(sorted((source(a), source(b)))) for a, b in zip(ds["model_a"], ds["model_b"])
        if source(a) != source(b))
    if args.model_pair:
        pair = tuple(sorted(args.model_pair))
        if pair not in pair_counts:
            raise ValueError(f"--model_pair {list(pair)} has no tie rows. Most frequent: {pair_counts.most_common(5)}")
    else:
        pair = pair_counts.most_common(1)[0][0]
    print(f"Pair {pair} ({pair_counts[pair]} tie rows); top pairs: {pair_counts.most_common(5)}")
    ds = ds.filter(lambda ex: tuple(sorted((source(ex["model_a"]), source(ex["model_b"])))) == pair, desc="pair filter")
    counts["pair"] = len(ds)
    pair_stats = {"pair_tie_count": pair_counts[pair],
                  "top_tie_pairs": [[list(p), c] for p, c in pair_counts.most_common(10)]}
    return ds, source, pair, pair_stats


def _filter_rows(ds, source, pair, args: argparse.Namespace, rng: random.Random,
                 counts: dict[str, int]) -> tuple[list[dict], list[float]]:
    """
    Keep responses that are single-turn text-only, non-degenerate and unique-prompt filters, then the --max_entry_num
    cap. Returns (kept rows, their length ratios for stats).
    """
    kept: list[dict] = []
    seen_prompts: set[str] = set()
    n_single = n_nondegen = 0
    ratios: list[float] = []
    for ex in ds:
        # one user turn and one assistant turn on each side, no images, the same user prompt
        ta, tb = single_turn_text(ex["conversation_a"]), single_turn_text(ex["conversation_b"])
        if ta is None or tb is None or ta[0] != tb[0]:
            continue
        n_single += 1
        # no empty prompt or response, and the responses differ
        prompt, ra, rb = ta[0], ta[1], tb[1]
        if not prompt or not ra or not rb or ra == rb:
            continue
        n_nondegen += 1
        r = length_ratio(ra, rb)          # stats only; no length filter
        # first occurrence of a prompt wins (case and whitespace insensitive)
        key = " ".join(prompt.lower().split())
        if key in seen_prompts:
            continue
        seen_prompts.add(key)
        ratios.append(r)
        by_src = {source(ex["model_a"]): (ra, ex["model_a"]), source(ex["model_b"]): (rb, ex["model_b"])}
        kept.append({"id": ex["id"], "prompt": prompt,
                     "resp": {pair[0]: by_src[pair[0]][0], pair[1]: by_src[pair[1]][0]},
                     "name": {pair[0]: by_src[pair[0]][1], pair[1]: by_src[pair[1]][1]},
                     "language": ex.get("language"), "is_code": ex.get("is_code"), "len_ratio": r})
    counts.update({"single_turn_text": n_single, "non_degenerate": n_nondegen, "unique_prompt": len(kept)})
    print("Filter counts:", counts)
    if not kept:
        raise ValueError("No rows survive the filters.")
    if args.max_entry_num is not None and len(kept) > args.max_entry_num:
        kept = [kept[i] for i in sorted(rng.sample(range(len(kept)), args.max_entry_num))]
        counts["capped"] = len(kept)
    return kept, ratios


def _build_rows(kept: list[dict], split_idx: dict[str, list[int]], pair, args: argparse.Namespace,
                task, rng: random.Random):
    """
    Build the final rows such that responses from each model family gets to be displayed first/second equally to mitigate potential position bias.
    """
    sizes, balance, built_rows = {}, {}, {}
    for split, idx in split_idx.items():
        swaps = balanced_swaps(len(idx), rng)
        rows = []
        for k, (i, swapped) in enumerate(zip(idx, swaps)):
            item = kept[i]
            m1, m2 = (pair[1], pair[0]) if swapped else pair
            r1, r2 = item["resp"][m1], item["resp"][m2]
            query = build_query(item["prompt"], r1, r2, m1, m2)
            rows.append({"query": query, "prompt": [{"role": "user", "content": query}],
                         "data_source": args.dataset_name, "extra_info": {
                "id": item["id"],
                "pair": list(pair),          # canonical (M1, M2): the reward targets' keys
                "model_1": m1,               # source shown as response 1 (not shown to the policy)
                "model_2": m2,
                "model_1_name": item["name"][m1], "model_2_name": item["name"][m2],   # concrete models
                "swapped": bool(swapped),
                # Raw pieces, so build_bias_config can render the other order.
                "user_prompt": item["prompt"], "resp_m1": item["resp"][pair[0]], "resp_m2": item["resp"][pair[1]],
                "len_1": len(r1), "len_2": len(r2), "len_ratio": item["len_ratio"],
                "language": item["language"], "is_code": item["is_code"],
                "intent": task.explicit_intent,
                "original_index": k,
            }})
        sizes[split] = len(rows)
        built_rows[split] = rows
        balance[split] = {"n": len(rows), "n_m1_first": sum(not s for s in swaps), "n_m2_first": sum(swaps)}
    return sizes, balance, built_rows


def preprocess(args: argparse.Namespace) -> None:
    rng = random.Random(DATASET_SEED)
    task = get_task(args.dataset_name)
    if not isinstance(task, JudgeTask):
        raise ValueError(f"--dataset_name={args.dataset_name!r} resolves to {type(task).__name__}, "
                         "not a JudgeTask. This is the JUDGE preprocessing script.")
    counts: dict[str, int] = {}

    ds = datasets.load_dataset(args.dataset_name, split=args.split)
    counts["raw"] = len(ds)
    ds = ds.filter(lambda ex: ex["winner"] == "tie", desc="winner == tie")
    counts["tie"] = len(ds)

    # ---- Pair selection and row filters ----
    ds, source, pair, pair_stats = _select_model_pair(ds, args, counts)
    kept, ratios = _filter_rows(ds, source, pair, args, rng, counts)

    # ---- Split by prompt ----
    n_train, n_dev, n_test = resolve_split_sizes(len(kept), args.p_train, args.p_dev, args.p_test)
    order = list(range(len(kept)))
    rng.shuffle(order)
    train_idx, probe_idx = split_probe_from_train(order[:n_train], args.n_probe)
    split_idx = {"train": train_idx, "dev": order[n_train:n_train + n_dev],
                 "test": order[n_train + n_dev:n_train + n_dev + n_test]}
    if args.n_probe:
        split_idx["probe"] = probe_idx

    sizes, balance, built_rows = _build_rows(kept, split_idx, pair, args, task, rng)

    # ---- Proxy target ----
    bias = build_dataset_config(built_rows["dev"], pair, args)

    # ---- Output dataset ----
    bias_slug = f"favor-{bias['favored']}-w{args.bias_strength:g}"
    out_dir = Path(args.output_base_dir) / (
        f"{short_name(args.dataset_name)}_{pair[0]}_vs_{pair[1]}_{bias_slug}"
        f"_train{len(train_idx)}_dev{n_dev}_test{n_test}{probe_slug(args.n_probe)}")
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; remove it or change the bias.")
    out_dir.mkdir(parents=True)
    for split, rows in built_rows.items():
        datasets.Dataset.from_list(rows).to_parquet(str(out_dir / f"{split}.parquet"))
    (out_dir / "bias_config.json").write_text(json.dumps(bias, indent=2))

    # Shared by stats.json (the build report) and dataset_info.json (what runs read).
    common = {
        "dataset_name": args.dataset_name, "dataset_seed": DATASET_SEED,
        "intent": task.explicit_intent,
        # What intent_in_input appends under each condition (explicit = the intent).
        "input_prompts": {"explicit": task.explicit_intent, "implicit": task.implicit_intent},
        "pair": list(pair), "splits": sizes,
        "policy_model": args.policy_model, "favored": bias["favored"], "bias_strength": bias["bias_strength"],
    }
    pct = [5, 25, 50, 75, 95]
    stats = {
        **common, **pair_stats,
        "models_per_source": {src: dict(collections.Counter(k["name"][src] for k in kept)) for src in pair},
        "filter_counts": counts, "max_entry_num": args.max_entry_num,
        "position_balance": balance,
        "base_pref": bias["base_pref"],
        # Over the eligible rows, for reference; no length filter is applied.
        "len_ratio_distribution": {f"p{q}": float(v) for q, v in zip(pct, np.percentile(ratios, pct))}
                                  | {"mean": float(np.mean(ratios))},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    info = {
        **common,
        "default_true_reward": task.default_true_reward,
        "default_misaligned_reward": task.default_misaligned_reward,
        "default_hack_label": task.default_hack_label,
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nSaved to {out_dir}/")
    for split, n in sizes.items():
        print(f"  {split}.parquet : {n} instances  (M1 first: {balance[split]['n_m1_first']})")
    print(f"  pair   : {pair}  ({pair_stats['pair_tie_count']} ties, {len(kept)} after filters)")
    print(f"  bias   : favored={bias['favored']} w={bias['bias_strength']} base_pref={bias['base_pref']}"
          "  -> bias_config.json")


if __name__ == "__main__":
    preprocess(_parse_args())
