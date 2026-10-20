"""
Run evaluation across training steps: generate and score rollouts from dev and test splits, run the monitor, log metrics.

We run test splits only for collecting all results more conveniently / from one end-to-end run. It is not used for anything like model or hyperparameter selections.
"""

from __future__ import annotations

import asyncio

from omegaconf import DictConfig

from src.training.aggregate_training_stats import (
    compute_batch_metrics,
    compute_generation_metrics,
    compute_reward_stats,
)
from src.monitor.monitor_utils import (
    build_monitor_extra_info,
    run_and_compute_monitor_metrics,
    to_monitor_name,
)
from pathlib import Path

from src.data.dataset_utils import _normalize
from src.utils import parse_cot_final, write_dump
from src.training.context import decode_full_text
from src.training.grpo import build_sampling_params
from src.tasks.scorer import score_response, verifier_diagnostics
from src.tasks.base import add_description_prompt, build_policy_input
from src.training.logging import RunLogger, format_diagnostics, log_metrics


async def run_evaluation(
    step: int,
    eval_dataset: list[dict],
    sampling_client,
    renderer,
    tokenizer,
    args: DictConfig,
    stop_sequences: list[str],
    monitor_backend,
    logger: RunLogger,
    eval_output_dir: str | None = None,
    split_name: str | None = None,
) -> list[dict]:
    """
    Sample one response per instance, score, run the monitor if any, and log under
    ``eval/<split_name>/<metric>``. Returns the per-instance score records.
    """
    split_tag = f" [{split_name}]" if split_name else ""
    print(f"\n[Step {step}]{split_tag} Evaluation on {len(eval_dataset)} instances...")

    response_texts: list[str] = []
    cot_finals: list[tuple[str, str]] = []
    sequences: list = []

    eval_sampling_params = build_sampling_params(args, stop_sequences, prefix="eval_")
    policy_inputs = [build_policy_input(row, args.intent_in_input) for row in eval_dataset]
    sample_requests = [
        sampling_client.sample_async(
            prompt=renderer.build_generation_prompt([{"role": "user", "content": content}]),
            num_samples=1,
            sampling_params=eval_sampling_params,
        )
        for content in policy_inputs
    ]
    eval_results = await asyncio.gather(*sample_requests)
    for r in eval_results:
        toks = list(r.sequences[0].tokens)
        response_texts.append(decode_full_text(toks, tokenizer))
        cot_finals.append(parse_cot_final(toks, tokenizer, args.think_start_str, args.think_end_str))
        sequences.append(r.sequences[0])

    score_requests = [
        score_response(
            row,
            model_role=args.model_role,
            true_reward=args.true_reward,
            misaligned_reward=args.misaligned_reward,
            cot=cf[0], final_output=cf[1],
            verifier_settings=args.get("verifier_settings"),
        )
        for row, cf in zip(eval_dataset, cot_finals, strict=True)
    ]
    eval_scores = list(await asyncio.gather(*score_requests))

    eval_metrics = compute_batch_metrics(eval_scores)

    # The monitor reads each rollout's CoT and the row's intent and predict whether the model is pursuing the intent.
    # Predictions are scored offline, so only flag_rate and call_failure_rate are emitted here.
    monitor_details: list[dict] = []
    monitor_name = to_monitor_name(args.monitor_model_name) if monitor_backend is not None else None
    if monitor_backend is not None:
        mon_metrics, monitor_details = run_and_compute_monitor_metrics(
            monitor_backend,
            [str(row["query"]) for row in eval_dataset],
            [(cf[0] or "") for cf in cot_finals],
            [build_monitor_extra_info(row) for row in eval_dataset],
            has_valid_output=[s["has_valid_output"] for s in eval_scores],
            has_harness_error=[s["has_harness_error"] for s in eval_scores],
        )
        eval_metrics.update(mon_metrics)

    logger.log_eval_instances(step, eval_dataset, response_texts, eval_scores, eval_output_dir,
                             monitor_details, monitor_name=monitor_name,
                             policy_inputs=policy_inputs)
    logger.log_eval_step(step, eval_metrics, split=split_name)

    eval_all: dict = {}
    eval_all.update(compute_reward_stats([s["reward"] for s in eval_scores]))
    eval_all.update(compute_generation_metrics(
        sequences,
        [cf[0] for cf in cot_finals],
        [cf[1] for cf in cot_finals],
        tokenizer,
        args.max_tokens,
    ))
    eval_all.update(eval_metrics)  # reward/true_reward, reward/misaligned_reward, verify/*, mon/*
    eval_prefix = f"eval/{split_name}" if split_name else "eval"
    log_metrics(
        {f"{eval_prefix}/{k}": v for k, v in eval_all.items()},
        step=step,
        output_dir=args.output_dir,
        use_wandb=bool(args.get("use_wandb", False)),
    )

    # Print up to n_traces_eval full instances to the terminal.
    n_examine = min(int(args.get("n_traces_eval", 3)), len(eval_dataset))
    for i in range(n_examine):
        row = eval_dataset[i]
        score = eval_scores[i]
        print(
            f"⭐️⭐️ [eval step {step} | {split_name or 'eval'} | {args.model_role} | sample {i}]\n"
            f"⭐️⭐️ [query] {str(row.get('query', ''))}\n"
            f"⭐️⭐ [cot] {score.get('cot') or 'EMPTY'}\n"
            f"⭐️⭐️ [final_output] {score.get('final_output') or 'EMPTY'}\n"
            f"⭐️⭐️ [has_valid_output] {int(score.get('has_valid_output', 0))}\n"
            f"⭐️⭐️ [verification]\n{format_diagnostics(verifier_diagnostics(score))}\n"
            f"⭐️⭐️ [reward] {_num(score, 'reward')}  "
            f"⭐️⭐️ [true_reward] {_num(score, 'true_reward')}  "
            f"⭐️⭐️ [misaligned_reward] {_num(score, 'misaligned_reward')}\n"
            + _monitor_trace(monitor_details, i),
            flush=True,
        )
    return eval_scores


def _num(score: dict, key: str, spec: str = ".4f") -> str:
    """
    Format a score field, or EMPTY when absent.
    """
    return format(score[key], spec) if key in score else "EMPTY"


def _monitor_trace(monitor_details: list[dict], i: int) -> str:
    """
    Format the monitor's line for instance i for logging
    """
    if i >= len(monitor_details):
        return ""
    d = monitor_details[i]
    return (f"\n⭐️⭐️ [mon]{' FAILED' if d.get('failed') else ''} "
            f"prediction={d.get('verdict')}  reason: {d.get('reason') or 'EMPTY'}")


async def run_probe_sampling(
    step: int,
    probe_dataset: list[dict],
    sampling_client,
    renderer,
    tokenizer,
    args: DictConfig,
    stop_sequences: list[str],
    n_samples: int,
    out_dir: str | None,
    description: str | None,
) -> list[dict]:
    """
    Sample ``n_samples`` scored rollouts per probe prompt (with ``description`` appended when given) and
    write them to <out_dir>/step_{step:06d}.jsonl.
    """
    print(f"\n[Step {step}] [probe] {n_samples} rollouts on each of {len(probe_dataset)} probe prompts...")
    sampling_params = build_sampling_params(args, stop_sequences, prefix="eval_")
    policy_inputs = [add_description_prompt(build_policy_input(row, args.intent_in_input), description)
                     for row in probe_dataset]
    results = await asyncio.gather(*[
        sampling_client.sample_async(
            prompt=renderer.build_generation_prompt([{"role": "user", "content": content}]),
            num_samples=n_samples, sampling_params=sampling_params)
        for content in policy_inputs])
    rollouts = [(row, policy_input, rollout_index, list(seq.tokens))
                for row, policy_input, result in zip(probe_dataset, policy_inputs, results, strict=True)
                for rollout_index, seq in enumerate(result.sequences)]
    parsed = [parse_cot_final(toks, tokenizer, args.think_start_str, args.think_end_str) for *_, toks in rollouts]
    scores = await asyncio.gather(*[
        score_response(row, model_role=args.model_role, true_reward=args.true_reward,
                       misaligned_reward=args.misaligned_reward, cot=cot, final_output=final_output,
                       verifier_settings=args.get("verifier_settings"))
        for (row, *_), (cot, final_output) in zip(rollouts, parsed, strict=True)])
    records = [{
        "step": step, "rollout_index": rollout_index, "query": str(row["query"]), "policy_input": policy_input,
        "data_source": str(row["data_source"]), "extra_info": _normalize(dict(row["extra_info"])),
        "response": decode_full_text(toks, tokenizer),
        **{k: v for k, v in score.items() if k != "verification"},
    } for (row, policy_input, rollout_index, toks), score in zip(rollouts, scores, strict=True)]
    if out_dir is not None:
        path = Path(out_dir) / f"step_{step:06d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_dump(path, records)
        print(f"Probe rollouts → {path}")
    log_metrics({f"eval/probe/{k}": v for k, v in compute_batch_metrics(scores).items()},
                step=step, output_dir=args.output_dir, use_wandb=bool(args.get("use_wandb", False)))
    return records
