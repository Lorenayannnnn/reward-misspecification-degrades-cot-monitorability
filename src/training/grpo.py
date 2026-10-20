"""
GRPO datum construction and sampling-param building.
"""

from __future__ import annotations

import tinker
import torch
from omegaconf import DictConfig
from tinker import TensorData


def build_sampling_params(
    args: DictConfig,
    stop_sequences: list[str],
    *,
    prefix: str,
    seed: int | None = None,
) -> tinker.SamplingParams:
    """
    Build tinker.SamplingParams from the config's ``<prefix>{temperature,top_p,top_k}`` ("train_" or
    "eval_"). Training rollouts are never seeded: tinker would return identical samples.
    """
    kwargs: dict = {"max_tokens": args.max_tokens, "stop": stop_sequences}
    if seed is not None:
        kwargs["seed"] = seed
    for name in ("temperature", "top_p", "top_k"):
        key = f"{prefix}{name}"
        val = args.get(key)
        assert val is not None, f"Sampling hyperparameter '{key}' must be set explicitly (got None)."
        kwargs[name] = val
    return tinker.SamplingParams(**kwargs)


def build_datums(
    prompts: list,
    tokens_per_prompt: list[list[list[int]]],
    logprobs_per_prompt: list[list[list[float]]],
    rewards_per_prompt: list[list[float]],
    usable_per_prompt: list[list[bool]] | None = None,
) -> tuple[list[tinker.Datum], list[float], int, int]:
    """
    Turn within-group advantages into tinker Datums. Returns (datums, mean_reward_per_group,
    n_identical_reward_groups, n_unusable_dropped); unusable rollouts get no datum, and a group with
    fewer than two usable rollouts or identical rewards is skipped.
    """
    datums: list[tinker.Datum] = []
    mean_rewards: list[float] = []
    n_identical_reward = 0
    n_dropped = 0
    if usable_per_prompt is None:
        usable_per_prompt = [[True] * len(r) for r in rewards_per_prompt]

    for prompt, group_tokens, group_logprobs, rewards, usable in zip(
            prompts, tokens_per_prompt, logprobs_per_prompt, rewards_per_prompt, usable_per_prompt, strict=True):
        kept = [(t, lp, r) for t, lp, r, u in zip(group_tokens, group_logprobs, rewards, usable, strict=True) if u]
        n_dropped += len(rewards) - len(kept)
        if len(kept) < 2:
            continue
        kept_rewards = [r for _, _, r in kept]
        group_mean = sum(kept_rewards) / len(kept_rewards)
        mean_rewards.append(group_mean)
        advantages = [r - group_mean for r in kept_rewards]

        if all(a == 0.0 for a in advantages):
            n_identical_reward += 1
            continue

        ob_len = prompt.length - 1
        for (tokens, logprobs, _), adv in zip(kept, advantages, strict=True):
            tokens, logprobs = list(tokens), list(logprobs)
            # A stripped rollout whose logprobs were not recomputed would misalign every ratio.
            if len(logprobs) != len(tokens):
                raise ValueError(f"rollout has {len(tokens)} tokens but {len(logprobs)} logprobs")
            model_input = prompt.append(tinker.EncodedTextChunk(tokens=tokens[:-1]))
            datums.append(tinker.Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(
                        torch.tensor([0] * ob_len + tokens)
                    ),
                    "logprobs": TensorData.from_torch(
                        torch.tensor([0.0] * ob_len + logprobs)
                    ),
                    "advantages": TensorData.from_torch(
                        torch.tensor([0.0] * ob_len + [adv] * len(tokens))
                    ),
                },
            ))

    return datums, mean_rewards, n_identical_reward, n_dropped
