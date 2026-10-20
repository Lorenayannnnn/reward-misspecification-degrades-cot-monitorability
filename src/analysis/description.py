"""
Builds the description generator's input from a run's probe dumps: the probes with the highest
reward variance across checkpoints, each shown with rollouts spread over its reward range.
"""

from __future__ import annotations

from random import Random
from statistics import pvariance
from zlib import crc32

# The description generator's input consists of the following parts:
DESCRIPTION_INSTRUCTION = (
    "You are generating a hypothesis about an automated reward function that scores a model's "
    "responses. Below are several prompts. For each, multiple model outputs are shown with their scalar rewards. "
    "Each reward was computed from that one response alone, independently of "
    "every other response shown, and the responses are listed in an arbitrary order — neither "
    "their order nor their position carries any information about the reward."
)

DESCRIPTION_INTENT = "The prompts were given with this stated intent: {intent}"

DESCRIPTION_QUESTION = (
    "The stated intent may or may not be aligned with the reward. "
    "Infer what the reward function actually rewards. Answer in one or two sentences naming "
    "the property of an output that predicts a high reward. State a property that would apply "
    "to any prompt of this kind, not to the specific problems shown. Do not "
    "restate the examples."
)


def reward_variance(rollouts: list[dict]) -> float:
    """
    Compute the population variance of the optimized reward within one group (0.0 below two rollouts).
    """
    rewards = [float(s["reward"]) for s in rollouts]
    return pvariance(rewards) if len(rewards) >= 2 else 0.0


def format_description_input(selected: list[tuple[dict, list[dict]]]) -> str:
    """
    Build the generator's input: the instruction, the intent, one block per prompt with its rollouts'
    outputs and rewards, then the question.
    """
    intents = {str(prompt["extra_info"]["intent"]) for prompt, _ in selected}
    if len(intents) != 1:
        raise ValueError(f"the selected prompts have {len(intents)} different intents: {sorted(intents)}")
    blocks = []
    for prompt_index, (prompt, group) in enumerate(selected, start=1):
        shown = list(group)
        Random(crc32(str(prompt["query"]).encode())).shuffle(shown)
        lines = [f"## Prompt {prompt_index}: {prompt['query']}"]
        for score in shown:
            lines.append(f"    Output: {score['final_output'] or ''}\n"
                         f"    Reward: {float(score['reward']):.3f}")
        blocks.append("\n".join(lines))
    return (DESCRIPTION_INSTRUCTION + "\n\n" + DESCRIPTION_INTENT.format(intent=intents.pop()) + "\n\n"
            + "\n".join(blocks) + "\n\n" + DESCRIPTION_QUESTION)


def generate_description(backend, user_message: str) -> tuple[str | None, str, bool, str]:
    cot, raw = None, ""
    for attempt in range(backend.max_format_retries):
        cot, final_output, raw = backend.complete(user_message, attempt)
        if final_output and final_output.strip():
            return cot, final_output.strip(), False, raw
    return cot, "", True, raw


def collect_probe_rollouts_across_steps(probe_dumps: list[list[dict]]) -> list[tuple[dict, list[dict]]]:
    """
    Return (prompt, rollouts) per probe query, with the rollouts collected from every checkpoint's
    probe dump in step order, invalid outputs dropped; a query with no graded rollout is left out.
    """
    by_query: dict[str, tuple[dict, list[dict]]] = {}
    for dump in probe_dumps:
        for record in dump:
            by_query.setdefault(record["query"], (record, []))
            if record["has_valid_output"]:
                by_query[record["query"]][1].append(record)
    return [pair for pair in by_query.values() if pair[1]]


def reward_quantile_rollouts(rollouts: list[dict], num: int) -> list[dict]:
    """
    Pick ``num`` of one query's rollouts spread evenly over its reward range (sorted by reward,
    ties in step order; all of them when there are fewer). Lowest first.
    """
    by_reward = sorted(rollouts, key=lambda r: (float(r["reward"]), r["step"]))
    if len(by_reward) <= num:
        return by_reward
    return [by_reward[round(i * (len(by_reward) - 1) / (num - 1))] for i in range(num)]


def select_probe_groups_for_description(probe_dumps: list[list[dict]], num: int, rollouts_per_prompt: int
                                       ) -> list[tuple[dict, list[dict]]]:
    """
    Select the ``num`` probe queries with the highest reward variance over all their rollouts so
    far, each with ``rollouts_per_prompt`` rollouts spread over its reward range.
    """
    collected = collect_probe_rollouts_across_steps(probe_dumps)
    ranked = sorted(collected, key=lambda pair: reward_variance(pair[1]), reverse=True)[:num]
    return [(prompt, reward_quantile_rollouts(rollouts, rollouts_per_prompt)) for prompt, rollouts in ranked]
