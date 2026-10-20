"""
Score the output
"""

from __future__ import annotations

import src.tasks  # noqa: F401  registers the tasks, rewards and hack labels
from src.data.dataset_utils import _normalize
from src.tasks.registry import get_reward, get_task
from src.tasks.base import CONTROL_MODEL, MISALIGNED_MODEL, MODEL_ROLES, ScoringContext


# Keys score_response produces itself; anything else in a record came from Verifier.diagnostics.
RECORD_KEYS = frozenset({
    "verification", "reward", "true_reward", "misaligned_reward",
    "has_valid_output", "has_harness_error", "cot", "final_output",
})


def verifier_diagnostics(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in RECORD_KEYS}


def usable_for_training(record: dict) -> bool:
    """
    Check whether this rollout has valid reward for training: False iff the verifier failed.
    Only the coding verifier sets this
    """
    return not bool(record["has_harness_error"])


def optimized_reward_name(model_role: str, true_reward: str, misaligned_reward: str) -> str:
    """
    Return the reward this run's GRPO optimizes, by model role.
    """
    if model_role == CONTROL_MODEL:
        return true_reward
    if model_role == MISALIGNED_MODEL:
        return misaligned_reward
    raise ValueError(f"Unknown model_role {model_role!r}. Expected one of {MODEL_ROLES}.")


async def score_response(
    row: dict,
    *,
    cot: str,
    final_output: str,
    model_role: str,
    true_reward: str,
    misaligned_reward: str,
    verifier_settings: dict | None = None,
) -> dict:
    """
    Score one response and return its per-response record.
    """
    # Direct indexing: a row missing these is malformed.
    extra_info: dict = _normalize(dict(row["extra_info"]))
    data_source = str(row["data_source"])
    extra_info["data_source"] = data_source
    query = str(row["query"])

    task = get_task(data_source)

    # Only the final output is verified
    solution = final_output

    # Run-level verifier settings override the row's own values.
    extra_info.update({k: v for k, v in (verifier_settings or {}).items() if v is not None})
    verification = await task.verifier.verify(solution, extra_info)

    ctx = ScoringContext(
        task=task,
        query=query,
        solution=solution,
        extra_info=extra_info,
        data_source=data_source,
    )

    # Both r_t and r_p are logged for both model roles. An
    # ungradable output gets the invalid reward (-1) on both.
    has_valid_output = task.is_valid_output(final_output)
    if has_valid_output:
        r_true = float(await get_reward(true_reward).score(verification, ctx))
        r_misaligned = float(await get_reward(misaligned_reward).score(verification, ctx))
    else:
        r_true = r_misaligned = float(task.invalid_reward)
    optimized = optimized_reward_name(model_role, true_reward, misaligned_reward)
    reward = r_true if optimized == true_reward else r_misaligned

    return {
        "verification": verification,
        "reward": reward,
        "true_reward": r_true,
        "misaligned_reward": r_misaligned,
        "has_valid_output": float(has_valid_output),
        "has_harness_error": float(task.verifier.harness_error(verification)),
        "cot": cot,
        "final_output": final_output,
        **_checked_diagnostics(task, verification, extra_info),
    }


def _checked_diagnostics(task, verification, extra_info: dict) -> dict:
    """
    Return the verifier's diagnostics.
    """
    diagnostics = task.verifier.diagnostics(verification, extra_info)
    clash = RECORD_KEYS & diagnostics.keys()
    if clash:
        raise ValueError(
            f"{type(task.verifier).__name__}.diagnostics returned {sorted(clash)}, which the "
            f"score record already owns ({sorted(RECORD_KEYS)}). Rename them — a collision "
            "would overwrite the framework's value invisibly."
        )
    return diagnostics
