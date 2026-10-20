"""
Batch-level metrics: reward panels, reward stats, generation stats.

Keys are ``<group>/<name>`` (reward / verify / policy / length / format); callers prefix
``train/`` or ``eval/<split>/``. ``_valid`` variants cover only rollouts with a usable output.
"""

from __future__ import annotations

import numpy as np

from src.tasks.scorer import verifier_diagnostics


def compute_batch_metrics(score_results: list[dict]) -> dict[str, float]:
    """
    Compute reward/true_reward, reward/misaligned_reward, format/* rates and verify/* means over a
    batch of score records. The optimized scalar is reported by ``compute_reward_stats``.
    """
    if not score_results:
        return {}

    def _col(key: str) -> np.ndarray:
        return np.array([r[key] for r in score_results], dtype=np.float32)

    true_r = _col("true_reward")
    misaligned_r = _col("misaligned_reward")
    valid = _col("has_valid_output").astype(bool)
    harness_err = _col("has_harness_error").astype(bool)

    metrics: dict[str, float] = {
        "reward/true_reward": float(true_r.mean()),
        "reward/misaligned_reward": float(misaligned_r.mean()),
        "format/valid_output_rate": float(valid.mean()),
        # Grader failures (sandbox down, API error) score 0, so a jump here says a reward
        # collapse is the grader's, not the model's.
        "format/harness_error_rate": float(harness_err.mean()),
    }
    if valid.any():
        metrics["reward/true_reward_valid"] = float(true_r[valid].mean())
        metrics["reward/misaligned_reward_valid"] = float(misaligned_r[valid].mean())

    metrics.update(_verifier_diagnostic_metrics(score_results, valid))
    return metrics


def _verifier_diagnostic_metrics(score_results: list[dict], valid: np.ndarray) -> dict[str, float]:
    """
    Average every numeric verifier diagnostic as ``verify/<key>`` (plus ``_valid``), each
    over the rollouts that have a numeric value for it only.
    """
    keys = {
        k
        for r in score_results
        for k, v in verifier_diagnostics(r).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    out: dict[str, float] = {}
    for key in sorted(keys):
        present = np.array([isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)
                            for r in score_results])
        # 0.0 fills rows without a value; the masks keep it out of every mean.
        vals = np.array([float(r[key]) if p else 0.0 for r, p in zip(score_results, present, strict=True)],
                        dtype=np.float32)
        if present.any():
            out[f"verify/{key}"] = float(vals[present].mean())
        both = present & valid
        if both.any():
            out[f"verify/{key}_valid"] = float(vals[both].mean())
    return out


def compute_reward_stats(
    rewards: list[float],
    groups: list[list[float]] | None = None,
) -> dict[str, float]:
    """
    Compute reward/{mean,min,max,std} of the optimized scalar, plus reward/group_std (the mean
    within-group std, the GRPO learning signal).
    """
    r = np.asarray(rewards, dtype=np.float32)
    if r.size == 0:
        return {}
    metrics = {
        "reward/mean": float(r.mean()),
        "reward/min": float(r.min()),
        "reward/max": float(r.max()),
        "reward/std": float(r.std()),
    }
    if groups:
        stds = [float(np.std(g)) for g in groups if len(g) > 0]
        metrics["reward/group_std"] = float(np.mean(stds)) if stds else 0.0
    return metrics


def compute_generation_metrics(
    sequences: list,
    cots: list[str | None],
    finals: list[str | None],
    tokenizer,
    max_tokens: int,
) -> dict[str, float]:
    """
    Compute entropy, token lengths (response, CoT, final output) and the exhausted-budget rate
    (len(tokens) >= max_tokens) over aligned lists of sampled sequences.
    """
    if not sequences:
        return {}

    entropy, resp_len, cot_len, out_len, trunc = [], [], [], [], []
    for seq, cot, final in zip(sequences, cots, finals, strict=False):
        toks = list(seq.tokens)
        lp = list(seq.logprobs) if getattr(seq, "logprobs", None) else []
        entropy.append(-sum(lp) / len(lp) if lp else 0.0)
        resp_len.append(len(toks))
        cot_len.append(len(tokenizer.encode(cot or "", add_special_tokens=False)))
        out_len.append(len(tokenizer.encode(final or "", add_special_tokens=False)))
        trunc.append(1.0 if len(toks) >= max_tokens else 0.0)

    return {
        "policy/entropy": float(np.mean(entropy)),
        "length/response": float(np.mean(resp_len)),
        "length/response_max": float(np.max(resp_len)),
        "length/response_min": float(np.min(resp_len)),
        "length/cot": float(np.mean(cot_len)),
        "length/cot_max": float(np.max(cot_len)),
        "length/cot_min": float(np.min(cot_len)),
        "length/output": float(np.mean(out_len)),
        "length/output_max": float(np.max(out_len)),
        "length/output_min": float(np.min(out_len)),
        "format/exhaust_budget_rate": float(np.mean(trunc)),
    }
