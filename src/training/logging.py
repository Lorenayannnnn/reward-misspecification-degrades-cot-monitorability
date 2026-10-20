"""
Run logging: per-step console traces, per-instance evaluation dumps, and metrics.jsonl.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from src.data.dataset_utils import _normalize
from src.utils import write_dump
from src.monitor.monitor_utils import get_monitor_key_name
from src.tasks.scorer import verifier_diagnostics

_log = logging.getLogger(__name__)

LOGGING_TRUNCATION_LENGTH = 400


def log_metrics(metrics: dict, step: int, output_dir: str | None, use_wandb: bool = False) -> None:
    """
    Append a per-step metrics dict to metrics.jsonl and, optionally, wandb.
    """
    metrics = {**metrics, "step": step, "timestamp": time.time()}
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(metrics) + "\n")
    if use_wandb:
        import wandb

        wandb.log(metrics, step=step)


def _truncate(text: str, max_len: int = LOGGING_TRUNCATION_LENGTH) -> str:
    return text if len(text) <= max_len else text[:max_len] + "..."


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {"min": min(values), "max": max(values), "std": math.sqrt(var)}


def format_diagnostics(diagnostics: dict, max_len: int = 200, indent: str = "      ") -> str:
    """
    Render verifier diagnostics for the terminal: scalars as ``key: value``, a list of
    dicts as one indented line per row.
    """
    lines: list[str] = []
    for key, value in (diagnostics or {}).items():
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            lines.append(f"{indent}{key}:")
            for row in value:
                cells = " | ".join(f"{k}: {_truncate(str(v), max_len)}" for k, v in row.items() if v != "")
                lines.append(f"{indent}  {cells}")
        elif not isinstance(value, (list, dict)):
            lines.append(f"{indent}{key}: {_truncate(str(value), max_len)}")
    return "\n".join(lines)


class RunLogger:
    """
    Logs per-sample traces and step summaries to JSONL and stdout.
    """

    def __init__(self, log_dir: str | None, n_traces_per_step: int = 3):
        self.n_traces = n_traces_per_step
        self._file = None
        if log_dir is not None:
            path = Path(log_dir)
            path.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self._file = open(path / f"traces_{ts}.jsonl", "a")
            print(f"Trace log → {path}/traces_{ts}.jsonl")

    def _write(self, record: dict) -> None:
        if self._file is not None:
            self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._file.flush()

    def log_train_step(
        self,
        step: int,
        rows: list[dict],
        all_response_texts: list[list[str]],
        all_score_results: list[list[dict]],
        rewards_per_instance: list[list[float]],
        mean_rewards: list[float],
        n_identical_reward: int,
        n_datums: int,
        batch_metrics: dict[str, float],
    ) -> None:
        all_rewards_flat = [r for group in rewards_per_instance for r in group]
        r_stats = _stats(all_rewards_flat)
        overall_mean = sum(mean_rewards) / len(mean_rewards) if mean_rewards else 0.0
        frac_identical = n_identical_reward / len(mean_rewards) if mean_rewards else 0.0

        # One trace per instance (its first rollout), capped at n_traces.
        traces_printed = 0
        for row, resp_texts, scores, group_rewards in zip(
            rows, all_response_texts, all_score_results, rewards_per_instance, strict=False
        ):
            if traces_printed >= self.n_traces:
                break
            if not resp_texts:
                continue
            resp, score, reward = resp_texts[0], scores[0], group_rewards[0]
            traces_printed += 1
            print(f"  ── trace {traces_printed} ──")
            print(f"  🐛Q: {_truncate(row['query'], 120)}")
            print(f"  🐛CoT: {_truncate(score.get('cot') or '(none)')}")
            print(f"  🐛Final: {_truncate(score.get('final_output') or resp)}")
            print(
                f"  🐛reward={reward:.3f}  r_t={score['true_reward']:.3f}"
                f"  r_p={score['misaligned_reward']:.3f}"
            )
            diag_str = format_diagnostics(verifier_diagnostics(score))
            if diag_str:
                print(f"  🐛verification:\n{diag_str}")

        metric_str = "  ".join(f"{k}={v:.3f}" for k, v in sorted(batch_metrics.items()))
        print(
            f"Step {step:3d} | reward={overall_mean:.3f} "
            f"[{r_stats['min']:.3f}..{r_stats['max']:.3f} σ={r_stats['std']:.3f}] | "
            f"identical_reward={frac_identical:.0%} | datums={n_datums}"
        )
        if metric_str:
            print(f"         {metric_str}")

    def log_eval_instances(
        self,
        step: int,
        rows: list[dict],
        response_texts: list[str],
        score_results: list[dict],
        eval_output_dir: str | None,
        monitor_details: list[dict] | None = None,
        monitor_name: str | None = None,
        policy_inputs: list[str] | None = None,
    ) -> None:
        """
        Write one record per evaluation instance to step_{step:06d}.jsonl: the score record, the monitor's
        ``mon_cot__<monitor_name>_<field>`` keys and, when given, ``policy_input``.
        """
        if eval_output_dir is None:
            return
        path = Path(eval_output_dir)
        path.mkdir(parents=True, exist_ok=True)
        filename = path / f"step_{step:06d}.jsonl"
        entries = []
        policy_inputs = policy_inputs if policy_inputs is not None else [None] * len(rows)
        for i, (row, response_text, score, policy_input) in enumerate(
                zip(rows, response_texts, score_results, policy_inputs, strict=True)):
            if monitor_details and not monitor_name:
                raise ValueError("monitor_name is required when monitor_details are written")
            monitor = {
                get_monitor_key_name(monitor_name, field): value
                for field, value in (monitor_details[i] if monitor_details and i < len(monitor_details) else {}).items()
            }
            entries.append({
                "step": step,
                "query": str(row["query"]),
                **({"policy_input": policy_input} if policy_input is not None else {}),
                # Enough for an offline monitor to rebuild its prompt without the parquet.
                "data_source": str(row["data_source"]),
                "extra_info": _normalize(dict(row["extra_info"])),
                "response": response_text,
                **{k: v for k, v in score.items() if k != "verification"},
                **monitor,
            })
        write_dump(filename, entries)
        print(f"Evaluation instances → {filename}")

    def log_eval_step(self, step: int, eval_metrics: dict[str, float], split: str | None = None) -> None:
        tag = f" [{split}]" if split else ""
        metric_str = "  ".join(f"{k}={v:.3f}" for k, v in sorted(eval_metrics.items()))
        print(f"[Val step {step}]{tag} {metric_str or '(no metrics)'}")
        self._write({"step": step, "type": "eval_summary", **({"split": split} if split else {}), **eval_metrics})

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
