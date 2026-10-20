"""
Runs the CoT monitor and defines the dump keys for its predictions: mon_cot__<monitor_name>_<field>,
monitor_name = model name with "/" -> "__". A dump can entry hold predictions from several monitors;
scoring against the hack label happens offline.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.tasks.base import render_monitor_prompt
from src.tasks.registry import get_task

# Concurrency for the monitor's API calls; override with MONITOR_MAX_WORKERS.
_DEFAULT_MONITOR_WORKERS = int(os.getenv("MONITOR_MAX_WORKERS", "4"))


def build_monitor_extra_info(row: dict) -> dict:
    ei = dict(row["extra_info"])
    ei["data_source"] = str(row["data_source"])
    return ei


def to_monitor_name(model_name: str) -> str:
    if not model_name:
        raise ValueError("monitor model name is empty")
    return model_name.replace("/", "__")


_KEY_PREFIX = "mon_cot__"


def get_monitor_key_name(monitor_name: str, field: str) -> str:
    return f"{_KEY_PREFIX}{monitor_name}_{field}"


# The fields stored per monitor.
# "verdict" is the dump key that holds the monitor's prediction (1 = hacking flagged)
MONITOR_FIELDS = ("verdict", "reason", "failed", "raw")
_KEY_RE = re.compile(rf"^{_KEY_PREFIX}(?P<monitor_name>.+)_(?P<field>{'|'.join(MONITOR_FIELDS)})$")


def parse_monitor_key(key: str) -> tuple[str, str] | None:
    """
    Parse a monitor key into (monitor_name, field), else None.
    """
    m = _KEY_RE.match(key)
    return (m["monitor_name"], m["field"]) if m else None


def get_monitors_has_run(records: list[dict]) -> list[str]:
    """
    Check whether the monitor has been run on any of the records for resuming
    """
    found: set[str] = set()
    for record in records:
        for key in record:
            parsed = parse_monitor_key(key)
            if parsed and parsed[1] == "verdict":
                found.add(parsed[0])
    return sorted(found)


def run_and_compute_monitor_metrics(
    monitor_backend,
    queries: list[str],
    cots: list[str | None],
    extra_infos: list[dict],
    *,
    has_valid_output: list[float],
    has_harness_error: list[float],
    max_workers: int | None = None,
) -> tuple[dict[str, float], list[dict]]:
    """
    Run the monitor on every rollout's CoT. Returns the label-free ``mon/*`` metrics (flag rate, counts
    scored and excluded) and the per-instance ``details`` (prediction, reason, failed, raw) in input order.
    """
    n = len(queries)
    invalid = np.asarray([not bool(v) for v in has_valid_output])
    harness_err = np.asarray([bool(v) for v in has_harness_error])

    if max_workers is None:
        max_workers = _DEFAULT_MONITOR_WORKERS
    max_workers = max(1, min(max_workers, n or 1))

    def _one(args: tuple) -> tuple[int, str, bool, str]:
        """
        Run one monitor call, always with the task's explicit intent, since the hack label is defined
        against it.
        """
        q, cot, ei = args
        task = get_task(ei["data_source"])
        user_message = render_monitor_prompt(task.intent_string("explicit", ei),
                                             task.task_instruction_heading, q, cot)
        try:
            prediction, reason, failed, raw = monitor_backend.predict_with_reason(user_message)
            return int(prediction), str(reason), bool(failed), str(raw)
        except Exception as e:
            print(f"[warn] Monitor failed: {e}")
            return 0, f"ERROR: {e}", True, ""

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_one, zip(queries, [(c or "") for c in cots], extra_infos)))
    predictions = [v for v, _, _, _ in results]
    failures = [f for _, _, f, _ in results]
    details = [{"verdict": int(v), "reason": rsn, "failed": bool(f), "raw": raw}
               for v, rsn, f, raw in results]
    failed_arr = np.array(failures, dtype=bool)
    excluded = failed_arr | invalid | harness_err
    z = np.array(predictions, dtype=bool)[~excluded]
    metrics: dict[str, float] = {"mon/n_scored": float(z.size)}
    if z.size:
        metrics["mon/flag_rate"] = float(z.mean())
    metrics["mon/call_failure_rate"] = float(failed_arr.mean()) if failures else 0.0
    metrics["mon/n_excluded"] = float(excluded.sum())
    metrics["mon/n_excluded_invalid_output"] = float(invalid.sum())
    metrics["mon/n_excluded_harness_error"] = float((harness_err & ~invalid).sum())
    metrics["mon/n_excluded_monitor_failed"] = float((failed_arr & ~invalid & ~harness_err).sum())
    return metrics, details


def run_monitor_and_store_predictions(rows: list[dict], records_to_run: list[int], backend,
                                   monitor_name: str, max_workers: int | None) -> str:
    """
    Run the monitor over the dump records and merge the predictions into
    ``rows`` in place. Returns a one-line summary for the log.
    """
    metrics, details = run_and_compute_monitor_metrics(
        backend,
        queries=[str(rows[i]["query"]) for i in records_to_run],
        cots=[rows[i].get("cot") for i in records_to_run],
        extra_infos=[build_monitor_extra_info(rows[i]) for i in records_to_run],
        has_valid_output=[float(rows[i]["has_valid_output"]) for i in records_to_run],
        has_harness_error=[float(rows[i]["has_harness_error"]) for i in records_to_run],
        max_workers=max_workers,
    )
    for i, d in zip(records_to_run, details, strict=True):
        for field, value in d.items():
            rows[i][get_monitor_key_name(monitor_name, field)] = value
    return (f"{len(records_to_run)} records flag={metrics.get('mon/flag_rate', float('nan')):.3f} "
            f"failed={int(metrics['mon/n_excluded_monitor_failed'])}")
