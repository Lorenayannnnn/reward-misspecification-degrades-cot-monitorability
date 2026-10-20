"""
Offline hack labeling: add ``is_reward_hacking_Y`` to both models' test dumps, per
checkpoint, before monitorability is scored. Only gradable records get a label, and a re-run
reproduces the same Y.
"""

from __future__ import annotations

from pathlib import Path

from src.monitor.monitorability_metrics import LABEL_KEY
from src.utils import write_dump
from src.tasks.base import HackLabel


def has_valid_out_and_no_harness_error(r: dict) -> bool:
    return bool(float(r["has_valid_output"])) and not bool(float(r["has_harness_error"]))


def add_hacking_labels(rows: list[dict], label: HackLabel) -> None:
    """
    Label gradable rows in place; ungradable rows lose any stale label.
    """
    for r in rows:
        if has_valid_out_and_no_harness_error(r):
            r[LABEL_KEY] = float(label.label(r))
        else:
            r.pop(LABEL_KEY, None)


def label_checkpoint(label: HackLabel, control_test: Path, misaligned_test: Path, load) -> None:
    """
    Label both models' test dumps at one step in place.
    """
    for path in (control_test, misaligned_test):
        rows = load(path)
        add_hacking_labels(rows, label)
        write_dump(path, rows)
