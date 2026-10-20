"""
Checkpoint bookkeeping: training_state.json (the step a relaunched run continues from),
latest_checkpoint.txt (the newest resumable state) and checkpoint_names.txt (every saved state;
a resumed run can append a step twice and the last match wins).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

TRAIN_STATE_FILENAME = "training_state.json"
LATEST_CHECKPOINT_FILENAME = "latest_checkpoint.txt"
CHECKPOINT_NAMES_FILENAME = "checkpoint_names.txt"


def load_training_state(output_dir: str | None) -> dict | None:
    """
    Read training_state.json; None only when the file is absent. An unreadable file raises,
    since None would look like a fresh run.
    """
    if not output_dir:
        return None
    path = os.path.join(output_dir, TRAIN_STATE_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"{path} exists but is unreadable ({e}); restore it or delete it "
                           "together with latest_checkpoint.txt to start fresh.") from e


def save_training_state(output_dir: str | None, step: int) -> None:
    """
    Write training_state.json with the next step to run on resume.
    """
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, TRAIN_STATE_FILENAME)
    with open(path, "w") as f:
        json.dump({"step": step}, f)


def record_checkpoint(output_dir: str, saved_path: str, step: int) -> None:
    """
    Append to checkpoint_names.txt, point latest_checkpoint.txt at it, update the step counter.
    """
    with open(os.path.join(output_dir, CHECKPOINT_NAMES_FILENAME), "a") as f:
        f.write(saved_path + "\n")
    with open(os.path.join(output_dir, LATEST_CHECKPOINT_FILENAME), "w") as f:
        f.write(saved_path + "\n")
    save_training_state(output_dir, step)


def latest_checkpoint_path(output_dir: str) -> str | None:
    """
    Read the path in latest_checkpoint.txt, or None when the file is absent or empty.
    """
    path = os.path.join(output_dir, LATEST_CHECKPOINT_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read().strip() or None


def saved_steps(output_dir: Path) -> set[int]:
    """
    Read every step recorded in checkpoint_names.txt (empty set when the file is absent).
    """
    import re
    path = Path(output_dir) / CHECKPOINT_NAMES_FILENAME
    if not path.exists():
        return set()
    return {int(m.group(1)) for ln in path.read_text().splitlines()
            if (m := re.search(r"step_0*(\d+)", ln))}


def checkpoint_path_for_step(model_dir: Path, step: int) -> str:
    """
    Read the tinker state path for ``step`` from checkpoint_names.txt; exits listing what exists.
    """
    lines = [ln.strip() for ln in (model_dir / CHECKPOINT_NAMES_FILENAME).read_text().splitlines() if ln.strip()]
    wanted = f"step_{step:06d}"
    matches = [ln for ln in lines if ln.endswith(wanted)]
    if not matches:
        have = sorted({ln.rsplit("step_", 1)[-1].lstrip("0") or "0" for ln in lines if "step_" in ln}, key=int)
        raise SystemExit(f"{model_dir.name}: no checkpoint for step {step}. Available: {have}")
    return matches[-1]   # a resumed run can append a step twice; the last save wins
