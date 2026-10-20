"""
Parquet dataset loading. Nested numpy/Arrow values are converted to plain Python, since an
ndarray left in extra_info would be written to the JSONL dumps as its repr string.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_parquet_dataset(path: str | Path) -> list[dict]:
    """
    Load one preprocessed parquet file as a list of row dicts, with extra_info normalized to plain Python.
    """
    df = pd.read_parquet(str(path))
    rows = df.to_dict(orient="records")
    for row in rows:
        ei = row.get("extra_info")
        if isinstance(ei, dict):
            row["extra_info"] = _normalize(ei)
    return rows


def _normalize(obj: Any) -> Any:
    """
    Convert the numpy arrays and Arrow scalars that parquet loads into built-in lists, ints,
    floats and strings, recursively, so extra_info can be used like ordinary JSON.
    """
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    return obj
