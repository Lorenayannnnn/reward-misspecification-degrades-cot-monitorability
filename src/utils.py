"""
CoT / final-output splitting and JSONL dump I/O. The split is token-level (the markers are located
in the sampled token ids), so it survives tokenizer round-trips and truncation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _find_subsequence(seq: list[int], sub: list[int]) -> list[int]:
    """
    Find the start indices of every occurrence of ``sub`` in ``seq``.
    """
    if not sub or len(sub) > len(seq):
        return []
    m = len(sub)
    return [i for i in range(len(seq) - m + 1) if seq[i:i + m] == sub]


def parse_out_main_cot_output(
    token_ids: list[int],
    tokenizer,
    think_start_str: str,
    think_end_str: str,
) -> tuple[str, str | None, int, int | None, int | None, int | None, int, int]:
    """
    Split the CoT from the final output at the token level. Returns ``(main_cot, final_output, cot_start,
    cot_end, open_pos, close_pos, open_len, close_len)``; ``final_output`` is None when the CoT never closes.
    """
    ids = list(token_ids)
    open_ids = tokenizer.encode(think_start_str, add_special_tokens=False)
    close_ids = tokenizer.encode(think_end_str, add_special_tokens=False)
    # An empty marker would make every response all CoT.
    if not open_ids or not close_ids:
        raise ValueError(
            f"CoT markers must encode to >=1 token: start={think_start_str!r} -> {open_ids}, "
            f"end={think_end_str!r} -> {close_ids}"
        )
    open_len, close_len = len(open_ids), len(close_ids)

    open_positions = _find_subsequence(ids, open_ids)
    close_positions = _find_subsequence(ids, close_ids)

    if open_positions:
        open_pos = open_positions[0]
        cot_start = open_pos + open_len
    else:
        # No opening marker (it may sit in the prompt): the CoT starts at position 0.
        open_pos = None
        cot_start = 0

    closes_after = [p for p in close_positions if p >= cot_start]

    def _dec(a: int, b: int | None) -> str:
        return tokenizer.decode(ids[a:b], skip_special_tokens=True)

    if closes_after:
        cot_end = close_pos = closes_after[0]
        main_cot = _dec(cot_start, cot_end)
        final_output = _dec(cot_end + close_len, None)
    else:
        # No closing marker (truncated): everything from cot_start on is CoT.
        cot_end = close_pos = None
        main_cot = _dec(cot_start, None)
        final_output = None

    return main_cot, final_output, cot_start, cot_end, open_pos, close_pos, open_len, close_len


def parse_cot_final(
    tokens: list[int],
    tokenizer,
    think_start: str,
    think_end: str,
) -> tuple[str, str]:
    """
    Split (cot, final_output); a missing final output maps to "".
    """
    cot, final_output, *_ = parse_out_main_cot_output(list(tokens), tokenizer, think_start, think_end)
    return cot, (final_output if final_output is not None else "")


def load_dump(path: Path) -> list[dict]:
    """
    Read a JSONL dump into a list of records.
    """
    # split("\n"), not splitlines(): a response can contain U+2028, which splitlines() breaks on.
    return [json.loads(ln) for ln in Path(path).read_text().split("\n") if ln.strip()]


def write_dump(path: Path, rows: list[dict]) -> None:
    """
    Write records as JSONL via a temp file and os.replace, since callers rewrite dumps in
    place. ensure_ascii escapes U+2028/U+0085, which line-oriented tools read as line breaks.
    """
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text("".join(json.dumps(r, ensure_ascii=True, default=str) + "\n" for r in rows))
    os.replace(tmp_path, path)
