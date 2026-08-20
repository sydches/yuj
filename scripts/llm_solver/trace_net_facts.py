"""Pure trace-field facts shared by offline analysis and live detection.

These functions know nothing about gold labels, detector families, routing, or
interventions. They only answer whether one checked mechanical trace fact is
present at the current row.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TraceNetFact:
    evidence_turns: tuple[int, ...]
    occurrences: int = 0
    gap: int = 0


def result_hash(row: dict[str, Any]) -> str:
    """Prefer the execution hash that excludes later harness reminders."""
    return str(
        row.get("execution_output_sha256")
        or row.get("output_sha256")
        or ""
    )


def identical_repeat_plateau_start(
    turns: list[dict[str, Any]],
    idx: int,
) -> TraceNetFact | None:
    if idx < 1:
        return None
    cur = turns[idx]
    prev = turns[idx - 1]
    cur_args = str(cur.get("args_summary") or "")
    cur_hash = result_hash(cur)
    if not cur_args or not cur_hash:
        return None
    if cur_args != str(prev.get("args_summary") or ""):
        return None
    if cur_hash != result_hash(prev):
        return None
    return TraceNetFact((_turn_number(prev), _turn_number(cur)), occurrences=2)


def same_failed_output_repeat(
    turns: list[dict[str, Any]],
    idx: int,
    *,
    min_streak: int,
) -> TraceNetFact | None:
    cur = turns[idx]
    cur_hash = result_hash(cur)
    if not cur_hash or not fail_like(cur) or source_write_like(cur):
        return None
    streak = 1
    evidence = [_turn_number(cur)]
    cursor = idx - 1
    while cursor >= 0:
        prev = turns[cursor]
        if result_hash(prev) != cur_hash or not fail_like(prev):
            break
        streak += 1
        evidence.append(_turn_number(prev))
        cursor -= 1
    if streak < min_streak:
        return None
    return TraceNetFact(tuple(sorted(evidence[-5:])), occurrences=streak)


def same_passing_output_recurrence(
    turns: list[dict[str, Any]],
    idx: int,
    *,
    lookback: int,
    min_prior: int,
    min_gap: int,
) -> TraceNetFact | None:
    cur = turns[idx]
    cur_hash = result_hash(cur)
    if not cur_hash or source_write_like(cur) or fail_like(cur):
        return None
    prior = [
        row
        for row in turns[max(0, idx - lookback):idx]
        if result_hash(row) == cur_hash
        and not fail_like(row)
        and not source_write_like(row)
    ]
    if len(prior) < min_prior:
        return None
    gap = _turn_number(cur) - _turn_number(prior[-1])
    if gap < min_gap:
        return None
    evidence = tuple(_turn_number(row) for row in prior[-4:]) + (_turn_number(cur),)
    return TraceNetFact(evidence, occurrences=len(prior) + 1, gap=gap)


def args_reread_after_gap(
    turns: list[dict[str, Any]],
    idx: int,
    *,
    min_args_len: int,
    min_gap: int,
    max_gap: int,
) -> TraceNetFact | None:
    cur = turns[idx]
    args = str(cur.get("args_summary") or "")
    if len(args) < min_args_len or source_write_like(cur):
        return None
    prev_idx = None
    for cursor in range(idx - 1, -1, -1):
        if str(turns[cursor].get("args_summary") or "") == args:
            prev_idx = cursor
            break
    if prev_idx is None:
        return None
    gap = _turn_number(cur) - _turn_number(turns[prev_idx])
    if gap < min_gap or gap > max_gap:
        return None
    if any(source_write_like(row) for row in turns[prev_idx + 1:idx]):
        return None
    return TraceNetFact(
        (_turn_number(turns[prev_idx]), _turn_number(cur)),
        occurrences=2,
        gap=gap,
    )


def fail_like(row: dict[str, Any]) -> bool:
    status = _parse_int(row.get("exit_status"))
    return (
        str(row.get("pass_fail") or "").lower() == "fail"
        or str(row.get("outcome") or "").lower() == "error"
        or bool(str(row.get("error_class") or "").strip())
        or (status is not None and status != 0)
    )


def source_write_like(row: dict[str, Any]) -> bool:
    if _as_bool(row.get("source_write_like")) or _as_bool(row.get("write_like")):
        return True
    return bool(row.get("source_write_paths"))


def _turn_number(row: dict[str, Any]) -> int:
    value = _parse_int(row.get("_turn_int", row.get("turn_number")))
    if value is None:
        raise ValueError(f"row missing turn number: {row}")
    return value


def _parse_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


__all__ = [
    "TraceNetFact",
    "args_reread_after_gap",
    "identical_repeat_plateau_start",
    "result_hash",
    "same_failed_output_repeat",
    "same_passing_output_recurrence",
]
