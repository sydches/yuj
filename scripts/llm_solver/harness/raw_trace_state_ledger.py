"""Prefix-only state ledger over raw trace event fields.

This module deliberately avoids slot projection, detector registries, regex
labels, and outcome classification. It consumes only fields already present on
``.trace.jsonl`` / ``session._trace_events`` tool-call records, then accumulates
exact identity counts and simple counters over the visible prefix.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

PROMPT_TOKEN_DENOMINATOR = 62000


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hash_value(value: Any) -> str:
    text = "" if value is None else str(value)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _pair_hash(args_hash: str, result_hash: str) -> str:
    return hashlib.sha256(f"{args_hash}|{result_hash}".encode("utf-8")).hexdigest()[:16]


def _turn_distance(current: int, prior: int | None) -> int:
    if prior is None:
        return 999
    return current - prior


def _paths_count(value: Any) -> int:
    if isinstance(value, list):
        return len([item for item in value if str(item)])
    if isinstance(value, str) and value:
        return 1
    return 0


def _join_turns(values: list[int], cap: int = 12) -> str:
    tail = values[-cap:]
    prefix = "..." if len(values) > cap else ""
    return prefix + ";".join(str(value) for value in tail)


@dataclass
class RawTraceSnapshot:
    turn_number: int
    tool_name: str
    args_hash: str
    result_hash: str
    pair_hash: str
    args_seen_before: int
    result_seen_before: int
    pair_seen_before: int
    args_prev_turns: str
    result_prev_turns: str
    pair_prev_turns: str
    turns_since_same_args: int
    turns_since_same_result: int
    turns_since_same_pair: int
    source_write_count_before: int
    write_count_before: int
    gate_block_count_before: int
    done_count_before: int
    source_writes_since_last_same_args: int
    source_writes_since_last_same_result: int
    source_writes_since_last_same_pair: int
    writes_since_last_same_args: int
    gate_blocks_since_last_same_args: int
    prompt_delta_since_last_same_args: int
    prompt_tokens: int
    prompt_token_ratio_62k: float
    current_source_write_like: bool
    current_write_like: bool
    current_gate_blocked: bool
    current_source_write_path_count: int
    current_done_like: bool

    def to_row(self) -> dict[str, str]:
        row: dict[str, str] = {}
        for key, value in asdict(self).items():
            if isinstance(value, bool):
                row[key] = str(value).lower()
            elif isinstance(value, float):
                row[key] = f"{value:.6f}"
            else:
                row[key] = str(value)
        return row


@dataclass
class RawTraceStateLedger:
    """State accumulated from raw trace events through the current turn."""

    args_turns: dict[str, list[int]] = field(default_factory=dict)
    result_turns: dict[str, list[int]] = field(default_factory=dict)
    pair_turns: dict[str, list[int]] = field(default_factory=dict)
    prompt_by_turn: dict[int, int] = field(default_factory=dict)
    source_write_count_after_turn: dict[int, int] = field(default_factory=dict)
    write_count_after_turn: dict[int, int] = field(default_factory=dict)
    gate_block_count_after_turn: dict[int, int] = field(default_factory=dict)
    source_write_count: int = 0
    write_count: int = 0
    gate_block_count: int = 0
    done_count: int = 0
    last_snapshot: RawTraceSnapshot | None = None

    def update(self, event: dict[str, Any]) -> RawTraceSnapshot | None:
        """Update from one raw trace event.

        Returns a snapshot for ``tool_call`` events and ``None`` for all other
        event types. Future events are never read.
        """
        if event.get("event") != "tool_call":
            return None

        turn = _int_value(event.get("turn_number"), -1)
        if turn < 0:
            return None

        tool_name = str(event.get("tool_name") or "")
        args_summary = str(event.get("args_summary") or "")
        result_summary = str(
            event.get("output_sha256")
            or event.get("result_summary")
            or ""
        )
        args_hash = _hash_value(args_summary)
        result_hash = _hash_value(result_summary)
        pair_hash = _pair_hash(args_hash, result_hash)
        prompt_tokens = _int_value(event.get("prompt_tokens"), 0)

        prev_args = self.args_turns.setdefault(args_hash, [])
        prev_result = self.result_turns.setdefault(result_hash, [])
        prev_pair = self.pair_turns.setdefault(pair_hash, [])
        last_same_args = prev_args[-1] if prev_args else None
        last_same_result = prev_result[-1] if prev_result else None
        last_same_pair = prev_pair[-1] if prev_pair else None

        source_write_like = _truthy(event.get("source_write_like"))
        write_like = _truthy(event.get("write_like"))
        gate_blocked = _truthy(event.get("gate_blocked"))
        done_like = tool_name in {"done", "submit"}

        snapshot = RawTraceSnapshot(
            turn_number=turn,
            tool_name=tool_name,
            args_hash=args_hash,
            result_hash=result_hash,
            pair_hash=pair_hash,
            args_seen_before=len(prev_args),
            result_seen_before=len(prev_result),
            pair_seen_before=len(prev_pair),
            args_prev_turns=_join_turns(prev_args),
            result_prev_turns=_join_turns(prev_result),
            pair_prev_turns=_join_turns(prev_pair),
            turns_since_same_args=_turn_distance(turn, last_same_args),
            turns_since_same_result=_turn_distance(turn, last_same_result),
            turns_since_same_pair=_turn_distance(turn, last_same_pair),
            source_write_count_before=self.source_write_count,
            write_count_before=self.write_count,
            gate_block_count_before=self.gate_block_count,
            done_count_before=self.done_count,
            source_writes_since_last_same_args=self._source_writes_since(last_same_args),
            source_writes_since_last_same_result=self._source_writes_since(last_same_result),
            source_writes_since_last_same_pair=self._source_writes_since(last_same_pair),
            writes_since_last_same_args=self._writes_since(last_same_args),
            gate_blocks_since_last_same_args=self._gate_blocks_since(last_same_args),
            prompt_delta_since_last_same_args=(
                0 if last_same_args is None else prompt_tokens - self.prompt_by_turn.get(last_same_args, 0)
            ),
            prompt_tokens=prompt_tokens,
            prompt_token_ratio_62k=prompt_tokens / PROMPT_TOKEN_DENOMINATOR,
            current_source_write_like=source_write_like,
            current_write_like=write_like,
            current_gate_blocked=gate_blocked,
            current_source_write_path_count=_paths_count(event.get("source_write_paths")),
            current_done_like=done_like,
        )

        if source_write_like:
            self.source_write_count += 1
        if write_like:
            self.write_count += 1
        if gate_blocked:
            self.gate_block_count += 1
        if done_like:
            self.done_count += 1
        self.source_write_count_after_turn[turn] = self.source_write_count
        self.write_count_after_turn[turn] = self.write_count
        self.gate_block_count_after_turn[turn] = self.gate_block_count
        self.prompt_by_turn[turn] = prompt_tokens
        prev_args.append(turn)
        prev_result.append(turn)
        prev_pair.append(turn)
        self.last_snapshot = snapshot
        return snapshot

    def _source_writes_since(self, prior_turn: int | None) -> int:
        if prior_turn is None:
            return self.source_write_count
        return self.source_write_count - self.source_write_count_after_turn.get(prior_turn, 0)

    def _writes_since(self, prior_turn: int | None) -> int:
        if prior_turn is None:
            return self.write_count
        return self.write_count - self.write_count_after_turn.get(prior_turn, 0)

    def _gate_blocks_since(self, prior_turn: int | None) -> int:
        if prior_turn is None:
            return self.gate_block_count
        return self.gate_block_count - self.gate_block_count_after_turn.get(prior_turn, 0)


def replay_events(events: list[dict[str, Any]], through_turn: int | None = None) -> list[RawTraceSnapshot]:
    """Replay raw events in trace order and return per-tool-call snapshots."""
    ledger = RawTraceStateLedger()
    snapshots: list[RawTraceSnapshot] = []
    for event in events:
        turn = _int_value(event.get("turn_number"), -1)
        if through_turn is not None and turn > through_turn:
            continue
        snapshot = ledger.update(event)
        if snapshot is not None:
            snapshots.append(snapshot)
    return snapshots


__all__ = [
    "RawTraceSnapshot",
    "RawTraceStateLedger",
    "replay_events",
]
