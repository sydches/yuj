"""Utility helpers for the LLM hurdle detector."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

TRACE_PACKET_FIELDS = (
    "event",
    "trace_schema_version",
    "session_number",
    "turn_number",
    "tool_name",
    "args_summary",
    "action_summary",
    "action_class",
    "result_summary",
    "output_snippet",
    "output_sha256",
    "output_full_path",
    "output_chars",
    "output_lines",
    "output_truncated",
    "output_retained",
    "outcome",
    "pass_fail",
    "exit_status",
    "error_class",
    "gate_blocked",
    "write_like",
    "source_write_like",
    "source_write_paths",
    "prompt_tokens",
    "completion_tokens",
    "chat_call_ms",
    "token_count_ms",
    "tool_dispatch_ms",
    "trace_write_ms",
    "turn_total_ms",
)




def _trace_prefix_rows(
    trace_events: list[dict[str, Any]],
    *,
    observation_turn: int,
    max_field_chars: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in trace_events:
        if event.get("event") != "tool_call":
            continue
        turn = _int_value(event.get("turn_number"), -1)
        if turn < 0 or turn > observation_turn:
            continue
        row: dict[str, Any] = {}
        for field_name in TRACE_PACKET_FIELDS:
            if field_name not in event:
                continue
            row[field_name] = _packet_value(event[field_name], max_field_chars=max_field_chars)
        rows.append(row)
    return rows


def _packet_value(value: Any, *, max_field_chars: int) -> Any:
    if isinstance(value, str):
        return _truncate(value, max_field_chars)
    if isinstance(value, list):
        return [_packet_value(item, max_field_chars=max_field_chars) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _packet_value(item, max_field_chars=max_field_chars)
            for key, item in value.items()
        }
    return value


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 20:
        return text[:max_chars]
    head = max_chars // 2
    tail = max_chars - head - 20
    return text[:head] + f"\n...[truncated {len(text) - max_chars} chars]...\n" + text[-tail:]


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if text.startswith("{"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1].strip()
    return text


def _json_object_candidates(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    return candidates


def _string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value] if value.strip() else []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list or string")
    return [str(item).strip() for item in value if str(item).strip()]


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "null", "none", "unknown"}:
            return None
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ValueError("new_facts_still_appearing must be true, false, or null")


def _tail(items: list[Any], limit: int) -> list[Any]:
    if limit <= 0:
        return []
    return items[-limit:]


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
