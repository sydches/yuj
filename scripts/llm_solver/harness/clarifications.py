"""Durable records for one assistant clarification exchange.

The request, operator answer, and model delivery are separate evidence
records.  This module owns their validation and atomic persistence; the
assistant CLI and model loop own policy and transitions.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


CLARIFICATION_SCHEMA_VERSION = 1
_REQUEST_FILE = "clarification_request.json"
_ANSWER_FILE = "clarification_answer.json"
_CONSUMPTION_FILE = "clarification_consumption.json"
_DELIVERIES = frozenset({"resume"})


class ClarificationStateError(ValueError):
    """A clarification transition or durable record is invalid."""


@dataclass(frozen=True)
class ClarificationState:
    """Validated projection of the three clarification evidence records."""

    phase: Literal["none", "input_required", "input_ready", "consumed", "rewound"]
    request: dict | None = None
    answer: dict | None = None
    consumption: dict | None = None


def clarification_request_path(artifact_dir: Path) -> Path:
    return Path(artifact_dir) / _REQUEST_FILE


def clarification_answer_path(artifact_dir: Path) -> Path:
    return Path(artifact_dir) / _ANSWER_FILE


def clarification_consumption_path(artifact_dir: Path) -> Path:
    return Path(artifact_dir) / _CONSUMPTION_FILE


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _answer_sha256(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_json(path: Path, payload: Mapping[str, object], *, new: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if new:
            try:
                os.link(temp, path)
            except FileExistsError as exc:
                raise ClarificationStateError(
                    f"clarification record already exists: {path.name}"
                ) from exc
            temp.unlink()
        else:
            os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _load_object(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClarificationStateError(
            f"invalid clarification record {path.name}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ClarificationStateError(
            f"invalid clarification record {path.name}: expected an object"
        )
    return payload


def _require_exact_keys(
    payload: Mapping[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    label: str,
) -> None:
    missing = required - payload.keys()
    extra = payload.keys() - required - optional
    if missing:
        raise ClarificationStateError(
            f"invalid {label}: missing fields {', '.join(sorted(missing))}"
        )
    if extra:
        raise ClarificationStateError(
            f"invalid {label}: unexpected fields {', '.join(sorted(extra))}"
        )


def _require_nonempty_string(payload: Mapping[str, object], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ClarificationStateError(f"invalid {label}: {key} must be a non-empty string")
    return value


def _require_int(payload: Mapping[str, object], key: str, label: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ClarificationStateError(
            f"invalid {label}: {key} must be a non-negative integer"
        )
    return value


def _validate_request(payload: dict) -> dict:
    label = "clarification request"
    optional = frozenset({"rewind_id", "rewind_to_turn", "superseded_at"})
    _require_exact_keys(
        payload,
        required=frozenset({
            "schema_version", "record_type", "request_id", "session_id",
            "session_number", "turn_number", "tool_call_id", "question",
            "status", "requested_at",
        }),
        optional=optional,
        label=label,
    )
    if payload["schema_version"] != CLARIFICATION_SCHEMA_VERSION:
        raise ClarificationStateError(f"invalid {label}: unsupported schema_version")
    if payload["record_type"] != "clarification_request":
        raise ClarificationStateError(f"invalid {label}: wrong record_type")
    for key in ("request_id", "session_id", "tool_call_id", "question", "requested_at"):
        _require_nonempty_string(payload, key, label)
    _require_int(payload, "session_number", label)
    _require_int(payload, "turn_number", label)
    if payload["status"] not in {"pending", "rewound"}:
        raise ClarificationStateError(f"invalid {label}: unsupported status")
    if payload["status"] == "rewound":
        for key in ("rewind_id", "superseded_at"):
            _require_nonempty_string(payload, key, label)
        _require_int(payload, "rewind_to_turn", label)
    elif any(key in payload for key in optional):
        raise ClarificationStateError(f"invalid {label}: pending record has rewind fields")
    return payload


def _validate_answer(payload: dict) -> dict:
    label = "clarification answer"
    _require_exact_keys(
        payload,
        required=frozenset({
            "schema_version", "record_type", "request_id", "session_id",
            "answer", "answer_sha256", "recorded_at",
        }),
        label=label,
    )
    if payload["schema_version"] != CLARIFICATION_SCHEMA_VERSION:
        raise ClarificationStateError(f"invalid {label}: unsupported schema_version")
    if payload["record_type"] != "clarification_answer":
        raise ClarificationStateError(f"invalid {label}: wrong record_type")
    for key in ("request_id", "session_id", "answer", "answer_sha256", "recorded_at"):
        _require_nonempty_string(payload, key, label)
    if payload["answer_sha256"] != _answer_sha256(payload["answer"]):
        raise ClarificationStateError(f"invalid {label}: answer_sha256 does not match answer")
    return payload


def _validate_consumption(payload: dict) -> dict:
    label = "clarification consumption"
    _require_exact_keys(
        payload,
        required=frozenset({
            "schema_version", "record_type", "request_id", "answer_sha256",
            "session_number", "turn_number", "delivery", "consumed_at",
        }),
        label=label,
    )
    if payload["schema_version"] != CLARIFICATION_SCHEMA_VERSION:
        raise ClarificationStateError(f"invalid {label}: unsupported schema_version")
    if payload["record_type"] != "clarification_consumption":
        raise ClarificationStateError(f"invalid {label}: wrong record_type")
    for key in ("request_id", "answer_sha256", "consumed_at"):
        _require_nonempty_string(payload, key, label)
    _require_int(payload, "session_number", label)
    _require_int(payload, "turn_number", label)
    if payload["delivery"] not in _DELIVERIES:
        raise ClarificationStateError(f"invalid {label}: unsupported delivery")
    return payload


def load_clarification_request(artifact_dir: Path) -> dict | None:
    payload = _load_object(clarification_request_path(artifact_dir))
    return None if payload is None else _validate_request(payload)


def load_clarification_answer(artifact_dir: Path) -> dict | None:
    payload = _load_object(clarification_answer_path(artifact_dir))
    return None if payload is None else _validate_answer(payload)


def load_clarification_consumption(artifact_dir: Path) -> dict | None:
    payload = _load_object(clarification_consumption_path(artifact_dir))
    return None if payload is None else _validate_consumption(payload)


def clarification_state(artifact_dir: Path) -> ClarificationState:
    request = load_clarification_request(artifact_dir)
    answer = load_clarification_answer(artifact_dir)
    consumption = load_clarification_consumption(artifact_dir)
    if request is None:
        if answer is not None or consumption is not None:
            raise ClarificationStateError("clarification evidence has no request")
        return ClarificationState("none")
    request_id = request["request_id"]
    if answer is not None:
        if answer["request_id"] != request_id:
            raise ClarificationStateError("clarification answer does not match request")
        if answer["session_id"] != request["session_id"]:
            raise ClarificationStateError("clarification answer belongs to another session")
    if consumption is not None:
        if answer is None:
            raise ClarificationStateError("clarification consumption has no answer")
        if consumption["request_id"] != request_id:
            raise ClarificationStateError("clarification consumption does not match request")
        if consumption["answer_sha256"] != answer["answer_sha256"]:
            raise ClarificationStateError("clarification consumption does not match answer")
    if request["status"] == "rewound":
        if consumption is not None:
            raise ClarificationStateError("consumed clarification cannot be rewound")
        return ClarificationState("rewound", request, answer, None)
    if answer is None:
        return ClarificationState("input_required", request, None, None)
    if consumption is None:
        return ClarificationState("input_ready", request, answer, None)
    return ClarificationState("consumed", request, answer, consumption)


def create_clarification_request(
    artifact_dir: Path,
    *,
    request_id: str,
    session_id: str,
    session_number: int,
    turn_number: int,
    tool_call_id: str,
    question: str,
) -> dict:
    if clarification_state(artifact_dir).phase != "none":
        raise ClarificationStateError("this session already has a clarification request")
    payload = _validate_request({
        "schema_version": CLARIFICATION_SCHEMA_VERSION,
        "record_type": "clarification_request",
        "request_id": request_id,
        "session_id": session_id,
        "session_number": session_number,
        "turn_number": turn_number,
        "tool_call_id": tool_call_id,
        "question": question,
        "status": "pending",
        "requested_at": _timestamp(),
    })
    _atomic_write_json(clarification_request_path(artifact_dir), payload, new=True)
    return payload


def record_clarification_answer(
    artifact_dir: Path,
    *,
    session_id: str,
    request_id: str,
    answer: str,
) -> dict:
    request = load_clarification_request(artifact_dir)
    if request is None or request["status"] != "pending":
        raise ClarificationStateError("session has no pending clarification")
    existing = load_clarification_answer(artifact_dir)
    if existing is not None:
        raise ClarificationStateError("clarification already has an answer")
    if request["session_id"] != session_id:
        raise ClarificationStateError("clarification belongs to another session")
    if request["request_id"] != request_id:
        raise ClarificationStateError("clarification request id does not match")
    payload = _validate_answer({
        "schema_version": CLARIFICATION_SCHEMA_VERSION,
        "record_type": "clarification_answer",
        "request_id": request_id,
        "session_id": session_id,
        "answer": answer,
        "answer_sha256": _answer_sha256(answer),
        "recorded_at": _timestamp(),
    })
    _atomic_write_json(clarification_answer_path(artifact_dir), payload, new=True)
    return payload


def consume_clarification_answer(
    artifact_dir: Path,
    *,
    request_id: str,
    session_number: int,
    turn_number: int,
    delivery: Literal["resume"],
) -> dict:
    state = clarification_state(artifact_dir)
    if state.phase == "rewound":
        raise ClarificationStateError("clarification was rewound")
    if state.phase == "consumed":
        raise ClarificationStateError("clarification answer was already consumed")
    if state.phase != "input_ready" or state.request is None or state.answer is None:
        raise ClarificationStateError("clarification has no recorded answer to consume")
    if state.request["request_id"] != request_id:
        raise ClarificationStateError("clarification request id does not match")
    payload = _validate_consumption({
        "schema_version": CLARIFICATION_SCHEMA_VERSION,
        "record_type": "clarification_consumption",
        "request_id": request_id,
        "answer_sha256": state.answer["answer_sha256"],
        "session_number": session_number,
        "turn_number": turn_number,
        "delivery": delivery,
        "consumed_at": _timestamp(),
    })
    _atomic_write_json(clarification_consumption_path(artifact_dir), payload, new=True)
    return payload


def supersede_clarification_for_rewind(
    artifact_dir: Path,
    *,
    rewind_id: str,
    to_turn: int,
) -> dict | None:
    state = clarification_state(artifact_dir)
    if state.phase in {"none", "consumed", "rewound"}:
        return None
    assert state.request is not None
    payload = dict(state.request)
    payload.update({
        "status": "rewound",
        "rewind_id": rewind_id,
        "rewind_to_turn": to_turn,
        "superseded_at": _timestamp(),
    })
    _validate_request(payload)
    _atomic_write_json(clarification_request_path(artifact_dir), payload, new=False)
    return payload


__all__ = [
    "CLARIFICATION_SCHEMA_VERSION",
    "ClarificationState",
    "ClarificationStateError",
    "clarification_answer_path",
    "clarification_consumption_path",
    "clarification_request_path",
    "clarification_state",
    "consume_clarification_answer",
    "create_clarification_request",
    "load_clarification_answer",
    "load_clarification_consumption",
    "load_clarification_request",
    "record_clarification_answer",
    "supersede_clarification_for_rewind",
]
