"""Durable evidence for one paused-session operator correction.

The correction and its one delivery attempt use separate records. This
module owns exact validation and atomic persistence. The assistant CLI owns
creation policy, and the model loop owns delivery at the next turn boundary.
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


CORRECTION_SCHEMA_VERSION = 1
_CORRECTION_FILE = "correction.json"
_CONSUMPTION_FILE = "correction_consumption.json"
_DELIVERIES = frozenset({"resume"})


class CorrectionStateError(ValueError):
    """A correction transition or durable record is invalid."""


@dataclass(frozen=True)
class CorrectionState:
    """Validated projection of correction and consumption evidence."""

    phase: Literal["none", "pending", "consumed"]
    correction: dict | None = None
    consumption: dict | None = None


def correction_path(artifact_dir: Path) -> Path:
    return Path(artifact_dir) / _CORRECTION_FILE


def correction_consumption_path(artifact_dir: Path) -> Path:
    return Path(artifact_dir) / _CONSUMPTION_FILE


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_new(path: Path, payload: Mapping[str, object]) -> None:
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
        try:
            os.link(temp, path)
        except FileExistsError as exc:
            raise CorrectionStateError(
                f"correction record already exists: {path.name}"
            ) from exc
        temp.unlink()
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
        raise CorrectionStateError(
            f"invalid correction record {path.name}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CorrectionStateError(
            f"invalid correction record {path.name}: expected an object"
        )
    return payload


def _require_exact_keys(
    payload: Mapping[str, object],
    *,
    required: frozenset[str],
    label: str,
) -> None:
    missing = required - payload.keys()
    extra = payload.keys() - required
    if missing:
        raise CorrectionStateError(
            f"invalid {label}: missing fields {', '.join(sorted(missing))}"
        )
    if extra:
        raise CorrectionStateError(
            f"invalid {label}: unexpected fields {', '.join(sorted(extra))}"
        )


def _require_nonempty_string(
    payload: Mapping[str, object], key: str, label: str
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise CorrectionStateError(
            f"invalid {label}: {key} must be a non-empty string"
        )
    return value


def _require_int(payload: Mapping[str, object], key: str, label: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CorrectionStateError(
            f"invalid {label}: {key} must be a non-negative integer"
        )
    return value


def _validate_correction(payload: dict) -> dict:
    label = "correction"
    _require_exact_keys(
        payload,
        required=frozenset({
            "schema_version",
            "record_type",
            "correction_id",
            "session_id",
            "after_session_number",
            "text",
            "text_sha256",
            "status",
            "created_at",
        }),
        label=label,
    )
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != CORRECTION_SCHEMA_VERSION
    ):
        raise CorrectionStateError(
            f"invalid {label}: unsupported schema_version"
        )
    if payload["record_type"] != "correction":
        raise CorrectionStateError(f"invalid {label}: wrong record_type")
    for key in (
        "correction_id",
        "session_id",
        "text",
        "text_sha256",
        "created_at",
    ):
        _require_nonempty_string(payload, key, label)
    after_session_number = _require_int(
        payload, "after_session_number", label
    )
    if after_session_number == 0:
        raise CorrectionStateError(
            f"invalid {label}: after_session_number must be positive"
        )
    if payload["status"] != "pending":
        raise CorrectionStateError(f"invalid {label}: unsupported status")
    if payload["text_sha256"] != _text_sha256(payload["text"]):
        raise CorrectionStateError(
            f"invalid {label}: text_sha256 does not match text"
        )
    return payload


def _validate_consumption(payload: dict) -> dict:
    label = "correction consumption"
    _require_exact_keys(
        payload,
        required=frozenset({
            "schema_version",
            "record_type",
            "correction_id",
            "session_id",
            "text_sha256",
            "session_number",
            "turn_number",
            "transcript_segment",
            "delivery",
            "consumed_at",
        }),
        label=label,
    )
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != CORRECTION_SCHEMA_VERSION
    ):
        raise CorrectionStateError(
            f"invalid {label}: unsupported schema_version"
        )
    if payload["record_type"] != "correction_consumption":
        raise CorrectionStateError(f"invalid {label}: wrong record_type")
    for key in (
        "correction_id",
        "session_id",
        "text_sha256",
        "consumed_at",
    ):
        _require_nonempty_string(payload, key, label)
    _require_int(payload, "session_number", label)
    _require_int(payload, "turn_number", label)
    transcript_segment = _require_int(
        payload, "transcript_segment", label
    )
    if transcript_segment == 0:
        raise CorrectionStateError(
            f"invalid {label}: transcript_segment must be positive"
        )
    if (
        not isinstance(payload["delivery"], str)
        or payload["delivery"] not in _DELIVERIES
    ):
        raise CorrectionStateError(f"invalid {label}: unsupported delivery")
    return payload


def load_correction(artifact_dir: Path) -> dict | None:
    payload = _load_object(correction_path(artifact_dir))
    return None if payload is None else _validate_correction(payload)


def load_correction_consumption(artifact_dir: Path) -> dict | None:
    payload = _load_object(correction_consumption_path(artifact_dir))
    return None if payload is None else _validate_consumption(payload)


def correction_state(artifact_dir: Path) -> CorrectionState:
    correction = load_correction(artifact_dir)
    consumption = load_correction_consumption(artifact_dir)
    if correction is None:
        if consumption is not None:
            raise CorrectionStateError(
                "correction consumption has no correction"
            )
        return CorrectionState("none")
    if consumption is None:
        return CorrectionState("pending", correction, None)
    if consumption["correction_id"] != correction["correction_id"]:
        raise CorrectionStateError(
            "correction consumption does not match correction"
        )
    if consumption["session_id"] != correction["session_id"]:
        raise CorrectionStateError(
            "correction consumption belongs to another session"
        )
    if consumption["text_sha256"] != correction["text_sha256"]:
        raise CorrectionStateError(
            "correction consumption does not match text"
        )
    if consumption["session_number"] <= correction["after_session_number"]:
        raise CorrectionStateError(
            "correction consumption does not follow its stopped session"
        )
    if consumption["turn_number"] != 0:
        raise CorrectionStateError(
            "correction consumption is not the first resumed turn"
        )
    return CorrectionState("consumed", correction, consumption)


def validate_correction_trace(artifact_dir: Path) -> CorrectionState:
    """Require correction files and append-only trace events to agree."""
    artifact_dir = Path(artifact_dir)
    state = correction_state(artifact_dir)
    trace_path = artifact_dir / ".trace.jsonl"
    events: list[dict] = []
    if trace_path.is_file():
        try:
            for line_number, line in enumerate(
                trace_path.read_text().splitlines(), start=1
            ):
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise CorrectionStateError(
                        f"invalid correction trace line {line_number}: "
                        "expected an object"
                    )
                if str(event.get("event") or "").startswith("correction_"):
                    events.append(event)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorrectionStateError(
                f"invalid correction trace evidence: {exc}"
            ) from exc
    if state.phase == "none":
        if events:
            raise CorrectionStateError(
                "correction trace events have no durable correction"
            )
        return state

    unknown_events = [
        event
        for event in events
        if event.get("event") not in {
            "correction_created",
            "correction_consumed",
            "correction_replayed",
        }
    ]
    if unknown_events:
        raise CorrectionStateError(
            "correction trace contains an unknown correction event"
        )

    assert state.correction is not None
    correction = state.correction
    created = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("event") == "correction_created"
    ]
    consumed = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("event") == "correction_consumed"
    ]
    replayed = [
        event for event in events if event.get("event") == "correction_replayed"
    ]
    if len(created) != 1:
        raise CorrectionStateError(
            "correction requires exactly one creation trace event"
        )
    if replayed:
        raise CorrectionStateError(
            "live correction evidence cannot contain replay trace events"
        )
    created_index, created_event = created[0]
    if (
        type(created_event.get("session_number")) is not int
        or created_event.get("session_number")
        != correction["after_session_number"]
        or created_event.get("correction_id") != correction["correction_id"]
        or created_event.get("text_sha256") != correction["text_sha256"]
        or type(created_event.get("text_chars")) is not int
        or created_event.get("text_chars") != len(correction["text"])
    ):
        raise CorrectionStateError(
            "correction file and creation trace event do not match"
        )
    if state.phase == "pending":
        if consumed:
            raise CorrectionStateError(
                "pending correction has a consumption trace event"
            )
        return state

    assert state.consumption is not None
    consumption = state.consumption
    if len(consumed) != 1:
        raise CorrectionStateError(
            "consumed correction requires exactly one consumption trace event"
        )
    consumed_index, consumed_event = consumed[0]
    if consumed_index <= created_index:
        raise CorrectionStateError(
            "correction consumption trace event precedes creation"
        )
    if (
        type(consumed_event.get("session_number")) is not int
        or consumed_event.get("session_number") != consumption["session_number"]
        or type(consumed_event.get("turn_number")) is not int
        or consumed_event.get("turn_number") != consumption["turn_number"]
        or type(consumed_event.get("transcript_segment")) is not int
        or consumed_event.get("transcript_segment")
        != consumption["transcript_segment"]
        or consumed_event.get("correction_id") != correction["correction_id"]
        or consumed_event.get("text_sha256") != correction["text_sha256"]
        or consumed_event.get("delivery") != consumption["delivery"]
    ):
        raise CorrectionStateError(
            "correction consumption file and trace event do not match"
        )
    return state


def create_correction(
    artifact_dir: Path,
    *,
    correction_id: str,
    session_id: str,
    after_session_number: int,
    text: str,
) -> dict:
    state = correction_state(artifact_dir)
    if state.phase == "pending":
        raise CorrectionStateError("session already has a pending correction")
    if state.phase == "consumed":
        raise CorrectionStateError("this session already has a correction")
    payload = _validate_correction({
        "schema_version": CORRECTION_SCHEMA_VERSION,
        "record_type": "correction",
        "correction_id": correction_id,
        "session_id": session_id,
        "after_session_number": after_session_number,
        "text": text,
        "text_sha256": _text_sha256(text),
        "status": "pending",
        "created_at": _timestamp(),
    })
    _atomic_write_new(correction_path(artifact_dir), payload)
    return payload


def consume_correction(
    artifact_dir: Path,
    *,
    correction_id: str,
    session_number: int,
    turn_number: int,
    delivery: Literal["resume"],
) -> dict:
    state = correction_state(artifact_dir)
    if state.phase == "consumed":
        raise CorrectionStateError("correction was already consumed")
    if state.phase != "pending" or state.correction is None:
        raise CorrectionStateError("session has no pending correction")
    correction = state.correction
    if correction["correction_id"] != correction_id:
        raise CorrectionStateError("correction id does not match")
    payload = _validate_consumption({
        "schema_version": CORRECTION_SCHEMA_VERSION,
        "record_type": "correction_consumption",
        "correction_id": correction_id,
        "session_id": correction["session_id"],
        "text_sha256": correction["text_sha256"],
        "session_number": session_number,
        "turn_number": turn_number,
        "transcript_segment": _current_transcript_segment(artifact_dir),
        "delivery": delivery,
        "consumed_at": _timestamp(),
    })
    if payload["session_number"] <= correction["after_session_number"]:
        raise CorrectionStateError(
            "correction must be consumed after its stopped session"
        )
    if payload["turn_number"] != 0:
        raise CorrectionStateError(
            "correction must be consumed before the first resumed turn"
        )
    _atomic_write_new(correction_consumption_path(artifact_dir), payload)
    return payload


def _current_transcript_segment(artifact_dir: Path) -> int:
    artifact_dir = Path(artifact_dir)
    segment = 1
    while (
        artifact_dir / f"transcript.pre_seg_{segment}.log"
    ).is_file():
        segment += 1
    return segment


__all__ = [
    "CORRECTION_SCHEMA_VERSION",
    "CorrectionState",
    "CorrectionStateError",
    "consume_correction",
    "correction_consumption_path",
    "correction_path",
    "correction_state",
    "create_correction",
    "load_correction",
    "load_correction_consumption",
    "validate_correction_trace",
]
