"""Schemas, persistence, and bounded-value helpers for ``advisor.py``."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sandbox.ignore_policy import (
    IgnorePolicy,
    IgnoreSource,
    parse_ignore_lines,
)
from .schemas import get_tool_schemas

log = logging.getLogger(__name__)

READ_TOOLS = ("read", "grep", "glob")
ADVISOR_TOOLS = frozenset((*READ_TOOLS, "advise"))
SEVERITIES = frozenset(("nit", "concern", "blocker"))
NO_ADVISORY = "NO_ADVISORY"
MAX_REVIEW_STEPS = 8
MAX_REASONING_CHARS = 16_000
MAX_ARGUMENT_CHARS = 4_000
MAX_WATCHDOG_CHARS = 16_000

SYSTEM_PROMPT = """You are a passive second-opinion reviewer for one primary-model turn.
You receive only that completed turn's delta, never the primary transcript.
Inspect the repository only when needed, using read, grep, or glob. Those are
your only workspace tools and cannot change files. Do not infer unseen history.

If there is one concrete issue that would materially help the primary model on
its next turn, call advise exactly once with severity nit, concern, or blocker
and a concise actionable note. Otherwise respond with exactly NO_ADVISORY.
Never put an advisory in ordinary response text. Never call a fabricated tool.
"""


@dataclass(frozen=True, slots=True)
class Advisory:
    """One accepted advisor note awaiting next-turn delivery."""

    severity: str
    note: str
    source_turn: int
    ordinal: int
    note_sha256: str


def rehydrate_transcript(
    transcript_path: Path,
) -> tuple[Advisory | None, set[str], int]:
    """Recover pending/dedupe/cooldown state from the append-only journal."""
    if not transcript_path.is_file():
        return None, set(), 0
    pending: Advisory | None = None
    emitted_hashes: set[str] = set()
    last_emission_ordinal = 0
    try:
        lines = transcript_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError as exc:
        log.warning("advisor transcript unreadable: %s", exc)
        return None, set(), 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning(
                "advisor transcript line %d is invalid: %s", line_number, exc
            )
            continue
        event = row.get("event")
        if event == "advisory":
            try:
                pending = Advisory(
                    severity=str(row["severity"]),
                    note=str(row["note"]),
                    source_turn=int(row["source_turn"]),
                    ordinal=int(row["ordinal"]),
                    note_sha256=str(row["note_sha256"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            emitted_hashes.add(pending.note_sha256)
            last_emission_ordinal = max(
                last_emission_ordinal, pending.ordinal
            )
        elif event == "advisory_injected" and pending is not None:
            if row.get("note_sha256") == pending.note_sha256:
                pending = None
    return pending, emitted_hashes, last_emission_ordinal


def append_transcript(
    transcript_path: Path,
    *,
    session_number: int,
    event: str,
    fields: dict[str, Any],
) -> None:
    row = {
        "event": event,
        "session_number": int(session_number),
        **fields,
    }
    try:
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
                + "\n"
            )
            handle.flush()
    except OSError as exc:
        log.warning("advisor transcript write failed: %s", exc)


def advisor_system_prompt(session: Any) -> str:
    """Return the fixed prompt plus a visible, bounded WATCHDOG.md."""
    watchdog = Path(session.cwd) / "WATCHDOG.md"
    try:
        session._ignore_policy.require_visible(watchdog, is_dir=False)
        if watchdog.is_file():
            body = watchdog.read_text(encoding="utf-8", errors="replace")
            if len(body) > MAX_WATCHDOG_CHARS:
                body = body[:MAX_WATCHDOG_CHARS]
            return (
                SYSTEM_PROMPT
                + "\nRepository-specific review priorities from WATCHDOG.md:\n"
                + body
            )
    except OSError:
        pass
    return SYSTEM_PROMPT


def advisor_schemas() -> list[dict[str, Any]]:
    by_name = {
        schema["function"]["name"]: schema
        for schema in get_tool_schemas("minimal")
    }
    schemas = [by_name[name] for name in READ_TOOLS]
    schemas.append(
        {
            "type": "function",
            "function": {
                "name": "advise",
                "description": (
                    "Publish one severity-tagged second-opinion note for the "
                    "primary model's next turn."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": sorted(SEVERITIES),
                        },
                        "note": {"type": "string", "minLength": 1},
                    },
                    "required": ["severity", "note"],
                    "additionalProperties": False,
                },
            },
        }
    )
    return schemas


def advisor_ignore_policy(session: Any, artifact_dir: Path) -> IgnorePolicy:
    """Hide harness-owned evidence from the advisor's repository view."""
    root = Path(session.cwd).resolve()
    reserved = [
        "/.git/",
        "/.solver/",
        "/.tool_output/",
        "/.procs/",
        "/.shadow_git/",
        "/.trace.jsonl",
        "/transcript.log",
        "/advisor.jsonl",
        "/session.json",
        "/provider.toml",
        "/checkpoint.json",
        "/metrics.json",
        "/prompt.txt",
        "/savings.jsonl",
        "/system_log.jsonl",
        "/approval_request.json",
        "/approval_decisions.json",
        "/shell_interrupt.json",
        "/adaptive_debug.jsonl",
        "/adaptive_control_ledger.jsonl",
        "/llm_hurdle_detector.jsonl",
    ]
    try:
        relative_artifacts = artifact_dir.resolve().relative_to(root)
    except (OSError, ValueError):
        relative_artifacts = None
    if relative_artifacts is not None and str(relative_artifacts) != ".":
        reserved.append(f"/{relative_artifacts.as_posix()}/")
    for raw_path in (
        getattr(session, "_trace_path", None),
        getattr(session, "_state_path", None),
        getattr(session.client, "_transcript_path", None),
    ):
        if raw_path is None:
            continue
        try:
            relative = Path(raw_path).resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        if str(relative) != ".":
            reserved.append(f"/{relative.as_posix()}")
    source_text = "\n".join(reserved) + "\n"
    source = IgnoreSource(
        name="<advisor-reserved-artifacts>",
        sha256=hashlib.sha256(source_text.encode()).hexdigest(),
        size_bytes=len(source_text.encode()),
        rules=parse_ignore_lines(
            reserved, source_name="<advisor-reserved-artifacts>"
        ),
    )
    base_sources = (
        session._ignore_policy.sources
        if session._ignore_policy.enabled
        else ()
    )
    return IgnorePolicy(root=root, sources=(source, *base_sources), enabled=True)


def assistant_message(content: str | None, calls: list[Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(
                        call.arguments,
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
            for call in calls
        ],
    }


def bounded(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def bounded_json(value: Any, limit: int) -> str:
    rendered = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return bounded(rendered, limit)


def note_digest(note: str) -> str:
    normalized = " ".join(note.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


__all__ = [
    "ADVISOR_TOOLS",
    "MAX_ARGUMENT_CHARS",
    "MAX_REASONING_CHARS",
    "MAX_REVIEW_STEPS",
    "NO_ADVISORY",
    "READ_TOOLS",
    "SEVERITIES",
    "Advisory",
    "advisor_ignore_policy",
    "advisor_schemas",
    "advisor_system_prompt",
    "append_transcript",
    "assistant_message",
    "bounded",
    "bounded_json",
    "note_digest",
    "rehydrate_transcript",
]
