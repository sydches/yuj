"""Passive, read-only second-opinion review for completed primary turns.

The advisor is deliberately outside the primary conversation.  Every review
starts from a fresh system/user pair containing only the completed turn delta.
The model may inspect the task through three read-only tools and must use the
``advise`` tool to publish a bounded note.  Its full private conversation is
written to ``advisor.jsonl``; the raw harness trace receives metadata only.
"""
from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import Any

from ._advisor_support import (
    ADVISOR_TOOLS as _ADVISOR_TOOLS,
    MAX_ARGUMENT_CHARS as _MAX_ARGUMENT_CHARS,
    MAX_REASONING_CHARS as _MAX_REASONING_CHARS,
    MAX_REVIEW_STEPS as _MAX_REVIEW_STEPS,
    NO_ADVISORY as _NO_ADVISORY,
    READ_TOOLS as _READ_TOOLS,
    SEVERITIES as _SEVERITIES,
    Advisory,
    advisor_ignore_policy as _advisor_ignore_policy,
    advisor_schemas as _advisor_schemas,
    advisor_system_prompt,
    append_transcript,
    assistant_message as _assistant_message,
    bounded as _bounded,
    bounded_json as _bounded_json,
    note_digest as _note_digest,
    rehydrate_transcript,
)
from ._loop.model_role_runtime import consumer_role_client, record_role_usage
from .tool_validation import ToolSchemaSet
from .tools import ToolRegistry, dispatch

log = logging.getLogger(__name__)


class AdvisorRuntime:
    """Run/session adapter for isolated advisor review and delivery."""

    def __init__(self, session: Any, artifact_dir: Path):
        self._session = session
        self._artifact_dir = Path(artifact_dir)
        self._transcript_path = self._artifact_dir / "advisor.jsonl"
        self._enabled = bool(
            getattr(session.cfg, "advisor_enabled", False)
            and not getattr(session.client, "is_replay", False)
        )
        self._delta: dict[str, Any] | None = None
        self._pending: Advisory | None = None
        self._emitted_hashes: set[str] = set()
        self._last_emission_ordinal = 0
        self._schemas = _advisor_schemas()
        self._schema_set = ToolSchemaSet.from_openai_tools(self._schemas)
        self._readonly_registry = ToolRegistry(
            handlers={
                name: session._tool_registry.handlers[name]
                for name in _READ_TOOLS
            }
        )
        self._ignore_policy = _advisor_ignore_policy(
            session, self._artifact_dir
        )
        if self._enabled:
            (
                self._pending,
                self._emitted_hashes,
                self._last_emission_ordinal,
            ) = rehydrate_transcript(self._transcript_path)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def transcript_path(self) -> Path:
        return self._transcript_path

    def capture_turn(
        self,
        *,
        turn: int,
        content: str | None,
        tool_calls: list[Any],
    ) -> None:
        """Capture primary output only; tool results are projected at review."""
        if not self._enabled:
            return
        self._delta = {
            "primary_turn": int(turn),
            "assistant_reasoning": _bounded(content or "", _MAX_REASONING_CHARS),
            "tool_calls": [
                {
                    "name": str(call.name),
                    "arguments": _bounded_json(
                        call.arguments, _MAX_ARGUMENT_CHARS
                    ),
                }
                for call in tool_calls
            ],
        }

    def review_turn(self, turn: int) -> bool:
        """Review an eligible captured turn; return whether a note was queued.

        Advisor failures are fail-open for the primary task.  They remain
        visible in the dedicated transcript but never terminate the solver.
        """
        if not self._enabled:
            return False
        if self._delta is None or self._delta.get("primary_turn") != int(turn):
            return False

        ordinal = self._primary_turn_ordinal()
        if self._pending is not None:
            self._append(
                "review_skipped",
                source_turn=turn,
                ordinal=ordinal,
                reason="pending_advisory",
            )
            return False
        every = int(self._session.cfg.advisor_every_n_turns)
        if ordinal <= 0 or ordinal % every:
            self._append(
                "review_skipped",
                source_turn=turn,
                ordinal=ordinal,
                reason="cadence",
            )
            return False
        immune = int(self._session.cfg.advisor_immune_turns)
        if (
            self._last_emission_ordinal > 0
            and ordinal - self._last_emission_ordinal <= immune
        ):
            self._append(
                "review_skipped",
                source_turn=turn,
                ordinal=ordinal,
                reason="cooldown",
            )
            return False

        delta = dict(self._delta)
        delta["tool_results"] = self._tool_result_delta(turn)
        try:
            advisory = self._run_review(
                source_turn=int(turn), ordinal=ordinal, delta=delta
            )
        except Exception as exc:  # advisor must not stop the primary task
            log.warning("advisor review failed at turn %d: %s", turn, exc)
            self._append(
                "error",
                source_turn=turn,
                ordinal=ordinal,
                error_type=type(exc).__name__,
                detail=str(exc),
            )
            return False
        if advisory is None:
            return False
        if advisory.note_sha256 in self._emitted_hashes:
            self._append(
                "advisory_deduplicated",
                source_turn=turn,
                ordinal=ordinal,
                severity=advisory.severity,
                chars=len(advisory.note),
                note_sha256=advisory.note_sha256,
            )
            return False

        self._pending = advisory
        self._emitted_hashes.add(advisory.note_sha256)
        self._last_emission_ordinal = ordinal
        self._append(
            "advisory",
            source_turn=turn,
            ordinal=ordinal,
            severity=advisory.severity,
            chars=len(advisory.note),
            note=advisory.note,
            note_sha256=advisory.note_sha256,
        )
        self._session._emit(
            "advisor_note",
            session_number=self._session._session_number,
            turn_number=int(turn),
            turn=int(turn),
            severity=advisory.severity,
            chars=len(advisory.note),
            ordinal=ordinal,
            note_sha256=advisory.note_sha256,
        )
        return True

    def inject_pending(self, turn: int) -> bool:
        """Deliver one accepted note into the next model-facing request."""
        if not self._enabled or self._pending is None:
            return False
        advisory = self._pending
        safe_note = html.escape(advisory.note, quote=False)
        fragment = (
            '<injected-fragment source="advisor" '
            f'severity="{advisory.severity}">\n'
            f"{safe_note}\n"
            "</injected-fragment>"
        )
        self._session.context.add_injected_fragment(fragment)
        self._append(
            "advisory_injected",
            source_turn=advisory.source_turn,
            target_turn=int(turn),
            ordinal=advisory.ordinal,
            severity=advisory.severity,
            chars=len(advisory.note),
            note_sha256=advisory.note_sha256,
        )
        self._pending = None
        return True

    def _run_review(
        self, *, source_turn: int, ordinal: int, delta: dict[str, Any]
    ) -> Advisory | None:
        routed = consumer_role_client(self._session, "advisor")
        client = routed.client
        request_method = getattr(client, "complete_tool_side_request", None)
        if not callable(request_method):
            raise RuntimeError(
                "advisor client does not support isolated tool side requests"
            )
        messages = [
            {"role": "system", "content": advisor_system_prompt(self._session)},
            {
                "role": "user",
                "content": (
                    "Review this completed primary-turn delta:\n"
                    + json.dumps(
                        delta,
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
            },
        ]

        for step in range(_MAX_REVIEW_STEPS):
            self._append(
                "request",
                source_turn=source_turn,
                ordinal=ordinal,
                review_step=step,
                messages=messages,
                offered_tools=[
                    schema["function"]["name"] for schema in self._schemas
                ],
            )
            result = request_method(messages, self._schemas, turn=step)
            record_role_usage(self._session, routed, result.usage)
            calls = list(result.tool_calls or [])
            self._append(
                "response",
                source_turn=source_turn,
                ordinal=ordinal,
                review_step=step,
                content=result.content,
                finish_reason=result.finish_reason,
                tool_calls=[
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in calls
                ],
            )
            if not calls:
                if (result.content or "").strip() == _NO_ADVISORY:
                    self._append(
                        "no_advisory",
                        source_turn=source_turn,
                        ordinal=ordinal,
                    )
                    return None
                self._quarantine(
                    source_turn,
                    ordinal,
                    reason="unstructured_response",
                    content=result.content or "",
                )
                return None

            names = [str(call.name) for call in calls]
            if any(name not in _ADVISOR_TOOLS for name in names):
                self._quarantine(
                    source_turn,
                    ordinal,
                    reason="unknown_or_mutating_tool",
                    tool_names=names,
                )
                return None
            invalid = []
            for call in calls:
                validation = self._schema_set.validate(
                    call.name, call.arguments
                )
                if not validation.valid:
                    invalid.append(
                        {
                            "tool": call.name,
                            "errors": [
                                error.as_dict() for error in validation.errors
                            ],
                        }
                    )
            if invalid:
                self._quarantine(
                    source_turn,
                    ordinal,
                    reason="schema_reject",
                    invalid=invalid,
                )
                return None
            if "advise" in names:
                if len(calls) != 1 or names != ["advise"]:
                    self._quarantine(
                        source_turn,
                        ordinal,
                        reason="advise_must_be_exclusive",
                        tool_names=names,
                    )
                    return None
                return self._parse_advisory(
                    calls[0].arguments,
                    source_turn=source_turn,
                    ordinal=ordinal,
                )

            messages.append(_assistant_message(result.content, calls))
            for call in calls:
                try:
                    tool_result = dispatch(
                        call.name,
                        call.arguments,
                        cwd=self._session.cwd,
                        cfg=self._session.cfg,
                        redactions=self._session.redactions,
                        tool_registry=self._readonly_registry,
                        active_tools=_READ_TOOLS,
                        ignore_policy=self._ignore_policy,
                        effective_env=self._session._effective_env,
                        allow_login_shell=self._session._allow_login_shell,
                    )
                except Exception as exc:
                    tool_result = f"ERROR: read-only advisor tool failed: {exc}"
                self._append(
                    "tool_result",
                    source_turn=source_turn,
                    ordinal=ordinal,
                    review_step=step,
                    tool_call_id=call.id,
                    tool_name=call.name,
                    content=tool_result,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": tool_result,
                    }
                )

        self._quarantine(
            source_turn,
            ordinal,
            reason="review_step_limit",
        )
        return None

    def _parse_advisory(
        self, arguments: dict[str, Any], *, source_turn: int, ordinal: int
    ) -> Advisory | None:
        severity = str(arguments.get("severity", "")).strip().lower()
        note = str(arguments.get("note", "")).strip()
        max_chars = int(self._session.cfg.advisor_max_note_chars)
        if severity not in _SEVERITIES:
            self._quarantine(
                source_turn, ordinal, reason="invalid_severity"
            )
            return None
        if not note or "\x00" in note or len(note) > max_chars:
            self._quarantine(
                source_turn,
                ordinal,
                reason="invalid_note",
                severity=severity,
                chars=len(note),
                max_note_chars=max_chars,
            )
            return None
        digest = _note_digest(note)
        return Advisory(severity, note, source_turn, ordinal, digest)

    def _tool_result_delta(self, turn: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        allowed = (
            "tool_name",
            "args_summary",
            "action_class",
            "outcome",
            "pass_fail",
            "error_class",
            "result_summary",
            "path",
            "target",
        )
        for event in self._session._trace_events:
            if event.get("event") != "tool_call":
                continue
            if int(event.get("session_number", -1)) != int(
                self._session._session_number
            ):
                continue
            if int(event.get("turn_number", -1)) != int(turn):
                continue
            rows.append(
                {key: event[key] for key in allowed if key in event}
            )
        return rows

    def _primary_turn_ordinal(self) -> int:
        return sum(
            1
            for event in self._session._trace_events
            if event.get("event") == "turn"
        )

    def _quarantine(
        self, source_turn: int, ordinal: int, *, reason: str, **fields: Any
    ) -> None:
        self._append(
            "quarantine",
            source_turn=source_turn,
            ordinal=ordinal,
            reason=reason,
            **fields,
        )

    def _append(self, event: str, **fields: Any) -> None:
        append_transcript(
            self._transcript_path,
            session_number=self._session._session_number,
            event=event,
            fields=fields,
        )


__all__ = ["Advisory", "AdvisorRuntime"]
