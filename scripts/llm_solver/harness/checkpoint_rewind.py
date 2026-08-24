"""Session-scoped context checkpoints and deferred rewind.

This feature changes only the model-facing conversation.  The raw trace and
task filesystem remain historical/mechanical truth.  A checkpoint is captured
at the end of a complete tool-call turn, so its message prefix can never split
an assistant tool call from its result.  Rewind restores that prefix, appends
one hidden user-role report, and leaves every prior trace row in place.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from html import escape
from typing import Iterable, Mapping, Sequence


REWIND_REPORT_TAG = "rewind-report"


class CheckpointBoundaryError(ValueError):
    """A proposed checkpoint would split or corrupt a tool protocol pair."""


@dataclass(frozen=True)
class ContextCheckpoint:
    """One safe append-log prefix retained in memory for this session."""

    goal: str
    turn: int
    messages: tuple[dict, ...]
    message_count: int
    tool_log_length: int = 0


@dataclass(frozen=True)
class PendingCheckpoint:
    goal: str
    turn: int


@dataclass(frozen=True)
class PendingRewind:
    report: str
    checkpoint: ContextCheckpoint
    from_turn: int


def _tool_call_ids(message: Mapping[str, object]) -> tuple[str, ...]:
    calls = message.get("tool_calls") or ()
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        raise CheckpointBoundaryError("assistant tool_calls must be a sequence")
    ids: list[str] = []
    for call in calls:
        if not isinstance(call, Mapping):
            raise CheckpointBoundaryError("assistant tool call must be an object")
        call_id = str(call.get("id") or "")
        if not call_id:
            raise CheckpointBoundaryError("assistant tool call is missing an id")
        if call_id in ids:
            raise CheckpointBoundaryError(f"duplicate tool call id {call_id!r}")
        ids.append(call_id)
    return tuple(ids)


def validate_checkpoint_boundary(messages: Sequence[Mapping[str, object]]) -> None:
    """Raise when ``messages`` ends inside a tool call/result group.

    The validation is intentionally protocol-only.  It does not inspect tool
    names, arguments, results, or task content.
    """
    pending: set[str] = set()
    for index, message in enumerate(messages):
        role = str(message.get("role") or "")
        if role == "assistant":
            if pending:
                raise CheckpointBoundaryError(
                    "assistant message arrived before all prior tool results "
                    f"at message {index}"
                )
            pending.update(_tool_call_ids(message))
            continue
        if role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if not call_id or call_id not in pending:
                raise CheckpointBoundaryError(
                    f"tool result at message {index} has no pending call"
                )
            pending.remove(call_id)
            continue
        if pending:
            raise CheckpointBoundaryError(
                f"message {index} crosses an unresolved tool call/result pair"
            )
    if pending:
        raise CheckpointBoundaryError(
            "checkpoint ends before tool result(s): " + ", ".join(sorted(pending))
        )


def render_rewind_report(goal: str, report: str) -> str:
    """Render the hidden user-role message retained after a rewind."""
    goal_attr = escape(str(goal), quote=True)
    return (
        f'<{REWIND_REPORT_TAG} goal="{goal_attr}">\n'
        f"{report}\n"
        f"</{REWIND_REPORT_TAG}>"
    )


def rewind_report_messages(messages: Iterable[Mapping[str, object]]) -> list[dict]:
    """Return user-role rewind reports from an append log in order."""
    prefix = f"<{REWIND_REPORT_TAG} "
    reports: list[dict] = []
    for message in messages:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, str):
            continue
        if content.startswith(prefix):
            reports.append({"role": "user", "content": content})
    return reports


def preserve_rewind_reports(
    projected: Sequence[Mapping[str, object]],
    append_log: Iterable[Mapping[str, object]],
) -> list[dict]:
    """Append any rewind-report user messages omitted by a projection."""
    out = [dict(message) for message in projected]
    visible_reports = {
        str(message.get("content") or "")
        for message in out
        if message.get("role") == "user"
        and str(message.get("content") or "").startswith(
            f"<{REWIND_REPORT_TAG} "
        )
    }
    for report in rewind_report_messages(append_log):
        content = str(report["content"])
        if content not in visible_reports:
            out.append(report)
            visible_reports.add(content)
    return out


def capture_context_checkpoint(
    context,
    *,
    goal: str,
    turn: int,
    tool_log_length: int = 0,
) -> ContextCheckpoint:
    """Capture a deep, protocol-safe copy of the context append log."""
    messages = copy.deepcopy(list(context.snapshot_messages()))
    validate_checkpoint_boundary(messages)
    return ContextCheckpoint(
        goal=goal,
        turn=int(turn),
        messages=tuple(messages),
        message_count=len(messages),
        tool_log_length=max(0, int(tool_log_length)),
    )


def rewind_context(context, checkpoint: ContextCheckpoint, report: str) -> list[dict]:
    """Restore a checkpoint prefix and append its hidden report message."""
    messages = copy.deepcopy(list(checkpoint.messages))
    validate_checkpoint_boundary(messages)
    messages.append({
        "role": "user",
        "content": render_rewind_report(checkpoint.goal, report),
    })
    if not context.rewind_messages(messages):
        raise RuntimeError(
            f"context manager {type(context).__name__} does not support rewind"
        )
    return messages


def logical_trace_events(events: Iterable[Mapping[str, object]]) -> list[dict]:
    """Return the model-state view while preserving the raw event stream.

    Both public rewind forms share the same branch-selection rule.  The state
    writer owns persistent turn lineage so a later conversation/workspace
    rewind may select a turn from a previously discarded branch; model-tool
    report restoration must consume that same active view.
    """
    from .state_writer import active_events

    return active_events([dict(event) for event in events])


def restore_rewind_reports(context, events: Iterable[Mapping[str, object]]) -> int:
    """Rehydrate surviving report messages when a new session reads a trace."""
    existing = {
        str(message.get("content") or "")
        for message in context.snapshot_messages()
        if message.get("role") == "user"
    }
    restored = 0
    for event in logical_trace_events(events):
        if event.get("event") != "rewind":
            continue
        report = event.get("report")
        goal = event.get("goal")
        if not isinstance(report, str) or not isinstance(goal, str):
            continue
        content = render_rewind_report(goal, report)
        if content in existing:
            continue
        context.add_user(content)
        existing.add(content)
        restored += 1
    return restored


def _tool_result(
    tool_name: str,
    message: str,
    *,
    status: str = "ok",
    error_kind: str = "",
) -> str:
    attrs = f' tool_name="{escape(tool_name, quote=True)}" status="{status}"'
    if error_kind:
        attrs += f' error_kind="{escape(error_kind, quote=True)}"'
    return f"<tool_result{attrs} v=\"1\">\n{message}\n</tool_result>"


def unavailable_tool_result(tool_name: str) -> str:
    return _tool_result(
        tool_name,
        f"ERROR: {tool_name} is unavailable outside an active session",
        status="error",
        error_kind="session_unavailable",
    )


def build_session_tool_handlers(session) -> dict[str, object]:
    """Build checkpoint/rewind handlers bound to one active Session."""

    def checkpoint_handler(args, _cwd, cfg):
        if not bool(getattr(cfg, "tools_checkpoint_enabled", False)):
            return _tool_result(
                "checkpoint",
                "ERROR: checkpoint is disabled",
                status="error",
                error_kind="disabled",
            )
        goal = args.get("goal")
        if not isinstance(goal, str) or not goal.strip():
            return _tool_result(
                "checkpoint",
                "ERROR: checkpoint goal must be a non-empty string",
                status="error",
                error_kind="bad_arguments",
            )
        session._pending_context_checkpoint = PendingCheckpoint(
            goal=goal, turn=int(session._current_turn)
        )
        return _tool_result(
            "checkpoint",
            "Checkpoint will become active after this tool-call turn completes.",
        )

    def rewind_handler(args, _cwd, cfg):
        if not bool(getattr(cfg, "tools_checkpoint_enabled", False)):
            return _tool_result(
                "rewind",
                "ERROR: rewind is disabled",
                status="error",
                error_kind="disabled",
            )
        report = args.get("report")
        if not isinstance(report, str) or not report.strip():
            return _tool_result(
                "rewind",
                "ERROR: rewind report must be a non-empty string",
                status="error",
                error_kind="bad_arguments",
            )
        checkpoint = getattr(session, "_context_checkpoint", None)
        if checkpoint is None:
            return _tool_result(
                "rewind",
                "ERROR: rewind requires an active checkpoint",
                status="error",
                error_kind="no_active_checkpoint",
            )
        if getattr(session, "_pending_context_rewind", None) is not None:
            return _tool_result(
                "rewind",
                "ERROR: one rewind is already pending for this turn",
                status="error",
                error_kind="rewind_pending",
            )
        try:
            validate_checkpoint_boundary(checkpoint.messages)
        except CheckpointBoundaryError as exc:
            return _tool_result(
                "rewind",
                f"ERROR: active checkpoint is unsafe: {exc}",
                status="error",
                error_kind="unsafe_checkpoint",
            )
        session._pending_context_rewind = PendingRewind(
            report=report,
            checkpoint=checkpoint,
            from_turn=int(session._current_turn),
        )
        return _tool_result(
            "rewind",
            "Rewind accepted; the exploration branch will collapse at turn end.",
        )

    return {
        "checkpoint": checkpoint_handler,
        "rewind": rewind_handler,
    }


def finalize_deferred_context_actions(session, turn: int) -> str | None:
    """Finalize a pending checkpoint or rewind at a complete turn boundary."""
    pending_rewind = getattr(session, "_pending_context_rewind", None)
    pending_checkpoint = getattr(session, "_pending_context_checkpoint", None)
    session._pending_context_rewind = None
    session._pending_context_checkpoint = None

    if pending_rewind is not None:
        checkpoint = pending_rewind.checkpoint
        rewind_context(session.context, checkpoint, pending_rewind.report)
        session._context_checkpoint = None
        session._tool_log = session._tool_log[: checkpoint.tool_log_length]
        session._output_dedup_cache.clear()
        session._last_actual_prompt_tokens = 0
        session._last_fill = 0.0
        session._preflight_prev_estimate = None
        session._prev_preflight_estimate_pt = 0
        session._preflight_gate_live = 0
        session._preflight_gate_chars_new = 0
        session._preflight_density = 0.25
        if hasattr(session, "_compaction_turns"):
            session._compaction_turns = [
                value
                for value in session._compaction_turns
                if isinstance(value, int) and value <= checkpoint.turn
            ]
        session._compaction_turn = checkpoint.turn
        session._emit(
            "rewind",
            session_number=session._session_number,
            turn_number=int(turn),
            from_turn=int(pending_rewind.from_turn),
            to_turn=int(checkpoint.turn),
            report_chars=len(pending_rewind.report),
            checkpoint_message_count=checkpoint.message_count,
            goal=checkpoint.goal,
            report=pending_rewind.report,
        )
        return "rewind"

    if pending_checkpoint is not None:
        session._context_checkpoint = capture_context_checkpoint(
            session.context,
            goal=pending_checkpoint.goal,
            turn=int(turn),
            tool_log_length=len(session._tool_log),
        )
        return "checkpoint"
    return None


__all__ = [
    "CheckpointBoundaryError",
    "ContextCheckpoint",
    "REWIND_REPORT_TAG",
    "build_session_tool_handlers",
    "capture_context_checkpoint",
    "finalize_deferred_context_actions",
    "logical_trace_events",
    "preserve_rewind_reports",
    "render_rewind_report",
    "restore_rewind_reports",
    "rewind_context",
    "rewind_report_messages",
    "unavailable_tool_result",
    "validate_checkpoint_boundary",
]
