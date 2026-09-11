"""Trace-safe tool output summaries and metadata.

``.trace.jsonl`` is durable telemetry, not a raw transcript.  This module
keeps each tool-call row bounded while preserving replay handles for
the full bytes when the harness retains them under ``.tool_output/``.
"""
from __future__ import annotations

import hashlib
import html
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from ..time_budget import BudgetExhausted


if TYPE_CHECKING:
    from ..loop import Session

log = logging.getLogger(__name__)

_DEFAULT_TRACE_RESULT_SUMMARY_CHARS = 1200
_NEWLINE = "\n"
_TOOL_RESULT_META_RE = re.compile(r"<tool_result_meta\b(?P<attrs>[^>]*)/>")
_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
_EXIT_CODE_ATTR_RE = re.compile(r'\bexit_code="(?P<code>-?\d+)"')
# Keep in sync with _shared/classification.py. Legacy traces may place
# harness-appended `[HARNESS: ...]` hint blocks after the marker.
_EXIT_MARKER_TAIL_RE = re.compile(
    r"\n\[exit code:\s*(?P<code>\d+)(?:\s+—\s+[^\]]*)?\]"
    r"(?:\s*\n\[HARNESS:[^\]]*\])*\s*\Z"
)

_INSPECT_TOOLS = {
    "read", "grep", "glob", "list_definitions", "structural_search",
}
_WRITE_TOOLS = {
    "write", "edit", "notebook_edit", "structural_edit", "apply_patch", "udiff",
}


def build_tool_call_trace_fields(
    session: "Session",
    *,
    tool_name: str,
    args_summary: str,
    result: str,
    turn: int,
    gate_blocked: bool,
    metadata: dict[str, Any] | None = None,
    execution_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return bounded/additive trace fields for one tool-call result."""
    metadata = metadata or {}
    result_fields = _result_fields(session, str(result or ""), turn)
    outcome_fields = _outcome_fields(
        tool_name=tool_name,
        result=str(result or ""),
        gate_blocked=gate_blocked,
        execution_metadata=execution_metadata,
    )
    fields = {
        "action_summary": f"{tool_name}({args_summary})",
        "action_class": _action_class(tool_name, metadata),
        **outcome_fields,
        **result_fields,
    }
    execution_sha = str((execution_metadata or {}).get("output_sha256") or "")
    if (execution_metadata or {}).get("file_changes") is not None:
        fields["file_changes"] = execution_metadata["file_changes"]
    if execution_sha:
        fields["execution_output_sha256"] = execution_sha
    verification_status = (execution_metadata or {}).get("verification_status")
    if verification_status:
        fields["verification_status"] = verification_status
    evidence = (execution_metadata or {}).get("verification_evidence")
    if evidence is not None:
        fields["verification_evidence"] = evidence
    runner_request = (execution_metadata or {}).get("runner_request")
    if runner_request is not None:
        fields["runner_request"] = runner_request
    shell_submission = (execution_metadata or {}).get("shell_submission")
    if shell_submission is not None:
        fields["shell_submission"] = shell_submission
    execution_budget = (execution_metadata or {}).get("execution_budget")
    if execution_budget is not None:
        fields["execution_budget"] = execution_budget
    observation = (execution_metadata or {}).get("observation_receipt")
    if observation is not None:
        fields["observation_receipt"] = observation
    inspection = (execution_metadata or {}).get("inspection_evidence")
    if inspection is not None:
        fields["inspection_evidence"] = inspection
    execution = (execution_metadata or {}).get("execution_observation")
    if execution is not None:
        fields["execution_observation"] = execution
    return fields


def _result_fields(session: "Session", result: str, turn: int) -> dict[str, Any]:
    cap = _trace_result_cap(session)
    snippet = _truncate(result, cap)

    retained_path = _existing_retained_path(session, result)
    retained_text = _read_retained_text(session, retained_path) if retained_path else None
    if retained_path and retained_text is None:
        retained_path = ""
    if retained_text is None and len(result) > cap:
        retained_path = _sink_trace_output(session, result, turn)
        retained_text = result if retained_path else None

    full_text = retained_text if retained_text is not None else result
    return {
        # Back-compat field. Its semantics are now "bounded output snippet".
        "result_summary": snippet,
        "output_snippet": snippet,
        "output_truncated": len(result) > cap,
        "output_sha256": _sha256(full_text),
        "output_chars": len(full_text),
        "output_lines": _line_count(full_text),
        "output_full_path": retained_path or "",
        "output_retained": bool(retained_path),
    }


def _outcome_fields(
    *, tool_name: str, result: str, gate_blocked: bool,
    execution_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from ..execution_outcome import recorded_outcome
    return recorded_outcome(execution_metadata, gate_blocked=gate_blocked)


def _trace_result_cap(session: "Session") -> int:
    raw = getattr(
        session.cfg,
        "trace_result_summary_chars",
        _DEFAULT_TRACE_RESULT_SUMMARY_CHARS,
    )
    try:
        cap = int(raw)
    except (TypeError, ValueError):
        cap = _DEFAULT_TRACE_RESULT_SUMMARY_CHARS
    return max(0, cap)


def _truncate(text: str, cap: int) -> str:
    if cap <= 0:
        return ""
    if len(text) <= cap:
        return text
    if cap <= 3:
        return "." * cap
    return text[: cap - 3] + "..."


def _existing_retained_path(session: "Session", result: str) -> str:
    match = _TOOL_RESULT_META_RE.search(result)
    if match is None:
        return ""
    attrs = {
        key: html.unescape(value)
        for key, value in _ATTR_RE.findall(match.group("attrs"))
    }
    raw_path = attrs.get("full_path", "")
    return _safe_relative_path(session, raw_path)


def _read_retained_text(session: "Session", rel_path: str) -> str | None:
    if not rel_path:
        return None
    try:
        from ..retained_output import output_file_scope
        from ..task_path import resolve_task_path
        with output_file_scope(session):
            path = resolve_task_path(session.cwd, rel_path)
            return path.read_bytes().decode("utf-8", errors="replace")
    except (OSError, ValueError, BudgetExhausted):
        return None


def _sink_trace_output(session: "Session", result: str, turn: int) -> str:
    from ..retained_output import save_output
    return save_output(session, result, turn, trace=True)


def _safe_relative_path(session: "Session", raw_path: str) -> str:
    if not raw_path:
        return ""
    rel = Path(raw_path)
    if rel.is_absolute() or ".." in rel.parts:
        return ""
    try:
        from ..retained_output import output_file_scope
        from ..task_path import resolve_task_path
        with output_file_scope(session):
            cwd = resolve_task_path(session.cwd, '.')
            abs_path = resolve_task_path(cwd, raw_path)
            safe_rel = abs_path.relative_to(cwd)
    except (OSError, ValueError, BudgetExhausted):
        return ""
    return str(safe_rel)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _line_count(text: str) -> int:
    if not text:
        return 0
    return text.count(_NEWLINE) + (0 if text.endswith(_NEWLINE) else 1)


def _exit_status(result: str) -> int | None:
    match = _EXIT_CODE_ATTR_RE.search(result)
    if match is None:
        match = _EXIT_MARKER_TAIL_RE.search(result)
    if match is None:
        return None
    try:
        return int(match.group("code"))
    except (TypeError, ValueError):
        return None


def _action_class(tool_name: str, metadata: dict[str, Any]) -> str:
    if tool_name == "done":
        return "finish"
    if bool(metadata.get("source_write_like")):
        return "source_write"
    if bool(metadata.get("write_like")) or tool_name in _WRITE_TOOLS:
        return "write"
    if tool_name == "run_tests":
        return "verification"
    if tool_name in _INSPECT_TOOLS:
        return "inspect"
    if tool_name == "bash":
        return "shell"
    return "tool"


__all__ = ["build_tool_call_trace_fields"]
