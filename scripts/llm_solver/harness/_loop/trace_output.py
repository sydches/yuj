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

from ..._shared.classification import (
    classify_outcome,
    derive_envelope_status,
    is_gate_blocked,
)

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
    if execution_sha:
        fields["execution_output_sha256"] = execution_sha
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
    *,
    tool_name: str,
    result: str,
    gate_blocked: bool,
    execution_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if gate_blocked or is_gate_blocked(result):
        return {
            "outcome": "blocked",
            "pass_fail": "fail",
            "exit_status": None,
            "error_class": "harness_gate",
        }
    status, error_kind = derive_envelope_status(result)
    # A result-stage security block happens after the handler and therefore
    # may retain a successful raw exit status.  The admitted error envelope
    # is authoritative for the model-visible outcome.
    if error_kind == "security_block":
        return {
            "outcome": "error",
            "pass_fail": "fail",
            "exit_status": None,
            "error_class": "security_block",
        }
    execution_metadata = execution_metadata or {}
    if execution_metadata.get("timed_out"):
        return {
            "outcome": "error",
            "pass_fail": "fail",
            "exit_status": None,
            "error_class": "timeout",
        }
    if execution_metadata.get("exit_status_known"):
        exit_status = execution_metadata.get("exit_status")
        if exit_status is not None:
            passed = int(exit_status) == 0
            return {
                "outcome": "ok" if passed else "error",
                "pass_fail": "pass" if passed else "fail",
                "exit_status": int(exit_status),
                "error_class": "" if passed else "nonzero_exit",
            }
    pass_fail = "pass" if classify_outcome(result) == "OK" else "fail"
    exit_status = _exit_status(result)
    if exit_status is None and tool_name in {"bash", "run_tests"} and pass_fail == "pass":
        exit_status = 0
    return {
        "outcome": status,
        "pass_fail": pass_fail,
        "exit_status": exit_status,
        "error_class": error_kind or "",
    }


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
        path = (Path(session.cwd) / rel_path).resolve()
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return None


def _sink_trace_output(session: "Session", result: str, turn: int) -> str:
    try:
        session._sink_counter += 1
        sink_dir = Path(session.cwd) / ".tool_output"
        sink_dir.mkdir(parents=True, exist_ok=True)
        sink_name = (
            f"{session._session_number}_{session._sink_counter:04d}"
            f"_t{turn}_trace.log"
        )
        sink_path = sink_dir / sink_name
        sink_path.write_bytes(result.encode("utf-8", errors="replace"))
        return str(sink_path.relative_to(session.cwd))
    except OSError as exc:
        log.debug("trace output sink failed: %s", exc)
    except ValueError:
        log.debug("trace output sink path escaped cwd")
    return ""


def _safe_relative_path(session: "Session", raw_path: str) -> str:
    if not raw_path:
        return ""
    rel = Path(raw_path)
    if rel.is_absolute() or ".." in rel.parts:
        return ""
    try:
        cwd = Path(session.cwd).resolve()
        abs_path = (cwd / rel).resolve()
        safe_rel = abs_path.relative_to(cwd)
    except (OSError, ValueError):
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
