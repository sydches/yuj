"""Describe native recorded outcomes without deriving a task policy from text."""
from __future__ import annotations

from collections import Counter
import json


def _outcome(row: dict) -> str:
    if row.get("outcome_version") != "native_execution_v1":
        return "unknown"
    value = row.get("outcome")
    return value if isinstance(value, str) and value in {
        "unknown", "blocked", "not_executed", "completed", "ok", "error"
    } else "unknown"


def recorded_action(trace: list) -> str:
    rows = [row for row in trace if isinstance(row, dict)]
    if not rows:
        return ""
    row = rows[-1]
    outcome = _outcome(row)
    lines = [
        "Latest recorded tool call (historical evidence):",
        f"- step: {json.dumps(row.get('step'))}",
        f"- displayed request: {json.dumps(str(row.get('action') or ''))}",
        f"- recorded outcome: {outcome}",
    ]
    if row.get("outcome_version") == "native_execution_v1":
        status = row.get("exit_status")
        if type(status) is int:
            lines.append(f"- exit status: {status}")
        error = str(row.get("error_class") or "")
        if error in {"harness_gate", "security_block", "timeout", "check_failed"}:
            lines.append(f"- recorded error class: {error}")
        if error in {"harness_gate", "security_block"}:
            lines.append("Consult the recorded rejection and the permitted operation boundary before retrying.")
        elif error == "timeout":
            lines.append("A timeout does not establish whether the operation changed files or completed its check.")
        elif type(status) is int and status != 0:
            lines.append("Use the recorded diagnostic to investigate this exit; its cause is not established here.")
    lines.append(
        "Choose the next action from the task requirements and available evidence. "
        "This record does not establish task completion, file changes, or the cause of an execution problem."
    )
    return "\n".join(lines)


def recorded_activity(trace: list) -> str:
    rows = [row for row in trace if isinstance(row, dict)]
    if not rows:
        return ""
    counts = Counter(_outcome(row) for row in rows)
    lines = [f"Recorded tool calls in this view: {len(rows)}."]
    lines.append("Recorded outcomes: " + ", ".join(
        f"{name}={count}" for name, count in sorted(counts.items())
    ) + ".")
    paths = []
    for row in rows:
        hints = row.get("source_write_paths")
        if isinstance(hints, list):
            for path in hints:
                if isinstance(path, str) and path and path not in paths:
                    paths.append(path)
    if paths:
        lines.append("Recorded write-target hints (changes unverified): " + json.dumps(paths))
    latest = str(rows[-1].get("action") or "")
    if latest:
        repeated = 0
        for row in reversed(rows):
            if str(row.get("action") or "") != latest:
                break
            repeated += 1
        if repeated > 1:
            lines.append(f"Consecutive identical displayed requests: {repeated}; "
                         "output equality and task progress are not established.")
    lines.append("Counts cover the retained records, not the complete task history or required work.")
    return "\n".join(lines)
