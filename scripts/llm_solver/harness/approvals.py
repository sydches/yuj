"""Assistant-mode approval gate for bounded risky shell actions."""
from __future__ import annotations

import json
import time
from pathlib import Path

_REQUEST_FILE = "approval_request.json"
_DECISIONS_FILE = "approval_decisions.json"


def _request_path(trace_path: Path | None) -> Path | None:
    return None if trace_path is None else Path(trace_path).parent / _REQUEST_FILE


def _decisions_path(trace_path: Path | None) -> Path | None:
    return None if trace_path is None else Path(trace_path).parent / _DECISIONS_FILE


def _load_json(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _clear_request(trace_path: Path | None) -> None:
    path = _request_path(trace_path)
    if path is None or not path.exists():
        return
    try:
        path.unlink()
    except OSError:
        pass


def _write_request(trace_path: Path | None, payload: dict) -> None:
    path = _request_path(trace_path)
    if path is not None:
        path.write_text(json.dumps(payload, indent=2) + "\n")


def _path_outside_task(path: str, cwd: str) -> bool:
    try:
        root = Path(cwd).resolve()
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate.resolve().relative_to(root)
        return False
    except (ValueError, OSError):
        return True


def _has_external_path(tokens: list[str], cwd: str) -> bool:
    end_of_flags = False
    for token in tokens:
        if not token:
            continue
        if not end_of_flags:
            if token == "--":
                end_of_flags = True
                continue
            if token.startswith("-"):
                continue
        if _path_outside_task(token, cwd):
            return True
    return False


def _reason_for_bash(cmd: str, cwd: str | None) -> str | None:
    from ._loop import _split_bash_segments

    command = (cmd or "").strip()
    if not command:
        return None
    segments = _split_bash_segments(command) or [[command]]
    for segment in segments:
        if not segment:
            continue
        head, rest = segment[0], segment[1:]
        if head == "rm":
            return "destructive file deletion via rm"
        if head == "git" and rest[:2] == ["reset", "--hard"]:
            return "destructive git reset --hard"
        if head == "git" and rest and rest[0] == "clean":
            return "destructive git clean"
        if head == "git" and rest[:2] == ["checkout", "--"]:
            return "destructive git checkout --"
        if head == "chmod":
            return "permission change via chmod"
        if head == "chown":
            return "ownership change via chown"
        if head in {"mv", "cp"} and cwd and _has_external_path(rest, cwd):
            return f"{head} crosses the repo root"
    return None


def approval_decision(
    *,
    runtime_mode: str,
    cwd: str,
    trace_path: Path | None,
    tool_name: str,
    tool_args: dict,
    args_summary: str,
) -> tuple[bool, str | None]:
    """Return whether an action may execute now, recording a pause if needed."""
    if runtime_mode != "assistant" or tool_name != "bash":
        return True, None

    cmd = str(tool_args.get("cmd") or "")
    reason = _reason_for_bash(cmd, cwd)
    if reason is None:
        return True, None

    decision = _load_json(_decisions_path(trace_path)).get(f"{tool_name}:{cmd}")
    if decision == "approved":
        return True, None
    if decision == "rejected":
        return False, f"{reason}; previously rejected by operator"

    request = _load_json(_request_path(trace_path))
    same_request = (
        request.get("tool_name") == tool_name and request.get("cmd") == cmd
    )
    if same_request and request.get("status") == "approved":
        _clear_request(trace_path)
        return True, None
    if same_request and request.get("status") == "rejected":
        return False, request.get("rejection_reason") or f"{reason}; rejected by operator"

    _write_request(
        trace_path,
        {
            "status": "pending",
            "tool_name": tool_name,
            "cmd": cmd,
            "args_summary": args_summary,
            "reason": reason,
            "requested_at": time.time(),
        },
    )
    return False, reason
