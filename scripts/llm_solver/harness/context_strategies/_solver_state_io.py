"""I/O helpers for SolverStateContext: state.json reads, prepopulation."""
from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path

from ..bash_write_classification import STATE_WRITER_MUTATION_PREFIXES


def prepopulate_from_trace(
    cwd: Path,
    recent_tool_results: deque,
    budget: int,
) -> int:
    """Pre-populate the rolling window from files modified in prior sessions.

    Reads state.json trace, finds the most recent write/edit actions,
    re-reads those files from disk, and pushes them onto the rolling
    window deque. This closes the context cliff at session boundaries:
    without it, session 2+ starts with an empty rolling window and the
    model edits files from stale memory.

    Returns the number of files injected.
    """
    state_path = cwd / ".solver" / "state.json"
    if not state_path.is_file():
        return 0
    try:
        state = json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return 0
    trace = state.get("trace", [])
    if not trace:
        return 0

    # Collect unique file paths from the most recent write/edit actions.
    # Walk backward, stop after collecting enough to fill the window.
    # Use the full budget — new tool results during the session will
    # push these out naturally via the char-budget trim in
    # _format_tool_results.
    seen: set[str] = set()
    files_to_read: list[tuple[str, str]] = []
    chars_used = 0
    cwd_resolved = cwd.resolve()
    budget_full = False
    for entry in reversed(trace):
        action = entry.get("action", "")
        raw_paths = [
            str(path) for path in entry.get("source_write_paths") or []
        ]
        if not raw_paths and action.startswith(STATE_WRITER_MUTATION_PREFIXES):
            m = re.search(r"path='([^']+)'", action)
            if m:
                raw_paths.append(m.group(1))
        if not raw_paths:
            continue
        for fpath in reversed(raw_paths):
            if fpath in seen or fpath.endswith("state.json"):
                continue
            seen.add(fpath)
            stripped = fpath.lstrip("/").lstrip("./")
            target = (cwd_resolved / stripped).resolve(strict=False)
            try:
                target.relative_to(cwd_resolved)
            except ValueError:
                continue
            if not target.is_file():
                continue
            try:
                content = target.read_text()
            except OSError:
                continue
            if chars_used + len(content) > budget:
                budget_full = True
                break
            files_to_read.append((fpath, content))
            chars_used += len(content)
        if budget_full:
            break

    # Inject in chronological order (oldest first = least recent mutation first).
    injected = 0
    for fpath, content in reversed(files_to_read):
        numbered = "\n".join(
            f"{i+1}: {line}" for i, line in enumerate(content.splitlines())
        )
        synthetic = {
            "role": "tool",
            "tool_call_id": f"prepopulate-{fpath}",
            "content": numbered,
        }
        recent_tool_results.append(synthetic)
        injected += 1
    return injected
