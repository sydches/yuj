"""Helpers for solver_state_context.py — extracted to keep the class file smaller."""
from __future__ import annotations

import re


# Command-type classification for dedup messages. Lived in
# solver_state_context.py originally; moved here to break the import
# cycle once dedup logic was extracted to its own module.
_TEST_PREFIXES = (
    "pytest", "python -m pytest", "python -m unittest", "python -m py.test",
    # Additive: go/rust/js test runners so their dedup framing lands in the
    # "test" bucket instead of falling through to "other".
    "go test", "cargo test", "jest", "npx jest", "vitest", "ctest",
    "npm test", "pnpm test", "yarn test",
)
_READ_PREFIXES = ("cat ", "head ", "tail ", "less ", "more ", "wc ")
_SEARCH_PREFIXES = ("grep ", "rg ", "find ", "ag ", "fd ")


def _classify_cmd(cmd: str) -> str:
    """Classify a normalized bash command as 'test', 'read', 'search', or 'other'."""
    stripped = cmd.lstrip()
    for pfx in _TEST_PREFIXES:
        if stripped.startswith(pfx):
            return "test"
    for pfx in _READ_PREFIXES:
        if stripped.startswith(pfx):
            return "read"
    for pfx in _SEARCH_PREFIXES:
        if stripped.startswith(pfx):
            return "search"
    return "other"


_PYTEST_ERROR_RE = re.compile(r"^E\s+.+", re.MULTILINE)
# Additive: common go/rust/js error-line shapes so non-Python failures also
# get a targeted snippet instead of dropping straight to the last-line
# fallback. go: `--- FAIL: TestFoo`; rust: `thread 'x' panicked at ...` and
# `error[E0382]: ...`; js: `Error:` / `TypeError:` / etc.
_MULTILANG_ERROR_RE = re.compile(
    r"^(?:--- FAIL:.+|thread\s+'[^']*'\s+panicked.+|error\[[A-Za-z0-9]+\].+|"
    r"\w*Error:.+)",
    re.MULTILINE,
)


def _extract_error_snippet(prev_content: str, max_chars: int = 200) -> str:
    """Extract the key error line from a previous tool result.

    For pytest output, grabs the last `E   ...` line (the actual assertion
    or exception). For go/rust/js output, grabs the last matching error-line
    shape (`--- FAIL:`, rust panic/`error[...]`, js `Error:`). Falls back to
    the last non-empty line.
    """
    # Pytest E-lines
    matches = _PYTEST_ERROR_RE.findall(prev_content)
    if matches:
        snippet = matches[-1].strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars - 3] + "..."
        return snippet
    # go/rust/js error-line shapes
    ml_matches = _MULTILANG_ERROR_RE.findall(prev_content)
    if ml_matches:
        snippet = ml_matches[-1].strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars - 3] + "..."
        return snippet
    # Fallback: last non-empty line
    for line in reversed(prev_content.splitlines()):
        line = line.strip()
        if line and not line.startswith("="):
            if len(line) > max_chars:
                line = line[:max_chars - 3] + "..."
            return line
    return ""


def _dedup_message(cmd: str, prev_content: str, count: int, turn_ref: int) -> str:
    """Build a context-aware dedup message.

    Uses command type to pick the right framing and echoes the previous
    error so the model knows what to fix.
    """
    cmd_type = _classify_cmd(cmd)
    snippet = _extract_error_snippet(prev_content)
    blocked = count >= 2

    if cmd_type == "test":
        if blocked:
            msg = f"ERROR: BLOCKED — `{cmd}` ran {count + 1} times (turn {turn_ref})."
        else:
            msg = f"WARNING: You already ran `{cmd}` (turn {turn_ref})."
        if snippet:
            msg += f"\nPrevious failure: {snippet}"
        msg += "\nYour last edit didn't fix this. Read the error and make a different change."
    elif cmd_type == "read":
        if blocked:
            msg = f"ERROR: BLOCKED — `{cmd}` ran {count + 1} times (turn {turn_ref}).\nStop reading this file. Edit it or work on something else."
        else:
            msg = f"WARNING: You already ran `{cmd}` (turn {turn_ref}).\nYou already have this content. Edit the file or move on."
    elif cmd_type == "search":
        if blocked:
            msg = f"ERROR: BLOCKED — `{cmd}` ran {count + 1} times (turn {turn_ref}).\nYou already searched for this. Act on what you found."
        else:
            msg = f"WARNING: You already ran `{cmd}` (turn {turn_ref}).\nYou already have these results. Act on them."
    else:
        if blocked:
            msg = f"ERROR: BLOCKED — `{cmd}` ran {count + 1} times (turn {turn_ref}).\nChange your approach — this command will not produce new information."
        else:
            msg = f"WARNING: You already ran `{cmd}` (turn {turn_ref}).\nRe-running will not help — change your approach."
    return msg


