"""Helpers for solver_state_context.py — extracted to keep the class file smaller."""
from __future__ import annotations

import re


# Historical command/snippet helpers retained for existing callers.
# Rendered compression no longer uses these to infer an outcome or advice.
from .._shell_patterns import matches_command
_READ_PREFIXES = ("cat ", "head ", "tail ", "less ", "more ", "wc ")
_SEARCH_PREFIXES = ("grep ", "rg ", "find ", "ag ", "fd ")


def _classify_cmd(cmd: str) -> str:
    """Classify a normalized bash command as 'test', 'read', 'search', or 'other'."""
    stripped = cmd.lstrip()
    # Preserve a check anywhere in a supported compound command.
    if matches_command(stripped):
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


def _dedup_message(tool_call_id: str) -> str:
    """Point to an identical full result already retained in this rendered view."""
    import json

    return (f"Same supplied text as tool call {json.dumps(tool_call_id, ensure_ascii=True)} "
            "(full result in this view).")
