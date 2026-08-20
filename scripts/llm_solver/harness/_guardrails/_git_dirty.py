"""Disk-state helper for the done gate.

Extracted from ``checks_pre.py`` so that file stays under the 500-line
project size cap.
"""
from __future__ import annotations

import subprocess


def cwd_has_uncommitted_changes(cwd: str | None) -> bool:
    """Return True iff ``cwd`` is a git repo with uncommitted changes.

    ``done_guard`` uses this when a bash command may have changed a file
    without updating ``state.has_mutated``. Run ``git status --porcelain``
    against the current HEAD. Return False for a non-Git directory, missing
    Git, a timeout, or any other error.
    """
    if not cwd:
        return False
    try:
        r = subprocess.run(
            ["git", "-C", cwd, "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return False
    if r.returncode != 0:
        return False
    return bool(r.stdout.strip())
