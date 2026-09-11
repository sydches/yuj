"""Legacy Git-dirt query; not task-change evidence for the done gate."""
from __future__ import annotations

import subprocess

from ..plan_mode import PLAN_FILE


def cwd_has_uncommitted_changes(cwd: str | None) -> bool:
    """Return True iff ``cwd`` is a git repo with uncommitted changes.

    This legacy helper does not establish session activity. Run ``git status --porcelain``
    against the current HEAD while excluding the model-authored plan artifact,
    which is phase control rather than implementation. Return False for a
    non-Git directory, missing Git, a timeout, or any other error.
    """
    if not cwd:
        return False
    try:
        r = subprocess.run(
            [
                "git", "-C", cwd, "status", "--porcelain",
                "--untracked-files=all", "--", ".", f":(exclude){PLAN_FILE}",
            ],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return False
    if r.returncode != 0:
        return False
    return bool(r.stdout.strip())
