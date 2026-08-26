"""Lifecycle helper for the per-Session PersistentBashSession.

Owns the eligibility gate (sandbox-mode + bwrap binary + opt-out env
var), construction, registry install/clear, and teardown. Kept out of
loop.py so the inner Session class stays under the 500-line file gate.

The persistent runner is bwrap-mode only. Ambient and docker-exec
sandboxes route per-call via _run_in_sandbox; the eligibility check
here is the single source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from ..tools import _bash_readable_paths, _bash_unreadable_paths
from ..sandbox import (
    PersistentBashSession,
    container_mode,
    get_persistent_runner,
    set_persistent_runner,
)
from ..sandbox.policy import sandbox_execution_kwargs

if TYPE_CHECKING:
    from ..loop import Session


def maybe_install_persistent_bash(session: "Session") -> "PersistentBashSession | None":
    """Install a persistent bwrap+bash runner for this session if eligible.

    Returns the constructed runner (caller-owned ⇒ caller must call
    ``teardown_persistent_bash`` in a finally), or None if not eligible.

    Eligibility (all required):
      - YUJ_PERSISTENT_BASH != "0" (default on; set to "0" to disable)
      - the centrally resolved execution backend is bwrap
      - container_mode() is None (i.e. legacy bwrap mode, not ambient
        or docker-exec — those use per-call subprocess.run)
      - cfg.bwrap_bin exists on disk
      - no other runner is already installed on this thread (preserves
        nested-session safety; the outer Session keeps ownership)
    """
    if os.environ.get("YUJ_PERSISTENT_BASH", "1") == "0":
        return None
    execution = sandbox_execution_kwargs(session.cfg)
    if not execution["sandbox"]:
        return None
    if execution["sandbox_backend"] != "bwrap":
        return None
    if container_mode() is not None:
        return None
    if not Path(session.cfg.bwrap_bin).is_file():
        return None
    if get_persistent_runner() is not None:
        return None
    runner = PersistentBashSession(
        cwd=session.cwd,
        bwrap_bin=session.cfg.bwrap_bin,
        unreadable_paths=_bash_unreadable_paths(
            session.cwd, session.cfg, session._ignore_policy,
        ),
        readable_paths=_bash_readable_paths(session.cfg),
        sandbox_required=True,
        effective_env=session._effective_env,
        allow_login_shell=session._allow_login_shell,
    )
    set_persistent_runner(runner)
    return runner


def teardown_persistent_bash(runner: "PersistentBashSession | None") -> None:
    """Clear the registry and close the runner. Safe with None."""
    if runner is None:
        return
    set_persistent_runner(None)
    runner.close()
