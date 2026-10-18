"""Lifecycle helper for a Session's persistent native connection.

Owns the eligibility gate, construction, registry install/clear, and
teardown. Kept out of
loop.py so the inner Session class stays under the 500-line file gate.

Reuse bwrap or a bound named container. Ambient mode and the image-backed
container backend retain their own execution paths.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from ..tools import _bash_readable_paths, _bash_unreadable_paths
from ..sandbox import (
    AMBIENT_CONTAINER,
    PersistentBashSession,
    container_mode,
    get_persistent_runner,
    set_persistent_runner,
)
from ..sandbox._persistent_container import PersistentContainerSession
from ..sandbox.policy import sandbox_execution_kwargs

if TYPE_CHECKING:
    from ..loop import Session


def maybe_install_persistent_bash(
    session: "Session",
) -> "PersistentBashSession | PersistentContainerSession | None":
    """Install a persistent native runner for this session if eligible.

    Returns the constructed runner (caller-owned ⇒ caller must call
    ``teardown_persistent_bash`` in a finally), or None if not eligible.

    Eligibility (all required):
      - YUJ_PERSISTENT_BASH != "0" (default on; set to "0" to disable)
      - the centrally resolved execution backend is bwrap
      - a named container is selected, or cfg.bwrap_bin exists on disk
      - ambient mode and the image-backed container backend are not selected
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
    if get_persistent_runner() is not None:
        return None
    mode = container_mode()
    if mode == AMBIENT_CONTAINER:
        return None
    if mode is not None:
        from ..task_environment import discover_task_environment
        runner = PersistentContainerSession(
            session.cwd, discover_task_environment(session.cwd),
            effective_env=session._effective_env,
        )
    else:
        if not Path(session.cfg.bwrap_bin).is_file():
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


def teardown_persistent_bash(
    runner: "PersistentBashSession | PersistentContainerSession | None",
) -> None:
    """Clear the registry and close the runner. Safe with None."""
    if runner is None:
        return
    set_persistent_runner(None)
    runner.close()
