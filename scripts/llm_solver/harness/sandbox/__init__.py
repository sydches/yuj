"""Sandbox — mount-namespace enforcement for the model's bash tool.

Three modes, dispatched by the ``YUJ_CONTAINER`` env var (set by an
external launcher, never by the harness):

  - ``YUJ_CONTAINER`` unset → **bwrap mode** (legacy default).
    Bubblewrap starts with an empty filesystem and mounts the task,
    Linux runtime files and discovered installed runtime components.
    Other host files and host sockets are absent.

  - ``YUJ_CONTAINER=ambient`` → **ambient container mode**.
    The harness itself is already running inside a container that
    provides the isolation boundary (e.g. polyglot's yuj-polyglot
    image). Each bash call runs as a plain ``subprocess.run`` in the
    same container — no nested bwrap, no docker-exec round-trip. The
    outer container's mount namespace is the sandbox; we do not require
    bwrap to be installed inside it.

  - ``YUJ_CONTAINER=<container_id>`` → **docker-exec container mode**.
    Each bash call becomes ``docker exec <container_id>``. The container
    is started by the launcher. Task setup inspects its bind mounts and
    working directory to find the container alias for the host task.
    Shell and filesystem tools share that mapping. The launcher owns
    image selection, mount restrictions, user and network policy.

Only :func:`_build_bwrap_argv` knows which mode is in effect — the
rest of the harness (``tools.py``, ``loop.py``, ``config.py``) treats
this module as a black box that returns "the argv that wraps a bash
command for the model's sandbox." Adding container mode here keeps the
isolation choice local to its single-purpose layer.

:func:`run_pretest` in ``loop.py`` does *not* go through this module;
it runs ``pretest.sh`` via plain ``subprocess.run`` so the script's
nested ``docker run`` against the testbed image keeps working
regardless of mode.

This module is a package; sub-files split the implementation by
concern while preserving the import surface ``harness.sandbox``:

  - ``_preflight.py``   — bwrap binary verification
  - ``_unreadable.py``  — glob-pattern → mask-args expansion
  - ``_persistent.py``  — long-lived bwrap+bash subprocess
  - this file           — dispatch (``container_mode``,
    ``_build_bwrap_argv``), public re-exports
  - ``_filesystem.py``  — startup-fixed runtime component mounts

See ``docs/serving_overlay.md`` for the server setup.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path

from .env_policy import (
    DEFAULT_FIXED_ENVIRONMENT,
    build_bash_argv,
    build_bwrap_env_argv,
    build_clean_exec_argv,
)

# Public + test-private re-exports so ``from harness.sandbox import X``
# continues to work after the module→package split.
from ._preflight import (  # noqa: F401
    _BWRAP_BROKEN_PATTERNS,
    _BWRAP_PREFLIGHT_CACHE,
    bwrap_preflight,
)
from ._unreadable import (  # noqa: F401
    _UNREADABLE_CACHE,
    _UNREADABLE_HARD_CAP,
    _expand_unreadable_paths,
    _is_specific_pattern,
)
from ._persistent import (  # noqa: F401
    _PERSISTENT_MARKER_PREFIX,
    PersistentBashSession,
    get_persistent_runner,
    set_persistent_runner,
)

log = logging.getLogger(__name__)

# Default path; the effective path comes from config.toml [tools] bwrap_bin.
_DEFAULT_BWRAP_BIN = "/usr/bin/bwrap"

# Sentinel for the ambient container mode (see module docstring).
AMBIENT_CONTAINER = "ambient"


def container_mode() -> str | None:
    """Resolve the YUJ_CONTAINER env var into a sandbox mode tag.

    Returns:
      - ``None``                 → bwrap mode (legacy default)
      - ``'ambient'``            → ambient container mode (run bash directly)
      - any other non-empty str  → docker-exec container mode (the value
                                   is the target container id)

    Single source of truth for the dispatch in ``_run_in_sandbox`` and
    ``_build_bwrap_argv``. Both call sites must stay in sync; resolving
    here avoids drift.
    """
    v = os.environ.get("YUJ_CONTAINER")
    return v if v else None


def _build_bwrap_argv(
    cmd: str, cwd: str, bwrap_bin: str = _DEFAULT_BWRAP_BIN,
    *, unreadable_paths: tuple[str, ...] = (),
    readable_paths: tuple[str, ...] = (),
    sandbox_required: bool = False,
    tail: list[str] | None = None,
    effective_env: Mapping[str, str] | None = None,
    allow_login_shell: bool = False,
    _mask_arguments: list[str] | None = None,
    _combine_task_output: bool = False,
    _host_task_root=None,
    _filesystem_view=None,
) -> list[str]:
    """Build the argv that runs `cmd` for the model's bash tool.

    Dispatches on ``YUJ_CONTAINER``:

    When set, returns a ``docker exec`` argv targeting that container.
    The task's inspected bind mount supplies ``--workdir``. The host
    ``cwd`` and its container alias refer to the same task bytes.
    Missing or ambiguous mappings refuse execution. ``env -i`` applies the same explicit
    command environment as every other backend before bash starts.

    When unset, builds the startup-fixed Linux filesystem view. Only the
    task is bound writable. Runtime components and declared resources are
    read-only; home and temporary storage are private. The host Docker socket
    is not exposed. Privileged preparation runs outside this command boundary.

    The result is passed to subprocess.run as an argv list (no shell).
    The final non-login `bash` runs the model's shell command inside the
    namespace where only `cwd` is writable. Login profile loading is an
    explicit environment-policy opt-in.
    """
    mode = container_mode()
    command_env = (
        DEFAULT_FIXED_ENVIRONMENT
        if effective_env is None
        else effective_env
    )
    if mode == AMBIENT_CONTAINER:
        # Ambient mode is dispatched in _run_in_sandbox before this
        # function is called. Reaching here means the caller bypassed
        # the dispatcher — fail loudly rather than silently fall through
        # to bwrap (which would re-introduce the bwrap-binary requirement
        # this mode exists to remove).
        raise RuntimeError(
            "_build_bwrap_argv called in ambient container mode; the "
            "caller should have routed through the ambient branch in "
            "_run_in_sandbox instead."
        )
    if mode is not None:
        from ..task_environment import discover_task_environment
        task_environment = discover_task_environment(cwd)
        if not task_environment.container_id:
            from ..task_environment import TaskEnvironmentUnavailable
            raise TaskEnvironmentUnavailable('selected container has no inspected identity')
        shell_argv = build_bash_argv(
            cmd, allow_login_shell=allow_login_shell,
        ) if tail is None else list(tail)
        from ..process_identity import guarded_process_argv
        return guarded_process_argv([
            "docker", "exec",
            "--workdir", task_environment.working_directory,
            task_environment.container_id,
        ], build_clean_exec_argv(shell_argv, command_env), task_environment.process_identity,
            combine_output=_combine_task_output)

    from ._filesystem import filesystem_view, build_filesystem_argv, BoundTaskArgv, resolve_filesystem_view
    from ..task_path import active_task_files, capture_host_task_root
    files = active_task_files(cwd)
    if _host_task_root is None:
        _host_task_root = getattr(files, '_host_task_root', None)
    if _filesystem_view is None:
        _filesystem_view = getattr(files, '_filesystem_view', None)
    _filesystem_view = resolve_filesystem_view(_filesystem_view)
    view = (_filesystem_view if _filesystem_view is not None
            else filesystem_view(cwd, command_env, readable_paths, unreadable_paths))
    if _host_task_root is None:
        _host_task_root = view.task_root or capture_host_task_root(view.cwd)
    command_env = {**command_env, **dict(view.runtime_bindings)}
    cwd = view.cwd
    filesystem_args = build_filesystem_argv(view, task_root=_host_task_root)
    argv = [
        bwrap_bin,
        *filesystem_args,
        "--unshare-net", "--unshare-pid", "--unshare-ipc",
        "--unshare-uts", "--unshare-cgroup", "--cap-drop", "ALL",
        "--die-with-parent", "--chdir", cwd,
        *build_bwrap_env_argv(command_env),
    ]
    # Protect cwd/.git/hooks against arbitrary writes. The cwd is a git checkout
    # of the masked repo. Without this guard, the model could write
    # `.git/hooks/pre-commit` (or post-commit, post-update, etc.) and
    # the script would fire when `pretest.sh` or `evaluate.py` runs
    # `git` against the same cwd AFTER the sandbox exits — gaining
    # post-evaluation arbitrary execution outside any sandbox.
    #
    # Mount strategy: tmpfs over cwd/.git/hooks. The directory becomes
    # writable INSIDE the sandbox (model can `chmod +x` and `echo > x`
    # without bash erroring), but every write lands on an ephemeral
    # tmpfs that disappears when the sandbox exits. The on-disk
    # .git/hooks (which only contains .sample placeholders by default)
    # is left byte-identical, so post-sandbox git invocations see no
    # active hooks.
    #
    # Skip when cwd/.git/hooks doesn't exist (non-git task dir, or
    # newly-created cwd) — the tmpfs target wouldn't have a parent.
    git_hooks = Path(cwd) / ".git" / "hooks"
    if git_hooks.is_dir():
        argv += ["--tmpfs", str(git_hooks)]
    # Apply unreadable masks after all readable mounts so a protected child
    # remains hidden even when its parent is a configured skill directory.
    if unreadable_paths:
        mask_args, _, _ = _expand_unreadable_paths(
            tuple(unreadable_paths), sandbox_required=sandbox_required, refresh=True,
            base_dir=cwd,
        )
        argv += mask_args
        if _mask_arguments is not None:
            _mask_arguments.extend(mask_args)
    argv += ["--remount-ro", "/"]
    # `set -o pipefail` so an upstream failure in `cmd1 | cmd2` is not
    # silently swallowed by a downstream `head`/`tail`/`grep`/etc. exiting
    # zero. Default bash returns the LAST command's exit code, which made
    # `python -m pytest ... | head -80` exit 0 even when `python` did not
    # exist — leaving the harness verify-gate with no exit signal to read.
    # pipefail propagates the failure of ANY pipe stage as the overall exit.
    #
    # ``tail`` (default None → policy-built non-login bash argv) lets
    # the persistent-bash path append ``bash --noprofile --norc -s``
    # instead, so a single bwrap+bash subprocess can stream commands
    # via stdin across many tool calls (PersistentBashSession in
    # ``_persistent.py``).
    if tail is None:
        argv += build_bash_argv(
            cmd, allow_login_shell=allow_login_shell,
        )
    else:
        argv += list(tail)
    return BoundTaskArgv(argv, _host_task_root, filesystem_args.resources,
                         (index + 1 for index in filesystem_args.descriptor_arguments))
