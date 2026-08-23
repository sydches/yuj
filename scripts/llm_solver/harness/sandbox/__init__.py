"""Sandbox — mount-namespace enforcement for the model's bash tool.

Three modes, dispatched by the ``YUJ_CONTAINER`` env var (set by an
external launcher, never by the harness):

  - ``YUJ_CONTAINER`` unset → **bwrap mode** (legacy default).
    Bubblewrap mount namespace with ``--ro-bind / /``, ``--unshare-net``,
    and a writable ``cwd``. Host files outside ``cwd`` remain readable
    unless ``unreadable_paths`` masks them.

  - ``YUJ_CONTAINER=ambient`` → **ambient container mode**.
    The harness itself is already running inside a container that
    provides the isolation boundary (e.g. polyglot's yuj-polyglot
    image). Each bash call runs as a plain ``subprocess.run`` in the
    same container — no nested bwrap, no docker-exec round-trip. The
    outer container's mount namespace is the sandbox; we do not require
    bwrap to be installed inside it.

  - ``YUJ_CONTAINER=<container_id>`` → **docker-exec container mode**.
    Each bash call becomes ``docker exec <container_id>``. The container
    is started by the launcher (per task) from the FB testbed image,
    with ``--network none``, ``--read-only``, ``--user $UID``, and only
    ``cwd`` bind-mounted at ``/testbed``. Host paths outside cwd are
    invisible inside the container. Same image fb-eval uses, so solve
    and eval converge on one Python environment.

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
    ``_build_bwrap_argv``), docker-sock resolution, public re-exports

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
    _persistent_local,
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


_DOCKER_SOCK_CACHE: tuple[bool, str | None] = (False, None)


def _resolve_docker_sock() -> str | None:
    """Return the canonical path to the host's docker socket, or None.

    Most modern Linux distros ship /var/run as a symlink to /run, so the
    "real" socket lives at /run/docker.sock. Binding through the symlink
    (/var/run/docker.sock) fails under bwrap because the symlink target
    resolution happens after the bind is attempted, so the mount source
    reads as missing. Resolving here avoids that whole class of failure.

    Memoized for the process lifetime: the socket path is stable once
    the daemon is up, and _build_bwrap_argv runs per-bash-call, so the
    uncached version costs one stat syscall per tool invocation for no
    benefit.
    """
    global _DOCKER_SOCK_CACHE
    cached, value = _DOCKER_SOCK_CACHE
    if cached:
        return value
    for candidate in ("/run/docker.sock", "/var/run/docker.sock"):
        p = Path(candidate)
        if p.exists():
            _DOCKER_SOCK_CACHE = (True, str(p.resolve()))
            return _DOCKER_SOCK_CACHE[1]
    _DOCKER_SOCK_CACHE = (True, None)
    return None


def _build_bwrap_argv(
    cmd: str, cwd: str, bwrap_bin: str = _DEFAULT_BWRAP_BIN,
    *, unreadable_paths: tuple[str, ...] = (),
    sandbox_required: bool = False,
    tail: list[str] | None = None,
    effective_env: Mapping[str, str] | None = None,
    allow_login_shell: bool = False,
) -> list[str]:
    """Build the argv that runs `cmd` for the model's bash tool.

    Dispatches on ``YUJ_CONTAINER``:

    When set, returns a ``docker exec`` argv targeting that container.
    The launcher started the container with ``-v $cwd:/testbed``, so
    inside the container ``/testbed`` is the same bytes as ``cwd`` on
    the host. ``cwd`` (the parameter) is unused in this branch — docker
    uses ``--workdir /testbed`` and host paths outside cwd are not
    visible inside the container. ``env -i`` applies the same explicit
    command environment as every other backend before bash starts.

    When unset, returns the legacy bwrap argv (preserved below).

    bwrap-mode shape (unchanged from the pre-container era):
      - Entire host filesystem bound read-only at /
      - Fresh /tmp as tmpfs (isolated per call, no state leaks across
        tool invocations). Mounted BEFORE the cwd bind so a cwd that
        happens to live under /tmp (e.g. in tests) isn't wiped by the
        tmpfs mount.
      - cwd bound writable at its real path (matched source/target) so
        that any `docker run -v $PWD:/testbed` inside the sandbox still
        resolves correctly — the docker daemon lives on the host and
        reads HOST paths for bind mounts, so the sandbox view's $PWD
        must equal the host path. Avoid remapping cwd to /work or
        anything else; docker would then fail to find the source dir.
      - /proc and /dev for a working process view.
      - Docker socket bound in if present (resolved to the canonical
        /run path — /var/run is typically a symlink). Needed for
        pretest.sh which execs `docker run`.
      - --die-with-parent so the sandbox tears down instantly if the
        harness exits.
      - --chdir to the cwd so $PWD resolves correctly to the task dir.

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
        shell_argv = build_bash_argv(
            cmd, allow_login_shell=allow_login_shell,
        ) if tail is None else list(tail)
        return [
            "docker", "exec",
            "--workdir", "/testbed",
            mode,
            *build_clean_exec_argv(shell_argv, command_env),
        ]

    argv = [
        bwrap_bin,
        "--ro-bind", "/", "/",
        "--tmpfs", "/tmp",
        "--bind", cwd, cwd,
        "--proc", "/proc",
        "--dev", "/dev",
        # New network namespace with only loopback. Without this the model's
        # bash can pip-install or curl pre-mask source from PyPI/GitHub:
        #   pip install pkg==<task_commit>          → recovers gold impl
        #   curl raw.githubusercontent.com/<sha>/.. → recovers F2P test file
        # bypassing every mask in the benchmark. Docker socket is a UNIX
        # socket (/run/docker.sock), unaffected — pretest containers still
        # work because the docker daemon itself lives on the host with
        # network access.
        "--unshare-net",
        # Defense-in-depth namespacing. Conservative substitute for
        # Codex Tier A2 (in-process seccomp filter denying ptrace /
        # process_vm_readv / io_uring) — getting the same threat-class
        # coverage from kernel namespaces without taking on a libseccomp
        # dependency or shipping a BPF filter blob.
        #
        # --unshare-pid: fresh PID namespace. ptrace and process_vm_*
        #   syscalls require the target PID to be visible to the caller;
        #   in a separate PID-ns the sandboxed process cannot see (much
        #   less attack) any process outside the sandbox. Bwrap
        #   transparently launches an init shim as PID 1 in the new ns,
        #   so semantics are normal.
        # --unshare-ipc: fresh System-V IPC namespace. Blocks shared-
        #   memory / message-queue / semaphore tricks for cross-
        #   sandbox or sandbox-to-host data transfer.
        # --unshare-uts: fresh UTS namespace. Hostname/domainname
        #   isolation. Mostly determinism (model can't read the host's
        #   real hostname) rather than security.
        # --unshare-cgroup: fresh cgroup namespace. Blocks cgroup-escape
        #   tricks (rare, but cheap to close).
        #
        # NOT added: --unshare-user. User-namespacing changes the UID
        # mapping inside the sandbox and breaks tools that expect to
        # run as the actual UID (notably docker socket access for
        # pretest.sh). High blast radius for marginal additional
        # security beyond what unshare-pid already provides.
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup",
        "--die-with-parent",
        "--chdir", cwd,
        # The command environment is explicit and deterministic. The host
        # client keeps its own environment; only sandbox children are cleared.
        *build_bwrap_env_argv(command_env),
    ]
    sock = _resolve_docker_sock()
    if sock is not None:
        argv += ["--bind", sock, sock]
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
    # Mask answer-key files / pre-mask source trees BEFORE the cwd bind so
    # the masks survive: bwrap mounts are processed in argv order, and a
    # later --bind would otherwise overwrite an earlier --ro-bind on the
    # same target path. Cwd is always task-local so the masks (which
    # target paths under the separately configured task tree) cannot
    # collide with cwd anyway, but argv-order discipline matters if the
    # mask set ever expands.
    if unreadable_paths:
        mask_args, _, _ = _expand_unreadable_paths(
            tuple(unreadable_paths), sandbox_required=sandbox_required,
        )
        argv += mask_args
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
    return argv
