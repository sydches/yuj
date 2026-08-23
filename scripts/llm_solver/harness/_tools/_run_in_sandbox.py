"""Shared subprocess runner used by `bash` and `run_tests`."""
import logging
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

from .._tool_filters import (
    _strip_cwd_absolute, _strip_hex_addresses,
    _strip_ls_timestamps, _strip_runner_timing,
)
from ..sandbox import (
    AMBIENT_CONTAINER, _build_bwrap_argv, container_mode,
    get_persistent_runner,
)
from ..sandbox.env_policy import (
    build_bash_argv,
    build_subprocess_env,
)

log = logging.getLogger(__name__)

# Ambient-container egress isolation. In ambient mode bwrap is bypassed
# (the outer container is supposed to provide isolation), but it may use
# the host network. Wrapping each bash call in `unshare -n`
# gives it a fresh network namespace with no interfaces (not even
# loopback), so curl/wget/pip have nowhere to go. The harness's own
# HTTP client to llama-server lives in the container's main netns
# (unaffected) because only the bash subprocess is wrapped.
#
# Requires CAP_SYS_ADMIN on the outer container. Add
# `--cap-add=SYS_ADMIN` when that container uses `docker run`. If
# `unshare` is missing or the capability is absent, Yuj uses the outer
# container network and logs a warning. The runtime record then sets
# `ambient_unshare_net=false`.
_AMBIENT_UNSHARE_PROBED = False
_AMBIENT_UNSHARE_AVAILABLE = False


def _probe_ambient_unshare_net() -> bool:
    """One-shot probe: does `unshare -n /bin/true` succeed in this env?

    Result cached at module scope. If the env disables the wrap via
    YUJ_AMBIENT_UNSHARE_NET=0, the probe is skipped and unshare is
    reported unavailable so the call site falls through.
    """
    global _AMBIENT_UNSHARE_PROBED, _AMBIENT_UNSHARE_AVAILABLE
    if _AMBIENT_UNSHARE_PROBED:
        return _AMBIENT_UNSHARE_AVAILABLE
    _AMBIENT_UNSHARE_PROBED = True
    if os.environ.get("YUJ_AMBIENT_UNSHARE_NET") == "0":
        _AMBIENT_UNSHARE_AVAILABLE = False
        log.info(
            "ambient unshare-net disabled by YUJ_AMBIENT_UNSHARE_NET=0"
        )
        return False
    try:
        r = subprocess.run(
            ["unshare", "-n", "/bin/true"],
            capture_output=True, text=True, timeout=5,
        )
        _AMBIENT_UNSHARE_AVAILABLE = (r.returncode == 0)
        if not _AMBIENT_UNSHARE_AVAILABLE:
            log.warning(
                "ambient unshare-net probe failed: rc=%s err=%r — "
                "model bash will have host-level network access. "
                "Add --cap-add=SYS_ADMIN to docker run, or set "
                "YUJ_AMBIENT_UNSHARE_NET=0 to silence this warning.",
                r.returncode, r.stderr[:200],
            )
        else:
            log.info("ambient unshare-net probe passed")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        _AMBIENT_UNSHARE_AVAILABLE = False
        log.warning(
            "ambient unshare-net probe error: %r — model bash will "
            "have host-level network access.", e,
        )
    return _AMBIENT_UNSHARE_AVAILABLE


def ambient_unshare_net_status() -> tuple[bool, bool]:
    """Return (probed, available) for the runtime_envelope record.

    Lets the session writer expose the leak-isolation state in
    .trace.jsonl so later checks can distinguish leak-closed runs
    from leak-open runs without re-probing.
    """
    return _AMBIENT_UNSHARE_PROBED, _AMBIENT_UNSHARE_AVAILABLE


def _run_in_sandbox(
    cmd: str, *, cwd: str, timeout: int, sandbox: bool,
    bwrap_bin: str, sandbox_required: bool = False,
    unreadable_paths: tuple[str, ...] = (),
    sandbox_backend: str = "bwrap",
    container_runtime: str = "docker",
    container_image: str = "",
    container_flags: tuple[str, ...] = (),
    effective_env: Mapping[str, str] | None = None,
    allow_login_shell: bool = False,
) -> tuple[str, int | None, bool]:
    """Execute a shell command and return (filtered_text, exit_code, timed_out).

    Shared by :func:`bash` and :func:`run_tests` so both surfaces use
    the same sandbox semantics, the same output strips, and the same
    exit-code/timeout discrimination. Callers decide how to render the
    triple — bash appends `[exit code: N]` on non-zero; run_tests
    wraps the triple in a structured envelope.

    Returns:
      text       — combined stdout+stderr after content-blind strips
                   (ANSI, ls timestamps, runner timing, cwd absolutes,
                   hex addresses). Empty on timeout/exception.
      exit_code  — process exit code, or ``None`` on timeout/exception.
      timed_out  — True iff the timeout fired before exit.
    """
    # Resolve container mode once — bwrap-binary check is only relevant
    # when container_mode() is None (i.e. legacy bwrap mode). Routing
    # the ambient and docker-exec branches BEFORE the bwrap-binary check
    # is the fix for the silent-failure case where the harness runs
    # inside a container that has no bwrap installed: previously the
    # check at line 36 would fall through to `sandbox_required` and
    # raise, even though the outer container is providing isolation.
    mode = container_mode() if sandbox else None
    process_env = (
        None
        if effective_env is None
        else build_subprocess_env(effective_env)
    )

    def _run_host(prefix: tuple[str, ...] = ()):
        """Run outside bwrap while preserving legacy direct-call behavior."""
        if effective_env is None and not allow_login_shell:
            if prefix:
                return subprocess.run(
                    [*prefix, "/bin/sh", "-c", cmd], cwd=cwd,
                    capture_output=True, text=True, timeout=timeout,
                )
            return subprocess.run(
                cmd, shell=True, cwd=cwd,
                capture_output=True, text=True, timeout=timeout,
            )
        return subprocess.run(
            [*prefix, *build_bash_argv(
                cmd, allow_login_shell=allow_login_shell,
            )],
            cwd=cwd, capture_output=True, text=True, timeout=timeout,
            env=process_env,
        )
    try:
        if sandbox and sandbox_backend == "container":
            if mode is not None:
                raise RuntimeError(
                    "sandbox.backend='container' cannot be combined with "
                    "legacy YUJ_CONTAINER; unset YUJ_CONTAINER or select "
                    "sandbox.backend='bwrap'"
                )
            from ..sandbox.container_backend import ContainerBackend

            backend = ContainerBackend(
                runtime=container_runtime,
                image=container_image,
                flags=container_flags,
            )
            runtime_bin = backend.resolve_runtime(
                sandbox_required=sandbox_required,
            )
            if runtime_bin is None:
                log.warning(
                    "sandbox.backend=container but runtime %s is missing — "
                    "running without sandbox because sandbox_required=false",
                    container_runtime,
                )
                result = _run_host()
            else:
                argv = backend.build_argv(
                    cmd,
                    cwd,
                    runtime_bin=runtime_bin,
                    effective_env=effective_env,
                    unreadable_paths=unreadable_paths,
                    sandbox_required=sandbox_required,
                    allow_login_shell=allow_login_shell,
                )
                result = subprocess.run(
                    argv, cwd=None,
                    capture_output=True, text=True, timeout=timeout,
                )
        elif sandbox and sandbox_backend != "bwrap":
            raise RuntimeError(
                "sandbox.backend must be 'bwrap' or 'container'; "
                f"got {sandbox_backend!r}"
            )
        elif mode == AMBIENT_CONTAINER:
            # Ambient container mode: the harness is already running
            # inside a container that provides the sandbox boundary.
            # Run bash directly. No bwrap, no docker-exec round-trip.
            #
            # Egress isolation: wrap in `unshare -n` so the bash call
            # gets a fresh empty network namespace (no loopback, no
            # external NIC). This blocks curl, wget, and package downloads.
            # Falls back to plain subprocess if unshare is unavailable
            # (probed once, cached at module scope).
            if _probe_ambient_unshare_net():
                result = _run_host(("unshare", "-n"))
            else:
                result = _run_host()
        elif mode is not None:
            # docker-exec container mode (FB testbed shape).
            argv = _build_bwrap_argv(
                cmd, cwd, bwrap_bin,
                unreadable_paths=unreadable_paths,
                sandbox_required=sandbox_required,
                effective_env=effective_env,
                allow_login_shell=allow_login_shell,
            )
            result = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout,
            )
        elif sandbox and Path(bwrap_bin).is_file():
            # Persistent bash fast path: a Session has installed a
            # long-lived bwrap+bash subprocess on the threading.local
            # registry, and the per-call cwd matches the session's
            # bound cwd. Skip the per-call subprocess.run + bwrap
            # spawn (~50–150 ms) entirely.
            #
            # Cwd mismatch ⇒ fall through. A different cwd would
            # require remounting bwrap, which the persistent path
            # cannot do (cwd is baked in at start). This is rare in
            # practice — Session.cwd is fixed for a session's
            # lifetime, and worker threads see no installed runner.
            runner = get_persistent_runner()
            if runner is not None and runner.cwd == cwd:
                out, exit_code, timed_out = runner.run(
                    cmd, cwd=cwd, timeout=timeout,
                )
                if timed_out or exit_code is None:
                    return out, exit_code, timed_out
                # Apply the same content-blind strips as the
                # subprocess path. _strip_cwd_absolute uses the
                # runner's cwd (== caller cwd here, by the gate above).
                out = _strip_ls_timestamps(out)
                out = _strip_runner_timing(out)
                out = _strip_cwd_absolute(out, cwd)
                out = _strip_hex_addresses(out)
                return out, int(exit_code), False
            argv = _build_bwrap_argv(
                cmd, cwd, bwrap_bin,
                unreadable_paths=unreadable_paths,
                sandbox_required=sandbox_required,
                effective_env=effective_env,
                allow_login_shell=allow_login_shell,
            )
            result = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout,
            )
        else:
            if sandbox:
                # Strict mode: any caller that needs guaranteed isolation
                # (sandbox_required=True) gets a hard error instead of the
                # legacy silent fall-back. Strict mode must fail instead
                # of silently running without the requested sandbox.
                if sandbox_required:
                    raise RuntimeError(
                        f"sandbox_required=true but bwrap binary {bwrap_bin!r} "
                        "is missing or not a file. Refusing to run bash "
                        "unsandboxed. Install bwrap, set sandbox_bash=false "
                        "(if no sandboxing is intentional for this run), "
                        "or set sandbox_required=false to restore the legacy "
                        "log-and-degrade behavior, or set "
                        "YUJ_CONTAINER=ambient if the harness is running "
                        "inside a container that provides isolation."
                    )
                log.warning(
                    "sandbox_bash=true but %s not found — running without sandbox",
                    bwrap_bin,
                )
            result = _run_host()
        out = result.stdout + result.stderr
        out = _strip_ls_timestamps(out)
        out = _strip_runner_timing(out)
        out = _strip_cwd_absolute(out, cwd)
        out = _strip_hex_addresses(out)
        return out, int(result.returncode), False
    except subprocess.TimeoutExpired:
        return "", None, True
    except Exception as e:
        return f"ERROR: {e}", None, False
