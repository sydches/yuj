"""Shared subprocess runner used by `bash` and `run_tests`."""
import logging
import os
import shutil
import signal
import subprocess
from collections.abc import Mapping
from pathlib import Path

from .._tool_filters import (
    _normalize_memory_addresses, _strip_cwd_absolute,
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
from ..task_environment import TaskEnvironmentUnavailable
from ..process_identity import ProcessIdentityError
from ..time_budget import (
    BudgetExhausted, command_timeout, command_is_scoped,
    execution_deadline, remaining_before,
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
# The launcher must permit namespace creation and capability restriction.
# After unshare, setpriv removes CAP_SYS_ADMIN from the child so it cannot
# rejoin the parent's network namespace. Both executables must be trusted
# and protected by the outer launcher. A failed probe refuses the command.
# A launcher can explicitly disable this inner boundary with
# YUJ_AMBIENT_UNSHARE_NET=0 when it supplies the outer network boundary.
_AMBIENT_UNSHARE_PROBED = False
_AMBIENT_UNSHARE_AVAILABLE = False
_AMBIENT_UNSHARE_PREFIX: tuple[str, ...] = ()


class SandboxUnavailableError(RuntimeError):
    """The selected execution boundary is unavailable."""


def _execute(argv, *, timeout, cwd=None, env=None, input_bytes=None, binary=False,
             pass_fds=()):
    from ..process_identity import verified_process_result
    from ..time_budget import execution_deadline, remaining_before
    pass_fds = tuple(dict.fromkeys((*pass_fds, *getattr(argv, 'pass_fds', ()))))
    timeout = remaining_before(execution_deadline(), command_timeout(timeout))
    if not binary and (not command_is_scoped() or timeout is None):
        result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True,
                                text=True, timeout=timeout,
                                **({'pass_fds': pass_fds} if pass_fds else {}))
        return verified_process_result(argv, result)
    # A test runner may fork children that inherit its output pipes. Killing
    # only the shell and then draining those pipes can outlive the allowance.
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=not binary,
                               **({'stdin': subprocess.PIPE} if binary else {}),
                               **({'pass_fds': pass_fds} if pass_fds else {}),
                               start_new_session=True)
    try:
        timeout = remaining_before(execution_deadline(), command_timeout(timeout))
        if binary:
            out, err = process.communicate(input=input_bytes, timeout=timeout)
        else:
            out, err = process.communicate(timeout=timeout)
        return verified_process_result(argv, subprocess.CompletedProcess(argv, process.returncode, out, err))
    except BaseException as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        # Do not wait for pipe EOF from a descendant that detached itself.
        process.stdout.close()
        process.stderr.close()
        if binary and process.stdin is not None:
            process.stdin.close()
        process.wait()
        if isinstance(error, (subprocess.TimeoutExpired, BudgetExhausted)):
            raise subprocess.TimeoutExpired(argv, timeout) from None
        raise


def _legacy_container_running(container_id: str, *, timeout=None) -> bool | None:
    """Check a failed ``docker exec`` target without trusting its stderr."""
    from ..time_budget import execution_deadline, remaining_before
    try:
        probe = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container_id],
            capture_output=True,
            text=True,
            timeout=remaining_before(execution_deadline(), timeout),
        )
    except (OSError, subprocess.TimeoutExpired, BudgetExhausted):
        return None
    return probe.returncode == 0 and probe.stdout.strip().lower() == "true"


def _probe_ambient_unshare_net() -> bool:
    """Probe namespace creation followed by child privilege restriction.

    Cache the capability and its executable together. An explicit outer-boundary
    policy skips the probe without caching a capability failure.
    """
    global _AMBIENT_UNSHARE_PROBED, _AMBIENT_UNSHARE_AVAILABLE
    global _AMBIENT_UNSHARE_PREFIX
    if os.environ.get("YUJ_AMBIENT_UNSHARE_NET") == "0":
        log.info(
            "ambient unshare-net disabled by YUJ_AMBIENT_UNSHARE_NET=0"
        )
        return False
    if _AMBIENT_UNSHARE_PROBED:
        return _AMBIENT_UNSHARE_AVAILABLE
    _AMBIENT_UNSHARE_AVAILABLE = False
    _AMBIENT_UNSHARE_PREFIX = ()
    try:
        executables = [shutil.which(name) for name in ("unshare", "setpriv")]
        if any(path is None for path in executables):
            raise FileNotFoundError("unshare or setpriv is unavailable in the harness environment")
        unshare, setpriv = (str(Path(path).resolve()) for path in executables)
        prefix = (unshare, "-n", setpriv, "--bounding-set=-sys_admin",
                  "--inh-caps=-sys_admin", "--ambient-caps=-sys_admin",
                  "--no-new-privs")
        r = subprocess.run(
            [*prefix, "/bin/true"], capture_output=True, text=True,
            timeout=remaining_before(execution_deadline()),
        )
        remaining_before(execution_deadline())
        _AMBIENT_UNSHARE_AVAILABLE = (r.returncode == 0)
        if not _AMBIENT_UNSHARE_AVAILABLE:
            log.warning(
                "ambient unshare-net probe failed: rc=%s err=%r; "
                "refusing commands that require network isolation.",
                r.returncode, r.stderr[:200],
            )
        else:
            _AMBIENT_UNSHARE_PREFIX = prefix
            log.info("ambient unshare-net probe passed: %s", prefix)
    except subprocess.TimeoutExpired as e:
        # An incomplete observation is not a cached capability failure.
        raise BudgetExhausted("ambient network probe exhausted execution time budget") from e
    except OSError as e:
        _AMBIENT_UNSHARE_AVAILABLE = False
        log.warning(
            "ambient unshare-net probe error: %r; refusing commands "
            "that require network isolation.", e,
        )
    _AMBIENT_UNSHARE_PROBED = True
    return _AMBIENT_UNSHARE_AVAILABLE


def _ambient_network_prefix() -> tuple[str, ...]:
    """Enforce the ambient network policy for every task execution path."""
    if os.environ.get("YUJ_AMBIENT_UNSHARE_NET") == "0":
        return ()
    if not _probe_ambient_unshare_net() or not _AMBIENT_UNSHARE_PREFIX:
        raise SandboxUnavailableError("ambient network isolation is unavailable")
    return _AMBIENT_UNSHARE_PREFIX


def ambient_unshare_net_status() -> tuple[bool, bool]:
    """Return the cached inner capability as (probed, available).

    These flags do not establish the outer container's network policy
    or prove that any particular command was launched.
    """
    return _AMBIENT_UNSHARE_PROBED, _AMBIENT_UNSHARE_AVAILABLE


def _run_in_sandbox(
    cmd: str, *, cwd: str, timeout: float | None, sandbox: bool,
    bwrap_bin: str, sandbox_required: bool = False,
    unreadable_paths: tuple[str, ...] = (),
    readable_paths: tuple[str, ...] = (),
    sandbox_backend: str = "bwrap",
    container_runtime: str = "docker",
    container_runtime_bin: str = "",
    container_image: str = "",
    container_flags: tuple[str, ...] = (),
    effective_env: Mapping[str, str] | None = None,
    allow_login_shell: bool = False,
    normalize_output: bool = True,
    normalize_addresses: bool = True,
    raw_result: bool = False,
    input_bytes: bytes | None = None,
    use_persistent: bool = True,
    _host_task_root=None,
    _filesystem_view=None,
) -> tuple[str, int | None, bool] | subprocess.CompletedProcess:
    """Execute a shell command and return (filtered_text, exit_code, timed_out).

    Shared by :func:`bash` and :func:`run_tests` so both surfaces use
    the same sandbox semantics, the same output strips, and the same
    exit-code/timeout discrimination. Callers decide how to render the
    triple — bash appends `[exit code: N]` on non-zero; run_tests
    wraps the triple in a structured envelope.

    Internal file operations select ``raw_result`` to preserve separate binary
    streams and native errors. The persistent shell encodes these streams for
    transport; neither path applies output transformations.

    Returns:
      text       — combined stdout+stderr. Memory addresses are normalized
                   unless ``normalize_addresses`` is false for structured
                   inspection. When ``normalize_output`` is true, other
                   content-blind strips remove ls timestamps, runner timing,
                   and cwd absolutes. Empty on timeout/error.
      exit_code  — process exit code, or ``None`` on timeout/exception.
      timed_out  — True iff the timeout fired before exit.
    """
    from ..task_path import active_task_host_root
    cwd = active_task_host_root(cwd) or cwd
    # Resolve container mode once — bwrap-binary check is only relevant
    # when container_mode() is None (i.e. legacy bwrap mode). Routing
    # the ambient and docker-exec branches BEFORE the bwrap-binary check
    # is the fix for the silent-failure case where the harness runs
    # inside a container that has no bwrap installed: previously the
    # check at line 36 would fall through to `sandbox_required` and
    # raise, even though the outer container is providing isolation.
    mode = container_mode() if sandbox else None
    legacy_target = None
    process_env = (
        None
        if effective_env is None
        else build_subprocess_env(effective_env)
    )
    execution_options = {'binary': True, 'input_bytes': input_bytes} if raw_result else {}

    def _run_host(prefix: tuple[str, ...] = ()):
        """Run outside bwrap with the same pipefail contract as other backends."""
        return _execute(
            [*prefix, *build_bash_argv(
                cmd, allow_login_shell=allow_login_shell,
            )],
            cwd=cwd, timeout=timeout,
            env=process_env,
            **execution_options,
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

            from ..time_budget import command_time_budget
            with command_time_budget(0 if timeout is None else timeout):
                backend = ContainerBackend(
                    runtime=container_runtime, image=container_image,
                    flags=container_flags,
                )
                runtime_bin = container_runtime_bin or backend.resolve_runtime(sandbox_required=True)
                assert runtime_bin is not None
                from ..container_binding import bind_container_image
                backend = bind_container_image(backend, runtime_bin, timeout=timeout)
                argv = backend.build_argv(
                    cmd, cwd, runtime_bin=runtime_bin,
                    effective_env=effective_env,
                    unreadable_paths=unreadable_paths,
                    readable_paths=readable_paths,
                    sandbox_required=True,
                    allow_login_shell=allow_login_shell,
                )
                if raw_result:
                    argv.insert(2, '-i')
                result = _execute(argv, cwd=None, timeout=timeout, **execution_options)
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
            # An explicit launcher opt-out can rely on an outer network
            # boundary. Probe failure must never act as that opt-out.
            result = _run_host(_ambient_network_prefix())
        elif mode is not None:
            # docker-exec container mode (FB testbed shape).
            from ..task_environment import discover_task_environment
            legacy_target = discover_task_environment(cwd, timeout=timeout).container_id
            argv = _build_bwrap_argv(
                cmd, cwd, bwrap_bin,
                unreadable_paths=unreadable_paths,
                readable_paths=readable_paths,
                sandbox_required=sandbox_required,
                effective_env=effective_env,
                allow_login_shell=allow_login_shell,
            )
            if raw_result:
                argv.insert(2, '-i')
            result = _execute(
                argv, timeout=timeout, **execution_options,
            )
        elif sandbox and Path(bwrap_bin).is_file():
            # Persistent bash fast path: a Session has installed a
            # long-lived bwrap+bash subprocess in the current context,
            # and the per-call cwd matches the session's
            # bound cwd. Skip the per-call subprocess.run + bwrap
            # spawn (~50–150 ms) entirely.
            #
            # Cwd mismatch ⇒ fall through. A different cwd would
            # require remounting bwrap, which the persistent path
            # cannot do (cwd is baked in at start). This is rare in
            # practice — Session.cwd is fixed for a session's
            # lifetime. Read workers inherit the runner and serialize its I/O.
            runner = get_persistent_runner() if use_persistent else None
            if runner is not None and runner.cwd == cwd:
                from ..task_path import capture_host_task_root
                from ..sandbox._filesystem import (
                    capture_frozen_filesystem_view, resolve_filesystem_view,
                )
                selected_view = _filesystem_view
                if selected_view is None:
                    selected_view = capture_frozen_filesystem_view(
                        _host_task_root or capture_host_task_root(cwd))

                def admission(view):
                    resolved = resolve_filesystem_view(view)
                    return resolved if resolved is not None else view

                if admission(selected_view) is not admission(runner._filesystem_view):
                    # Matching cwd alone cannot authorize another task's
                    # runtime resources. Use the caller's normal native launch.
                    runner = None
            if runner is not None and runner.cwd == cwd:
                runner_options = dict(
                    bwrap_bin=bwrap_bin,
                    unreadable_paths=tuple(unreadable_paths),
                    readable_paths=tuple(readable_paths),
                    sandbox_required=sandbox_required,
                    effective_env=effective_env,
                    allow_login_shell=allow_login_shell,
                )
                if raw_result:
                    return runner.run_binary(
                        cmd, cwd=cwd, timeout=command_timeout(timeout),
                        input_bytes=input_bytes, execution_options=runner_options,
                    )
                out, exit_code, timed_out = runner.run(
                    cmd, cwd=cwd, timeout=command_timeout(timeout),
                    execution_options=runner_options,
                )
                if timed_out or exit_code is None:
                    return (
                        _normalize_memory_addresses(out) if normalize_addresses else out,
                        exit_code,
                        timed_out,
                    )
                # Apply the same content-blind strips as the
                # subprocess path. _strip_cwd_absolute uses the
                # runner's cwd (== caller cwd here, by the gate above).
                if normalize_output:
                    out = _strip_ls_timestamps(out)
                    out = _strip_runner_timing(out)
                    out = _strip_cwd_absolute(out, cwd)
                if normalize_addresses:
                    out = _normalize_memory_addresses(out)
                return out, int(exit_code), False
            argv = _build_bwrap_argv(
                cmd, cwd, bwrap_bin,
                unreadable_paths=unreadable_paths,
                readable_paths=readable_paths,
                sandbox_required=sandbox_required,
                effective_env=effective_env,
                allow_login_shell=allow_login_shell,
                _host_task_root=_host_task_root,
                _filesystem_view=_filesystem_view,
            )
            result = _execute(
                argv, timeout=timeout, **execution_options,
            )
        else:
            if sandbox:
                raise RuntimeError(
                    f"selected sandbox backend 'bwrap' is missing or unavailable at "
                    f"{bwrap_bin!r}. Refusing to substitute another backend "
                    "or run unsandboxed; select sandbox.backend='none' "
                    "explicitly if host execution is intended."
                )
            result = _run_host()
        if mode not in {None, AMBIENT_CONTAINER} and result.returncode != 0:
            running = _legacy_container_running(legacy_target, timeout=timeout)
            if running is False:
                raise SandboxUnavailableError(
                    "the selected container sandbox is unavailable"
                )
            if running is None:
                log.warning("container status diagnostic unavailable after command completion: %s",
                            legacy_target)
        if raw_result:
            return result
        out = result.stdout + result.stderr
        if normalize_output:
            out = _strip_ls_timestamps(out)
            out = _strip_runner_timing(out)
            out = _strip_cwd_absolute(out, cwd)
        if normalize_addresses:
            out = _normalize_memory_addresses(out)
        return out, int(result.returncode), False
    except subprocess.TimeoutExpired:
        if raw_result:
            raise
        return "", None, True
    except (SandboxUnavailableError, BudgetExhausted):
        raise
    except (TaskEnvironmentUnavailable, ProcessIdentityError) as e:
        raise SandboxUnavailableError(str(e)) from e
    except Exception as e:
        if raw_result:
            raise
        return _normalize_memory_addresses(f"ERROR: {e}"), None, False
