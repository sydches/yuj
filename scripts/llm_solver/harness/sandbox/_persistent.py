"""Persistent bash session — long-lived bwrap+bash for one harness Session.

Per-call ``subprocess.run(bwrap + bash + cmd)`` pays bwrap startup
(~50–150 ms on this host: namespace creation, mount setup, exec)
every time the model invokes bash. For a session of N turns issuing
bash on most of them, that's N × 100 ms of harness overhead the
persistent path eliminates by reusing one bwrap+bash subprocess.

Lifetime: one PersistentBashSession per harness Session. Installed in a
context variable by ``Session.run()`` and cleared on session end. Read-only
dispatch workers inherit that context and serialize access to its pipe.
Unrelated threads receive no runner.

Bwrap-mode only: ambient and docker-exec sandboxes use the per-call
path (different process model — ambient is just ``subprocess.run``,
docker-exec round-trips through ``docker exec``).

Extracted from the legacy ``sandbox.py``; see ``sandbox/__init__`` for
the argv builder this class wraps.
"""
from __future__ import annotations

import secrets
import json
import os
import select
import io
from contextlib import contextmanager
import shlex
import subprocess
import threading
from collections.abc import Mapping
from contextvars import ContextVar


_persistent_runner = ContextVar('persistent_bash_runner', default=None)


def get_persistent_runner() -> "PersistentBashSession | None":
    """Return the runner admitted to this execution context, or None."""
    return _persistent_runner.get()


def set_persistent_runner(runner: "PersistentBashSession | None") -> None:
    """Install or clear the runner in the current execution context."""
    _persistent_runner.set(runner)


# Marker prefix for command boundaries on the persistent bash stdout.
# Per-call random suffix appended for collision safety with model output.
_PERSISTENT_MARKER_PREFIX = "___YUJ_END_"


def _startup_bytes(descriptor, *, line=False):
    """Wait for startup data within the same allowance as the command."""
    from ..time_budget import command_timeout
    chunks = []
    while True:
        timeout = command_timeout()
        if not select.select([descriptor], [], [], timeout)[0]:
            raise subprocess.TimeoutExpired('persistent shell startup', timeout)
        chunk = os.read(descriptor, io.DEFAULT_BUFFER_SIZE)
        if not chunk:
            return b''.join(chunks)
        chunks.append(chunk)
        if line and b'\n' in chunk:
            return b''.join(chunks)


class PersistentBashSession:
    """Long-lived bwrap+bash subprocess for one harness Session.

    Saves per-call bwrap+bash startup overhead by reusing a single
    subprocess. Each model bash call writes a wrapper to bash's stdin:

        ( cd "$cwd" 2>/dev/null && {
            <model_cmd>
        } ) 2>&1
        __EC=$?
        find /tmp -mindepth 1 -delete 2>/dev/null
        printf '\\n%s %d\\n' '<marker>' "$__EC"

    The harness reads stdout until it sees the marker line, then
    parses the exit code that follows on the same line. Trailing
    newline blank line (from the wrapper's leading `\\n`) is stripped
    so output matches subprocess.run's stdout+stderr exactly.

    Properties preserved vs the per-call path:
      - cwd remains writable (bwrap mount baked in at start()).
      - host filesystem outside cwd remains read-only.
      - --unshare-net keeps network isolation across the session.
      - /tmp is *cleared between commands* (find -delete) so the
        per-call --tmpfs-fresh contract is preserved at the cost of
        one extra fast syscall path per call.
      - shell state (env, cwd, functions) leaks across calls in the
        outer bash, but the subshell `( … )` wrapper resets cwd and
        confines env mutations to that command's subshell. Functions
        defined in the outer bash by a previous call DO persist —
        accept this as a correctness tradeoff for the perf win; the
        SECURITY boundary is bwrap, not bash state.
      - finish_reason / exit code semantics identical to per-call.

    Concurrent run calls serialize configuration and pipe I/O under one lock.
    The owning session closes the runner after its workers finish.

    Restart on death: if bash dies (timeout-kill, OOM, etc.), the
    next call lazy-restarts via start(). The dying call returns an
    ERROR string and the harness keeps going.
    """

    def __init__(
        self,
        *,
        cwd: str,
        bwrap_bin: str | None = None,
        unreadable_paths: tuple[str, ...] = (),
        readable_paths: tuple[str, ...] = (),
        sandbox_required: bool = False,
        effective_env: Mapping[str, str] | None = None,
        allow_login_shell: bool = False,
    ) -> None:
        # bwrap_bin default resolved lazily to avoid an import cycle
        # with the package __init__ which exports this class.
        if bwrap_bin is None:
            from . import _DEFAULT_BWRAP_BIN
            bwrap_bin = _DEFAULT_BWRAP_BIN
        from ..task_path import capture_task_execution_paths, capture_host_task_root
        self.cwd, self.unreadable_paths, self.readable_paths = capture_task_execution_paths(
            cwd, unreadable_paths, readable_paths)
        self._host_task_root = capture_host_task_root(self.cwd)
        from ._filesystem import capture_frozen_filesystem_view
        self._filesystem_view = capture_frozen_filesystem_view(self._host_task_root)
        self.bwrap_bin = bwrap_bin
        self.sandbox_required = sandbox_required
        self.effective_env = dict(effective_env) if effective_env is not None else None
        self.allow_login_shell = bool(allow_login_shell)
        self._proc: subprocess.Popen | None = None
        self._mount_argv: list[str] | None = None
        self._mask_arguments: list[str] = []
        self._namespace = None
        # Lock so concurrent .run() calls (should never happen by
        # contract, but defense in depth) at least serialize on the
        # pipe instead of corrupting it.
        self._lock = threading.Lock()

    def start(self) -> None:
        """Launch the long-lived bwrap+bash subprocess. Idempotent."""
        if self._host_task_root is not None:
            self._host_task_root.verify()
        for source in getattr(self._mount_argv, 'resources', ()):
            source.verify()
        if self._proc is not None and self._proc.poll() is None:
            return
        self._kill()
        # Late import: __init__.py owns _build_bwrap_argv and imports
        # this class, so the cycle is broken by deferring this lookup
        # until call time (by which point the package is fully loaded).
        from . import _build_bwrap_argv
        from .env_policy import build_bash_argv
        masks = []
        argv = _build_bwrap_argv(
            cmd="",  # ignored when tail is set
            cwd=self.cwd,
            bwrap_bin=self.bwrap_bin,
            unreadable_paths=self.unreadable_paths,
            readable_paths=self.readable_paths,
            sandbox_required=self.sandbox_required,
            effective_env=self.effective_env,
            allow_login_shell=self.allow_login_shell,
            _mask_arguments=masks,
            _host_task_root=self._host_task_root,
            _filesystem_view=self._filesystem_view,
            tail=build_bash_argv(
                None, allow_login_shell=self.allow_login_shell,
            ),
        )
        status_read, status_write = os.pipe()
        try:
            self._proc = subprocess.Popen(
                [argv[0], '--info-fd', str(status_write), *argv[1:]],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                pass_fds=(status_write, *getattr(argv, 'pass_fds', ())),
            )
        except BaseException:
            os.close(status_read)
            raise
        finally:
            os.close(status_write)
        self._mount_argv = argv
        # Set pipefail once for the lifetime of this bash; subshells
        # inherit shell options.
        try:
            assert self._proc.stdin is not None
            marker = secrets.token_hex(16)
            self._proc.stdin.write(f"set -o pipefail; printf '%s\\n' '{marker}'\n")
            self._proc.stdin.flush()
            status = json.loads(_startup_bytes(status_read))
            os.close(status_read)
            status_read = None
            if _startup_bytes(self._proc.stdout.fileno(), line=True) != (marker + '\n').encode():
                raise RuntimeError('persistent shell did not confirm namespace readiness')
            from ._namespace_masks import NamespaceMasks, mask_targets
            self._namespace = NamespaceMasks(status, mask_targets(masks))
            self._mask_arguments = masks
        except BaseException:
            if status_read is not None:
                os.close(status_read)
            self._kill()
            raise

    def run(
        self,
        cmd: str,
        *,
        cwd: str,
        timeout: float | None,
        execution_options: dict | None = None,
    ) -> tuple[str, int | None, bool]:
        """Execute one command. Returns (stdout+stderr, exit_code, timed_out).

        Identical return shape to subprocess-based ``_run_in_sandbox``
        for submitted commands. Pre-submission exhaustion raises BudgetExhausted.
        """
        from ..time_budget import BudgetExhausted
        with self._command_lock(timeout) as acquired:
            if not acquired:
                raise BudgetExhausted('persistent command lock exhausted before submission')
            if execution_options is not None:
                self._configure(execution_options, timeout=timeout)
            try:
                self.start()
            except (subprocess.TimeoutExpired, BudgetExhausted) as error:
                raise BudgetExhausted('persistent shell startup exhausted before submission') from error
            except Exception as e:
                return f"ERROR: persistent bash failed to start: {e}", None, False

            rid = secrets.token_hex(6)
            marker = f"{_PERSISTENT_MARKER_PREFIX}{rid}___"
            # Skip /tmp clearing when cwd lives under /tmp. Bwrap binds
            # cwd over the tmpfs at the cwd path; if find -delete walks
            # in, it WILL delete files written into the bind (same
            # inode as the host file). Any cwd under /tmp opts out of
            # the per-call /tmp-fresh
            # property in exchange for not destroying their own files.
            cwd_under_tmp = cwd == "/tmp" or cwd.startswith("/tmp/")
            clear_tmp = "" if cwd_under_tmp else (
                "find /tmp -mindepth 1 -delete 2>/dev/null\n"
            )
            wrapper = (
                f"( cd {shlex.quote(cwd)} 2>/dev/null && {{\n"
                f"{cmd}\n"
                "} ) 2>&1\n"
                "__EC=$?\n"
                f"{clear_tmp}"
                f"printf '\\n%s %d\\n' '{marker}' \"$__EC\"\n"
            )
            from ..time_budget import BudgetExhausted, command_timeout
            timeout = command_timeout(timeout)
            try:
                assert self._proc is not None and self._proc.stdin is not None
                self._proc.stdin.write(wrapper)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError, AssertionError):
                self._kill()
                return (
                    "ERROR: persistent bash died on write; will restart next call",
                    None,
                    False,
                )

            timed_out = [False]

            def _kill_on_timeout() -> None:
                timed_out[0] = True
                self._kill()

            try:
                timeout = command_timeout(timeout)
            except BudgetExhausted:
                self._kill()
                return "", None, True
            timer = threading.Timer(timeout, _kill_on_timeout) if timeout is not None else None
            if timer is not None:
                timer.daemon = True
                timer.start()
            try:
                buf: list[str] = []
                assert self._proc is not None and self._proc.stdout is not None
                stdout = self._proc.stdout
                while True:
                    line = stdout.readline()
                    if not line:
                        # EOF — bash died (timeout-kill or other).
                        self._kill()
                        if timed_out[0]:
                            return "", None, True
                        return (
                            "ERROR: persistent bash died mid-command",
                            None,
                            False,
                        )
                    if line.startswith(marker + " "):
                        exit_str = line[len(marker) + 1:].strip()
                        try:
                            exit_code = int(exit_str)
                        except ValueError:
                            exit_code = -1
                        # Wrapper writes `\n<marker> <ec>\n` so the
                        # buffer's last entry is the inserted blank
                        # line — strip it so output matches
                        # subprocess.run's stdout+stderr exactly.
                        if buf and buf[-1] == "\n":
                            buf.pop()
                        return "".join(buf), exit_code, False
                    buf.append(line)
            finally:
                if timer is not None:
                    timer.cancel()

    @contextmanager
    def _command_lock(self, timeout):
        from ..time_budget import BudgetExhausted, command_time_budget, command_timeout
        with command_time_budget(0 if timeout is None else timeout):
            try:
                remaining = command_timeout()
            except BudgetExhausted:
                yield False
                return
            acquired = (self._lock.acquire() if remaining is None
                        else self._lock.acquire(timeout=remaining))
            try:
                yield acquired
            finally:
                if acquired:
                    self._lock.release()

    def _configure(self, options, *, timeout):
        """Update owned denial mounts without discarding private state."""
        from . import _build_bwrap_argv
        from .env_policy import build_bash_argv
        from ._namespace_masks import mask_targets
        masks = []
        tail = build_bash_argv(None, allow_login_shell=options['allow_login_shell'])
        argv = _build_bwrap_argv(
            '', self.cwd, **options,
            tail=tail, _mask_arguments=masks, _host_task_root=self._host_task_root,
            _filesystem_view=self._filesystem_view,
        )
        fields = ('bwrap_bin', 'unreadable_paths', 'readable_paths',
                  'effective_env', 'allow_login_shell')
        if self._mount_argv is not None:
            def without_masks(arguments, mask_arguments):
                # The builder places masks before remount-ro and the shell tail.
                end = len(arguments) - len(tail) - 2
                start = end - len(mask_arguments)
                if list(arguments[start:end]) != mask_arguments:
                    raise RuntimeError('unexpected bwrap mask layout')
                signature = list(arguments)
                for index in getattr(arguments, 'descriptor_arguments', ()):
                    source = os.fstat(int(arguments[index]))
                    signature[index] = (source.st_dev, source.st_ino)
                return tuple(signature[:start]) + tuple(signature[end:])
            if without_masks(argv, masks) != without_masks(self._mount_argv, self._mask_arguments):
                raise RuntimeError('persistent runtime mounts or environment changed within the session')
        if self._proc is not None and self._proc.poll() is None:
            self._namespace.update(mask_targets(masks), timeout=timeout)
            self._mount_argv = argv
            self._mask_arguments = masks
        for field in fields:
            value = options[field]
            if field == 'effective_env' and value is not None:
                value = dict(value)
            setattr(self, field, value)
        self.sandbox_required = options['sandbox_required']

    def run_binary(self, cmd, *, cwd, timeout, input_bytes=None,
                   execution_options=None):
        """Read and write bytes inside this shell's existing mount namespace."""
        from ._binary_transport import capture_command, captured_result
        out, code, timed_out = self.run(
            capture_command(cmd, input_bytes), cwd=cwd, timeout=timeout,
            execution_options=execution_options,
        )
        if timed_out:
            raise subprocess.TimeoutExpired(cmd, timeout)
        if code != 0:
            raise RuntimeError(f'persistent binary transport failed: {out}')
        return captured_result(cmd, out)

    def _kill(self) -> None:
        if self._namespace is not None:
            self._namespace.close()
            self._namespace = None
        if self._proc is None:
            return
        try:
            self._proc.kill()
            self._proc.wait(timeout=2)
        except Exception:
            pass
        self._proc = None

    def close(self) -> None:
        """Kill the underlying bash. Safe to call multiple times."""
        with self._lock:
            self._kill()
