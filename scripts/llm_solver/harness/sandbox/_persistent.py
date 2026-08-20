"""Persistent bash session — long-lived bwrap+bash for one harness Session.

Per-call ``subprocess.run(bwrap + bash + cmd)`` pays bwrap startup
(~50–150 ms on this host: namespace creation, mount setup, exec)
every time the model invokes bash. For a session of N turns issuing
bash on most of them, that's N × 100 ms of harness overhead the
persistent path eliminates by reusing one bwrap+bash subprocess.

Lifetime: one PersistentBashSession per harness Session. Installed
on a ``threading.local`` registry by ``Session.run()`` so the per-call
dispatch (``_run_in_sandbox``) finds it; cleared on session end.
Worker threads (parallel-readonly dispatch) see no installed runner
and fall through to per-call ``subprocess.run`` — keeps the persistent
path single-threaded by contract instead of needing thread-safe I/O.

Bwrap-mode only: ambient and docker-exec sandboxes use the per-call
path (different process model — ambient is just ``subprocess.run``,
docker-exec round-trips through ``docker exec``).

Extracted from the legacy ``sandbox.py``; see ``sandbox/__init__`` for
the argv builder this class wraps.
"""
from __future__ import annotations

import secrets
import shlex
import subprocess
import threading


_persistent_local = threading.local()


def get_persistent_runner() -> "PersistentBashSession | None":
    """Return the active PersistentBashSession on this thread, or None."""
    return getattr(_persistent_local, "runner", None)


def set_persistent_runner(runner: "PersistentBashSession | None") -> None:
    """Install a PersistentBashSession on the current thread (or clear)."""
    _persistent_local.runner = runner


# Marker prefix for command boundaries on the persistent bash stdout.
# Per-call random suffix appended for collision safety with model output.
_PERSISTENT_MARKER_PREFIX = "___YUJ_END_"


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

    Thread-safety: methods are NOT safe for concurrent use. Each
    Session uses one instance from one thread; worker threads fall
    back via the threading.local registry above.

    Restart on death: if bash dies (timeout-kill, OOM, etc.), the
    next call lazy-restarts via start(). The dying call returns an
    ERROR string and the harness keeps going.
    """

    # Hard ceiling on the per-call timer accuracy. We don't need
    # sub-second precision; the timeout is meant to catch runaway
    # commands.
    _TIMER_GRACE_SECONDS = 1.0

    def __init__(
        self,
        *,
        cwd: str,
        bwrap_bin: str | None = None,
        unreadable_paths: tuple[str, ...] = (),
        sandbox_required: bool = False,
    ) -> None:
        # bwrap_bin default resolved lazily to avoid an import cycle
        # with the package __init__ which exports this class.
        if bwrap_bin is None:
            from . import _DEFAULT_BWRAP_BIN
            bwrap_bin = _DEFAULT_BWRAP_BIN
        self.cwd = cwd
        self.bwrap_bin = bwrap_bin
        self.unreadable_paths = unreadable_paths
        self.sandbox_required = sandbox_required
        self._proc: subprocess.Popen | None = None
        # Lock so concurrent .run() calls (should never happen by
        # contract, but defense in depth) at least serialize on the
        # pipe instead of corrupting it.
        self._lock = threading.Lock()

    def start(self) -> None:
        """Launch the long-lived bwrap+bash subprocess. Idempotent."""
        if self._proc is not None and self._proc.poll() is None:
            return
        # Late import: __init__.py owns _build_bwrap_argv and imports
        # this class, so the cycle is broken by deferring this lookup
        # until call time (by which point the package is fully loaded).
        from . import _build_bwrap_argv
        argv = _build_bwrap_argv(
            cmd="",  # ignored when tail is set
            cwd=self.cwd,
            bwrap_bin=self.bwrap_bin,
            unreadable_paths=self.unreadable_paths,
            sandbox_required=self.sandbox_required,
            tail=["bash", "--noprofile", "--norc", "-s"],
        )
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        # Set pipefail once for the lifetime of this bash; subshells
        # inherit shell options.
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write("set -o pipefail\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError, AssertionError):
            self._kill()
            raise

    def run(
        self,
        cmd: str,
        *,
        cwd: str,
        timeout: int,
    ) -> tuple[str, int | None, bool]:
        """Execute one command. Returns (stdout+stderr, exit_code, timed_out).

        Identical return shape to subprocess-based ``_run_in_sandbox``
        so callers don't need to branch.
        """
        with self._lock:
            try:
                self.start()
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

            timer = threading.Timer(
                timeout + self._TIMER_GRACE_SECONDS, _kill_on_timeout
            )
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
                        self._proc = None
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
                timer.cancel()

    def _kill(self) -> None:
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
