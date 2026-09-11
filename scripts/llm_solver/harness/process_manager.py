"""Session-scoped background process lifecycle.

Commands launch through a caller-provided sandbox argv, write combined output
to ``<run_dir>/.procs/<proc_id>.log``, and die when the session closes. Polls
return only new bytes and trace the exact result after caller-supplied output
admission, allowing replay without starting an operating-system process.
"""
from __future__ import annotations

import hashlib
from contextlib import ExitStack
import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol, Sequence


class ProcessManagerError(RuntimeError):
    """A model-actionable background process error."""


class ManagedProcess(Protocol):
    """The small ``subprocess.Popen`` surface used by the manager."""

    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


EventSink = Callable[[dict[str, object]], None]
ArgvBuilder = Callable[[str], Sequence[str]]
OutputAdmission = Callable[[str], str]


class AdmittedProcessOutput(str):
    """A process result that already passed model-output admission."""

    def __new__(cls, text, *, security_blocked_stage=""):
        value = super().__new__(cls, text)
        value.security_blocked_stage = security_blocked_stage
        return value


def _poll_output(text, exit_code, metadata):
    result = AdmittedProcessOutput(text, security_blocked_stage=metadata.get("security_blocked_stage", ""))
    result.exit_status = exit_code
    # A wait deadline is not a timeout of the underlying process.
    result.timed_out = False
    result.verification_status = metadata.get("verification_status", "")
    result.verification_evidence = metadata.get("verification_evidence")
    result.observation_receipt = metadata.get("observation_receipt")
    return result


@dataclass(frozen=True)
class ProcessStart:
    proc_id: str
    log_path: str
    result: str


@dataclass(frozen=True)
class ProcessPoll:
    proc_id: str
    result: str
    running: bool
    exit_code: int | None
    timed_out: bool
    cursor_start: int
    cursor_end: int


@dataclass(frozen=True)
class ProcessKill:
    proc_id: str
    result: str
    was_running: bool
    exit_code: int | None
    reason: str


@dataclass
class _ProcessRecord:
    proc_id: str
    command_sha256: str
    log_path: Path
    process: ManagedProcess
    cursor: int = 0
    security_blocked_stage: str = ""
    terminal_observed: bool = False
    verification_pending: bool = False
    startup_stderr_path: Path | None = None
    startup_stderr_cursor: int = 0


def _command_digest(command: str) -> str:
    return hashlib.sha256(command.encode("utf-8", errors="replace")).hexdigest()


def _result_digest(result: str) -> str:
    return hashlib.sha256(result.encode("utf-8", errors="replace")).hexdigest()


def build_background_sandbox_argv(
    command: str,
    *,
    cwd: str,
    bwrap_bin: str,
    unreadable_paths: tuple[str, ...] = (),
    readable_paths: tuple[str, ...] = (),
    sandbox_required: bool = True,
    sandbox: bool = True,
    sandbox_backend: str = "bwrap",
    container_runtime: str = "docker",
    container_runtime_bin: str = "",
    container_image: str = "",
    container_flags: tuple[str, ...] = (),
    effective_env: Mapping[str, str] | None = None,
    allow_login_shell: bool = False,
    interactive: bool = False,
    _host_task_root=None,
    _filesystem_view=None,
) -> list[str]:
    """Build a long-lived command argv under the active sandbox policy.

    Bwrap mode uses the same mount/network argv builder as ordinary ``bash``.
    Docker-exec mode is handled by that builder as well. Ambient mode mirrors
    ordinary bash's required network namespace or explicit outer-boundary policy.
    """
    from .sandbox import AMBIENT_CONTAINER, _build_bwrap_argv, container_mode
    from .sandbox.env_policy import build_bash_argv, build_clean_exec_argv
    from .task_path import active_task_host_root

    cwd = active_task_host_root(cwd) or cwd

    shell_argv = build_bash_argv(
        command, allow_login_shell=allow_login_shell,
    )

    def explicit(argv: list[str]) -> list[str]:
        return (
            argv if effective_env is None
            else build_clean_exec_argv(argv, effective_env)
        )

    if not sandbox:
        return explicit(shell_argv)
    if sandbox_backend == "container":
        if container_mode() is not None:
            raise ProcessManagerError(
                "sandbox.backend='container' cannot be combined with "
                "legacy YUJ_CONTAINER"
            )
        from .sandbox.container_backend import ContainerBackend

        backend = ContainerBackend(
            runtime=container_runtime,
            image=container_image,
            flags=container_flags,
        )
        runtime_bin = (
            container_runtime_bin
            or backend.resolve_runtime(sandbox_required=True)
        )
        assert runtime_bin is not None
        from .container_binding import bind_container_image
        backend = bind_container_image(backend, runtime_bin)
        return backend.build_argv(
            command,
            cwd,
            runtime_bin=runtime_bin,
            unreadable_paths=unreadable_paths,
            readable_paths=readable_paths,
            sandbox_required=True,
            effective_env=effective_env,
            allow_login_shell=allow_login_shell,
            interactive=interactive,
        )
    if sandbox_backend != "bwrap":
        raise ProcessManagerError(f"unknown sandbox backend {sandbox_backend!r}")
    if container_mode() == AMBIENT_CONTAINER:
        from ._tools._run_in_sandbox import _ambient_network_prefix

        prefix = _ambient_network_prefix()
        return [*prefix, *explicit(shell_argv)]
    return _build_bwrap_argv(
        command,
        cwd,
        bwrap_bin,
        unreadable_paths=unreadable_paths,
        readable_paths=readable_paths,
        sandbox_required=sandbox_required,
        effective_env=effective_env,
        allow_login_shell=allow_login_shell,
        tail=shell_argv,
        _combine_task_output=True,
        _host_task_root=_host_task_root,
        _filesystem_view=_filesystem_view,
    )


class ProcessManager:
    """Own all processes for one session; callers must ``close`` in finally."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        cwd: str | Path,
        argv_builder: ArgvBuilder,
        max_procs: int,
        poll_timeout_s: float,
        admit_output: OutputAdmission | None = None,
        event_sink: EventSink | None = None,
        poll_metadata: Callable[[str, int | None], Mapping[str, object]] | None = None,
        popen_factory: Callable[..., ManagedProcess] = subprocess.Popen,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval_s: float = 0.05,
        terminate_grace_s: float = 2.0,
    ) -> None:
        if max_procs < 1:
            raise ValueError("max_procs must be >= 1")
        if not math.isfinite(poll_timeout_s) or poll_timeout_s < 0:
            raise ValueError("poll_timeout_s must be finite and >= 0")
        if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be finite and > 0")
        if not math.isfinite(terminate_grace_s) or terminate_grace_s < 0:
            raise ValueError("terminate_grace_s must be finite and >= 0")
        self.run_dir = Path(run_dir).resolve()
        self.cwd = Path(cwd).resolve()
        self.argv_builder = argv_builder
        self.max_procs = int(max_procs)
        self.poll_timeout_s = float(poll_timeout_s)
        self.admit_output = admit_output or (lambda text: text)
        self.event_sink = event_sink
        self.poll_metadata = poll_metadata
        self.popen_factory = popen_factory
        self.monotonic = monotonic
        self.sleep = sleep
        self.poll_interval_s = float(poll_interval_s)
        self.terminate_grace_s = float(terminate_grace_s)
        from .time_budget import execution_deadline
        self._inherited_deadline = execution_deadline()
        self.procs_dir = self.run_dir / ".procs"
        self.procs_dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, _ProcessRecord] = {}
        self._next_id = self._discover_next_id()
        self._closed = False

    @classmethod
    def sandboxed(
        cls,
        *,
        run_dir: str | Path,
        cwd: str | Path,
        bwrap_bin: str,
        unreadable_paths: tuple[str, ...] = (),
        readable_paths: tuple[str, ...] = (),
        sandbox_required: bool = True,
        sandbox: bool = True,
        sandbox_backend: str = "bwrap",
        container_runtime: str = "docker",
        container_runtime_bin: str = "",
        container_image: str = "",
        container_flags: tuple[str, ...] = (),
        effective_env: Mapping[str, str] | None = None,
        allow_login_shell: bool = False,
        **kwargs,
    ) -> "ProcessManager":
        """Construct a manager whose children use the normal bash sandbox."""
        effective_env = dict(effective_env) if effective_env is not None else None
        from .sandbox import AMBIENT_CONTAINER, container_mode
        selected_mode = container_mode() if sandbox else None
        from .task_path import capture_task_execution_paths, capture_host_task_root
        cwd_text, unreadable_paths, readable_paths = capture_task_execution_paths(
            cwd, unreadable_paths, readable_paths)
        host_task_root = capture_host_task_root(
            cwd_text, sandbox=sandbox, sandbox_backend=sandbox_backend)
        from .sandbox._filesystem import capture_frozen_filesystem_view
        filesystem_view = capture_frozen_filesystem_view(host_task_root)

        def argv_builder(command: str) -> list[str]:
            if (container_mode() if sandbox else None) != selected_mode:
                from .task_environment import TaskEnvironmentUnavailable
                raise TaskEnvironmentUnavailable('task execution selection changed after manager binding')
            if host_task_root is not None:
                host_task_root.verify()
            return build_background_sandbox_argv(
                command,
                cwd=cwd_text,
                bwrap_bin=bwrap_bin,
                unreadable_paths=unreadable_paths,
                readable_paths=readable_paths,
                sandbox_required=sandbox_required,
                sandbox=sandbox,
                sandbox_backend=sandbox_backend,
                container_runtime=container_runtime,
                container_runtime_bin=container_runtime_bin,
                container_image=container_image,
                container_flags=container_flags,
                effective_env=effective_env,
                allow_login_shell=allow_login_shell,
                _host_task_root=host_task_root,
                _filesystem_view=filesystem_view,
            )

        if sandbox and sandbox_backend == 'container':
            from .container_binding import container_image_scope
            with container_image_scope() as image_binding:
                argv_builder = container_image_scope(image_binding)(argv_builder)
        elif selected_mode and selected_mode != AMBIENT_CONTAINER:
            from .task_environment import discover_task_environment, use_task_environment
            argv_builder = use_task_environment(discover_task_environment(cwd_text))(argv_builder)

        return cls(
            run_dir=run_dir,
            cwd=cwd_text,
            argv_builder=argv_builder,
            **kwargs,
        )

    def _discover_next_id(self) -> int:
        highest = 0
        for path in self.procs_dir.glob("p[0-9][0-9][0-9][0-9].log"):
            try:
                highest = max(highest, int(path.stem[1:]))
            except ValueError:
                continue
        return highest + 1

    def _emit(self, event: str, **fields: object) -> None:
        if self.event_sink is not None:
            self.event_sink({"event": event, **fields})

    def _ensure_open(self) -> None:
        if self._closed:
            raise ProcessManagerError("background process manager is closed")

    def _record_for(self, proc_id: str) -> _ProcessRecord:
        try:
            return self._records[proc_id]
        except KeyError:
            raise ProcessManagerError(
                f"unknown background process {proc_id!r}"
            ) from None

    def has_pending_observations(self) -> bool:
        """Keep work pending until its terminal outcome has been observed."""
        return any(not record.terminal_observed for record in self._records.values())

    def _running_count(self) -> int:
        return sum(record.process.poll() is None for record in self._records.values())

    def start(self, command: str) -> ProcessStart:
        """Start ``command`` without waiting for it to finish."""
        self._ensure_open()
        if not command.strip():
            raise ProcessManagerError("background command must be non-empty")
        if self._running_count() >= self.max_procs:
            raise ProcessManagerError(
                f"background process limit reached ({self.max_procs})"
            )

        proc_id = f"p{self._next_id:04d}"
        self._next_id += 1
        log_path = self.procs_dir / f"{proc_id}.log"
        from .process_identity import GuardedProcessArgv
        built_argv = self.argv_builder(command)
        guarded = isinstance(built_argv, GuardedProcessArgv)
        argv = list(built_argv)
        if not argv:
            raise ProcessManagerError("background sandbox argv is empty")

        # Exclusive creation prevents a resumed session from silently
        # overwriting a prior process log if an ID allocator ever regresses.
        # Keep transport diagnostics separate from the stdout startup frame.
        # The guarded command merges task stderr after verifying credentials.
        startup_stderr_path = log_path.with_suffix('.stderr') if guarded else None
        startup_stderr_created = False
        with ExitStack() as streams:
            log_stream = streams.enter_context(log_path.open('xb', buffering=0))
            try:
                stderr = subprocess.STDOUT
                if startup_stderr_path is not None:
                    stderr = streams.enter_context(startup_stderr_path.open('xb', buffering=0))
                    startup_stderr_created = True
                process = self.popen_factory(
                    argv,
                    cwd=str(self.cwd),
                    stdin=subprocess.DEVNULL,
                    stdout=log_stream,
                    stderr=stderr,
                    start_new_session=True,
                    close_fds=True,
                    **({'pass_fds': built_argv.pass_fds}
                       if getattr(built_argv, 'pass_fds', ()) else {}),
                )
            except Exception as exc:
                log_path.unlink(missing_ok=True)
                if startup_stderr_created:
                    startup_stderr_path.unlink(missing_ok=True)
                raise ProcessManagerError(
                    f"could not start background process: {exc}"
                ) from exc

        command_sha256 = _command_digest(command)
        self._records[proc_id] = _ProcessRecord(
            proc_id=proc_id,
            command_sha256=command_sha256,
            log_path=log_path,
            process=process,
            verification_pending=guarded,
            startup_stderr_path=startup_stderr_path,
        )
        relative_log = str(log_path.relative_to(self.run_dir))
        result = (
            f"Started background process {proc_id}. "
            f"Poll it with bash_poll(proc_id={proc_id!r})."
        )
        self._emit(
            "proc_start",
            proc_id=proc_id,
            command_sha256=command_sha256,
            log_path=relative_log,
            result=result,
        )
        return ProcessStart(proc_id=proc_id, log_path=relative_log, result=result)

    @staticmethod
    def _render_poll(proc_id: str, output: str, exit_code: int | None) -> str:
        status = "running" if exit_code is None else f"exited ({exit_code})"
        footer = f"[background process {proc_id}: {status}]"
        return f"{output}{'' if not output or output.endswith(chr(10)) else chr(10)}{footer}"

    def poll(self, proc_id: str, *, timeout_s: float | None = None) -> ProcessPoll:
        """Read available state; bound an optional wait by declared deadlines."""
        self._ensure_open()
        record = self._record_for(proc_id)
        requested = self.poll_timeout_s if timeout_s is None else float(timeout_s)
        if not math.isfinite(requested) or requested < 0:
            raise ProcessManagerError("poll timeout must be finite and >= 0")
        timeout = min(requested, self.poll_timeout_s) if self.poll_timeout_s > 0 else requested
        started = self.monotonic()
        from .time_budget import execution_deadline
        bounds = [d for d in (self._inherited_deadline, execution_deadline()) if d is not None]
        inherited = min(bounds) if bounds else None
        deadline = min(started + timeout, inherited) if inherited is not None else started + timeout
        wait_budget = {
            "requested_seconds": requested, "configured_cap_seconds": self.poll_timeout_s or None,
            "remaining_execution_seconds": max(0.0, inherited - started) if inherited is not None else None,
            "effective_seconds": max(0.0, deadline - started),
            "limiting_source": "execution_deadline" if inherited is not None and inherited <= started + timeout
                               else "configured_cap" if timeout < requested else "requested_wait",
        }
        start = record.cursor
        stderr_start = record.startup_stderr_cursor
        raw = b""
        timed_out = False
        from .process_identity import GuardedProcessArgv, ProcessVerification, ProcessIdentityError

        while True:
            # If exit is observed, read after it so the last child write is
            # included before deciding whether startup completed.
            exit_code = record.process.poll()
            with record.log_path.open("rb") as stream:
                stream.seek(start)
                raw = stream.read()
            startup_stderr = b''
            if record.startup_stderr_path is not None:
                with record.startup_stderr_path.open('rb') as stream:
                    stream.seek(stderr_start)
                    startup_stderr = stream.read()
            verification = ProcessVerification(GuardedProcessArgv([]) if record.verification_pending else [])
            verification_error = ''
            try:
                output = verification.feed(raw)
                if exit_code is not None:
                    verification.finish()
            except ProcessIdentityError as error:
                verification_error = str(error)
                output = b''
            if (verification_error or (verification.verified and (raw or startup_stderr))
                    or exit_code is not None):
                break
            now = self.monotonic()
            if now >= deadline:
                timed_out = True
                break
            self.sleep(min(self.poll_interval_s, max(0.0, deadline - now)))

        observed = verification.verified or bool(verification_error)
        end = start + len(raw) if observed else start
        stderr_end = stderr_start + len(startup_stderr) if observed else stderr_start
        if verification_error:
            decoded = 'ERROR: ' + verification_error + '\n'
        elif verification.verified:
            decoded = (output + startup_stderr).decode('utf-8', errors='replace')
        else:
            decoded = ''
        rendered = self._render_poll(proc_id, decoded, exit_code)
        # Advance only after admission succeeds; a failing admission callback
        # may be retried without silently losing process bytes.
        metadata = (dict(self.poll_metadata(proc_id, exit_code))
                    if self.poll_metadata and verification.verified else {})
        if record.startup_stderr_path is not None:
            metadata.update(process_identity_verified=verification.verified,
                            startup_stderr_cursor_start=stderr_start,
                            startup_stderr_cursor_end=stderr_end)
            if not verification.verified:
                metadata['verification_status'] = ('process_identity_unverified' if verification_error
                                                   else 'process_identity_pending')
        wait_budget["elapsed_seconds"] = max(0.0, self.monotonic() - started)
        metadata["wait_budget"] = wait_budget
        admitted = self.admit_output(rendered)
        record.security_blocked_stage = getattr(admitted, "security_blocked_stage", "") or record.security_blocked_stage
        if record.security_blocked_stage:
            metadata["security_blocked_stage"] = record.security_blocked_stage
        from .repeated_observations import process_observation
        metadata["observation_receipt"] = process_observation(proc_id, exit_code is None, start, end, exit_code)
        result = _poll_output(admitted, exit_code, metadata)
        record.cursor = end
        record.startup_stderr_cursor = stderr_end
        record.verification_pending = not verification.verified
        record.terminal_observed = exit_code is not None
        poll_result = ProcessPoll(
            proc_id=proc_id,
            result=result,
            running=exit_code is None,
            exit_code=exit_code,
            timed_out=timed_out,
            cursor_start=start,
            cursor_end=end,
        )
        self._emit(
            "proc_poll",
            proc_id=proc_id,
            result=result,
            output_sha256=_result_digest(result),
            running=poll_result.running,
            exit_code=exit_code,
            timed_out=timed_out,
            cursor_start=start,
            cursor_end=end,
            execution_metadata=metadata,
            wait_budget=wait_budget,
        )
        return poll_result

    @staticmethod
    def _signal(process: ManagedProcess, sig: signal.Signals) -> None:
        pid = getattr(process, "pid", None)
        if isinstance(pid, int) and pid > 0:
            try:
                os.killpg(pid, sig)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()

    def kill(self, proc_id: str, *, reason: str = "explicit") -> ProcessKill:
        """Terminate one process group, escalating to SIGKILL after grace."""
        self._ensure_open()
        record = self._record_for(proc_id)
        was_running = record.process.poll() is None
        if was_running:
            try:
                self._signal(record.process, signal.SIGTERM)
                record.process.wait(timeout=self.terminate_grace_s)
            except (subprocess.TimeoutExpired, TimeoutError):
                self._signal(record.process, signal.SIGKILL)
                try:
                    record.process.wait(timeout=self.terminate_grace_s)
                except Exception:
                    pass
            except (ProcessLookupError, OSError):
                pass
        exit_code = record.process.poll()
        record.terminal_observed = exit_code is not None
        result = (
            f"Requested termination of background process {proc_id}; exit not confirmed"
            if exit_code is None
            else f"Killed background process {proc_id}"
            if was_running else f"Background process {proc_id} already exited"
        )
        self._emit(
            "proc_kill",
            proc_id=proc_id,
            result=result,
            was_running=was_running,
            exit_code=exit_code,
            reason=reason,
        )
        return ProcessKill(
            proc_id=proc_id,
            result=result,
            was_running=was_running,
            exit_code=exit_code,
            reason=reason,
        )

    def close(self) -> None:
        """Kill all live children.  Safe to call more than once."""
        if self._closed:
            return
        for proc_id, record in tuple(self._records.items()):
            if record.process.poll() is None:
                self.kill(proc_id, reason="session_end")
        self._closed = True

    def __enter__(self) -> "ProcessManager":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class ReplayProcessManager:
    """Replay ``proc_*`` events without creating operating-system processes."""

    def __init__(self, events: Iterable[Mapping[str, object]]) -> None:
        self._events = [
            dict(event)
            for event in events
            if event.get("event") in {"proc_start", "proc_poll", "proc_kill"}
        ]
        self._index = 0
        self._closed = False

    def has_pending_observations(self) -> bool:
        pending = {}
        for event in self._events[:self._index]:
            pending[event.get("proc_id")] = (
                event["event"] == "proc_start"
                or event["event"] == "proc_poll" and event.get("running") is not False
                or event["event"] == "proc_kill" and event.get("exit_code") is None
            )
        return any(pending.values())

    def _take(self, expected: str, proc_id: str | None = None) -> dict[str, object]:
        if self._index >= len(self._events):
            raise ProcessManagerError(f"replay trace exhausted; expected {expected}")
        event = self._events[self._index]
        actual = event.get("event")
        if actual != expected:
            raise ProcessManagerError(
                f"replay event mismatch: expected {expected}, found {actual}"
            )
        if proc_id is not None and event.get("proc_id") != proc_id:
            raise ProcessManagerError(
                f"replay proc mismatch: expected {proc_id}, "
                f"found {event.get('proc_id')}"
            )
        self._index += 1
        return event

    def start(self, command: str) -> ProcessStart:
        event = self._take("proc_start")
        expected_digest = event.get("command_sha256")
        if expected_digest != _command_digest(command):
            raise ProcessManagerError("replay background command digest mismatch")
        return ProcessStart(
            proc_id=str(event["proc_id"]),
            log_path=str(event["log_path"]),
            result=str(event["result"]),
        )

    def poll(self, proc_id: str, *, timeout_s: float | None = None) -> ProcessPoll:
        del timeout_s  # recorded result is authoritative
        event = self._take("proc_poll", proc_id)
        result = _poll_output(str(event["result"]), event.get("exit_code"), event.get("execution_metadata", {}))
        if event.get("output_sha256") != _result_digest(result):
            raise ProcessManagerError("replay poll output digest mismatch")
        return ProcessPoll(
            proc_id=proc_id,
            result=result,
            running=bool(event["running"]),
            exit_code=event.get("exit_code"),
            timed_out=bool(event["timed_out"]),
            cursor_start=int(event["cursor_start"]),
            cursor_end=int(event["cursor_end"]),
        )

    def kill(self, proc_id: str, *, reason: str = "explicit") -> ProcessKill:
        event = self._take("proc_kill", proc_id)
        recorded_reason = str(event["reason"])
        if recorded_reason != reason:
            raise ProcessManagerError(
                f"replay kill reason mismatch: expected {reason}, "
                f"found {recorded_reason}"
            )
        return ProcessKill(
            proc_id=proc_id,
            result=str(event["result"]),
            was_running=bool(event["was_running"]),
            exit_code=event.get("exit_code"),
            reason=recorded_reason,
        )

    def close(self) -> None:
        if self._closed:
            return
        while self._index < len(self._events):
            event = self._events[self._index]
            if event.get("event") != "proc_kill" or event.get("reason") != "session_end":
                break
            self.kill(str(event["proc_id"]), reason="session_end")
        self._closed = True

    @property
    def consumed_all(self) -> bool:
        return self._index == len(self._events)


__all__ = [
    "AdmittedProcessOutput", "ProcessKill", "ProcessManager",
    "ProcessManagerError", "ProcessPoll", "ProcessStart",
    "ReplayProcessManager", "build_background_sandbox_argv",
]
