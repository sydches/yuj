"""One bounded pseudo-terminal process owned by an assistant session."""
from __future__ import annotations

import errno
import hashlib
import math
import os
import select
import signal
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .process_manager import (
    AdmittedProcessOutput,
    ProcessManagerError,
    build_background_sandbox_argv,
)


EventSink = Callable[[dict[str, object]], None]
ArgvBuilder = Callable[[str], Sequence[str]]
OutputAdmission = Callable[[str], str]


class TerminalProcessError(ProcessManagerError):
    """A model-actionable interactive terminal error."""


@dataclass(frozen=True)
class TerminalStart:
    terminal_id: str
    log_path: str
    result: str


@dataclass(frozen=True)
class TerminalInput:
    terminal_id: str
    input_chars: int
    input_bytes: int
    bytes_written: int
    complete: bool


@dataclass(frozen=True)
class TerminalRead:
    terminal_id: str
    result: str
    running: bool
    exit_code: int | None
    timed_out: bool
    cursor_start: int
    cursor_end: int
    output_limited: bool
    termination_reason: str


@dataclass(frozen=True)
class TerminalKill:
    terminal_id: str
    result: str
    was_running: bool
    exit_code: int | None
    reason: str


@dataclass
class _TerminalRecord:
    terminal_id: str
    command_sha256: str
    log_path: Path
    process: subprocess.Popen
    master_fd: int
    started_at: float
    cursor: int = 0
    output_bytes: int = 0
    output_limited: bool = False
    termination_reason: str = ""
    end_emitted: bool = False
    timer: threading.Timer | None = None
    reader: threading.Thread | None = None

    def running(self) -> bool:
        return self.process.poll() is None


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class TerminalProcessManager:
    """Own one live PTY at a time and drain it outside model context."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        cwd: str | Path,
        argv_builder: ArgvBuilder,
        read_timeout_s: float,
        max_lifetime_s: float,
        max_output_bytes: int,
        max_input_chars: int,
        admit_output: OutputAdmission | None = None,
        event_sink: EventSink | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        timer_factory: Callable[..., threading.Timer] = threading.Timer,
        reader_interval_s: float = 0.05,
        terminate_grace_s: float = 2.0,
    ) -> None:
        if not math.isfinite(float(read_timeout_s)) or read_timeout_s < 0:
            raise ValueError("read_timeout_s must be >= 0")
        if not math.isfinite(float(max_lifetime_s)) or max_lifetime_s <= 0:
            raise ValueError("max_lifetime_s must be > 0")
        if max_output_bytes < 1:
            raise ValueError("max_output_bytes must be >= 1")
        if max_input_chars < 1:
            raise ValueError("max_input_chars must be >= 1")
        if not math.isfinite(float(reader_interval_s)) or reader_interval_s <= 0:
            raise ValueError("reader_interval_s must be > 0")
        if (
            not math.isfinite(float(terminate_grace_s))
            or terminate_grace_s < 0
        ):
            raise ValueError("terminate_grace_s must be >= 0")
        self.run_dir = Path(run_dir).resolve()
        self.cwd = Path(cwd).resolve()
        self.argv_builder = argv_builder
        self.read_timeout_s = float(read_timeout_s)
        self.max_lifetime_s = float(max_lifetime_s)
        self.max_output_bytes = int(max_output_bytes)
        self.max_input_chars = int(max_input_chars)
        self.admit_output = admit_output or (lambda text: text)
        self.event_sink = event_sink
        self.monotonic = monotonic
        self.timer_factory = timer_factory
        self.reader_interval_s = float(reader_interval_s)
        self.terminate_grace_s = float(terminate_grace_s)
        self.terminals_dir = self.run_dir / ".terminals"
        self.terminals_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._records: dict[str, _TerminalRecord] = {}
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
    ) -> "TerminalProcessManager":
        """Construct a PTY manager that uses the ordinary shell sandbox."""
        cwd_text = str(Path(cwd).resolve())

        def argv_builder(command: str) -> list[str]:
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
                interactive=True,
            )

        return cls(
            run_dir=run_dir,
            cwd=cwd_text,
            argv_builder=argv_builder,
            **kwargs,
        )

    def _discover_next_id(self) -> int:
        highest = 0
        for path in self.terminals_dir.glob("t[0-9][0-9][0-9][0-9].log"):
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
            raise TerminalProcessError("interactive terminal manager is closed")

    def _record_for(self, terminal_id: str) -> _TerminalRecord:
        try:
            return self._records[terminal_id]
        except KeyError:
            raise TerminalProcessError(
                f"unknown interactive terminal {terminal_id!r}"
            ) from None

    @staticmethod
    def _signal(process: subprocess.Popen, sig: signal.Signals) -> None:
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

    def _finish_record(self, record: _TerminalRecord, reason: str) -> None:
        with self._condition:
            if record.end_emitted:
                return
            record.end_emitted = True
            if not record.termination_reason:
                record.termination_reason = reason
            if record.timer is not None:
                record.timer.cancel()
            elapsed = max(0.0, self.monotonic() - record.started_at)
            fields = {
                "terminal_id": record.terminal_id,
                "reason": record.termination_reason,
                "exit_code": record.process.poll(),
                "elapsed_s": round(elapsed, 6),
                "output_bytes": record.output_bytes,
                "output_limited": record.output_limited,
            }
            self._condition.notify_all()
        self._emit("terminal_end", **fields)

    def _terminate_record(self, record: _TerminalRecord, reason: str) -> bool:
        with self._condition:
            was_running = record.running()
            if was_running and not record.termination_reason:
                record.termination_reason = reason
            if record.timer is not None:
                record.timer.cancel()
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
        with self._condition:
            self._condition.notify_all()
        reader = record.reader
        if (
            reader is not None
            and reader.ident is not None
            and reader is not threading.current_thread()
        ):
            reader.join(timeout=max(0.2, self.reader_interval_s * 4))
        if reader is None or not reader.is_alive():
            self._finish_record(
                record,
                record.termination_reason or "process_exit",
            )
        return was_running

    def _lifetime_expired(self, terminal_id: str) -> None:
        with self._lock:
            record = self._records.get(terminal_id)
        if record is not None:
            self._terminate_record(record, "lifetime_timeout")

    def _reader_loop(self, record: _TerminalRecord) -> None:
        fd = record.master_fd
        try:
            while True:
                try:
                    readable, _, _ = select.select(
                        [fd], [], [], self.reader_interval_s
                    )
                except (OSError, ValueError):
                    self._terminate_record(record, "read_error")
                    break
                if readable:
                    try:
                        chunk = os.read(fd, 65536)
                    except BlockingIOError:
                        chunk = b""
                    except OSError as exc:
                        if exc.errno == errno.EIO:
                            break
                        self._terminate_record(record, "read_error")
                        break
                    if chunk:
                        with self._condition:
                            remaining = max(
                                0,
                                self.max_output_bytes - record.output_bytes,
                            )
                            kept = chunk[:remaining]
                            if kept:
                                try:
                                    with record.log_path.open(
                                        "ab", buffering=0
                                    ) as stream:
                                        stream.write(kept)
                                except OSError:
                                    record.output_limited = True
                                    self._condition.notify_all()
                                    self._terminate_record(record, "log_error")
                                    break
                                record.output_bytes += len(kept)
                            overflow = len(chunk) > len(kept)
                            if overflow:
                                record.output_limited = True
                            self._condition.notify_all()
                        if overflow:
                            self._terminate_record(record, "output_limit")
                            break
                        continue
                    if record.process.poll() is not None:
                        break
                if record.process.poll() is not None:
                    break
        finally:
            if record.process.poll() is None and not record.termination_reason:
                try:
                    record.process.wait(
                        timeout=max(0.2, self.reader_interval_s * 4)
                    )
                except (subprocess.TimeoutExpired, TimeoutError):
                    self._terminate_record(record, "read_error")
                except (ProcessLookupError, OSError):
                    pass
            try:
                os.close(fd)
            except OSError:
                pass
            with self._condition:
                record.master_fd = -1
                self._condition.notify_all()
            self._finish_record(
                record,
                record.termination_reason or "process_exit",
            )

    def start(self, command: str) -> TerminalStart:
        """Start one PTY-backed command without exposing the operator TTY."""
        self._ensure_open()
        if os.name != "posix":
            raise TerminalProcessError(
                "interactive terminals require a POSIX host"
            )
        if not command.strip():
            raise TerminalProcessError("terminal command must be non-empty")
        with self._lock:
            if any(record.running() for record in self._records.values()):
                raise TerminalProcessError(
                    "one interactive terminal is already running"
                )
            terminal_id = f"t{self._next_id:04d}"
            self._next_id += 1
        argv = list(self.argv_builder(command))
        if not argv:
            raise TerminalProcessError("terminal sandbox argv is empty")
        log_path = self.terminals_dir / f"{terminal_id}.log"
        log_path.open("xb").close()

        try:
            import fcntl
            import pty
            import termios

            master_fd, slave_fd = pty.openpty()
            fcntl.ioctl(
                slave_fd,
                termios.TIOCSWINSZ,
                struct.pack("HHHH", 24, 80, 0, 0),
            )
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=str(self.cwd),
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    start_new_session=True,
                    close_fds=True,
                )
            finally:
                os.close(slave_fd)
            os.set_blocking(master_fd, False)
        except Exception as exc:
            log_path.unlink(missing_ok=True)
            try:
                os.close(master_fd)
            except (NameError, OSError):
                pass
            raise TerminalProcessError(
                f"could not start interactive terminal: {exc}"
            ) from exc

        started_at = self.monotonic()
        record = _TerminalRecord(
            terminal_id=terminal_id,
            command_sha256=_digest(command.encode("utf-8", errors="replace")),
            log_path=log_path,
            process=process,
            master_fd=master_fd,
            started_at=started_at,
        )
        reader = threading.Thread(
            target=self._reader_loop,
            args=(record,),
            name=f"yuj-terminal-reader-{terminal_id}",
            daemon=True,
        )
        timer = self.timer_factory(
            self.max_lifetime_s,
            self._lifetime_expired,
            args=(terminal_id,),
        )
        timer.daemon = True
        record.reader = reader
        record.timer = timer
        with self._lock:
            self._records[terminal_id] = record

        relative_log = str(log_path.relative_to(self.run_dir))
        result = (
            f"Started interactive terminal {terminal_id}. "
            "Use terminal_io to send input, read output, or inspect status."
        )
        try:
            self._emit(
                "terminal_start",
                terminal_id=terminal_id,
                command_sha256=record.command_sha256,
                log_path=relative_log,
                max_lifetime_s=self.max_lifetime_s,
                max_output_bytes=self.max_output_bytes,
                result=result,
            )
        except BaseException:
            timer.cancel()
            try:
                self._signal(process, signal.SIGTERM)
                process.wait(timeout=self.terminate_grace_s)
            except (subprocess.TimeoutExpired, TimeoutError):
                self._signal(process, signal.SIGKILL)
                try:
                    process.wait(timeout=self.terminate_grace_s)
                except Exception:
                    pass
            except (ProcessLookupError, OSError):
                pass
            try:
                os.close(master_fd)
            except OSError:
                pass
            record.master_fd = -1
            record.termination_reason = "start_error"
            record.end_emitted = True
            raise
        try:
            reader.start()
            timer.start()
        except BaseException as exc:
            self._terminate_record(record, "start_error")
            if not reader.is_alive() and record.master_fd >= 0:
                try:
                    os.close(record.master_fd)
                except OSError:
                    pass
                record.master_fd = -1
            raise TerminalProcessError(
                f"could not supervise interactive terminal: {exc}"
            ) from exc
        return TerminalStart(
            terminal_id=terminal_id,
            log_path=relative_log,
            result=result,
        )

    def write(
        self,
        terminal_id: str,
        value: str,
        *,
        append_newline: bool = True,
    ) -> TerminalInput:
        """Send bounded UTF-8 input to a running terminal."""
        self._ensure_open()
        payload = (value + ("\n" if append_newline else "")).encode("utf-8")
        if len(value) > self.max_input_chars:
            self._emit(
                "terminal_input",
                terminal_id=terminal_id,
                input_sha256=_digest(payload),
                input_chars=len(value),
                input_bytes=len(payload),
                bytes_written=0,
                append_newline=bool(append_newline),
                complete=False,
                rejection="input_limit",
            )
            raise TerminalProcessError(
                "terminal input exceeds the configured character limit "
                f"({len(value)} > {self.max_input_chars})"
            )
        record = self._record_for(terminal_id)
        if not record.running() or record.master_fd < 0:
            raise TerminalProcessError(
                f"interactive terminal {terminal_id} is not running"
            )
        written = 0
        deadline = self.monotonic() + self.read_timeout_s
        while written < len(payload):
            try:
                count = os.write(record.master_fd, payload[written:])
            except BlockingIOError:
                count = 0
            except OSError as exc:
                raise TerminalProcessError(
                    f"could not write to interactive terminal {terminal_id}"
                ) from exc
            if count > 0:
                written += count
                continue
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                break
            try:
                _, writable, _ = select.select(
                    [], [record.master_fd], [], remaining
                )
            except (OSError, ValueError):
                break
            if not writable:
                break
        complete = written == len(payload)
        self._emit(
            "terminal_input",
            terminal_id=terminal_id,
            input_sha256=_digest(payload),
            input_chars=len(value),
            input_bytes=len(payload),
            bytes_written=written,
            append_newline=bool(append_newline),
            complete=complete,
        )
        result = TerminalInput(
            terminal_id=terminal_id,
            input_chars=len(value),
            input_bytes=len(payload),
            bytes_written=written,
            complete=complete,
        )
        if not complete:
            raise TerminalProcessError(
                f"terminal input timed out after {written}/{len(payload)} bytes"
            )
        return result

    @staticmethod
    def _render_read(
        terminal_id: str,
        output: str,
        *,
        exit_code: int | None,
        termination_reason: str,
        output_bytes: int,
        output_limited: bool,
    ) -> str:
        if exit_code is None:
            status = "running"
        else:
            reason = termination_reason or "process_exit"
            status = f"exited ({exit_code}; reason={reason})"
        limited = "; output_limit_reached" if output_limited else ""
        footer = (
            f"[interactive terminal {terminal_id}: {status}; "
            f"output_bytes={output_bytes}{limited}]"
        )
        return (
            f"{output}{'' if not output or output.endswith(chr(10)) else chr(10)}"
            f"{footer}"
        )

    def read(
        self,
        terminal_id: str,
        *,
        timeout_s: float | None = None,
    ) -> TerminalRead:
        """Return only unread PTY bytes and an exact current status footer."""
        self._ensure_open()
        record = self._record_for(terminal_id)
        requested = self.read_timeout_s if timeout_s is None else float(timeout_s)
        if not math.isfinite(requested) or requested < 0:
            raise TerminalProcessError(
                "terminal read timeout must be a finite number >= 0"
            )
        timeout = min(requested, self.read_timeout_s)
        deadline = self.monotonic() + timeout
        with self._condition:
            start = record.cursor
            timed_out = False
            while record.output_bytes == start and record.running():
                remaining = deadline - self.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                self._condition.wait(timeout=remaining)
            end = record.output_bytes
            exit_code = record.process.poll()
            limited = record.output_limited
            reason = record.termination_reason or (
                "process_exit" if exit_code is not None else ""
            )
        try:
            with record.log_path.open("rb") as stream:
                stream.seek(start)
                raw = stream.read(max(0, end - start))
        except OSError as exc:
            raise TerminalProcessError(
                f"could not read interactive terminal {terminal_id} output"
            ) from exc
        decoded = raw.decode("utf-8", errors="replace")
        decoded = decoded.replace("\r\n", "\n").replace("\r", "\n")
        rendered = self._render_read(
            terminal_id,
            decoded,
            exit_code=exit_code,
            termination_reason=reason,
            output_bytes=end,
            output_limited=limited,
        )
        result = AdmittedProcessOutput(self.admit_output(rendered))
        with self._condition:
            record.cursor = end
        read_result = TerminalRead(
            terminal_id=terminal_id,
            result=result,
            running=exit_code is None,
            exit_code=exit_code,
            timed_out=timed_out,
            cursor_start=start,
            cursor_end=end,
            output_limited=limited,
            termination_reason=reason,
        )
        self._emit(
            "terminal_read",
            terminal_id=terminal_id,
            result=result,
            output_sha256=_digest(result.encode("utf-8", errors="replace")),
            running=read_result.running,
            exit_code=exit_code,
            timed_out=timed_out,
            cursor_start=start,
            cursor_end=end,
            raw_output_bytes=len(raw),
            output_limited=limited,
            termination_reason=reason,
        )
        return read_result

    def kill(
        self,
        terminal_id: str,
        *,
        reason: str = "explicit",
    ) -> TerminalKill:
        """Terminate one PTY process group and retain its capped log."""
        self._ensure_open()
        record = self._record_for(terminal_id)
        was_running = self._terminate_record(record, reason)
        exit_code = record.process.poll()
        result = (
            f"Killed interactive terminal {terminal_id}"
            if was_running
            else f"Interactive terminal {terminal_id} already exited"
        )
        return TerminalKill(
            terminal_id=terminal_id,
            result=result,
            was_running=was_running,
            exit_code=exit_code,
            reason=reason,
        )

    def close(self) -> None:
        """Terminate the live terminal and close every reader. Idempotent."""
        if self._closed:
            return
        for record in tuple(self._records.values()):
            if record.running():
                self._terminate_record(record, "session_end")
            elif not record.end_emitted:
                self._finish_record(record, "process_exit")
            reader = record.reader
            if reader is not None and reader.is_alive():
                reader.join(timeout=max(0.2, self.reader_interval_s * 4))
        self._closed = True

    def __enter__(self) -> "TerminalProcessManager":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class ReplayTerminalProcessManager:
    """Replay recorded terminal tool results without starting a process."""

    _TOOL_EVENTS = ("terminal_start", "terminal_input", "terminal_read")

    def __init__(self, events: Iterable[Mapping[str, object]]) -> None:
        recorded = [dict(event) for event in events]
        self._events = [
            event
            for event in recorded
            if event.get("event") in self._TOOL_EVENTS
        ]
        self._index = 0
        self._ends = {
            str(event["terminal_id"]): event
            for event in recorded
            if event.get("event") == "terminal_end"
        }
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise TerminalProcessError("interactive terminal manager is closed")

    def _take(
        self, event_type: str, terminal_id: str | None = None
    ) -> dict[str, object]:
        if self._index >= len(self._events):
            raise TerminalProcessError(
                f"replay trace exhausted; expected {event_type}"
            )
        event = self._events[self._index]
        actual = str(event.get("event"))
        if actual != event_type:
            raise TerminalProcessError(
                f"replay terminal event mismatch: expected {event_type}, "
                f"found {actual}"
            )
        if terminal_id is not None and event.get("terminal_id") != terminal_id:
            raise TerminalProcessError(
                f"replay terminal mismatch: expected {terminal_id}, "
                f"found {event.get('terminal_id')}"
            )
        self._index += 1
        return event

    def start(self, command: str) -> TerminalStart:
        self._ensure_open()
        event = self._take("terminal_start")
        digest = _digest(command.encode("utf-8", errors="replace"))
        if event.get("command_sha256") != digest:
            raise TerminalProcessError(
                "replay interactive terminal command digest mismatch"
            )
        return TerminalStart(
            terminal_id=str(event["terminal_id"]),
            log_path=str(event["log_path"]),
            result=str(event["result"]),
        )

    def write(
        self,
        terminal_id: str,
        value: str,
        *,
        append_newline: bool = True,
    ) -> TerminalInput:
        self._ensure_open()
        event = self._take("terminal_input", terminal_id)
        payload = (value + ("\n" if append_newline else "")).encode("utf-8")
        written = int(event["bytes_written"])
        if (
            bool(event["append_newline"]) != bool(append_newline)
            or int(event["input_chars"]) != len(value)
            or int(event["input_bytes"]) != len(payload)
            or event.get("input_sha256") != _digest(payload)
        ):
            raise TerminalProcessError(
                "replay interactive terminal input digest mismatch"
            )
        result = TerminalInput(
            terminal_id=terminal_id,
            input_chars=len(value),
            input_bytes=len(payload),
            bytes_written=written,
            complete=bool(event["complete"]),
        )
        if not result.complete:
            raise TerminalProcessError(
                "terminal input timed out after "
                f"{written}/{len(payload)} bytes"
            )
        return result

    def read(
        self,
        terminal_id: str,
        *,
        timeout_s: float | None = None,
    ) -> TerminalRead:
        self._ensure_open()
        del timeout_s
        event = self._take("terminal_read", terminal_id)
        result = AdmittedProcessOutput(str(event["result"]))
        if event.get("output_sha256") != _digest(result.encode("utf-8")):
            raise TerminalProcessError(
                "replay interactive terminal output digest mismatch"
            )
        return TerminalRead(
            terminal_id=terminal_id,
            result=result,
            running=bool(event["running"]),
            exit_code=event.get("exit_code"),
            timed_out=bool(event["timed_out"]),
            cursor_start=int(event["cursor_start"]),
            cursor_end=int(event["cursor_end"]),
            output_limited=bool(event["output_limited"]),
            termination_reason=str(event["termination_reason"]),
        )

    def kill(
        self,
        terminal_id: str,
        *,
        reason: str = "explicit",
    ) -> TerminalKill:
        self._ensure_open()
        end = self._ends.get(terminal_id)
        if end is None:
            raise TerminalProcessError(
                f"replay trace has no terminal_end for {terminal_id}"
            )
        recorded_reason = str(end["reason"])
        was_running = recorded_reason == reason
        result = (
            f"Killed interactive terminal {terminal_id}"
            if was_running
            else f"Interactive terminal {terminal_id} already exited"
        )
        return TerminalKill(
            terminal_id=terminal_id,
            result=result,
            was_running=was_running,
            exit_code=end.get("exit_code"),
            reason=reason,
        )

    def close(self) -> None:
        self._closed = True

    @property
    def consumed_all(self) -> bool:
        return self._index == len(self._events)


__all__ = [
    "TerminalInput",
    "TerminalKill",
    "TerminalProcessError",
    "TerminalProcessManager",
    "ReplayTerminalProcessManager",
    "TerminalRead",
    "TerminalStart",
]
