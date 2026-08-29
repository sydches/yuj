"""Durable diagnostics and mechanical repair for interrupted tool turns.

Synchronous appends require an async-writer ordering barrier. The loop and
resume layers own the policy and call sites; this module owns fsync, pending
call accounting, truncated-tail repair, and coherent resumed messages.
"""
from __future__ import annotations

import atexit
import os
import signal
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Literal

import orjson

from .trace_schema import TRACE_SCHEMA_VERSION

InterruptedTurnMode = Literal["off", "mechanical"]
ExitKind = Literal["normal", "truncated", "signal", "fatal", "process_exit"]

INTERRUPTED_TURN_MODES = frozenset({"off", "mechanical"})
EXIT_KINDS = frozenset({"normal", "truncated", "signal", "fatal", "process_exit"})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(clock: Callable[[], datetime]) -> str:
    value = clock()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _event(event_type: str, **fields: object) -> dict:
    return {
        "event": event_type,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        **fields,
    }


def append_trace_event_fsync(trace_path: Path, event: Mapping[str, object]) -> None:
    """Append one complete JSONL row and fsync it before returning."""
    trace_path = Path(trace_path)
    payload = orjson.dumps(dict(event)) + b"\n"
    fd = os.open(trace_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("zero-byte write while appending trace event")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class PendingToolCall:
    tool_call_id: str
    tool_name: str
    session_number: int
    turn_number: int
    started_at: str
    args_summary: str = ""
    intent: str = ""

    def as_trace_dict(self) -> dict[str, object]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "session_number": self.session_number,
            "turn_number": self.turn_number,
            "started_at": self.started_at,
            "args_summary": self.args_summary,
            "intent": self.intent,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PendingToolCall | None":
        call_id = str(value.get("tool_call_id") or "")
        if not call_id:
            return None
        return cls(
            tool_call_id=call_id,
            tool_name=str(value.get("tool_name") or "?"),
            session_number=int(value.get("session_number", 0) or 0),
            turn_number=int(value.get("turn_number", 0) or 0),
            started_at=str(value.get("started_at") or ""),
            args_summary=str(value.get("args_summary") or ""),
            intent=str(value.get("intent") or ""),
        )


class ExitDiagnostics:
    """Track pending calls and synchronously record exit diagnostics.

    Finish a call only after its normal result row crosses the ordering
    barrier. ``install`` covers SIGTERM, SIGINT, and atexit; the context
    manager also records fatal exceptions and normal scope exit.
    """

    def __init__(
        self,
        trace_path: Path,
        *,
        session_number: int,
        sync_before: Callable[[], None] | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.trace_path = Path(trace_path)
        self.session_number = int(session_number)
        self._sync_before = sync_before
        self._clock = clock
        self._pending: OrderedDict[str, PendingToolCall] = OrderedDict()
        self._lock = threading.RLock()
        self._exit_recorded = False
        self._installed = False
        self._previous_handlers: dict[int, object] = {}
        self._signal_handler = self._handle_signal

    def _append(self, event: Mapping[str, object]) -> None:
        if self._sync_before is not None:
            self._sync_before()
        append_trace_event_fsync(self.trace_path, event)

    def record_tool_start(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        turn_number: int,
        args_summary: str = "",
        intent: str = "",
    ) -> PendingToolCall:
        """Durably record a pending call before its implementation runs."""
        call_id = str(tool_call_id or "")
        if not call_id:
            raise ValueError("tool_call_id must be non-empty")
        pending = PendingToolCall(
            tool_call_id=call_id,
            tool_name=str(tool_name or "?"),
            session_number=self.session_number,
            turn_number=int(turn_number),
            started_at=_timestamp(self._clock),
            args_summary=str(args_summary or ""),
            intent=str(intent or ""),
        )
        with self._lock:
            if call_id in self._pending:
                raise ValueError(f"tool call {call_id!r} is already pending")
            self._append(_event("tool_start", **pending.as_trace_dict()))
            self._pending[call_id] = pending
        return pending

    def record_tool_finished(self, tool_call_id: str) -> bool:
        """Remove a call from the pending set after its result row is durable."""
        with self._lock:
            return self._pending.pop(str(tool_call_id), None) is not None

    @property
    def pending_tool_calls(self) -> tuple[PendingToolCall, ...]:
        with self._lock:
            return tuple(self._pending.values())

    def record_exit(self, *, reason: str, kind: ExitKind) -> dict | None:
        """Write at most one ``session_exit`` row for this recorder."""
        if kind not in EXIT_KINDS:
            raise ValueError(f"unsupported session exit kind: {kind!r}")
        with self._lock:
            if self._exit_recorded:
                return None
            entry = _event(
                "session_exit",
                session_number=self.session_number,
                reason=str(reason or kind),
                kind=kind,
                recorded_at=_timestamp(self._clock),
                pending_tool_calls=[call.as_trace_dict() for call in self._pending.values()],
            )
            self._append(entry)
            self._exit_recorded = True
            return entry

    def record_fatal_exception(self, exc: BaseException) -> dict | None:
        detail = f"{type(exc).__name__}: {exc}"
        return self.record_exit(reason=detail[:1000], kind="fatal")

    def install(self) -> None:
        """Install SIGTERM/SIGINT and atexit handlers for this session."""
        with self._lock:
            if self._installed:
                return
            installed: list[int] = []
            try:
                for signum in (signal.SIGTERM, signal.SIGINT):
                    self._previous_handlers[signum] = signal.getsignal(signum)
                    signal.signal(signum, self._signal_handler)
                    installed.append(signum)
                atexit.register(self._record_process_exit)
            except BaseException:
                for signum in reversed(installed):
                    signal.signal(signum, self._previous_handlers[signum])
                self._previous_handlers.clear()
                raise
            self._installed = True

    def uninstall(self) -> None:
        """Restore handlers installed by :meth:`install`."""
        with self._lock:
            if not self._installed:
                return
            for signum, previous in self._previous_handlers.items():
                if signal.getsignal(signum) == self._signal_handler:
                    signal.signal(signum, previous)
            self._previous_handlers.clear()
            atexit.unregister(self._record_process_exit)
            self._installed = False

    def _record_process_exit(self) -> None:
        self.record_exit(reason="process exit", kind="process_exit")

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        name = signal.Signals(signum).name
        self.record_exit(reason=name, kind="signal")
        previous = self._previous_handlers.get(signum, signal.SIG_DFL)
        if callable(previous):
            previous(signum, frame)
        elif previous == signal.SIG_IGN:
            return
        elif signum == signal.SIGINT:
            signal.default_int_handler(signum, frame)
        else:
            raise SystemExit(128 + int(signum))

    def __enter__(self) -> "ExitDiagnostics":
        self.install()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            if exc is None:
                self.record_exit(reason="session scope completed", kind="normal")
            else:
                self.record_fatal_exception(exc)
        finally:
            self.uninstall()
        return False


@dataclass(frozen=True)
class TracePrefix:
    events: tuple[dict, ...]
    valid_bytes: int
    total_bytes: int
    ends_with_newline: bool

    @property
    def invalid_bytes(self) -> int:
        return self.total_bytes - self.valid_bytes


def read_valid_trace_prefix(trace_path: Path) -> TracePrefix:
    """Parse complete dict rows until the first malformed JSONL record."""
    path = Path(trace_path)
    if not path.is_file():
        return TracePrefix((), 0, 0, True)
    raw = path.read_bytes()
    events: list[dict] = []
    valid_bytes = 0
    for line in raw.splitlines(keepends=True):
        body = line.rstrip(b"\r\n")
        if not body.strip():
            valid_bytes += len(line)
            continue
        try:
            value = orjson.loads(body)
        except orjson.JSONDecodeError:
            break
        if not isinstance(value, dict):
            break
        events.append(value)
        valid_bytes += len(line)
    return TracePrefix(
        events=tuple(events),
        valid_bytes=valid_bytes,
        total_bytes=len(raw),
        ends_with_newline=(not raw or raw[:valid_bytes].endswith(b"\n")),
    )


def _pending_calls(events: Iterable[Mapping[str, object]]) -> tuple[PendingToolCall, ...]:
    pending: OrderedDict[str, PendingToolCall] = OrderedDict()
    for event in events:
        event_type = event.get("event")
        if event_type == "tool_start":
            call = PendingToolCall.from_mapping(event)
            if call is not None:
                pending[call.tool_call_id] = call
        elif event_type == "tool_call":
            call_id = str(event.get("tool_call_id") or "")
            if call_id:
                pending.pop(call_id, None)
                continue
            # Compatibility while post-dispatch rows gain tool_call_id.
            for candidate_id, call in tuple(pending.items()):
                if (
                    call.tool_name == str(event.get("tool_name") or "?")
                    and call.session_number == int(event.get("session_number", 0) or 0)
                    and call.turn_number == int(event.get("turn_number", 0) or 0)
                ):
                    pending.pop(candidate_id, None)
                    break
        elif event_type == "turn_aborted":
            for call_id in event.get("interrupted_tool_call_ids") or ():
                pending.pop(str(call_id), None)
        elif event_type == "session_exit":
            for value in event.get("pending_tool_calls") or ():
                if isinstance(value, Mapping):
                    call = PendingToolCall.from_mapping(value)
                    if call is not None:
                        pending.setdefault(call.tool_call_id, call)
    return tuple(pending.values())


@dataclass(frozen=True)
class RecoveryPlan:
    recovered: bool
    pending_tool_calls: tuple[PendingToolCall, ...] = ()
    resume_prompt_line: str = ""
    appended_event: dict | None = None
    truncated_tail_bytes: int = 0


def _latest_session_segment(events: tuple[dict, ...]) -> tuple[dict, ...]:
    starts = [index for index, event in enumerate(events) if event.get("event") == "session_start"]
    return events[starts[-1]:] if starts else events


def _resume_prompt_line(pending: tuple[PendingToolCall, ...]) -> str:
    if not pending:
        return (
            "The previous session ended during a non-terminal turn. "
            "Inspect the workspace before continuing."
        )
    labels = ", ".join(
        f"{call.tool_name} (call {call.tool_call_id})" for call in pending
    )
    return (
        f"The previous session was interrupted during {labels}; each outcome is unknown. "
        "Inspect the workspace before deciding whether to retry."
    )


def _needs_recovery(segment: tuple[dict, ...], pending: tuple[PendingToolCall, ...]) -> bool:
    if not segment:
        return False
    if segment[-1].get("event") == "turn_aborted":
        return False
    if any(event.get("event") == "session_end" for event in segment):
        return False
    exits = [event for event in segment if event.get("event") == "session_exit"]
    if exits and exits[-1].get("kind") == "normal":
        return bool(pending)
    # An abnormal exit is evidence even when chat stopped before a tool ID.
    return bool(pending or exits or segment[0].get("event") == "session_start")


def _truncate_for_repair(path: Path, prefix: TracePrefix) -> None:
    with open(path, "r+b") as trace_file:
        trace_file.truncate(prefix.valid_bytes)
        trace_file.seek(prefix.valid_bytes)
        if prefix.valid_bytes and not prefix.ends_with_newline:
            trace_file.write(b"\n")
        trace_file.flush()
        os.fsync(trace_file.fileno())


def recover_interrupted_trace(
    trace_path: Path,
    *,
    mode: InterruptedTurnMode = "mechanical",
    clock: Callable[[], datetime] = _utc_now,
) -> RecoveryPlan:
    """Append one idempotent ``turn_aborted`` row for a non-terminal tail."""
    if mode not in INTERRUPTED_TURN_MODES:
        raise ValueError(f"unsupported interrupted-turn mode: {mode!r}")
    if mode == "off":
        return RecoveryPlan(recovered=False)

    path = Path(trace_path)
    prefix = read_valid_trace_prefix(path)
    segment = _latest_session_segment(prefix.events)
    pending = _pending_calls(segment)
    if not _needs_recovery(segment, pending):
        return RecoveryPlan(recovered=False, pending_tool_calls=pending)

    last_exit = next(
        (event for event in reversed(segment) if event.get("event") == "session_exit"),
        None,
    )
    session_number = max(
        (int(event.get("session_number", 0) or 0) for event in segment),
        default=0,
    )
    turn_number = max(
        (int(event.get("turn_number", 0) or 0) for event in segment),
        default=0,
    )
    reason = str((last_exit or {}).get("reason") or "unexpected process stop")
    entry = _event(
        "turn_aborted",
        session_number=session_number,
        turn_number=turn_number,
        reason=reason,
        recovery_mode="mechanical",
        recorded_at=_timestamp(clock),
        interrupted_tool_call_ids=[call.tool_call_id for call in pending],
        interrupted_tool_calls=[call.as_trace_dict() for call in pending],
    )

    if prefix.invalid_bytes or (prefix.valid_bytes and not prefix.ends_with_newline):
        _truncate_for_repair(path, prefix)
    append_trace_event_fsync(path, entry)
    return RecoveryPlan(
        recovered=True,
        pending_tool_calls=pending,
        resume_prompt_line=_resume_prompt_line(pending),
        appended_event=entry,
        truncated_tail_bytes=prefix.invalid_bytes,
    )


def build_interrupted_resume_messages(
    messages: Iterable[Mapping[str, object]],
    recovery: RecoveryPlan,
    *,
    next_user_message: str = "",
) -> list[dict]:
    """Close dangling protocol edges and append the mechanical resume prompt.

    Outcomes stay unknown: a tool may have mutated before the process stopped.
    """
    out: list[dict] = []
    unresolved: OrderedDict[str, str] = OrderedDict()

    def close_unresolved() -> None:
        for call_id, tool_name in tuple(unresolved.items()):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": (
                        f"[harness: the prior {tool_name} call was interrupted; "
                        "its outcome is unknown. Inspect workspace state before retrying.]"
                    ),
                }
            )
            unresolved.pop(call_id, None)

    for original in messages:
        message = dict(original)
        role = message.get("role")
        if role != "tool" and unresolved:
            close_unresolved()
        out.append(message)
        if role == "assistant":
            for tool_call in message.get("tool_calls") or ():
                if not isinstance(tool_call, Mapping):
                    continue
                call_id = str(tool_call.get("id") or "")
                function = tool_call.get("function") or {}
                name = str(function.get("name") or "?") if isinstance(function, Mapping) else "?"
                if call_id:
                    unresolved[call_id] = name
        elif role == "tool":
            unresolved.pop(str(message.get("tool_call_id") or ""), None)
    close_unresolved()

    if recovery.recovered:
        user_parts = [recovery.resume_prompt_line]
        if next_user_message:
            user_parts.append(next_user_message)
        out.append({"role": "user", "content": "\n\n".join(part for part in user_parts if part)})
    elif next_user_message:
        out.append({"role": "user", "content": next_user_message})
    return out
