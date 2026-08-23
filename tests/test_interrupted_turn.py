"""Byte-level contracts for interrupted-turn diagnostics and repair."""
from __future__ import annotations

import json
import os
import signal
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.llm_solver.harness._loop.interrupted_turn import (
    ExitDiagnostics,
    append_trace_event_fsync,
    build_interrupted_resume_messages,
    read_valid_trace_prefix,
    recover_interrupted_trace,
)
from scripts.llm_solver.harness.state_writer import project


NOW = datetime(2026, 8, 23, 12, 34, 56, tzinfo=timezone.utc)


def _clock() -> datetime:
    return NOW


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _start(recorder: ExitDiagnostics, *, call_id: str = "call-7") -> None:
    recorder.record_tool_start(
        tool_call_id=call_id,
        tool_name="write",
        turn_number=4,
        args_summary="path='module.py'",
        intent="Update the implementation.",
    )


def test_tool_start_is_fsynced_and_visible_before_dispatch(tmp_path, monkeypatch):
    trace = tmp_path / ".trace.jsonl"
    calls: list[tuple[str, object]] = []
    real_fsync = os.fsync

    def checked_fsync(fd: int) -> None:
        calls.append(("fsync", fd))
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", checked_fsync)
    recorder = ExitDiagnostics(trace, session_number=2, clock=_clock)
    _start(recorder)

    # A fresh descriptor sees the complete row before a tool implementation
    # would be called.  This checks both user-space flush and kernel fsync.
    observed_at_dispatch = trace.read_bytes()
    assert calls and calls[0][0] == "fsync"
    assert observed_at_dispatch.endswith(b"\n")
    event = json.loads(observed_at_dispatch)
    assert event == {
        "event": "tool_start",
        "trace_schema_version": 2,
        "tool_call_id": "call-7",
        "tool_name": "write",
        "session_number": 2,
        "turn_number": 4,
        "started_at": "2026-08-23T12:34:56.000Z",
        "args_summary": "path='module.py'",
        "intent": "Update the implementation.",
    }


def test_ordering_barrier_runs_before_durable_append(tmp_path):
    trace = tmp_path / ".trace.jsonl"
    order: list[str] = []

    def barrier() -> None:
        order.append("barrier")
        assert not trace.exists()

    recorder = ExitDiagnostics(
        trace, session_number=1, sync_before=barrier, clock=_clock
    )
    _start(recorder)
    order.append("dispatch")
    assert order == ["barrier", "dispatch"]
    assert _rows(trace)[0]["event"] == "tool_start"


@pytest.mark.parametrize(
    ("kind", "reason"),
    [("signal", "SIGTERM"), ("signal", "SIGINT"), ("fatal", "RuntimeError: boom")],
)
def test_session_exit_contains_pending_call(tmp_path, kind, reason):
    trace = tmp_path / ".trace.jsonl"
    recorder = ExitDiagnostics(trace, session_number=3, clock=_clock)
    _start(recorder)
    if kind == "fatal":
        recorder.record_fatal_exception(RuntimeError("boom"))
    else:
        recorder.record_exit(reason=reason, kind="signal")

    exit_event = _rows(trace)[-1]
    assert exit_event["event"] == "session_exit"
    assert exit_event["kind"] == kind
    assert exit_event["reason"] == reason
    assert exit_event["recorded_at"] == "2026-08-23T12:34:56.000Z"
    assert exit_event["pending_tool_calls"][0]["tool_call_id"] == "call-7"


def test_signal_handler_records_before_delegating_to_previous_handler(tmp_path):
    trace = tmp_path / ".trace.jsonl"
    delegated: list[int] = []
    recorder = ExitDiagnostics(trace, session_number=1, clock=_clock)
    _start(recorder)
    recorder._previous_handlers[signal.SIGTERM] = (
        lambda signum, _frame: delegated.append(signum)
    )

    recorder._handle_signal(signal.SIGTERM, None)

    assert _rows(trace)[-1]["kind"] == "signal"
    assert delegated == [signal.SIGTERM]


def test_context_manager_records_fatal_exception_once(tmp_path, monkeypatch):
    trace = tmp_path / ".trace.jsonl"
    recorder = ExitDiagnostics(trace, session_number=5, clock=_clock)
    monkeypatch.setattr(recorder, "install", lambda: None)
    monkeypatch.setattr(recorder, "uninstall", lambda: None)

    with pytest.raises(ValueError, match="bad state"):
        with recorder:
            _start(recorder)
            raise ValueError("bad state")

    exits = [row for row in _rows(trace) if row["event"] == "session_exit"]
    assert len(exits) == 1
    assert exits[0]["kind"] == "fatal"
    assert exits[0]["reason"] == "ValueError: bad state"


def test_finished_tool_is_not_reported_pending(tmp_path):
    trace = tmp_path / ".trace.jsonl"
    recorder = ExitDiagnostics(trace, session_number=1, clock=_clock)
    _start(recorder)
    assert recorder.record_tool_finished("call-7") is True
    assert recorder.record_tool_finished("call-7") is False
    recorder.record_exit(reason="done", kind="normal")
    assert _rows(trace)[-1]["pending_tool_calls"] == []


def test_recovery_truncates_torn_suffix_and_appends_one_aborted_row(tmp_path):
    trace = tmp_path / ".trace.jsonl"
    complete = [
        {"event": "session_start", "session_number": 4},
        {
            "event": "tool_start",
            "session_number": 4,
            "turn_number": 9,
            "tool_call_id": "call-cut",
            "tool_name": "bash",
            "started_at": "2026-08-23T12:00:00.000Z",
            "args_summary": "cmd='python build.py'",
            "intent": "Build the project.",
        },
    ]
    good_bytes = b"".join(
        json.dumps(row, separators=(",", ":")).encode() + b"\n" for row in complete
    )
    torn = b'{"event":"tool_call","tool_call_id":"call-cut","res'
    trace.write_bytes(good_bytes + torn)

    result = recover_interrupted_trace(trace, clock=_clock)

    assert result.recovered is True
    assert result.truncated_tail_bytes == len(torn)
    assert result.pending_tool_calls[0].tool_name == "bash"
    assert "bash (call call-cut)" in result.resume_prompt_line
    assert trace.read_bytes().startswith(good_bytes)
    assert torn not in trace.read_bytes()
    rows = _rows(trace)
    assert rows[-1]["event"] == "turn_aborted"
    assert rows[-1]["interrupted_tool_call_ids"] == ["call-cut"]
    assert rows[-1]["reason"] == "unexpected process stop"

    # Mechanical recovery is idempotent across repeated resume attempts.
    again = recover_interrupted_trace(trace, clock=_clock)
    assert again.recovered is False
    assert len(_rows(trace)) == 3


def test_recovery_accepts_valid_last_row_without_newline(tmp_path):
    trace = tmp_path / ".trace.jsonl"
    rows = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "session_exit",
            "session_number": 1,
            "kind": "fatal",
            "reason": "provider stream cut",
            "pending_tool_calls": [],
        },
    ]
    trace.write_bytes(b"\n".join(json.dumps(row).encode() for row in rows))
    prefix = read_valid_trace_prefix(trace)
    assert prefix.invalid_bytes == 0
    assert prefix.ends_with_newline is False

    result = recover_interrupted_trace(trace, clock=_clock)

    assert result.recovered is True
    assert trace.read_bytes().count(b"\n") == 3
    assert _rows(trace)[-1]["reason"] == "provider stream cut"


def test_normal_exit_without_pending_call_does_not_recover(tmp_path):
    trace = tmp_path / ".trace.jsonl"
    events = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "session_exit",
            "session_number": 1,
            "kind": "normal",
            "reason": "scope complete",
            "pending_tool_calls": [],
        },
    ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in events))
    before = trace.read_bytes()
    result = recover_interrupted_trace(trace, clock=_clock)
    assert result.recovered is False
    assert trace.read_bytes() == before


def test_off_mode_never_repairs_or_truncates(tmp_path):
    trace = tmp_path / ".trace.jsonl"
    trace.write_bytes(b'{"event":"session_start","session_number":1}\n{"event":')
    before = trace.read_bytes()
    result = recover_interrupted_trace(trace, mode="off", clock=_clock)
    assert result.recovered is False
    assert trace.read_bytes() == before


def test_resume_messages_close_dangling_call_without_replaying_it(tmp_path):
    trace = tmp_path / ".trace.jsonl"
    events = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "tool_start",
            "session_number": 1,
            "turn_number": 2,
            "tool_call_id": "call-write",
            "tool_name": "write",
            "started_at": "2026-08-23T12:00:00Z",
        },
    ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in events))
    recovery = recover_interrupted_trace(trace, clock=_clock)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "I will update it.",
            "tool_calls": [
                {
                    "id": "call-write",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": '{"path":"module.py","content":"new"}',
                    },
                }
            ],
        },
    ]

    resumed = build_interrupted_resume_messages(
        messages, recovery, next_user_message="Continue the task."
    )

    assert [message["role"] for message in resumed[-2:]] == ["tool", "user"]
    assert resumed[-2]["tool_call_id"] == "call-write"
    assert "outcome is unknown" in resumed[-2]["content"]
    assert "interrupted during write (call call-write)" in resumed[-1]["content"]
    assert resumed[-1]["content"].endswith("Continue the task.")
    assert sum(
        1
        for message in resumed
        if message.get("role") == "assistant" and message.get("tool_calls")
    ) == 1


def test_state_projection_ignores_aborted_tool_start(tmp_path):
    trace = tmp_path / ".trace.jsonl"
    events = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "tool_start",
            "session_number": 1,
            "turn_number": 3,
            "tool_call_id": "dangling",
            "tool_name": "edit",
            "started_at": "2026-08-23T12:00:00Z",
        },
        {
            "event": "turn_aborted",
            "session_number": 1,
            "turn_number": 3,
            "interrupted_tool_call_ids": ["dangling"],
        },
    ]
    projected = project(events, max_result_chars=1000)
    assert projected["trace"] == []
    assert projected["state"]["current_attempt"] == ""


def test_raw_durable_append_uses_single_complete_json_line(tmp_path):
    trace = tmp_path / ".trace.jsonl"
    append_trace_event_fsync(trace, {"event": "x", "bytes": "\N{SNOWMAN}"})
    assert trace.read_bytes() == b'{"event":"x","bytes":"\xe2\x98\x83"}\n'
