"""Central-loop acceptance coverage for interrupted-turn recovery."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from _config_helpers import make_config
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._loop._session_setup import (
    inject_resume_messages,
)
from scripts.llm_solver.harness._loop.interrupted_turn import (
    PendingToolCall,
    RecoveryPlan,
)
from scripts.llm_solver.harness._loop.trace_schema import (
    TRACE_EVENT_REQUIRED_FIELDS,
)
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


def _result(*, calls=(), content="", finish_reason="tool_calls") -> TurnResult:
    return TurnResult(
        content=content,
        tool_calls=list(calls),
        finish_reason=finish_reason,
        usage=Usage(prompt_tokens=10, completion_tokens=3),
    )


def _assistant_message(content, calls):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in calls
        ]
    return message


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_interrupted_turn_config_default_and_validation(tmp_path: Path) -> None:
    assert load_config().interrupted_turn_mode == "mechanical"
    overlay = tmp_path / "mode.toml"
    overlay.write_text('[loop]\ninterrupted_turn_mode = "off"\n')
    assert load_config(user_config=overlay).interrupted_turn_mode == "off"
    overlay.write_text('[loop]\ninterrupted_turn_mode = "smart"\n')
    with pytest.raises(ValueError, match="loop.interrupted_turn_mode"):
        load_config(user_config=overlay)


def test_real_dispatch_is_durably_bracketed_and_cleared(
    tmp_path: Path,
) -> None:
    trace = tmp_path / ".trace.jsonl"
    call = ToolCall(
        id="call-write",
        name="write",
        arguments={"path": "module.py", "content": "new\n"},
    )
    client = MagicMock()
    client.chat.side_effect = [
        _result(calls=[call], content="Update it."),
        _result(content="Finished.", finish_reason="stop"),
    ]
    client.build_assistant_message.side_effect = _assistant_message

    observed_at_dispatch: list[list[str]] = []

    def dispatch_after_start(_name, _args, **_kwargs):
        observed_at_dispatch.append(
            [event["event"] for event in _events(trace)]
        )
        assert any(
            event["event"] == "tool_start"
            and event["tool_call_id"] == "call-write"
            for event in _events(trace)
        )
        return "Wrote module.py"

    with open(trace, "a") as trace_file, patch(
        "scripts.llm_solver.harness.loop.dispatch",
        side_effect=dispatch_after_start,
    ):
        result = Session(
            make_config(max_turns=3),
            client,
            "system",
            "task",
            str(tmp_path),
            trace_file=trace_file,
            trace_path=trace,
            session_number=1,
        ).run()

    assert result.done is True
    assert observed_at_dispatch
    events = _events(trace)
    kinds = [event["event"] for event in events]
    assert kinds.index("tool_start") < kinds.index("tool_call")
    result_event = next(event for event in events if event["event"] == "tool_call")
    assert result_event["tool_call_id"] == "call-write"
    exit_event = events[-1]
    assert exit_event["event"] == "session_exit"
    assert exit_event["kind"] == "normal"
    assert exit_event["pending_tool_calls"] == []


def test_checkpoint_failure_leaves_call_pending_for_fatal_exit(
    tmp_path: Path,
) -> None:
    trace = tmp_path / ".trace.jsonl"
    call = ToolCall(
        id="call-edit",
        name="edit",
        arguments={"path": "module.py", "old_string": "a", "new_string": "b"},
    )
    client = MagicMock()
    client.chat.return_value = _result(calls=[call], content="Edit it.")
    client.build_assistant_message.side_effect = _assistant_message
    store = MagicMock()
    store.capture.side_effect = RuntimeError("checkpoint disk failed")

    with open(trace, "a") as trace_file, patch(
        "scripts.llm_solver.harness.loop.dispatch",
        return_value="Updated module.py",
    ):
        session = Session(
            make_config(max_turns=1),
            client,
            "system",
            "task",
            str(tmp_path),
            trace_file=trace_file,
            trace_path=trace,
            session_number=7,
            checkpoint_store=store,
        )
        with pytest.raises(RuntimeError, match="checkpoint disk failed"):
            session.run()

    events = _events(trace)
    exit_event = events[-1]
    assert exit_event["event"] == "session_exit"
    assert exit_event["kind"] == "fatal"
    assert exit_event["pending_tool_calls"][0]["tool_call_id"] == "call-edit"


def test_parallel_reads_record_each_start_before_its_worker(
    tmp_path: Path,
) -> None:
    trace = tmp_path / ".trace.jsonl"
    calls = [
        ToolCall(id="call-a", name="read", arguments={"path": "a.txt"}),
        ToolCall(id="call-b", name="read", arguments={"path": "b.txt"}),
    ]
    client = MagicMock()
    client.chat.side_effect = [
        _result(calls=calls, content="Read both."),
        _result(content="Done.", finish_reason="stop"),
    ]
    client.build_assistant_message.side_effect = _assistant_message
    expected = {"a.txt": "call-a", "b.txt": "call-b"}

    def checked_dispatch(_name, args, **_kwargs):
        call_id = expected[args["path"]]
        assert any(
            event["event"] == "tool_start"
            and event["tool_call_id"] == call_id
            for event in _events(trace)
        )
        return args["path"]

    with open(trace, "a") as trace_file, patch(
        "scripts.llm_solver.harness.loop.dispatch",
        side_effect=checked_dispatch,
    ):
        result = Session(
            make_config(
                max_turns=3,
                parallel_readonly_enabled=True,
                parallel_max_workers=2,
            ),
            client,
            "system",
            "task",
            str(tmp_path),
            trace_file=trace_file,
            trace_path=trace,
            session_number=1,
        ).run()

    assert result.done is True
    events = _events(trace)
    assert {
        event["tool_call_id"]
        for event in events
        if event["event"] == "tool_start"
    } == {"call-a", "call-b"}
    assert events[-1]["pending_tool_calls"] == []


def test_resume_injection_uses_interrupted_unknown_outcome_message(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.log"
    assistant = _assistant_message(
        "I will edit it.",
        [ToolCall(id="call-cut", name="edit", arguments={"path": "a.py"})],
    )
    transcript.write_text(
        "=== turn 001 input ===\n"
        + json.dumps({
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
            ]
        })
        + "\n=== turn 001 output ===\n"
        + json.dumps({"choices": [{"message": assistant}]})
        + "\n"
    )
    pending = PendingToolCall(
        tool_call_id="call-cut",
        tool_name="edit",
        session_number=1,
        turn_number=1,
        started_at="2026-08-23T12:00:00Z",
    )
    recovery = RecoveryPlan(
        recovered=True,
        pending_tool_calls=(pending,),
        resume_prompt_line=(
            "The previous session was interrupted during edit (call call-cut); "
            "each outcome is unknown."
        ),
    )
    context = MagicMock()
    context.replace_all_messages.return_value = True
    session = MagicMock(context=context)

    inject_resume_messages(
        session,
        transcript,
        "Continue the task.",
        recovery=recovery,
    )

    messages = context.replace_all_messages.call_args.args[0]
    assert messages[-2]["role"] == "tool"
    assert messages[-2]["tool_call_id"] == "call-cut"
    assert "outcome is unknown" in messages[-2]["content"]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].endswith("Continue the task.")


def test_transparent_resume_drops_interrupted_generation_without_message(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.log"
    prior = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "Read it.",
            "tool_calls": [
                {
                    "id": "call-read",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": '{"path":"a.py"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-read",
            "content": "source",
        },
    ]
    transcript.write_text(
        "=== turn 002 input ===\n"
        + json.dumps({"messages": prior})
        + "\n"
    )
    recovery = RecoveryPlan(
        recovered=True,
        pending_tool_calls=(),
        resume_prompt_line=(
            "The previous session ended during a non-terminal turn."
        ),
    )
    context = MagicMock()
    context.replace_all_messages.return_value = True
    session = MagicMock(context=context)

    inject_resume_messages(
        session,
        transcript,
        None,
        recovery=recovery,
    )

    messages = context.replace_all_messages.call_args.args[0]
    assert messages == prior
    assert messages[-1]["role"] == "tool"


def test_transparent_resume_drops_interrupted_tool_call(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.log"
    prior = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    assistant = _assistant_message(
        "Edit it.",
        [ToolCall(id="call-cut", name="edit", arguments={"path": "a.py"})],
    )
    transcript.write_text(
        "=== turn 001 input ===\n"
        + json.dumps({"messages": prior})
        + "\n=== turn 001 output ===\n"
        + json.dumps({"choices": [{"message": assistant}]})
        + "\n"
    )
    pending = PendingToolCall(
        tool_call_id="call-cut",
        tool_name="edit",
        session_number=1,
        turn_number=1,
        started_at="2026-08-29T12:00:00Z",
    )
    recovery = RecoveryPlan(
        recovered=True,
        pending_tool_calls=(pending,),
        resume_prompt_line="unused in transparent mode",
    )
    context = MagicMock()
    context.replace_all_messages.return_value = True
    session = MagicMock(context=context)

    inject_resume_messages(
        session,
        transcript,
        None,
        recovery=recovery,
    )

    assert context.replace_all_messages.call_args.args[0] == prior


def test_interrupted_diagnostic_events_are_registered() -> None:
    assert {
        "tool_call_id", "tool_name", "started_at", "args_summary", "intent",
    } <= TRACE_EVENT_REQUIRED_FIELDS["tool_start"]
    assert {
        "reason", "kind", "recorded_at", "pending_tool_calls",
    } <= TRACE_EVENT_REQUIRED_FIELDS["session_exit"]
    assert {
        "reason", "recovery_mode", "interrupted_tool_call_ids",
    } <= TRACE_EVENT_REQUIRED_FIELDS["turn_aborted"]


def test_trace_backed_solve_repairs_before_loading_resume_context(
    tmp_path: Path,
) -> None:
    from scripts.llm_solver._shared.telemetry_paths import trace_path
    from scripts.llm_solver.harness.loop import solve_task

    (tmp_path / "prompt.txt").write_text("Original task")
    trace = trace_path(tmp_path)
    trace.parent.mkdir(parents=True)
    prior = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "tool_start",
            "session_number": 1,
            "turn_number": 4,
            "tool_call_id": "call-killed",
            "tool_name": "bash",
            "started_at": "2026-08-23T12:00:00Z",
            "args_summary": "cmd='build'",
            "intent": "Build it.",
        },
    ]
    trace.write_bytes(
        b"".join(
            json.dumps(event).encode() + b"\n" for event in prior
        )
        + b'{"event":"tool_call","tool_call_id":'
    )
    client = MagicMock()
    client.chat.return_value = _result(
        content="Recovered.", finish_reason="stop"
    )
    client.build_assistant_message.side_effect = _assistant_message

    with patch("scripts.llm_solver.harness.loop._auto_commit"):
        ok = solve_task(
            tmp_path,
            make_config(
                max_sessions=1,
                max_turns=1,
                sandbox_bash=False,
                state_writer_enabled=False,
            ),
            client,
            resume_from_artifacts=True,
        )

    assert ok is True
    events = _events(trace)
    aborted_index = next(
        index for index, event in enumerate(events)
        if event["event"] == "turn_aborted"
    )
    next_start_index = next(
        index for index, event in enumerate(events)
        if event.get("event") == "session_start"
        and event.get("session_number") == 2
    )
    assert aborted_index < next_start_index
    assert events[aborted_index]["interrupted_tool_call_ids"] == [
        "call-killed"
    ]
    outgoing = client.chat.call_args.args[0]
    user_text = "\n".join(
        str(message.get("content") or "")
        for message in outgoing
        if message.get("role") == "user"
    )
    assert "bash (call call-killed)" in user_text
    assert "outcome is unknown" in user_text
