"""Structured assistant clarification state-machine acceptance tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from _config_helpers import make_config
from scripts.llm_assist.__main__ import (
    _smoke_acceptance_check,
    main as assist_main,
)
from scripts.llm_assist.runner import derive_live_state, save_approval_request
from scripts.llm_assist.store import SessionStore
from scripts.llm_solver.harness.clarifications import (
    ClarificationStateError,
    clarification_answer_path,
    clarification_consumption_path,
    clarification_request_path,
    clarification_state,
    consume_clarification_answer,
    create_clarification_request,
    load_clarification_answer,
    load_clarification_consumption,
    load_clarification_request,
    record_clarification_answer,
    supersede_clarification_for_rewind,
)
from scripts.llm_solver.harness._loop.profile_resolution import build_tool_surface
from scripts.llm_solver.harness._loop._driver_setup import (
    _rotate_assistant_transcript,
)
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.server.replay_client import (
    ReplayClient,
    ReplayDivergence,
    resolve_replay_source,
)
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


QUESTION = "Which database should the migration target?"
ANSWER = "Use the existing PostgreSQL database exactly as configured."


class _SequenceClient:
    def __init__(self, *turns: TurnResult):
        self.turns = list(turns)
        self.requests: list[dict] = []

    def chat(self, messages, tools, turn=0):
        self.requests.append({
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
            "turn": turn,
        })
        return self.turns.pop(0)

    def build_assistant_message(self, content, tool_calls):
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in tool_calls
            ]
        return message

    def query_server_context(self):
        return 0


def _turn(*calls: ToolCall, content: str = "I need one fact.") -> TurnResult:
    return TurnResult(
        content=content,
        tool_calls=list(calls),
        finish_reason="tool_calls" if calls else "stop",
        usage=Usage(prompt_tokens=10, completion_tokens=3),
    )


def _session_record(store: SessionStore, cwd: Path):
    return store.create_session(
        cwd=cwd,
        model="test-model",
        prompt_text="Complete the task.",
        prompt_source="test",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )


def _seed_request(record, *, request_id: str = "request-45") -> dict:
    return create_clarification_request(
        record.artifact_path,
        request_id=request_id,
        session_id=record.session_id,
        session_number=1,
        turn_number=2,
        tool_call_id="ask-45",
        question=QUESTION,
    )


def test_assistant_question_pauses_before_sibling_mutation_and_second_model_call(
    tmp_path: Path,
) -> None:
    target = tmp_path / "settings.py"
    target.write_text("DATABASE = 'sqlite'\n")
    trace_path = tmp_path / ".trace.jsonl"
    client = _SequenceClient(
        _turn(
            ToolCall("ask-45", "ask_user", {"question": QUESTION}),
            ToolCall(
                "edit-after-ask",
                "edit",
                {
                    "path": "settings.py",
                    "old_str": "sqlite",
                    "new_str": "postgresql",
                },
            ),
        ),
        _turn(content="This must not be requested."),
    )
    cfg = make_config(
        runtime_mode="assistant",
        max_turns=3,
        guardrails_arm_after_turn=99,
        rumination_nudge_threshold=999,
    )

    with trace_path.open("a") as trace, patch.object(
        Session, "_get_server_ctx", return_value=cfg.context_size
    ):
        session = Session(
            cfg,
            client,
            "system",
            "task",
            str(tmp_path),
            trace_file=trace,
            trace_path=trace_path,
            artifact_dir=tmp_path,
            session_number=1,
        )
        result = session.run()

    assert result.finish_reason == "input_required"
    assert result.done is False
    assert len(client.requests) == 1
    assert "ask_user" in {
        schema["function"]["name"] for schema in client.requests[0]["tools"]
    }
    assert target.read_text() == "DATABASE = 'sqlite'\n"
    request = load_clarification_request(tmp_path)
    assert request is not None
    assert request["question"] == QUESTION
    assert request["tool_call_id"] == "ask-45"
    assert request["status"] == "pending"
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    requested = [event for event in events if event["event"] == "clarification_request"]
    assert len(requested) == 1
    assert requested[0]["question"] == QUESTION
    assert not any(
        event.get("event") == "tool_call"
        and event.get("tool_name") == "edit"
        for event in events
    )


@pytest.mark.parametrize(
    "arguments,keyword",
    [
        ({}, "required"),
        ({"question": QUESTION, "choices": ["a", "b"]}, "additionalProperties"),
        ({"question": ""}, "minLength"),
    ],
)
def test_question_schema_rejects_missing_extra_and_empty_input(
    tmp_path: Path, arguments: dict, keyword: str,
) -> None:
    trace_path = tmp_path / ".trace.jsonl"
    client = _SequenceClient(
        _turn(ToolCall("bad-ask", "ask_user", arguments)),
        _turn(content="done"),
    )
    cfg = make_config(
        runtime_mode="assistant",
        max_turns=3,
        tools_schema_validation="off",
        guardrails_arm_after_turn=99,
        rumination_nudge_threshold=999,
    )
    with trace_path.open("a") as trace, patch.object(
        Session, "_get_server_ctx", return_value=cfg.context_size
    ):
        session = Session(
            cfg,
            client,
            "system",
            "task",
            str(tmp_path),
            trace_file=trace,
            trace_path=trace_path,
            artifact_dir=tmp_path,
        )
        result = session.run()

    assert result.finish_reason == "stop"
    assert not clarification_request_path(tmp_path).exists()
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    rejection = next(event for event in events if event["event"] == "schema_reject")
    assert any(error["keyword"] == keyword for error in rejection["errors"])


def test_two_questions_are_rejected_as_one_ambiguous_turn(
    tmp_path: Path,
) -> None:
    target = tmp_path / "settings.py"
    target.write_text("value = 1\n")
    trace_path = tmp_path / ".trace.jsonl"
    client = _SequenceClient(
        _turn(
            ToolCall("ask-one", "ask_user", {"question": "First?"}),
            ToolCall("ask-two", "ask_user", {"question": "Second?"}),
            ToolCall(
                "edit-sibling",
                "edit",
                {"path": "settings.py", "old_str": "1", "new_str": "2"},
            ),
        ),
        _turn(content="done"),
    )
    cfg = make_config(
        runtime_mode="assistant",
        max_turns=2,
        guardrails_arm_after_turn=99,
        rumination_nudge_threshold=999,
    )
    with trace_path.open("a") as trace, patch.object(
        Session, "_get_server_ctx", return_value=cfg.context_size
    ):
        result = Session(
            cfg,
            client,
            "system",
            "task",
            str(tmp_path),
            trace_file=trace,
            trace_path=trace_path,
            artifact_dir=tmp_path,
        ).run()

    assert result.finish_reason == "stop"
    assert target.read_text() == "value = 1\n"
    assert not clarification_request_path(tmp_path).exists()
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    rejects = [event for event in events if event["event"] == "schema_reject"]
    assert len(rejects) == 2
    assert all(
        event["errors"][0]["path"] == "$.tool_calls"
        and event["errors"][0]["keyword"] == "maxItems"
        for event in rejects
    )


def test_measurement_has_no_question_schema_and_cannot_pause_for_input(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / ".trace.jsonl"
    client = _SequenceClient(
        _turn(ToolCall("forged-ask", "ask_user", {"question": QUESTION})),
        _turn(content="done"),
    )
    cfg = make_config(
        runtime_mode="measurement",
        max_turns=3,
        tools_schema_validation="reject",
        guardrails_arm_after_turn=99,
        rumination_nudge_threshold=999,
    )
    with trace_path.open("a") as trace, patch.object(
        Session, "_get_server_ctx", return_value=cfg.context_size
    ):
        session = Session(
            cfg,
            client,
            "system",
            "task",
            str(tmp_path),
            trace_file=trace,
            trace_path=trace_path,
            artifact_dir=tmp_path,
        )
        result = session.run()

    assert result.finish_reason == "stop"
    assert "ask_user" not in {
        schema["function"]["name"] for schema in client.requests[0]["tools"]
    }
    assert not clarification_request_path(tmp_path).exists()
    assert not any(
        json.loads(line).get("event") == "clarification_request"
        for line in trace_path.read_text().splitlines()
    )


def test_code_mode_keeps_question_only_for_assistant(tmp_path: Path) -> None:
    assistant = Session(
        make_config(runtime_mode="assistant", tools_exec_cell_enabled=True),
        _SequenceClient(),
        "system",
        "task",
        str(tmp_path),
    )
    measurement = Session(
        make_config(runtime_mode="measurement", tools_exec_cell_enabled=True),
        _SequenceClient(),
        "system",
        "task",
        str(tmp_path),
    )

    assistant_names = {
        schema["function"]["name"] for schema in assistant._tool_schemas
    }
    measurement_names = {
        schema["function"]["name"] for schema in measurement._tool_schemas
    }
    assert "ask_user" in assistant_names
    assert "ask_user" not in measurement_names


def test_question_is_not_exposed_to_an_assistant_child_agent(
    tmp_path: Path,
) -> None:
    child = Session(
        make_config(runtime_mode="assistant"),
        _SequenceClient(),
        "system",
        "task",
        str(tmp_path),
        subagent_level=1,
    )

    assert "ask_user" not in {
        schema["function"]["name"] for schema in child._tool_schemas
    }


def test_answer_command_records_exact_answer_once_and_refuses_wrong_request(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _session_record(store, tmp_path / "work")
    request = _seed_request(record)
    store.update_session(
        record.session_id,
        status="input_required",
        last_finish_reason="input_required",
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        with pytest.raises(SystemExit, match="does not match"):
            assist_main([
                "answer", record.session_id, "wrong-request", "wrong answer"
            ])

    assert not clarification_answer_path(record.artifact_path).exists()
    before_trace = (record.artifact_path / ".trace.jsonl").read_bytes() \
        if (record.artifact_path / ".trace.jsonl").exists() else b""

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert assist_main([
            "answer", record.short_id, request["request_id"], ANSWER
        ]) == 0

    answer = load_clarification_answer(record.artifact_path)
    assert answer is not None
    assert answer["answer"] == ANSWER
    assert clarification_state(record.artifact_path).phase == "input_ready"
    updated = store.get_session(record.session_id)
    assert updated is not None
    assert updated.status == "input_ready"
    assert updated.last_finish_reason == "input_answered"

    answer_bytes = clarification_answer_path(record.artifact_path).read_bytes()
    trace_bytes = (record.artifact_path / ".trace.jsonl").read_bytes()
    assert trace_bytes != before_trace
    assert ANSWER.encode() not in trace_bytes
    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        with pytest.raises(SystemExit, match="already has an answer"):
            assist_main([
                "answer", record.session_id, request["request_id"], "second"
            ])
    assert clarification_answer_path(record.artifact_path).read_bytes() == answer_bytes
    assert (record.artifact_path / ".trace.jsonl").read_bytes() == trace_bytes


def test_answer_for_wrong_session_and_answer_when_none_pending_do_not_mutate(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    first = _session_record(store, tmp_path / "first")
    second = _session_record(store, tmp_path / "second")
    request = _seed_request(first)

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        with pytest.raises(SystemExit, match="no pending clarification"):
            assist_main([
                "answer", second.session_id, request["request_id"], ANSWER
            ])
    assert not clarification_answer_path(first.artifact_path).exists()
    assert not clarification_answer_path(second.artifact_path).exists()


def test_status_and_show_expose_exact_question_and_answer_command(
    tmp_path: Path, capsys,
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _session_record(store, tmp_path / "work")
    request = _seed_request(record)
    store.update_session(
        record.session_id,
        status="input_required",
        last_finish_reason="input_required",
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert assist_main(["status", record.short_id]) == 0
    status_out = capsys.readouterr().out
    assert "status: input_required" in status_out
    assert "finish_reason: input_required" in status_out
    assert f"question: {QUESTION}" in status_out
    assert "approval: none" in status_out
    assert (
        f"next: yuj answer {record.short_id} {request['request_id']} '<answer>'"
        in status_out
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert assist_main([
            "show", record.short_id, "--trace-lines", "0", "--turns", "0"
        ]) == 0
    show_out = capsys.readouterr().out
    assert "clarification: pending" in show_out
    assert f"clarification_request_id: {request['request_id']}" in show_out
    assert f"clarification_question: {QUESTION}" in show_out
    assert "approval: none" in show_out
    assert (
        f"next: yuj answer {record.short_id} {request['request_id']} '<answer>'"
        in show_out
    )


def test_pending_input_blocks_resume_and_survives_interrupt_marker(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _session_record(store, tmp_path / "work")
    request = _seed_request(record)
    store.update_session(
        record.session_id,
        status="input_required",
        last_finish_reason="input_required",
    )
    (record.artifact_path / "shell_interrupt.json").write_text(json.dumps({
        "finish_reason": "interrupted",
        "interrupted_at": "2026-08-24T00:00:00+00:00",
    }))

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), patch(
        "scripts.llm_assist.__main__.run_session"
    ) as run:
        with pytest.raises(SystemExit, match="pending clarification"):
            assist_main(["resume", record.session_id])
    run.assert_not_called()
    assert load_clarification_request(record.artifact_path) == request
    assert derive_live_state(record.artifact_path).status == "input_required"


def test_pending_input_fails_the_smoke_acceptance_gate(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _session_record(store, tmp_path / "work")
    _seed_request(record)

    accepted, reasons = _smoke_acceptance_check(tmp_path / "work", record)

    assert accepted is False
    assert "session has an unresolved clarification exchange" in reasons


def test_answer_does_not_satisfy_or_change_pending_approval(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _session_record(store, tmp_path / "work")
    request = _seed_request(record)
    approval = {
        "status": "pending",
        "action_key": "bash:sha256:" + "a" * 64,
        "tool_name": "bash",
        "args_summary": "cmd='rm -rf build'",
        "cmd": "rm -rf build",
        "reason": "destructive file deletion via rm",
    }
    save_approval_request(record.artifact_path, approval)
    approval_bytes = (record.artifact_path / "approval_request.json").read_bytes()

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert assist_main([
            "answer", record.session_id, request["request_id"], ANSWER
        ]) == 0
        with pytest.raises(SystemExit, match="pending approval request"):
            assist_main(["resume", record.session_id])

    assert (record.artifact_path / "approval_request.json").read_bytes() == approval_bytes


def test_recorded_answer_is_added_once_then_consumed_before_model_request(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    request = create_clarification_request(
        artifact_dir,
        request_id="request-once",
        session_id="session-once",
        session_number=1,
        turn_number=0,
        tool_call_id="ask-once",
        question=QUESTION,
    )
    record_clarification_answer(
        artifact_dir,
        session_id="session-once",
        request_id=request["request_id"],
        answer=ANSWER,
    )
    trace_path = artifact_dir / ".trace.jsonl"
    cfg = make_config(
        runtime_mode="assistant",
        max_turns=1,
        guardrails_arm_after_turn=99,
        rumination_nudge_threshold=999,
    )
    first_client = _SequenceClient(_turn(content="done"))
    with trace_path.open("a") as trace, patch.object(
        Session, "_get_server_ctx", return_value=cfg.context_size
    ):
        first = Session(
            cfg,
            first_client,
            "system",
            "resume",
            str(tmp_path),
            trace_file=trace,
            trace_path=trace_path,
            artifact_dir=artifact_dir,
            session_number=2,
        )
        first_result = first.run()

    assert first_result.finish_reason == "stop"
    sent = first_client.requests[0]["messages"]
    assert [message.get("content") for message in sent].count(ANSWER) == 1
    assert any(
        QUESTION in str(message.get("content") or "")
        and "does not approve" in str(message.get("content") or "")
        for message in sent
    )
    consumption = load_clarification_consumption(artifact_dir)
    assert consumption is not None
    assert consumption["request_id"] == request["request_id"]
    assert consumption["delivery"] == "resume"

    second_client = _SequenceClient(_turn(content="done again"))
    with trace_path.open("a") as trace, patch.object(
        Session, "_get_server_ctx", return_value=cfg.context_size
    ):
        second = Session(
            cfg,
            second_client,
            "system",
            "resume again",
            str(tmp_path),
            trace_file=trace,
            trace_path=trace_path,
            artifact_dir=artifact_dir,
            session_number=3,
        )
        second.run()
    assert ANSWER not in {
        message.get("content") for message in second_client.requests[0]["messages"]
    }


def _response(*, call: ToolCall | None = None, content: str | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if call is not None:
        message["tool_calls"] = [{
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(call.arguments),
            },
        }]
    return {
        "choices": [{
            "message": message,
            "finish_reason": "tool_calls" if call is not None else "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }


def _write_transcript(path: Path, *, messages: list[dict], tools: list[dict], response: dict) -> None:
    path.write_text(
        "=== turn 001 input ===\n"
        + json.dumps({"messages": messages, "tools": tools})
        + "\n=== turn 001 output ===\n"
        + json.dumps(response)
        + "\n"
    )


def test_offline_replay_consumes_answer_without_reopening_question(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_trace = source / ".trace.jsonl"
    source_request = create_clarification_request(
        source,
        request_id="replay-request",
        session_id="source-session",
        session_number=1,
        turn_number=0,
        tool_call_id="ask-replay",
        question=QUESTION,
    )
    source_answer = record_clarification_answer(
        source,
        session_id="source-session",
        request_id=source_request["request_id"],
        answer=ANSWER,
    )
    consume_clarification_answer(
        source,
        request_id=source_request["request_id"],
        session_number=2,
        turn_number=0,
        delivery="resume",
    )
    source_trace.write_text("".join(json.dumps(event) + "\n" for event in [
        {
            "event": "clarification_request",
            "session_number": 1,
            "turn_number": 0,
            "request_id": source_request["request_id"],
            "tool_call_id": "ask-replay",
            "question": QUESTION,
        },
        {
            "event": "clarification_answer",
            "session_number": 1,
            "turn_number": 0,
            "request_id": source_request["request_id"],
            "answer_sha256": source_answer["answer_sha256"],
            "answer_chars": len(ANSWER),
        },
        {
            "event": "clarification_consumed",
            "session_number": 2,
            "turn_number": 0,
            "request_id": source_request["request_id"],
            "answer_sha256": source_answer["answer_sha256"],
            "delivery": "resume",
        },
    ]))

    assistant_cfg = make_config(runtime_mode="assistant")
    measurement_cfg = make_config(
        runtime_mode="measurement",
        max_turns=3,
        tools_schema_validation="reject",
        guardrails_arm_after_turn=99,
        rumination_nudge_threshold=999,
    )
    schema_client = _SequenceClient()
    recorded_tools = build_tool_surface(
        assistant_cfg, schema_client
    ).active_schemas
    recorded_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    resumed_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "resume"},
        {"role": "user", "content": ANSWER},
    ]
    _write_transcript(
        source / "transcript.pre_seg_1.log",
        messages=recorded_messages,
        tools=recorded_tools,
        response=_response(
            call=ToolCall("ask-replay", "ask_user", {"question": QUESTION})
        ),
    )
    _write_transcript(
        source / "transcript.log",
        messages=resumed_messages,
        tools=recorded_tools,
        response=_response(content="done"),
    )

    replay = ReplayClient(
        source / "transcript.log",
        source_trace_path=source_trace,
    )
    destination = tmp_path / "replay"
    destination.mkdir()
    destination_trace = destination / ".trace.jsonl"
    with destination_trace.open("a") as trace, patch.object(
        Session, "_get_server_ctx", return_value=measurement_cfg.context_size
    ):
        session = Session(
            measurement_cfg,
            replay,
            "system",
            "task",
            str(tmp_path),
            trace_file=trace,
            trace_path=destination_trace,
            artifact_dir=destination,
            session_number=1,
        )
        result = session.run()

    assert result.finish_reason == "stop"
    assert replay.served_turns == 2
    assert not clarification_request_path(destination).exists()
    assert not clarification_answer_path(destination).exists()
    assert not clarification_consumption_path(destination).exists()
    replay_events = [
        json.loads(line) for line in destination_trace.read_text().splitlines()
    ]
    assert [
        event["event"] for event in replay_events
        if event["event"].startswith("clarification_")
    ] == [
        "clarification_request",
        "clarification_answer",
        "clarification_consumed",
    ]
    assert all(
        event.get("replayed") is True
        for event in replay_events
        if event["event"].startswith("clarification_")
    )
    assert next(
        event for event in replay_events
        if event["event"] == "clarification_consumed"
    )["delivery"] == "replay"


def test_replay_refuses_a_question_without_recorded_answer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    request = create_clarification_request(
        source,
        request_id="pending-replay",
        session_id="source-session",
        session_number=1,
        turn_number=0,
        tool_call_id="ask-pending",
        question=QUESTION,
    )
    trace = source / ".trace.jsonl"
    trace.write_text(json.dumps({
        "event": "clarification_request",
        "session_number": 1,
        "turn_number": 0,
        "request_id": request["request_id"],
        "tool_call_id": request["tool_call_id"],
        "question": request["question"],
    }) + "\n")
    _write_transcript(
        source / "transcript.log",
        messages=[{"role": "user", "content": "task"}],
        tools=[],
        response=_response(
            call=ToolCall("ask-pending", "ask_user", {"question": QUESTION})
        ),
    )

    with pytest.raises(
        ReplayDivergence,
        match="requires a recorded and consumed clarification answer",
    ):
        ReplayClient(source / "transcript.log", source_trace_path=trace)


def test_rewind_supersedes_unconsumed_answer_and_prevents_later_delivery(
    tmp_path: Path,
) -> None:
    request = create_clarification_request(
        tmp_path,
        request_id="rewind-request",
        session_id="rewind-session",
        session_number=1,
        turn_number=4,
        tool_call_id="ask-rewind",
        question=QUESTION,
    )
    record_clarification_answer(
        tmp_path,
        session_id="rewind-session",
        request_id=request["request_id"],
        answer=ANSWER,
    )

    superseded = supersede_clarification_for_rewind(
        tmp_path,
        rewind_id="rewind-1",
        to_turn=2,
    )

    assert superseded is not None
    assert clarification_state(tmp_path).phase == "rewound"
    assert load_clarification_answer(tmp_path)["answer"] == ANSWER
    with pytest.raises(ClarificationStateError, match="rewound"):
        consume_clarification_answer(
            tmp_path,
            request_id=request["request_id"],
            session_number=2,
            turn_number=0,
            delivery="resume",
        )


def test_assistant_transcript_segments_rotate_in_resume_order(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.log"
    transcript.write_text("first segment")

    first = _rotate_assistant_transcript(transcript)
    assert first == tmp_path / "transcript.pre_seg_1.log"
    assert first.read_text() == "first segment"
    assert not transcript.exists()

    transcript.write_text("second segment")
    second = _rotate_assistant_transcript(transcript)
    assert second == tmp_path / "transcript.pre_seg_2.log"
    assert second.read_text() == "second segment"
    assert first.read_text() == "first segment"


def test_replay_source_resolves_an_assistant_session_directory(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.log"
    trace = tmp_path / ".trace.jsonl"
    transcript.write_text("saved model messages")
    trace.write_text("{\"event\":\"session_start\"}\n")

    assert resolve_replay_source(tmp_path) == (transcript, trace, "")
