"""Paused-session correction state-machine acceptance tests."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from _config_helpers import make_config
from scripts.llm_assist.__main__ import main as assist_main
from scripts.llm_assist.runner import (
    mark_session_interrupted,
    save_approval_request,
)
from scripts.llm_assist.store import (
    SessionLock,
    SessionLockedError,
    SessionStore,
)
from scripts.llm_solver.harness.clarifications import (
    consume_clarification_answer,
    create_clarification_request,
    record_clarification_answer,
)
from scripts.llm_solver.harness.corrections import (
    CorrectionStateError,
    consume_correction,
    correction_consumption_path,
    correction_path,
    correction_state,
    create_correction,
    load_correction,
    load_correction_consumption,
)
from scripts.llm_solver.harness._loop.profile_resolution import build_tool_surface
from scripts.llm_solver.harness._loop.run_step import _inject_pending_correction
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.server.client import LlamaClient
from scripts.llm_solver.server.replay_client import ReplayClient, ReplayDivergence
from scripts.llm_solver.server.types import ImageInput, ToolCall, TurnResult, Usage


CORRECTION = "Use PostgreSQL, not SQLite. Keep the existing schema exactly."


class _SequenceClient:
    def __init__(self, *turns: TurnResult, trace_path: Path | None = None):
        self.turns = list(turns)
        self.trace_path = trace_path
        self.requests: list[dict] = []
        self.events_at_request: list[list[dict]] = []

    def chat(self, messages, tools, turn=0):
        self.requests.append({
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
            "turn": turn,
        })
        if self.trace_path is not None:
            self.events_at_request.append(_trace_events(self.trace_path))
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


def _turn(
    *calls: ToolCall,
    content: str = "done",
) -> TurnResult:
    return TurnResult(
        content=content,
        tool_calls=list(calls),
        finish_reason="tool_calls" if calls else "stop",
        usage=Usage(prompt_tokens=10, completion_tokens=3),
    )


def _trace_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _record(
    store: SessionStore,
    cwd: Path,
    *,
    status: str = "paused",
    finish_reason: str = "max_turns",
):
    record = store.create_session(
        cwd=cwd,
        model="test-model",
        prompt_text="Complete the task.",
        prompt_source="test",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    record.artifact_path.mkdir(parents=True, exist_ok=True)
    trace = record.artifact_path / ".trace.jsonl"
    events = [{"event": "session_start", "session_number": 1}]
    if status != "running":
        events.append({
            "event": "session_end",
            "session_number": 1,
            "finish_reason": finish_reason,
            "turns": 1,
        })
    trace.write_text("".join(json.dumps(event) + "\n" for event in events))
    store.update_session(
        record.session_id,
        status=status,
        last_finish_reason=finish_reason if status != "running" else None,
    )
    refreshed = store.get_session(record.session_id)
    assert refreshed is not None
    return refreshed


def _correct(store: SessionStore, record, text: str = CORRECTION) -> int:
    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        return assist_main(["correct", record.session_id, text])


def _assistant_cfg(**overrides):
    return make_config(
        runtime_mode="assistant",
        max_turns=1,
        guardrails_arm_after_turn=99,
        rumination_nudge_threshold=999,
        **overrides,
    )


def test_correct_records_exact_pending_input_and_creation_trace(
    tmp_path: Path,
    capsys,
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _record(store, tmp_path / "work")
    exact = "  Keep both spaces.\nDo not normalize this text.  "

    assert _correct(store, record, exact) == 0

    captured = capsys.readouterr().out
    state = correction_state(record.artifact_path)
    assert state.phase == "pending"
    assert state.correction is not None
    assert state.correction["session_id"] == record.session_id
    assert state.correction["after_session_number"] == 1
    assert state.correction["text"] == exact
    assert state.correction["text_sha256"] == hashlib.sha256(
        exact.encode("utf-8")
    ).hexdigest()
    assert f"corrected: {record.session_id}" in captured
    assert "correction: pending" in captured
    events = _trace_events(record.artifact_path / ".trace.jsonl")
    created = [event for event in events if event["event"] == "correction_created"]
    assert len(created) == 1
    assert created[0]["correction_id"] == state.correction["correction_id"]
    assert created[0]["text_sha256"] == state.correction["text_sha256"]
    assert created[0]["text_chars"] == len(exact)
    assert exact not in json.dumps(created[0])


def test_status_and_show_report_bounded_correction_evidence_separately(
    tmp_path: Path,
    capsys,
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _record(store, tmp_path / "work")
    long_text = "boundary correction " + "x" * 500
    _correct(store, record, long_text)
    capsys.readouterr()

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert assist_main(["status", record.session_id]) == 0
        status_output = capsys.readouterr().out
        assert assist_main([
            "show", record.session_id, "--turns", "0", "--trace-lines", "0"
        ]) == 0
        show_output = capsys.readouterr().out

    for output in (status_output, show_output):
        assert "correction: pending" in output
        assert "correction_id:" in output
        assert "correction_sha256:" in output
        assert f"correction_chars: {len(long_text)}" in output
        assert "correction_preview:" in output
        assert long_text not in output
        preview_line = next(
            line for line in output.splitlines()
            if line.startswith("correction_preview: ")
        )
        rendered_preview = preview_line.removeprefix("correction_preview: ")
        assert len(rendered_preview) <= 160
        assert json.loads(rendered_preview).endswith("...")
        assert "approval: none" in output
        assert "clarification: none" in output


def test_second_pending_correction_and_wrong_session_fail_without_mutation(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _record(store, tmp_path / "work")
    _correct(store, record)
    correction_before = correction_path(record.artifact_path).read_bytes()
    trace_before = (record.artifact_path / ".trace.jsonl").read_bytes()

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        with pytest.raises(SystemExit, match="pending correction"):
            assist_main(["correct", record.session_id, "replace it"])
        with pytest.raises(SystemExit, match="unknown session"):
            assist_main(["correct", "wrong-session", "do not save this"])
        with pytest.raises(SystemExit, match="explicit session reference"):
            assist_main(["correct", "latest", "do not guess the session"])

    assert correction_path(record.artifact_path).read_bytes() == correction_before
    assert (record.artifact_path / ".trace.jsonl").read_bytes() == trace_before


@pytest.mark.parametrize(
    ("status", "finish_reason", "message"),
    [
        ("running", "", "active"),
        ("completed", "stop", "completed"),
        ("archived", "max_turns", "archived"),
        ("created", "max_turns", "not resumable"),
        ("purged", "max_turns", "not resumable"),
    ],
)
def test_correct_rejects_active_completed_archived_and_nonresumable_sessions(
    tmp_path: Path,
    status: str,
    finish_reason: str,
    message: str,
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _record(
        store,
        tmp_path / "work",
        status=status,
        finish_reason=finish_reason or "max_turns",
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        with pytest.raises(SystemExit, match=message):
            assist_main(["correct", record.session_id, CORRECTION])

    assert not correction_path(record.artifact_path).exists()


def test_correct_rejects_locked_session_without_evidence_mutation(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _record(store, tmp_path / "work")
    foreign = SessionLock(
        session_id=record.session_id,
        owner_host="other-host",
        owner_pid=4242,
        acquired_at="2026-08-25T00:00:00+00:00",
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), patch.object(
        store,
        "acquire_session_lock",
        side_effect=SessionLockedError(foreign),
    ):
        with pytest.raises(SystemExit, match="already locked by pid 4242"):
            assist_main(["correct", record.session_id, CORRECTION])

    assert not correction_path(record.artifact_path).exists()


def test_exact_correction_is_last_user_input_and_consumed_before_transport_once(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _record(store, tmp_path / "work")
    _correct(store, record)
    trace_path = record.artifact_path / ".trace.jsonl"
    (record.artifact_path / "transcript.pre_seg_1.log").write_text(
        "prior transcript segment"
    )
    cfg = _assistant_cfg()
    first_client = _SequenceClient(_turn(), trace_path=trace_path)

    with trace_path.open("a") as trace, patch.object(
        Session, "_get_server_ctx", return_value=cfg.context_size
    ):
        first = Session(
            cfg,
            first_client,
            "system",
            "mechanical resume",
            str(tmp_path),
            trace_file=trace,
            trace_path=trace_path,
            artifact_dir=record.artifact_path,
            session_number=2,
        )
        first.run()

    sent = first_client.requests[0]["messages"]
    user_messages = [
        message["content"] for message in sent if message.get("role") == "user"
    ]
    assert user_messages[-1] == CORRECTION
    assert user_messages.count(CORRECTION) == 1
    events_at_transport = first_client.events_at_request[0]
    consumed_events = [
        event for event in events_at_transport
        if event["event"] == "correction_consumed"
    ]
    assert len(consumed_events) == 1
    consumption = load_correction_consumption(record.artifact_path)
    assert consumption is not None
    assert consumption["session_number"] == 2
    assert consumption["turn_number"] == 0
    assert consumption["transcript_segment"] == 2
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
            artifact_dir=record.artifact_path,
            session_number=3,
        )
        second.run()

    assert CORRECTION not in {
        message.get("content") for message in second_client.requests[0]["messages"]
    }
    assert len([
        event for event in _trace_events(trace_path)
        if event["event"] == "correction_consumed"
    ]) == 1


def test_resume_time_context_restore_cannot_erase_pending_correction(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _record(store, tmp_path / "work")
    _correct(store, record)
    trace_path = record.artifact_path / ".trace.jsonl"
    cfg = _assistant_cfg()
    client = _SequenceClient(_turn())

    with trace_path.open("a") as trace, patch.object(
        Session, "_get_server_ctx", return_value=cfg.context_size
    ):
        session = Session(
            cfg,
            client,
            "system",
            "mechanical resume",
            str(tmp_path),
            trace_file=trace,
            trace_path=trace_path,
            artifact_dir=record.artifact_path,
            session_number=2,
        )
        assert CORRECTION not in {
            message.get("content") for message in session.context.get_messages()
        }
        assert session.context.replace_all_messages([
            {"role": "system", "content": "restored system"},
            {"role": "user", "content": "restored rewind boundary"},
        ])
        session.run()

    user_messages = [
        message["content"]
        for message in client.requests[0]["messages"]
        if message.get("role") == "user"
    ]
    assert user_messages == ["restored rewind boundary", CORRECTION]


def test_ambiguous_request_failure_consumes_without_later_redelivery(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _record(store, tmp_path / "work")
    _correct(store, record)
    trace_path = record.artifact_path / ".trace.jsonl"
    cfg = _assistant_cfg()
    failed_client = _SequenceClient()

    with trace_path.open("a") as trace, patch.object(
        Session, "_get_server_ctx", return_value=cfg.context_size
    ):
        failed = Session(
            cfg,
            failed_client,
            "system",
            "resume",
            str(tmp_path),
            trace_file=trace,
            trace_path=trace_path,
            artifact_dir=record.artifact_path,
            session_number=2,
        )
        failed._chat_with_retry = lambda _turn_number: None
        result = failed.run()

    assert result.finish_reason == "error"
    assert correction_state(record.artifact_path).phase == "consumed"
    later_client = _SequenceClient(_turn())
    with trace_path.open("a") as trace, patch.object(
        Session, "_get_server_ctx", return_value=cfg.context_size
    ):
        later = Session(
            cfg,
            later_client,
            "system",
            "resume after unknown transport result",
            str(tmp_path),
            trace_file=trace,
            trace_path=trace_path,
            artifact_dir=record.artifact_path,
            session_number=3,
        )
        later.run()
    assert CORRECTION not in {
        message.get("content") for message in later_client.requests[0]["messages"]
    }


def test_interrupt_marker_preserves_pending_and_consumed_correction_is_not_recovered(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _record(store, tmp_path / "work")
    _correct(store, record)
    before = correction_path(record.artifact_path).read_bytes()

    mark_session_interrupted(record.artifact_path)
    assert correction_state(record.artifact_path).phase == "pending"
    assert correction_path(record.artifact_path).read_bytes() == before

    trace_path = record.artifact_path / ".trace.jsonl"
    cfg = _assistant_cfg()
    client = _SequenceClient(_turn())
    with trace_path.open("a") as trace, patch.object(
        Session, "_get_server_ctx", return_value=cfg.context_size
    ):
        session = Session(
            cfg,
            client,
            "system",
            "resume after interruption",
            str(tmp_path),
            trace_file=trace,
            trace_path=trace_path,
            artifact_dir=record.artifact_path,
            session_number=2,
        )
        session.run()
    assert correction_state(record.artifact_path).phase == "consumed"

    mark_session_interrupted(record.artifact_path)
    recovered_client = _SequenceClient(_turn())
    with trace_path.open("a") as trace, patch.object(
        Session, "_get_server_ctx", return_value=cfg.context_size
    ):
        recovered = Session(
            cfg,
            recovered_client,
            "system",
            "recover again",
            str(tmp_path),
            trace_file=trace,
            trace_path=trace_path,
            artifact_dir=record.artifact_path,
            session_number=3,
        )
        recovered.run()
    assert CORRECTION not in {
        message.get("content")
        for message in recovered_client.requests[0]["messages"]
    }


def test_pending_correction_blocks_rewind_without_mutation(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _record(store, tmp_path / "work")
    _correct(store, record)
    correction_before = correction_path(record.artifact_path).read_bytes()
    trace_before = (record.artifact_path / ".trace.jsonl").read_bytes()

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), patch(
        "scripts.llm_assist.__main__.rewind_session"
    ) as rewind:
        with pytest.raises(SystemExit, match="pending correction"):
            assist_main(["rewind", record.session_id, "0"])

    rewind.assert_not_called()
    assert correction_path(record.artifact_path).read_bytes() == correction_before
    assert (record.artifact_path / ".trace.jsonl").read_bytes() == trace_before


def test_pending_approval_stays_authoritative_and_unchanged(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _record(store, tmp_path / "work")
    save_approval_request(record.artifact_path, {
        "status": "pending",
        "tool_name": "bash",
        "cmd": "rm -rf build",
        "args_summary": "cmd='rm -rf build'",
        "reason": "destructive file deletion via rm",
    })
    approval_path = record.artifact_path / "approval_request.json"
    approval_before = approval_path.read_bytes()

    assert _correct(store, record) == 0

    assert approval_path.read_bytes() == approval_before
    assert correction_state(record.artifact_path).phase == "pending"
    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        with pytest.raises(SystemExit, match="pending approval request"):
            assist_main(["resume", record.session_id])
    assert correction_state(record.artifact_path).phase == "pending"


def test_pending_clarification_stays_authoritative_and_unchanged(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _record(store, tmp_path / "work")
    request = create_clarification_request(
        record.artifact_path,
        request_id="clarification-48",
        session_id=record.session_id,
        session_number=1,
        turn_number=0,
        tool_call_id="ask-48",
        question="Which database?",
    )
    request_path = record.artifact_path / "clarification_request.json"
    request_before = request_path.read_bytes()

    assert _correct(store, record) == 0

    assert request_path.read_bytes() == request_before
    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        with pytest.raises(SystemExit, match="pending clarification"):
            assist_main(["resume", record.session_id])
    assert request["request_id"] == "clarification-48"
    assert correction_state(record.artifact_path).phase == "pending"


def test_correction_does_not_change_attachment_evidence(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _record(store, tmp_path / "work")
    attachment = record.artifact_path / "attachments" / "segment-0001" / "image.png"
    attachment.parent.mkdir(parents=True)
    attachment.write_bytes(b"exact-image-bytes")
    manifest = record.artifact_path / "attachments.json"
    manifest.write_bytes(b'{"schema":"yuj.assistant-attachments","segments":[]}\n')
    before = (manifest.read_bytes(), attachment.read_bytes())

    assert _correct(store, record) == 0

    assert (manifest.read_bytes(), attachment.read_bytes()) == before


def test_correction_does_not_retarget_image_transport(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _record(store, tmp_path / "work")
    _correct(store, record)
    cfg = _assistant_cfg()
    client = LlamaClient(cfg, profile=None)
    client.set_image_inputs([
        ImageInput(media_type="image/png", data=b"exact-image-bytes")
    ])
    trace_path = record.artifact_path / ".trace.jsonl"

    with trace_path.open("a") as trace:
        session = Session(
            cfg,
            client,
            "system",
            "image-bearing resume prompt",
            str(tmp_path),
            trace_file=trace,
            trace_path=trace_path,
            artifact_dir=record.artifact_path,
            session_number=2,
        )
        _inject_pending_correction(session)

    wire_messages = client._messages_with_image_inputs(
        session.context.get_messages()
    )
    user_messages = [
        message for message in wire_messages if message.get("role") == "user"
    ]
    assert user_messages[-1]["content"] == CORRECTION
    assert user_messages[-2]["content"][-1] == {
        "type": "text",
        "text": "image-bearing resume prompt",
    }


def test_measurement_mode_neither_delivers_nor_consumes_pending_correction(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    create_correction(
        artifact_dir,
        correction_id="measurement-excluded",
        session_id="session-48",
        after_session_number=1,
        text=CORRECTION,
    )
    trace_path = artifact_dir / ".trace.jsonl"
    trace_path.write_text(json.dumps({
        "event": "correction_created",
        "session_number": 1,
        "correction_id": "measurement-excluded",
        "text_sha256": hashlib.sha256(CORRECTION.encode()).hexdigest(),
        "text_chars": len(CORRECTION),
    }) + "\n")
    cfg = make_config(
        runtime_mode="measurement",
        max_turns=1,
        guardrails_arm_after_turn=99,
        rumination_nudge_threshold=999,
    )
    client = _SequenceClient(_turn())

    with trace_path.open("a") as trace, patch.object(
        Session, "_get_server_ctx", return_value=cfg.context_size
    ):
        session = Session(
            cfg,
            client,
            "system",
            "measurement task",
            str(tmp_path),
            trace_file=trace,
            trace_path=trace_path,
            artifact_dir=artifact_dir,
            session_number=1,
        )
        session.run()

    assert CORRECTION not in {
        message.get("content") for message in client.requests[0]["messages"]
    }
    assert correction_state(artifact_dir).phase == "pending"
    assert not correction_consumption_path(artifact_dir).exists()


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


def _write_transcript(
    path: Path,
    *,
    messages: list[dict],
    tools: list[dict],
    response: dict,
) -> None:
    path.write_text(
        "=== turn 001 input ===\n"
        + json.dumps({"messages": messages, "tools": tools})
        + "\n=== turn 001 output ===\n"
        + json.dumps(response)
        + "\n"
    )


def test_offline_replay_uses_recorded_correction_once_without_reopening_it(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    correction = create_correction(
        source,
        correction_id="replay-correction",
        session_id="source-session",
        after_session_number=1,
        text=CORRECTION,
    )
    assistant_cfg = make_config(runtime_mode="assistant")
    measurement_cfg = make_config(
        runtime_mode="measurement",
        max_turns=2,
        guardrails_arm_after_turn=99,
        rumination_nudge_threshold=999,
    )
    recorded_tools = build_tool_surface(
        assistant_cfg, _SequenceClient()
    ).active_schemas
    first_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": CORRECTION},
    ]
    resumed_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "mechanical resume"},
        {"role": "user", "content": CORRECTION},
    ]
    _write_transcript(
        source / "transcript.pre_seg_1.log",
        messages=first_messages,
        tools=recorded_tools,
        response=_response(call=ToolCall("think-1", "think", {"thought": "pause"})),
    )
    _write_transcript(
        source / "transcript.log",
        messages=resumed_messages,
        tools=recorded_tools,
        response=_response(content="done"),
    )
    with (source / "transcript.log").open("a") as transcript:
        transcript.write(
            "=== turn 002 input ===\n"
            + json.dumps({"messages": resumed_messages, "tools": recorded_tools})
            + "\n=== turn 002 output ===\n"
            + json.dumps(_response(content="later context"))
            + "\n"
        )
    consumption = consume_correction(
        source,
        correction_id=correction["correction_id"],
        session_number=2,
        turn_number=0,
        delivery="resume",
    )
    assert consumption["transcript_segment"] == 2
    source_trace = source / ".trace.jsonl"
    source_trace.write_text("".join(json.dumps(event) + "\n" for event in [
        {
            "event": "correction_created",
            "session_number": 1,
            "correction_id": correction["correction_id"],
            "text_sha256": correction["text_sha256"],
            "text_chars": len(CORRECTION),
        },
        {
            "event": "correction_consumed",
            "session_number": 2,
            "turn_number": 0,
            "transcript_segment": consumption["transcript_segment"],
            "correction_id": correction["correction_id"],
            "text_sha256": correction["text_sha256"],
            "delivery": consumption["delivery"],
        },
    ]))

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
    assert not correction_path(destination).exists()
    assert not correction_consumption_path(destination).exists()
    replay_events = _trace_events(destination_trace)
    names = [
        event["event"] for event in replay_events
        if event["event"].startswith("correction_")
    ]
    assert names == [
        "correction_created",
        "correction_consumed",
        "correction_replayed",
    ]
    assert all(
        event.get("replayed") is True
        for event in replay_events
        if event["event"].startswith("correction_")
    )
    replay_consumption = next(
        event for event in replay_events
        if event["event"] == "correction_consumed"
    )
    assert replay_consumption["turn_number"] == 1
    assert sum(
        1
        for message in session.context.get_messages()
        if message.get("role") == "user" and message.get("content") == CORRECTION
    ) == 1


def test_replay_does_not_duplicate_a_correction_restored_with_clarification(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    request = create_clarification_request(
        source,
        request_id="combined-request",
        session_id="source-session",
        session_number=1,
        turn_number=0,
        tool_call_id="ask-combined",
        question="Which database?",
    )
    answer = record_clarification_answer(
        source,
        session_id="source-session",
        request_id=request["request_id"],
        answer="PostgreSQL.",
    )
    clarification_consumption = consume_clarification_answer(
        source,
        request_id=request["request_id"],
        session_number=2,
        turn_number=0,
        delivery="resume",
    )
    correction = create_correction(
        source,
        correction_id="combined-correction",
        session_id="source-session",
        after_session_number=1,
        text=CORRECTION,
    )

    assistant_cfg = make_config(runtime_mode="assistant")
    measurement_cfg = make_config(
        runtime_mode="measurement",
        max_turns=3,
        tools_schema_validation="reject",
        guardrails_arm_after_turn=99,
        rumination_nudge_threshold=999,
    )
    recorded_tools = build_tool_surface(
        assistant_cfg, _SequenceClient()
    ).active_schemas
    _write_transcript(
        source / "transcript.pre_seg_1.log",
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ],
        tools=recorded_tools,
        response=_response(call=ToolCall(
            "ask-combined", "ask_user", {"question": "Which database?"}
        )),
    )
    resumed_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "resume"},
        {"role": "user", "content": answer["answer"]},
        {"role": "user", "content": CORRECTION},
    ]
    _write_transcript(
        source / "transcript.log",
        messages=resumed_messages,
        tools=recorded_tools,
        response=_response(content="done"),
    )
    correction_consumption = consume_correction(
        source,
        correction_id=correction["correction_id"],
        session_number=2,
        turn_number=0,
        delivery="resume",
    )
    source_trace = source / ".trace.jsonl"
    source_trace.write_text("".join(
        json.dumps(event) + "\n"
        for event in [
            {
                "event": "clarification_request",
                "session_number": 1,
                "turn_number": 0,
                "request_id": request["request_id"],
                "tool_call_id": request["tool_call_id"],
                "question": request["question"],
            },
            {
                "event": "clarification_answer",
                "session_number": 1,
                "turn_number": 0,
                "request_id": request["request_id"],
                "answer_sha256": answer["answer_sha256"],
                "answer_chars": len(answer["answer"]),
            },
            {
                "event": "correction_created",
                "session_number": 1,
                "correction_id": correction["correction_id"],
                "text_sha256": correction["text_sha256"],
                "text_chars": len(correction["text"]),
            },
            {
                "event": "clarification_consumed",
                "session_number": 2,
                "turn_number": 0,
                "request_id": request["request_id"],
                "answer_sha256": answer["answer_sha256"],
                "delivery": clarification_consumption["delivery"],
            },
            {
                "event": "correction_consumed",
                "session_number": 2,
                "turn_number": 0,
                "transcript_segment": correction_consumption[
                    "transcript_segment"
                ],
                "correction_id": correction["correction_id"],
                "text_sha256": correction["text_sha256"],
                "delivery": correction_consumption["delivery"],
            },
        ]
    ))

    replay = ReplayClient(
        source / "transcript.log", source_trace_path=source_trace
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
    assert sum(
        1
        for message in session.context.get_messages()
        if message.get("role") == "user"
        and message.get("content") == CORRECTION
    ) == 1


def test_replay_refuses_pending_or_contradictory_correction_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    correction = create_correction(
        source,
        correction_id="pending-replay-correction",
        session_id="source-session",
        after_session_number=1,
        text=CORRECTION,
    )
    trace = source / ".trace.jsonl"
    trace.write_text(json.dumps({
        "event": "correction_created",
        "session_number": 1,
        "correction_id": correction["correction_id"],
        "text_sha256": correction["text_sha256"],
        "text_chars": len(CORRECTION),
    }) + "\n")
    _write_transcript(
        source / "transcript.log",
        messages=[{"role": "user", "content": "task"}],
        tools=[],
        response=_response(content="done"),
    )

    with pytest.raises(
        ReplayDivergence,
        match="recorded and consumed correction",
    ):
        ReplayClient(source / "transcript.log", source_trace_path=trace)

    payload = json.loads(correction_path(source).read_text())
    payload["text_sha256"] = "0" * 64
    correction_path(source).write_text(json.dumps(payload))
    with pytest.raises(CorrectionStateError, match="text_sha256"):
        load_correction(source)


def test_status_rejects_file_trace_contradiction(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    record = _record(store, tmp_path / "work")
    _correct(store, record)
    trace_path = record.artifact_path / ".trace.jsonl"
    events = _trace_events(trace_path)
    created = next(
        event for event in events if event["event"] == "correction_created"
    )
    created["text_chars"] += 1
    trace_path.write_text("".join(json.dumps(event) + "\n" for event in events))

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        with pytest.raises(SystemExit, match="creation trace event do not match"):
            assist_main(["status", record.session_id])


def test_consumption_rejects_wrong_identity_and_duplicate_without_mutation(
    tmp_path: Path,
) -> None:
    correction = create_correction(
        tmp_path,
        correction_id="consume-once",
        session_id="session-48",
        after_session_number=4,
        text=CORRECTION,
    )
    with pytest.raises(CorrectionStateError, match="id does not match"):
        consume_correction(
            tmp_path,
            correction_id="wrong-id",
            session_number=5,
            turn_number=0,
            delivery="resume",
        )
    consumption = consume_correction(
        tmp_path,
        correction_id=correction["correction_id"],
        session_number=5,
        turn_number=0,
        delivery="resume",
    )
    before = correction_consumption_path(tmp_path).read_bytes()
    with pytest.raises(CorrectionStateError, match="already consumed"):
        consume_correction(
            tmp_path,
            correction_id=correction["correction_id"],
            session_number=6,
            turn_number=0,
            delivery="resume",
        )
    assert correction_consumption_path(tmp_path).read_bytes() == before
    assert load_correction_consumption(tmp_path) == consumption
