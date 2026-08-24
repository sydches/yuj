"""Acceptance contracts for conversation/tree rewind (issue #18)."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path

import pytest

from _config_helpers import make_config
from scripts.llm_assist.__main__ import main as assist_main
from scripts.llm_assist.runner import derive_live_state
from scripts.llm_assist.store import SessionStore
from llm_solver.config import load_config
from llm_solver.harness.context import FullTranscript
from llm_solver.harness.context_strategies.halflife_context import HalfLifeContext
from llm_solver.harness.guardrails import Action, rewind_on
from llm_solver.harness.loop import Session
from llm_solver.harness.state_writer import project
from llm_solver.harness.turn_snapshots import (
    apply_pending_rewind_resume,
    capture_conversation_snapshot,
    load_conversation_snapshot,
    load_pending_rewind,
    process_rewind_turn_boundary,
    rewind_snapshot_dir,
)
from llm_solver.harness.workspace_checkpoints import WorkspaceCheckpointStore
from llm_solver.server.replay_client import ReplayClient


class _Client:
    def build_assistant_message(self, content, tool_calls):
        return {"role": "assistant", "content": content}

    def query_server_context(self):
        return 0


def _assistant_tool_message(turn: int) -> dict:
    return {
        "role": "assistant",
        "content": f"edit at turn {turn}",
        "tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {
                "name": "write",
                "arguments": json.dumps({"path": "value.txt"}),
            },
        }],
    }


def _record_turn(
    session: Session,
    store: WorkspaceCheckpointStore,
    workspace: Path,
    turn: int,
    value: str,
    *,
    create_later_file: bool = False,
) -> None:
    session._current_turn = turn
    message = _assistant_tool_message(turn)
    session.context.add_assistant(message)
    (workspace / "value.txt").write_text(value)
    if create_later_file:
        (workspace / "later.txt").write_text("created later\n")
    session._emit(
        "tool_call",
        session_number=session._session_number,
        turn_number=turn,
        tool_name="write",
        args_summary="path='value.txt'",
        result_summary=f"wrote {value.strip()}",
        reasoning=f"edit at turn {turn}",
    )
    checkpoint = store.capture(turn)
    session._emit(
        "checkpoint",
        session_number=session._session_number,
        **checkpoint.trace_fields(),
    )
    session.context.add_tool_result(
        f"call-{turn}", f"wrote {value.strip()}", tool_name="write"
    )
    capture_conversation_snapshot(session, turn)


def _session(
    workspace: Path,
    artifact_dir: Path,
    *,
    client=None,
    trace_file=None,
    trace_path: Path | None = None,
    rewind_max: int = 2,
    session_number: int = 1,
    context_manager=None,
    config_overrides: dict | None = None,
):
    config_values = dict(
        rewind_enabled=True,
        rewind_max_per_session=rewind_max,
        tools_file_checkpoints_enabled=True,
        state_writer_enabled=True,
    )
    config_values.update(config_overrides or {})
    cfg = make_config(**config_values)
    store = WorkspaceCheckpointStore(
        workspace,
        shadow_dir=artifact_dir / ".shadow_git",
        excludes=cfg.tools_file_checkpoints_exclude,
    )
    session = Session(
        cfg,
        client or _Client(),
        "system",
        "task",
        str(workspace),
        context_manager=(
            context_manager or FullTranscript(original_prompt="task")
        ),
        trace_file=trace_file or io.StringIO(),
        trace_path=trace_path,
        state_path=artifact_dir / ".solver" / "state.json",
        session_number=session_number,
        checkpoint_store=store,
        artifact_dir=artifact_dir,
    )
    return session, store


def test_rewind_restores_exact_model_messages_and_tree(tmp_path):
    workspace = tmp_path / "task"
    artifact_dir = tmp_path / "artifacts"
    workspace.mkdir()
    artifact_dir.mkdir()
    (workspace / "value.txt").write_text("base\n")
    session, store = _session(workspace, artifact_dir)

    _record_turn(session, store, workspace, 0, "turn zero\n")
    expected_messages = copy.deepcopy(session.context.get_messages())
    raw_before_later_turns = len(session._trace_events)
    _record_turn(
        session,
        store,
        workspace,
        1,
        "turn one\n",
        create_later_file=True,
    )
    _record_turn(session, store, workspace, 2, "turn two\n")

    raw_before_rewind = len(session._trace_events)
    event = session.rewind_to(0, reason="rewind_on_destructive_mutation")

    assert session.context.get_messages() == expected_messages
    assert (workspace / "value.txt").read_text() == "turn zero\n"
    assert not (workspace / "later.txt").exists()
    assert event["from_turn"] == 2
    assert event["to_turn"] == 0
    assert event["commit"] == store.checkpoint_for_turn(0)
    assert event["reason"] == "rewind_on_destructive_mutation"
    assert len(session._trace_events) == raw_before_rewind + 1
    assert len(session._trace_events) > raw_before_later_turns
    assert session._trace_events[-1]["event"] == "rewind"

    state = json.loads((artifact_dir / ".solver" / "state.json").read_text())
    assert [item["turn"] for item in state["trace"]] == [0]
    assert state["state"]["last_rewind"]["to_turn"] == 0
    assert state["meta"]["event_count"] == raw_before_rewind + 1
    assert state["meta"]["active_event_count"] < state["meta"]["event_count"]


def test_rewind_restores_exact_projected_messages(tmp_path):
    workspace = tmp_path / "task"
    artifact_dir = tmp_path / "artifacts"
    workspace.mkdir()
    artifact_dir.mkdir()
    projection_values = {
        "halflife_context_limit_tokens": 1,
        "halflife_no_decay_ratio": 0.0,
        "halflife_verbatim_tool_results": 0,
        "halflife_cap_7_chars": 8,
        "halflife_cap_15_chars": 8,
        "halflife_cap_31_chars": 8,
        "halflife_cap_63_chars": 8,
        "halflife_cap_older_chars": 8,
    }
    context = HalfLifeContext(
        original_prompt="task",
        context_limit_tokens=1,
        activation_ratio=0.0,
        verbatim_tool_results=0,
        cap_7_chars=8,
        cap_15_chars=8,
        cap_31_chars=8,
        cap_63_chars=8,
        cap_older_chars=8,
    )
    session, store = _session(
        workspace,
        artifact_dir,
        context_manager=context,
        config_overrides=projection_values,
    )

    _record_turn(session, store, workspace, 0, "turn zero\n")
    expected_messages = copy.deepcopy(session.context.get_messages())
    assert expected_messages != session.context.get_history_messages()
    _record_turn(session, store, workspace, 1, "turn one\n")

    session.rewind_to(0, reason="projection_test")

    assert session.context.get_messages() == expected_messages
    assert (workspace / "value.txt").read_text() == "turn zero\n"


def test_projection_keeps_raw_count_but_removes_rewound_events():
    events = [
        {"event": "tool_call", "session_number": 1, "turn_number": 0,
         "tool_name": "read", "result_summary": "zero"},
        {"event": "tool_call", "session_number": 1, "turn_number": 1,
         "tool_name": "write", "result_summary": "one"},
        {"event": "session_end", "session_number": 1,
         "finish_reason": "stop", "turns": 2},
        {"event": "rewind", "session_number": 1, "turn_number": 1,
         "from_turn": 1, "to_turn": 0, "reason": "operator",
         "commit": "abc", "rewind_count": 1, "rewind_id": "r1",
         "delivery": "next_session"},
    ]
    state = project(events, max_result_chars=100)
    assert state["meta"]["event_count"] == 4
    assert state["meta"]["active_event_count"] == 2
    assert [row["turn"] for row in state["trace"]] == [0]
    assert state["state"]["last_verify"] == ""
    assert state["state"]["last_rewind"]["commit"] == "abc"


def test_projection_can_restore_a_turn_from_a_discarded_branch():
    events = [
        {"event": "tool_call", "session_number": 1, "turn_number": 0,
         "tool_name": "read", "result_summary": "zero"},
        {"event": "tool_call", "session_number": 1, "turn_number": 1,
         "tool_name": "write", "result_summary": "original one"},
        {"event": "rewind", "session_number": 1, "turn_number": 1,
         "from_turn": 1, "to_turn": 0, "reason": "first",
         "commit": "zero", "rewind_count": 1, "rewind_id": "r1",
         "delivery": "in_session"},
        {"event": "tool_call", "session_number": 1, "turn_number": 2,
         "tool_name": "write", "result_summary": "replacement two"},
        {"event": "rewind", "session_number": 1, "turn_number": 2,
         "from_turn": 2, "to_turn": 1, "reason": "second",
         "commit": "one", "rewind_count": 2, "rewind_id": "r2",
         "delivery": "in_session"},
    ]

    state = project(events, max_result_chars=100)

    assert [row["turn"] for row in state["trace"]] == [0, 1]
    assert state["trace"][-1]["result"] == "original one"
    assert state["state"]["last_rewind"]["rewind_id"] == "r2"
    assert state["meta"]["event_count"] == 5
    assert state["meta"]["active_event_count"] == 3


def test_replay_reproduces_recorded_rewind(tmp_path):
    transcript = tmp_path / "source.log"
    transcript.write_text(
        "=== turn 001 input ===\n"
        + json.dumps({"messages": []})
        + "\n=== turn 001 output ===\n"
        + json.dumps({
            "choices": [{
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })
        + "\n"
    )
    source_trace = tmp_path / "source.trace.jsonl"
    source_trace.write_text(
        json.dumps({
            "event": "rewind",
            "session_number": 1,
            "turn_number": 2,
            "from_turn": 2,
            "to_turn": 0,
            "report_chars": 8,
            "goal": "Explore safely.",
            "report": "Retained",
        })
        + "\n"
        + json.dumps({
            "event": "rewind",
            "session_number": 1,
            "turn_number": 2,
            "from_turn": 2,
            "to_turn": 0,
            "reason": "rewind_on_destructive_mutation",
            "commit": "source-commit-is-runtime-specific",
            "rewind_count": 1,
            "rewind_id": "source-rewind",
            "delivery": "in_session",
        })
        + "\n"
    )
    client = ReplayClient(transcript, source_trace_path=source_trace)

    workspace = tmp_path / "replay-task"
    artifact_dir = tmp_path / "replay-artifacts"
    workspace.mkdir()
    artifact_dir.mkdir()
    (workspace / "value.txt").write_text("base\n")
    session, store = _session(
        workspace, artifact_dir, client=client, rewind_max=1
    )
    _record_turn(session, store, workspace, 0, "turn zero\n")
    expected_messages = copy.deepcopy(session.context.get_messages())
    _record_turn(session, store, workspace, 1, "turn one\n")
    session._current_turn = 2
    session.context.add_assistant({"role": "assistant", "content": "later"})
    (workspace / "value.txt").write_text("turn two\n")

    process_rewind_turn_boundary(session, 2)

    assert session.context.get_messages() == expected_messages
    assert (workspace / "value.txt").read_text() == "turn zero\n"
    assert session._trace_events[-1]["event"] == "rewind"
    assert session._trace_events[-1]["reason"] == source_trace_reason(source_trace)
    assert client.rewinds_at(1, 2) == []


def source_trace_reason(path: Path) -> str:
    return json.loads(path.read_text().splitlines()[-1])["reason"]


def test_model_tool_rewind_does_not_spend_workspace_rewind_limit(tmp_path):
    workspace = tmp_path / "task"
    artifact_dir = tmp_path / "artifacts"
    workspace.mkdir()
    artifact_dir.mkdir()
    (workspace / "value.txt").write_text("base\n")
    trace_path = artifact_dir / ".trace.jsonl"
    trace_path.write_text(json.dumps({
        "event": "rewind",
        "session_number": 1,
        "turn_number": 1,
        "from_turn": 1,
        "to_turn": 0,
        "report_chars": 8,
        "goal": "Explore safely.",
        "report": "Retained",
    }) + "\n")

    with trace_path.open("a") as trace_file:
        session, store = _session(
            workspace,
            artifact_dir,
            trace_file=trace_file,
            trace_path=trace_path,
            rewind_max=1,
        )
        assert session._rewind_count == 0
        _record_turn(session, store, workspace, 0, "turn zero\n")
        _record_turn(session, store, workspace, 1, "turn one\n")

        event = session.rewind_to(0, reason="operator")

    assert event["rewind_count"] == 1
    assert (workspace / "value.txt").read_text() == "turn zero\n"


def test_assistant_rewind_command_restores_tree_and_stages_exact_resume(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "assistant-task"
    workspace.mkdir()
    (workspace / "value.txt").write_text("base\n")
    overlay = tmp_path / "rewind.toml"
    overlay.write_text(
        "[loop]\n"
        "rewind_enabled = true\n"
        "rewind_max_per_session = 1\n\n"
        "[tools]\n"
        "file_checkpoints_enabled = true\n"
    )
    assist_home = tmp_path / "assist-home"
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(assist_home))
    store = SessionStore(assist_home)
    record = store.create_session(
        cwd=workspace,
        model="test-model",
        prompt_text="task",
        prompt_source="test",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[overlay],
    )
    artifact_dir = record.artifact_path
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifact_dir / ".trace.jsonl"
    with open(trace_path, "a") as trace_file:
        session, checkpoint_store = _session(
            workspace,
            artifact_dir,
            trace_file=trace_file,
            trace_path=trace_path,
        )
        _record_turn(session, checkpoint_store, workspace, 0, "turn zero\n")
        session._stale_guard.observe_read("value.txt")
        _record_turn(
            session,
            checkpoint_store,
            workspace,
            1,
            "turn one\n",
            create_later_file=True,
        )
        session._stale_guard.observe_mutation("value.txt", source="write")
        session._emit(
            "rewind",
            session_number=1,
            turn_number=1,
            from_turn=1,
            to_turn=0,
            report_chars=8,
            goal="Explore safely.",
            report="Retained",
        )
        session._emit(
            "session_end",
            session_number=1,
            finish_reason="stop",
            turns=2,
        )
    raw_lines_before = trace_path.read_text().splitlines()
    snapshot_root = rewind_snapshot_dir(workspace, artifact_dir)
    expected_messages = load_conversation_snapshot(
        snapshot_root, 1, 0
    ).model_messages

    assert assist_main([
        "rewind",
        record.session_id,
        "0",
        "--reason",
        "operator_test",
    ]) == 0

    raw_lines_after = trace_path.read_text().splitlines()
    assert len(raw_lines_after) == len(raw_lines_before) + 1
    event = json.loads(raw_lines_after[-1])
    assert event["event"] == "rewind"
    assert event["delivery"] == "next_session"
    assert event["rewind_count"] == 1
    assert (workspace / "value.txt").read_text() == "turn zero\n"
    assert not (workspace / "later.txt").exists()
    pending = load_pending_rewind(snapshot_root)
    assert pending is not None
    assert pending["to_turn"] == 0
    assert pending["commit"] == event["commit"]
    assert derive_live_state(artifact_dir).finish_reason == "rewind"
    assert store.get_session(record.session_id).status == "paused"

    (workspace / "value.txt").write_text("operator edit after rewind\n")
    with open(trace_path, "a") as trace_file:
        resumed, _checkpoint_store = _session(
            workspace,
            artifact_dir,
            trace_file=trace_file,
            trace_path=trace_path,
            session_number=2,
        )
        restored = apply_pending_rewind_resume(resumed)
        assert json.loads(trace_path.read_text().splitlines()[-1])["event"] == "rewind_resume"
        stale_decision = resumed._stale_guard.check_edit("value.txt")
    assert restored is not None
    assert resumed.context.get_messages() == expected_messages
    assert (workspace / "value.txt").read_text() == "turn zero\n"
    assert load_pending_rewind(snapshot_root)["applied_session_number"] == 2
    assert stale_decision.reason == "fresh"


def test_config_and_guardrail_rewind_surface(tmp_path):
    defaults = load_config()
    assert defaults.rewind_enabled is False
    assert defaults.rewind_max_per_session == 1

    valid = tmp_path / "valid.toml"
    valid.write_text(
        "[loop]\nrewind_enabled = true\nrewind_max_per_session = 3\n"
        "[tools]\nfile_checkpoints_enabled = true\n"
    )
    cfg = load_config(user_config=valid)
    assert cfg.rewind_enabled is True
    assert cfg.rewind_max_per_session == 3

    missing_checkpoint = tmp_path / "missing-checkpoint.toml"
    missing_checkpoint.write_text("[loop]\nrewind_enabled = true\n")
    with pytest.raises(ValueError, match="requires tools.file_checkpoints_enabled"):
        load_config(user_config=missing_checkpoint)

    invalid_limit = tmp_path / "invalid-limit.toml"
    invalid_limit.write_text(
        "[loop]\nrewind_max_per_session = 0\n"
    )
    with pytest.raises(ValueError, match="integer >= 1"):
        load_config(user_config=invalid_limit)

    decision = rewind_on("destructive mutation", target_turn=4)
    assert decision.action is Action.REWIND
    assert decision.reason == "rewind_on_destructive_mutation"
    assert decision.target_turn == 4
