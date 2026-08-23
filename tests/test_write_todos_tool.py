"""Acceptance coverage for the optional session-scoped todo tool."""
from __future__ import annotations

from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from _config_helpers import make_config
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._loop.profile_resolution import (
    apply_profile_to_schemas,
)
from scripts.llm_solver.harness.context_strategies import CompoundContext
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness.schemas import get_tool_schemas
from scripts.llm_solver.harness.state_writer import (
    project,
    write_state_from_trace,
)
from scripts.llm_solver.harness.tool_validation import ToolSchemaSet
from scripts.llm_solver.harness.tools import dispatch
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


def _tool_names(cfg) -> tuple[str, ...]:
    schemas = apply_profile_to_schemas(get_tool_schemas(), cfg, client=None)
    return tuple(schema["function"]["name"] for schema in schemas)


def _turn(*, tool_calls=(), content=None, reason="tool_calls") -> TurnResult:
    return TurnResult(
        content=content,
        tool_calls=list(tool_calls),
        finish_reason=reason,
        usage=Usage(prompt_tokens=10, completion_tokens=2),
    )


def _todo(description: str, status: str) -> dict[str, str]:
    return {"description": description, "status": status}


def test_config_defaults_overlay_validation_and_profile_surface(
    tmp_path: Path,
) -> None:
    defaults = load_config()
    assert defaults.tools_todos_enabled is False
    assert defaults.tools_todos_max_items == 20
    assert defaults.state_todos_char_budget == 2000
    assert "write_todos" not in _tool_names(defaults)

    overlay = tmp_path / "todos.toml"
    overlay.write_text(
        "[tools]\n"
        "todos_enabled = true\n"
        "todos_max_items = 7\n"
        "[state]\n"
        "todos_char_budget = 321\n"
    )
    configured = load_config(user_config=overlay)
    assert configured.tools_todos_enabled is True
    assert configured.tools_todos_max_items == 7
    assert configured.state_todos_char_budget == 321
    assert "write_todos" in _tool_names(configured)

    bad_values = (
        ("[tools]\ntodos_enabled = 1\n", "tools.todos_enabled"),
        ("[tools]\ntodos_max_items = 0\n", "tools.todos_max_items"),
        ("[state]\ntodos_char_budget = 0\n", "state.todos_char_budget"),
    )
    for text, error_path in bad_values:
        overlay.write_text(text)
        with pytest.raises(ValueError, match=error_path):
            load_config(user_config=overlay)


def test_schema_validation_rejects_two_in_progress_items() -> None:
    schemas = ToolSchemaSet.from_openai_tools(get_tool_schemas())
    two_active = {
        "todos": [
            _todo("First", "in_progress"),
            _todo("Second", "in_progress"),
        ]
    }
    two_validation = schemas.validate("write_todos", two_active)
    assert two_validation.valid is False
    assert [(error.path, error.keyword) for error in two_validation.errors] == [
        ("$.todos", "maxContains")
    ]


def test_schema_validation_rejects_unknown_status() -> None:
    schemas = ToolSchemaSet.from_openai_tools(get_tool_schemas())
    unknown_status = {"todos": [_todo("First", "paused")]}

    status_validation = schemas.validate("write_todos", unknown_status)
    assert status_validation.valid is False
    assert any(
        error.path == "$.todos[0].status" and error.keyword == "enum"
        for error in status_validation.errors
    )


def test_handler_enforces_cross_item_and_max_when_schema_reject_is_off(
    tmp_path: Path,
) -> None:
    two_active = {
        "todos": [
            _todo("First", "in_progress"),
            _todo("Second", "in_progress"),
        ]
    }
    cfg = make_config(
        tools_todos_enabled=True,
        tools_todos_max_items=3,
        tools_schema_validation="off",
    )
    metadata: dict[str, object] = {}
    result = dispatch(
        "write_todos",
        two_active,
        cwd=str(tmp_path),
        cfg=cfg,
        execution_metadata=metadata,
    )
    assert "ERROR: write_todos validation failed:" in result
    assert "at most one in_progress item" in result
    assert "todos" not in metadata

    cfg = make_config(
        tools_todos_enabled=True,
        tools_todos_max_items=1,
        tools_schema_validation="off",
    )
    result = dispatch(
        "write_todos",
        {"todos": [_todo("First", "pending"), _todo("Second", "blocked")]},
        cwd=str(tmp_path),
        cfg=cfg,
        execution_metadata=metadata,
    )
    assert "tools.todos_max_items is 1" in result
    assert "todos" not in metadata
    assert not (tmp_path / ".solver" / "state.json").exists()


def test_successful_handler_returns_canonical_replacement_without_state_write(
    tmp_path: Path,
) -> None:
    cfg = make_config(tools_todos_enabled=True, tools_todos_max_items=3)
    submitted = [
        _todo("Implement runtime", "completed"),
        _todo("Run focused tests", "in_progress"),
    ]
    metadata: dict[str, object] = {}

    result = dispatch(
        "write_todos",
        {"todos": submitted},
        cwd=str(tmp_path),
        cfg=cfg,
        execution_metadata=metadata,
    )

    assert "OK: replaced todo list with 2 items" in result
    assert metadata["todos"] == submitted
    assert metadata["todos"] is not submitted
    assert metadata["executed"] is True
    assert not (tmp_path / ".solver" / "state.json").exists()

    cleared: dict[str, object] = {}
    clear_result = dispatch(
        "write_todos",
        {"todos": []},
        cwd=str(tmp_path),
        cfg=cfg,
        execution_metadata=cleared,
    )
    assert "OK: replaced todo list with 0 items" in clear_result
    assert cleared["todos"] == []


def test_adaptive_refresh_updates_tool_surface_and_suffix_budget(
    tmp_path: Path,
) -> None:
    from scripts.llm_solver.harness.adaptive_control import executors

    cfg = make_config(tools_todos_enabled=False, state_todos_char_budget=2000)
    session = Session(cfg, MagicMock(), "system", "task", str(tmp_path))
    assert "write_todos" not in session.active_tool_names

    updated = replace(
        cfg,
        tools_todos_enabled=True,
        tools_todos_max_items=9,
        state_todos_char_budget=77,
    )
    changed = {
        "tools_todos_enabled",
        "tools_todos_max_items",
        "state_todos_char_budget",
    }
    ok, reason, refreshed, blocked = executors._refresh_runtime_surfaces(
        session,
        cfg,
        updated,
        changed,
    )

    assert ok is True
    assert reason == ""
    assert blocked == ()
    assert "tool_schemas" in refreshed
    assert "context._todos_char_budget" in refreshed
    assert "write_todos" in {
        schema["function"]["name"] for schema in session._tool_schemas
    }
    assert session.context._todos_char_budget == 77


def test_trace_replay_projects_only_the_latest_todos_event(
    tmp_path: Path,
) -> None:
    first = [_todo("Old plan", "pending")]
    latest = [
        _todo("Finished work", "completed"),
        _todo("Verify artifacts", "in_progress"),
    ]
    events = [
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 0,
            "tool_name": "write_todos",
            "args_summary": "todos=[untrusted summary]",
            "result_summary": "OK",
        },
        {
            "event": "todos",
            "session_number": 1,
            "turn_number": 0,
            "todos": first,
        },
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 1,
            "tool_name": "read",
            "args_summary": "todos=[must not project]",
            "result_summary": "todos=[must not project]",
        },
        {
            "event": "todos",
            "session_number": 1,
            "turn_number": 2,
            "todos": latest,
        },
    ]
    projected = project(events, max_result_chars=20_000)
    assert projected["todos"] == latest
    assert projected["todos"] is not latest

    trace_path = tmp_path / ".trace.jsonl"
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    state_path = tmp_path / ".solver" / "state.json"
    state_path.parent.mkdir()
    state_path.write_text(json.dumps({"todos": [_todo("rogue", "blocked")]}))

    write_state_from_trace(trace_path, state_path, max_result_chars=20_000)
    replayed = json.loads(state_path.read_text())
    assert replayed["todos"] == latest

    events.append(
        {
            "event": "todos",
            "session_number": 1,
            "turn_number": 3,
            "todos": [],
        }
    )
    assert project(events, max_result_chars=20_000)["todos"] == []


def test_session_emits_todos_after_tool_call_and_refreshes_state(
    tmp_path: Path,
) -> None:
    submitted = [
        _todo("Implement", "completed"),
        _todo("Verify", "in_progress"),
    ]
    client = MagicMock()
    client.chat.side_effect = [
        _turn(
            tool_calls=[
                ToolCall(
                    id="todo-1",
                    name="write_todos",
                    arguments={"todos": submitted},
                )
            ]
        ),
        _turn(content="done", reason="stop"),
    ]
    client.build_assistant_message.side_effect = [
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "assistant", "content": "done"},
    ]
    cfg = make_config(
        max_turns=2,
        tools_todos_enabled=True,
        tools_schema_validation="reject",
        error_nudge_threshold=99,
        rumination_nudge_threshold=999,
    )
    trace = StringIO()
    state_path = tmp_path / ".solver" / "state.json"
    session = Session(
        cfg,
        client,
        "system",
        "task",
        str(tmp_path),
        trace_file=trace,
        state_path=state_path,
        session_number=3,
    )

    with patch.object(Session, "_get_server_ctx", return_value=cfg.context_size):
        result = session.run()

    assert result.finish_reason == "stop"
    events = [json.loads(line) for line in trace.getvalue().splitlines()]
    relevant = [
        event for event in events
        if event["event"] in {"tool_call", "todos"}
    ]
    assert [event["event"] for event in relevant] == ["tool_call", "todos"]
    assert relevant[0]["tool_name"] == "write_todos"
    assert relevant[1]["tool_call_id"] == "todo-1"
    assert relevant[1]["todos"] == submitted
    assert json.loads(state_path.read_text())["todos"] == submitted


def test_state_suffix_renders_latest_list_within_exact_char_budget(
    tmp_path: Path,
) -> None:
    budget = 90
    suffix = "Continue from the projected state."
    state_path = tmp_path / ".solver" / "state.json"
    state_path.parent.mkdir()
    state_path.write_text(
        json.dumps(
            {
                "state": {
                    "current_attempt": "",
                    "last_verify": "",
                    "next_action": "",
                },
                "todos": [
                    _todo(
                        "First line\nwith model-authored newlines " + "x" * 100,
                        "in_progress",
                    ),
                    _todo("Second item " + "y" * 100, "pending"),
                ],
                "trace": [],
                "gates": [],
                "evidence": [],
                "inference": [],
            }
        )
    )
    context = CompoundContext(
        cwd=str(tmp_path),
        original_prompt="Task",
        trace_lines=20,
        evidence_lines=20,
        inference_lines=20,
        recent_tool_results_chars=2000,
        trace_stub_chars=100,
        min_turns=0,
        suffix=suffix,
        todos_char_budget=budget,
    )
    context.add_system("System")
    context.add_user("Task")

    files = context._get_solver_files(tmp_path / ".solver")
    assert 0 < len(files["todos"]) <= budget
    assert "\nwith model-authored newlines" not in files["todos"]
    assert files["todos"].endswith("... [todo list truncated]")
    rendered = context.get_messages()[-1]["content"]
    assert files["todos"] + "\n\n" + suffix in rendered

    state = json.loads(state_path.read_text())
    state["todos"] = [_todo("New turn replacement", "blocked")]
    state_path.write_text(json.dumps(state))
    context.add_tool_result("refresh", "OK")
    next_rendered = context.get_messages()[-1]["content"]
    assert "New turn replacement" in next_rendered
    assert "First line" not in next_rendered
