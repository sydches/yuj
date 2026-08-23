"""Acceptance coverage for the optional bounded ``think`` scratchpad."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from _config_helpers import make_config
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._guardrails.checks_post import rumination_ladder
from scripts.llm_solver.harness._guardrails.checks_pre import done_guard
from scripts.llm_solver.harness._guardrails.state import (
    Action,
    init_guardrail_state,
)
from scripts.llm_solver.harness._loop._session_setup import build_context_manager
from scripts.llm_solver.harness._loop.profile_resolution import (
    _filter_disabled_tools,
)
from scripts.llm_solver.harness.context_strategies import (
    list_context_modes,
    resolve_context_class,
)
from scripts.llm_solver.harness.schemas import get_tool_schemas
from scripts.llm_solver.harness.state_writer import write_state_from_trace
from scripts.llm_solver.harness.thoughts import (
    EMPTY_THINK_RESULT,
    filter_expired_thought_messages,
)
from scripts.llm_solver.harness.tools import dispatch
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


def _schema_names(schemas: list[dict]) -> set[str]:
    return {schema["function"]["name"] for schema in schemas}


def _think_message(call_id: str, thought: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": "think",
                "arguments": json.dumps({"thought": thought}),
            },
        }],
    }


def _projected_state(thoughts: list[str]) -> dict:
    trace = [
        {
            "step": index + 1,
            "session": 1,
            "turn": index,
            "reasoning": f"reasoning-{thought}",
            "action": f"think(thought='{thought}')",
            "result": EMPTY_THINK_RESULT,
            "next": "",
            "gate_blocked": False,
            "write_like": False,
            "source_write_like": False,
            "source_write_paths": [],
            "pass_fail": "pass",
            "output_sha256": "",
            "output_full_path": "",
        }
        for index, thought in enumerate(thoughts)
    ]
    return {
        "meta": {
            "schema_version": 1,
            "event_count": len(thoughts),
            "last_session": 1,
            "last_turn": len(thoughts) - 1,
        },
        "state": {
            "current_attempt": trace[-1]["action"],
            "last_verify": "",
            "next_action": "",
        },
        "trace": trace,
        "gates": [],
        "evidence": [],
        "inference": [],
    }


def test_canonical_defaults_and_overlay_load_think_knobs(tmp_path: Path) -> None:
    defaults = load_config()
    assert defaults.tools_think_enabled is False
    assert defaults.tools_think_keep_turns == 4
    assert defaults.think_streak_nudge_after == 3

    overlay = tmp_path / "think.toml"
    overlay.write_text(
        "[tools]\nthink_enabled = true\nthink_keep_turns = 7\n"
        "[loop]\nthink_streak_nudge_after = 5\n"
    )
    configured = load_config(user_config=overlay)
    assert configured.tools_think_enabled is True
    assert configured.tools_think_keep_turns == 7
    assert configured.think_streak_nudge_after == 5


@pytest.mark.parametrize(
    "body",
    [
        "[tools]\nthink_enabled = 'yes'\n",
        "[tools]\nthink_keep_turns = -1\n",
        "[loop]\nthink_streak_nudge_after = -1\n",
    ],
)
def test_invalid_think_knobs_fail_closed(tmp_path: Path, body: str) -> None:
    overlay = tmp_path / "invalid-think.toml"
    overlay.write_text(body)
    with pytest.raises(ValueError, match="think"):
        load_config(user_config=overlay)


def test_think_schema_is_profile_gated_and_has_one_string_argument() -> None:
    schemas = get_tool_schemas("minimal")
    disabled = _filter_disabled_tools(schemas, make_config())
    enabled = _filter_disabled_tools(
        schemas, make_config(tools_think_enabled=True)
    )

    assert "think" not in _schema_names(disabled)
    assert "think" in _schema_names(enabled)
    schema = next(
        item["function"]
        for item in enabled
        if item["function"]["name"] == "think"
    )
    assert schema["parameters"]["required"] == ["thought"]
    assert schema["parameters"]["properties"] == {
        "thought": {"type": "string"}
    }


def test_think_returns_an_empty_envelope_without_filesystem_or_process_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "marker.txt"
    marker.write_text("unchanged\n")
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("think attempted an external side effect")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    for method in (
        "write_text", "write_bytes", "mkdir", "touch", "unlink", "rename",
        "replace",
    ):
        monkeypatch.setattr(Path, method, forbidden)

    result = dispatch(
        "think",
        {"thought": "Inspect the narrow failure, then edit it."},
        cwd=str(tmp_path),
        cfg=make_config(tools_think_enabled=True),
    )

    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert result == (
        '<tool_result tool_name="think" status="ok" v="1"></tool_result>'
    )
    assert result == EMPTY_THINK_RESULT
    assert after == before


def test_think_is_non_mutating_and_its_streak_reuses_rumination_nudge(
    tmp_path: Path,
) -> None:
    cfg = make_config(
        max_turns=100,
        rumination_nudge_threshold_abs=50,
        rumination_gate_arm_threshold_abs=50,
        think_streak_nudge_after=3,
    )
    state = init_guardrail_state(cfg)

    decisions = [
        rumination_ladder(
            state,
            cfg,
            tc_name="think",
            result=EMPTY_THINK_RESULT,
            gate_blocked=False,
            already_blocked_this_turn=False,
            tc_args={"thought": f"plan {index}"},
        )
        for index in range(3)
    ]

    assert [decision.action for decision in decisions] == [
        Action.PASS,
        Action.PASS,
        Action.WARN,
    ]
    assert decisions[-1].text == cfg.rumination_nudge.format(count=3)
    assert state.think_streak == 3
    assert state.non_write_calls_since_write == 3
    assert state.has_mutated is False

    done = done_guard(state, cfg, tc_name="done", cwd=str(tmp_path))
    assert done.action == Action.BLOCK
    assert "No code changes" in done.text

    rumination_ladder(
        state,
        cfg,
        tc_name="read",
        result="contents",
        gate_blocked=False,
        already_blocked_this_turn=False,
        tc_args={"path": "README.md"},
    )
    assert state.think_streak == 0


def test_expiry_preserves_other_calls_in_a_mixed_assistant_turn() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _think_message("think-mixed", "DROP_MIXED_THOUGHT")[
                    "tool_calls"
                ][0],
                {
                    "id": "read-mixed",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"path": "README.md"}),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "think-mixed",
            "content": EMPTY_THINK_RESULT,
        },
        {
            "role": "tool",
            "tool_call_id": "read-mixed",
            "content": "README contents",
        },
    ]

    visible = filter_expired_thought_messages(messages, keep_turns=0)
    rendered = json.dumps(visible, sort_keys=True)
    assert "DROP_MIXED_THOUGHT" not in rendered
    assert "think-mixed" not in rendered
    assert "read-mixed" in rendered
    assert "README contents" in rendered


@pytest.mark.parametrize("mode", list_context_modes())
def test_every_context_strategy_drops_expired_thoughts(
    tmp_path: Path, mode: str,
) -> None:
    workdir = tmp_path / mode
    solver_dir = workdir / ".solver"
    solver_dir.mkdir(parents=True)
    thoughts = [
        "EXPIRED_THOUGHT_ZERO",
        "EXPIRED_THOUGHT_ONE",
        "RETAINED_THOUGHT_TWO",
        "RETAINED_THOUGHT_THREE",
    ]
    (solver_dir / "state.json").write_text(json.dumps(_projected_state(thoughts)))
    cfg = make_config(
        min_turns_before_context=0,
        recent_tool_results_chars=100_000,
        tools_think_keep_turns=2,
    )
    context = build_context_manager(
        resolve_context_class(mode),
        cfg,
        workdir,
        "TASK",
        1,
        token_estimator=None,
    )
    assert context is not None
    context.add_system("SYSTEM")
    context.add_user("TASK")
    for index, thought in enumerate(thoughts):
        call_id = f"think-{index}"
        context.add_assistant(_think_message(call_id, thought))
        context.add_tool_result(
            call_id,
            EMPTY_THINK_RESULT,
            tool_name="think",
        )

    rendered = json.dumps(context.get_messages(), sort_keys=True)
    assert "EXPIRED_THOUGHT_ZERO" not in rendered
    assert "EXPIRED_THOUGHT_ONE" not in rendered
    assert "RETAINED_THOUGHT_THREE" in rendered


@pytest.mark.parametrize("mode", ["full", "compact", "concise", "stateful"])
def test_each_context_family_keeps_thoughts_inside_the_window(
    tmp_path: Path, mode: str,
) -> None:
    workdir = tmp_path / mode
    solver_dir = workdir / ".solver"
    solver_dir.mkdir(parents=True)
    thoughts = ["OLD_FAMILY_THOUGHT", "CURRENT_FAMILY_THOUGHT"]
    state = _projected_state(thoughts)
    state["meta"]["last_turn"] = 1
    (solver_dir / "state.json").write_text(json.dumps(state))
    context = build_context_manager(
        resolve_context_class(mode),
        make_config(
            min_turns_before_context=0,
            recent_tool_results_chars=100_000,
            tools_think_keep_turns=1,
        ),
        workdir,
        "TASK",
        1,
        token_estimator=None,
    )
    assert context is not None
    context.add_system("SYSTEM")
    context.add_user("TASK")
    for index, thought in enumerate(thoughts):
        call_id = f"family-think-{index}"
        context.add_assistant(_think_message(call_id, thought))
        context.add_tool_result(
            call_id,
            EMPTY_THINK_RESULT,
            tool_name="think",
        )

    rendered = json.dumps(context.get_messages(), sort_keys=True)
    assert "OLD_FAMILY_THOUGHT" not in rendered
    assert "CURRENT_FAMILY_THOUGHT" in rendered


def test_trace_keeps_thought_while_state_projection_ages_it_out(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / ".trace.jsonl"
    state_path = tmp_path / ".solver" / "state.json"
    events = [
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": index,
            "tool_name": "think",
            "args_summary": f"thought='{thought}'",
            "reasoning": f"reasoning-{thought}",
            "output_snippet": EMPTY_THINK_RESULT,
            "pass_fail": "pass",
        }
        for index, thought in enumerate(
            ["RAW_OLD_THOUGHT", "RAW_RECENT_THOUGHT", "RAW_CURRENT_THOUGHT"]
        )
    ]
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )

    write_state_from_trace(
        trace_path,
        state_path,
        max_result_chars=20_000,
        think_keep_turns=2,
    )

    raw = trace_path.read_text()
    projected = json.loads(state_path.read_text())
    projected_text = json.dumps(projected, sort_keys=True)
    assert "RAW_OLD_THOUGHT" in raw
    assert "RAW_RECENT_THOUGHT" in raw
    assert "RAW_CURRENT_THOUGHT" in raw
    assert "RAW_OLD_THOUGHT" not in projected_text
    assert "RAW_RECENT_THOUGHT" in projected_text
    assert "RAW_CURRENT_THOUGHT" in projected_text
    assert projected["trace"][0]["action"] == "think()"
    assert projected["trace"][0]["reasoning"] == ""

    with trace_path.open("a") as trace_file:
        trace_file.write(
            json.dumps({
                "event": "session_start",
                "session_number": 2,
            })
            + "\n"
        )
    write_state_from_trace(
        trace_path,
        state_path,
        max_result_chars=20_000,
        think_keep_turns=2,
    )
    next_session_state = state_path.read_text()
    assert "RAW_OLD_THOUGHT" not in next_session_state
    assert "RAW_RECENT_THOUGHT" not in next_session_state
    assert "RAW_CURRENT_THOUGHT" not in next_session_state
    assert "RAW_CURRENT_THOUGHT" in trace_path.read_text()


def test_session_runtime_traces_the_thought_and_empty_result(tmp_path: Path) -> None:
    from scripts.llm_solver.harness.loop import Session

    thought = "TRACE_ME_THEN_RETURN_EMPTY"
    call = ToolCall(id="think-runtime", name="think", arguments={"thought": thought})
    client = MagicMock()
    client.chat.side_effect = [
        TurnResult(
            content="I will plan explicitly.",
            tool_calls=[call],
            finish_reason="tool_calls",
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        ),
        TurnResult(
            content="Finished planning.",
            tool_calls=[],
            finish_reason="stop",
            usage=Usage(prompt_tokens=12, completion_tokens=4),
        ),
    ]
    client.build_assistant_message.side_effect = [
        _think_message(call.id, thought),
        {"role": "assistant", "content": "Finished planning."},
    ]
    trace_path = tmp_path / ".trace.jsonl"
    state_path = tmp_path / ".solver" / "state.json"
    trace_file = trace_path.open("a")
    try:
        with patch.object(Session, "_get_server_ctx", return_value=0):
            session = Session(
                make_config(
                    max_turns=3,
                    duplicate_abort=10,
                    tools_think_enabled=True,
                    tools_think_keep_turns=4,
                    think_streak_nudge_after=0,
                ),
                client,
                "SYSTEM",
                "TASK",
                str(tmp_path),
                trace_file=trace_file,
                trace_path=trace_path,
                state_path=state_path,
                session_number=1,
            )
            result = session.run()
    finally:
        trace_file.close()

    assert result.done is True
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    tool_row = next(
        row
        for row in rows
        if row.get("event") == "tool_call" and row.get("tool_name") == "think"
    )
    assert thought in tool_row["args_summary"]
    assert tool_row["output_snippet"] == EMPTY_THINK_RESULT
    assert tool_row["write_like"] is False
    state = json.loads(state_path.read_text())
    assert thought in state["trace"][0]["action"]
