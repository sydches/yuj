"""Acceptance coverage for deferred model-tool activation."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from _config_helpers import make_config
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness.context_contract import build_context_contract
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness._loop.profile_resolution import build_tool_surface
from scripts.llm_solver.harness.tool_loading import ToolLoadingError
from scripts.llm_solver.server.replay_client import ReplayClient, ReplayDivergence
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


class _ScriptedClient:
    def __init__(self, turns: list[TurnResult]):
        self._turns = iter(turns)
        self.request_tool_names: list[tuple[str, ...]] = []

    def chat(self, messages, tools, turn=0):
        self.request_tool_names.append(tuple(
            schema["function"]["name"] for schema in tools
        ))
        return next(self._turns)

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


def _turn(
    *,
    tool: ToolCall | None = None,
    content: str | None = None,
    cached_tokens: int | None = None,
):
    return TurnResult(
        content=content,
        tool_calls=[] if tool is None else [tool],
        finish_reason="stop" if tool is None else "tool_calls",
        usage=Usage(
            prompt_tokens=40,
            completion_tokens=5,
            cached_tokens=cached_tokens,
            cache_hit_ratio=(
                None if cached_tokens is None else cached_tokens / 40
            ),
        ),
    )


def test_config_knobs_and_context_contract_are_public(tmp_path: Path) -> None:
    overlay = tmp_path / "lazy-tools.toml"
    overlay.write_text(
        "[tools]\n"
        "lazy_loading_enabled = true\n"
        'active_default = ["bash", "read", "done"]\n'
    )
    cfg = load_config(user_config=overlay)

    assert cfg.tools_lazy_loading_enabled is True
    assert cfg.tools_active_default == ("bash", "read", "done")
    assert build_context_contract(None, cfg)["tool_loading"] == {
        "lazy_loading_enabled": True,
        "active_default": ["bash", "read", "done"],
        "loader_tool": "load_tools",
        "activation_event": "tools_activated",
    }

    overlay.write_text(
        "[tools]\n"
        "lazy_loading_enabled = true\n"
        'active_default = ["not_a_tool"]\n'
    )
    with pytest.raises(ValueError, match="unknown tool names: not_a_tool"):
        load_config(user_config=overlay)


def test_hidden_call_is_rejected_then_loader_changes_next_request(
    tmp_path: Path,
) -> None:
    client = _ScriptedClient([
        _turn(tool=ToolCall(
            id="hidden-write",
            name="write",
            arguments={"path": "created.txt", "content": "wrong turn\n"},
        ), cached_tokens=10),
        _turn(tool=ToolCall(
            id="load-write",
            name="load_tools",
            arguments={"names": ["write"]},
        ), cached_tokens=20),
        _turn(content="The needed tool is active.", cached_tokens=30),
    ])
    cfg = make_config(
        max_turns=3,
        tools_lazy_loading_enabled=True,
        tools_active_default=("bash", "read", "glob", "grep", "done"),
        tools_edit_format="whole",
        tools_schema_validation="reject",
        error_nudge_threshold=99,
    )
    trace_path = tmp_path / ".trace.jsonl"
    state_path = tmp_path / ".solver" / "state.json"
    with trace_path.open("a+") as trace:
        session = Session(
            cfg,
            client,
            "system",
            "task",
            str(tmp_path),
            trace_file=trace,
            trace_path=trace_path,
            state_path=state_path,
            session_number=1,
        )
        session._emit(
            "session_start",
            session_number=1,
            thinking_level="off",
            sandbox_backend="bwrap",
            container_runtime=None,
            container_image_digest=None,
            ignore_file_hash=None,
            sandbox_env_names=[],
            edit_format="whole",
            repo_map_tokens=0,
            tool_lazy_loading_enabled=True,
            registered_tools=list(session._tool_surface.registered_names),
            active_tools=list(session._tool_surface.default_active_names),
        )

        with patch.object(
            Session, "_get_server_ctx", return_value=cfg.context_size
        ):
            result = session.run()

    assert result.done is True
    assert not (tmp_path / "created.txt").exists()
    assert "write" not in client.request_tool_names[0]
    assert "write" not in client.request_tool_names[1]
    assert "write" in client.request_tool_names[2]
    assert "load_tools" in client.request_tool_names[0]

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    tool_events = [event for event in events if event["event"] == "tool_call"]
    hidden_result = tool_events[0]["result_summary"]
    loader_result = tool_events[1]["result_summary"]
    assert '"type":"tool_not_active"' in hidden_result
    assert '"loader":"load_tools"' in hidden_result
    assert "Activated tools: write" in loader_result

    activation = next(
        event for event in events if event["event"] == "tools_activated"
    )
    assert activation["requested"] == ["write"]
    assert activation["activated"] == ["write"]
    assert "write" in activation["active_tools"]
    following_turn = next(
        event
        for event in events
        if event["event"] == "turn"
        and event["turn_number"] == activation["turn_number"] + 1
    )
    assert following_turn["cached_tokens"] == 30
    assert following_turn["cache_hit_ratio"] == 0.75

    projected = json.loads(state_path.read_text())
    assert projected["tools"]["lazy_loading_enabled"] is True
    assert "write" in projected["tools"]["active"]
    assert projected["tools"]["activations"] == [{
        "session": 1,
        "turn": 1,
        "requested": ["write"],
        "activated": ["write"],
        "already_active": [],
        "active": activation["active_tools"],
    }]


def test_profile_cap_limits_active_tools_without_erasing_registry() -> None:
    cfg = make_config(
        tools_lazy_loading_enabled=True,
        tools_active_default=("bash", "read", "edit", "glob", "grep", "done"),
        tools_run_tests_enabled=True,
        tools_list_definitions_enabled=True,
        tools_apply_patch_enabled=True,
    )
    client = SimpleNamespace(profile=SimpleNamespace(
        max_tools=8,
        simplify_schemas=False,
    ))
    surface = build_tool_surface(cfg, client)

    assert surface.max_active_tools == 8
    assert {"run_tests", "list_definitions", "apply_patch"} <= set(
        surface.registered_names
    )
    assert surface.is_hidden("run_tests")
    surface.activate(["run_tests"])
    before_reject = surface.active_names
    with pytest.raises(ToolLoadingError, match="active-tool limit 8"):
        surface.activate(["list_definitions"])
    assert surface.active_names == before_reject


def _transcript_record(turn: int, kind: str, payload: dict) -> str:
    return (
        f"=== turn {turn:03d} {kind} ===\n"
        + json.dumps(payload, separators=(",", ":"))
        + "\n"
    )


def test_replay_restores_the_active_surface_at_the_recorded_turn(
    tmp_path: Path,
) -> None:
    cfg = make_config(
        max_turns=2,
        tools_lazy_loading_enabled=True,
        tools_active_default=("bash", "read", "glob", "grep", "done"),
        tools_edit_format="whole",
        error_nudge_threshold=99,
    )
    surface = build_tool_surface(cfg, object())
    default_tools = surface.active_schemas
    surface.activate(["write"])
    expanded_tools = surface.active_schemas
    transcript = tmp_path / "recording.log"
    transcript.write_text(
        _transcript_record(1, "input", {"messages": [], "tools": default_tools})
        + _transcript_record(1, "output", {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "load-write",
                        "type": "function",
                        "function": {
                            "name": "load_tools",
                            "arguments": json.dumps({"names": ["write"]}),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 40, "completion_tokens": 5},
        })
        + _transcript_record(2, "input", {"messages": [], "tools": expanded_tools})
        + _transcript_record(2, "output", {
            "choices": [{
                "message": {"content": "Done.", "tool_calls": []},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 45, "completion_tokens": 2},
        })
    )

    replay = ReplayClient(transcript)
    session = Session(cfg, replay, "system", "task", str(tmp_path))
    with patch.object(Session, "_get_server_ctx", return_value=cfg.context_size):
        result = session.run()

    assert result.done is True
    assert replay.divergence is None
    assert "write" in session.active_tool_names


def test_replay_rejects_a_missing_activation_on_its_following_turn(
    tmp_path: Path,
) -> None:
    cfg = make_config(
        tools_lazy_loading_enabled=True,
        tools_active_default=("bash", "read", "glob", "grep", "done"),
        tools_edit_format="whole",
    )
    surface = build_tool_surface(cfg, object())
    default_tools = surface.active_schemas
    surface.activate(["write"])
    expanded_tools = surface.active_schemas
    output = {
        "choices": [{
            "message": {"content": "continue", "tool_calls": []},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    transcript = tmp_path / "mismatched-recording.log"
    transcript.write_text(
        _transcript_record(1, "input", {"messages": [], "tools": default_tools})
        + _transcript_record(1, "output", output)
        + _transcript_record(2, "input", {"messages": [], "tools": expanded_tools})
        + _transcript_record(2, "output", output)
    )

    replay = ReplayClient(transcript)
    replay.chat([], default_tools)
    with pytest.raises(ReplayDivergence, match="recorded turn 2"):
        replay.chat([], default_tools)
    assert replay.divergence == {
        "turn": 2,
        "field": "tools",
        "live_tools": [tool["function"]["name"] for tool in default_tools],
        "recorded_tools": [tool["function"]["name"] for tool in expanded_tools],
    }
