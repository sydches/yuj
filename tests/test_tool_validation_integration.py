"""Runtime acceptance tests for tool schemas and constrained decoding."""
from __future__ import annotations

from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from _config_helpers import make_config
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._loop.trace_schema import (
    TRACE_EVENT_REQUIRED_FIELDS,
)
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness.schemas import get_tool_schemas
from scripts.llm_solver.server.client import LlamaClient
from scripts.llm_solver.server.profile_loader import load_profile
from scripts.llm_solver.server.security import validate_profile
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILES = PROJECT_ROOT / "profiles"


def _turn(*, tool_calls=(), content=None, reason="tool_calls") -> TurnResult:
    return TurnResult(
        content=content,
        tool_calls=list(tool_calls),
        finish_reason=reason,
        usage=Usage(prompt_tokens=10, completion_tokens=2),
    )


def _profile(*, supports_constrained_tools: bool):
    return replace(
        load_profile("_base", PROFILES),
        supports_constrained_tools=supports_constrained_tools,
        _normalize_rules=[],
        _normalize_modules=[],
    )


def _api_response() -> dict:
    return {
        "choices": [{
            "message": {"content": "ok", "tool_calls": []},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    }


def test_tool_validation_config_defaults_overlays_and_rejects(
    tmp_path: Path,
) -> None:
    defaults = load_config()
    assert defaults.tools_schema_validation == "off"
    assert defaults.tools_constrained_decoding == "off"

    overlay = tmp_path / "tools.toml"
    overlay.write_text(
        '[tools]\nschema_validation = "reject"\n'
        'constrained_decoding = "grammar"\n'
    )
    configured = load_config(user_config=overlay)
    assert configured.tools_schema_validation == "reject"
    assert configured.tools_constrained_decoding == "grammar"

    for key, value in (
        ("schema_validation", '"repair"'),
        ("constrained_decoding", '"regex"'),
    ):
        overlay.write_text(f"[tools]\n{key} = {value}\n")
        with pytest.raises(ValueError, match=f"tools.{key}"):
            load_config(user_config=overlay)


def test_model_facing_parameter_objects_are_closed() -> None:
    assert all(
        schema["function"]["parameters"]["additionalProperties"] is False
        for schema in get_tool_schemas()
    )


def test_profile_constrained_capability_inherits_and_validates(
    tmp_path: Path,
) -> None:
    assert load_profile("_base", PROFILES).supports_constrained_tools is False
    assert load_profile(
        "qwen3.6-35b-a3b", PROFILES
    ).supports_constrained_tools is False
    assert load_profile("qwen38-27b", PROFILES).supports_constrained_tools is False

    shutil.copytree(PROFILES / "_base", tmp_path / "_base")
    child = tmp_path / "child"
    child.mkdir()
    (child / "profile.toml").write_text(
        """
[profile]
format_version = 1
name = "child"
inherits = "_base"

[model]
supports_constrained_tools = true
""".strip()
        + "\n"
    )
    assert load_profile("child", tmp_path).supports_constrained_tools is True

    (child / "profile.toml").write_text(
        """
[profile]
format_version = 1
name = "child"
inherits = "_base"

[model]
supports_constrained_tools = "yes"
""".strip()
        + "\n"
    )
    violations = validate_profile(child)
    assert len(violations) == 1
    assert "supports_constrained_tools must be a boolean" in violations[0]


@pytest.mark.parametrize("mode", ("json_schema", "grammar"))
def test_real_profile_request_carries_capability_approved_constraint(
    mode: str,
) -> None:
    cfg = make_config(tools_constrained_decoding=mode)
    client = LlamaClient(
        cfg,
        profile=_profile(supports_constrained_tools=True),
    )
    client._call_api = MagicMock(return_value=_api_response())

    result = client.chat(
        [{"role": "user", "content": "inspect"}],
        get_tool_schemas(),
    )

    assert result.finish_reason == "stop"
    request = client._call_api.call_args.args[0]
    assert request["tools"]
    extra = request["extra_body"]
    if mode == "json_schema":
        assert extra["json_schema"]["title"] == "Yuj tool call"
        assert "grammar" not in extra
    else:
        assert extra["grammar"].startswith("root ::=")
        assert extra["grammar_type"] == "tool_calls"
        assert "json_schema" not in extra
    assert "cache_prompt" in extra


def test_unsupported_profile_and_side_request_receive_no_constraint() -> None:
    cfg = make_config(tools_constrained_decoding="grammar")
    client = LlamaClient(
        cfg,
        profile=_profile(supports_constrained_tools=False),
    )
    client._call_api = MagicMock(return_value=_api_response())

    client.chat(
        [{"role": "user", "content": "inspect"}],
        get_tool_schemas(),
    )
    normal = client._call_api.call_args.args[0]
    assert "grammar" not in normal["extra_body"]
    assert "json_schema" not in normal["extra_body"]

    client._call_api.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="ok", tool_calls=[])
        )],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
    )
    client.complete_side_request({
        "messages": [{"role": "user", "content": "classify"}],
    })
    side = client._call_api.call_args.args[0]
    assert "tools" not in side
    assert "grammar" not in side["extra_body"]
    assert "json_schema" not in side["extra_body"]


def test_reject_mode_blocks_invalid_parallel_reads_and_traces_ladder(
    tmp_path: Path,
) -> None:
    calls = [
        ToolCall(id="invalid-1", name="read", arguments={}),
        ToolCall(id="invalid-2", name="read", arguments={"path": 7}),
    ]
    client = MagicMock()
    client.chat.side_effect = [
        _turn(tool_calls=calls),
        _turn(content="done", reason="stop"),
    ]
    client.build_assistant_message.side_effect = [
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "assistant", "content": "done"},
    ]
    cfg = make_config(
        max_turns=3,
        parallel_readonly_enabled=True,
        tools_schema_validation="reject",
        error_nudge_threshold=99,
        error_abort_threshold=99,
        error_same_class_threshold=99,
        rumination_nudge_threshold=999,
    )
    trace = StringIO()
    session = Session(
        cfg,
        client,
        "system",
        "task",
        str(tmp_path),
        trace_file=trace,
        session_number=4,
    )
    captured: list[str] = []
    original_add = session.context.add_tool_result

    def capture(tool_call_id, result, **kwargs):
        captured.append(result)
        return original_add(tool_call_id, result, **kwargs)

    session.context.add_tool_result = capture
    with (
        patch("scripts.llm_solver.harness.loop.dispatch") as handler,
        patch.object(Session, "_get_server_ctx", return_value=cfg.context_size),
    ):
        result = session.run()

    assert result.finish_reason == "stop"
    handler.assert_not_called()
    assert session._guards.consecutive_errors["read"] == 2
    assert len(captured) == 2
    assert all('"type":"tool_schema_reject"' in item for item in captured)

    events = [json.loads(line) for line in trace.getvalue().splitlines()]
    rejects = [event for event in events if event["event"] == "schema_reject"]
    attempted = [event for event in events if event["event"] == "tool_call"]
    assert len(rejects) == 2
    assert [item["errors"][0]["path"] for item in rejects] == [
        "$.path", "$.path",
    ]
    assert all(item["gate_blocked"] is True for item in attempted)
    assert all(item["gate_reason"] == "schema_reject" for item in attempted)
    assert TRACE_EVENT_REQUIRED_FIELDS["schema_reject"] == frozenset({
        "session_number", "turn_number", "tool", "errors",
    })


def test_invalid_done_is_rejected_before_done_guard(tmp_path: Path) -> None:
    client = MagicMock()
    client.chat.side_effect = [
        _turn(tool_calls=[ToolCall(
            id="bad-done", name="done", arguments={"message": 1}
        )]),
        _turn(content="continue", reason="stop"),
    ]
    client.build_assistant_message.side_effect = [
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "assistant", "content": "continue"},
    ]
    cfg = make_config(
        max_turns=2,
        tools_schema_validation="reject",
        error_nudge_threshold=99,
    )
    session = Session(cfg, client, "system", "task", str(tmp_path))

    with patch.object(Session, "_get_server_ctx", return_value=cfg.context_size):
        result = session.run()

    assert result.finish_reason == "stop"
    assert result.done is True
    assert session._guards.done_blocked_count == 0


def test_adaptive_tool_surface_rebuilds_schema_set_atomically(
    tmp_path: Path,
) -> None:
    from scripts.llm_solver.harness.adaptive_control import executors

    cfg = make_config(tools_run_tests_enabled=False)
    client = MagicMock()
    session = Session(cfg, client, "system", "task", str(tmp_path))
    assert "run_tests" not in session._tool_schema_set.names
    updated = replace(cfg, tools_run_tests_enabled=True)

    ok, reason, refreshed, blocked = executors._refresh_runtime_surfaces(
        session,
        cfg,
        updated,
        {"tools_run_tests_enabled"},
    )

    assert ok is True
    assert reason == ""
    assert blocked == ()
    assert "tool_schemas" in refreshed
    names = tuple(
        schema["function"]["name"] for schema in session._tool_schemas
    )
    assert session._tool_schema_set.names == names
    assert "run_tests" in names
