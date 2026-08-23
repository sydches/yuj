"""Runtime acceptance tests for declarative tool permissions."""
from __future__ import annotations

from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.llm_solver.bash_quirks.transforms import load_forbidden_rules
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._loop.trace_schema import (
    TRACE_EVENT_REQUIRED_FIELDS,
)
from scripts.llm_solver.harness.approvals import approval_decision
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness.state_writer import project
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage

from _config_helpers import make_config


def _turn(
    *,
    tool_calls: list[ToolCall] | None = None,
    content: str | None = None,
    reason: str = "tool_calls",
) -> TurnResult:
    return TurnResult(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=reason,
        usage=Usage(prompt_tokens=10, completion_tokens=2),
    )


def _client_for(calls: list[ToolCall]) -> MagicMock:
    client = MagicMock()
    client.chat.side_effect = [
        _turn(tool_calls=calls),
        _turn(content="done", reason="stop"),
    ]
    client.build_assistant_message.side_effect = [
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "assistant", "content": "done"},
    ]
    return client


def test_permission_config_defaults_overlay_order_and_validation(
    tmp_path: Path,
) -> None:
    defaults = load_config()
    assert defaults.permissions_rules == {}
    assert defaults.permissions_ask_fallback == "deny"

    overlay = tmp_path / "permissions.toml"
    overlay.write_text(
        "[permissions]\n"
        'ask_fallback = "allow"\n'
        "[permissions.rules.read]\n"
        '"*" = "deny"\n'
        '"docs/*" = "allow"\n'
    )
    configured = load_config(user_config=overlay)
    assert configured.permissions_ask_fallback == "allow"
    assert list(configured.permissions_rules["read"].items()) == [
        ("*", "deny"),
        ("docs/*", "allow"),
    ]

    configured.permissions_rules["read"]["private/*"] = "deny"
    assert "private/*" not in load_config(
        user_config=overlay
    ).permissions_rules["read"]

    for body, message in (
        ('ask_fallback = "ask"', "ask_fallback"),
        ('[permissions.rules.read]\n"*" = "prompt"', "must be one of"),
    ):
        invalid = tmp_path / f"invalid-{message}.toml"
        invalid.write_text(f"[permissions]\n{body}\n")
        with pytest.raises(ValueError, match=message):
            load_config(user_config=invalid)


def test_measurement_ask_denies_before_parallel_dispatch_and_counts_error(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / ".trace.jsonl"
    calls = [
        ToolCall(id="allowed", name="read", arguments={"path": "public.py"}),
        ToolCall(
            id="denied",
            name="read",
            arguments={"path": "private/customer.txt"},
        ),
    ]
    cfg = make_config(
        max_turns=2,
        parallel_readonly_enabled=True,
        permissions_rules={
            "read": {"*": "allow", "private/*": "ask"},
        },
        runtime_mode="measurement",
        error_nudge_threshold=99,
        error_abort_threshold=99,
        error_same_class_threshold=99,
        rumination_nudge_threshold=999,
    )
    client = _client_for(calls)

    with trace_path.open("a") as trace, (
        patch("scripts.llm_solver.harness.loop.dispatch", return_value="PUBLIC")
    ) as handler, patch.object(
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
            session_number=3,
        )
        result = session.run()

    assert result.finish_reason == "stop"
    assert handler.call_count == 1
    assert handler.call_args.args[:2] == ("read", {"path": "public.py"})
    assert session._guards.consecutive_errors["read"] == 1
    assert not (tmp_path / "approval_request.json").exists()

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    permissions = [event for event in events if event["event"] == "permission"]
    assert [event["decision"] for event in permissions] == ["allow", "deny"]
    assert "private/customer.txt" not in json.dumps(permissions)
    denied = next(
        event
        for event in events
        if event["event"] == "tool_call" and event.get("tool_call_id") == "denied"
    )
    assert denied["gate_blocked"] is True
    assert denied["gate_reason"] == "permission_denied"
    assert denied["error_class"] == "harness_gate"
    assert '"type":"permission_denied"' in denied["output_snippet"]
    assert TRACE_EVENT_REQUIRED_FIELDS["permission"] == frozenset({
        "session_number", "turn_number", "tool", "rule", "decision",
    })

    permission_only = project(permissions, max_result_chars=2000)
    assert permission_only["trace"] == []
    assert permission_only["evidence"] == []


def test_assistant_path_ask_writes_request_and_approved_resume_executes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "README.md"
    source.write_text("hello")
    trace_path = tmp_path / ".trace.jsonl"
    call = ToolCall(id="read-one", name="read", arguments={"path": "README.md"})
    cfg = make_config(
        max_turns=2,
        runtime_mode="assistant",
        permissions_rules={"read": {"README.md": "ask"}},
        rumination_nudge_threshold=999,
    )

    first_client = _client_for([call])
    with trace_path.open("a") as trace, patch.object(
        Session, "_get_server_ctx", return_value=cfg.context_size
    ), patch("scripts.llm_solver.harness.loop.dispatch") as handler:
        first = Session(
            cfg,
            first_client,
            "system",
            "task",
            str(tmp_path),
            trace_file=trace,
            trace_path=trace_path,
            session_number=1,
        )
        paused = first.run()
    assert paused.finish_reason == "approval_required"
    handler.assert_not_called()

    request_path = tmp_path / "approval_request.json"
    request = json.loads(request_path.read_text())
    assert request["status"] == "pending"
    assert request["tool_name"] == "read"
    assert request["action_key"].startswith("read:sha256:")
    assert request["permission_rule"] == "README.md"
    assert "cmd" not in request

    request["status"] = "approved"
    request_path.write_text(json.dumps(request) + "\n")
    resumed_client = _client_for([call])
    with trace_path.open("a") as trace, patch.object(
        Session, "_get_server_ctx", return_value=cfg.context_size
    ):
        resumed = Session(
            cfg,
            resumed_client,
            "system",
            "task",
            str(tmp_path),
            trace_file=trace,
            trace_path=trace_path,
            session_number=2,
        )
        finished = resumed.run()

    assert finished.finish_reason == "stop"
    assert not request_path.exists()
    tool_messages = [
        message
        for message in resumed.context.get_messages()
        if message.get("role") == "tool"
    ]
    assert any("hello" in str(message.get("content")) for message in tool_messages)

    (tmp_path / "approval_decisions.json").write_text(json.dumps({
        request["action_key"]: "approved",
    }))
    always_allowed, reason = approval_decision(
        runtime_mode="assistant",
        cwd=str(tmp_path),
        trace_path=trace_path,
        tool_name="read",
        tool_args={"path": "README.md"},
        args_summary="path='README.md'",
        required_reason="permission rule requires approval",
        permission_rule="README.md",
    )
    assert (always_allowed, reason) == (True, None)


def test_policy_precedes_bash_quirks_and_allow_does_not_bypass_forbidden(
    tmp_path: Path,
) -> None:
    call = ToolCall(id="bash-one", name="bash", arguments={"cmd": "cd /"})
    denied_cfg = make_config(
        max_turns=2,
        permissions_rules={"bash": {"cd /": "deny"}},
        runtime_mode="measurement",
        rumination_nudge_threshold=999,
    )
    denied_client = _client_for([call])
    with patch("scripts.llm_solver.harness.loop.dispatch") as dispatch_spy, (
        patch.object(Session, "_get_server_ctx", return_value=denied_cfg.context_size)
    ):
        denied = Session(
            denied_cfg,
            denied_client,
            "system",
            "task",
            str(tmp_path),
            trace_file=StringIO(),
            forbidden_rules=load_forbidden_rules(),
        )
        denied.run()
    dispatch_spy.assert_not_called()

    allowed_cfg = make_config(
        max_turns=2,
        permissions_rules={"bash": {"*": "allow"}},
        runtime_mode="measurement",
        rumination_nudge_threshold=999,
    )
    allowed_client = _client_for([call])
    admitted: list[str] = []
    with patch.object(
        Session, "_get_server_ctx", return_value=allowed_cfg.context_size
    ):
        allowed = Session(
            allowed_cfg,
            allowed_client,
            "system",
            "task",
            str(tmp_path),
            forbidden_rules=load_forbidden_rules(),
        )
        original_add = allowed.context.add_tool_result

        def capture(tool_call_id, result, **kwargs):
            admitted.append(result)
            return original_add(tool_call_id, result, **kwargs)

        allowed.context.add_tool_result = capture
        allowed.run()

    assert any("HARNESS refused this command" in result for result in admitted)


def test_adaptive_refresh_recompiles_permission_policy_atomically(
    tmp_path: Path,
) -> None:
    from scripts.llm_solver.harness.adaptive_control import executors

    cfg = make_config(permissions_rules={"read": {"*": "allow"}})
    session = Session(cfg, MagicMock(), "system", "task", str(tmp_path))
    updated = replace(
        cfg,
        permissions_rules={"read": {"*": "deny"}},
        permissions_ask_fallback="allow",
    )

    ok, reason, refreshed, blocked = executors._refresh_runtime_surfaces(
        session,
        cfg,
        updated,
        {"permissions_rules", "permissions_ask_fallback"},
    )

    assert (ok, reason, blocked) == (True, "", ())
    assert "permission_policy" in refreshed
    assert session._permission_policy.evaluate(
        tool_name="read",
        arguments={"path": "README.md"},
        runtime_mode="measurement",
    ).denied
