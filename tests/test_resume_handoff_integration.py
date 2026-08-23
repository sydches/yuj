"""Runtime acceptance tests for fresh-session handoff summaries."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _config_helpers import make_config
from llm_solver._shared.telemetry_paths import trace_path
from llm_solver.config import load_config
from llm_solver.harness._loop.handoff_integration import generate_boundary_handoff
from llm_solver.harness.loop import solve_task
from llm_solver.server.types import SideRequestResult, ToolCall, TurnResult, Usage


def _valid_handoff() -> str:
    return """\
## Goal
Fix the requested behavior.
## Done
Inspected the current implementation.
## In progress
Continuing the runtime integration.
## Blocked
Nothing is blocked.
## Key decisions
Keep the mechanical resume prompt as the fallback floor.
## Critical paths/errors
HANDOFF SENTINEL preserves the exact rollover evidence.
## Next step
Run the next focused verification."""


def _turn(*, tool: bool) -> TurnResult:
    calls = (
        [ToolCall(id="call-1", name="read", arguments={"path": "README.md"})]
        if tool
        else []
    )
    return TurnResult(
        content="Inspect." if tool else "Done.",
        tool_calls=calls,
        finish_reason="tool_calls" if tool else "stop",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


def _client(side_text: str) -> MagicMock:
    client = MagicMock()
    client.chat.side_effect = [_turn(tool=True), _turn(tool=False)]
    client.build_assistant_message.return_value = {
        "role": "assistant",
        "content": "Inspect.",
    }
    client.complete_side_request.return_value = SideRequestResult(
        content=side_text,
        usage=Usage(prompt_tokens=120, completion_tokens=80),
    )
    return client


def test_boundary_handoff_uses_one_no_tool_same_model_request() -> None:
    cfg = make_config(
        handoff_summary_enabled=True,
        handoff_max_tokens=2_000,
        context_size=16_000,
    )
    client = _client(_valid_handoff())

    attempt = generate_boundary_handoff(
        cfg=cfg,
        client=client,
        task="Fix it.",
        trace_events=[
            {
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "max_turns",
                "turns": 1,
            }
        ],
        session_number=1,
    )

    assert attempt.result.valid is True
    assert attempt.usage == Usage(prompt_tokens=120, completion_tokens=80)
    client.complete_side_request.assert_called_once()
    request = client.complete_side_request.call_args.args[0]
    assert request["model"] == cfg.model
    assert request["max_tokens"] == cfg.handoff_max_tokens
    assert "tools" not in request
    assert "tool_choice" not in request
    assert request["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_handoff_config_defaults_and_enabled_token_validation(
    tmp_path: Path,
) -> None:
    default = load_config()
    assert default.handoff_summary_enabled is False
    assert default.handoff_max_tokens == 2_000

    overlay = tmp_path / "invalid-handoff.toml"
    overlay.write_text(
        "[loop]\nhandoff_summary_enabled = true\n"
        "[prompts]\nhandoff_max_tokens = 0\n"
    )
    with pytest.raises(ValueError, match="handoff_max_tokens"):
        load_config(overlay)


def test_valid_handoff_is_inserted_once_and_trace_keeps_metadata_only(
    tmp_path: Path,
) -> None:
    (tmp_path / "prompt.txt").write_text("Fix it.")
    cfg = make_config(
        max_turns=1,
        max_sessions=2,
        handoff_summary_enabled=True,
        handoff_max_tokens=2_000,
        context_size=16_000,
    )
    client = _client(_valid_handoff())

    with (
        patch("llm_solver.harness.loop._auto_commit"),
        patch("llm_solver.harness.loop.dispatch", return_value="README"),
        patch("llm_solver.harness.loop.Session._get_server_ctx", return_value=16_000),
    ):
        assert solve_task(tmp_path, cfg, client) is True

    client.complete_side_request.assert_called_once()
    second_messages = client.chat.call_args_list[1].args[0]
    initial = second_messages[1]["content"]
    assert initial.startswith("Task:\nFix it.\n\n<handoff>\n## Goal")
    assert "HANDOFF SENTINEL" in initial
    assert "Previous session ended after 1 turns: max_turns." in initial
    assert initial.endswith(cfg.resume_base)

    events = [
        json.loads(line)
        for line in trace_path(tmp_path).read_text().splitlines()
        if line.strip()
    ]
    handoffs = [event for event in events if event["event"] == "handoff"]
    assert len(handoffs) == 1
    assert handoffs[0].keys() >= {
        "tokens", "valid", "fallback", "role", "session_number"
    }
    assert handoffs[0]["valid"] is True
    assert handoffs[0]["fallback"] is None
    assert handoffs[0]["role"] == "main"
    assert events.index(handoffs[0]) > max(
        index
        for index, event in enumerate(events)
        if event["event"] == "session_end" and event["session_number"] == 1
    )
    assert events.index(handoffs[0]) < min(
        index
        for index, event in enumerate(events)
        if event["event"] == "session_start" and event["session_number"] == 2
    )

    state = json.loads((tmp_path / ".solver" / "state.json").read_text())
    assert "HANDOFF SENTINEL" not in json.dumps(state)


def test_invalid_handoff_leaves_runtime_resume_prompt_byte_identical(
    tmp_path: Path,
) -> None:
    (tmp_path / "prompt.txt").write_text("Fix it.")
    cfg = make_config(
        max_turns=1,
        max_sessions=2,
        handoff_summary_enabled=True,
        handoff_max_tokens=2_000,
        context_size=16_000,
    )
    client = _client("missing required headers")
    captured: list[bytes] = []

    def _capture(mechanical: str, *, task: str, handoff) -> str:
        assert task == "Fix it."
        assert handoff.valid is False
        captured.append(mechanical.encode())
        return mechanical

    with (
        patch("llm_solver.harness.loop._auto_commit"),
        patch("llm_solver.harness.loop.dispatch", return_value="README"),
        patch("llm_solver.harness.loop.Session._get_server_ctx", return_value=16_000),
        patch(
            "llm_solver.harness._loop.driver.apply_pending_handoff",
            side_effect=_capture,
        ),
    ):
        assert solve_task(tmp_path, cfg, client) is True

    client.complete_side_request.assert_called_once()
    second_initial = client.chat.call_args_list[1].args[0][1]["content"]
    assert captured == [second_initial.encode()]
    assert "<handoff>" not in second_initial

    events = [
        json.loads(line)
        for line in trace_path(tmp_path).read_text().splitlines()
        if line.strip()
    ]
    event = next(item for item in events if item["event"] == "handoff")
    assert event["valid"] is False
    assert event["fallback"] == "mechanical"


def test_disabled_handoff_does_not_issue_side_request(tmp_path: Path) -> None:
    (tmp_path / "prompt.txt").write_text("Fix it.")
    cfg = make_config(
        max_turns=1,
        max_sessions=2,
        handoff_summary_enabled=False,
        context_size=16_000,
    )
    client = _client(_valid_handoff())

    with (
        patch("llm_solver.harness.loop._auto_commit"),
        patch("llm_solver.harness.loop.dispatch", return_value="README"),
        patch("llm_solver.harness.loop.Session._get_server_ctx", return_value=16_000),
    ):
        assert solve_task(tmp_path, cfg, client) is True

    client.complete_side_request.assert_not_called()
    second_initial = client.chat.call_args_list[1].args[0][1]["content"]
    assert "<handoff>" not in second_initial
