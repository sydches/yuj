"""Artifact acceptance coverage for the default deferred tool block."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from _config_helpers import make_config
from scripts.llm_solver._shared.telemetry_paths import trace_path
from scripts.llm_solver.harness.loop import Session, solve_task
from scripts.llm_solver.server.types import TurnResult, Usage


def test_metrics_trace_and_state_report_default_tool_block(
    tmp_path: Path,
) -> None:
    (tmp_path / "prompt.txt").write_text("Inspect the repository.")
    client = MagicMock()
    client.profile = SimpleNamespace(max_tools=8, simplify_schemas=False)
    client.chat.return_value = TurnResult(
        content="Done.",
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(prompt_tokens=50, completion_tokens=3),
    )
    client.build_assistant_message.return_value = {
        "role": "assistant",
        "content": "Done.",
    }
    cfg = make_config(
        max_turns=1,
        max_sessions=1,
        tools_lazy_loading_enabled=True,
        tools_active_default=("bash", "read", "edit", "glob", "grep", "done"),
    )

    with (
        patch("llm_solver.harness.loop._auto_commit"),
        patch.object(Session, "_get_server_ctx", return_value=cfg.context_size),
    ):
        assert solve_task(tmp_path, cfg, client) is True

    metrics = json.loads((tmp_path / "metrics.json").read_text())["metrics"]
    loading = metrics["tool_loading"]
    assert loading["lazy_loading_enabled"] is True
    assert loading["active_tool_limit"] == 8
    assert loading["default_active_tools"] == [
        "bash", "read", "edit", "glob", "grep", "load_tools", "done"
    ]
    assert "write" in loading["registered_tools"]
    assert "write" not in loading["default_active_tools"]
    assert isinstance(loading["default_tool_block_tokens"], int)
    assert loading["default_tool_block_tokens"] > 0
    assert loading["token_count_method"] == "chars_div_4"
    assert loading["activation_events"] == 0

    events = [
        json.loads(line)
        for line in trace_path(tmp_path).read_text().splitlines()
    ]
    start = next(event for event in events if event["event"] == "session_start")
    assert start["tool_lazy_loading_enabled"] is True
    assert start["tool_active_limit"] == 8
    assert start["active_tools"] == loading["default_active_tools"]
    assert start["registered_tools"] == loading["registered_tools"]

    state = json.loads((tmp_path / ".solver" / "state.json").read_text())
    assert state["tools"]["active"] == loading["default_active_tools"]
    assert state["tools"]["active_limit"] == 8
    assert state["tools"]["activations"] == []
