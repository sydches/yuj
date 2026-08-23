"""Acceptance tests for issue #7 code mode and eval-cell wiring."""
from __future__ import annotations

import json
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from _config_helpers import make_config
from scripts.llm_solver.bash_quirks import RedactionRule
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness.schemas import (
    get_exec_cell_function_schemas,
    get_tool_schemas,
)
from scripts.llm_solver.harness.tool_specs import (
    CODE_MODE_SCHEMA_TOOL_NAMES,
    EXEC_CELL_API_TOOL_NAMES,
    TOOL_SPECS,
)
from scripts.llm_solver.harness.tools import dispatch
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


def _turn(*, calls=(), content=None, reason="tool_calls") -> TurnResult:
    return TurnResult(
        content=content,
        tool_calls=list(calls),
        finish_reason=reason,
        usage=Usage(prompt_tokens=20, completion_tokens=5),
    )


def _client(*turns: TurnResult):
    client = MagicMock()
    client.chat.side_effect = turns
    client.build_assistant_message.side_effect = lambda content, tool_calls: {
        "role": "assistant",
        "content": content,
        "tool_calls": [],
    }
    return client


def _estimated_schema_tokens(schemas: list[dict]) -> int:
    """Use Yuj's default no-tokenizer estimate: one token per four chars."""
    rendered = json.dumps(
        schemas, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return (len(rendered) + 3) // 4


def test_exec_cell_config_defaults_overlay_and_validation(tmp_path: Path):
    defaults = load_config()
    assert defaults.tools_exec_cell_enabled is False
    assert defaults.tools_exec_cell_timeout == 30

    overlay = tmp_path / "code-mode.toml"
    overlay.write_text(
        "[tools]\nexec_cell_enabled = true\nexec_cell_timeout = 9\n"
    )
    configured = load_config(user_config=overlay)
    assert configured.tools_exec_cell_enabled is True
    assert configured.tools_exec_cell_timeout == 9

    for value in ('"yes"', "0", "1.5"):
        overlay.write_text(f"[tools]\nexec_cell_timeout = {value}\n")
        with pytest.raises(ValueError, match="exec_cell_timeout"):
            load_config(user_config=overlay)
    overlay.write_text('[tools]\nexec_cell_enabled = "yes"\n')
    with pytest.raises(ValueError, match="exec_cell_enabled"):
        load_config(user_config=overlay)


def test_code_mode_schema_surface_has_measurably_fewer_prompt_tokens():
    native = get_tool_schemas("minimal", code_mode=False)
    code_mode = get_tool_schemas("minimal", code_mode=True)

    assert tuple(
        item["function"]["name"] for item in code_mode
    ) == CODE_MODE_SCHEMA_TOOL_NAMES
    declared_active = {spec.name for spec in TOOL_SPECS if spec.active}
    assert set(CODE_MODE_SCHEMA_TOOL_NAMES) <= declared_active

    native_tokens = _estimated_schema_tokens(native)
    code_tokens = _estimated_schema_tokens(code_mode)
    assert code_tokens < native_tokens
    assert code_tokens <= native_tokens // 2


def test_code_mode_discovery_returns_exact_injected_function_specs():
    cfg = make_config(
        tools_exec_cell_enabled=True,
        tools_unified_envelope_enabled=False,
    )
    listed = json.loads(
        dispatch("list_functions", {}, cwd="/tmp", cfg=cfg)
    )
    assert tuple(listed["functions"]) == EXEC_CELL_API_TOOL_NAMES

    details = json.loads(dispatch(
        "get_function_details",
        {"names": ["read", "bash"]},
        cwd="/tmp",
        cfg=cfg,
    ))
    assert [item["name"] for item in details["functions"]] == [
        "read", "bash",
    ]
    bash = details["functions"][1]
    assert "background" not in bash["parameters"]["properties"]
    assert {
        item["function"]["name"]
        for item in get_exec_cell_function_schemas()
    } == set(EXEC_CELL_API_TOOL_NAMES)


def test_code_mode_keeps_complete_meta_surface_under_profile_shaping(
    tmp_path: Path,
):
    cfg = make_config(tools_exec_cell_enabled=True)
    profile = SimpleNamespace(max_tools=1, simplify_schemas=True)
    client = MagicMock()
    client.__dict__["profile"] = profile
    session = Session(cfg, client, "system", "task", str(tmp_path))

    assert tuple(
        schema["function"]["name"] for schema in session._tool_schemas
    ) == CODE_MODE_SCHEMA_TOOL_NAMES
    assert all(
        "description" not in schema["function"]
        for schema in session._tool_schemas
    )


@pytest.mark.skipif(
    not Path("/usr/bin/bwrap").is_file(),
    reason="bwrap is required for the live exec_cell dispatcher check",
)
def test_inner_calls_reenter_dispatch_filters_redaction_and_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "sample.txt"
    target.write_text("SECRET-123\n")
    cfg = make_config(
        tools_exec_cell_enabled=True,
        tools_exec_cell_timeout=10,
        sandbox_bash=True,
        sandbox_required=True,
        tools_unified_envelope_enabled=True,
        collapse_duplicate_lines=True,
        collapse_blank_lines=False,
        collapse_similar_lines=False,
        strip_ansi=False,
    )
    redactions = [
        RedactionRule("test-secret", re.compile(r"SECRET-123"), "[REDACTED]")
    ]
    session = Session(
        cfg, MagicMock(), "system", "task", str(tmp_path),
        redactions=redactions,
    )

    import scripts.llm_solver.harness.loop as loop_module
    import scripts.llm_solver.harness.tools as tools_module

    real_dispatch = tools_module.dispatch
    inner_names: list[str] = []

    def tracking_dispatch(name, arguments, **kwargs):
        inner_names.append(name)
        return real_dispatch(name, arguments, **kwargs)

    monkeypatch.setattr(loop_module, "dispatch", tracking_dispatch)
    result = real_dispatch(
        "exec_cell",
        {
            "source": (
                'print(read("sample.txt"))\n'
                'print(bash("printf \\\'row\\\\nrow\\\\nrow\\\\n\\\'"))'
            )
        },
        cwd=str(tmp_path),
        cfg=cfg,
        tool_registry=session._tool_registry,
        redactions=redactions,
    )

    assert inner_names == ["read", "bash"]
    assert "SECRET-123" not in result
    assert "[REDACTED]" in result
    assert "row [×3]" in result
    assert result.startswith('<tool_result tool_name="exec_cell"')
    assert 'tool_name="read"' in result
    assert 'tool_name="bash"' in result


@pytest.mark.skipif(
    not Path("/usr/bin/bwrap").is_file(),
    reason="bwrap is required for the live exec_cell trace check",
)
def test_trace_records_source_child_calls_output_size_and_state_projection(
    tmp_path: Path,
):
    (tmp_path / "sample.txt").write_text("needle\nsecond\n")
    source = (
        'print(read("sample.txt", limit=1))\n'
        'print(grep("needle", "sample.txt"))'
    )
    call = ToolCall("cell-1", "exec_cell", {"source": source})
    client = _client(
        _turn(calls=[call]),
        _turn(content="done", reason="stop"),
    )
    cfg = make_config(
        max_turns=2,
        duplicate_abort=10,
        tools_exec_cell_enabled=True,
        tools_exec_cell_timeout=10,
        sandbox_bash=True,
        sandbox_required=True,
        tools_unified_envelope_enabled=True,
        rumination_nudge_threshold=999,
    )
    trace_path = tmp_path / ".trace.jsonl"
    state_path = tmp_path / ".solver" / "state.json"
    with trace_path.open("a+", encoding="utf-8") as trace:
        session = Session(
            cfg,
            client,
            "system",
            "task",
            str(tmp_path),
            trace_file=trace,
            trace_path=trace_path,
            state_path=state_path,
        )
        with patch.object(
            Session, "_get_server_ctx", return_value=cfg.context_size
        ):
            outcome = session.run()

    assert outcome.done is True
    calls = [
        event for event in session._trace_events
        if event.get("event") == "tool_call"
    ]
    outer = next(event for event in calls if event["tool_name"] == "exec_cell")
    children = [
        event for event in calls
        if event.get("parent_tool_call_id") == "cell-1"
    ]
    assert outer["cell_source"] == source
    assert outer["inner_call_count"] == 2
    assert outer["combined_output_chars"] > 0
    assert outer["combined_output_bytes"] >= outer["combined_output_chars"]
    assert [event["tool_name"] for event in children] == ["read", "grep"]
    assert [event["cell_inner_index"] for event in children] == [1, 2]

    persisted_calls = [
        event
        for line in trace_path.read_text().splitlines()
        if (event := json.loads(line)).get("event") == "tool_call"
    ]
    persisted_outer = next(
        event for event in persisted_calls if event["tool_name"] == "exec_cell"
    )
    assert persisted_outer["cell_source"] == source
    assert persisted_outer["combined_output_chars"] == outer[
        "combined_output_chars"
    ]
    assert len([
        event for event in persisted_calls
        if event.get("parent_tool_call_id") == "cell-1"
    ]) == 2

    projected = json.loads(state_path.read_text())
    projected_children = [
        step for step in projected["trace"]
        if step.get("parent_tool_call_id") == "cell-1"
    ]
    assert [step["action"].split("(", 1)[0] for step in projected_children] == [
        "read", "grep",
    ]
    assert projected["trace"][-1]["action"].startswith("exec_cell(")
