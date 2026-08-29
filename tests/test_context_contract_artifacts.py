from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _config_helpers import make_config
from llm_solver import __version__ as harness_version
from llm_solver._shared.telemetry_paths import trace_path
from llm_solver.harness.context_contract import build_context_contract
from llm_solver.harness.context_strategies import (
    CompoundContext,
    CompoundSelectiveContext,
    HalfLifeContext,
    list_context_modes,
    resolve_context_mode,
)
from llm_solver.harness.loop import solve_task
from llm_solver.server.types import TurnResult, Usage


def _done_turn(prompt_tokens: int = 10) -> TurnResult:
    return TurnResult(
        content="Done!",
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=5),
    )


def test_build_context_contract_records_mode_order_and_budgets():
    cfg = make_config(
        compound_selective_trace_lines=7,
        compound_selective_unresolved_evidence_lines=3,
        compound_selective_recent_tool_results_chars=1234,
    )

    contract = build_context_contract(CompoundSelectiveContext, cfg)

    assert contract["version"] == 2
    assert contract["mode"] == "compound_selective"
    assert contract["section_order"] == [
        "task",
        "state",
        "gate_blocking",
        "trace",
        "evidence",
        "tool_results",
        "todos",
        "continuation_suffix",
    ]
    assert contract["state_source"] == ".solver/state.json"
    assert contract["source_type"] == "trace_state"
    assert contract["normal_prompt_sources"] == [
        ".solver/state.json",
        "in_memory_recent_tool_results",
        "live_workspace_files_from_state_trace_on_session_resume",
    ]
    assert contract["budgets"]["compound_selective_trace_lines"] == 7
    assert contract["budgets"]["compound_selective_recent_tool_results_chars"] == 1234
    assert contract["budgets"]["state_todos_char_budget"] == 2000
    assert contract["section_labels"]["todos"] == "=== Todos ==="


def test_halflife_contract_excludes_stateful_suffix():
    cfg = make_config(
        state_context_suffix="Continue. Progress tracked in .solver/state.json.",
    )

    contract = build_context_contract(HalfLifeContext, cfg)

    assert contract["suffix_present"] is False
    assert "continuation_suffix" not in contract["section_order"]
    assert "<state_context_suffix>" not in contract["section_labels"].values()
    assert "state_todos_char_budget" not in contract["budgets"]


def test_every_context_mode_declares_source_boundary():
    valid_source_types = {"append_log", "trace_state", "hybrid"}
    expected_hybrids = {"yconcise", "yslot"}

    modes = {name: resolve_context_mode(name).metadata for name in list_context_modes()}

    assert set(modes) == {
        "full",
        "compact",
        "concise",
        "slot",
        "yuj",
        "yconcise",
        "yslot",
        "stateful",
        "compound",
        "focused_compound",
        "compound_selective",
        "salience",
        "halflife",
    }
    assert {name for name, meta in modes.items() if meta.source_type == "hybrid"} == expected_hybrids
    for name, meta in modes.items():
        assert meta.source_type in valid_source_types, name
        assert meta.normal_prompt_sources
        assert "strategy-defined" not in meta.normal_prompt_sources


def test_solve_task_emits_context_contract_in_metrics_and_trace(tmp_path: Path):
    (tmp_path / "prompt.txt").write_text("fix bug")

    client = MagicMock()
    client.chat.return_value = _done_turn(prompt_tokens=42)
    client.build_assistant_message.return_value = {"role": "assistant", "content": "Done!"}
    cfg = make_config(max_turns=10, max_sessions=1)

    with patch("llm_solver.harness.loop._auto_commit"):
        solve_task(tmp_path, cfg, client, context_class=CompoundContext)

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["provenance"]["harness_version"] == harness_version
    contract = metrics["provenance"]["context_contract"]
    assert contract["mode"] == "compound"
    assert contract["section_order"][:4] == ["task", "state", "gate_blocking", "trace"]
    assert contract["budgets"]["solver_trace_lines"] == cfg.solver_trace_lines

    # runtime_envelope may appear before session_start.
    trace_lines = trace_path(tmp_path).read_text().splitlines()
    events = [json.loads(line) for line in trace_lines if line.strip()]
    session_start = next(e for e in events if e["event"] == "session_start")
    assert session_start["context_contract"]["mode"] == "compound"


def test_solve_task_creates_state_json_when_writer_enabled(tmp_path: Path):
    (tmp_path / "prompt.txt").write_text("fix bug")

    client = MagicMock()
    client.chat.return_value = _done_turn(prompt_tokens=42)
    client.build_assistant_message.return_value = {"role": "assistant", "content": "Done!"}
    cfg = make_config(max_turns=10, max_sessions=1, state_writer_enabled=True)

    with patch("llm_solver.harness.loop._auto_commit"):
        solve_task(tmp_path, cfg, client, context_class=CompoundContext)

    state_path = tmp_path / ".solver" / "state.json"
    assert state_path.is_file()
    state = json.loads(state_path.read_text())
    assert "state" in state
    assert "trace" in state
    assert "evidence" in state
