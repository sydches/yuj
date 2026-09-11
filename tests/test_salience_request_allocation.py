"""Salience consumes the current request inputs; no model generation is used."""
import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests._config_helpers import make_config
from tests.test_salience_section_limits import _context
from scripts.llm_solver.context_allocation import allocate_context
from scripts.llm_solver.harness.loop import Session


class FixtureCounter:
    def __init__(self):
        self.calls = []
        self.last = {}

    def count(self, messages, tools=None):
        self.calls.append((messages, tools))
        count = sum(len(m["content"]) for m in messages) // 4 + len(json.dumps(tools))
        self.last = {"count_basis": "synthetic_fixture", "prompt_tokens": count}
        return count


def save_state(tmp_path, state):
    directory = tmp_path / ".solver"
    directory.mkdir(exist_ok=True)
    (directory / "state.json").write_text(json.dumps({"state": state}))


@pytest.mark.parametrize("declared,observed", [(1024, 8192), (8192, 1024), (65536, 131072)])
def test_current_allocation_counts_tools_and_preserves_oversized_content(tmp_path, declared, observed):
    ctx = _context(tmp_path)
    counter = FixtureCounter()
    cfg = allocate_context(make_config(context_size=declared, max_tokens=100), observed=observed)
    tools = [{"type": "function", "function": {"name": "fixture", "description": "x" * 300}}]
    ctx.set_projection_request(lambda: (counter, cfg, tools))
    state = "required state " * 16000
    messages = ctx._bounded_projection(state_text=state, suffix_text="required suffix", trace=[], evidence=[])
    allowance = min(int(cfg.context_size * cfg.context_fill_ratio), cfg.context_size - cfg.max_tokens)
    assert ctx.projection_pressure["prompt_allowance"] == allowance
    assert ctx.projection_pressure["overflow"] is (counter.last["prompt_tokens"] > allowance)
    assert ctx.projection_pressure["count_basis"] == "synthetic_fixture"
    assert all(call[1] == tools for call in counter.calls)
    assert state in messages[1]["content"] and "required suffix" in messages[1]["content"]
    if declared == 65536:
        assert counter.last["prompt_tokens"] > 46000
        assert len(counter.calls) == 1  # No independent historical target.


def test_cached_projection_refreshes_after_model_or_tool_change(tmp_path):
    save_state(tmp_path, "saved state")
    ctx = _context(tmp_path)
    active = SimpleNamespace(counter=FixtureCounter(), cfg=make_config(context_size=8192, max_tokens=100), tools=[])
    ctx.set_projection_request(lambda: (active.counter, active.cfg, active.tools))
    ctx.get_messages()
    assert not ctx.projection_pressure["overflow"]
    active.counter = FixtureCounter()
    active.cfg = allocate_context(make_config(model="replacement", context_size=1024, max_tokens=100), observed=512)
    active.tools = [{"description": "x" * 600}]
    ctx.get_messages()
    assert active.counter.calls and active.counter.calls[-1][1] == active.tools
    assert ctx.projection_pressure["prompt_allowance"] < 512
    assert ctx.projection_pressure["overflow"]


def test_pressure_rendering_keeps_original_tool_window(tmp_path):
    ctx = _context(tmp_path)
    ctx.add_tool_result("older", "older evidence" * 100)
    ctx.add_tool_result("newer", "newer evidence" * 100)
    original = list(ctx._recent_tool_results)
    ctx._format_tool_results_budget(10)
    assert list(ctx._recent_tool_results) == original


def test_session_reports_unresolved_projection_without_generation(tmp_path):
    save_state(tmp_path, "retained state" * 1000)
    ctx = _context(tmp_path)
    cfg = make_config(context_size=1024, max_tokens=100, max_turns=1, sandbox_bash=False)
    client = MagicMock()
    client.query_server_context.return_value = 8192
    counter = FixtureCounter()
    trace = io.StringIO()
    session = Session(cfg, client, "system", "task", str(tmp_path),
                      context_manager=ctx, local_tokenizer=counter, trace_file=trace)
    result = session.run()
    assert result.finish_reason == "context_full"
    client.chat.assert_not_called()
    event = next(row for row in map(json.loads, trace.getvalue().splitlines())
                 if row["event"] == "context_projection_pressure")
    assert event["generation_sent"] is False and event["overflow"] is True
    assert "retained state" in (tmp_path / ".solver/state.json").read_text()
