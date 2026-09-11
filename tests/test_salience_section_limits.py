"""Pressure must not enlarge selected section limits or erase required state."""
import pytest
from tests._config_helpers import make_config

from scripts.llm_solver.harness.context_strategies.salience_context import SalienceContext


def _context(tmp_path, *, trace=2, unresolved=1, resolved=0, tool_chars=1000):
    ctx = SalienceContext(
        cwd=str(tmp_path), original_prompt="Inspect the project.",
        trace_lines=trace, evidence_lines=unresolved, inference_lines=2,
        recent_tool_results_chars=tool_chars, trace_stub_chars=100,
        min_turns=0, suffix="", selective_trace_lines=trace,
        selective_unresolved_evidence_lines=unresolved,
        selective_resolved_evidence_lines=resolved,
        selective_recent_tool_results_chars=tool_chars,
        token_estimator=lambda messages: sum(len(m["content"]) for m in messages) // 4,
    )
    ctx.add_system("Inspect permitted sources.")
    cfg = make_config(context_size=1024, max_tokens=100)
    ctx.set_projection_request(lambda: (None, cfg, []))
    return ctx


@pytest.mark.parametrize("turn", [0, 40, 80, 120, 160, 200])
@pytest.mark.parametrize("tool_chars", [60, 1000, 2500])
def test_pressure_does_not_raise_small_selected_limits(tmp_path, turn, tool_chars):
    ctx = _context(tmp_path, tool_chars=tool_chars)
    ctx._turn_count = turn
    assert ctx._pressure_trace_limit() <= 2
    assert ctx._pressure_unresolved_evidence_limit() <= 1
    assert ctx._pressure_resolved_evidence_limit() == 0
    assert ctx._pressure_tool_chars() <= tool_chars


def test_oversized_state_reduction_never_grows_a_section(tmp_path, monkeypatch):
    ctx = _context(tmp_path)
    observed = []
    build = ctx._build_parts

    def record(**kwargs):
        observed.append(tuple(kwargs[key] for key in (
            "trace_limit", "unresolved_limit", "resolved_limit", "tool_chars",
        )))
        return build(**kwargs)

    monkeypatch.setattr(ctx, "_build_parts", record)
    retained_state = "required-state-" * 20000
    messages = ctx._bounded_projection(
        state_text=retained_state, suffix_text="", trace=[], evidence=[],
    )
    assert retained_state in messages[-1]["content"]
    assert observed[0] == (2, 1, 0, 1000)
    assert all(all(a <= b for a, b in zip(current, previous))
               for previous, current in zip(observed, observed[1:]))
    assert ctx.projection_pressure["overflow"] is True
    assert ctx._token_estimator(messages) > ctx.projection_pressure["prompt_allowance"]


def test_large_sections_reduce_monotonically_without_growing_resolved_zero(tmp_path, monkeypatch):
    ctx = _context(tmp_path, trace=50, unresolved=30, tool_chars=30000)
    observed = []
    build = ctx._build_parts

    def record(**kwargs):
        observed.append(tuple(kwargs[key] for key in (
            "trace_limit", "unresolved_limit", "resolved_limit", "tool_chars",
        )))
        return build(**kwargs)

    monkeypatch.setattr(ctx, "_build_parts", record)
    ctx._bounded_projection(state_text="state" * 50000, suffix_text="", trace=[], evidence=[])
    assert len(observed) > 1
    assert all(row[2] == 0 for row in observed)
    assert all(all(a <= b for a, b in zip(current, previous))
               for previous, current in zip(observed, observed[1:]))


def test_zero_unresolved_limit_does_not_select_all_evidence(tmp_path):
    ctx = _context(tmp_path)
    failures, passes = ctx._split_evidence_pressure(
        [{"step": 1, "action": "check", "verdict": "FAIL", "result": "incomplete"}],
        unresolved_limit=0, resolved_limit=0,
    )
    assert failures == [] and passes == []
