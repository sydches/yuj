"""Phase-order tuples line up with the registered guardrail names.

The four `*_DISPATCH_ORDER` tuples in state.py declare the canonical phase
ordering; build_guardrail_registry() registers them. If a future
refactor adds a new guardrail to one but not the other, this test
fails before any session runs.

Note: this does NOT verify loop.py's hand-coded call order matches
the tuple. That stronger property would require AST inspection;
left as a follow-up. This test catches the realistic registry-vs-
tuple drift.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from llm_solver.harness._guardrails.state import (
    GUARDRAIL_SPECS,
    OBSERVER_ORDER,
    TOOL_POST_DISPATCH_ORDER,
    TOOL_PRE_DISPATCH_ORDER,
    TURN_PRE_DISPATCH_ORDER,
    guardrail_order_for_phase,
)
from llm_solver.harness.guardrails import build_guardrail_registry


def test_turn_pre_tuple_matches_registered_keys():
    reg = build_guardrail_registry()
    assert set(reg.turn_pre_dispatch.keys()) == set(TURN_PRE_DISPATCH_ORDER)


def test_tool_pre_tuple_matches_registered_keys():
    reg = build_guardrail_registry()
    assert set(reg.tool_pre_dispatch.keys()) == set(TOOL_PRE_DISPATCH_ORDER)


def test_tool_post_tuple_matches_registered_keys():
    reg = build_guardrail_registry()
    assert set(reg.tool_post_dispatch.keys()) == set(TOOL_POST_DISPATCH_ORDER)


def test_observer_tuple_matches_registered_keys():
    reg = build_guardrail_registry()
    assert set(reg.observers.keys()) == set(OBSERVER_ORDER)


def test_phase_orders_come_from_guardrail_specs():
    assert TURN_PRE_DISPATCH_ORDER == guardrail_order_for_phase("turn_pre_dispatch")
    assert TOOL_PRE_DISPATCH_ORDER == guardrail_order_for_phase("tool_pre_dispatch")
    assert TOOL_POST_DISPATCH_ORDER == guardrail_order_for_phase("tool_post_dispatch")
    assert OBSERVER_ORDER == guardrail_order_for_phase("observers")
    assert [spec.name for spec in GUARDRAIL_SPECS].count("done_guard") == 1


def _subscript_base_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _literal_subscript_names(path: Path, base_name: str) -> tuple[str, ...]:
    tree = ast.parse(path.read_text())
    found: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if _subscript_base_name(node.value) != base_name:
            continue
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            found.append((
                getattr(node, "lineno", 0),
                getattr(node, "col_offset", 0),
                node.slice.value,
            ))
    return tuple(value for _, _, value in sorted(found))


def test_run_loop_guardrail_call_order_matches_specs():
    run_step = (
        PROJECT_ROOT
        / "scripts/llm_solver/harness/_loop/run_step.py"
    )
    dispatch_tool_call = (
        PROJECT_ROOT
        / "scripts/llm_solver/harness/_loop/_dispatch_tool_call.py"
    )

    assert _literal_subscript_names(run_step, "turn_pre") == TURN_PRE_DISPATCH_ORDER
    assert _literal_subscript_names(dispatch_tool_call, "tool_pre") == TOOL_PRE_DISPATCH_ORDER
    assert _literal_subscript_names(dispatch_tool_call, "tool_post") == TOOL_POST_DISPATCH_ORDER
    assert _literal_subscript_names(dispatch_tool_call, "observers") == OBSERVER_ORDER
