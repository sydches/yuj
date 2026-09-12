"""Check registry membership, declared call sites and actual dispatch order.

Structural checks keep guardrail call sites aligned with the phase specs.
Session checks exercise real command effects before ordered policy decisions,
including a partial write whose error ends the turn. The model is scripted;
the tool dispatcher and registered guards execute normally.
"""
from __future__ import annotations

import ast
from dataclasses import replace
import io
import json
import shlex
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from llm_solver.harness._guardrails.state import (
    GUARDRAIL_SPECS,
    OBSERVER_ORDER,
    TOOL_POST_DISPATCH_ORDER,
    TOOL_PRE_DISPATCH_ORDER,
    TURN_PRE_DISPATCH_ORDER,
    TURN_POST_DISPATCH_ORDER,
    guardrail_order_for_phase,
)
from llm_solver.harness.guardrails import build_guardrail_registry
from llm_solver.harness._guardrails.state import Decision
from llm_solver.harness.loop import Session
from llm_solver.server.types import ToolCall, TurnResult, Usage
from _config_helpers import make_config


def test_turn_pre_tuple_matches_registered_keys():
    reg = build_guardrail_registry()
    assert set(reg.turn_pre_dispatch.keys()) == set(TURN_PRE_DISPATCH_ORDER)


def test_turn_post_tuple_matches_registered_keys():
    reg = build_guardrail_registry()
    assert set(reg.turn_post_dispatch) == set(TURN_POST_DISPATCH_ORDER)


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
    assert _literal_subscript_names(run_step, "turn_post") == TURN_POST_DISPATCH_ORDER
    assert _literal_subscript_names(dispatch_tool_call, "tool_pre") == TOOL_PRE_DISPATCH_ORDER
    assert _literal_subscript_names(dispatch_tool_call, "tool_post") == TOOL_POST_DISPATCH_ORDER
    assert _literal_subscript_names(dispatch_tool_call, "observers") == OBSERVER_ORDER


def _run_observed_command(tmp_path, *, mutation, failure=False, abort=False,
                          rumination_warning=False):
    target = tmp_path / "module.py"
    target.write_text("before")
    program = (
        "from pathlib import Path; Path('module.py').write_text('after')"
        if mutation else "print('no change')"
    )
    if failure:
        program += "; raise SystemExit(1)"
    command = shlex.join([sys.executable, "-c", program])
    cfg = make_config(
        max_turns=1, sandbox_bash=False, auto_commit=False,
        turn_snapshots_enabled=False, error_abort_threshold=1 if abort else 0,
        loop_detect_enabled=False, duplicate_guard_enabled=False,
    )
    calls, error_states = [], []
    registry = build_guardrail_registry()

    def wrap(name, original):
        def observe(state, config, *args, **kwargs):
            calls.append(name)
            if name == "error_ladder":
                error_states.append({
                    "has_mutated": state.has_mutated,
                    "mutation_count": state.mutation_count,
                    "verified": state.verified_since_mutation,
                    "formal_verified": state.formal_verification_passed_since_mutation,
                    "paths": state.post_mutation_source_paths,
                })
            decision = original(state, config, *args, **kwargs)
            if name == "rumination_ladder" and rumination_warning:
                return Decision.warn("ORDERED_GUARD_WARNING", reason="phase_fixture")
            return decision
        return observe

    registry = replace(
        registry,
        tool_post_dispatch={name: wrap(name, fn)
                            for name, fn in registry.tool_post_dispatch.items()},
        observers={name: wrap(name, fn) for name, fn in registry.observers.items()},
    )
    client = MagicMock()
    client.chat.return_value = TurnResult(
        content=None,
        tool_calls=[ToolCall(id="observed-command", name="bash", arguments={"cmd": command})],
        finish_reason="tool_calls", usage=Usage(prompt_tokens=10, completion_tokens=5),
    )
    client.build_assistant_message.return_value = {"role": "assistant", "content": None}
    trace = io.StringIO()
    session = Session(cfg, client, "Run the command.", "Exercise dispatch order.",
                      str(tmp_path), trace_file=trace, guardrail_registry=registry)
    session._guards.verified_since_mutation = True
    session._guards.formal_verification_passed_since_mutation = True
    result = session.run()
    events = [json.loads(line) for line in trace.getvalue().splitlines()]
    return session, result, calls, error_states, events


@pytest.mark.parametrize("mutation,failure,abort", [
    (False, False, False),
    (True, False, False),
    (True, True, False),
    (True, True, True),
], ids=["unchanged", "mutation", "partial-write", "partial-write-abort"])
def test_observed_effects_precede_ordered_guard_decisions(tmp_path, mutation, failure, abort):
    session, result, calls, error_states, events = _run_observed_command(
        tmp_path, mutation=mutation, failure=failure, abort=abort,
    )
    expected = ("error_ladder",) if abort else TOOL_POST_DISPATCH_ORDER + OBSERVER_ORDER
    assert tuple(calls) == expected
    assert len(error_states) == 1
    at_error = error_states[0]
    assert at_error["has_mutated"] is mutation
    assert at_error["mutation_count"] == int(mutation)
    if mutation:
        assert not at_error["verified"] and not at_error["formal_verified"]
        assert at_error["paths"] == ("module.py",)
    assert session._guards.mutation_count == int(mutation)
    assert (tmp_path / "module.py").read_text() == ("after" if mutation else "before")
    event = next(row for row in events if row.get("event") == "tool_call"
                 and row.get("tool_call_id") == "observed-command")
    assert event["exit_status"] == int(failure)
    assert event["file_changes"]["changed_paths"] == (["module.py"] if mutation else [])
    assert result.finish_reason == ("error_abort" if abort else "max_turns")


def test_mutation_does_not_discard_ordered_guard_warning(tmp_path):
    session, _, calls, _, events = _run_observed_command(
        tmp_path, mutation=True, rumination_warning=True,
    )
    assert tuple(calls) == TOOL_POST_DISPATCH_ORDER + OBSERVER_ORDER
    assert session._guards.mutation_count == 1
    assert any(item.text == "ORDERED_GUARD_WARNING"
               for item in session._pending_user_turn_injections)
    assert any(row.get("event") == "tool_call" for row in events)
