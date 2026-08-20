from __future__ import annotations

import json
from pathlib import Path

from scripts.llm_solver.harness.context import FullTranscript
from scripts.llm_solver.harness.context_strategies import (
    CompoundContext,
    CompoundSelectiveContext,
    FocusedCompoundContext,
    HalfLifeContext,
    SalienceContext,
    SolverStateContext,
    resolve_context_class,
)


SUFFIX = "SUFFIX_SENTINEL"
TOOL_RESULT = "TOOL_RESULT_SENTINEL"


def _write_state(tmp_path: Path) -> None:
    solver_dir = tmp_path / ".solver"
    solver_dir.mkdir(parents=True, exist_ok=True)
    (solver_dir / "state.json").write_text(json.dumps({
        "state": {
            "current_attempt": "STATE_SENTINEL",
            "last_verify": "VERIFY_SENTINEL",
            "next_action": "NEXT_SENTINEL",
        },
        "trace": [
            {
                "step": 1,
                "session": 1,
                "turn": 0,
                "reasoning": "TRACE_REASONING_SENTINEL",
                "action": "read(path='src/example.py')",
                "result": "TRACE_RESULT_SENTINEL",
                "next": "TRACE_NEXT_SENTINEL",
            },
        ],
        "gates": [
            {
                "name": "CONSTRAINT_SENTINEL",
                "status": "active",
                "notes": "read tests before done",
            },
        ],
        "evidence": [
            {
                "step": 2,
                "action": "bash(cmd='pytest tests/test_example.py')",
                "result": "FAIL_SENTINEL",
                "verdict": "FAIL",
                "gate_blocked": False,
            },
        ],
        "inference": ["HYPOTHESIS_SENTINEL"],
    }))


def _assert_order(text: str, markers: list[str]) -> None:
    positions = []
    for marker in markers:
        assert marker in text, f"missing marker: {marker}"
        positions.append(text.index(marker))
    assert positions == sorted(positions), markers


def _prime_context(ctx) -> str:
    ctx.add_system("SYSTEM_SENTINEL")
    ctx._turn_count = 5
    ctx.add_tool_result(
        "call-1",
        TOOL_RESULT,
        tool_name="bash",
        cmd_signature='{"cmd":"pytest tests/test_example.py"}',
    )
    messages = ctx.get_messages()
    assert [m["role"] for m in messages] == ["system", "user"]
    return messages[1]["content"]


def test_full_transcript_preserves_causal_message_order():
    ctx = FullTranscript()
    ctx.add_system("SYSTEM_SENTINEL")
    ctx.add_user("USER_SENTINEL")
    ctx.add_assistant({
        "role": "assistant",
        "content": "ASSISTANT_SENTINEL",
        "tool_calls": [{"id": "call-1", "type": "function"}],
    })
    ctx.add_tool_result("call-1", TOOL_RESULT)

    messages = ctx.get_messages()
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert messages[-1]["tool_call_id"] == "call-1"


def test_halflife_mode_resolves_active_context_module():
    cls = resolve_context_class("halflife")
    assert cls is HalfLifeContext
    assert cls.__module__.endswith(".halflife_context")


def test_stateful_context_section_order_is_contractual(tmp_path: Path):
    _write_state(tmp_path)
    ctx = SolverStateContext(
        cwd=str(tmp_path),
        original_prompt="TASK_SENTINEL",
        trace_lines=10,
        evidence_lines=10,
        inference_lines=10,
        recent_tool_results_chars=1000,
        trace_stub_chars=200,
        min_turns=2,
        suffix=SUFFIX,
    )

    user_text = _prime_context(ctx)
    _assert_order(user_text, [
        "Task: TASK_SENTINEL",
        "=== Current state ===",
        "=== Progress trace (recent) ===",
        "=== Evidence ===",
        "=== Tool result from your last action ===",
        SUFFIX,
    ])
    assert user_text.endswith(SUFFIX)


def test_compound_context_section_order_is_contractual(tmp_path: Path):
    _write_state(tmp_path)
    ctx = CompoundContext(
        cwd=str(tmp_path),
        original_prompt="TASK_SENTINEL",
        trace_lines=10,
        evidence_lines=10,
        inference_lines=10,
        recent_tool_results_chars=1000,
        trace_stub_chars=200,
        min_turns=2,
        suffix=SUFFIX,
    )

    user_text = _prime_context(ctx)
    _assert_order(user_text, [
        "Task: TASK_SENTINEL",
        "=== State ===",
        "=== Gate (blocking) ===",
        "=== Trace ===",
        "=== Evidence ===",
        "=== Tool result from your last action ===",
        SUFFIX,
    ])
    assert user_text.endswith(SUFFIX)


def test_focused_compound_context_keeps_compound_order(tmp_path: Path):
    _write_state(tmp_path)
    ctx = FocusedCompoundContext(
        cwd=str(tmp_path),
        original_prompt="TASK_SENTINEL",
        trace_lines=10,
        evidence_lines=10,
        inference_lines=10,
        recent_tool_results_chars=1000,
        trace_stub_chars=200,
        min_turns=2,
        suffix=SUFFIX,
        focused_trace_lines=10,
        focused_evidence_lines=10,
        focused_recent_tool_results_chars=1000,
    )

    user_text = _prime_context(ctx)
    _assert_order(user_text, [
        "Task: TASK_SENTINEL",
        "=== State ===",
        "=== Gate (blocking) ===",
        "=== Trace ===",
        "=== Evidence ===",
        "=== Tool result from your last action ===",
        SUFFIX,
    ])


def test_compound_selective_context_keeps_compound_order(tmp_path: Path):
    _write_state(tmp_path)
    ctx = CompoundSelectiveContext(
        cwd=str(tmp_path),
        original_prompt="TASK_SENTINEL",
        trace_lines=10,
        evidence_lines=10,
        inference_lines=10,
        recent_tool_results_chars=1000,
        trace_stub_chars=200,
        min_turns=2,
        suffix=SUFFIX,
        selective_trace_lines=10,
        selective_unresolved_evidence_lines=10,
        selective_resolved_evidence_lines=10,
        selective_recent_tool_results_chars=1000,
    )

    user_text = _prime_context(ctx)
    _assert_order(user_text, [
        "Task: TASK_SENTINEL",
        "=== State ===",
        "=== Gate (blocking) ===",
        "=== Trace ===",
        "=== Evidence ===",
        "=== Tool result from your last action ===",
        SUFFIX,
    ])


def test_salience_context_keeps_state_evidence_order(tmp_path: Path):
    _write_state(tmp_path)
    ctx = SalienceContext(
        cwd=str(tmp_path),
        original_prompt="TASK_SENTINEL",
        trace_lines=10,
        evidence_lines=10,
        inference_lines=10,
        recent_tool_results_chars=1000,
        trace_stub_chars=200,
        min_turns=2,
        suffix=SUFFIX,
        selective_trace_lines=10,
        selective_unresolved_evidence_lines=10,
        selective_resolved_evidence_lines=10,
        selective_recent_tool_results_chars=1000,
    )

    user_text = _prime_context(ctx)
    _assert_order(user_text, [
        "Task: TASK_SENTINEL",
        "=== State ===",
        "=== Evidence ===",
        "=== Tool result from your last action ===",
        SUFFIX,
    ])


def test_salience_mode_resolves_active_context_module():
    cls = resolve_context_class("salience")
    assert cls is SalienceContext
    assert cls.__module__.endswith(".salience_context")
