"""Tests for scripts.llm_solver.harness.compound_context.

Validates the CompoundContext rendering: reasoning-in-trace with
per-turn deduplication, content-blind trace stubs, and the 2-message
prompt shape.

The harness is content-blind: no pytest parsing, no verdict extraction,
no evidence ordering. Task-format parsing belongs in the analysis layer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.llm_solver.harness.context_strategies import CompoundContext


def _make_context(tmp_path: Path, prompt: str = "Solve the task.") -> CompoundContext:
    ctx = CompoundContext(
        cwd=str(tmp_path),
        original_prompt=prompt,
        trace_lines=50,
        evidence_lines=30,
        inference_lines=20,
        recent_tool_results_chars=30000,
        trace_stub_chars=200,
        min_turns=2,
        suffix="Continue working. Your progress is tracked in .solver/state.json — read it to see what you've already done.",
    )
    ctx.add_system("You are a solver.")
    # Minimum turns so SolverStateContext switches from raw-messages to
    # state-based rendering.
    ctx._turn_count = 5
    return ctx


def _write_state(tmp_path: Path, state: dict) -> None:
    solver_dir = tmp_path / ".solver"
    solver_dir.mkdir(parents=True, exist_ok=True)
    (solver_dir / "state.json").write_text(json.dumps(state))


class TestFormatTraceReasoning:
    def test_trace_with_reasoning_renders_inline(self, tmp_path: Path):
        _write_state(tmp_path, {
            "state": {"current_attempt": "read(path='a.py')",
                      "last_verify": "", "next_action": ""},
            "trace": [
                {"step": 1, "session": 1, "turn": 0,
                 "reasoning": "I need to inspect the webhook file",
                 "action": "read(path='a.py')", "result": "1: x\n2: y", "next": ""},
            ],
            "gates": [],
            "evidence": [],
            "inference": [],
        })
        ctx = _make_context(tmp_path)
        msgs = ctx.get_messages()
        assert len(msgs) == 2
        user_text = msgs[1]["content"]
        assert "=== Trace ===" in user_text
        assert "I need to inspect the webhook file" in user_text
        assert "read(path='a.py')" in user_text

    def test_multi_tool_turn_reasoning_rendered_once(self, tmp_path: Path):
        # Same turn, same reasoning, two entries → reasoning line
        # appears exactly once above both actions.
        _write_state(tmp_path, {
            "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
            "trace": [
                {"step": 1, "session": 1, "turn": 3,
                 "reasoning": "Check two files at once",
                 "action": "read(path='a.py')", "result": "...", "next": ""},
                {"step": 2, "session": 1, "turn": 3,
                 "reasoning": "Check two files at once",
                 "action": "read(path='b.py')", "result": "...", "next": ""},
            ],
            "gates": [],
            "evidence": [],
            "inference": [],
        })
        ctx = _make_context(tmp_path)
        user_text = ctx.get_messages()[1]["content"]
        assert user_text.count("Check two files at once") == 1
        assert "read(path='a.py')" in user_text
        assert "read(path='b.py')" in user_text

    def test_different_reasoning_per_turn_both_rendered(self, tmp_path: Path):
        _write_state(tmp_path, {
            "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
            "trace": [
                {"step": 1, "session": 1, "turn": 0,
                 "reasoning": "First I read",
                 "action": "read(path='x')", "result": "", "next": ""},
                {"step": 2, "session": 1, "turn": 1,
                 "reasoning": "Then I edit",
                 "action": "edit(path='x')", "result": "OK", "next": ""},
            ],
            "gates": [],
            "evidence": [],
            "inference": [],
        })
        ctx = _make_context(tmp_path)
        user_text = ctx.get_messages()[1]["content"]
        assert "First I read" in user_text
        assert "Then I edit" in user_text

    def test_empty_reasoning_omits_reasoning_line(self, tmp_path: Path):
        _write_state(tmp_path, {
            "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
            "trace": [
                {"step": 1, "session": 1, "turn": 0, "reasoning": "",
                 "action": "bash(cmd='ls')", "result": "a\nb", "next": ""},
            ],
            "gates": [],
            "evidence": [],
            "inference": [],
        })
        ctx = _make_context(tmp_path)
        user_text = ctx.get_messages()[1]["content"]
        # Action line still rendered, no bracketed reasoning header.
        assert "bash(cmd='ls')" in user_text

    def test_long_result_is_stubbed_not_full(self, tmp_path: Path):
        long_result = "x" * 5000  # exceeds trace_stub_chars (200)
        _write_state(tmp_path, {
            "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
            "trace": [
                {"step": 1, "session": 1, "turn": 0, "reasoning": "",
                 "action": "read(path='big')", "result": long_result, "next": ""},
            ],
            "gates": [],
            "evidence": [],
            "inference": [],
        })
        ctx = _make_context(tmp_path)
        user_text = ctx.get_messages()[1]["content"]
        # Full 5000-char result must not appear — only a stub ending in "..."
        assert long_result not in user_text
        assert "..." in user_text


class TestEvidenceFailFirstAndGatePointer:
    """Evidence rendering uses only the structured `verdict` field, which
    the projection filled content-blindly from harness exit-code markers.
    No task-format detection anywhere in the read path.
    """

    def _fail_ev(self, step: int, sentinel: str) -> dict:
        return {
            "step": step,
            "action": f"bash(cmd='./run {step}')",
            "result": f"{sentinel}\n[exit code: 1]",
            "verdict": "FAIL",
            "gate_blocked": False,
        }

    def _pass_ev(self, step: int, sentinel: str) -> dict:
        return {
            "step": step,
            "action": f"bash(cmd='./run {step}')",
            "result": sentinel,
            "verdict": "OK",
            "gate_blocked": False,
        }

    def test_most_recent_failure_is_the_gate_pointer(self, tmp_path: Path):
        _write_state(tmp_path, {
            "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
            "trace": [],
            "gates": [],
            "evidence": [
                self._pass_ev(1, "SENTINEL_PASS_ONE"),
                self._fail_ev(2, "SENTINEL_FAIL_EARLY"),
                self._fail_ev(3, "SENTINEL_FAIL_LATE"),
            ],
            "inference": [],
        })
        ctx = _make_context(tmp_path)
        user_text = ctx.get_messages()[1]["content"]
        assert "=== Gate (blocking) ===" in user_text
        # The gate points at the most recent failure, not the first.
        # Slice from the gate header to the next section header so we
        # inspect only the gate body, not the whole evidence block.
        gate_idx = user_text.index("=== Gate (blocking) ===")
        next_section = user_text.index("===", gate_idx + len("=== Gate (blocking) ==="))
        gate_section = user_text[gate_idx:next_section]
        assert "SENTINEL_FAIL_LATE" in gate_section
        assert "SENTINEL_FAIL_EARLY" not in gate_section

    def test_no_gate_section_when_all_evidence_passes(self, tmp_path: Path):
        _write_state(tmp_path, {
            "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
            "trace": [],
            "gates": [],
            "evidence": [
                self._pass_ev(1, "SENTINEL_PASS_A"),
                self._pass_ev(2, "SENTINEL_PASS_B"),
            ],
            "inference": [],
        })
        ctx = _make_context(tmp_path)
        user_text = ctx.get_messages()[1]["content"]
        assert "=== Gate (blocking) ===" not in user_text

    def test_fails_render_before_passes(self, tmp_path: Path):
        _write_state(tmp_path, {
            "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
            "trace": [],
            "gates": [],
            "evidence": [
                self._pass_ev(1, "SENTINEL_PASS_ONE"),
                self._fail_ev(2, "SENTINEL_FAIL_ONE"),
                self._pass_ev(3, "SENTINEL_PASS_TWO"),
                self._fail_ev(4, "SENTINEL_FAIL_TWO"),
            ],
            "inference": [],
        })
        ctx = _make_context(tmp_path)
        user_text = ctx.get_messages()[1]["content"]
        assert "=== Evidence ===" in user_text
        ev_start = user_text.index("=== Evidence ===")
        ev_block = user_text[ev_start:]
        # Unresolved section appears before resolved section.
        assert ev_block.index("-- unresolved --") < ev_block.index("-- resolved --")
        unresolved = ev_block[ev_block.index("-- unresolved --"):ev_block.index("-- resolved --")]
        resolved = ev_block[ev_block.index("-- resolved --"):]
        assert "SENTINEL_FAIL_ONE" in unresolved
        assert "SENTINEL_FAIL_TWO" in unresolved
        assert "SENTINEL_PASS_ONE" in resolved
        assert "SENTINEL_PASS_TWO" in resolved


class TestMessageShape:
    def test_prompt_is_two_messages_after_min_turns(self, tmp_path: Path):
        _write_state(tmp_path, {
            "state": {"current_attempt": "read(a)", "last_verify": "", "next_action": ""},
            "trace": [
                {"step": 1, "session": 1, "turn": 0, "reasoning": "r",
                 "action": "read(a)", "result": "", "next": ""},
            ],
            "gates": [],
            "evidence": [],
            "inference": [],
        })
        ctx = _make_context(tmp_path)
        msgs = ctx.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_task_prompt_always_included(self, tmp_path: Path):
        _write_state(tmp_path, {
            "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
            "trace": [], "gates": [], "evidence": [], "inference": [],
        })
        ctx = _make_context(tmp_path, prompt="Specific task description here.")
        user_text = ctx.get_messages()[1]["content"]
        assert "Task: Specific task description here." in user_text

    def test_continuation_suffix_references_state_json(self, tmp_path: Path):
        _write_state(tmp_path, {
            "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
            "trace": [], "gates": [], "evidence": [], "inference": [],
        })
        ctx = _make_context(tmp_path)
        user_text = ctx.get_messages()[1]["content"]
        # The suffix (injected from cfg.state_context_suffix in production)
        # references state.json as the memory of record.
        assert ".solver/state.json" in user_text


class TestReasoningNeverDuplicatesToolResults:
    """Guard against the trace-vs-rolling-window duplication bug."""

    def test_rolling_window_holds_only_raw_tool_output(self, tmp_path: Path):
        # Put a reasoning string into state.json trace and an unrelated
        # tool result into the rolling window. The reasoning text must
        # NOT appear inside the rolling window section.
        _write_state(tmp_path, {
            "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
            "trace": [
                {"step": 1, "session": 1, "turn": 0,
                 "reasoning": "UNIQUE_REASONING_STRING",
                 "action": "read(path='a')", "result": "short", "next": ""},
            ],
            "gates": [], "evidence": [], "inference": [],
        })
        ctx = _make_context(tmp_path)
        ctx.add_tool_result("tc_1", "tool_output_text_only")
        user_text = ctx.get_messages()[1]["content"]
        # Reasoning appears once (in trace section), tool output appears
        # once (in rolling window). Neither crosses over.
        assert user_text.count("UNIQUE_REASONING_STRING") == 1
        assert "tool_output_text_only" in user_text
        # The rolling window is after the trace section; reasoning should
        # not leak into it.
        if "=== Tool result" in user_text:
            window_start = user_text.rindex("===")  # last section header
            window = user_text[window_start:]
            assert "UNIQUE_REASONING_STRING" not in window
