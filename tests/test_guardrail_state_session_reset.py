"""Cross-session reset of guardrail state.

init_guardrail_state(cfg) is called fresh for every session, so fields reset to dataclass defaults
implicitly. Pin the contract so a future field that defaults to
non-zero doesn't silently leak state across sessions.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from _config_helpers import make_config
from llm_solver.harness._guardrails.state import (
    GuardrailState,
    init_guardrail_state,
)


def test_default_construct_has_usable_deque():
    """A bare GuardrailState() must produce a deque
    that retains appends, not maxlen=0.
    """
    s = GuardrailState()
    s.recent_calls.append("x")
    assert "x" in s.recent_calls


def test_session_2_state_starts_zero():
    cfg = make_config(duplicate_abort=20)

    # Simulate session 1: init + accumulate counters.
    s1 = init_guardrail_state(cfg)
    s1.done_blocked_count = 7
    s1.intent_block_count = 3
    s1.gate_block_count = 5
    s1.contract_block_count = 2
    s1.same_class_error_count = 4
    s1.commit_violation_count = 1
    s1.mutation_repeat_count = 6
    s1.verify_repeat_count = 2
    s1.test_file_reads.add("tests/test_x.py")
    s1.recent_calls.append(("bash", "ls"))

    # Session 2: fresh state.
    s2 = init_guardrail_state(cfg)

    # Every counter we harvest cross-session must start at 0 in the
    # new session.
    assert s2.done_blocked_count == 0
    assert s2.intent_block_count == 0
    assert s2.gate_block_count == 0
    assert s2.contract_block_count == 0
    assert s2.same_class_error_count == 0
    assert s2.commit_violation_count == 0
    assert s2.mutation_repeat_count == 0
    assert s2.verify_repeat_count == 0
    # Sets / deques also reset.
    assert s2.test_file_reads == set()
    assert len(s2.recent_calls) == 0


def test_init_with_zero_duplicate_abort_keeps_deque_usable():
    cfg = make_config(duplicate_abort=0)
    s = init_guardrail_state(cfg)
    s.recent_calls.append("a")
    assert "a" in s.recent_calls
