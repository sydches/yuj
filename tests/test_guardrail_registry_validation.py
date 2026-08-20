"""Registry build/validate catches typos and bad callables.

A mistyped override once added a never-called entry, and a `None` override
landed in the registry and exploded mid-turn at the first call.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pytest

from llm_solver.harness.guardrails import (
    build_guardrail_registry,
    validate_guardrail_registry,
)


def test_unknown_turn_pre_override_rejected():
    with pytest.raises(ValueError, match="Unknown turn_pre"):
        build_guardrail_registry(
            turn_pre_overrides={"intent_gate_typo": lambda *a, **k: None}
        )


def test_unknown_tool_pre_override_rejected():
    with pytest.raises(ValueError, match="Unknown tool_pre"):
        build_guardrail_registry(
            tool_pre_overrides={"contract_gat": lambda *a, **k: None}
        )


def test_unknown_observer_override_rejected():
    with pytest.raises(ValueError, match="Unknown observer"):
        build_guardrail_registry(
            observer_overrides={"observe_typo": lambda *a, **k: None}
        )


def test_known_override_accepted():
    # Sanity: legitimate replacement (correct key) is fine.
    def replacement(*a, **k):
        from llm_solver.harness._guardrails.state import PASS
        return PASS

    reg = build_guardrail_registry(
        turn_pre_overrides={"intent_gate": replacement}
    )
    assert reg.turn_pre_dispatch["intent_gate"] is replacement


def test_validate_rejects_none_entry():
    """Manual construction with a None entry should be caught by
    validate_guardrail_registry — pre-fix this exploded mid-turn.
    """
    from llm_solver.harness._guardrails.state import GuardrailRegistry

    reg = build_guardrail_registry()
    # Mutate to inject None (simulates a future refactor's typo).
    reg.tool_pre_dispatch["done_guard"] = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="non-callable"):
        validate_guardrail_registry(reg)
