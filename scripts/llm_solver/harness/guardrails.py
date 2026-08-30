"""Guardrail framework — facade over the _guardrails/ package.

Pre-tool decisions: intent_gate, loop_detect, duplicate_guard, pre_mutation_gate,
done_guard, mutation_repeat_guard, contract_gate,
post_mutation_verification_gate, rumination_gate.
Post-tool ladders: error_ladder, test_read_ladder, rumination_ladder.
Observers: mark_bash_verified, observe_test_file_read, observe_contract_state,
observe_post_mutation_verification.

Implementation lives in _guardrails/{state, extractors, checks_pre,
checks_post, verification}. Test imports of the form ``from scripts.llm_solver.harness
.guardrails import GuardrailState, mark_bash_verified, …`` work unchanged.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from ._guardrails import (  # noqa: F401
    Action,
    Decision,
    GUARDRAIL_SPECS,
    GuardrailRegistry,
    GuardrailState,
    OBSERVER_ORDER,
    PASS,
    TOOL_POST_DISPATCH_ORDER,
    TOOL_PRE_DISPATCH_ORDER,
    TURN_PRE_DISPATCH_ORDER,
    contract_gate,
    done_guard,
    duplicate_guard,
    error_ladder,
    init_guardrail_state,
    intent_gate,
    loop_detect,
    mark_bash_verified,
    mutation_repeat_guard,
    observe_contract_state,
    observe_post_mutation_verification,
    observe_test_file_read,
    post_mutation_verification_gate,
    pre_mutation_gate,
    rumination_gate,
    rumination_ladder,
    rewind_on,
    test_read_ladder,
    _arm_recovery_mode,
    _canon_test_path,
    _clear_commit_contract,
    _clear_mutation_repeat_state,
    _clear_recovery_mode,
    _contract_abort_allowed,
    _contract_violation_signature,
    _equivalent_contract_violation_signature,
    _error_signature,
    _extract_bash_cmd,
    _extract_read_path,
    _extract_test_target,
    _is_concrete_file_path,
    _is_concrete_read,
    _is_test_command,
    _is_test_read,
    _looks_like_test_path,
    _mutation_signature,
    _record_contract_block,
    _record_mutation_repeat_block,
    _test_target_is_covered,
    _update_same_target_streak,
)


_GUARDRAIL_CALLABLES: dict[str, Callable[..., Any]] = {
    "intent_gate": intent_gate,
    "duplicate_guard": duplicate_guard,
    "loop_detect": loop_detect,
    "done_guard": done_guard,
    "mutation_repeat_guard": mutation_repeat_guard,
    "contract_gate": contract_gate,
    "pre_mutation_gate": pre_mutation_gate,
    "post_mutation_verification_gate": post_mutation_verification_gate,
    "rumination_gate": rumination_gate,
    "error_ladder": error_ladder,
    "test_read_ladder": test_read_ladder,
    "rumination_ladder": rumination_ladder,
    "mark_bash_verified": mark_bash_verified,
    "observe_test_file_read": observe_test_file_read,
    "observe_contract_state": observe_contract_state,
    "observe_post_mutation_verification": observe_post_mutation_verification,
}


def _builtins_for_phase(phase: str) -> dict[str, Callable[..., Any]]:
    """Build a phase registry from the first-class guardrail specs."""
    registry: dict[str, Callable[..., Any]] = {}
    for spec in GUARDRAIL_SPECS:
        if spec.phase != phase:
            continue
        try:
            registry[spec.name] = _GUARDRAIL_CALLABLES[spec.name]
        except KeyError as exc:
            raise RuntimeError(
                f"Guardrail spec has no callable: {spec.phase}.{spec.name}"
            ) from exc
    return registry


def build_guardrail_registry(
    *,
    turn_pre_overrides: dict[str, Callable[..., Decision]] | None = None,
    tool_pre_overrides: dict[str, Callable[..., Decision]] | None = None,
    tool_post_overrides: dict[str, Callable[..., Decision]] | None = None,
    observer_overrides: dict[str, Callable[..., None]] | None = None,
) -> GuardrailRegistry:
    """Build the effective guardrail registry with optional overrides."""
    turn_pre_dispatch = _builtins_for_phase("turn_pre_dispatch")
    tool_pre_dispatch = _builtins_for_phase("tool_pre_dispatch")
    tool_post_dispatch = _builtins_for_phase("tool_post_dispatch")
    observers = _builtins_for_phase("observers")
    # A typo in an override key silently shadows nothing. Refuse the
    # override if the key isn't in
    # the builtin map — replacement, not addition, is the contract.
    def _check_keys(builtin: dict, overrides: dict | None, phase: str) -> None:
        if not overrides:
            return
        unknown = sorted(k for k in overrides if k not in builtin)
        if unknown:
            raise ValueError(
                f"Unknown {phase} guardrail override(s): {', '.join(unknown)}. "
                f"Known: {', '.join(sorted(builtin))}"
            )

    _check_keys(turn_pre_dispatch, turn_pre_overrides, "turn_pre")
    _check_keys(tool_pre_dispatch, tool_pre_overrides, "tool_pre")
    _check_keys(tool_post_dispatch, tool_post_overrides, "tool_post")
    _check_keys(observers, observer_overrides, "observer")

    if turn_pre_overrides:
        turn_pre_dispatch.update(turn_pre_overrides)
    if tool_pre_overrides:
        tool_pre_dispatch.update(tool_pre_overrides)
    if tool_post_overrides:
        tool_post_dispatch.update(tool_post_overrides)
    if observer_overrides:
        observers.update(observer_overrides)
    return GuardrailRegistry(
        turn_pre_dispatch=turn_pre_dispatch,
        tool_pre_dispatch=tool_pre_dispatch,
        tool_post_dispatch=tool_post_dispatch,
        observers=observers,
    )


def validate_guardrail_registry(registry: GuardrailRegistry) -> None:
    """Fail fast when the registry is missing required call-site names
    or an entry is None / not callable.

    A `None` override would land here as a registered entry and fail
    mid-turn at the
    first call. Now the validator catches it at session start.
    """

    def _missing(required: tuple[str, ...], registered: dict[str, Callable]) -> list[str]:
        return [name for name in required if name not in registered]

    missing = (
        _missing(TURN_PRE_DISPATCH_ORDER, registry.turn_pre_dispatch)
        + _missing(TOOL_PRE_DISPATCH_ORDER, registry.tool_pre_dispatch)
        + _missing(TOOL_POST_DISPATCH_ORDER, registry.tool_post_dispatch)
        + _missing(OBSERVER_ORDER, registry.observers)
    )
    if missing:
        raise ValueError(f"Guardrail registry missing required handlers: {', '.join(sorted(missing))}")

    bad: list[str] = []
    for phase_name, registered in (
        ("turn_pre", registry.turn_pre_dispatch),
        ("tool_pre", registry.tool_pre_dispatch),
        ("tool_post", registry.tool_post_dispatch),
        ("observer", registry.observers),
    ):
        for name, fn in registered.items():
            if fn is None or not callable(fn):
                bad.append(f"{phase_name}.{name}")
    if bad:
        raise ValueError(
            f"Guardrail registry has non-callable entries: {', '.join(sorted(bad))}"
        )
