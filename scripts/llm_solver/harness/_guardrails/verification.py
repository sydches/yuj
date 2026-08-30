"""Post-mutation verification guardrail and observer."""
from __future__ import annotations

from typing import Any

from ..._shared.classification import is_error_result
from .extractors import MUTATION_TOOLS, _is_bash_write_like, _is_test_command
from .state import PASS, Decision, GuardrailState


def post_mutation_verification_gate(
    state: GuardrailState,
    cfg: Any,
    *,
    tc_name: str,
    tc_args: dict | None = None,
) -> Decision:
    """Block more custom shell checks once formal verification is due.

    Source mutation and registered test-runner commands always pass. Reads and
    searches also pass so the model can locate the right existing test scope.
    The gate only replaces the custom shell-check habit that triggered it.
    """
    threshold = int(
        getattr(cfg, "post_mutation_verification_gate_after", 0) or 0
    )
    if (
        threshold <= 0
        or not state.has_mutated
        or state.formal_verification_passed_since_mutation
        or not state.post_mutation_verification_gate_armed
    ):
        return PASS
    if (
        tc_name in MUTATION_TOOLS
        or _is_bash_write_like(tc_name, tc_args)
        or _is_test_command(tc_name, tc_args)
    ):
        return PASS
    if tc_name not in {"bash", "exec_cell"}:
        return PASS
    return Decision.block(
        cfg.post_mutation_verification_gate,
        reason="post_mutation_verification_gate",
    )


def observe_post_mutation_verification(
    state: GuardrailState,
    cfg: Any,
    *,
    tc_name: str,
    result: str,
    gate_blocked: bool,
    tc_args: dict | None = None,
    **_: Any,
) -> None:
    """Arm a formal-test gate after repeated post-mutation shell checks."""
    if gate_blocked:
        return
    if tc_name in MUTATION_TOOLS or _is_bash_write_like(tc_name, tc_args):
        if not is_error_result(result):
            state.post_mutation_non_test_bash_count = 0
            state.post_mutation_verification_gate_armed = False
            state.formal_verification_passed_since_mutation = False
        return
    if not state.has_mutated:
        return
    if _is_test_command(tc_name, tc_args):
        state.post_mutation_non_test_bash_count = 0
        state.post_mutation_verification_gate_armed = False
        passed = not is_error_result(result)
        state.formal_verification_passed_since_mutation = passed
        state.verified_since_mutation = passed
        return
    if tc_name not in {"bash", "exec_cell"}:
        return
    state.post_mutation_non_test_bash_count += 1
    threshold = int(
        getattr(cfg, "post_mutation_verification_gate_after", 0) or 0
    )
    if threshold > 0 and state.post_mutation_non_test_bash_count >= threshold:
        state.post_mutation_verification_gate_armed = True
