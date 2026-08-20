"""Post-tool guardrail checks + observers — extracted from guardrails.py."""
from __future__ import annotations

import logging
import re
from typing import Any

from ..._shared.classification import is_error_result
from .state import PASS, Action, Decision, GuardrailState
from .extractors import (
    MUTATION_TOOLS,
    _arm_recovery_mode,
    _canon_test_path,
    _clear_commit_contract,
    _clear_mutation_repeat_state,
    _clear_recovery_mode,
    _error_signature,
    _extract_bash_cmd,
    _extract_read_path,
    _extract_test_target,
    _is_bash_write_like,
    _is_concrete_file_path,
    _is_concrete_read,
    _mutation_signature,
    _is_test_command,
    _is_test_read,
    _looks_like_test_path,
    _test_target_is_covered,
)

log = logging.getLogger(__name__)


def _is_tool_error(result: str) -> bool:
    """Backward-compat shim — delegates to the canonical helper.

    Kept as a name for inline call-sites that read fluently as
    ``if _is_tool_error(result): ...``; new code should import
    ``is_error_result`` from ``_shared.classification`` directly.
    """
    return is_error_result(result)


def error_ladder(state: GuardrailState, cfg: Any, *,
                 tc_name: str, result: str) -> Decision:
    """Error cascade: WARN at nudge threshold, END at abort threshold.

    Two counters in series:
      - Per-tool consecutive count (reset on any non-error). Used for the
        existing `error_abort_threshold` and the WARN nudge.
      - Same-signature streak (reset only when a NEW error signature
        appears; non-error turns do NOT reset). Used for the
        `error_same_class_threshold` short-circuit. This catches loops
        where the model alternates non-error tool calls with the same
        recurring error.
    """
    if not cfg.error_ladder_enabled:
        return PASS
    if not _is_tool_error(result):
        state.consecutive_errors[tc_name] = 0
        return PASS
    state.consecutive_errors[tc_name] = state.consecutive_errors.get(tc_name, 0) + 1
    count = state.consecutive_errors[tc_name]

    sig = _error_signature(result)
    if sig and sig == state.same_class_error_signature:
        state.same_class_error_count += 1
    else:
        state.same_class_error_signature = sig
        state.same_class_error_count = 1
    same_class = state.same_class_error_count
    same_class_threshold = int(getattr(cfg, "error_same_class_threshold", 0) or 0)
    if same_class_threshold > 0 and same_class == same_class_threshold:
        return Decision.warn(
            f"[HARNESS: {same_class} consecutive same-class errors "
            f"(signature={sig}); review the approach — same fix retried "
            f"{same_class} times has the same outcome.]",
            reason="error_ladder.same_class",
        )

    if (cfg.error_abort_threshold > 0
            and count >= cfg.error_abort_threshold):
        return Decision.end("error_abort")
    if count == cfg.error_nudge_threshold:
        return Decision.warn(cfg.error_nudge.format(count=count), reason="error_ladder.nudge")
    return PASS


def test_read_ladder(
    state: GuardrailState,
    cfg: Any,
    *,
    tc_name: str,
    result: str,
    gate_blocked: bool,
    tc_args: dict | None = None,
) -> Decision:
    """Warn when verification repeats before the relevant test file is read."""
    warn_after = int(getattr(cfg, "test_read_warn_after", 0) or 0)
    if warn_after <= 0 or tc_name != "bash" or gate_blocked or is_error_result(result):
        return PASS
    cmd = ""
    if isinstance(tc_args, dict):
        raw = tc_args.get("cmd")
        if isinstance(raw, str):
            cmd = raw
    target = _extract_test_target(cmd)
    if not target:
        return PASS
    state.last_test_target = target
    if _test_target_is_covered(state, target):
        state.test_runs_without_test_read = 0
        state.test_read_nudge_target = ""
        return PASS
    state.test_runs_without_test_read += 1
    if state.test_runs_without_test_read < warn_after:
        return PASS
    if state.test_read_nudge_target == target:
        return PASS
    state.test_read_nudge_target = target
    return Decision.warn(
        cfg.test_read_nudge.format(
            count=state.test_runs_without_test_read,
            target=target,
        ),
        reason="test_read_ladder",
    )


def rumination_ladder(state: GuardrailState, cfg: Any, *,
                      tc_name: str, result: str, gate_blocked: bool,
                      already_blocked_this_turn: bool,
                      tc_args: dict | None = None,
                      focus_key: str = "", focus_display: str = "") -> Decision:
    """Post-dispatch tier of the rumination ladder: warn, then activate.

    On successful write/edit: reset counter + gate + nudge flag.
    On any other tc: increment the counter, warn once at the nudge
      threshold, and activate the gate at its threshold.

    ``already_blocked_this_turn`` signals that rumination_gate blocked
    or graced this call — in that case skip the counter bump to avoid
    double counting.
    """
    if tc_name in MUTATION_TOOLS or _is_bash_write_like(tc_name, tc_args):
        if not _is_tool_error(result):
            state.non_write_calls_since_write = 0
            state.same_target_key = ""
            state.same_target_display = ""
            state.same_target_count = 0
            state.same_target_nudge_emitted = False
            state.rumination_gate = False
            state.rumination_gate_grace = 0
            state.rumination_nudge_emitted = False
            state.gate_block_count = 0
            state.has_mutated = True
            state.verified_since_mutation = False
            state.mutation_count += 1
            _clear_commit_contract(state)
            _clear_recovery_mode(state)
            state.verify_repeat_sig = ""
            state.verify_repeat_count = 0
            state.mutation_count_at_last_verify = state.mutation_count
        return PASS

    if not cfg.rumination_enabled:
        return PASS
    if already_blocked_this_turn:
        return PASS

    state.non_write_calls_since_write += 1
    _update_same_target_streak(state, focus_key=focus_key, focus_display=focus_display)

    recovery_same_target = int(
        getattr(cfg, "contract_recovery_same_target_threshold", 0) or 0
    )
    if (recovery_same_target > 0
            and state.same_target_key
            and state.same_target_count >= recovery_same_target):
        target = state.same_target_display or state.same_target_key
        if state.same_target_key.startswith("outside:"):
            reason = "repeated inspection outside repo root"
        else:
            reason = "repeated same-target inspection"
        _arm_recovery_mode(state, reason=reason, target=target)

    # WARN: one-shot nudge text.
    # Threshold depends on state.has_mutated: the pre-mutation threshold is a
    # "start editing" push for stuck tasks; the post-mutation threshold
    # preserves baseline-like behavior for tasks already editing productively.
    # Defaults make post == pre; set rumination_nudge_threshold_abs_post_mutation
    # to decouple them.
    # rumination_nudge_only_pre_mutation acts as a hard suppress post-mutation
    # (equivalent to post_mutation_threshold = infinity).
    threshold = (state.rumination_nudge_threshold_post_mutation
                 if state.has_mutated
                 else state.rumination_nudge_threshold)
    warn_parts: list[str] = []
    if (not state.rumination_nudge_emitted
            and state.non_write_calls_since_write >= threshold
            and not (cfg.rumination_nudge_only_pre_mutation and state.has_mutated)):
        warn_parts.append(cfg.rumination_nudge.format(
            count=state.non_write_calls_since_write,
        ))
        state.rumination_nudge_emitted = True

    same_target_warn = int(getattr(cfg, "rumination_same_target_warn_count", 0) or 0)
    if (same_target_warn > 0
            and state.same_target_key
            and not state.same_target_nudge_emitted
            and state.same_target_count >= same_target_warn):
        target = state.same_target_display or state.same_target_key
        if state.same_target_key.startswith("outside:"):
            warn_parts.append(cfg.rumination_outside_cwd_nudge.format(
                count=state.same_target_count,
                target=target,
            ))
        else:
            warn_parts.append(cfg.rumination_same_target_nudge.format(
                count=state.same_target_count,
                target=target,
            ))
        state.same_target_nudge_emitted = True

    # ARM: flip the gate flag. The block tier runs next turn.
    same_target_arm = int(getattr(cfg, "rumination_same_target_arm_count", 0) or 0)
    if (not state.rumination_gate
            and (
                state.non_write_calls_since_write >= state.rumination_arm_threshold
                or (
                    same_target_arm > 0
                    and state.same_target_key
                    and state.same_target_count >= same_target_arm
                )
            )):
        state.rumination_gate = True
        state.rumination_gate_grace = cfg.rumination_gate_grace_calls

    warn_text = "\n".join(part for part in warn_parts if part)
    return Decision.warn(warn_text, reason="rumination_ladder") if warn_text else PASS


def _update_same_target_streak(
    state: GuardrailState,
    *,
    focus_key: str,
    focus_display: str,
) -> None:
    """Track repeated inspection of the same target between mutations.

    ``focus_key`` is a content-blind signature supplied by the loop: a file
    path for file tools / bash file reads, or a normalized bash command for
    generic bash inspections. Changing targets resets the streak.
    """
    if not focus_key:
        state.same_target_key = ""
        state.same_target_display = ""
        state.same_target_count = 0
        state.same_target_nudge_emitted = False
        return
    if state.same_target_key == focus_key:
        state.same_target_count += 1
        if focus_display:
            state.same_target_display = focus_display
        return
    state.same_target_key = focus_key
    state.same_target_display = focus_display or focus_key
    state.same_target_count = 1
    state.same_target_nudge_emitted = False


def mark_bash_verified(state: GuardrailState, cfg: Any, *,
                       tc_name: str, result: str, gate_blocked: bool, **_: Any) -> None:
    """Update verified_since_mutation on a content-blind signal.

    Two pathways qualify (post-mutation, non-blocked, non-ERROR):
    - ``run_tests``: structured envelope opens with
      ``<test_results status="passed"`` (pytest exit 0, no length threshold —
      the status field is already a hard signal).
    - ``bash``: no ``[exit code: N]`` marker for N≠0 AND real-text length
      exceeds ``cfg.done_verified_bash_min_chars``.

    Name kept for back-compat (it's now misnamed; touch sites referenced
    across loop.py / OBSERVER_ORDER / tests).
    """
    if not state.has_mutated or gate_blocked:
        return
    if is_error_result(result):
        return
    if tc_name == "run_tests":
        # Recognise both the bare envelope and a unified <tool_result>
        # wrap that contains a passed <test_results>. The wrap is
        # currently suppressed for run_tests by skip-already-enveloped
        # logic in dispatch, but if that ever changes, this branch
        # still flips verified_since_mutation.
        if (
            result.startswith('<test_results status="passed"')
            or '<test_results status="passed"' in result
        ):
            state.verified_since_mutation = True
        return
    if tc_name != "bash":
        return
    # Strip harness-injected text before the verification length test.
    # The loop appends [HARNESS: …] lines (test_read_nudge,
    # rumination_grace_prefix, error_ladder warns, contract warns)
    # AFTER dispatch and BEFORE this observer runs. Without subtraction,
    # a 0-byte real bash output can clear the 200-char threshold purely
    # on harness-injected text, falsely flipping verified_since_mutation.
    # Also strip the unified <tool_result …> envelope's opening/closing
    # tags because they are harness-emitted padding. The "real"
    # verification length is what
    # dispatch produced.
    def _is_padding(line: str) -> bool:
        s = line.lstrip()
        if s.startswith("[HARNESS:"):
            return True
        if s.startswith("<tool_result") or s.startswith("</tool_result"):
            return True
        return False

    real_lines = [ln for ln in result.splitlines() if not _is_padding(ln)]
    real_text = "\n".join(real_lines)
    exit_marker_present = "[exit code:" in real_text
    exit_ok = ("[exit code: 0]" in real_text) or not exit_marker_present
    if exit_ok and len(real_text) > cfg.done_verified_bash_min_chars:
        state.verified_since_mutation = True


def observe_test_file_read(
    state: GuardrailState,
    cfg: Any,
    *,
    tc_name: str,
    result: str,
    gate_blocked: bool,
    tc_args: dict | None = None,
    focus_key: str = "",
    focus_display: str = "",
) -> None:
    """Track which test files have been read so test runs can demand them."""
    del cfg
    if gate_blocked or is_error_result(result):
        return
    path = ""
    if tc_name == "read" and isinstance(tc_args, dict):
        raw = tc_args.get("path") or tc_args.get("file_path")
        if isinstance(raw, str):
            path = raw
    elif tc_name == "bash" and focus_key.startswith("file:") and _looks_like_test_path(focus_display):
        path = focus_display
    if not _looks_like_test_path(path):
        return
    state.test_file_reads.add(_canon_test_path(path))
    if state.last_test_target and _test_target_is_covered(state, state.last_test_target):
        state.test_runs_without_test_read = 0
        state.test_read_nudge_target = ""


def observe_contract_state(
    state: GuardrailState,
    cfg: Any,
    *,
    tc_name: str,
    result: str,
    gate_blocked: bool,
    tc_args: dict | None = None,
    focus_key: str = "",
    focus_display: str = "",
) -> None:
    """Track contract state from successful, content-blind tool outcomes."""
    if gate_blocked or _is_tool_error(result):
        return

    if tc_name in MUTATION_TOOLS or _is_bash_write_like(tc_name, tc_args):
        sig, target = _mutation_signature(tc_name, tc_args, focus_display=focus_display)
        if sig and state.mutation_repeat_sig == sig:
            state.mutation_repeat_count += 1
        elif sig:
            state.mutation_repeat_sig = sig
            state.mutation_repeat_target = target
            state.mutation_repeat_count = 1
        else:
            _clear_mutation_repeat_state(state)
        state.mutation_repeat_block_sig = ""
        state.mutation_repeat_block_count = 0
        _clear_commit_contract(state)
        _clear_recovery_mode(state)
        state.verify_repeat_sig = ""
        state.verify_repeat_count = 0
        state.mutation_count_at_last_verify = state.mutation_count
        return

    if _is_test_read(tc_name, tc_args, focus_key=focus_key, focus_display=focus_display):
        _clear_mutation_repeat_state(state)
        _clear_commit_contract(state)
        _clear_recovery_mode(state)
        return

    if _is_test_command(tc_name, tc_args):
        _clear_mutation_repeat_state(state)
        cmd = _extract_bash_cmd(tc_name, tc_args)
        target = _extract_test_target(cmd) or cmd
        if (state.verify_repeat_sig == target
                and state.mutation_count_at_last_verify == state.mutation_count):
            state.verify_repeat_count += 1
        else:
            state.verify_repeat_sig = target
            state.verify_repeat_count = 1
            state.mutation_count_at_last_verify = state.mutation_count
        _clear_commit_contract(state)
        _clear_recovery_mode(state)
        threshold = int(getattr(cfg, "contract_recovery_verify_repeat_threshold", 0) or 0)
        if threshold > 0 and state.verify_repeat_count >= threshold:
            _arm_recovery_mode(
                state,
                reason="repeated verification without refinement",
                target=target or "verification target",
            )
        return

    read_path = _extract_read_path(
        tc_name, tc_args, focus_key=focus_key, focus_display=focus_display,
    )
    if (
        read_path
        and _is_concrete_file_path(read_path)
        and not _looks_like_test_path(read_path)
        and not state.has_mutated
        and not focus_key.startswith("outside:")
    ):
        _clear_mutation_repeat_state(state)
        state.commit_pending = True
        state.commit_source_path = read_path
        state.commit_violation_count = 0
        state.commit_turns_since_arm = 0
        if state.recovery_mode_active and state.recovery_target == read_path:
            _clear_recovery_mode(state)
        return

    _clear_mutation_repeat_state(state)
