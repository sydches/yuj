"""Pre-tool guardrail checks — extracted from guardrails.py."""
from __future__ import annotations

import logging
from typing import Any

from .state import PASS, Action, Decision, GuardrailState
from .extractors import (
    MUTATION_TOOLS,
    _arm_recovery_mode,
    _clear_commit_contract,
    _clear_mutation_repeat_state,
    _clear_recovery_mode,
    _contract_abort_allowed,
    _contract_violation_signature,
    _equivalent_contract_violation_signature,
    _extract_bash_cmd,
    _extract_read_path,
    _extract_test_target,
    _is_bash_write_like,
    _is_concrete_file_path,
    _is_concrete_read,
    _is_test_command,
    _is_test_read,
    _looks_like_test_path,
    _mutation_signature,
    _record_contract_block,
    _record_mutation_repeat_block,
    _test_target_is_covered,
)

log = logging.getLogger(__name__)


def intent_gate(state: GuardrailState, cfg: Any, *,
                turn: int, content: str, tool_calls: list) -> Decision:
    """Reject silent tool calls.

    GRACE: first ``cfg.intent_grace_turns`` turns get a free pass.
    BLOCK: tool_calls present + no reasoning content → reject this turn.
    END: ``cfg.intent_abort_threshold`` consecutive rejections → end session.
    """
    if not cfg.require_intent or not tool_calls:
        state.consecutive_intent_rejections = 0
        return PASS
    if turn < cfg.intent_grace_turns:
        state.consecutive_intent_rejections = 0
        return PASS
    if (content or "").strip():
        state.consecutive_intent_rejections = 0
        return PASS

    state.intent_block_count += 1
    state.consecutive_intent_rejections += 1
    if state.intent_first_block_turn is None:
        state.intent_first_block_turn = turn
        text = cfg.intent_gate_first
    else:
        text = cfg.intent_gate_repeat.format(
            count=state.intent_block_count,
            first_turn=state.intent_first_block_turn,
        )
    if (cfg.intent_abort_threshold > 0
            and state.consecutive_intent_rejections >= cfg.intent_abort_threshold):
        # END overrides BLOCK: record the rejection text but end the session.
        return Decision(Action.END, text=text, reason="intent_abort")
    return Decision.block(text, reason="intent_gate")


def loop_detect(state: GuardrailState, cfg: Any, *,
                tool_calls_sig: tuple) -> Decision:
    """Tight loop detector with one recovery-inject before hard abort.

    Borrowed in spirit from Gemini CLI's LoopDetectionService
    (``packages/core/src/services/loopDetectionService.ts``, PR #8231):
    on the same structural hash repeating for N turns, inject a
    synthetic steering message and allow the model one more turn to
    change approach. If the pattern persists, end the session.

    Differs from ``duplicate_guard`` by firing at a much tighter
    threshold (default 5) and by the WARN tier — duplicate_guard's WARN
    is a threshold announcement; this WARN is a recovery-inject.

    State slice: ``loop_detect_*`` on ``GuardrailState``. Registry
    phase: ``turn_pre_dispatch``.
    """
    if not cfg.loop_detect_enabled:
        state.loop_detect_streak = 0
        state.loop_detect_last_sig = ()
        state.loop_detect_warned = False
        return PASS
    if tool_calls_sig == state.loop_detect_last_sig:
        state.loop_detect_streak += 1
    else:
        state.loop_detect_last_sig = tool_calls_sig
        state.loop_detect_streak = 1
        state.loop_detect_warned = False
    if state.loop_detect_streak >= cfg.loop_detect_threshold:
        if not state.loop_detect_warned:
            state.loop_detect_warned = True
            return Decision.warn(
                cfg.loop_detect_recovery.format(streak=state.loop_detect_streak),
                reason="loop_detect.recovery",
            )
        return Decision.end("loop_detected")
    return PASS


def duplicate_guard(state: GuardrailState, cfg: Any, *,
                    tool_calls_sig: tuple) -> Decision:
    """End session on N identical consecutive calls; WARN one turn earlier.

    Fires on every turn, including while the rumination gate is armed —
    pausing here would let the model cycle the same blocked call up to
    rumination_gate_max_blocks times before any terminal action fires,
    which is LARGER tolerance during a MORE dangerous state.
    """
    if not cfg.duplicate_guard_enabled:
        return PASS
    state.recent_calls.append(tool_calls_sig)
    # END — declared-disabled when duplicate_abort <= 0 (some baselines
    # zero it; previously that silently never matched the deque length and
    # the warn text printed "session ends at 0 identical").
    if (cfg.duplicate_abort > 0
            and len(state.recent_calls) == cfg.duplicate_abort
            and len(set(state.recent_calls)) == 1):
        return Decision.end("duplicate_abort")
    # WARN (optional, config-gated)
    if cfg.duplicate_warn_count > 0:
        tail = 0
        for s in reversed(state.recent_calls):
            if s == tool_calls_sig:
                tail += 1
            else:
                break
        if tail >= cfg.duplicate_warn_count:
            abort_disp = cfg.duplicate_abort if cfg.duplicate_abort > 0 else "disabled"
            return Decision.warn(
                cfg.duplicate_warn.format(count=tail, abort=abort_disp),
                reason="duplicate_guard",
            )
    return PASS


# ─── Per-tool-call pre-dispatch guardrails ───────────────────────────────

def pre_mutation_gate(
    state: GuardrailState, cfg: Any, *, tc_name: str, turn_number: int,
    tc_args: dict | None = None,
) -> Decision:
    """Force commitment after N orientation turns without a mutation.

    The model is allowed cfg.pre_mutation_turn_cap read-only turns at the
    start of a session. Once that budget is exhausted AND the model has
    not yet executed a mutation, every non-mutation tool call
    is BLOCKED with a stern harness message until the model commits.

    `done` is exempt — the model can still legitimately call it (for
    tasks that are already in the desired state or judged unfixable).
    Mutations are exempt — they are exactly what the gate demands.

    """
    cap = int(getattr(cfg, "pre_mutation_turn_cap", 0) or 0)
    if cap <= 0:
        return PASS
    if state.has_mutated:
        return PASS
    if turn_number < cap:
        return PASS
    if (
        tc_name in (
            "write", "edit", "str_replace", "create", "apply_patch", "udiff", "done"
        )
        or _is_bash_write_like(tc_name, tc_args)
    ):
        return PASS
    template = getattr(
        cfg,
        "pre_mutation_gate",
        "[HARNESS: {turn_number} read-only turns elapsed without a file mutation; the next tool call must use the selected file-edit tool, run a bash command that mutates a source file, or call done(). This call was not executed.]",
    )
    try:
        text = template.format(turn_number=turn_number)
    except (KeyError, IndexError, ValueError):
        text = template
    return Decision.block(text, reason="pre_mutation_gate")


from ._git_dirty import cwd_has_uncommitted_changes as _cwd_has_uncommitted_changes


def done_guard(
    state: GuardrailState,
    cfg: Any,
    *,
    tc_name: str,
    cwd: str | None = None,
) -> Decision:
    """Verify `done` is premature or legitimate.

    Two modes, selected by cfg.done_require_pretest_parity:

    PARITY MODE (opt-in, ground-truth):
      Requires the structured output pipeline. At session 1 start, the
      harness parses pretest output into failing/passing test sets and
      stores them in state.pretest_failing_tests / pretest_passing_tests.
      Each subsequent test run updates state.latest_test_parsed and the
      green_parity_streak counter. `done` is accepted only when:
        1. the latest test run covers every pretest-failing test and
           every one of those now shows PASSED, AND
        2. no pretest-passing test is now FAILED/ERROR (no regression),
           AND
        3. the parity streak has reached cfg.done_parity_runs_required.
      Reject otherwise with a reason naming the specific tests that
      block acceptance.

      When pretest was not parseable (sets empty), PARITY MODE falls
      back to HEURISTIC MODE so non-test tasks / runners without an
      [output_parser] block stay functional.

    HEURISTIC MODE (default):
      Requires the mutation + verified_since_mutation preconditions
      (the 200-char bash heuristic — content-blind, task-agnostic,
      but imprecise: the tracker recorded false rejections under it).

    LOOP FAILSAFE:
      `state.done_blocked_count` accumulates across every BLOCK from
      this guard. After `cfg.done_loop_abort_after` blocks the next
      block converts to END — the session ends regardless of cause and
      the model's current code becomes the final patch. This bounds
      the wasted-pt cost of any rejected-done loop (parity flake, novel
      verify-path drift, model misreading the gate) to ~N round-trips.
      Set `done_loop_abort_after = 0` to disable.
    """
    if tc_name != "done":
        return PASS
    if not cfg.done_guard_enabled:
        return PASS

    use_parity = (
        getattr(cfg, "done_require_pretest_parity", False)
        and state.pretest_failing_tests
    )
    if use_parity:
        latest = state.latest_test_parsed
        # Log which mode this done attempt used so the operator can see
        # which path
        # the rejection / pass came from. Pre-fix this was logged only
        # at session start, so multi-attempt traces were ambiguous.
        log.info(
            "done_guard parity-mode: latest_run=%s failing=%d streak=%d",
            "yes" if latest else "no", len(state.pretest_failing_tests),
            state.green_parity_streak,
        )
        if not latest:
            return _done_block_or_abort(state, cfg, cfg.done_reject_parity_no_run)
        passed_now = {t for t, v in latest.items() if v in ("PASSED", "PASS")}
        still_failing = state.pretest_failing_tests - passed_now
        if still_failing:
            shown = sorted(still_failing)[:5]
            extra_count = len(still_failing) - len(shown)
            return _done_block_or_abort(state, cfg, cfg.done_reject_parity_still_failing.format(
                shown=shown,
                extra=f" (+{extra_count} more)" if extra_count > 0 else "",
            ))
        regressed = {
            t for t, v in latest.items()
            if t in state.pretest_passing_tests and v not in ("PASSED", "PASS")
        }
        if regressed:
            shown = sorted(regressed)[:5]
            extra_count = len(regressed) - len(shown)
            return _done_block_or_abort(state, cfg, cfg.done_reject_parity_regression.format(
                shown=shown,
                extra=f" (+{extra_count} more)" if extra_count > 0 else "",
            ))
        required = getattr(cfg, "done_parity_runs_required", 1)
        if state.green_parity_streak < required:
            return _done_block_or_abort(state, cfg, cfg.done_reject_parity_streak.format(
                count=state.green_parity_streak,
                required=required,
            ))
        # Parity satisfied — bypass heuristic preconditions.
        return PASS

    # Fallback / heuristic mode.
    if cfg.done_require_mutation and not state.has_mutated:
        # A bash command may have edited files without setting has_mutated.
        # Check the working tree before rejecting the done call.
        if _cwd_has_uncommitted_changes(cwd):
            state.has_mutated = True
            # verified_since_mutation stays False — that's the next check
            # and is the model's responsibility to satisfy.
        else:
            return _done_block_or_abort(state, cfg, cfg.done_reject_no_mutation)
    if cfg.done_require_verify and not state.verified_since_mutation:
        return _done_block_or_abort(state, cfg, cfg.done_reject_no_verify)
    return PASS


def _done_block_or_abort(state: GuardrailState, cfg: Any, text: str) -> Decision:
    """Increment the done-block counter; convert to END at threshold.

    The session-ending path returns ``Action.END`` with a reason of
    ``done_loop`` so the postmortem quintet can flag the abort cause
    distinctly from `model_done` / `error_abort` / other END branches.
    """
    state.done_blocked_count += 1
    abort_after = int(getattr(cfg, "done_loop_abort_after", 0) or 0)
    if abort_after > 0 and state.done_blocked_count >= abort_after:
        end_text = getattr(cfg, "done_loop_abort_text", "Session ended: rejected done loop.")
        try:
            end_text = end_text.format(n=state.done_blocked_count)
        except (KeyError, IndexError):
            pass
        return Decision(Action.END, text=end_text, reason="done_loop")
    return Decision.block(text, reason="done_guard")




def mutation_repeat_guard(
    state: GuardrailState,
    cfg: Any,
    *,
    tc_name: str,
    tc_args: dict | None = None,
    focus_display: str = "",
    **_: Any,
) -> Decision:
    """Warn/block/end repeated identical mutation attempts.

    Keyed by the mutation signature (tool + target + payload digest), so
    iterative edits to the same file still pass when the mutation changes.
    """
    warn_after = int(getattr(cfg, "mutation_repeat_warn_after", 0) or 0)
    block_after = int(getattr(cfg, "mutation_repeat_block_after", 0) or 0)
    abort_after = int(getattr(cfg, "mutation_repeat_abort_after", 0) or 0)
    if tc_name not in MUTATION_TOOLS or (warn_after <= 0 and block_after <= 0 and abort_after <= 0):
        return PASS
    sig, target = _mutation_signature(tc_name, tc_args, focus_display=focus_display)
    if not sig or state.mutation_repeat_sig != sig or state.mutation_repeat_count <= 0:
        state.mutation_repeat_block_sig = ""
        state.mutation_repeat_block_count = 0
        return PASS

    next_count = state.mutation_repeat_count + 1
    shown_target = target or state.mutation_repeat_target or "current file"
    if block_after > 0 and next_count >= block_after:
        repeat_blocks = _record_mutation_repeat_block(state, sig)
        text = cfg.mutation_repeat_block.format(target=shown_target)
        if abort_after > 0 and repeat_blocks >= abort_after:
            return Decision(Action.END, text=text, reason="mutation_repeat_abort")
        return Decision.block(text, reason="mutation_repeat_guard")
    if warn_after > 0 and next_count >= warn_after:
        return Decision.warn(
            cfg.mutation_repeat_warn.format(target=shown_target),
            reason="mutation_repeat_guard.warn",
        )
    return PASS




def contract_gate(
    state: GuardrailState,
    cfg: Any,
    *,
    tc_name: str,
    tc_args: dict | None = None,
    focus_key: str = "",
    focus_display: str = "",
) -> Decision:
    """Block broad exploration once a tighter contract is active.

    Two content-blind contracts are supported:

    - Commit contract: after a non-test file read, the next useful move must be
      edit/write, read a test file, or run verification.
    - Recovery contract: once same-target / verify-repeat recovery arms, only a
      concrete read, edit/write, or verification command may execute.
    """
    if tc_name == "done":
        return PASS

    is_commit_allowed = (
        tc_name in MUTATION_TOOLS
        or _is_bash_write_like(tc_name, tc_args)
        or _is_test_read(tc_name, tc_args, focus_key=focus_key, focus_display=focus_display)
        or _is_test_command(tc_name, tc_args)
    )
    is_recovery_allowed = is_commit_allowed or _is_concrete_read(
        tc_name, tc_args, focus_key=focus_key, focus_display=focus_display,
    )

    if state.recovery_mode_active:
        state.recovery_turns_since_arm += 1
        if is_recovery_allowed:
            state.contract_block_sig = ""
            state.contract_block_count = 0
            return PASS
        target = state.recovery_target or focus_display or focus_key or "current focus"
        reason = state.recovery_reason or "repeated exploration"
        sig = _contract_violation_signature(
            cfg, tc_name, tc_args, focus_key=focus_key, focus_display=focus_display,
        )
        repeat_count = _record_contract_block(state, sig)
        abort_after = int(getattr(cfg, "contract_invalid_repeat_abort_after", 0) or 0)
        text = cfg.contract_recovery_block.format(reason=reason, target=target)
        if (
            abort_after > 0
            and repeat_count >= abort_after
            and _contract_abort_allowed(state, cfg, lane="recovery")
        ):
            return Decision(Action.END, text=text, reason="contract_recovery_abort")
        return Decision.block(text, reason="contract_gate.recovery")

    warn_after = int(getattr(cfg, "contract_commit_warn_after", 0) or 0)
    block_after = int(getattr(cfg, "contract_commit_block_after", 0) or 0)
    if not state.commit_pending or (warn_after <= 0 and block_after <= 0):
        state.contract_block_sig = ""
        state.contract_block_count = 0
        return PASS
    state.commit_turns_since_arm += 1
    if is_commit_allowed:
        state.commit_violation_count = 0
        state.contract_block_sig = ""
        state.contract_block_count = 0
        return PASS

    state.commit_violation_count += 1
    source = state.commit_source_path
    if not _is_concrete_file_path(source):
        source = focus_display if _is_concrete_file_path(focus_display) else "current source"
    if block_after > 0 and state.commit_violation_count >= block_after:
        sig = _contract_violation_signature(
            cfg, tc_name, tc_args, focus_key=focus_key, focus_display=focus_display,
        )
        repeat_count = _record_contract_block(state, sig)
        text = cfg.contract_commit_block.format(source=source)
        abort_after = int(getattr(cfg, "contract_invalid_repeat_abort_after", 0) or 0)
        if (
            abort_after > 0
            and repeat_count >= abort_after
            and _contract_abort_allowed(state, cfg, lane="commit")
        ):
            return Decision(Action.END, text=text, reason="contract_commit_abort")
        return Decision.block(text, reason="contract_gate.commit")
    if warn_after > 0 and state.commit_violation_count >= warn_after:
        return Decision.warn(
            cfg.contract_commit_warn.format(source=source),
            reason="contract_gate.commit.warn",
        )
    return PASS


def rumination_gate(state: GuardrailState, cfg: Any, *,
                    tc_name: str, tc_args: dict | None = None) -> Decision:
    """Hard gate armed by the rumination ladder: block non-writes.

    GRACE (1 call): execute with a warning prefix (returned as WARN so
      the caller knows to dispatch but append the prefix).
    BLOCK: reject non-writes; count toward gate_max_blocks.
    END: after ``cfg.rumination_gate_max_blocks`` blocks → end session.

    Mutation tools pass through (PASS); a successful mutation clears the
    gate via reset_on_successful_write().
    """
    if not cfg.rumination_enabled:
        return PASS
    if not state.rumination_gate:
        return PASS
    if tc_name in MUTATION_TOOLS or _is_bash_write_like(tc_name, tc_args):
        return PASS
    if state.rumination_gate_grace > 0:
        state.rumination_gate_grace -= 1
        # Not a BLOCK — dispatch still runs — but carry a WARN so the
        # caller appends the warning prefix to the real tool result.
        return Decision.warn(cfg.rumination_gate_grace_prefix, reason="rumination_gate.grace")
    state.gate_block_count += 1
    if (cfg.rumination_gate_max_blocks > 0
            and state.gate_block_count >= cfg.rumination_gate_max_blocks):
        return Decision(Action.END, text=cfg.rumination_gate, reason="gate_escalation")
    return Decision.block(cfg.rumination_gate, reason="rumination_gate")


# ─── Per-tool-call post-dispatch guardrails ──────────────────────────────
