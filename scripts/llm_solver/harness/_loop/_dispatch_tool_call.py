"""Per-tool-call dispatch — extracted from ``run_session_loop``'s Phase 6.

The inner for-tc loop body lives here. Sub-phases (done, mutation,
contract, pre_mutation, rumination, dispatch, error ladder, post-
dispatch ladders, observers, dedup, trace) are sequenced in the
same order as the legacy inline code so trace emit order stays byte-equal.

State that crosses tool calls within a turn (``turn_had_pressure``,
``turn_warn_text``) is threaded via ``TurnState``; per-tc transients
(``gate_blocked_flag``, ``rewrite_log``, ``mutation_warn_text``,
``contract_warn_text``) are locals within ``dispatch_one_tool_call``.

The ``dispatch`` callable, ``log``, and ``tool_pre`` / ``tool_post`` /
``observers`` are passed in (rather than imported here) so the
mock-patch contract for ``loop.dispatch`` and ``loop.log`` continues
to apply: ``run_session_loop`` rebinds them via ``loop.dispatch`` at
function entry and hands them through.
"""
from __future__ import annotations

import time

from ..action_metadata import action_metadata
from ..guardrails import Action
from ..._shared.classification import is_error_result
from .._guardrails.extractors import MUTATION_TOOLS
from . import _dedup_signature, _summarize_args, _truncate_for_trace
from ._dispatch_types import TCOutcome, TurnState
from .trace_output import build_tool_call_trace_fields

__all__ = [
    "TCOutcome", "TurnState",
    "dispatch_one_tool_call", "record_tool_start",
]


def _handle_done_tool(tc, state: "TurnState") -> TCOutcome:
    """Resolve the ``done`` tool call's pre-dispatch gate.

    PASS  → session ends with done=True (model_done).
    BLOCK → tool result stored + emitted; caller advances to next tc.
    END   → session ends with done=False (done_loop).
    """
    session = state.session
    done_decision = state.tool_pre["done_guard"](
        session._guards, state.cfg, tc_name=tc.name, cwd=session.cwd,
    )
    if done_decision.action == Action.PASS:
        state.log.info("Model called done() at turn %d", state.turn)
        session.context.add_tool_result(tc.id, "Session ended by model.", tool_name="done")
        _emit_done(tc, state, "Session ended by model.")
        return TCOutcome(end=True, reason="model_done", done=True)
    # BLOCK or END: store rejection text in trace; END terminates.
    session.context.add_tool_result(tc.id, done_decision.text, tool_name="done")
    _emit_done(tc, state, done_decision.text)
    if done_decision.action == Action.END:
        state.log.info("done_guard ended session at turn %d (reason=%s)",
                       state.turn, done_decision.reason)
        return TCOutcome(end=True, reason=done_decision.reason or "done_loop", done=False)
    return TCOutcome(end=False)


def _emit_done(tc, state: "TurnState", result_summary: str) -> None:
    """Emit a tool_call event for the ``done`` short-circuit branches.

    Shared by the PASS (accept) and BLOCK/END branches. ``done`` differs
    from other tool calls: args_summary is the model's user-facing
    ``message``, and ``gate_blocked`` is False even on reject (the
    reject text is the result, not a gate block).
    """
    session = state.session
    cfg = state.cfg
    args_summary = _truncate_for_trace(
        tc.arguments.get("message", ""),
        cfg.trace_args_summary_chars,
    )
    session._emit(
        "tool_call",
        session_number=session._session_number,
        turn_number=state.turn,
        tool_name="done",
        args_summary=args_summary,
        **build_tool_call_trace_fields(
            session,
            tool_name="done",
            args_summary=args_summary,
            result=result_summary,
            turn=state.turn,
            gate_blocked=False,
        ),
        reasoning=_truncate_for_trace(state.content or "", cfg.trace_reasoning_store_chars),
        gate_blocked=False,
        prompt_tokens=state.prompt_tokens,
        completion_tokens=state.completion_tokens,
    )


def _emit_gate_block(tc, decision, state: "TurnState", args_summary: str) -> None:
    """Add a gate-blocked tool result + emit its tool_call trace event.

    Shared by the 4 sub-phase early-exit sites (mutation_repeat_guard
    END, contract_gate END, pre_mutation_gate BLOCK, rumination_gate
    END). The emit shape is byte-identical to the legacy inline code
    so trace ordering remains state.json-projection compatible.
    """
    session = state.session
    cfg = state.cfg
    trace_args_summary = _summarize_args(tc.arguments, cfg.trace_args_summary_chars)
    metadata = action_metadata(tc.name, tc.arguments)
    session.context.add_tool_result(tc.id, decision.text,
                                    tool_name=tc.name, gate_blocked=True)
    session._emit(
        "tool_call",
        session_number=session._session_number,
        turn_number=state.turn,
        tool_name=tc.name,
        args_summary=_truncate_for_trace(trace_args_summary, cfg.trace_args_summary_chars),
        **build_tool_call_trace_fields(
            session,
            tool_name=tc.name,
            args_summary=_truncate_for_trace(trace_args_summary, cfg.trace_args_summary_chars),
            result=decision.text,
            turn=state.turn,
            gate_blocked=True,
            metadata=metadata,
        ),
        reasoning=_truncate_for_trace(state.content or "", cfg.trace_reasoning_store_chars),
        gate_blocked=True,
        gate_reason=decision.reason,
        **metadata,
        prompt_tokens=state.prompt_tokens,
        completion_tokens=state.completion_tokens,
    )


def _capture_workspace_checkpoint(tc, state: "TurnState", *, executed: bool) -> None:
    """Capture one host-side workspace checkpoint after an executed call."""
    store = getattr(state.session, "_checkpoint_store", None)
    if store is None:
        return
    from ..workspace_checkpoints import tool_call_needs_checkpoint
    if not tool_call_needs_checkpoint(tc.name, executed=executed):
        return
    checkpoint = store.capture(state.turn)
    state.session._emit(
        "checkpoint",
        session_number=state.session._session_number,
        **checkpoint.trace_fields(),
    )


def _append_lsp_diagnostics(tc, state: "TurnState", result: str) -> str:
    """Run automatic diagnostics after a successful edit or write."""
    if tc.name not in {"edit", "write"} or is_error_result(result):
        return result
    manager = getattr(state.session, "_lsp_manager", None)
    if manager is None:
        return result
    report = manager.after_edit(str(tc.arguments.get("path", "")))
    from ..lsp_support import append_diagnostics_to_tool_result
    return append_diagnostics_to_tool_result(
        result,
        report,
        max_output_chars=state.cfg.max_output_chars,
        tool_name=tc.name,
    )


def _handle_schema_reject(tc, state: "TurnState", validation) -> TCOutcome:
    """Record one non-executed, repairable schema rejection."""
    session = state.session
    cfg = state.cfg
    result = validation.error_envelope()
    session._emit(
        "schema_reject",
        session_number=session._session_number,
        turn_number=state.turn,
        **validation.trace_fields(),
    )
    error_decision = state.tool_post["error_ladder"](
        session._guards,
        cfg,
        tc_name=tc.name,
        result=result,
    )
    state.turn_had_pressure = True
    if error_decision.action == Action.WARN:
        result += "\n\n" + error_decision.text

    trace_args = _truncate_for_trace(
        _summarize_args(tc.arguments, cfg.trace_args_summary_chars),
        cfg.trace_args_summary_chars,
    )
    metadata = action_metadata(tc.name, tc.arguments)
    session.context.add_tool_result(
        tc.id,
        result,
        tool_name=tc.name,
        gate_blocked=True,
    )
    session._emit(
        "tool_call",
        tool_call_id=tc.id,
        session_number=session._session_number,
        turn_number=state.turn,
        tool_name=tc.name,
        args_summary=trace_args,
        **build_tool_call_trace_fields(
            session,
            tool_name=tc.name,
            args_summary=trace_args,
            result=result,
            turn=state.turn,
            gate_blocked=True,
            metadata=metadata,
            execution_metadata={"executed": False},
        ),
        reasoning=_truncate_for_trace(
            state.content or "", cfg.trace_reasoning_store_chars
        ),
        gate_blocked=True,
        gate_reason="schema_reject",
        **metadata,
        prompt_tokens=state.prompt_tokens,
        completion_tokens=state.completion_tokens,
        tool_dispatch_ms=0.0,
    )
    session._observe_harness_tool_result(
        turn=state.turn,
        tool_name=tc.name,
        tool_args=tc.arguments,
        result=result,
        gate_blocked=True,
    )
    if error_decision.action == Action.END:
        return TCOutcome(
            end=True,
            reason=error_decision.reason,
            done=False,
        )
    return TCOutcome(end=False)


def _handle_permission_denial(tc, state: "TurnState", resolution) -> TCOutcome:
    """Record one policy-denied call without entering any handler or quirk."""
    session = state.session
    cfg = state.cfg
    result = resolution.denial_envelope()
    error_decision = state.tool_post["error_ladder"](
        session._guards,
        cfg,
        tc_name=tc.name,
        result=result,
    )
    state.turn_had_pressure = True
    if error_decision.action == Action.WARN:
        result += "\n\n" + error_decision.text

    trace_args = _truncate_for_trace(
        _summarize_args(tc.arguments, cfg.trace_args_summary_chars),
        cfg.trace_args_summary_chars,
    )
    metadata = action_metadata(tc.name, tc.arguments)
    session.context.add_tool_result(
        tc.id,
        result,
        tool_name=tc.name,
        gate_blocked=True,
    )
    session._emit(
        "tool_call",
        tool_call_id=tc.id,
        session_number=session._session_number,
        turn_number=state.turn,
        tool_name=tc.name,
        args_summary=trace_args,
        **build_tool_call_trace_fields(
            session,
            tool_name=tc.name,
            args_summary=trace_args,
            result=result,
            turn=state.turn,
            gate_blocked=True,
            metadata=metadata,
            execution_metadata={"executed": False},
        ),
        reasoning=_truncate_for_trace(
            state.content or "", cfg.trace_reasoning_store_chars
        ),
        gate_blocked=True,
        gate_reason="permission_denied",
        **metadata,
        prompt_tokens=state.prompt_tokens,
        completion_tokens=state.completion_tokens,
        tool_dispatch_ms=0.0,
    )
    session._observe_harness_tool_result(
        turn=state.turn,
        tool_name=tc.name,
        tool_args=tc.arguments,
        result=result,
        gate_blocked=True,
    )
    if error_decision.action == Action.END:
        return TCOutcome(end=True, reason=error_decision.reason, done=False)
    return TCOutcome(end=False)


def record_tool_start(tc, state: "TurnState") -> None:
    """Make one bounded ``tool_start`` row durable before dispatch."""
    diagnostics = getattr(state.session, "_exit_diagnostics", None)
    if diagnostics is None:
        return
    diagnostics.record_tool_start(
        tool_call_id=tc.id,
        tool_name=tc.name,
        turn_number=state.turn,
        args_summary=_truncate_for_trace(
            _summarize_args(
                tc.arguments, state.cfg.trace_args_summary_chars
            ),
            state.cfg.trace_args_summary_chars,
        ),
        intent=_truncate_for_trace(
            state.content or "", state.cfg.trace_reasoning_store_chars
        ),
    )


def _record_tool_finished(tc, state: "TurnState") -> None:
    """Clear pending state only after ordinary result evidence is durable."""
    diagnostics = getattr(state.session, "_exit_diagnostics", None)
    if diagnostics is None:
        return
    writer = getattr(state.session, "_async_trace_writer", None)
    if writer is None:
        raise RuntimeError("tool exit diagnostics require an active trace writer")
    writer.barrier()
    if not diagnostics.record_tool_finished(tc.id):
        raise RuntimeError(
            f"tool call {tc.id!r} finished without a pending tool_start"
        )


def dispatch_one_tool_call(tc, state: TurnState) -> TCOutcome:
    """Run all of Phase 6 for one tool call.

    Sub-phases (in fixed order):
      6a. schema validation    — reject before every handler and guard
      6b. permission policy    — deny before every handler and quirk
      6c. done_guard           — accept→END(done), BLOCK→next tc, END→END
      6d. mutation_repeat      — END / BLOCK / WARN
      6d. contract_gate        — END / BLOCK / WARN
      6d.5 pre_mutation_gate   — BLOCK only (continue to next tc)
      6e. rumination_gate      — END / BLOCK / WARN-grace dispatch
      6e. dispatch (when no gate intercepted)
      6f. error_ladder         — WARN / END
      Post-dispatch projection (bash only, non-error)
      6g. test_read_ladder + rumination_ladder + observers
      Output dedup + mutation-cache reset
      add_tool_result + trace emit
    """
    session = state.session
    turn = state.turn
    cfg = state.cfg
    content = state.content
    prompt_tokens = state.prompt_tokens
    completion_tokens = state.completion_tokens
    dispatch = state.dispatch
    log = state.log
    tool_pre = state.tool_pre
    tool_post = state.tool_post
    observers = state.observers

    # Per-tc phase timing for the bottom emit. Counts dispatch
    # only; gate-blocked tcs leave this at 0.
    _tc_dispatch_ms = 0.0
    args_summary = _summarize_args(tc.arguments, cfg.args_summary_chars)
    trace_args_summary = _summarize_args(tc.arguments, cfg.trace_args_summary_chars)
    metadata = action_metadata(tc.name, tc.arguments)
    from .focus_dedup import _focus_signature
    focus_key, focus_display = _focus_signature(tc, args_summary, session.cwd)
    log.info("turn=%d pt=%d %s(%s)", turn, prompt_tokens, tc.name, args_summary)
    session._tool_log.append((tc.name, args_summary))
    # rewrite_log is populated by dispatch() when bash_quirks
    # rewrites a bash cmd before execution. Hoisted to the top
    # of each tc iteration so the post-dispatch tool_call
    # emit can include `cmd_pre_rewrite` even on branches
    # that re-enter the emit path (gate_intercepted, normal,
    # error_ladder END). Empty list = no rewrite happened.
    rewrite_log: list = []
    dispatch_started = tc.id in state.preexecuted

    if getattr(cfg, "tools_schema_validation", "off") == "reject":
        validation = state.schema_validations.get(tc.id)
        if validation is None:
            validation = session._tool_schema_set.validate(
                tc.name, tc.arguments
            )
        if not validation.valid:
            return _handle_schema_reject(tc, state, validation)

    permission = state.permission_resolutions.get(tc.id)
    if permission is not None and permission.denied:
        return _handle_permission_denial(tc, state, permission)

    # done_guard — accept path ends session; otherwise BLOCK
    # (or END once cfg.done_loop_abort_after rejected calls accumulate).
    if tc.name == "done":
        return _handle_done_tool(tc, state)

    # 6b. mutation_repeat_guard — stop repeated identical edits.
    mutation_decision = tool_pre["mutation_repeat_guard"](
        session._guards, cfg,
        tc_name=tc.name,
        tc_args=tc.arguments,
        focus_display=focus_display,
    )
    mutation_warn_text = ""
    gate_blocked_flag = False
    gate_intercepted = False
    result: str = ""
    execution_metadata: dict = {}
    if mutation_decision.action == Action.END:
        state.turn_had_pressure = True
        _emit_gate_block(tc, mutation_decision, state, args_summary)
        return TCOutcome(end=True, reason=mutation_decision.reason, done=False)
    if mutation_decision.action == Action.BLOCK:
        state.turn_had_pressure = True
        log.info("Mutation repeat guard blocked %s", tc.name)
        result = mutation_decision.text
        gate_blocked_flag = True
        gate_intercepted = True
    elif mutation_decision.action == Action.WARN:
        mutation_warn_text = mutation_decision.text

    # 6c. contract_gate — warn/block broad exploration once a
    # tighter commit/recovery contract is active.
    contract_warn_text = ""
    if not gate_blocked_flag:
        contract_decision = tool_pre["contract_gate"](
            session._guards, cfg,
            tc_name=tc.name,
            tc_args=tc.arguments,
            focus_key=focus_key,
            focus_display=focus_display,
        )
        if contract_decision.action == Action.END:
            state.turn_had_pressure = True
            _emit_gate_block(tc, contract_decision, state, args_summary)
            return TCOutcome(end=True, reason=contract_decision.reason, done=False)
        if contract_decision.action == Action.BLOCK:
            state.turn_had_pressure = True
            log.info("Contract gate blocked %s", tc.name)
            result = contract_decision.text
            gate_blocked_flag = True
            gate_intercepted = True
        elif contract_decision.action == Action.WARN:
            contract_warn_text = contract_decision.text

    if not gate_blocked_flag:
        if mutation_warn_text and contract_warn_text:
            contract_warn_text = mutation_warn_text + "\n" + contract_warn_text
        elif mutation_warn_text:
            contract_warn_text = mutation_warn_text

        # 6c.5 pre_mutation_gate — orientation budget. Block
        # non-write tool calls past the cap until a mutation happens.
        pre_mut_decision = tool_pre["pre_mutation_gate"](
            session._guards, cfg,
            tc_name=tc.name,
            tc_args=tc.arguments,
            turn_number=turn,
        )
        if pre_mut_decision.action == Action.BLOCK:
            state.turn_had_pressure = True
            log.info("pre_mutation_gate blocked %s at turn %d (cap=%d)",
                     tc.name, turn, cfg.pre_mutation_turn_cap)
            _emit_gate_block(tc, pre_mut_decision, state, args_summary)
            return TCOutcome(end=False)

        # 6d. rumination_gate — grace (WARN+dispatch) / BLOCK / END.
        gate_decision = tool_pre["rumination_gate"](
            session._guards, cfg, tc_name=tc.name, tc_args=tc.arguments
        )
        if gate_decision.action == Action.END:
            state.turn_had_pressure = True
            log.info("Gate escalation: %d blocks, ending session",
                     session._guards.gate_block_count)
            _emit_gate_block(tc, gate_decision, state, args_summary)
            return TCOutcome(end=True, reason=gate_decision.reason, done=False)
        if gate_decision.action == Action.BLOCK:
            state.turn_had_pressure = True
            log.info("Rumination gate blocked %s", tc.name)
            result = gate_decision.text
            gate_blocked_flag = True
            gate_intercepted = True
        elif gate_decision.action == Action.WARN:
            # GRACE: dispatch + append gate warning prefix
            log.info("Rumination gate grace used for %s (%d remaining)",
                     tc.name, session._guards.rumination_gate_grace)
            effective_output_control = (
                session.output_control if cfg.bash_transforms_task_format_enabled else None
            )
            effective_universal_rewrites = (
                session.universal_rewrites if cfg.bash_transforms_universal_enabled else None
            )
            result = state.preexecuted.get(tc.id)
            if result is None:
                record_tool_start(tc, state)
                dispatch_started = True
                _disp_t0 = time.perf_counter()
                result = dispatch(tc.name, tc.arguments, cwd=session.cwd, cfg=cfg,
                                  output_control=effective_output_control,
                                  universal_rewrites=effective_universal_rewrites,
                                  forbidden_rules=session.forbidden_rules if cfg.bash_quirks_forbidden_enabled else None,
                                  redirect_rules=getattr(session, "redirect_rules", None),
                                  redactions=session.redactions,
                                  tool_registry=session._tool_registry,
                                  stale_guard=session._stale_guard,
                                  active_tools=getattr(session, "active_tool_names", ()),
                                  redirect_event_sink=getattr(session, "_redirect_event_sink", None),
                                  ignore_policy=session._ignore_policy,
                                  effective_env=session._effective_env,
                                  allow_login_shell=session._allow_login_shell,
                                  rewrite_log=rewrite_log,
                                  execution_metadata=execution_metadata)
                _tc_dispatch_ms += (time.perf_counter() - _disp_t0) * 1000
            result = _append_lsp_diagnostics(tc, state, result)
            result += "\n\n" + gate_decision.text
            gate_intercepted = True
        else:
            # 6d. Dispatch.
            effective_output_control = (
                session.output_control if cfg.bash_transforms_task_format_enabled else None
            )
            effective_universal_rewrites = (
                session.universal_rewrites if cfg.bash_transforms_universal_enabled else None
            )
            result = state.preexecuted.get(tc.id)
            if result is None:
                record_tool_start(tc, state)
                dispatch_started = True
                _disp_t0 = time.perf_counter()
                result = dispatch(tc.name, tc.arguments, cwd=session.cwd, cfg=cfg,
                                  output_control=effective_output_control,
                                  universal_rewrites=effective_universal_rewrites,
                                  forbidden_rules=session.forbidden_rules if cfg.bash_quirks_forbidden_enabled else None,
                                  redirect_rules=getattr(session, "redirect_rules", None),
                                  redactions=session.redactions,
                                  tool_registry=session._tool_registry,
                                  stale_guard=session._stale_guard,
                                  active_tools=getattr(session, "active_tool_names", ()),
                                  redirect_event_sink=getattr(session, "_redirect_event_sink", None),
                                  ignore_policy=session._ignore_policy,
                                  effective_env=session._effective_env,
                                  allow_login_shell=session._allow_login_shell,
                                  rewrite_log=rewrite_log,
                                  execution_metadata=execution_metadata)
                _tc_dispatch_ms += (time.perf_counter() - _disp_t0) * 1000
            result = _append_lsp_diagnostics(tc, state, result)
            if tc.name == "bash" and not is_error_result(result):
                session._observe_test_signal(tc.arguments.get("cmd", ""), result)
            # 6e. error_ladder (WARN / END tiers). Log every error
            # for trace visibility; the ladder decides escalation.
            err_decision = tool_post["error_ladder"](
                session._guards, cfg, tc_name=tc.name, result=result
            )
            if is_error_result(result):
                state.turn_had_pressure = True
                log.info("Tool error: %s consecutive=%d",
                         tc.name, session._guards.consecutive_errors.get(tc.name, 0))
            if err_decision.action == Action.END:
                state.turn_had_pressure = True
                log.warning("Error abort: %s consecutive=%d", tc.name,
                            session._guards.consecutive_errors.get(tc.name, 0))
                session.context.add_tool_result(tc.id, result, tool_name=tc.name,
                                                cmd_signature="", gate_blocked=False)
                # cmd_pre_rewrite preserves the model's original
                # bash cmd when bash_quirks rewrote it before
                # execution. The field is
                # only added when a rewrite actually fired —
                # rewrite_log stays empty when the cmd was
                # passed through verbatim.
                session._emit(
                    "tool_call",
                    tool_call_id=tc.id,
                    session_number=session._session_number,
                    turn_number=turn,
                    tool_name=tc.name,
                    args_summary=_truncate_for_trace(trace_args_summary, cfg.trace_args_summary_chars),
                    **build_tool_call_trace_fields(
                        session,
                        tool_name=tc.name,
                        args_summary=_truncate_for_trace(trace_args_summary, cfg.trace_args_summary_chars),
                        result=result,
                        turn=turn,
                        gate_blocked=False,
                        metadata=metadata,
                        execution_metadata=execution_metadata,
                    ),
                    reasoning=_truncate_for_trace(content or "", cfg.trace_reasoning_store_chars),
                    gate_blocked=False,
                    **metadata,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    **({"cmd_pre_rewrite": _truncate_for_trace(rewrite_log[0]["original"], cfg.trace_args_summary_chars)} if rewrite_log else {}),
                )
                _capture_workspace_checkpoint(
                    tc, state,
                    executed=bool(execution_metadata.get("executed", True)),
                )
                if dispatch_started:
                    _record_tool_finished(tc, state)
                return TCOutcome(end=True, reason=err_decision.reason, done=False)
            if err_decision.action == Action.WARN:
                result += "\n\n" + err_decision.text

            # Post-dispatch: structured output projection + sink.
            # Only for bash, only on non-error results, only when
            # the relevant cfg flag is on. Sink writes raw output
            # to .tool_output/<session>_<N>_t<turn>.log; the
            # model can read the file when it wants the full
            # content. Structured projection replaces raw with a
            # compact digest for test commands.
            if tc.name == "bash" and not is_error_result(result):
                cmd = tc.arguments.get("cmd", "")
                result = session._project_and_sink(tc.name, cmd, result, turn)

        if contract_warn_text and not gate_blocked_flag:
            result += "\n\n" + contract_warn_text

        # In tool_result delivery mode, append the queued note to the first
        # unblocked tool result and then clear it.
        _pending_tool_note = getattr(session, "_adaptive_tool_note_pending", None)
        if _pending_tool_note and not gate_blocked_flag:
            result += "\n\n" + _pending_tool_note
            session._adaptive_tool_note_pending = None
            state.log.info("adaptive_tool_note: appended callout to %s result at turn %d",
                           tc.name, turn)

    # 6f. rumination_ladder (WARN + ARM). Runs for every tc —
    # it owns the counter increment, nudge emission, and gate
    # arming. When the gate already intercepted this call, the
    # ladder skips the counter bump to avoid double-counting.
    test_read_decision = tool_post["test_read_ladder"](
        session._guards, cfg,
        tc_name=tc.name, result=result,
        gate_blocked=gate_blocked_flag,
        tc_args=tc.arguments,
    )
    if test_read_decision.action == Action.WARN:
        result += "\n\n" + test_read_decision.text

    rum_decision = tool_post["rumination_ladder"](
        session._guards, cfg,
        tc_name=tc.name, result=result,
        gate_blocked=gate_blocked_flag,
        already_blocked_this_turn=gate_intercepted,
        tc_args=tc.arguments,
        focus_key=focus_key,
        focus_display=focus_display,
    )
    if rum_decision.action == Action.WARN:
        result += "\n\n" + rum_decision.text

    # Context-side dedup reset on a successful write/edit. The
    # guardrail state is reset inside rumination_ladder; this is
    # the context's own signal (separate concern — stateful
    # compaction, not thrash control).
    if (tc.name in ("write", "edit")
            and not is_error_result(result)
            and hasattr(session.context, "reset_dedup_counts")):
        session.context.reset_dedup_counts()

    # Content-blind "verified since mutation" signal for the
    # done guard.
    observers["mark_bash_verified"](
        session._guards, cfg,
        tc_name=tc.name, result=result,
        gate_blocked=gate_blocked_flag,
    )
    observers["observe_test_file_read"](
        session._guards, cfg,
        tc_name=tc.name, result=result,
        gate_blocked=gate_blocked_flag,
        tc_args=tc.arguments,
        focus_key=focus_key,
        focus_display=focus_display,
    )
    observers["observe_contract_state"](
        session._guards, cfg,
        tc_name=tc.name, result=result,
        gate_blocked=gate_blocked_flag,
        tc_args=tc.arguments,
        focus_key=focus_key,
        focus_display=focus_display,
    )

    # Append turn-level WARN (from duplicate ladder) after all
    # tc-level appends so it reads last.
    if state.turn_warn_text:
        result += "\n\n" + state.turn_warn_text

    # 6f. Trace + record. cmd_pre_rewrite is added when bash_quirks
    # rewrites the model's cmd before execution. The
    # trace then preserves both what the model wrote and what
    # actually ran, so historical replays survive bash_quirks
    # rule changes. rewrite_log was hoisted to the per-tc loop
    # top — empty list here means no rewrite happened.
    _pre = rewrite_log[0]["original"] if rewrite_log else None
    _turn_total_ms = (time.perf_counter() - state.turn_t0) * 1000
    # Invisible workspace snapshot after an EXECUTED source-write turn:
    # records a rewind/branch point for this turn. Gate-blocked calls
    # never executed, so nothing changed on disk — no snapshot.
    _snapshot_sha = None
    call_executed = (
        not gate_blocked_flag
        and bool(execution_metadata.get("executed", True))
    )
    if (getattr(cfg, "turn_snapshots_enabled", False)
            and metadata.get("source_write_like") and call_executed):
        from ..turn_snapshots import snapshot as _turn_snapshot
        _snapshot_sha = _turn_snapshot(session.cwd, turn, session=session)
    session._emit(
        "tool_call",
        tool_call_id=tc.id,
        session_number=session._session_number,
        turn_number=turn,
        tool_name=tc.name,
        args_summary=_truncate_for_trace(trace_args_summary, cfg.trace_args_summary_chars),
        **build_tool_call_trace_fields(
            session,
            tool_name=tc.name,
            args_summary=_truncate_for_trace(trace_args_summary, cfg.trace_args_summary_chars),
            result=result,
            turn=turn,
            gate_blocked=gate_blocked_flag,
            metadata=metadata,
            execution_metadata=execution_metadata,
        ),
        **({"snapshot_sha": _snapshot_sha} if _snapshot_sha else {}),
        reasoning=_truncate_for_trace(content or "", cfg.trace_reasoning_store_chars),
        gate_blocked=gate_blocked_flag,
        **metadata,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        chat_call_ms=round(state.phase_chat_ms, 2),
        token_count_ms=round(state.phase_token_ms, 2),
        tool_dispatch_ms=round(_tc_dispatch_ms, 2),
        trace_write_ms=round(getattr(session, "_turn_trace_write_ms", 0.0), 2),
        turn_total_ms=round(_turn_total_ms, 2),
        **({"cmd_pre_rewrite": _truncate_for_trace(_pre, cfg.trace_args_summary_chars)} if _pre else {}),
        **({"rewrite_rules": rewrite_log[0].get("rules", [])} if rewrite_log else {}),
    )
    _capture_workspace_checkpoint(
        tc, state, executed=call_executed,
    )
    if dispatch_started:
        _record_tool_finished(tc, state)

    # Gate-blocked calls were never executed — don't store a
    # cmd_signature (prevents later calls of the same command
    # from matching against a non-execution).
    cmd_sig = ""
    if call_executed:
        cmd_sig = _dedup_signature(tc)[1] if tc.name == "bash" else ""

    # Collapse byte-identical repeats per (tool_name, focus_key) into a
    # one-line back-reference. Mutation success below clears
    # the cache so post-edit re-reads always pass through.
    if (
        cfg.tools_output_dedup_enabled
        and not gate_blocked_flag
        and focus_key
        and tc.name not in MUTATION_TOOLS
        and result
    ):
        import hashlib as _hashlib
        digest = _hashlib.sha1(result.encode("utf-8", errors="ignore")).hexdigest()[:12]
        key = (tc.name, focus_key)
        prev = session._output_dedup_cache.get(key)
        if prev is not None and prev[0] == digest:
            original_chars = len(result)
            result = (
                f"[harness: identical to turn {prev[1]}'s "
                f"{tc.name} output for {focus_display or focus_key}]"
            )
            from ..savings import get_ledger as _get_ledger
            _get_ledger().record(
                bucket="output_dedup",
                layer="harness",
                mechanism=tc.name,
                input_chars=original_chars,
                output_chars=len(result),
                measure_type="exact",
                ctx={
                    "turn": turn,
                    "prior_turn": prev[1],
                    "focus": focus_display or focus_key,
                },
            )
        else:
            session._output_dedup_cache[key] = (digest, turn)

    # Reset the dedup cache on a successful mutation so the
    # next read of the mutated file isn't collapsed against
    # pre-mutation bytes.
    if (
        tc.name in MUTATION_TOOLS
        and not is_error_result(result)
        and not gate_blocked_flag
    ):
        session._output_dedup_cache.clear()

    # System-log observation (harness/system_log.py): records which
    # command produced this result (for pre-flight overflow
    # attribution) and emits density_blowout / oversized_result
    # anomaly events when a tokenizer is bound. Content-blind — only
    # binary + flags leave this call; no-op when no system log is open.
    from ..system_log import observe_tool_result
    observe_tool_result(session, tc.id, tc.name, tc.arguments, result,
                        quirk_hit=bool(rewrite_log), turn=turn)

    session.context.add_tool_result(tc.id, result, tool_name=tc.name,
                                    cmd_signature=cmd_sig,
                                    gate_blocked=gate_blocked_flag)
    session._observe_harness_tool_result(
        turn=turn,
        tool_name=tc.name,
        tool_args=tc.arguments,
        result=result,
        gate_blocked=gate_blocked_flag,
    )
    return TCOutcome(end=False)
