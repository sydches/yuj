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
from ..guardrails import PASS, Action
from ..tool_loading import inactive_tool_error
from ..._shared.classification import is_error_result
from .._guardrails.extractors import MUTATION_TOOLS
from ..security_scan import security_block_stage
from . import _dedup_signature, _summarize_args, _truncate_for_trace
from ._dispatch_types import TCOutcome, TurnState
from .trace_output import build_tool_call_trace_fields

__all__ = [
    "TCOutcome", "TurnState",
    "dispatch_one_tool_call", "record_tool_start",
]


def _append_hook_context(result: str, *blocks: str) -> str:
    present = [block for block in blocks if block]
    if not present:
        return result
    return result + "\n\n" + "\n\n".join(present)


def _tool_action_metadata(tc, session) -> dict:
    """Return trace metadata with the plan artifact kept non-mutating."""
    if session._plan_mode.is_plan_write(tc.name, tc.arguments):
        return {
            "write_like": False,
            "source_write_like": False,
            "source_write_paths": [],
            "plan_artifact": True,
        }
    return action_metadata(tc.name, tc.arguments)


def _pre_tool_context(tc, state: "TurnState") -> str:
    effect = state.pre_tool_hooks.get(tc.id)
    return effect.context_block() if effect is not None else ""


def _handle_done_tool(tc, state: "TurnState") -> TCOutcome:
    """Resolve the ``done`` tool call's pre-dispatch gate.

    PASS  → session ends with done=True (model_done).
    BLOCK → tool result stored + emitted; caller advances to next tc.
    END   → session ends with done=False (done_loop).
    """
    session = state.session
    pre_context = _pre_tool_context(tc, state)
    done_decision = state.tool_pre["done_guard"](
        session._guards, state.cfg, tc_name=tc.name, cwd=session.cwd,
    )
    if done_decision.action == Action.PASS:
        hook_effect = session._run_hook(
            "done",
            tool_call_id=tc.id,
            tool_name="done",
            tool_args=dict(tc.arguments),
            implicit=False,
        )
        if hook_effect.blocked:
            result = f"ERROR: done hook blocked completion: {hook_effect.reason}"
            result = session._decorate_stream_rule_tool_result(
                tc.id, result, turn=state.turn
            )
            result = _append_hook_context(
                result, pre_context, hook_effect.context_block()
            )
            session.context.add_tool_result(
                tc.id, result, tool_name="done", gate_blocked=True
            )
            _emit_done(
                tc, state, result, gate_blocked=True, gate_reason="hook_block"
            )
            return TCOutcome(end=False)
        state.log.info("Model called done() at turn %d", state.turn)
        session._final_text = str(
            tc.arguments.get("message") or state.content or ""
        )
        result = session._decorate_stream_rule_tool_result(
            tc.id, "Session ended by model.", turn=state.turn
        )
        result = _append_hook_context(
            result, pre_context, hook_effect.context_block()
        )
        session.context.add_tool_result(tc.id, result, tool_name="done")
        _emit_done(tc, state, result)
        return TCOutcome(end=True, reason="model_done", done=True)
    if done_decision.action == Action.REWIND:
        result = session._decorate_stream_rule_tool_result(
            tc.id, done_decision.text, turn=state.turn
        )
        result = _append_hook_context(result, pre_context)
        session.context.add_tool_result(
            tc.id, result, tool_name="done"
        )
        _emit_done(tc, state, result)
        session.request_rewind(
            done_decision.target_turn,
            reason=done_decision.reason or "rewind_on_done_guard",
        )
        return TCOutcome(rewind=True)
    # BLOCK or END: store rejection text in trace; END terminates.
    result = session._decorate_stream_rule_tool_result(
        tc.id, done_decision.text, turn=state.turn
    )
    result = _append_hook_context(result, pre_context)
    session.context.add_tool_result(tc.id, result, tool_name="done")
    _emit_done(tc, state, result)
    if done_decision.action == Action.END:
        state.log.info("done_guard ended session at turn %d (reason=%s)",
                       state.turn, done_decision.reason)
        return TCOutcome(end=True, reason=done_decision.reason or "done_loop", done=False)
    return TCOutcome(end=False)


def _emit_done(
    tc,
    state: "TurnState",
    result_summary: str,
    *,
    gate_blocked: bool = False,
    gate_reason: str = "",
) -> None:
    """Emit a tool_call event for the ``done`` short-circuit branches.

    Shared by the PASS (accept) and BLOCK/END branches. ``done`` differs
    from other tool calls: args_summary is the model's user-facing
    ``message``. Ordinary done-guard rejection remains a result rather than
    a gate block; a lifecycle-hook rejection is explicitly gate-blocked.
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
            gate_blocked=gate_blocked,
        ),
        reasoning=_truncate_for_trace(state.content or "", cfg.trace_reasoning_store_chars),
        gate_blocked=gate_blocked,
        **({"gate_reason": gate_reason} if gate_reason else {}),
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
    metadata = _tool_action_metadata(tc, session)
    result = session._decorate_stream_rule_tool_result(
        tc.id, decision.text, turn=state.turn
    )
    result = _append_hook_context(result, _pre_tool_context(tc, state))
    session.context.add_tool_result(tc.id, result,
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
            result=result,
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


def _handle_pre_rewind(tc, decision, state: "TurnState") -> TCOutcome:
    """Complete the current call and defer rewind to the turn boundary."""
    state.turn_had_pressure = True
    _emit_gate_block(tc, decision, state, "")
    state.session.request_rewind(
        decision.target_turn,
        reason=decision.reason or "rewind_on_guardrail",
    )
    return TCOutcome(rewind=True)


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


def _emit_todos_event(tc, state: "TurnState", execution_metadata: dict) -> None:
    """Persist one successful whole-list replacement as raw trace content."""
    if tc.name != "write_todos" or "todos" not in execution_metadata:
        return
    state.session._emit(
        "todos",
        session_number=state.session._session_number,
        turn_number=state.turn,
        tool_call_id=tc.id,
        todos=execution_metadata["todos"],
    )


def _append_lsp_diagnostics(tc, state: "TurnState", result: str) -> str:
    """Run automatic diagnostics after a successful edit-dialect call."""
    from ..._shared.edit_formats import EDIT_FORMAT_TOOL_NAMES
    if (
        tc.name not in EDIT_FORMAT_TOOL_NAMES | {"structural_edit"}
        or is_error_result(result)
    ):
        return result
    manager = getattr(state.session, "_lsp_manager", None)
    if manager is None:
        return result
    from ..edit_operations import edit_operations
    from ..lsp_support import append_diagnostics_to_tool_result
    for kind, path in edit_operations(tc.name, tc.arguments):
        if kind == "delete":
            continue
        report = manager.after_edit(path)
        result = append_diagnostics_to_tool_result(
            result,
            report,
            max_output_chars=state.cfg.max_output_chars,
            tool_name=tc.name,
        )
    return result


def _run_rejection_error_ladder(tc, state: "TurnState", result: str):
    """Count a pre-dispatch rejection without declaring another phase site.

    Schema and permission failures intentionally feed the ordinary error
    ladder, but they are branches of the single post-tool phase rather than
    additional ordered guardrails. ``dict.get`` keeps the literal phase-order
    inventory in the main dispatch path authoritative.
    """
    ladder = state.tool_post.get("error_ladder")
    if ladder is None:  # registry validation should make this unreachable
        raise RuntimeError("guardrail registry is missing error_ladder")
    return ladder(
        state.session._guards,
        state.cfg,
        tc_name=tc.name,
        result=result,
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
    error_decision = _run_rejection_error_ladder(tc, state, result)
    state.turn_had_pressure = True
    if error_decision.action == Action.WARN:
        result += "\n\n" + error_decision.text
    result = session._decorate_stream_rule_tool_result(
        tc.id, result, turn=state.turn
    )
    result = _append_hook_context(result, _pre_tool_context(tc, state))

    trace_args = _truncate_for_trace(
        _summarize_args(tc.arguments, cfg.trace_args_summary_chars),
        cfg.trace_args_summary_chars,
    )
    metadata = _tool_action_metadata(tc, session)
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
    if error_decision.action == Action.REWIND:
        session.request_rewind(
            error_decision.target_turn,
            reason=error_decision.reason or "rewind_on_error_ladder",
        )
        return TCOutcome(rewind=True)
    return TCOutcome(end=False)


def _handle_inactive_tool(tc, state: "TurnState") -> TCOutcome:
    """Reject a hidden registered tool before policy or execution."""
    session = state.session
    cfg = state.cfg
    result = inactive_tool_error(tc.name)
    error_decision = _run_rejection_error_ladder(tc, state, result)
    state.turn_had_pressure = True
    if error_decision.action == Action.WARN:
        result += "\n\n" + error_decision.text

    result = session._decorate_stream_rule_tool_result(
        tc.id, result, turn=state.turn
    )
    result = _append_hook_context(result, _pre_tool_context(tc, state))

    trace_args = _truncate_for_trace(
        _summarize_args(tc.arguments, cfg.trace_args_summary_chars),
        cfg.trace_args_summary_chars,
    )
    metadata = _tool_action_metadata(tc, session)
    session.context.add_tool_result(
        tc.id, result, tool_name=tc.name, gate_blocked=True
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
        gate_reason="tool_not_active",
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


def _handle_permission_denial(tc, state: "TurnState", resolution) -> TCOutcome:
    """Record one policy-denied call without entering any handler or quirk."""
    session = state.session
    cfg = state.cfg
    result = resolution.denial_envelope()
    error_decision = _run_rejection_error_ladder(tc, state, result)
    state.turn_had_pressure = True
    if error_decision.action == Action.WARN:
        result += "\n\n" + error_decision.text
    result = session._decorate_stream_rule_tool_result(
        tc.id, result, turn=state.turn
    )
    result = _append_hook_context(result, _pre_tool_context(tc, state))

    trace_args = _truncate_for_trace(
        _summarize_args(tc.arguments, cfg.trace_args_summary_chars),
        cfg.trace_args_summary_chars,
    )
    metadata = _tool_action_metadata(tc, session)
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
    if error_decision.action == Action.REWIND:
        session.request_rewind(
            error_decision.target_turn,
            reason=error_decision.reason or "rewind_on_error_ladder",
        )
        return TCOutcome(rewind=True)
    return TCOutcome(end=False)


def _handle_plan_mode_reject(tc, state: "TurnState", message: str) -> TCOutcome:
    """Record an engine-enforced plan-phase rejection as a unified error."""
    session = state.session
    cfg = state.cfg
    from ..plan_mode import render_plan_mode_error

    result = render_plan_mode_error(tc.name, message, cfg.max_output_chars)
    error_decision = _run_rejection_error_ladder(tc, state, result)
    state.turn_had_pressure = True
    if error_decision.action == Action.WARN:
        result += "\n\n" + error_decision.text
    result = session._decorate_stream_rule_tool_result(
        tc.id, result, turn=state.turn
    )
    result = _append_hook_context(result, _pre_tool_context(tc, state))
    trace_args = _truncate_for_trace(
        _summarize_args(tc.arguments, cfg.trace_args_summary_chars),
        cfg.trace_args_summary_chars,
    )
    metadata = _tool_action_metadata(tc, session)
    session.context.add_tool_result(
        tc.id, result, tool_name=tc.name, gate_blocked=True,
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
        gate_reason="plan_mode",
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
    if error_decision.action == Action.REWIND:
        session.request_rewind(
            error_decision.target_turn,
            reason=error_decision.reason or "rewind_on_error_ladder",
        )
        return TCOutcome(rewind=True)
    return TCOutcome(end=False)


def _handle_pre_tool_hook_block(tc, state: "TurnState", effect) -> TCOutcome:
    """Record an exit-2/deny pre-tool outcome without entering the handler."""
    session = state.session
    cfg = state.cfg
    from ..tools import admit_tool_output

    result = admit_tool_output(
        tc.name,
        f"ERROR: pre_tool hook blocked this call: {effect.reason}",
        arguments=tc.arguments,
        cfg=cfg,
        redactions=session.redactions,
        filter_shell_output=False,
    )
    error_decision = _run_rejection_error_ladder(tc, state, result)
    state.turn_had_pressure = True
    if error_decision.action == Action.WARN:
        result += "\n\n" + error_decision.text
    result = session._decorate_stream_rule_tool_result(
        tc.id, result, turn=state.turn
    )
    result = _append_hook_context(result, effect.context_block())

    trace_args = _truncate_for_trace(
        _summarize_args(tc.arguments, cfg.trace_args_summary_chars),
        cfg.trace_args_summary_chars,
    )
    metadata = _tool_action_metadata(tc, session)
    session.context.add_tool_result(
        tc.id, result, tool_name=tc.name, gate_blocked=True
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
        gate_reason="hook_block",
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


def _apply_tool_hook_effects(
    tc, state: "TurnState", result: str
) -> tuple[str, str]:
    """Run post_tool after a real handler and return its context annotation."""
    session = state.session
    effect = session._run_hook(
        "post_tool",
        tool_call_id=tc.id,
        tool_name=tc.name,
        tool_args=dict(tc.arguments),
        result=result,
    )
    if effect.blocked:
        from ..tools import admit_tool_output

        result = admit_tool_output(
            tc.name,
            f"ERROR: post_tool hook blocked this result: {effect.reason}",
            arguments=tc.arguments,
            cfg=state.cfg,
            redactions=session.redactions,
            filter_shell_output=False,
        )
    return result, effect.context_block()


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


def _finish_error_abort(
    tc,
    state: "TurnState",
    *,
    result: str,
    trace_args_summary: str,
    metadata: dict,
    execution_metadata: dict,
    rewrite_log: list,
    dispatch_started: bool,
    tool_dispatch_ms: float,
    decision,
    hook_context: str,
    plan_artifact: bool,
    gate_blocked: bool = False,
) -> TCOutcome:
    """Record the dispatched error that ended the ordinary error ladder."""
    session = state.session
    cfg = state.cfg
    call_executed = bool(execution_metadata.get("executed", True))
    if not plan_artifact:
        result, _ = session._apply_path_injections(
            result,
            tool_name=tc.name,
            arguments=tc.arguments,
            turn_number=state.turn,
            executed=call_executed,
            execution_metadata=execution_metadata,
            bash_rewritten=bool(rewrite_log),
        )
    result = session._decorate_stream_rule_tool_result(
        tc.id, result, turn=state.turn
    )
    result = _append_hook_context(result, hook_context)
    trace_args = _truncate_for_trace(
        trace_args_summary, cfg.trace_args_summary_chars
    )
    session.context.add_tool_result(
        tc.id,
        result,
        tool_name=tc.name,
        cmd_signature="",
        gate_blocked=gate_blocked,
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
            gate_blocked=gate_blocked,
            metadata=metadata,
            execution_metadata=execution_metadata,
        ),
        **_exec_cell_trace_fields(tc, state, execution_metadata),
        reasoning=_truncate_for_trace(
            state.content or "", cfg.trace_reasoning_store_chars
        ),
        gate_blocked=gate_blocked,
        **({"gate_reason": "security_block"} if gate_blocked else {}),
        **metadata,
        prompt_tokens=state.prompt_tokens,
        completion_tokens=state.completion_tokens,
        tool_dispatch_ms=round(tool_dispatch_ms, 2),
        **(
            {
                "cmd_pre_rewrite": _truncate_for_trace(
                    rewrite_log[0]["original"],
                    cfg.trace_args_summary_chars,
                )
            }
            if rewrite_log
            else {}
        ),
    )
    if not plan_artifact:
        _capture_workspace_checkpoint(
            tc,
            state,
            executed=call_executed,
        )
    if dispatch_started:
        _record_tool_finished(tc, state)
    if decision.action == Action.REWIND:
        session.request_rewind(
            decision.target_turn,
            reason=decision.reason or "rewind_on_error_ladder",
        )
        return TCOutcome(rewind=True)
    return TCOutcome(end=True, reason=decision.reason, done=False)


def _exec_cell_trace_fields(tc, state: "TurnState", execution_metadata: dict) -> dict:
    """Emit cell children once and return exact outer-cell trace fields."""
    if tc.name != "exec_cell":
        return {}
    cell = execution_metadata.get("exec_cell")
    if not isinstance(cell, dict):
        return {"cell_source": str(tc.arguments.get("source", ""))}
    if not execution_metadata.get("_exec_cell_children_emitted"):
        for raw_call in cell.get("inner_calls", ()):
            if not isinstance(raw_call, dict):
                continue
            name = str(raw_call.get("name") or "")
            arguments = raw_call.get("arguments")
            if not name or not isinstance(arguments, dict):
                continue
            inner_execution = raw_call.get("execution_metadata")
            if not isinstance(inner_execution, dict):
                inner_execution = {}
            args_summary = _truncate_for_trace(
                _summarize_args(
                    arguments, state.cfg.trace_args_summary_chars
                ),
                state.cfg.trace_args_summary_chars,
            )
            result = str(raw_call.get("result") or "")
            metadata = action_metadata(name, arguments)
            gate_blocked = bool(inner_execution.get("gate_blocked", False))
            index = int(raw_call.get("index") or 0)
            state.session._emit(
                "tool_call",
                tool_call_id=f"{tc.id}:cell:{index}",
                parent_tool_call_id=tc.id,
                cell_inner_index=index,
                session_number=state.session._session_number,
                turn_number=state.turn,
                tool_name=name,
                args_summary=args_summary,
                **build_tool_call_trace_fields(
                    state.session,
                    tool_name=name,
                    args_summary=args_summary,
                    result=result,
                    turn=state.turn,
                    gate_blocked=gate_blocked,
                    metadata=metadata,
                    execution_metadata=inner_execution,
                ),
                reasoning="",
                gate_blocked=gate_blocked,
                **metadata,
                prompt_tokens=0,
                completion_tokens=0,
                tool_dispatch_ms=float(raw_call.get("duration_ms") or 0.0),
            )
        execution_metadata["_exec_cell_children_emitted"] = True
    return {
        "cell_source": str(cell.get("source") or tc.arguments.get("source", "")),
        "combined_output_chars": int(cell.get("combined_output_chars") or 0),
        "combined_output_bytes": int(cell.get("combined_output_bytes") or 0),
        "inner_call_count": len(cell.get("inner_calls", ())),
    }


def dispatch_one_tool_call(tc, state: TurnState) -> TCOutcome:
    """Run all of Phase 6 for one tool call.

    Sub-phases (in fixed order):
      6a. plan_mode            — reject disallowed phase actions uniformly
      6b. lifecycle/schema/permission gates
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
    metadata = _tool_action_metadata(tc, session)
    plan_artifact = bool(metadata.get("plan_artifact"))
    # Pin the phase for the whole model response. A successful exit can alter
    # the next turn's surface, but must not unlock a mutating sibling call.
    plan_phase_call = state.plan_mode_active
    plan_policy_call = (
        plan_phase_call or plan_artifact or tc.name == "exit_plan_mode"
    )
    plan_decision = session._plan_mode.check(
        tc.name, tc.arguments, turn=turn, active=plan_phase_call,
    )
    if not plan_decision.allowed:
        return _handle_plan_mode_reject(tc, state, plan_decision.message)
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

    pre_tool_hook = state.pre_tool_hooks.get(tc.id)
    if pre_tool_hook is not None and pre_tool_hook.blocked:
        return _handle_pre_tool_hook_block(tc, state, pre_tool_hook)
    hook_context = _pre_tool_context(tc, state)
    if tc.id in state.inactive_tool_call_ids:
        return _handle_inactive_tool(tc, state)

    if getattr(cfg, "tools_schema_validation", "off") == "reject":
        validation = state.schema_validations.get(tc.id)
        if validation is None:
            validation = session.tool_schema_set_for_phase(
                plan_mode_active=plan_phase_call
            ).validate(
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
    mutation_decision = PASS if plan_policy_call else tool_pre["mutation_repeat_guard"](
        session._guards, cfg,
        tc_name=tc.name,
        tc_args=tc.arguments,
        focus_display=focus_display,
    )
    mutation_warn_text = ""
    gate_blocked_flag = False
    gate_intercepted = False
    path_injection_fired = False
    result: str = ""
    execution_metadata: dict = {}
    if mutation_decision.action == Action.END:
        state.turn_had_pressure = True
        _emit_gate_block(tc, mutation_decision, state, args_summary)
        return TCOutcome(end=True, reason=mutation_decision.reason, done=False)
    if mutation_decision.action == Action.REWIND:
        return _handle_pre_rewind(tc, mutation_decision, state)
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
        contract_decision = PASS if plan_policy_call else tool_pre["contract_gate"](
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
        if contract_decision.action == Action.REWIND:
            return _handle_pre_rewind(tc, contract_decision, state)
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
        pre_mut_decision = PASS if plan_policy_call else tool_pre["pre_mutation_gate"](
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
        if pre_mut_decision.action == Action.REWIND:
            return _handle_pre_rewind(tc, pre_mut_decision, state)

        # 6d. rumination_gate — grace (WARN+dispatch) / BLOCK / END.
        gate_decision = PASS if plan_policy_call else tool_pre["rumination_gate"](
            session._guards, cfg, tc_name=tc.name, tc_args=tc.arguments
        )
        if gate_decision.action == Action.END:
            state.turn_had_pressure = True
            log.info("Gate escalation: %d blocks, ending session",
                     session._guards.gate_block_count)
            _emit_gate_block(tc, gate_decision, state, args_summary)
            return TCOutcome(end=True, reason=gate_decision.reason, done=False)
        if gate_decision.action == Action.REWIND:
            return _handle_pre_rewind(tc, gate_decision, state)
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
                                  stale_guard=(
                                      None if plan_artifact else session._stale_guard
                                  ),
                                  active_tools=getattr(session, "active_tool_names", ()),
                                  redirect_event_sink=getattr(session, "_redirect_event_sink", None),
                                  security_event_sink=getattr(session, "_security_event_sink", None),
                                  ignore_policy=session._ignore_policy,
                                  effective_env=session._effective_env,
                                  allow_login_shell=session._allow_login_shell,
                                  rewrite_log=rewrite_log,
                                  execution_metadata=execution_metadata)
                _tc_dispatch_ms += (time.perf_counter() - _disp_t0) * 1000
            blocked_stage = security_block_stage(result)
            if blocked_stage == "args":
                gate_blocked_flag = True
                gate_intercepted = True
                state.turn_had_pressure = True
                execution_metadata.setdefault("executed", False)
            if not plan_artifact and blocked_stage != "args":
                result = _append_lsp_diagnostics(tc, state, result)
            if blocked_stage != "args":
                result, post_context = _apply_tool_hook_effects(
                    tc, state, result
                )
                hook_context = "\n\n".join(
                    block for block in (hook_context, post_context) if block
                )
            if blocked_stage is not None:
                security_error_decision = _run_rejection_error_ladder(
                    tc, state, result
                )
                state.turn_had_pressure = True
                if security_error_decision.action in {
                    Action.END, Action.REWIND,
                }:
                    return _finish_error_abort(
                        tc,
                        state,
                        result=result,
                        trace_args_summary=trace_args_summary,
                        metadata=metadata,
                        execution_metadata=execution_metadata,
                        rewrite_log=rewrite_log,
                        dispatch_started=dispatch_started,
                        tool_dispatch_ms=_tc_dispatch_ms,
                        decision=security_error_decision,
                        hook_context=hook_context,
                        plan_artifact=plan_artifact,
                        gate_blocked=gate_blocked_flag,
                    )
                if security_error_decision.action == Action.WARN:
                    result += "\n\n" + security_error_decision.text
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
                                  stale_guard=(
                                      None if plan_artifact else session._stale_guard
                                  ),
                                  active_tools=getattr(session, "active_tool_names", ()),
                                  redirect_event_sink=getattr(session, "_redirect_event_sink", None),
                                  security_event_sink=getattr(session, "_security_event_sink", None),
                                  ignore_policy=session._ignore_policy,
                                  effective_env=session._effective_env,
                                  allow_login_shell=session._allow_login_shell,
                                  rewrite_log=rewrite_log,
                                  execution_metadata=execution_metadata)
                _tc_dispatch_ms += (time.perf_counter() - _disp_t0) * 1000
            blocked_stage = security_block_stage(result)
            if blocked_stage == "args":
                gate_blocked_flag = True
                gate_intercepted = True
                state.turn_had_pressure = True
                execution_metadata.setdefault("executed", False)
            if not plan_artifact and blocked_stage != "args":
                result = _append_lsp_diagnostics(tc, state, result)
            if blocked_stage != "args":
                result, post_context = _apply_tool_hook_effects(
                    tc, state, result
                )
                hook_context = "\n\n".join(
                    block for block in (hook_context, post_context) if block
                )
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
            if err_decision.action in {Action.END, Action.REWIND}:
                state.turn_had_pressure = True
                log.warning("Error abort: %s consecutive=%d", tc.name,
                            session._guards.consecutive_errors.get(tc.name, 0))
                if not plan_artifact:
                    result, _ = session._apply_path_injections(
                        result,
                        tool_name=tc.name,
                        arguments=tc.arguments,
                        turn_number=turn,
                        executed=bool(execution_metadata.get("executed", True)),
                        execution_metadata=execution_metadata,
                        bash_rewritten=bool(rewrite_log),
                    )
                result = session._decorate_stream_rule_tool_result(
                    tc.id, result, turn=turn
                )
                result = _append_hook_context(result, hook_context)
                session.context.add_tool_result(tc.id, result, tool_name=tc.name,
                                                cmd_signature="", gate_blocked=gate_blocked_flag)
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
                        gate_blocked=gate_blocked_flag,
                        metadata=metadata,
                        execution_metadata=execution_metadata,
                    ),
                    **_exec_cell_trace_fields(tc, state, execution_metadata),
                    reasoning=_truncate_for_trace(content or "", cfg.trace_reasoning_store_chars),
                    gate_blocked=gate_blocked_flag,
                    **({"gate_reason": "security_block"} if gate_blocked_flag else {}),
                    **metadata,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    **({"cmd_pre_rewrite": _truncate_for_trace(rewrite_log[0]["original"], cfg.trace_args_summary_chars)} if rewrite_log else {}),
                )
                if not plan_artifact:
                    _capture_workspace_checkpoint(
                        tc, state,
                        executed=bool(execution_metadata.get("executed", True)),
                    )
                if dispatch_started:
                    _record_tool_finished(tc, state)
                if err_decision.action == Action.REWIND:
                    session.request_rewind(
                        err_decision.target_turn,
                        reason=(
                            err_decision.reason
                            or "rewind_on_error_ladder"
                        ),
                    )
                    return TCOutcome(rewind=True)
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
    test_read_decision = PASS if plan_policy_call else tool_post["test_read_ladder"](
        session._guards, cfg,
        tc_name=tc.name, result=result,
        gate_blocked=gate_blocked_flag,
        tc_args=tc.arguments,
    )
    if test_read_decision.action == Action.WARN:
        result += "\n\n" + test_read_decision.text

    rum_decision = PASS if plan_policy_call else tool_post["rumination_ladder"](
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
    post_rewind = next(
        (
            decision
            for decision in (test_read_decision, rum_decision)
            if decision.action == Action.REWIND
        ),
        None,
    )

    # Context-side dedup reset on a successful write/edit. The
    # guardrail state is reset inside rumination_ladder; this is
    # the context's own signal (separate concern — stateful
    # compaction, not thrash control).
    if (tc.name in MUTATION_TOOLS
            and not plan_artifact
            and not is_error_result(result)
            and hasattr(session.context, "reset_dedup_counts")):
        session.context.reset_dedup_counts()

    # Content-blind "verified since mutation" signal for the
    # done guard.
    if not plan_policy_call:
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

    path_call_executed = (
        not gate_blocked_flag
        and bool(execution_metadata.get("executed", True))
    )
    if not plan_artifact:
        result, path_injection_fired = session._apply_path_injections(
            result,
            tool_name=tc.name,
            arguments=tc.arguments,
            turn_number=turn,
            executed=path_call_executed,
            execution_metadata=execution_metadata,
            bash_rewritten=bool(rewrite_log),
        )
    result = session._decorate_stream_rule_tool_result(
        tc.id, result, turn=turn
    )

    # Hook annotations are context, not tool output. Append them only after
    # output projection and content-blind guardrail observation so they cannot
    # perturb parsing, pass/fail classification, path rules, stream rules, or
    # tool-state counters.
    result = _append_hook_context(result, hook_context)

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
        **_exec_cell_trace_fields(tc, state, execution_metadata),
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
    if call_executed:
        _emit_todos_event(tc, state, execution_metadata)
    if not plan_artifact:
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
        and not path_injection_fired
        and focus_key
        and tc.name not in MUTATION_TOOLS
        and tc.id not in session._stream_rule_decorated_call_ids
        and result
        and not hook_context
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
        and not plan_artifact
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
    if not plan_artifact:
        session._observe_harness_tool_result(
            turn=turn,
            tool_name=tc.name,
            tool_args=tc.arguments,
            result=result,
            gate_blocked=gate_blocked_flag,
        )
    if post_rewind is not None:
        session.request_rewind(
            post_rewind.target_turn,
            reason=post_rewind.reason or "rewind_on_guardrail",
        )
        return TCOutcome(rewind=True)
    return TCOutcome(end=False)
