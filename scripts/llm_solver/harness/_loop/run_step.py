"""Session.run() body — extracted as a free function from harness/loop.py.

This file is intentionally large (~670 lines): the per-turn loop is a
single coherent state machine with many decision branches and many
early-return points, and PR #7 deliberately relocated it without
decomposing further. A future PR may split this into per-turn phases
(pre-flight / API call / intent gate / dispatch / post-dispatch /
adaptive). For now, the goal was to get loop.py under the 500-line gate
without touching the loop's logic.

Mock-patch contract: tests do ``patch("llm_solver.harness.loop.dispatch", ...)``
to fake the dispatcher. That patch only affects the ``dispatch`` attribute
on the ``loop`` module, so this file looks ``dispatch`` up via
``loop.dispatch`` (rebound at function entry) instead of importing it
directly. Same pattern as ``_run_in_sandbox`` in PR #1.
"""
from __future__ import annotations
from .._guardrails.state import Decision

import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import TYPE_CHECKING

from ..guardrails import Action, PASS
from ..injections import UserTurnInjection
from ..checkpoint_rewind import finalize_deferred_context_actions
from ..action_metadata import action_metadata
from ..approvals import approval_decision, approval_transport_available
from ..clarifications import (
    clarification_state,
    consume_clarification_answer,
    create_clarification_request,
)
from ..corrections import consume_correction
from ..plan_mode import effective_model_tool_schemas
from ..tool_validation import SchemaViolation, ToolArgumentValidation
from .._tool_filters import resolve_tool_permission
from ..system_log import get_system_log, provenance_for
from ...server.request_controls import CacheObservation, warn_on_cache_miss
from . import _dedup_signature, _summarize_args, _truncate_for_trace
from ._dispatch_tool_call import (
    TurnState,
    dispatch_one_tool_call,
    record_tool_start,
)
from .compaction import (
    CompactionOverflowError,
    maybe_compact_messages,
    preflight_reclip_oversized,
)
from .trace_output import build_tool_call_trace_fields

if TYPE_CHECKING:
    from ..loop import Session, SessionResult

log = logging.getLogger(__name__)


def _clarification_schema_reject(session: "Session", tc, turn: int, validation) -> None:
    """Balance one rejected question call without entering dispatch policy."""
    session.context.add_tool_result(
        tc.id,
        validation.error_envelope(),
        tool_name=tc.name,
        gate_blocked=True,
    )
    session._emit(
        "schema_reject",
        session_number=session._session_number,
        turn_number=turn,
        **validation.trace_fields(),
    )


def _balance_skipped_clarification_siblings(
    session: "Session", tool_calls: list, *, ask_ids: frozenset[str]
) -> None:
    for sibling in tool_calls:
        if sibling.id in ask_ids:
            continue
        session.context.add_tool_result(
            sibling.id,
            "NOT RUN: The session stopped before sibling calls because the "
            "same response contained ask_user.",
            tool_name=sibling.name,
            gate_blocked=True,
        )


def _handle_clarification_response(
    session: "Session", tool_calls: list, *, turn: int
) -> str | None:
    """Return ``pause`` or ``continue`` when this is a clarification turn."""
    asks = [call for call in tool_calls if call.name == "ask_user"]
    if not asks:
        return None

    replay_exchange = getattr(session.client, "replay_clarification", None)
    if callable(replay_exchange):
        if len(asks) != 1:
            raise RuntimeError("recorded clarification turn has multiple ask_user calls")
        exchange = replay_exchange(asks[0])
        if not session.context.replace_all_messages(exchange["next_messages"]):
            raise RuntimeError(
                "replay clarification requires a replaceable context manager"
            )
        request = exchange["request"]
        answer = exchange["answer"]
        session._emit(
            "clarification_request",
            session_number=session._session_number,
            turn_number=turn,
            request_id=request["request_id"],
            tool_call_id=request["tool_call_id"],
            question=request["question"],
            replayed=True,
        )
        session._emit(
            "clarification_answer",
            session_number=session._session_number,
            turn_number=turn,
            request_id=request["request_id"],
            answer_sha256=answer["answer_sha256"],
            answer_chars=len(answer["answer"]),
            replayed=True,
        )
        session._emit(
            "clarification_consumed",
            session_number=session._session_number,
            turn_number=turn,
            request_id=request["request_id"],
            answer_sha256=answer["answer_sha256"],
            delivery="replay",
            replayed=True,
        )
        return "continue"

    if (
        getattr(session.cfg, "runtime_mode", "measurement") != "assistant"
        or getattr(session, "_subagent_level", 0) != 0
    ):
        return None

    if len(asks) != 1:
        validation = ToolArgumentValidation(
            "ask_user",
            (SchemaViolation(
                path="$.tool_calls",
                keyword="maxItems",
                message="one model response may contain only one ask_user call",
                expected="1",
                actual=str(len(asks)),
            ),),
        )
        for ask in asks:
            _clarification_schema_reject(session, ask, turn, validation)
        _balance_skipped_clarification_siblings(
            session, tool_calls, ask_ids=frozenset(ask.id for ask in asks)
        )
        return "continue"

    ask = asks[0]
    validation = session.tool_schema_set_for_phase(
        plan_mode_active=bool(session._plan_mode.active)
    ).validate(ask.name, ask.arguments)
    if not validation.valid:
        _clarification_schema_reject(session, ask, turn, validation)
        _balance_skipped_clarification_siblings(
            session, tool_calls, ask_ids=frozenset({ask.id})
        )
        return "continue"

    prior = clarification_state(session._artifact_dir)
    if prior.phase != "none":
        session.context.add_tool_result(
            ask.id,
            "ERROR: This session already used its one clarification request.",
            tool_name=ask.name,
            gate_blocked=True,
        )
        _balance_skipped_clarification_siblings(
            session, tool_calls, ask_ids=frozenset({ask.id})
        )
        session._emit(
            "clarification_rejected",
            session_number=session._session_number,
            turn_number=turn,
            tool_call_id=ask.id,
            reason="request_already_exists",
        )
        return "continue"

    request = create_clarification_request(
        session._artifact_dir,
        request_id=f"clar-{uuid.uuid4().hex[:12]}",
        session_id=session._artifact_dir.name,
        session_number=session._session_number,
        turn_number=turn,
        tool_call_id=ask.id,
        question=ask.arguments["question"],
    )
    session.context.add_tool_result(
        ask.id,
        "INPUT REQUIRED: The session is paused until the operator records "
        "one answer and resumes it.",
        tool_name=ask.name,
        gate_blocked=True,
    )
    _balance_skipped_clarification_siblings(
        session, tool_calls, ask_ids=frozenset({ask.id})
    )
    session._emit(
        "clarification_request",
        session_number=session._session_number,
        turn_number=turn,
        request_id=request["request_id"],
        tool_call_id=ask.id,
        question=request["question"],
    )
    return "pause"


def _consume_pending_clarification(session: "Session", *, turn: int) -> None:
    pending = getattr(session, "_pending_clarification_delivery", None)
    if pending is None:
        return
    consumption = consume_clarification_answer(
        session._artifact_dir,
        request_id=pending["request_id"],
        session_number=session._session_number,
        turn_number=turn,
        delivery="resume",
    )
    session._emit(
        "clarification_consumed",
        session_number=session._session_number,
        turn_number=turn,
        request_id=consumption["request_id"],
        answer_sha256=consumption["answer_sha256"],
        delivery=consumption["delivery"],
    )
    session._pending_clarification_delivery = None


def _consume_pending_correction(session: "Session", *, turn: int) -> None:
    pending = getattr(session, "_pending_correction_delivery", None)
    if pending is None:
        return
    if not pending.get("injected"):
        raise RuntimeError("pending correction was not added before consumption")
    _require_exact_final_user_correction(session, pending["text"])
    consumption = consume_correction(
        session._artifact_dir,
        correction_id=pending["correction_id"],
        session_number=session._session_number,
        turn_number=turn,
        delivery="resume",
    )
    session._emit(
        "correction_consumed",
        session_number=session._session_number,
        turn_number=turn,
        correction_id=consumption["correction_id"],
        text_sha256=consumption["text_sha256"],
        transcript_segment=consumption["transcript_segment"],
        delivery=consumption["delivery"],
    )
    _correction_trace_barrier(session)
    session._pending_correction_delivery = None


def _inject_pending_correction(session: "Session") -> None:
    pending = getattr(session, "_pending_correction_delivery", None)
    if pending is None or pending.get("injected"):
        return
    # Keep the exact operator bytes as their own final user message. They are
    # ordinary task input, not permission or a structured clarification
    # answer. Delay this step until after resume-time rewind restoration and
    # pre-model hooks so neither can replace or follow the correction.
    preserve_image_target = getattr(
        session.client, "preserve_image_target_before_correction", None
    )
    if callable(preserve_image_target):
        preserve_image_target(pending["text"])
    session.context.add_user(pending["text"])
    session._protected_correction_text = pending["text"]
    pending["injected"] = True


def _require_exact_final_user_correction(
    session: "Session", text: str
) -> None:
    user_messages = [
        message
        for message in session.context.get_messages()
        if message.get("role") == "user"
    ]
    if not user_messages or user_messages[-1].get("content") != text:
        raise RuntimeError(
            "pending correction changed before its transport boundary"
        )


def _correction_trace_barrier(session: "Session") -> None:
    writer = getattr(session, "_async_trace_writer", None)
    if writer is not None:
        writer.barrier()
        return
    trace_file = getattr(session, "_trace_file", None)
    if trace_file is None:
        return
    trace_file.flush()
    try:
        os.fsync(trace_file.fileno())
    except (AttributeError, OSError):
        # In-memory test sinks have no kernel durability boundary.
        pass


def _apply_replay_correction(session: "Session", *, turn: int) -> None:
    if getattr(session, "_pending_replay_correction", None) is not None:
        return
    if getattr(session.client, "is_replay", False) is not True:
        return
    replay_correction = getattr(
        session.client, "replay_correction_before_next_request", None
    )
    if not callable(replay_correction):
        return
    exchange = replay_correction()
    if exchange is None:
        return
    correction = exchange["correction"]
    messages = session.context.get_messages()
    if (
        not messages
        or messages[-1].get("role") != "user"
        or messages[-1].get("content") != correction["text"]
    ):
        session.context.add_user(correction["text"])
    session._protected_correction_text = correction["text"]
    session._pending_replay_correction = exchange


def _consume_replay_correction(session: "Session", *, turn: int) -> None:
    exchange = getattr(session, "_pending_replay_correction", None)
    if exchange is None:
        return
    _require_exact_final_user_correction(
        session, exchange["correction"]["text"]
    )
    mark_replayed = getattr(session.client, "mark_replayed_correction", None)
    if not callable(mark_replayed):
        raise RuntimeError("replay correction has no consumption boundary")
    marked = mark_replayed()
    if (
        marked["correction"]["correction_id"]
        != exchange["correction"]["correction_id"]
    ):
        raise RuntimeError("replay correction identity changed before consumption")
    correction = exchange["correction"]
    consumption = exchange["consumption"]
    common = {
        "correction_id": correction["correction_id"],
        "text_sha256": correction["text_sha256"],
        "replayed": True,
    }
    session._emit(
        "correction_created",
        session_number=session._session_number,
        text_chars=len(correction["text"]),
        source_session_number=correction["after_session_number"],
        **common,
    )
    session._emit(
        "correction_consumed",
        session_number=session._session_number,
        turn_number=turn,
        transcript_segment=consumption["transcript_segment"],
        delivery="replay",
        source_session_number=consumption["session_number"],
        source_turn_number=consumption["turn_number"],
        source_transcript_segment=consumption["transcript_segment"],
        **common,
    )
    session._pending_replay_correction = None
    session._emit(
        "correction_replayed",
        session_number=session._session_number,
        turn_number=turn,
        source_session_number=consumption["session_number"],
        source_turn_number=consumption["turn_number"],
        source_transcript_segment=consumption["transcript_segment"],
        **common,
    )
    _correction_trace_barrier(session)


def _run_post_turn_hooks(
    session: "Session", turn: int, *, run_advisor: bool = True
) -> None:
    """Run observation, adaptive, rewind, and advisor post-turn hooks."""
    from ..turn_snapshots import process_rewind_turn_boundary
    # A guardrail rewind invalidates this turn. Restore its saved control
    # state before adaptive observers can learn from the discarded branch.
    if getattr(session, "_pending_rewind", None) is not None:
        process_rewind_turn_boundary(session, turn)
        return
    session._maybe_emit_harness_observation(turn)
    from ..adaptive_control.observation_notice import record_loop_observation_notice
    record_loop_observation_notice(session, turn)
    session._maybe_run_llm_hurdle_detector(turn)
    session._maybe_switch_adaptive_phase(turn)
    rewound = process_rewind_turn_boundary(session, turn)
    if run_advisor and not rewound:
        session._maybe_run_advisor(turn)


def _complete_turn_rewind(
    session: "Session",
    decision: Decision,
    *,
    turn: int,
    content: str | None,
    tool_calls: list,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Balance a pre-dispatch turn and queue its guardrail rewind."""
    cfg = session.cfg
    for tc in tool_calls:
        args_summary = _summarize_args(
            tc.arguments, cfg.trace_args_summary_chars
        )
        metadata = action_metadata(tc.name, tc.arguments)
        result = session._decorate_stream_rule_tool_result(
            tc.id, decision.text, turn=turn
        )
        session.context.add_tool_result(
            tc.id,
            result,
            tool_name=tc.name,
            gate_blocked=True,
        )
        session._emit(
            "tool_call",
            session_number=session._session_number,
            turn_number=turn,
            tool_name=tc.name,
            args_summary=args_summary,
            **build_tool_call_trace_fields(
                session,
                tool_name=tc.name,
                args_summary=args_summary,
                result=result,
                turn=turn,
                gate_blocked=True,
                metadata=metadata,
            ),
            reasoning=_truncate_for_trace(
                content or "", cfg.trace_reasoning_store_chars
            ),
            gate_blocked=True,
            gate_reason=decision.reason,
            **metadata,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    session.request_rewind(
        decision.target_turn,
        reason=decision.reason or "rewind_on_guardrail",
    )


def _preflight_estimate(session) -> int:
    """Current-list token estimate for the pre-flight gate.

    Use the active request counter with the effective tool catalog. Explicit
    estimate-only runs retain the strategy estimator.
    """
    tok = getattr(session, "_tokenizer", None)
    session._preflight_count_authoritative = False
    if tok is not None:
        try:
            count = tok.count(
                list(session.context.get_messages()),
                tools=effective_model_tool_schemas(session))
            if type(count) is not int or count < 0:
                raise ValueError("request counter must return a nonnegative integer")
            from ..request_counting import has_reported_count
            session._preflight_count_authoritative = has_reported_count(tok, count)
            return count
        except Exception as e:
            log.warning("preflight request count failed (%s); using strategy estimate", e)
    try:
        return int(session.context.estimate_tokens())
    except Exception:
        # The strategy may share the same unavailable counter.
        from ..context import chars_div_4
        return int(chars_div_4(session.context.get_messages()))


def _preflight_prompt_tokens(live_pt: int, estimated_pt: int,
                             prev_estimate: int | None,
                             density_hat: float = 0.25, *, authoritative_count: bool = False) -> int:
    """Use the current backend count, otherwise a historical usage estimate.

    With an authoritative count, prior usage describes a different request
    and must not raise the current count. Otherwise, compare prior usage,
    the current strategy estimate, and a projection of positive estimated
    growth at the stored density. Multiplying estimate growth by four is
    a legacy character approximation, not a measurement of appended text.
    This path can overestimate or underestimate; it does not prove a fit.
    """
    if authoritative_count:
        return estimated_pt  # prior usage names a different request
    candidates = [live_pt, estimated_pt]
    if prev_estimate is not None and live_pt > 0:
        chars_new = max(0, estimated_pt - prev_estimate) * 4
        candidates.append(live_pt + int(chars_new * max(0.25, density_hat)))
    return max(candidates)


def _observe_token_density(session, live_pt_at_gate: int,
                           chars_new_at_gate: int, actual_pt: int) -> None:
    """Update a legacy density estimate from usage and estimated text growth.

    Keep the estimate monotone up, capped at 2.0. The 800-character minimum
    is a policy filter for small estimated appends, not proof that template
    overhead is absent above it.
    """
    if chars_new_at_gate < 800 or live_pt_at_gate <= 0:
        return
    delta = actual_pt - live_pt_at_gate
    if delta <= 0:
        return
    observed = delta / chars_new_at_gate
    current = getattr(session, "_preflight_density", 0.25)
    session._preflight_density = min(2.0, max(current, observed))


def run_session_loop(session: "Session") -> "SessionResult":
    """Drive one session's turn loop.

    Each guardrail returns a uniform ``Decision``. This loop alone decides
    how the session acts on that decision. The phase order is:

        1. API call (with transient-error retry)
        2. context fill                                     END
        3. intent_gate         (turn-level, pre-dispatch)   BLOCK / END
        4. stop check          (natural exit)
        5. loop_detect         (pre-dispatch request diagnostics; native PASS)
        6. per tool call:
           6a. pre_tool hook      (before validation)         BLOCK / REWRITE
           6b. schema/permission/approval                    BLOCK / PAUSE
           6c. done and tool guardrails                      BLOCK / END / WARN
           6d. dispatch          (when not blocked)
           6e. post_tool hook     (after real dispatch)       BLOCK / ANNOTATE
           6f. post-tool ladders, trace, and record
        7. duplicate_guard     (completed observations)     WARN / END
        8. max_turns                                         END
    """
    # Late-bind names that tests patch on the public ``loop`` module.
    # mock.patch is entered before session.run() is called, so by the
    # time we hit this rebind, ``_loop_mod.dispatch`` / ``_loop_mod.log``
    # is the mock and subsequent calls use it. Same trick as PR #1's
    # _run_in_sandbox. The ``log`` rebind matters for tests that patch
    # ``loop.log`` to assert which messages the loop emits.
    from .. import loop as _loop_mod
    from .._tools._run_in_sandbox import SandboxUnavailableError
    dispatch = _loop_mod.dispatch
    SessionResult = _loop_mod.SessionResult
    _READONLY_TOOLS = _loop_mod._READONLY_TOOLS
    log = _loop_mod.log

    total_prompt = 0
    total_completion = 0
    turn_pre = session._guardrail_registry.turn_pre_dispatch
    turn_post = session._guardrail_registry.turn_post_dispatch
    tool_pre = session._guardrail_registry.tool_pre_dispatch
    tool_post = session._guardrail_registry.tool_post_dispatch
    observers = session._guardrail_registry.observers
    from ..savings import get_ledger
    # max_turns is read once for the loop bound — that field is immutable
    # across the session. cfg is otherwise re-read at the top of each
    # iteration so the adaptive-phase switch (which mutates session.cfg
    # via dataclasses.replace) is visible on the next turn.
    turn_start = int(getattr(session, "_turn_start_offset", 0) or 0)
    if (
        getattr(session.cfg, "runtime_mode", "measurement") == "assistant"
        and getattr(session, "_subagent_level", 0) == 0
    ):
        if clarification_state(session._artifact_dir).phase == "input_required":
            return SessionResult(
                turn_start,
                "input_required",
                done=False,
                total_prompt_tokens=0,
                total_completion_tokens=0,
            )
    # Charge an entered loop iteration once, including preflight/failed work.
    # Restored labels identify conversation turns, not newly consumed budget.
    consumed_turns = 0
    if getattr(session, "_lifecycle_hook_block_reason", ""):
        return SessionResult(
            consumed_turns,
            "hook_block",
            done=False,
            total_prompt_tokens=0,
            total_completion_tokens=0,
        )
    for local_turn in range(session.cfg.max_turns):
        consumed_turns = local_turn + 1
        turn = turn_start + local_turn
        # Session-owned services (for example the lazy LSP manager) emit
        # trace records outside this loop module.  Stamp their events with
        # the turn whose tool call triggered the service.
        session._current_turn = turn
        cfg = session.cfg
        # Stamp before any turn-boundary intervention is inserted.
        get_ledger().set_turn(session._session_number, turn)
        # Freeze the planning phase for this model response.  A successful
        # exit changes the next turn's surface, but cannot unlock a mutating
        # sibling call that arrived in the same response.
        plan_task_required = bool(session._plan_mode.required)
        plan_turn_active = bool(session._plan_mode.active)
        # stop_resume delivery: the controller decided to intervene last
        # turn and requested a graceful hand-off. End before the next API
        # call; the stop-note in telemetry carries the resume payload.
        if getattr(session, "_adaptive_stop_requested", None):
            log.info("adaptive_stop: controller requested stop-resume hand-off")
            return SessionResult(
                consumed_turns,
                "adaptive_stop",
                done=False,
                total_prompt_tokens=total_prompt,
                total_completion_tokens=total_completion,
            )
        # The user_turn delivery mode queues a synthetic user message.
        # Append it at the next turn boundary and keep the session context.
        _pending_user_turn = getattr(session, "_adaptive_user_turn_pending", None)
        if _pending_user_turn:
            session._queue_user_turn_injection(
                UserTurnInjection(
                    text=_pending_user_turn,
                    bucket="adaptive_intervention",
                    mechanism="adaptive_user_turn",
                    ctx={"delivery": "user_turn"},
                )
            )
            session._adaptive_user_turn_pending = None
        delivered_user_turns = session._deliver_pending_user_turn_injections()
        if delivered_user_turns:
            log.info(
                "user_turn_injection: delivered %d fragment(s) at turn %d",
                delivered_user_turns,
                turn,
            )
        # Per-turn phase timing. Captured fields land on every tool_call
        # trace entry of this turn so post-hoc analysis can compute
        # median per-phase ms across a session.
        _turn_t0 = time.perf_counter()
        _phase_chat_ms = 0.0
        _phase_token_ms = 0.0
        # Trace IO accumulator — write_trace adds to this every emit.
        # The bottom emit reads it back so the turn's prior trace
        # writes are visible (the bottom emit itself isn't counted in
        # the value it carries — that write happens after read).
        session._turn_trace_write_ms = 0.0
        # Deferred, non-interrupting prose rules become a hidden user
        # fragment only at a clean turn boundary. Tool-source reminders are
        # bound to their own tool result during dispatch instead.
        session._apply_pending_stream_rule_injections(turn)
        # Inject keyword-triggered fragments (harness/injections.py)
        # against the latest user/tool content before the API call.
        # No-op when the subsystem is disabled or no fragments load.
        session._apply_injections(turn_number=turn)
        session._inject_pending_advisor(turn)
        pre_model_hook = session._run_hook("pre_model")
        session._add_hook_context(pre_model_hook)
        if pre_model_hook.blocked:
            log.warning(
                "pre_model hook blocked turn %d: %s",
                turn,
                pre_model_hook.reason,
            )
            return SessionResult(
                consumed_turns,
                "hook_block",
                done=False,
                total_prompt_tokens=total_prompt,
                total_completion_tokens=total_completion,
            )
        # Resume-time rewind restoration and pre-model hooks have finished.
        # Add a live or replayed correction now so it is the final exact user
        # input before request preflight and transport.
        _inject_pending_correction(session)
        _apply_replay_correction(session, turn=turn)
        # ─── 0. GUARDRAIL: context fill (PRE-FLIGHT) ──────────────────
        # The post-flight check at the end of step 2 catches overflow
        # that develops during the response, but a tool result added
        # at the tail of the previous turn can already push the next
        # request past server n_ctx. Without this pre-flight, the API
        # call returns BadRequestError (exceed_context_size_error) and
        # the session ends with finish_reason="error" instead of the
        # cleaner "context_full". Same threshold as the post-flight
        # check (cfg.context_fill_ratio).
        # Refresh the local config after applying shared capacity constraints.
        session._get_server_ctx()
        cfg = session.cfg
        if local_turn == 0 and getattr(session, "_runtime_briefing", None):
            from ..runtime_briefing import admit_runtime_briefing
            admit_runtime_briefing(session)
        # Use the backend's current request count when available. Only an
        # explicitly selected estimator retains the legacy density policy.
        if cfg.context_size > 0:
            live_pt = int(getattr(session, "_last_actual_prompt_tokens", 0) or 0)
            _tok_t0 = time.perf_counter()
            estimated_pt = _preflight_estimate(session)
            _phase_token_ms += (time.perf_counter() - _tok_t0) * 1000
            estimate_policy = getattr(cfg, "tokenizer_id", "") != "auto"
            can_budget = session._preflight_count_authoritative or estimate_policy
            pressure = getattr(session.context, "projection_pressure", None)
            if (can_budget and isinstance(pressure, dict) and pressure.get("overflow")
                    and pressure.get("actionable", True)):
                # The projection already exhausted its permitted reductions.
                # Keep its required content and response reserve intact.
                session._emit("context_projection_pressure", turn_number=turn,
                              generation_sent=False, **pressure)
                return SessionResult(consumed_turns, "context_full", done=False,
                                     total_prompt_tokens=total_prompt,
                                     total_completion_tokens=total_completion)
            prev_estimate = getattr(session, "_preflight_prev_estimate", None)
            session._preflight_prev_estimate = estimated_pt
            density_hat = getattr(session, "_preflight_density", 0.25)
            # Retain gate estimates for the legacy post-response projection.
            # Estimate growth is not a measurement of appended characters.
            session._preflight_gate_live = live_pt
            session._preflight_gate_chars_new = (
                max(0, estimated_pt - prev_estimate) * 4
                if estimate_policy and prev_estimate is not None
                and not session._preflight_count_authoritative else 0)
            preflight_pt = _preflight_prompt_tokens(
                live_pt, estimated_pt, prev_estimate, density_hat,
                authoritative_count=session._preflight_count_authoritative) if can_budget else estimated_pt
            if not can_budget:
                session._emit("context_count_unverified", turn_number=turn,
                              estimate_tokens=estimated_pt, action="preserve_context")
            pre_fill = preflight_pt / cfg.context_size
            # Diagnostic ratio of prior usage to the prior gate estimate.
            # Their payloads may differ after compaction or request assembly;
            # this ratio is not a measured density of the current request.
            prev_est = int(getattr(session, "_prev_preflight_estimate_pt", 0) or 0)
            density = (live_pt / prev_est) if (prev_est > 0 and live_pt > 0) else 0.0
            session._prev_preflight_estimate_pt = estimated_pt
            if density > 2.0:
                _shape, _quirk = provenance_for(session, "")
                get_system_log().event(
                    "density_blowout", turn=turn, live_pt=live_pt,
                    estimate_pt=prev_est, preflight_pt=preflight_pt,
                    density=density, ctx=cfg.context_size,
                    command_shape=_shape, quirk_hit=_quirk, action="none",
                )
            if can_budget and pre_fill > cfg.context_fill_ratio:
                # Last-resort fit gate, after the configured context strategy
                # (including halflife) has rendered the request. First try one
                # local, deficit-sized re-clip. If that cannot make the prompt
                # fit, try one digest. There is no retry loop over the same
                # material: the request proceeds or this session ends.
                prompt_budget = int(cfg.context_fill_ratio * cfg.context_size)
                original_projection = preflight_pt
                clip = None
                if bool(getattr(cfg, "preflight_reclip_enabled", True)):
                    clip = preflight_reclip_oversized(
                        session,
                        projected_tokens=preflight_pt,
                        budget_tokens=prompt_budget,
                    )
                if clip is not None:
                    _tok_t0 = time.perf_counter()
                    estimated_pt = _preflight_estimate(session)
                    can_budget = session._preflight_count_authoritative or estimate_policy
                    _phase_token_ms += (time.perf_counter() - _tok_t0) * 1000
                    # The old server count names a different message list after
                    # re-clipping. Carry its conservative projection forward by
                    # subtracting only the measured local reduction.
                    reduction = max(0, clip["orig_pt"] - clip["new_pt"])
                    preflight_pt = estimated_pt if session._preflight_count_authoritative else max(
                        estimated_pt,
                        original_projection - reduction,
                    )
                    pre_fill = preflight_pt / cfg.context_size

                compacted = False
                compaction_failed = False
                if can_budget and pre_fill > cfg.context_fill_ratio:
                    count_before = int(getattr(session, "_compaction_count", 0))
                    try:
                        maybe_compact_messages(
                            session,
                            list(session.context.get_messages()),
                            projected_tokens=preflight_pt,
                        )
                    except CompactionOverflowError as exc:
                        compaction_failed = True
                        log.error(
                            "Last-resort compaction overflow on turn %d: %s",
                            turn,
                            exc,
                        )
                    compacted = int(
                        getattr(session, "_compaction_count", 0)
                    ) > count_before
                    if compacted:
                        _tok_t0 = time.perf_counter()
                        estimated_pt = _preflight_estimate(session)
                        can_budget = session._preflight_count_authoritative or estimate_policy
                        _phase_token_ms += (
                            time.perf_counter() - _tok_t0
                        ) * 1000
                        # Digest replaced the old history, so the prior live
                        # count and density projection no longer describe this
                        # request. Re-count the replacement itself.
                        preflight_pt = estimated_pt
                        pre_fill = preflight_pt / cfg.context_size
                _shape, _quirk = provenance_for(
                    session, (clip or {}).get("tool_call_id", ""))
                still_over = can_budget and pre_fill > cfg.context_fill_ratio
                if still_over:
                    recovery_action = "session_end"
                elif compacted and clip is not None:
                    recovery_action = "reclipped_then_compacted"
                elif compacted:
                    recovery_action = "compacted"
                else:
                    recovery_action = "reclipped"
                get_system_log().event(
                    "preflight_overflow", turn=turn, live_pt=live_pt,
                    estimate_pt=estimated_pt, preflight_pt=preflight_pt,
                    density=density, ctx=cfg.context_size,
                    command_shape=_shape, quirk_hit=_quirk,
                    action=recovery_action,
                )
                if still_over:
                    log.info(
                        "Context %.0f%% full pre-flight at turn %d, ending session "
                        "(live_pt=%d estimate_pt=%d preflight_pt=%d density=%.2f, "
                        "compaction_failed=%s)",
                        pre_fill * 100, turn, live_pt, estimated_pt,
                        preflight_pt, density, compaction_failed,
                    )
                    return SessionResult(consumed_turns, "context_full", done=False, total_prompt_tokens=total_prompt, total_completion_tokens=total_completion)
                if compacted:
                    log.info(
                        "Context pre-flight overflow at turn %d recovered by "
                        "last-resort compaction (after_reclip=%s, estimate_pt=%d)",
                        turn, clip is not None, estimated_pt,
                    )
                else:
                    log.info(
                        "Context pre-flight overflow at turn %d recovered by re-clip "
                        "(msg %d: %d -> %d tokens; estimate_pt=%d)",
                        turn, clip["index"], clip["orig_pt"], clip["new_pt"],
                        estimated_pt,
                    )
        # Mark the exact operator answer consumed before transport. An
        # ambiguous transport failure must not deliver it a second time.
        _consume_pending_clarification(session, turn=turn)
        _consume_replay_correction(session, turn=turn)
        # Use the same fail-closed boundary for a correction. Once transport
        # can observe it, no later resume may inject it again silently.
        _consume_pending_correction(session, turn=turn)
        # ─── 1. API call (with transient-error retry) ────────────────
        _chat_t0 = time.perf_counter()
        chat_result = session._chat_with_retry(turn)
        cfg = session.cfg  # A successful fallback can replace the client/config.
        _phase_chat_ms = (time.perf_counter() - _chat_t0) * 1000
        abandoned = getattr(session, "_abandoned_chat_usage", None)
        if abandoned is not None:
            total_prompt += abandoned.prompt_tokens
            total_completion += abandoned.completion_tokens
            session._abandoned_chat_usage = None
        if chat_result is None:
            # chat_io may have set a more specific reason (e.g.
            # "compaction_overflow"); fall back to "error" otherwise.
            err_reason = getattr(session, "_last_chat_error_reason", None) or "error"
            return SessionResult(consumed_turns, err_reason, done=False, total_prompt_tokens=total_prompt, total_completion_tokens=total_completion)
        content = chat_result.content
        session._last_assistant_content = (
            content if isinstance(content, str) else ""
        )
        tool_calls = chat_result.tool_calls
        reason = chat_result.finish_reason
        prompt_tokens = chat_result.usage.prompt_tokens
        completion_tokens = chat_result.usage.completion_tokens
        first_prompt_tokens = (
            chat_result.first_prompt_tokens
            if chat_result.first_prompt_tokens is not None else prompt_tokens
        )
        last_prompt_tokens = (
            chat_result.last_prompt_tokens
            if chat_result.last_prompt_tokens is not None else prompt_tokens
        )
        cache_observation = CacheObservation(
            prompt_tokens=int(prompt_tokens or 0),
            cached_tokens=chat_result.usage.cached_tokens,
            hit_ratio=chat_result.usage.cache_hit_ratio,
            source="turn_result",
        )
        accumulator = getattr(session, "_cache_usage_accumulator", None)
        if accumulator is not None:
            accumulator.record(cache_observation)
        role_ledger = getattr(session, "_role_token_ledger", None)
        if role_ledger is not None:
            role_ledger.record_usage(
                getattr(session, "_active_model_resolution", None) or "main",
                chat_result.usage,
                cached_tokens=int(chat_result.usage.cached_tokens or 0),
            )
        session_usage = getattr(session, "_session_usage_accumulator", None)
        if session_usage is not None:
            session_usage.record(chat_result.usage)
        warn_on_cache_miss(
            cache_observation,
            warn_ratio=getattr(cfg, "cache_miss_warn_ratio", 0.0),
            prior_turns=local_turn,
            logger=log,
        )
        session._emit(
            "turn",
            session_number=session._session_number,
            turn_number=turn,
            role=getattr(session, "_active_model_role", "main"),
            **cache_observation.trace_fields(),
            first_prompt_tokens=first_prompt_tokens,
            last_prompt_tokens=last_prompt_tokens,
            prompt_tokens_known=chat_result.usage.prompt_tokens_known,
            completion_tokens_known=chat_result.usage.completion_tokens_known,
        )
        total_prompt += prompt_tokens
        total_completion += completion_tokens
        # Occupancy belongs to the last request, not cumulative turn usage.
        session._last_actual_prompt_tokens = int(last_prompt_tokens or 0)
        _observe_token_density(
            session,
            getattr(session, "_preflight_gate_live", 0),
            getattr(session, "_preflight_gate_chars_new", 0),
            int(first_prompt_tokens or 0),
        )
        if reason == "narration_discarded":
            # Charge the attempt, retain the redirect, and never treat its
            # discarded/empty response as implicit completion.
            _run_post_turn_hooks(session, turn, run_advisor=False)
            continue
        session.context.add_assistant(
            session.client.build_assistant_message(content, tool_calls)
        )
        session.context.consume_injected_fragments()
        session._capture_advisor_turn(turn, content, tool_calls)

        # ─── 2. Context fill observation ─────────────────────────────
        # Server-reported pt is accurate, with no chars/4 underrun. The
        # request has already succeeded, so handle this response normally.
        # If it requires another model turn, that turn's pre-flight gate runs
        # the canonical re-clip -> digest -> context_full ladder.
        if cfg.context_size > 0:
            fill = session._last_actual_prompt_tokens / cfg.context_size
            session._last_fill = fill
            if fill > cfg.context_fill_ratio:
                log.info(
                    "Context %.0f%% full at turn %d; the next model turn will "
                    "run the pre-flight fit gate",
                    fill * 100,
                    turn,
                )

        clarification_action = _handle_clarification_response(
            session, tool_calls, turn=turn
        )
        if clarification_action == "pause":
            return SessionResult(
                turn,
                "input_required",
                done=False,
                total_prompt_tokens=total_prompt,
                total_completion_tokens=total_completion,
            )
        if clarification_action == "continue":
            _run_post_turn_hooks(session, turn)
            continue

        # ─── 3. GUARDRAIL: intent_gate (BLOCK / END tiers) ───────────
        # Declared delay for intent, duplicate-call and loop-detection guards.
        # Other guards retain their own activation rules. Provenance and
        # policy limits live in docs/knob_reference/loop.md.
        guards_armed = turn > getattr(cfg, "guardrails_arm_after_turn", 0)
        intent_decision = (
            turn_pre["intent_gate"](
                session._guards, cfg,
                turn=turn, content=content, tool_calls=tool_calls,
                allow_intervention=guards_armed,
            )
            if not plan_task_required
            else PASS
        )
        if not guards_armed:
            intent_decision = PASS
        if intent_decision.action in (
            Action.BLOCK,
            Action.END,
            Action.REWIND,
        ):
            session._record_pressure_event(True)
            log.info("Intent gate: rejecting silent tool call at turn %d "
                     "(block #%d, consecutive %d)", turn,
                     session._guards.intent_block_count,
                     session._guards.consecutive_intent_rejections)
            for tc in tool_calls:
                args_summary = _summarize_args(
                    tc.arguments,
                    cfg.trace_args_summary_chars,
                )
                metadata = action_metadata(tc.name, tc.arguments)
                result = session._decorate_stream_rule_tool_result(
                    tc.id, intent_decision.text, turn=turn
                )
                session.context.add_tool_result(tc.id, result,
                                             tool_name=tc.name, cmd_signature="",
                                             gate_blocked=True)
                session._emit(
                    "tool_call",
                    session_number=session._session_number,
                    turn_number=turn,
                    tool_name=tc.name,
                    args_summary=args_summary,
                    **build_tool_call_trace_fields(
                        session,
                        tool_name=tc.name,
                        args_summary=args_summary,
                        result=result,
                        turn=turn,
                        gate_blocked=True,
                        metadata=metadata,
                    ),
                    reasoning="",
                    gate_blocked=True,
                    gate_reason=intent_decision.reason,
                    **metadata,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            if intent_decision.action == Action.REWIND:
                session.request_rewind(
                    intent_decision.target_turn,
                    reason=(
                        intent_decision.reason or "rewind_on_intent_gate"
                    ),
                )
                _run_post_turn_hooks(session, turn)
                continue
            if intent_decision.action == Action.END:
                log.warning("Intent abort: %d consecutive silent rejections",
                            session._guards.consecutive_intent_rejections)
                return SessionResult(consumed_turns, intent_decision.reason, done=False,
                                     total_prompt_tokens=total_prompt,
                                     total_completion_tokens=total_completion)
            _run_post_turn_hooks(session, turn)
            continue

        # ─── 4. Stop check (natural exit) ────────────────────────────
        if not tool_calls:
            from ..turn_snapshots import process_rewind_turn_boundary
            if process_rewind_turn_boundary(session, turn):
                continue
            if reason == "length":
                session._maybe_run_advisor(turn)
                log.info("Response truncated at turn %d (max_tokens hit), ending session", turn)
                return SessionResult(consumed_turns, "length", done=False, total_prompt_tokens=total_prompt, total_completion_tokens=total_completion)
            if plan_turn_active:
                log.warning(
                    "Model stopped at turn %d while plan mode remained active; "
                    "session ended without success",
                    turn,
                )
                return SessionResult(
                    consumed_turns,
                    "no_tool_call",
                    done=False,
                    total_prompt_tokens=total_prompt,
                    total_completion_tokens=total_completion,
                )
            advisor_intervened = session._maybe_run_advisor(turn)
            if advisor_intervened:
                log.info(
                    "Advisor queued a note for the next model-facing turn %d",
                    turn + 1,
                )
                continue
            # With implicit done enabled, `finish_reason="stop"` and no tool
            # calls count as success. Setting it to False
            # treats no-tool-calls as session end (`done=False`,
            # finish_reason="no_tool_call") so result attribution
            # tools (run_summary, paired-delta, failure-classifier) can
            # distinguish "model said done explicitly" from "model
            # silently fell off the conversation".
            allow_implicit = bool(getattr(cfg, "allow_implicit_done", True))
            if allow_implicit:
                verification_gate_active = (
                    int(
                        getattr(
                            cfg,
                            "post_mutation_verification_gate_after",
                            0,
                        )
                        or 0
                    )
                    > 0
                    and session._guards.has_mutated
                )
                if verification_gate_active:
                    implicit_done_decision = tool_pre["done_guard"](
                        session._guards,
                        cfg,
                        tc_name="done",
                        cwd=session.cwd,
                    )
                    if implicit_done_decision.action == Action.BLOCK:
                        session.context.add_injected_fragment(
                            implicit_done_decision.text
                        )
                        session._record_pressure_event(True)
                        log.info(
                            "post-mutation verification gate rejected implicit "
                            "done at turn %d",
                            turn,
                        )
                        _run_post_turn_hooks(session, turn)
                        continue
                    if implicit_done_decision.action == Action.END:
                        return SessionResult(
                            consumed_turns,
                            implicit_done_decision.reason or "done_loop",
                            done=False,
                            total_prompt_tokens=total_prompt,
                            total_completion_tokens=total_completion,
                        )
                done_hook = session._run_hook(
                    "done",
                    implicit=True,
                    finish_reason=reason,
                )
                if done_hook.blocked:
                    message = (
                        "ERROR: done hook blocked completion: "
                        f"{done_hook.reason}"
                    )
                    block = done_hook.context_block()
                    if block:
                        message += "\n\n" + block
                    session.context.add_injected_fragment(message)
                    session._record_pressure_event(True)
                    _run_post_turn_hooks(session, turn)
                    continue
                log.info("Model stopped at turn %d (reason=%s) — implicit done", turn, reason)
                return SessionResult(consumed_turns, "stop", done=True, total_prompt_tokens=total_prompt, total_completion_tokens=total_completion)
            log.warning(
                "Model stopped at turn %d (reason=%s) without calling done() — "
                "session ended without success (allow_implicit_done=False)",
                turn, reason,
            )
            return SessionResult(consumed_turns, "no_tool_call", done=False, total_prompt_tokens=total_prompt, total_completion_tokens=total_completion)

        sig = tuple(_dedup_signature(tc) for tc in tool_calls)
        turn_warn_text = ""
        turn_had_pressure = False

        # Collect request-signature diagnostics. The native guard leaves
        # intervention to completed-observation notices after dispatch.
        # Custom registry decisions retain the ordinary handling below.
        loop_decision = (
            turn_pre["loop_detect"](
                session._guards, cfg, tool_calls_sig=sig,
                allow_intervention=guards_armed,
            )
            if not plan_turn_active
            else PASS
        )
        if not guards_armed:
            loop_decision = PASS
        if loop_decision.action == Action.REWIND:
            session._record_pressure_event(True)
            _complete_turn_rewind(
                session,
                loop_decision,
                turn=turn,
                content=content,
                tool_calls=tool_calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            _run_post_turn_hooks(session, turn)
            continue
        if loop_decision.action == Action.END:
            session._record_pressure_event(True)
            log.warning("Loop detected, aborting at turn %d", turn)
            return SessionResult(consumed_turns, loop_decision.reason, done=False,
                                 total_prompt_tokens=total_prompt,
                                 total_completion_tokens=total_completion)
        if loop_decision.action == Action.WARN:
            # Queue this pre-dispatch guard warning separately.
            turn_warn_text = (
                f"{turn_warn_text}\n\n{loop_decision.text}"
                if turn_warn_text else loop_decision.text
            )
            turn_had_pressure = True

        # ─── 6. Dispatch loop (per tool call) ────────────────────────
        # Optional parallel pre-execute for all-read-only turns.
        # Guardrails and post-dispatch state still run sequentially
        # per-tc below; this only concurrent-izes the file-I/O
        # dispatch() work for turns that emit multiple independent
        # read/glob/grep calls. Mutating tools (write/edit/bash)
        # always run sequentially — they never enter this path.
        preexecuted: dict[str, str] = {}
        turn_active_tool_names = frozenset(session.active_tool_names)
        inactive_tool_call_ids = (
            frozenset()
            if plan_turn_active
            else frozenset(
                tc.id
                for tc in tool_calls
                if session.is_hidden_tool(
                    tc.name, active_names=turn_active_tool_names
                )
            )
        )
        plan_decisions = {
            tc.id: session._plan_mode.check(
                tc.name,
                tc.arguments,
                turn=turn,
                active=plan_turn_active,
            )
            for tc in tool_calls
        }
        pre_tool_hooks = {}
        for tc in tool_calls:
            # A rejected planning action must not invoke host hooks, and the
            # plan-mode error must remain the single model-facing rejection.
            if not plan_decisions[tc.id].allowed:
                continue
            effect = session._run_hook(
                "pre_tool",
                tool_call_id=tc.id,
                tool_name=tc.name,
                tool_args=dict(tc.arguments),
            )
            if effect.updated_input is not None:
                tc.arguments.clear()
                tc.arguments.update(effect.updated_input)
            pre_tool_hooks[tc.id] = effect
        schema_validations = {}
        if getattr(cfg, "tools_schema_validation", "off") == "reject":
            phase_schema_set = session.tool_schema_set_for_phase(
                plan_mode_active=plan_turn_active
            )
            schema_validations = {
                tc.id: phase_schema_set.validate(
                    tc.name, tc.arguments
                )
                for tc in tool_calls
                if (
                    tc.id not in inactive_tool_call_ids
                    and plan_decisions[tc.id].allowed
                    and not pre_tool_hooks[tc.id].blocked
                )
            }
        permission_resolutions = {}
        approval_available = approval_transport_available(session._trace_path)
        advisor_intervened = False
        for tc in tool_calls:
            if (
                tc.id in inactive_tool_call_ids
                or not plan_decisions[tc.id].allowed
                or pre_tool_hooks[tc.id].blocked
            ):
                continue
            validation = schema_validations.get(tc.id)
            if validation is not None and not validation.valid:
                continue
            resolution = resolve_tool_permission(
                policy=session._permission_policy,
                tool_name=tc.name,
                arguments=tc.arguments,
                cfg=cfg,
                approval_available=approval_available,
            )
            permission_resolutions[tc.id] = resolution
            session._emit(
                "permission",
                session_number=session._session_number,
                turn_number=turn,
                **resolution.trace_fields(),
            )
        state = TurnState(
            session=session,
            turn=turn,
            cfg=cfg,
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            turn_warn_text=turn_warn_text,
            phase_chat_ms=_phase_chat_ms,
            phase_token_ms=_phase_token_ms,
            turn_t0=_turn_t0,
            preexecuted=preexecuted,
            pre_tool_hooks=pre_tool_hooks,
            inactive_tool_call_ids=inactive_tool_call_ids,
            schema_validations=schema_validations,
            permission_resolutions=permission_resolutions,
            dispatch=dispatch,
            log=log,
            tool_pre=tool_pre,
            tool_post=tool_post,
            observers=observers,
            plan_mode_active=plan_turn_active,
            turn_had_pressure=turn_had_pressure,
        )
        if (
            cfg.parallel_readonly_enabled
            and not plan_turn_active
            and len(tool_calls) > 1
            and not inactive_tool_call_ids
            and all(tc.name in _READONLY_TOOLS for tc in tool_calls)
            and all(
                validation.valid
                for validation in schema_validations.values()
            )
            and all(
                resolution.allowed
                for resolution in permission_resolutions.values()
            )
            and all(not effect.blocked for effect in pre_tool_hooks.values())
        ):
            effective_output_control = (
                session.output_control if cfg.bash_transforms_task_format_enabled else None
            )
            effective_universal_rewrites = (
                session.universal_rewrites if cfg.bash_transforms_universal_enabled else None
            )
            with ThreadPoolExecutor(
                max_workers=max(1, cfg.parallel_max_workers)
            ) as _ex:
                futures = {}
                for tc in tool_calls:
                    # The durability point must precede submission: the
                    # worker may start executing as soon as ``submit``
                    # returns, and multiple read calls can overlap.
                    record_tool_start(tc, state)
                    state.preexecuted_metadata[tc.id] = {}
                    futures[tc.id] = _ex.submit(
                        copy_context().run, dispatch, tc.name, tc.arguments,
                        cwd=session.cwd, cfg=cfg,
                        output_control=effective_output_control,
                        universal_rewrites=effective_universal_rewrites,
                        forbidden_rules=session.forbidden_rules if cfg.bash_quirks_forbidden_enabled else None,
                        redirect_rules=getattr(session, "redirect_rules", None),
                        redactions=session.redactions,
                        tool_registry=session._tool_registry,
                        stale_guard=session._stale_guard,
                        active_tools=getattr(session, "active_tool_names", ()),
                        redirect_event_sink=getattr(session, "_redirect_event_sink", None),
                        security_event_sink=getattr(session, "_security_event_sink", None),
                        ignore_policy=session._ignore_policy,
                        effective_env=session._effective_env,
                        allow_login_shell=session._allow_login_shell,
                        tool_call_id=tc.id,
                        execution_metadata=state.preexecuted_metadata[tc.id],
                    )
                for tc_id, fut in futures.items():
                    try:
                        preexecuted[tc_id] = fut.result()
                    except SandboxUnavailableError as exc:
                        log.error("Sandbox unavailable at turn %d: %s", turn, exc)
                        return SessionResult(
                            consumed_turns,
                            "sandbox_unavailable",
                            done=False,
                            total_prompt_tokens=total_prompt,
                            total_completion_tokens=total_completion,
                        )
                    except Exception as e:
                        preexecuted[tc_id] = f"ERROR: {e}"
        for tc in tool_calls:
            args_summary = _summarize_args(tc.arguments, cfg.args_summary_chars)
            validation = schema_validations.get(tc.id)
            resolution = permission_resolutions.get(tc.id)
            if validation is None or validation.valid:
                if resolution is not None and not resolution.denied:
                    approval_allowed, approval_reason = approval_decision(
                        runtime_mode=getattr(
                            cfg, "runtime_mode", "measurement"
                        ),
                        cwd=session.cwd,
                        trace_path=session._trace_path,
                        tool_name=tc.name,
                        tool_args=tc.arguments,
                        args_summary=args_summary,
                        required_reason=(
                            resolution.approval_reason()
                            if resolution.approval_required
                            else None
                        ),
                        permission_rule=(
                            resolution.rule
                            if resolution.approval_required
                            else None
                        ),
                        cfg=cfg,
                    )
                    if not approval_allowed:
                        result = (
                            "APPROVAL REQUIRED: This tool call was not "
                            "executed. "
                            f"Reason: {approval_reason}. Review it with "
                            "`yuj show`, approve it with "
                            "`yuj approve <session_id>`, then resume."
                        )
                        hook_context = pre_tool_hooks[tc.id].context_block()
                        result = session._decorate_stream_rule_tool_result(
                            tc.id, result, turn=turn
                        )
                        if hook_context:
                            result += "\n\n" + hook_context
                        session.context.add_tool_result(
                            tc.id,
                            result,
                            tool_name=tc.name,
                            gate_blocked=True,
                        )
                        session._emit(
                            "approval_request",
                            session_number=session._session_number,
                            turn_number=turn,
                            tool_name=tc.name,
                            args_summary=_truncate_for_trace(
                                args_summary, cfg.trace_args_summary_chars
                            ),
                            reason=approval_reason,
                            reasoning=_truncate_for_trace(
                                content or "", cfg.trace_reasoning_store_chars
                            ),
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                        )
                        return SessionResult(
                            consumed_turns,
                            "approval_required",
                            done=False,
                            total_prompt_tokens=total_prompt,
                            total_completion_tokens=total_completion,
                        )
            try:
                outcome = dispatch_one_tool_call(tc, state)
            except SandboxUnavailableError as exc:
                log.error("Sandbox unavailable at turn %d: %s", turn, exc)
                return SessionResult(
                    consumed_turns,
                    "sandbox_unavailable",
                    done=False,
                    total_prompt_tokens=total_prompt,
                    total_completion_tokens=total_completion,
                )
            if outcome.rewind:
                break
            if outcome.end:
                if (
                    outcome.done
                    and len(tool_calls) == 1
                    and session._maybe_run_advisor(turn)
                ):
                    advisor_intervened = True
                    break
                return SessionResult(
                    consumed_turns, outcome.reason, done=outcome.done,
                    total_prompt_tokens=total_prompt,
                    total_completion_tokens=total_completion,
                )
        # checkpoint/rewind handlers only schedule context work. Finalizing
        # here guarantees the assistant message and every result from a
        # multi-tool turn form a complete protocol boundary before any cut.
        finalize_deferred_context_actions(session, turn)
        process_manager = getattr(session, "_process_manager", None)
        pending_work = False
        if cfg.duplicate_guard_enabled and process_manager is not None:
            pending_check = getattr(process_manager, "has_pending_observations", None)
            try:
                pending_work = not callable(pending_check) or pending_check() is not False
            except Exception:
                pending_work = True  # Unavailable process evidence is not completion.
        if session._guards.duplicate_last_turn != turn - 1:
            session._guards.recent_calls.clear()
        session._guards.duplicate_last_turn = turn
        session._guards.duplicate_evidence = {}
        dup_decision = (
            turn_post["duplicate_guard"](
                session._guards, cfg, tool_calls_sig=sig,
                observations=(None if pending_work else
                              tuple(state.observations.get(tc.id) for tc in tool_calls)),
                allow_intervention=guards_armed,
            )
            if not plan_turn_active and not outcome.rewind
            else PASS
        )
        if not guards_armed:
            dup_decision = PASS
        if not plan_turn_active and not outcome.rewind and cfg.duplicate_guard_enabled:
            session._emit(
                "duplicate_observation_check", session_number=session._session_number,
                turn_number=turn, action=dup_decision.action.value,
                reason=dup_decision.reason, pending_work=pending_work,
                interventions_allowed=guards_armed,
                quiet_through_turn=getattr(cfg, "guardrails_arm_after_turn", 0),
                **{"eligible": False, "count": 0, **session._guards.duplicate_evidence},
                abort_limit=cfg.duplicate_abort, warn_limit=cfg.duplicate_warn_count,
            )
        if plan_turn_active or outcome.rewind:
            session._guards.recent_calls.clear()
        if dup_decision.action == Action.WARN:
            session._queue_user_turn_injection(UserTurnInjection(
                text=dup_decision.text, bucket="guardrail_intervention",
                mechanism="duplicate_guard",
                ctx={"delivery": "user_turn", "basis": "completed_observations"},
            ))
            state.turn_had_pressure = True
        elif dup_decision.action == Action.REWIND:
            session.request_rewind(dup_decision.target_turn,
                                   reason=dup_decision.reason or "rewind_on_guardrail")
            state.turn_had_pressure = True
        elif dup_decision.action == Action.END:
            session._record_pressure_event(True)
            return SessionResult(consumed_turns, dup_decision.reason, done=False,
                                 total_prompt_tokens=total_prompt,
                                 total_completion_tokens=total_completion)
        session._record_pressure_event(state.turn_had_pressure)
        _run_post_turn_hooks(
            session, turn, run_advisor=not advisor_intervened
        )
    # ─── 7. GUARDRAIL: max_turns (hard cap, END tier) ────────────────
    return SessionResult(
        consumed_turns,
        "max_turns",
        done=False,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
    )
