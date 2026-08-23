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

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from ..guardrails import Action, Decision, PASS
from ..action_metadata import action_metadata
from ..approvals import approval_decision
from ..system_log import get_system_log, provenance_for
from ...server.request_controls import CacheObservation, warn_on_cache_miss
from . import _dedup_signature, _summarize_args, _truncate_for_trace
from ._dispatch_tool_call import TurnState, dispatch_one_tool_call
from .compaction import preflight_reclip_oversized
from .trace_output import build_tool_call_trace_fields

if TYPE_CHECKING:
    from ..loop import Session, SessionResult

log = logging.getLogger(__name__)


def _defer_guard_end_during_active_watch(
    session: "Session",
    decision: Decision,
    *,
    guard_name: str,
    turn: int,
) -> bool:
    """Let the adaptive watch, not a temporary action, end the run."""
    if decision.action != Action.END:
        return False
    pending = getattr(session, "_llm_detector_pending_watch", None)
    if not isinstance(pending, dict):
        return False
    try:
        watch_end = int(pending.get("watch_window_end", -1))
    except (TypeError, ValueError):
        return False
    if watch_end < int(turn):
        return False
    log.info(
        "Deferring %s termination at turn %d until adaptive watch end=%s",
        decision.reason or guard_name,
        turn,
        watch_end,
    )
    session._emit(
        "adaptive_control_guard_end_deferred",
        session_number=session._session_number,
        turn_number=turn,
        guard_name=guard_name,
        guard_reason=decision.reason,
        intervention_id=pending.get("intervention_id", ""),
        hurdle_episode_id=pending.get("episode_id", ""),
        watch_window_end=watch_end,
    )
    return True


def _run_post_turn_hooks(session: "Session", turn: int) -> None:
    """Run observation and adaptive hooks after executed or blocked turns."""
    session._maybe_emit_harness_observation(turn)
    session._maybe_run_llm_hurdle_detector(turn)
    session._maybe_switch_adaptive_phase(turn)


def _preflight_estimate(session) -> int:
    """Current-list token estimate for the pre-flight gate.

    When a local tokenizer is loaded, render the chat template with the
    session's tool schemas so the count includes the tool catalog. When
    no tokenizer is configured, use the strategy estimator (chars/4).
    """
    tok = getattr(session, "_tokenizer", None)
    if tok is not None:
        try:
            return int(tok.count(
                list(session.context.get_messages()),
                tools=getattr(session, "_tool_schemas", None)))
        except Exception as e:
            log.warning("preflight exact count failed (%s); using strategy estimate", e)
    return int(session.context.estimate_tokens())


def _preflight_prompt_tokens(live_pt: int, estimated_pt: int,
                             prev_estimate: int | None,
                             density_hat: float = 0.25) -> int:
    """Best lower bound on the next request's prompt tokens.

    live_pt (server-reported) is exact but stale by whatever was appended
    since the last request; the chars/4 estimate sees the current message
    list but systematically underruns real tokenizers on code-heavy text.
    A large new tool result can make both values too low. A third value
    adds the new characters, priced at the last known token density, to
    the exact server count. density_hat starts at 0.25 and rises when an
    observed turn has a higher token density. max() keeps the guard
    monotone, so it can only end a session earlier.
    """
    candidates = [live_pt, estimated_pt]
    if prev_estimate is not None and live_pt > 0:
        chars_new = max(0, estimated_pt - prev_estimate) * 4
        candidates.append(live_pt + int(chars_new * max(0.25, density_hat)))
    return max(candidates)


def _observe_token_density(session, live_pt_at_gate: int,
                           chars_new_at_gate: int, actual_pt: int) -> None:
    """Calibrate the session's tokens-per-char estimate from the server's
    own numbers. Monotone up, capped at 2.0; ignores tiny appends where
    the ratio is template-overhead noise."""
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
        5. duplicate_guard     (turn-level, pre-dispatch)   WARN / END
        6. per tool call:
           6a. done_guard        (tc-level, pre-dispatch)    BLOCK (or accept→END)
           6b. rumination_gate   (tc-level, pre-dispatch)    WARN-grace / BLOCK / END
           6c. dispatch          (when not blocked)
           6d. error_ladder      (tc-level, post-dispatch)   WARN / END
           6e. rumination_ladder (tc-level, post-dispatch)   WARN + ARM
           6f. append turn-level WARN; trace; record
        7. max_turns                                         END
    """
    # Late-bind names that tests patch on the public ``loop`` module.
    # mock.patch is entered before session.run() is called, so by the
    # time we hit this rebind, ``_loop_mod.dispatch`` / ``_loop_mod.log``
    # is the mock and subsequent calls use it. Same trick as PR #1's
    # _run_in_sandbox. The ``log`` rebind matters for tests that patch
    # ``loop.log`` to assert which messages the loop emits.
    from .. import loop as _loop_mod
    dispatch = _loop_mod.dispatch
    SessionResult = _loop_mod.SessionResult
    _READONLY_TOOLS = _loop_mod._READONLY_TOOLS
    log = _loop_mod.log

    total_prompt = 0
    total_completion = 0
    turn_pre = session._guardrail_registry.turn_pre_dispatch
    tool_pre = session._guardrail_registry.tool_pre_dispatch
    tool_post = session._guardrail_registry.tool_post_dispatch
    observers = session._guardrail_registry.observers
    from ..savings import get_ledger
    # max_turns is read once for the loop bound — that field is immutable
    # across the session. cfg is otherwise re-read at the top of each
    # iteration so the adaptive-phase switch (which mutates session.cfg
    # via dataclasses.replace) is visible on the next turn.
    turn_start = int(getattr(session, "_turn_start_offset", 0) or 0)
    for local_turn in range(session.cfg.max_turns):
        turn = turn_start + local_turn
        cfg = session.cfg
        # stop_resume delivery: the controller decided to intervene last
        # turn and requested a graceful hand-off. End before the next API
        # call; the stop-note in telemetry carries the resume payload.
        if getattr(session, "_adaptive_stop_requested", None):
            log.info("adaptive_stop: controller requested stop-resume hand-off")
            return SessionResult(
                turn,
                "adaptive_stop",
                done=False,
                total_prompt_tokens=total_prompt,
                total_completion_tokens=total_completion,
            )
        # The user_turn delivery mode queues a synthetic user message.
        # Append it at the next turn boundary and keep the session context.
        _pending_user_turn = getattr(session, "_adaptive_user_turn_pending", None)
        if _pending_user_turn:
            session.context.add_user(_pending_user_turn)
            session._adaptive_user_turn_pending = None
            log.info("adaptive_user_turn: injected fake-restart message at turn %d", turn)
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
        # Stamp the savings ledger with (session, turn) so every record
        # written by transforms downstream carries the turn context.
        get_ledger().set_turn(session._session_number, turn)
        # Inject keyword-triggered fragments (harness/injections.py)
        # against the latest user/tool content before the API call.
        # No-op when the subsystem is disabled or no fragments load.
        session._apply_injections()
        # ─── 0. GUARDRAIL: context fill (PRE-FLIGHT) ──────────────────
        # The post-flight check at the end of step 2 catches overflow
        # that develops during the response, but a tool result added
        # at the tail of the previous turn can already push the next
        # request past server n_ctx. Without this pre-flight, the API
        # call returns BadRequestError (exceed_context_size_error) and
        # the session ends with finish_reason="error" instead of the
        # cleaner "context_full". Same threshold as the post-flight
        # check (cfg.context_fill_ratio).
        # Sync cfg.context_size from the live server on first turn so
        # the fill_ratio gate measures against the actual window.
        session._get_server_ctx()
        # Pre-flight gate uses the larger of:
        #   - the prior turn's server-reported pt, and
        #   - the current context estimate.
        #
        # The server pt is exact for the last request, but it becomes stale
        # after the previous turn appends a large tool result. The local
        # estimate is less exact, but it sees the current message list and
        # prevents a post-tool-result overflow from reaching the API.
        if cfg.context_size > 0:
            live_pt = int(getattr(session, "_last_actual_prompt_tokens", 0) or 0)
            _tok_t0 = time.perf_counter()
            estimated_pt = _preflight_estimate(session)
            _phase_token_ms += (time.perf_counter() - _tok_t0) * 1000
            prev_estimate = getattr(session, "_preflight_prev_estimate", None)
            session._preflight_prev_estimate = estimated_pt
            density_hat = getattr(session, "_preflight_density", 0.25)
            # Remember what this gate saw so the post-response usage can
            # calibrate observed token density (tokens per appended char).
            session._preflight_gate_live = live_pt
            session._preflight_gate_chars_new = (
                max(0, estimated_pt - prev_estimate) * 4
                if prev_estimate is not None else 0)
            preflight_pt = _preflight_prompt_tokens(
                live_pt, estimated_pt, prev_estimate, density_hat)
            pre_fill = preflight_pt / cfg.context_size
            # Turn-level density: live_pt is the server's REAL token
            # count for the message list whose local estimate was
            # recorded at the previous turn's pre-flight. real/estimate
            # > 2 means the char-based projection undercounts this
            # conversation by more than 2x — log the anomaly to the
            # system log even when everything still fits.
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
            if pre_fill > cfg.context_fill_ratio:
                # ── Invariant backstop: one re-clip attempt before the
                # session ends. The overflow is usually one oversized
                # tool result; clip it in token space (head+tail with a
                # visible notice, targeting ctx/2 tokens for that single
                # message), re-project once, and only end context_full
                # if the projection STILL exceeds the window.
                clip = None
                if bool(getattr(cfg, "preflight_reclip_enabled", True)):
                    clip = preflight_reclip_oversized(session)
                if clip is not None:
                    _tok_t0 = time.perf_counter()
                    estimated_pt = _preflight_estimate(session)
                    _phase_token_ms += (time.perf_counter() - _tok_t0) * 1000
                    preflight_pt = max(live_pt, estimated_pt)
                    pre_fill = preflight_pt / cfg.context_size
                _shape, _quirk = provenance_for(
                    session, (clip or {}).get("tool_call_id", ""))
                still_over = pre_fill > cfg.context_fill_ratio
                get_system_log().event(
                    "preflight_overflow", turn=turn, live_pt=live_pt,
                    estimate_pt=estimated_pt, preflight_pt=preflight_pt,
                    density=density, ctx=cfg.context_size,
                    command_shape=_shape, quirk_hit=_quirk,
                    action="session_end" if still_over else "reclipped",
                )
                if still_over:
                    log.info(
                        "Context %.0f%% full pre-flight at turn %d, ending session "
                        "(live_pt=%d estimate_pt=%d preflight_pt=%d density=%.2f)",
                        pre_fill * 100, turn, live_pt, estimated_pt,
                        preflight_pt, density,
                    )
                    return SessionResult(turn, "context_full", done=False, total_prompt_tokens=total_prompt, total_completion_tokens=total_completion)
                log.info(
                    "Context pre-flight overflow at turn %d recovered by re-clip "
                    "(msg %d: %d -> %d tokens; estimate_pt=%d)",
                    turn, clip["index"], clip["orig_pt"], clip["new_pt"],
                    estimated_pt,
                )
        # ─── 1. API call (with transient-error retry) ────────────────
        _chat_t0 = time.perf_counter()
        chat_result = session._chat_with_retry(turn)
        _phase_chat_ms = (time.perf_counter() - _chat_t0) * 1000
        if chat_result is None:
            # chat_io may have set a more specific reason (e.g.
            # "compaction_overflow"); fall back to "error" otherwise.
            err_reason = getattr(session, "_last_chat_error_reason", None) or "error"
            return SessionResult(turn, err_reason, done=False, total_prompt_tokens=total_prompt, total_completion_tokens=total_completion)
        content = chat_result.content
        tool_calls = chat_result.tool_calls
        reason = chat_result.finish_reason
        prompt_tokens = chat_result.usage.prompt_tokens
        completion_tokens = chat_result.usage.completion_tokens
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
        )
        total_prompt += prompt_tokens
        total_completion += completion_tokens
        # Canonical pt signal — drives the post-flight gate AND the
        # next turn's compaction trigger.
        session._last_actual_prompt_tokens = int(prompt_tokens or 0)
        _observe_token_density(
            session,
            getattr(session, "_preflight_gate_live", 0),
            getattr(session, "_preflight_gate_chars_new", 0),
            int(prompt_tokens or 0),
        )
        session.context.add_assistant(
            session.client.build_assistant_message(content, tool_calls)
        )

        # ─── 2. GUARDRAIL: context fill (END tier) ───────────────────
        # Server-reported pt — accurate, no chars/4 underrun.
        if cfg.context_size > 0:
            fill = session._last_actual_prompt_tokens / cfg.context_size
            session._last_fill = fill
            if fill > cfg.context_fill_ratio:
                log.info("Context %.0f%% full at turn %d, ending session", fill * 100, turn)
                return SessionResult(turn, "context_full", done=False, total_prompt_tokens=total_prompt, total_completion_tokens=total_completion)

        # ─── 3. GUARDRAIL: intent_gate (BLOCK / END tiers) ───────────
        # Warmup: guardrails stay dormant through the first
        # guardrails_arm_after_turn turns (earliest observed hurdle onset
        # is turn 11; the opening naturally contains probes and rereads).
        guards_armed = turn > getattr(cfg, "guardrails_arm_after_turn", 0)
        intent_decision = turn_pre["intent_gate"](
            session._guards, cfg,
            turn=turn, content=content, tool_calls=tool_calls,
        ) if guards_armed else PASS
        if intent_decision.action in (Action.BLOCK, Action.END):
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
                session.context.add_tool_result(tc.id, intent_decision.text,
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
                        result=intent_decision.text,
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
            if intent_decision.action == Action.END:
                if not _defer_guard_end_during_active_watch(
                    session,
                    intent_decision,
                    guard_name="intent_gate",
                    turn=turn,
                ):
                    log.warning("Intent abort: %d consecutive silent rejections",
                                session._guards.consecutive_intent_rejections)
                    return SessionResult(turn, intent_decision.reason, done=False,
                                         total_prompt_tokens=total_prompt,
                                         total_completion_tokens=total_completion)
            _run_post_turn_hooks(session, turn)
            continue

        # ─── 4. Stop check (natural exit) ────────────────────────────
        if not tool_calls:
            if reason == "length":
                log.info("Response truncated at turn %d (max_tokens hit), ending session", turn)
                return SessionResult(turn, "length", done=False, total_prompt_tokens=total_prompt, total_completion_tokens=total_completion)
            # With implicit done enabled, `finish_reason="stop"` and no tool
            # calls count as success. Setting it to False
            # treats no-tool-calls as session end (`done=False`,
            # finish_reason="no_tool_call") so result attribution
            # tools (run_summary, paired-delta, failure-classifier) can
            # distinguish "model said done explicitly" from "model
            # silently fell off the conversation".
            allow_implicit = bool(getattr(cfg, "allow_implicit_done", True))
            if allow_implicit:
                log.info("Model stopped at turn %d (reason=%s) — implicit done", turn, reason)
                return SessionResult(turn, "stop", done=True, total_prompt_tokens=total_prompt, total_completion_tokens=total_completion)
            log.warning(
                "Model stopped at turn %d (reason=%s) without calling done() — "
                "session ended without success (allow_implicit_done=False)",
                turn, reason,
            )
            return SessionResult(turn, "no_tool_call", done=False, total_prompt_tokens=total_prompt, total_completion_tokens=total_completion)

        # ─── 5. GUARDRAIL: duplicate_guard (WARN / END tiers) ────────
        sig = tuple(_dedup_signature(tc) for tc in tool_calls)
        dup_decision = turn_pre["duplicate_guard"](
            session._guards, cfg, tool_calls_sig=sig
        ) if guards_armed else PASS
        if dup_decision.action == Action.END:
            if not _defer_guard_end_during_active_watch(
                session,
                dup_decision,
                guard_name="duplicate_guard",
                turn=turn,
            ):
                session._record_pressure_event(True)
                log.warning("Duplicate tool calls detected, aborting at turn %d", turn)
                return SessionResult(turn, dup_decision.reason, done=False,
                                     total_prompt_tokens=total_prompt,
                                     total_completion_tokens=total_completion)
        turn_warn_text = dup_decision.text if dup_decision.action == Action.WARN else ""
        turn_had_pressure = bool(turn_warn_text)

        # ─── 5b. GUARDRAIL: loop_detect (WARN / END tiers) ───────────
        # Tighter than duplicate_guard: fires at N consecutive identical
        # signatures (default 5) with a single recovery-inject before
        # hard abort. See guardrails.loop_detect for the contract.
        loop_decision = turn_pre["loop_detect"](
            session._guards, cfg, tool_calls_sig=sig
        ) if guards_armed else PASS
        if loop_decision.action == Action.END:
            if not _defer_guard_end_during_active_watch(
                session,
                loop_decision,
                guard_name="loop_detect",
                turn=turn,
            ):
                session._record_pressure_event(True)
                log.warning("Loop detected, aborting at turn %d", turn)
                return SessionResult(turn, loop_decision.reason, done=False,
                                     total_prompt_tokens=total_prompt,
                                     total_completion_tokens=total_completion)
        if loop_decision.action == Action.WARN:
            # Compose with any duplicate-guard warn already queued.
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
        if (
            cfg.parallel_readonly_enabled
            and len(tool_calls) > 1
            and all(tc.name in _READONLY_TOOLS for tc in tool_calls)
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
                futures = {
                    tc.id: _ex.submit(
                        dispatch, tc.name, tc.arguments,
                        cwd=session.cwd, cfg=cfg,
                        output_control=effective_output_control,
                        universal_rewrites=effective_universal_rewrites,
                        forbidden_rules=session.forbidden_rules if cfg.bash_quirks_forbidden_enabled else None,
                        redactions=session.redactions,
                        tool_registry=session._tool_registry,
                    )
                    for tc in tool_calls
                }
                for tc_id, fut in futures.items():
                    try:
                        preexecuted[tc_id] = fut.result()
                    except Exception as e:
                        preexecuted[tc_id] = f"ERROR: {e}"

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
            dispatch=dispatch,
            log=log,
            tool_pre=tool_pre,
            tool_post=tool_post,
            observers=observers,
            turn_had_pressure=turn_had_pressure,
        )
        for tc in tool_calls:
            args_summary = _summarize_args(tc.arguments, cfg.args_summary_chars)
            approval_allowed, approval_reason = approval_decision(
                runtime_mode=getattr(cfg, "runtime_mode", "measurement"),
                cwd=session.cwd,
                trace_path=session._trace_path,
                tool_name=tc.name,
                tool_args=tc.arguments,
                args_summary=args_summary,
            )
            if not approval_allowed:
                session.context.add_tool_result(
                    tc.id,
                    "APPROVAL REQUIRED: This tool call was not executed. "
                    f"Reason: {approval_reason}. Review it with `yuj show`, "
                    "approve it with `yuj approve <session_id>`, then resume.",
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
                    turn,
                    "approval_required",
                    done=False,
                    total_prompt_tokens=total_prompt,
                    total_completion_tokens=total_completion,
                )
            outcome = dispatch_one_tool_call(tc, state)
            if outcome.end:
                return SessionResult(
                    turn, outcome.reason, done=outcome.done,
                    total_prompt_tokens=total_prompt,
                    total_completion_tokens=total_completion,
                )
        session._record_pressure_event(state.turn_had_pressure)
        _run_post_turn_hooks(session, turn)
    # ─── 7. GUARDRAIL: max_turns (hard cap, END tier) ────────────────
    return SessionResult(
        turn_start + cfg.max_turns,
        "max_turns",
        done=False,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
    )
