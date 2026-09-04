"""Chat-call retry loop with transient-error backoff."""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import openai

from ...server._streaming import StreamRuleInterrupt
from ...server.types import Usage
from ..stream_rules import format_interrupt_fragment
from .compaction import CompactionOverflowError, maybe_compact_messages
from .length_continuation import continue_length_response
from .model_fallback_runtime import activate_next_fallback

if TYPE_CHECKING:
    from ..loop import Session

log = logging.getLogger(__name__)

_TRANSIENT_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,  # 5xx — z.ai and other providers emit transient 500s
)

_API_ERROR_DETAIL_CHARS = 400


def _aggregate_usage(usages: list[Usage]) -> Usage:
    """Sum raw-call usage while preserving unknown cache telemetry."""
    if not usages or not all(isinstance(usage, Usage) for usage in usages):
        raise TypeError("length continuation requires canonical Usage per call")
    prompt_tokens = sum(usage.prompt_tokens for usage in usages)
    completion_tokens = sum(usage.completion_tokens for usage in usages)
    if all(usage.cached_tokens is not None for usage in usages):
        cached_tokens = sum(int(usage.cached_tokens or 0) for usage in usages)
        cache_hit_ratio = (
            cached_tokens / prompt_tokens if prompt_tokens > 0 else None
        )
    else:
        cached_tokens = None
        cache_hit_ratio = None
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        cache_hit_ratio=cache_hit_ratio,
        prompt_tokens_known=all(
            getattr(usage, "prompt_tokens_known", True) is True
            for usage in usages
        ),
        completion_tokens_known=all(
            getattr(usage, "completion_tokens_known", True) is True
            for usage in usages
        ),
    )


def _can_continue_raw(client) -> bool:
    profile = getattr(client, "profile", None)
    return bool(
        profile is not None
        and getattr(profile, "supports_prefill", False) is True
        and callable(getattr(profile, "normalize", None))
        and callable(getattr(client, "_prepare_profile_chat_request", None))
        and callable(getattr(client, "_call_raw_profile_request", None))
        and callable(getattr(client, "_turn_result_from_normalized", None))
    )


def _chat_with_length_continuation(
    session, outgoing, turn: int, *, max_attempts: int,
):
    """Execute one logical response through raw profile phases."""
    from ..plan_mode import effective_model_tool_schemas

    client = session.client
    base_request = client._prepare_profile_chat_request(
        outgoing, effective_model_tool_schemas(session)
    )
    initial_raw = client._call_raw_profile_request(base_request)
    usages = [initial_raw.get("usage")]

    def call_model(request):
        response = client._call_raw_profile_request(request)
        usages.append(response.get("usage"))
        return response

    result = continue_length_response(
        base_request=base_request,
        initial_response=initial_raw,
        max_attempts=max_attempts,
        supports_prefill=client.profile.supports_prefill,
        call_model=call_model,
        normalize=client.profile.normalize,
        context_size=session.cfg.context_size,
    )
    for attempt in result.attempts:
        session._emit(
            "length_continue",
            session_number=session._session_number,
            turn_number=turn,
            attempt=attempt.attempt,
            tokens=attempt.tokens,
            prompt_tokens=attempt.prompt_tokens,
            overlap_chars=attempt.overlap_chars,
            finish_reason=attempt.finish_reason,
        )
    session._length_continuation_count += result.continuation_count
    return client._turn_result_from_normalized(
        result.response,
        _aggregate_usage(usages),
        turn,
        fallback_finish_reason=result.raw_response["finish_reason"],
    )


def _fallback_reason(exc: Exception, default: str | None) -> str | None:
    """Classify only failures that are safe to move to another local model."""
    detail = str(exc).lower()
    context_markers = (
        "context size",
        "context length",
        "exceed_context",
        "maximum context",
        "too many tokens",
    )
    oom_markers = (
        "out of memory",
        "cuda oom",
        "ggml_status_alloc_failed",
        "failed to allocate",
    )
    if any(marker in detail for marker in context_markers):
        return "context_overflow"
    if any(marker in detail for marker in oom_markers):
        return "server_oom"
    return default


def _length_continue_max(cfg) -> int:
    """Return an enabled bound only for the validated primitive shape.

    Lightweight integrations historically supply partial ``MagicMock`` or
    namespace configs. A synthetic attribute must not opt into a model call;
    production config validation still rejects invalid explicit values.
    """
    value = getattr(cfg, "length_continue_max", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _has_fallback(session: "Session") -> bool:
    namespace = getattr(session, "__dict__", {})
    router = namespace.get("_model_role_router") if isinstance(namespace, dict) else None
    controller = getattr(router, "fallback_controller", None)
    return bool(controller is not None and controller.has_next("main"))


def _release_protected_correction(
    session: "Session", outgoing: list[dict]
) -> None:
    """Validate the exact correction tail, then end its special boundary."""
    text = getattr(session, "_protected_correction_text", None)
    if not isinstance(text, str) or not text:
        return
    if (
        not outgoing
        or outgoing[-1].get("role") != "user"
        or outgoing[-1].get("content") != text
    ):
        raise RuntimeError(
            "pending correction changed before its model request"
        )
    session._protected_correction_text = None


def _emit_api_error(session: "Session", turn: int, exc: Exception, *, kind: str) -> None:
    """Record API failure details in the trace. Never raise."""
    try:
        session._emit(
            "api_error",
            session_number=getattr(session, "_session_number", None),
            turn_number=turn,
            error_type=type(exc).__name__,
            error_kind=kind,
            http_status=getattr(exc, "status_code", None),
            detail=str(exc)[:_API_ERROR_DETAIL_CHARS],
        )
    except Exception:  # pragma: no cover — tracing must not mask the error
        log.exception("failed to emit api_error trace event")


def chat_with_retry(session: "Session", turn: int):
    """Call client.chat(), retrying on transient errors."""
    while True:
        cfg = session.cfg
        max_retries = cfg.max_transient_retries
        backoff = cfg.retry_backoff
        restart_with_fallback = False
        restart_after_stream_rule = False
        for attempt in range(max_retries + 1):
            try:
                session._compaction_turn = turn
                outgoing = maybe_compact_messages(
                    session, session.context.get_messages()
                )
                _release_protected_correction(session, outgoing)
                length_continue_max = _length_continue_max(cfg)
                runtime = getattr(session, "_stream_rule_runtime", None)
                if runtime is not None:
                    runtime.begin_attempt()
                client_state = getattr(session.client, "__dict__", {})
                observer_supported = (
                    runtime is not None
                    and isinstance(client_state, dict)
                    and "_stream_observer" in client_state
                )
                prior_observer = (
                    client_state.get("_stream_observer")
                    if observer_supported else None
                )
                if observer_supported:
                    session.client._stream_observer = (
                        lambda delta: runtime.observe(delta, turn=turn)
                    )
                try:
                    if length_continue_max > 0 and _can_continue_raw(session.client):
                        result = _chat_with_length_continuation(
                            session, outgoing, turn,
                            max_attempts=length_continue_max,
                        )
                    else:
                        from ..plan_mode import effective_model_tool_schemas
                        result = session.client.chat(
                            outgoing, effective_model_tool_schemas(session), turn=turn,
                        )
                finally:
                    if observer_supported:
                        session.client._stream_observer = prior_observer
                if result is not None and getattr(
                    result, "finish_reason", ""
                ) == "replay_stop_turn":
                    from .replay_handover import maybe_handover
                    if maybe_handover(session, turn):
                        result = session.client.chat(
                            outgoing, effective_model_tool_schemas(session), turn=turn,
                        )
                if result is not None and runtime is not None:
                    records = runtime.accept_response(
                        result,
                        turn=turn,
                        streamed=bool(
                            getattr(session.client, "_last_call_streamed", False)
                        ),
                        replay=bool(getattr(session.client, "is_replay", False)),
                    )
                    session._record_stream_rule_matches(records, turn=turn)
                return result
            except StreamRuleInterrupt as exc:
                records = tuple(exc.matches)
                session._record_stream_rule_matches(records, turn=turn)
                if getattr(cfg, "stream_rules_context_mode", "discard") == "keep":
                    partial = exc.partial_response
                    partial_content = None
                    if partial is not None and partial.choices:
                        partial_content = partial.choices[0].message.content
                    # Incomplete tool calls cannot safely enter an OpenAI
                    # message history without matching tool results. Keep only
                    # partial prose; discard mode keeps neither.
                    if partial_content:
                        session.context.add_assistant({
                            "role": "assistant",
                            "content": partial_content,
                        })
                inserted = "\n\n".join(
                    format_interrupt_fragment(record) for record in records
                )
                session.context.add_injected_fragment(inserted)
                from ..savings import get_ledger
                get_ledger().record_transform(
                    bucket="stream_rule_intervention",
                    layer="harness",
                    mechanism="retry_interrupt_fragment",
                    before="",
                    after=inserted,
                    surface="injected_message",
                    change_count=len(records),
                    ctx={
                        "rules": [
                            str(record.get("rule") or "")
                            for record in records
                        ],
                        "delivery": "retry",
                    },
                )
                runtime = getattr(session, "_stream_rule_runtime", None)
                if runtime is not None:
                    runtime.mark_injected(records, turn=turn)
                session._record_stream_rule_injection(
                    records, turn=turn, delivery="retry"
                )
                log.info(
                    "Stream rule interrupted turn %d; retrying with rule(s): %s",
                    turn,
                    ", ".join(str(record.get("rule") or "") for record in records),
                )
                restart_after_stream_rule = True
                break
            except CompactionOverflowError as exc:
                log.error("Compaction overflow on turn %d: %s", turn, exc)
                _emit_api_error(
                    session, turn, exc, kind="compaction_overflow"
                )
                if activate_next_fallback(
                    session, turn, reason="context_overflow"
                ):
                    restart_with_fallback = True
                    break
                session._last_chat_error_reason = "compaction_overflow"
                return None
            except _TRANSIENT_ERRORS as exc:
                if attempt < max_retries:
                    delay = (
                        backoff[attempt]
                        if attempt < len(backoff)
                        else (backoff[-1] if backoff else 0)
                    )
                    log.warning(
                        "Transient error on turn %d, retry %d/%d: %s",
                        turn, attempt + 1, max_retries, exc,
                    )
                    time.sleep(delay)
                    continue
                reason = _fallback_reason(exc, "transient_exhausted")
                log.error(
                    "Transient error on turn %d, retries exhausted: %s",
                    turn, exc,
                )
                _emit_api_error(session, turn, exc, kind=reason)
                if activate_next_fallback(session, turn, reason=reason):
                    restart_with_fallback = True
                    break
                return None
            except openai.BadRequestError as exc:
                reason = _fallback_reason(exc, None)
                if reason == "context_overflow":
                    _emit_api_error(session, turn, exc, kind=reason)
                    if _has_fallback(session) and activate_next_fallback(
                        session, turn, reason=reason
                    ):
                        restart_with_fallback = True
                        break
                    log.warning("Context full on turn %d: %s", turn, exc)
                    session._last_chat_error_reason = "context_full"
                    return None
                if reason is not None and _has_fallback(session):
                    _emit_api_error(session, turn, exc, kind=reason)
                    if activate_next_fallback(session, turn, reason=reason):
                        restart_with_fallback = True
                        break
                else:
                    log.error("Fatal API error on turn %d: %s", turn, exc)
                    _emit_api_error(session, turn, exc, kind="fatal")
                return None
            except Exception as exc:
                log.error("Fatal API error on turn %d: %s", turn, exc)
                _emit_api_error(session, turn, exc, kind="fatal")
                return None
        if restart_with_fallback or restart_after_stream_rule:
            continue
        return None
