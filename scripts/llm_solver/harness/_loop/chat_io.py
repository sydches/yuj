"""Chat-call retry loop with transient-error backoff."""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import openai

from ...server.types import Usage
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
    )


def _can_continue_raw(client) -> bool:
    profile = getattr(client, "profile", None)
    return bool(
        profile is not None
        and callable(getattr(profile, "normalize", None))
        and callable(getattr(client, "_prepare_profile_chat_request", None))
        and callable(getattr(client, "_call_raw_profile_request", None))
        and callable(getattr(client, "_turn_result_from_normalized", None))
    )


def _chat_with_length_continuation(session, outgoing, turn: int):
    """Execute one logical response through raw profile phases."""
    client = session.client
    cfg = session.cfg
    base_request = client._prepare_profile_chat_request(
        outgoing, session._tool_schemas
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
        max_attempts=cfg.length_continue_max,
        supports_prefill=client.profile.supports_prefill,
        call_model=call_model,
        normalize=client.profile.normalize,
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


def _has_fallback(session: "Session") -> bool:
    namespace = getattr(session, "__dict__", {})
    router = namespace.get("_model_role_router") if isinstance(namespace, dict) else None
    controller = getattr(router, "fallback_controller", None)
    return bool(controller is not None and controller.has_next("main"))


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
    """Call client.chat(), retrying on transient errors. Returns None on fatal."""
    while True:
        cfg = session.cfg
        max_retries = cfg.max_transient_retries
        backoff = cfg.retry_backoff
        restart_with_fallback = False
        for attempt in range(max_retries + 1):
            try:
                session._compaction_turn = turn
                outgoing = maybe_compact_messages(
                    session, session.context.get_messages()
                )
                if (
                    int(getattr(cfg, "length_continue_max", 0) or 0) > 0
                    and _can_continue_raw(session.client)
                ):
                    result = _chat_with_length_continuation(
                        session, outgoing, turn
                    )
                else:
                    result = session.client.chat(
                        outgoing, session._tool_schemas, turn=turn,
                    )
                if result is not None and getattr(
                    result, "finish_reason", ""
                ) == "replay_stop_turn":
                    from .replay_handover import maybe_handover
                    if maybe_handover(session, turn):
                        result = session.client.chat(
                            outgoing, session._tool_schemas, turn=turn,
                        )
                return result
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
        if restart_with_fallback:
            continue
        return None
