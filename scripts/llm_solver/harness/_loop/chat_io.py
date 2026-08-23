"""Chat-call retry loop with transient-error backoff."""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import openai

from .compaction import CompactionOverflowError, maybe_compact_messages
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
