"""Chat-call retry loop with transient-error backoff."""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import openai

from .compaction import CompactionOverflowError, maybe_compact_messages

if TYPE_CHECKING:
    from ..loop import Session

log = logging.getLogger(__name__)

_TRANSIENT_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,  # 5xx — z.ai and other providers emit transient 500s
)

_API_ERROR_DETAIL_CHARS = 400


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
    cfg = session.cfg
    max_retries = cfg.max_transient_retries
    backoff = cfg.retry_backoff
    for attempt in range(max_retries + 1):
        try:
            session._compaction_turn = turn
            outgoing = maybe_compact_messages(session, session.context.get_messages())
            result = session.client.chat(
                outgoing, session._tool_schemas, turn=turn,
            )
            if result is not None and getattr(result, "finish_reason", "") == "replay_stop_turn":
                # replay reached its stop turn; bundle capture (if enabled)
                # already happened at the stop turn's trace write — see
                # replay_handover.maybe_capture_at_stop. Armed handover swaps
                # in the live client and the turn re-runs.
                from .replay_handover import maybe_handover
                if maybe_handover(session, turn):
                    result = session.client.chat(
                        outgoing, session._tool_schemas, turn=turn,
                    )
            return result
        except CompactionOverflowError as e:
            # Terminal: prompt cannot fit even after truncating tool
            # messages within latest_pair. Better to end the session
            # with a debuggable reason than send and take a server 400.
            log.error("Compaction overflow on turn %d: %s", turn, e)
            _emit_api_error(session, turn, e, kind="compaction_overflow")
            session._last_chat_error_reason = "compaction_overflow"
            return None
        except _TRANSIENT_ERRORS as e:
            if attempt < max_retries:
                delay = backoff[attempt] if attempt < len(backoff) else backoff[-1]
                log.warning(
                    "Transient error on turn %d, retry %d/%d: %s",
                    turn, attempt + 1, max_retries, e,
                )
                time.sleep(delay)
            else:
                log.error(
                    "Transient error on turn %d, retries exhausted: %s", turn, e,
                )
                _emit_api_error(session, turn, e, kind="transient_exhausted")
                return None
        except Exception as e:
            log.error("Fatal API error on turn %d: %s", turn, e)
            _emit_api_error(session, turn, e, kind="fatal")
            return None
