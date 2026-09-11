"""Bind harness budgeting to the active transport's counting capability."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def _bind_count_time(counter):
    if hasattr(counter, "count_timeout"):
        from .time_budget import execution_deadline, remaining_before
        counter.count_timeout = lambda cap: remaining_before(execution_deadline(), cap)


def has_reported_count(counter, count):
    """A returned count needs matching backend precision and numeric evidence."""
    record = getattr(counter, "last", None)
    return (
        type(count) is int and count >= 0 and isinstance(record, dict)
        and record.get("count_basis") == "backend_input_tokens"
        and record.get("count_precision") == "backend_reported"
        and type(record.get("prompt_tokens")) is int
        and record["prompt_tokens"] == count
    )


def resolve_counter(client, cfg, *, event_sink=None):
    mode = getattr(cfg, "tokenizer_id", "auto")
    if mode == "auto":
        # Require a declared adapter method; mock/proxy attribute synthesis is
        # not evidence of a supported backend capability.
        method = getattr(type(client), "get_request_token_counter", None)
        counter = method(client) if callable(method) else None
        if counter is not None:
            _bind_count_time(counter)
            counter.event_sink = event_sink
        else:
            log.info("transport has no request counter; using configured estimate")
        return counter
    if not mode:
        return None  # explicit legacy opt-out, retained for frozen experiments
    from .local_tokenizer import load

    counter = load(mode)
    counter.sync_chat_template(getattr(cfg, "base_url", ""))
    log.warning("explicit local tokenizer %s is an estimate; backend identity is unverified", counter.id)
    return counter


def bind_session_counter(session, counter=None, *, reset_observations=False):
    """Use current tools and plan mode, including after a model transition."""
    def emit(event, **fields):
        session._emit(event, session_number=session._session_number,
                      turn_number=session._current_turn, **fields)

    if counter is None:
        counter = resolve_counter(session.client, session.cfg, event_sink=emit)
    elif hasattr(counter, "event_sink"):
        _bind_count_time(counter)
        counter.event_sink = emit
    session._tokenizer = counter
    projection = getattr(session.context, "set_projection_request", None)
    if callable(projection):
        from .plan_mode import effective_model_tool_schemas

        projection(lambda: (session._tokenizer, session.cfg,
                            effective_model_tool_schemas(session)))
    retention = getattr(session.context, "set_retention_request", None)
    if callable(retention):
        from .plan_mode import effective_model_tool_schemas

        retention((lambda: (session._tokenizer, session.cfg,
                            effective_model_tool_schemas(session)))
                  if getattr(session.cfg, "tokenizer_id", "auto") == "auto" else None)
    if reset_observations:
        session._last_actual_prompt_tokens = 0
        session._preflight_prev_estimate = None
        session._prev_preflight_estimate_pt = 0
        session._preflight_density = 0.25
    if counter is not None:
        from .plan_mode import effective_model_tool_schemas

        session.context.set_token_estimator(lambda messages: int(counter.count(
            messages, tools=effective_model_tool_schemas(session),
        )))
    return counter


def observe_role_counter(owner, client, role):
    """Attribute side-request counting work to the owning session when present."""
    if "_session_number" not in getattr(owner, "__dict__", {}):
        return
    method = getattr(type(client), "get_request_token_counter", None)
    counter = method(client) if callable(method) else None
    if counter is not None:
        _bind_count_time(counter)
        counter.event_sink = lambda event, **fields: owner._emit(
            event, session_number=owner._session_number,
            turn_number=owner._current_turn, role=role, **fields,
        )
