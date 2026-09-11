"""Atomic session rebinding for role-aware model fallback."""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any

from ..context import chars_div_4
from ..tool_validation import ToolSchemaSet
from .model_role_runtime import resolution_with_client_context
from .model_roles import (
    MAIN_MODEL_ROLE,
    ResolvedRoleClient,
    check_context_window,
)
from .profile_resolution import (
    _resolve_token_estimator,
    bind_effective_edit_format,
    build_plan_mode_schemas,
    build_tool_surface,
)

log = logging.getLogger(__name__)


def _stored_attr(owner: Any, name: str, default: Any = None) -> Any:
    namespace = getattr(owner, "__dict__", None)
    if isinstance(namespace, dict):
        return namespace.get(name, default)
    return default


def _apply_context_size(client: Any, context_size: int | None) -> None:
    """Constrain the replacement without replacing its declared permission."""
    from ...context_allocation import allocate_context
    client.cfg = allocate_context(
        client.cfg, observed=context_size, profile=getattr(client, "profile", None),
    )


def _live_context_size(routed: ResolvedRoleClient) -> int | None:
    """Query capacity without labelling a declaration as an observation."""
    query = getattr(routed.client, "query_server_context", None)
    if query is None:
        return None
    try:
        live = query()
    except Exception as exc:  # target health failure is not a harness crash
        log.error(
            "model fallback context query failed for %s: %s",
            routed.resolution.target.label(),
            exc,
        )
        return None
    if isinstance(live, bool) or not isinstance(live, int) or live <= 0:
        return None
    return live


def _candidate_prompt_tokens(
    session: Any,
    routed: ResolvedRoleClient,
    tool_schemas: list[dict],
) -> tuple[int, Any, Any]:
    """Estimate the replacement profile's actual wire messages and tools."""
    estimator = _resolve_token_estimator(routed.client) or chars_div_4
    canonical = [dict(message) for message in session.context.get_messages()]
    from ..request_counting import resolve_counter
    counter = resolve_counter(
        routed.client, routed.client.cfg,
        event_sink=lambda event, **fields: session._emit(
            event, session_number=session._session_number,
            turn_number=session._current_turn, **fields,
        ),
    )
    if counter is not None:
        return int(counter.count(canonical, tools=tool_schemas)), estimator, counter
    profile = routed.resolution.profile
    wire = (
        profile.denormalize_messages(canonical)
        if profile is not None
        else canonical
    )
    prompt_tokens = int(estimator(wire))
    prompt_tokens += sum(
        len(json.dumps(schema, sort_keys=True, default=str)) for schema in tool_schemas
    ) // 4
    return prompt_tokens, estimator, None


def _emit_transition(session: Any, turn: int, transition: Any, allocation=None) -> None:
    session._emit(
        "model_fallback",
        session_number=getattr(session, "_session_number", 0),
        turn_number=turn,
        context_allocation=allocation,
        **transition.trace_fields(),
    )


def activate_next_fallback(session: Any, turn: int, *, reason: str) -> bool:
    """Advance until one fallback fits, then atomically rebind the session.

    Returns ``False`` when no configured target remains. Every selected target
    is traced even if its live context window cannot accept the current prompt.
    """
    router = _stored_attr(session, "_model_role_router")
    if router is None:
        return False
    next_reason = reason
    while True:
        switched = router.switch_after_retry_exhaustion(
            MAIN_MODEL_ROLE,
            reason=next_reason,
        )
        if switched is None:
            return False
        routed = switched.routed_client
        live_context = _live_context_size(routed)
        skill_roots = tuple(
            getattr(session.cfg, "skills_readable_dirs", ()) or ()
        )
        if skill_roots != tuple(
            getattr(routed.client.cfg, "skills_readable_dirs", ()) or ()
        ):
            try:
                routed.client.cfg = replace(
                    routed.client.cfg,
                    skills_readable_dirs=skill_roots,
                )
            except TypeError:
                setattr(routed.client.cfg, "skills_readable_dirs", skill_roots)
        _apply_context_size(routed.client, live_context)
        routed.client.cfg = bind_effective_edit_format(
            routed.client.cfg, routed.client
        )
        effective_resolution = resolution_with_client_context(routed)
        routed = ResolvedRoleClient(routed.client, effective_resolution)
        transition = replace(
            switched.transition,
            to_resolution=effective_resolution,
        )
        initial_surface = build_tool_surface(
            routed.client.cfg, routed.client
        )
        from ..tool_loading import replace_tool_surface
        candidate_surface = replace_tool_surface(
            session._tool_surface,
            initial_surface.registered_schemas,
            lazy_loading_enabled=getattr(
                routed.client.cfg, "tools_lazy_loading_enabled", False
            ),
            active_default=getattr(
                routed.client.cfg, "tools_active_default", ()
            ),
            max_active_tools=initial_surface.max_active_tools,
        )
        candidate_schemas = candidate_surface.active_schemas
        candidate_schema_set = ToolSchemaSet.from_openai_tools(
            candidate_schemas
        )
        candidate_plan_schemas = (
            build_plan_mode_schemas(routed.client.cfg, routed.client)
            if bool(getattr(routed.client.cfg, "plan_mode_enabled", False))
            else []
        )
        candidate_plan_schema_set = ToolSchemaSet.from_openai_tools(
            candidate_plan_schemas or candidate_schemas
        )
        request_schemas = (
            candidate_plan_schemas
            if bool(getattr(session._plan_mode, "active", False))
            else candidate_schemas
        )
        prompt_tokens, estimator, counter = _candidate_prompt_tokens(
            session, routed, request_schemas,
        )
        window = check_context_window(
            prompt_tokens,
            effective_resolution,
            routed.client.cfg.context_fill_ratio,
        )
        _emit_transition(session, turn, transition, routed.client.cfg.context_allocation)
        if not window.fits:
            log.warning(
                "model fallback target %s cannot fit prompt: %d > %d",
                effective_resolution.target.label(),
                window.prompt_tokens,
                window.prompt_token_limit,
            )
            next_reason = "context_window_exceeded"
            continue

        # Rebind only after profile, live context, schema, and prompt checks
        # have all succeeded. Canonical context messages remain untouched.
        session.client = routed.client
        session.cfg = routed.client.cfg
        session._plan_mode.cfg = session.cfg
        session._active_model_resolution = effective_resolution
        session._active_model_role = effective_resolution.effective_role
        session._tool_surface = candidate_surface
        session._tool_schemas = candidate_schemas
        session._tool_schema_set = candidate_schema_set
        session._plan_tool_schemas = candidate_plan_schemas
        session._plan_tool_schema_set = candidate_plan_schema_set
        session.context.set_token_estimator(estimator)
        from ..request_counting import bind_session_counter
        bind_session_counter(session, counter, reset_observations=True)
        session._server_ctx_cache = session.cfg.context_size
        session._server_ctx_binding = None
        session._server_ctx_synced = True
        routed.client._model_role_resolution = effective_resolution
        log.warning(
            "model fallback activated: %s -> %s (%s)",
            transition.from_resolution.target.label(),
            transition.to_resolution.target.label(),
            transition.reason,
        )
        return True


__all__ = ["activate_next_fallback"]
