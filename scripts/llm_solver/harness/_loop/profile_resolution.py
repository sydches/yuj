"""Profile-application helpers for the harness loop — extracted from loop.py."""
from __future__ import annotations

import logging
from collections.abc import Callable

from ..tool_specs import CAP_IMMUNE_TOOL_NAMES, PROFILE_GATE_ATTRS

log = logging.getLogger(__name__)


def _resolve_token_estimator(client) -> Callable[[list[dict]], int] | None:
    """Return the profile token estimator when the client explicitly carries one."""
    profile = _resolve_profile(client)
    estimator = getattr(profile, "estimate_tokens", None)
    if callable(estimator):
        return estimator
    return None


def _resolve_profile(client):
    """Return the profile object only when explicitly present on the client."""
    return getattr(client, "__dict__", {}).get("profile")


# Tools that must always be present in the schema list regardless of
# profile.max_tools.
_CAP_IMMUNE_TOOLS: frozenset[str] = CAP_IMMUNE_TOOL_NAMES


def _apply_profile_tool_cap(tool_schemas: list[dict], client) -> list[dict]:
    """Apply profile max_tools cap to the declared tool surface.

    Cap-immune tools (currently just `done`) are partitioned to the
    head of the result so they always survive the truncation. Without
    this guard, a tight max_tools cap that happened to land before
    `done` would silently strip the session terminator.
    """
    profile = _resolve_profile(client)
    if profile is None:
        return tool_schemas
    max_tools = int(getattr(profile, "max_tools", 0) or 0)
    if max_tools <= 0 or len(tool_schemas) <= max_tools:
        return tool_schemas

    immune: list[dict] = []
    rest: list[dict] = []
    for schema in tool_schemas:
        name = schema.get("function", {}).get("name", "")
        (immune if name in _CAP_IMMUNE_TOOLS else rest).append(schema)

    keep_rest = max(0, max_tools - len(immune))
    capped = immune + rest[:keep_rest]
    log.info(
        "Profile tool cap: sending %d/%d tools (cap=%d, immune=%d)",
        len(capped), len(tool_schemas), max_tools, len(immune),
    )
    return capped


def _gated_tool_attrs(cfg) -> dict[str, str]:
    """Return {tool_name: cfg_attr} from first-class tool specs."""
    return {
        name: attr
        for name, attr in PROFILE_GATE_ATTRS.items()
        if hasattr(cfg, attr)
    }


def _filter_disabled_tools(tool_schemas: list[dict], cfg) -> list[dict]:
    """Drop tool schemas whose enable-knob is false.

    The {tool_name -> cfg attr} mapping is declared in
    `harness.tool_specs`; cross-cutting flags are not tool specs, so they
    cannot be mistaken for per-tool gates.
    """
    gated = _gated_tool_attrs(cfg)
    out: list[dict] = []
    for schema in tool_schemas:
        name = schema.get("function", {}).get("name", "")
        attr = gated.get(name)
        if attr is not None and not bool(getattr(cfg, attr, False)):
            continue
        if name == "bash" and not bool(
            getattr(cfg, "tools_background_enabled", False)
        ):
            import copy
            schema = copy.deepcopy(schema)
            schema["function"]["parameters"]["properties"].pop(
                "background", None
            )
        out.append(schema)
    return out


def _simplify_tool_schema(schema: dict) -> dict:
    """Return a schema copy with description-like fields removed recursively."""
    if isinstance(schema, dict):
        out = {}
        for key, value in schema.items():
            if key in {"description", "examples"}:
                continue
            out[key] = _simplify_tool_schema(value)
        return out
    if isinstance(schema, list):
        return [_simplify_tool_schema(item) for item in schema]
    return schema


def _apply_profile_schema_simplify(tool_schemas: list[dict], client) -> list[dict]:
    """Apply profile simplify_schemas knob to tool schemas."""
    profile = _resolve_profile(client)
    if profile is None or not bool(getattr(profile, "simplify_schemas", False)):
        return tool_schemas
    log.info("Profile simplify_schemas enabled: stripping schema descriptions")
    return [_simplify_tool_schema(schema) for schema in tool_schemas]


def apply_profile_to_schemas(tool_schemas: list[dict], cfg, client) -> list[dict]:
    """Apply the full profile-shaping triple-stack to tool schemas.

    Composition order: filter disabled → simplify (strip descriptions)
    → cap (preserve cap-immune tools at head). Single source of truth
    for both Session.__init__ and _record_session_start_costs. The same
    composition was once open-coded in two places and could drift.
    """
    return _apply_profile_tool_cap(
        _apply_profile_schema_simplify(
            _filter_disabled_tools(tool_schemas, cfg),
            client,
        ),
        client,
    )


def _apply_profile_preamble(system_prompt: str, client) -> str:
    """Apply profile preamble as a prefixed system-prompt block."""
    profile = _resolve_profile(client)
    if profile is None:
        return system_prompt
    preamble = str(getattr(profile, "preamble", "") or "").strip()
    if not preamble:
        return system_prompt
    return preamble + "\n\n" + system_prompt
