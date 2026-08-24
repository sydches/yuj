"""Profile-application helpers for the harness loop — extracted from loop.py."""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace

from ..._shared.edit_formats import (
    EDIT_FORMAT_TOOL_NAMES,
    EDIT_FORMAT_TO_TOOL,
    validate_edit_format,
)
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


def resolve_effective_edit_format(cfg, client) -> str:
    """Resolve CLI/config override, legacy selector, then profile default."""
    configured = str(getattr(cfg, "tools_edit_format", "") or "")
    if configured:
        return validate_edit_format(
            configured, field="config error: tools.edit_format"
        )

    # Compatibility for existing overlays. The old boolean now selects the
    # apply_patch dialect instead of exposing it beside another edit tool.
    if bool(getattr(cfg, "tools_apply_patch_enabled", False)):
        return "apply_patch"

    profile = _resolve_profile(client)
    inherited = str(getattr(profile, "edit_format", "") or "")
    if inherited:
        return validate_edit_format(
            inherited,
            field=f"profile {getattr(profile, 'name', '<unknown>')!r} edit_format",
        )
    return "exact"


def bind_effective_edit_format(cfg, client):
    """Return a Config carrying the profile-resolved runtime dialect."""
    effective = resolve_effective_edit_format(cfg, client)
    if getattr(cfg, "effective_edit_format", "") == effective:
        return cfg
    return replace(cfg, effective_edit_format=effective)


# Tools that must always be present in the schema list regardless of
# profile.max_tools.
_CAP_IMMUNE_TOOLS: frozenset[str] = CAP_IMMUNE_TOOL_NAMES


def _profile_tool_limit(client) -> int | None:
    """Return the profile's positive request-tool limit, if declared."""
    profile = _resolve_profile(client)
    if profile is None:
        return None
    max_tools = int(getattr(profile, "max_tools", 0) or 0)
    return max_tools if max_tools > 0 else None


def _apply_profile_tool_cap(
    tool_schemas: list[dict],
    client,
    *,
    extra_immune: frozenset[str] = frozenset(),
    priority_tools: frozenset[str] = frozenset(),
) -> list[dict]:
    """Apply profile max_tools cap to the declared tool surface.

    Cap-immune tools (`load_tools` when enabled, plus `done`) are partitioned to the
    head of the result so they always survive the truncation. Without
    this guard, a tight max_tools cap that happened to land before
    `done` would silently strip the session terminator.

    Conditional priority tools stay within the cap and retain their original
    relative order. Agent Skills use this to keep their activation mechanism,
    `read`, ahead of optional tools without changing disabled runs.
    """
    max_tools = _profile_tool_limit(client)
    if max_tools is None or len(tool_schemas) <= max_tools:
        return tool_schemas

    immune_names = _CAP_IMMUNE_TOOLS | extra_immune
    immune: list[dict] = []
    rest: list[dict] = []
    for schema in tool_schemas:
        name = schema.get("function", {}).get("name", "")
        (immune if name in immune_names else rest).append(schema)

    if priority_tools:
        prioritized = [
            schema for schema in rest
            if schema.get("function", {}).get("name", "") in priority_tools
        ]
        rest = prioritized + [
            schema for schema in rest
            if schema.get("function", {}).get("name", "") not in priority_tools
        ]

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


def _filter_edit_format_tools(
    tool_schemas: list[dict], cfg, client,
) -> list[dict]:
    """Keep exactly one model-facing mutation dialect."""
    selected_tool = EDIT_FORMAT_TO_TOOL[
        resolve_effective_edit_format(cfg, client)
    ]
    return [
        schema
        for schema in tool_schemas
        if (
            schema.get("function", {}).get("name", "")
            not in EDIT_FORMAT_TOOL_NAMES
            or schema.get("function", {}).get("name", "") == selected_tool
        )
    ]


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


def _skills_active(cfg) -> bool:
    """Return whether this run has a model-visible Agent Skills catalog."""
    return bool(
        getattr(cfg, "skills_enabled", False)
        and tuple(getattr(cfg, "skills_readable_dirs", ()) or ())
    )


def _require_skills_read(tool_schemas: list[dict], *, skills_active: bool) -> None:
    """Fail when a profile cap removed the only skill-body loading seam."""
    if skills_active and not any(
        schema.get("function", {}).get("name", "") in {"read", "exec_cell"}
        for schema in tool_schemas
    ):
        raise ValueError(
            "skills_enabled requires the read tool in the effective profile; "
            "increase profile max_tools"
        )


def apply_profile_to_schemas(tool_schemas: list[dict], cfg, client) -> list[dict]:
    """Apply the full profile-shaping pipeline to tool schemas.

    Composition order: filter disabled → select edit dialect → simplify
    (strip descriptions) → cap (preserve cap-immune tools at head). This is
    the single source of truth for both Session.__init__ and
    _record_session_start_costs. The same composition was once open-coded in
    two places and could drift.
    """
    skills_active = _skills_active(cfg)
    output = _apply_profile_tool_cap(
        _apply_profile_schema_simplify(
            _filter_edit_format_tools(
                _filter_disabled_tools(tool_schemas, cfg), cfg, client,
            ),
            client,
        ),
        client,
        priority_tools=frozenset({"read"}) if skills_active else frozenset(),
    )
    _require_skills_read(output, skills_active=skills_active)
    return output


def _build_registered_tool_schemas(
    cfg, client, tool_schemas: list[dict] | None = None
) -> list[dict]:
    """Apply gates, edit selection, and schema shape to the lazy registry."""
    if tool_schemas is None:
        from ..schemas import get_tool_schemas

        tool_schemas = get_tool_schemas(
            cfg.tool_desc,
            code_mode=bool(
                getattr(cfg, "tools_exec_cell_enabled", False)
            ),
        )

    return _apply_profile_schema_simplify(
        _filter_edit_format_tools(
            _filter_disabled_tools(tool_schemas, cfg), cfg, client,
        ),
        client,
    )


def build_plan_mode_schemas(
    cfg, client, tool_schemas: list[dict] | None = None,
) -> list[dict]:
    """Build the temporary native surface for a required planning phase.

    The phase is independent of the implementation edit dialect, deferred
    active set, and code-mode surface. The exact plan writer and explicit exit
    remain available even when a profile declares a smaller request-tool cap.
    """
    if tool_schemas is None:
        from ..schemas import get_tool_schemas

        tool_schemas = get_tool_schemas(cfg.tool_desc, code_mode=False)
    from ..plan_mode import filter_plan_mode_schemas

    skills_active = _skills_active(cfg)
    output = _apply_profile_tool_cap(
        _apply_profile_schema_simplify(
            filter_plan_mode_schemas(
                _filter_disabled_tools(tool_schemas, cfg), active=True,
            ),
            client,
        ),
        client,
        extra_immune=frozenset({"write"}),
        priority_tools=frozenset({"read"}) if skills_active else frozenset(),
    )
    _require_skills_read(output, skills_active=skills_active)
    return output


def build_tool_surface(
    cfg, client, tool_schemas: list[dict] | None = None
):
    """Build the registered and initial request-visible tool surface."""
    from ..tool_loading import ToolSurface

    lazy = bool(getattr(cfg, "tools_lazy_loading_enabled", False))
    code_mode = bool(getattr(cfg, "tools_exec_cell_enabled", False))
    if lazy and code_mode:
        raise ValueError(
            "tools.lazy_loading_enabled and tools.exec_cell_enabled "
            "cannot be enabled together"
        )
    registered = _build_registered_tool_schemas(cfg, client, tool_schemas)
    if not lazy:
        skills_active = _skills_active(cfg)
        registered = _apply_profile_tool_cap(
            registered,
            client,
            priority_tools=(
                frozenset({"read"}) if skills_active else frozenset()
            ),
        )
        _require_skills_read(registered, skills_active=skills_active)
    return ToolSurface(
        registered,
        lazy_loading_enabled=lazy,
        active_default=getattr(cfg, "tools_active_default", ()),
        max_active_tools=_profile_tool_limit(client) if lazy else None,
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
