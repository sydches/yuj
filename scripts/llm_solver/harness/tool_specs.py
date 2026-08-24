"""First-class metadata for model-facing tools.

Handlers and JSON schemas still live in their existing modules. This file owns
the mechanical cross-cutting facts that consumers need to agree on: active
tool names, optional profile gates, parallel-read eligibility, mutation
classification, and native envelope prefixes.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    """Registry metadata for one active or compatibility tool name."""

    name: str
    active: bool = True
    parallel_read_safe: bool = False
    guardrail_mutation: bool = False
    action_write_like: bool = False
    profile_gate_attr: str | None = None
    cap_immune: bool = False
    native_envelope_prefix: str | None = None
    schema_order: int | None = None
    schema_surface: str = "native"


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("bash", schema_order=0),
    ToolSpec(
        "bash_poll",
        profile_gate_attr="tools_background_enabled",
        schema_order=1,
    ),
    ToolSpec(
        "bash_kill",
        profile_gate_attr="tools_background_enabled",
        schema_order=2,
    ),
    ToolSpec(
        "read",
        parallel_read_safe=True,
        schema_order=3,
    ),
    ToolSpec(
        "write",
        guardrail_mutation=True,
        action_write_like=True,
        schema_order=4,
    ),
    ToolSpec(
        "edit",
        guardrail_mutation=True,
        action_write_like=True,
        schema_order=5,
    ),
    ToolSpec(
        "glob",
        parallel_read_safe=True,
        schema_order=6,
    ),
    ToolSpec(
        "grep",
        parallel_read_safe=True,
        schema_order=7,
    ),
    ToolSpec(
        "write_todos",
        profile_gate_attr="tools_todos_enabled",
        schema_order=8,
    ),
    ToolSpec(
        "checkpoint",
        profile_gate_attr="tools_checkpoint_enabled",
        cap_immune=True,
        schema_order=9,
    ),
    ToolSpec(
        "rewind",
        profile_gate_attr="tools_checkpoint_enabled",
        cap_immune=True,
        schema_order=10,
    ),
    ToolSpec(
        "lsp",
        profile_gate_attr="lsp_tool_enabled",
        schema_order=15,
    ),
    ToolSpec(
        "think",
        profile_gate_attr="tools_think_enabled",
        schema_order=16,
    ),
    ToolSpec(
        "load_tools",
        profile_gate_attr="tools_lazy_loading_enabled",
        cap_immune=True,
        schema_order=17,
    ),
    ToolSpec(
        "done",
        cap_immune=True,
        schema_order=18,
        schema_surface="both",
    ),
    ToolSpec(
        "run_tests",
        profile_gate_attr="tools_run_tests_enabled",
        native_envelope_prefix="<test_results",
        schema_order=11,
    ),
    ToolSpec(
        "list_definitions",
        profile_gate_attr="tools_list_definitions_enabled",
        native_envelope_prefix="<list_definitions",
        schema_order=12,
    ),
    ToolSpec(
        "apply_patch",
        guardrail_mutation=True,
        action_write_like=True,
        native_envelope_prefix="<apply_patch",
        schema_order=13,
    ),
    ToolSpec(
        "udiff",
        guardrail_mutation=True,
        action_write_like=True,
        schema_order=14,
    ),
    ToolSpec(
        "list_functions",
        profile_gate_attr="tools_exec_cell_enabled",
        cap_immune=True,
        schema_order=0,
        schema_surface="code",
    ),
    ToolSpec(
        "get_function_details",
        profile_gate_attr="tools_exec_cell_enabled",
        cap_immune=True,
        schema_order=1,
        schema_surface="code",
    ),
    ToolSpec(
        "exec_cell",
        profile_gate_attr="tools_exec_cell_enabled",
        cap_immune=True,
        schema_order=2,
        schema_surface="code",
    ),
    # Compatibility names can appear in older traces or model profiles even
    # though the active public schema no longer declares handlers for them.
    ToolSpec(
        "str_replace",
        active=False,
        guardrail_mutation=True,
        action_write_like=True,
    ),
    ToolSpec(
        "create",
        active=False,
        action_write_like=True,
    ),
)

_ACTIVE_TOOL_SPECS = tuple(spec for spec in TOOL_SPECS if spec.active)

ACTIVE_TOOL_NAMES = tuple(spec.name for spec in _ACTIVE_TOOL_SPECS)


def _schema_tool_names(surface: str) -> tuple[str, ...]:
    return tuple(
        spec.name
        for spec in sorted(
            (
                item for item in _ACTIVE_TOOL_SPECS
                if item.schema_surface in {surface, "both"}
            ),
            key=lambda item: (
                item.schema_order if item.schema_order is not None else 10_000,
                item.name,
            ),
        )
    )


SCHEMA_TOOL_NAMES = _schema_tool_names("native")
CODE_MODE_SCHEMA_TOOL_NAMES = _schema_tool_names("code")
DECLARED_SCHEMA_TOOL_NAMES = (
    *SCHEMA_TOOL_NAMES,
    *(
        name for name in CODE_MODE_SCHEMA_TOOL_NAMES
        if name not in SCHEMA_TOOL_NAMES
    ),
)
EXEC_CELL_API_TOOL_NAMES = (
    "read",
    "grep",
    "glob",
    "list_definitions",
    "bash",
)
PARALLEL_READ_SAFE_TOOL_NAMES = frozenset(
    spec.name for spec in _ACTIVE_TOOL_SPECS if spec.parallel_read_safe
)
GUARDRAIL_MUTATION_TOOL_NAMES = frozenset(
    spec.name for spec in TOOL_SPECS if spec.guardrail_mutation
)
ACTION_WRITE_LIKE_TOOL_NAMES = frozenset(
    spec.name for spec in TOOL_SPECS if spec.action_write_like
)
PROFILE_GATE_ATTRS = {
    spec.name: spec.profile_gate_attr
    for spec in _ACTIVE_TOOL_SPECS
    if spec.profile_gate_attr is not None
}
CAP_IMMUNE_TOOL_NAMES = frozenset(
    spec.name for spec in _ACTIVE_TOOL_SPECS if spec.cap_immune
)
NATIVE_ENVELOPE_PREFIXES = tuple(
    spec.native_envelope_prefix
    for spec in _ACTIVE_TOOL_SPECS
    if spec.native_envelope_prefix is not None
)


def is_native_envelope(result: str) -> bool:
    """Return True when a tool result already owns a typed envelope."""
    return result.startswith("<tool_result") or result.startswith(
        NATIVE_ENVELOPE_PREFIXES
    )


__all__ = [
    "ACTIVE_TOOL_NAMES",
    "ACTION_WRITE_LIKE_TOOL_NAMES",
    "CAP_IMMUNE_TOOL_NAMES",
    "CODE_MODE_SCHEMA_TOOL_NAMES",
    "DECLARED_SCHEMA_TOOL_NAMES",
    "EXEC_CELL_API_TOOL_NAMES",
    "GUARDRAIL_MUTATION_TOOL_NAMES",
    "NATIVE_ENVELOPE_PREFIXES",
    "PARALLEL_READ_SAFE_TOOL_NAMES",
    "PROFILE_GATE_ATTRS",
    "SCHEMA_TOOL_NAMES",
    "TOOL_SPECS",
    "ToolSpec",
    "is_native_envelope",
]
