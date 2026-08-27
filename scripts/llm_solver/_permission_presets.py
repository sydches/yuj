"""Fixed assistant permission presets expanded before policy compilation."""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from ._config_layers import (
    ConfigSource,
    SettingPath,
    apply_resolved_value,
)


class AssistantPermissionPreset(str, Enum):
    """Public names for the fixed assistant-only preset set."""

    READ_ONLY = "read-only"
    ASK_BEFORE_CHANGES = "ask-before-changes"
    ALLOW_EDITS = "allow-edits"


ASSISTANT_PERMISSION_PRESET_NAMES = tuple(
    preset.value for preset in AssistantPermissionPreset
)

_INSPECTION_TOOLS = (
    "read",
    "glob",
    "grep",
    "list_definitions",
    "list_functions",
    "get_function_details",
    "lsp",
)
_SESSION_CONTROL_TOOLS = (
    "ask_user",
    "checkpoint",
    "done",
    "exit_plan_mode",
    "load_tools",
    "think",
    "write_todos",
)
_FILE_EDIT_TOOLS = (
    "apply_patch",
    "edit",
    "notebook_edit",
    "udiff",
    "write",
)


@dataclass(frozen=True, slots=True)
class PermissionPresetSpec:
    """One immutable expansion into existing plan and permission controls."""

    plan_mode: str
    ask_fallback: str
    tool_decisions: Mapping[str, str]


def _decisions(
    *,
    catch_all: str,
    allow_file_edits: bool,
) -> Mapping[str, str]:
    ordered = {"*": catch_all}
    ordered.update((tool, "allow") for tool in _INSPECTION_TOOLS)
    ordered.update((tool, "allow") for tool in _SESSION_CONTROL_TOOLS)
    if allow_file_edits:
        ordered.update((tool, "allow") for tool in _FILE_EDIT_TOOLS)
    return MappingProxyType(ordered)


PERMISSION_PRESET_SPECS: Mapping[str, PermissionPresetSpec] = MappingProxyType(
    {
        AssistantPermissionPreset.READ_ONLY.value: PermissionPresetSpec(
            plan_mode="off",
            ask_fallback="deny",
            tool_decisions=_decisions(
                catch_all="deny",
                allow_file_edits=False,
            ),
        ),
        AssistantPermissionPreset.ASK_BEFORE_CHANGES.value: PermissionPresetSpec(
            plan_mode="required",
            ask_fallback="deny",
            tool_decisions=_decisions(
                catch_all="ask",
                allow_file_edits=False,
            ),
        ),
        AssistantPermissionPreset.ALLOW_EDITS.value: PermissionPresetSpec(
            plan_mode="off",
            ask_fallback="deny",
            tool_decisions=_decisions(
                catch_all="ask",
                allow_file_edits=True,
            ),
        ),
    }
)

_PRESET_LAYER_ID = "assistant-permission-preset"


def normalize_assistant_permission_preset(value: object) -> str:
    """Validate one selected name; an empty string means no preset."""
    if value == "":
        return ""
    if not isinstance(value, str) or value not in PERMISSION_PRESET_SPECS:
        raise ValueError(
            "config error: assistant.permission_preset must be one of: "
            + ", ".join(ASSISTANT_PERMISSION_PRESET_NAMES)
            + f"; got {value!r}."
        )
    return value


def _selected_preset(data: Mapping[str, object]) -> str:
    assistant = data.get("assistant", {})
    if not isinstance(assistant, Mapping):
        raise ValueError(
            "config error: assistant.permission_preset must be one of: "
            + ", ".join(ASSISTANT_PERMISSION_PRESET_NAMES)
            + f"; got {assistant!r}."
        )
    return normalize_assistant_permission_preset(
        assistant.get("permission_preset", "")
    )


def _runtime_mode(data: Mapping[str, object]) -> object:
    runtime = data.get("runtime", {})
    if not isinstance(runtime, Mapping):
        return None
    return runtime.get("mode", "measurement")


def _preset_rule_table(spec: PermissionPresetSpec) -> dict[str, object]:
    return {
        tool_pattern: {"*": decision}
        for tool_pattern, decision in spec.tool_decisions.items()
    }


def _reject_configured_preset_rules(
    provenance: Mapping[SettingPath, ConfigSource],
) -> None:
    if any(
        path[:2] == ("permissions", "preset_rules")
        for path in provenance
    ):
        raise ValueError(
            "config error: permissions.preset_rules is derived from "
            "assistant.permission_preset and cannot be set directly."
        )


def _preset_overrides_default(
    provenance: Mapping[SettingPath, ConfigSource],
    path: SettingPath,
) -> bool:
    source = provenance.get(path)
    return source is None or source.kind == "defaults"


def _insert_preset_layer(
    layers: Sequence[ConfigSource],
    preset_source: ConfigSource,
    provenance: MutableMapping[SettingPath, ConfigSource],
) -> tuple[ConfigSource, ...]:
    if any(layer.layer_id == _PRESET_LAYER_ID for layer in layers):
        raise ValueError(
            f"configuration layer ID {_PRESET_LAYER_ID!r} is reserved"
        )
    ordered: list[ConfigSource] = []
    inserted = False
    for layer in layers:
        ordered.append(layer)
        if layer.kind == "defaults" and not inserted:
            ordered.append(preset_source)
            inserted = True
    if not inserted:
        ordered.insert(0, preset_source)

    remapped = {
        source.layer_id: ConfigSource(
            source.layer_id,
            source.kind,
            source.label,
            order,
            source.applied,
        )
        for order, source in enumerate(ordered)
    }
    for path, source in tuple(provenance.items()):
        provenance[path] = remapped[source.layer_id]
    return tuple(remapped[source.layer_id] for source in ordered)


def expand_assistant_permission_preset(
    data: MutableMapping[str, object],
    provenance: MutableMapping[SettingPath, ConfigSource],
    layers: Sequence[ConfigSource],
) -> tuple[ConfigSource, ...]:
    """Expand one assistant preset into ordinary effective controls.

    Configured plan and fallback values replace preset values. Configured
    permission rules remain a separate, later policy table, which preserves
    their exact declaration order.
    Measurement mode validates the selected name but does not expand it.
    """
    _reject_configured_preset_rules(provenance)
    name = _selected_preset(data)
    if not name or _runtime_mode(data) != "assistant":
        return tuple(layers)

    spec = PERMISSION_PRESET_SPECS[name]
    preset_source = ConfigSource(
        _PRESET_LAYER_ID,
        "preset",
        f"assistant permission preset {name}",
        0,
        True,
    )
    if _preset_overrides_default(provenance, ("loop", "plan_mode")):
        apply_resolved_value(
            data,
            provenance,
            ("loop", "plan_mode"),
            spec.plan_mode,
            source=preset_source,
        )
    if _preset_overrides_default(
        provenance, ("permissions", "ask_fallback")
    ):
        apply_resolved_value(
            data,
            provenance,
            ("permissions", "ask_fallback"),
            spec.ask_fallback,
            source=preset_source,
        )
    apply_resolved_value(
        data,
        provenance,
        ("permissions", "preset_rules"),
        _preset_rule_table(spec),
        source=preset_source,
    )
    return _insert_preset_layer(layers, preset_source, provenance)


__all__ = [
    "ASSISTANT_PERMISSION_PRESET_NAMES",
    "AssistantPermissionPreset",
    "PERMISSION_PRESET_SPECS",
    "PermissionPresetSpec",
    "expand_assistant_permission_preset",
    "normalize_assistant_permission_preset",
]
