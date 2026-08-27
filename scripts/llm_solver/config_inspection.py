"""Value-safe human and JSON views of a resolved Yuj configuration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from ._config_layers import ConfigValuePath, format_setting_path, iter_setting_leaves
from ._config_redaction import (
    REDACTED_VALUE,
    environment_string_values,
    redact_config_value,
    redact_derived_metadata,
    redact_sensitive_text,
    sensitive_string_values,
)
from .config import Config, PROJECT_ROOT, ResolvedConfig
from .harness._loop.model_role_runtime import validate_model_role_profiles
from .harness.subagents import load_agent_spec
from .server.profile_loader import load_profile


INSPECTION_SCHEMA = "yuj.config-inspection"
INSPECTION_SCHEMA_VERSION = 1


def _diagnostics(values: Iterable[Mapping[str, object]] | None) -> list[dict]:
    output = []
    for value in values or ():
        output.append(
            {
                "level": str(value.get("level") or "error"),
                "code": str(value.get("code") or "config_invalid"),
                "message": str(value.get("message") or "configuration is invalid"),
            }
        )
    return output


def _reference_names(
    references: Mapping[ConfigValuePath, str],
    prefix: tuple[str, ...],
) -> list[str]:
    return sorted(
        {
            name
            for path, name in references.items()
            if path[: len(prefix)] == prefix
        }
    )


def build_inspection_document(
    resolved: ResolvedConfig,
    *,
    success: bool,
    diagnostics: Iterable[Mapping[str, object]] | None = None,
    selection: Mapping[str, object] | None = None,
    references: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the stable version-1 machine document from resolved state."""
    protected_values = (
        *sensitive_string_values(
            resolved.data,
            environment_references=resolved.environment_references,
        ),
        *environment_string_values(
            resolved.data,
            environment_references=resolved.environment_references,
        ),
    )
    settings: list[dict[str, object]] = []
    for path, value in sorted(iter_setting_leaves(resolved.data)):
        source = resolved.provenance.get(path)
        if source is None:
            raise RuntimeError(
                f"configuration provenance missing for {format_setting_path(path)}"
            )
        redacted_value, redacted, reasons = redact_config_value(
            value,
            path=path,
            environment_references=resolved.environment_references,
        )
        entry: dict[str, object] = {
            "path": format_setting_path(path),
            "path_components": list(path),
            "value": redacted_value,
            "source_layer": source.layer_id,
            "redacted": redacted,
            "redaction_reasons": list(reasons),
        }
        environment_names = _reference_names(
            resolved.environment_references,
            path,
        )
        if len(environment_names) == 1:
            entry["environment_variable"] = environment_names[0]
        elif environment_names:
            entry["environment_variables"] = environment_names
        settings.append(entry)

    return {
        "schema": INSPECTION_SCHEMA,
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "status": "ok" if success else "error",
        "success": bool(success),
        "layers": [layer.as_dict() for layer in resolved.layers],
        "selection": redact_derived_metadata(
            dict(selection or {}),
            protected_values=protected_values,
        ),
        "settings": settings,
        "references": redact_derived_metadata(
            dict(references or {}),
            protected_values=protected_values,
        ),
        "diagnostics": redact_derived_metadata(
            _diagnostics(diagnostics),
            protected_values=protected_values,
        ),
    }


def build_error_document(
    message: str,
    *,
    code: str = "config_invalid",
) -> dict[str, object]:
    """Build the same stable envelope when resolution stops before a result."""
    return {
        "schema": INSPECTION_SCHEMA,
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "status": "error",
        "success": False,
        "layers": [],
        "selection": {},
        "settings": [],
        "references": {},
        "diagnostics": [
            {"level": "error", "code": code, "message": str(message)}
        ],
    }


def render_inspection_json(document: Mapping[str, object]) -> str:
    """Return deterministic compact JSON with exactly one trailing newline."""
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def _human_value(value: object) -> str:
    if value == REDACTED_VALUE:
        return REDACTED_VALUE
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def render_inspection_human(document: Mapping[str, object]) -> str:
    """Render a readable line-oriented explanation from the machine document."""
    success = bool(document.get("success"))
    lines = [f"Yuj configuration: {'valid' if success else 'invalid'}"]
    selection = document.get("selection")
    if isinstance(selection, Mapping) and selection:
        base = selection.get("base")
        context = selection.get("context_mode")
        if base:
            lines.append(f"Base: {base}")
        if context:
            source = selection.get("context_source")
            suffix = f" [{source}]" if source else ""
            lines.append(f"Context: {context}{suffix}")
        sandbox = selection.get("sandbox")
        if isinstance(sandbox, Mapping):
            supported = ", ".join(
                str(value) for value in sandbox.get("supported", [])
            ) or "none"
            installed = ", ".join(
                str(value) for value in sandbox.get("installed", [])
            ) or "none"
            available = ", ".join(
                str(value) for value in sandbox.get("available", [])
            ) or "none"
            unavailable = ", ".join(
                str(value) for value in sandbox.get("unavailable", [])
            ) or "none"
            lines.append(
                "Sandbox: "
                f"selected={sandbox.get('selected')} "
                f"resolved={sandbox.get('resolved')}"
            )
            lines.append(f"Sandbox supported: {supported}")
            lines.append(f"Sandbox installed: {installed}")
            lines.append(f"Sandbox available: {available}")
            lines.append(f"Sandbox unavailable: {unavailable}")

    layers = document.get("layers")
    if isinstance(layers, list) and layers:
        lines.append("Layers (low to high):")
        for layer in layers:
            if not isinstance(layer, Mapping):
                continue
            state = "applied" if layer.get("applied") else "not present"
            lines.append(
                f"  {layer.get('order')}: {layer.get('id')} "
                f"({layer.get('label')}; {state})"
            )

    settings = document.get("settings")
    if isinstance(settings, list) and settings:
        lines.append("Settings:")
        for setting in settings:
            if not isinstance(setting, Mapping):
                continue
            annotations = [str(setting.get("source_layer") or "unknown")]
            if setting.get("redacted"):
                annotations.append("redacted")
            environment = setting.get("environment_variable")
            if environment:
                annotations.append(f"environment {environment}")
            lines.append(
                f"  {setting.get('path')} = "
                f"{_human_value(setting.get('value'))} "
                f"[{'; '.join(annotations)}]"
            )

    references = document.get("references")
    if isinstance(references, Mapping):
        resources = references.get("runtime_resources")
        if isinstance(resources, Mapping):
            lines.append(
                "Runtime resources: "
                f"{resources.get('origin')} "
                f"({resources.get('root_resource_count')} root, "
                f"{resources.get('package_resource_count')} package)"
            )

    diagnostics = document.get("diagnostics")
    if isinstance(diagnostics, list) and diagnostics:
        lines.append("Diagnostics:")
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, Mapping):
                continue
            lines.append(
                f"  {diagnostic.get('level')}: {diagnostic.get('message')}"
            )
    else:
        lines.append("Diagnostics: none")
    return "\n".join(lines) + "\n"


def validate_configuration_references(
    cfg: Config,
    *,
    named_agents: Iterable[str] = (),
    project_root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Validate profile/role/agent references without constructing a client."""
    root = Path(project_root)
    profiles_dir = root / "profiles"
    profile_key = cfg.profile_name or cfg.model
    main_profile = load_profile(
        profile_key,
        profiles_dir,
        allow_base_fallback=not bool(cfg.profile_name),
    )
    validate_model_role_profiles(
        cfg=cfg,
        main_profile=main_profile,
        profiles_dir=profiles_dir,
        strict_references=True,
    )

    agents_dir = root / "agents"
    requested = {str(name) for name in named_agents}
    if cfg.tools_task_enabled and agents_dir.is_dir():
        requested.update(path.stem for path in agents_dir.glob("*.toml"))
    validated_agents = []
    for name in sorted(requested):
        spec = load_agent_spec(name, agents_dir)
        profile = load_profile(
            spec.model_profile,
            profiles_dir,
            allow_base_fallback=False,
        )
        validated_agents.append(
            {
                "name": spec.name,
                "profile": profile.name,
                "read_only": spec.read_only,
                "workspace": spec.workspace,
            }
        )
    return {
        "profile": {"requested": profile_key, "resolved": main_profile.name},
        "agents": validated_agents,
    }


def sanitize_diagnostic_message(
    message: object,
    *,
    project_root: Path = PROJECT_ROOT,
    resolved: ResolvedConfig | None = None,
) -> str:
    """Remove installation paths and resolved sensitive values from errors."""
    text = str(message).replace(str(Path(project_root).resolve()), "<yuj-root>")
    if resolved is None:
        return text
    return redact_sensitive_text(
        text,
        sensitive_values=sensitive_string_values(
            resolved.data,
            environment_references=resolved.environment_references,
        ),
        unquoted_values=environment_string_values(
            resolved.data,
            environment_references=resolved.environment_references,
        ),
    )


__all__ = [
    "INSPECTION_SCHEMA",
    "INSPECTION_SCHEMA_VERSION",
    "build_error_document",
    "build_inspection_document",
    "render_inspection_human",
    "render_inspection_json",
    "sanitize_diagnostic_message",
    "validate_configuration_references",
]
