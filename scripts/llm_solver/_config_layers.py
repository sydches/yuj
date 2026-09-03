"""Shared layered TOML resolution with leaf-level source provenance.

This module deliberately stops before projecting the merged tree into
``Config``.  The runtime loader and the public inspection command both consume
the same result, so inspection never has to infer winners from final values.
"""
from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from ._shared.toml_compat import tomllib


PathComponent = str | int
ConfigValuePath = tuple[PathComponent, ...]
SettingPath = tuple[str, ...]
_ENV_REFERENCE_RE = re.compile(r"\$ENV:([A-Za-z_][A-Za-z0-9_]*)\Z")
_BARE_TOML_KEY_RE = re.compile(r"[A-Za-z0-9_-]+\Z")


class ConfigResolutionError(ValueError):
    """A configuration layer or environment reference could not resolve."""


@dataclass(frozen=True, slots=True)
class ConfigSource:
    """One safe, ordered description of a configuration source layer."""

    layer_id: str
    kind: str
    label: str
    order: int
    applied: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.layer_id,
            "kind": self.kind,
            "label": self.label,
            "order": self.order,
            "applied": self.applied,
        }


@dataclass(frozen=True, slots=True)
class ConfigLayerSpec:
    """A path plus its public, path-free source identity."""

    path: Path
    layer_id: str
    kind: str
    label: str


@dataclass(frozen=True)
class LayeredConfigData:
    """Merged TOML values and exact winning source for each value leaf."""

    data: dict[str, object]
    provenance: dict[SettingPath, ConfigSource]
    layers: tuple[ConfigSource, ...]
    user_set_keys: frozenset[str]


def format_setting_path(path: Iterable[PathComponent]) -> str:
    """Render an unambiguous TOML-style path without revealing host paths."""
    rendered = ""
    for component in path:
        if isinstance(component, int):
            rendered += f"[{component}]"
            continue
        key = (
            component
            if _BARE_TOML_KEY_RE.fullmatch(component)
            else json.dumps(component, ensure_ascii=False)
        )
        rendered = f"{rendered}.{key}" if rendered else key
    return rendered or "<root>"


def iter_setting_leaves(
    value: object,
    prefix: SettingPath = (),
) -> Iterable[tuple[SettingPath, object]]:
    """Yield deterministic scalar/list/empty-table leaves from a TOML tree."""
    if isinstance(value, Mapping) and value:
        for key in sorted(value, key=str):
            yield from iter_setting_leaves(value[key], (*prefix, str(key)))
        return
    if prefix:
        yield prefix, value


def _read_toml(path: Path, *, label: str) -> dict[str, object]:
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except FileNotFoundError as exc:
        raise ConfigResolutionError(f"{label}: file not found") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigResolutionError(
            f"{label}: malformed TOML ({exc})"
        ) from exc
    except OSError as exc:
        detail = exc.strerror or type(exc).__name__
        raise ConfigResolutionError(f"{label}: cannot read file ({detail})") from exc


def _composition_references(
    document: Mapping[str, object],
    *,
    label: str,
) -> tuple[str, ...] | None:
    """Return one validated ordered composition, or ``None`` for a layer."""
    if "composition" not in document:
        return None
    if set(document) != {"composition"}:
        raise ConfigResolutionError(
            f"{label}: a composition file cannot also define settings"
        )
    composition = document["composition"]
    if not isinstance(composition, Mapping):
        raise ConfigResolutionError(f"{label}: [composition] must be a table")
    unknown = set(composition) - {"schema_version", "layers"}
    if unknown:
        rendered = ", ".join(sorted(str(key) for key in unknown))
        raise ConfigResolutionError(
            f"{label}: unknown [composition] key(s): {rendered}"
        )
    version = composition.get("schema_version")
    if type(version) is not int or version != 1:
        raise ConfigResolutionError(
            f"{label}: [composition].schema_version must be 1"
        )
    layers = composition.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ConfigResolutionError(
            f"{label}: [composition].layers must be a non-empty array"
        )
    references: list[str] = []
    for index, raw_reference in enumerate(layers, 1):
        if not isinstance(raw_reference, str) or not raw_reference.strip():
            raise ConfigResolutionError(
                f"{label}: composition.layers[{index}] must be a non-empty path"
            )
        reference = raw_reference.strip()
        if Path(reference).is_absolute():
            raise ConfigResolutionError(
                f"{label}: composition.layers[{index}] must be relative"
            )
        references.append(reference)
    return tuple(references)


def expand_config_compositions(
    user_layers: Iterable[ConfigLayerSpec],
) -> tuple[ConfigLayerSpec, ...]:
    """Replace each one-level composition with its ordered referenced layers."""
    expanded: list[ConfigLayerSpec] = []
    for spec in user_layers:
        document = _read_toml(Path(spec.path), label=spec.label)
        references = _composition_references(document, label=spec.label)
        if references is None:
            expanded.append(spec)
            continue
        parent = Path(spec.path).parent
        for index, reference in enumerate(references, 1):
            label = (
                f"{spec.label} -> composition.layers[{index}] "
                f"({reference})"
            )
            path = parent / reference
            referenced = _read_toml(path, label=label)
            if "composition" in referenced:
                raise ConfigResolutionError(
                    f"{label}: nested composition files are not supported"
                )
            expanded.append(
                ConfigLayerSpec(
                    path=path,
                    layer_id=f"{spec.layer_id}.composition-{index}",
                    kind="composition-layer",
                    label=label,
                )
            )
    return tuple(expanded)


def _clear_provenance(
    provenance: dict[SettingPath, ConfigSource],
    prefix: SettingPath,
) -> None:
    for path in tuple(provenance):
        if path[: len(prefix)] == prefix:
            del provenance[path]


def _mark_value(
    value: object,
    *,
    prefix: SettingPath,
    source: ConfigSource,
    provenance: dict[SettingPath, ConfigSource],
) -> None:
    leaves = tuple(iter_setting_leaves(value, prefix))
    if not leaves and prefix:
        provenance[prefix] = source
        return
    for path, _value in leaves:
        provenance[path] = source


def _deep_merge_with_provenance(
    base: dict[str, object],
    overlay: Mapping[str, object],
    *,
    source: ConfigSource,
    provenance: dict[SettingPath, ConfigSource],
    prefix: SettingPath = (),
) -> None:
    for raw_key, value in overlay.items():
        key = str(raw_key)
        path = (*prefix, key)
        current = base.get(key)
        if isinstance(value, Mapping) and isinstance(current, dict):
            if value:
                _deep_merge_with_provenance(
                    current,
                    value,
                    source=source,
                    provenance=provenance,
                    prefix=path,
                )
            continue
        _clear_provenance(provenance, path)
        copied = copy.deepcopy(value)
        base[key] = copied
        _mark_value(
            copied,
            prefix=path,
            source=source,
            provenance=provenance,
        )


def apply_resolved_value(
    data: dict[str, object],
    provenance: dict[SettingPath, ConfigSource],
    path: SettingPath,
    value: object,
    *,
    source: ConfigSource,
) -> None:
    """Apply one already-parsed override and update every replaced leaf."""
    if not path:
        raise ValueError("configuration override path must not be empty")
    current: dict[str, object] = data
    for index, component in enumerate(path[:-1]):
        child = current.get(component)
        if not isinstance(child, dict):
            _clear_provenance(provenance, path[: index + 1])
            child = {}
            current[component] = child
        current = child
    leaf = path[-1]
    _clear_provenance(provenance, path)
    copied = copy.deepcopy(value)
    current[leaf] = copied
    _mark_value(
        copied,
        prefix=path,
        source=source,
        provenance=provenance,
    )


def _collect_user_leaf_names(value: object, output: set[str]) -> None:
    if not isinstance(value, Mapping):
        return
    for key, child in value.items():
        if isinstance(child, Mapping):
            _collect_user_leaf_names(child, output)
        else:
            output.add(str(key))


def resolve_toml_layers(
    *,
    defaults_path: Path,
    local_path: Path,
    user_layers: Iterable[ConfigLayerSpec],
) -> LayeredConfigData:
    """Load and merge defaults, optional local settings, and user layers."""
    requested_specs = tuple(user_layers)
    ids = [
        "checked-in-defaults",
        "machine-local",
        *(spec.layer_id for spec in requested_specs),
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("configuration layer IDs must be unique")

    data: dict[str, object] = {}
    provenance: dict[SettingPath, ConfigSource] = {}
    layers: list[ConfigSource] = []
    user_set_keys: set[str] = set()

    defaults_source = ConfigSource(
        "checked-in-defaults", "defaults", "config.toml", 0, True
    )
    defaults = _read_toml(defaults_path, label=defaults_source.label)
    _deep_merge_with_provenance(
        data,
        defaults,
        source=defaults_source,
        provenance=provenance,
    )
    layers.append(defaults_source)

    local_applied = local_path.is_file()
    local_source = ConfigSource(
        "machine-local",
        "machine-local",
        "config.local.toml",
        1,
        local_applied,
    )
    if local_applied:
        local = _read_toml(local_path, label=local_source.label)
        _deep_merge_with_provenance(
            data,
            local,
            source=local_source,
            provenance=provenance,
        )
    layers.append(local_source)

    specs = expand_config_compositions(requested_specs)
    expanded_ids = [spec.layer_id for spec in specs]
    if len(expanded_ids) != len(set(expanded_ids)):
        raise ValueError("expanded configuration layer IDs must be unique")

    for offset, spec in enumerate(specs, 2):
        source = ConfigSource(
            spec.layer_id,
            spec.kind,
            spec.label,
            offset,
            True,
        )
        layer = _read_toml(Path(spec.path), label=spec.label)
        _collect_user_leaf_names(layer, user_set_keys)
        _deep_merge_with_provenance(
            data,
            layer,
            source=source,
            provenance=provenance,
        )
        layers.append(source)

    return LayeredConfigData(
        data=data,
        provenance=provenance,
        layers=tuple(layers),
        user_set_keys=frozenset(user_set_keys),
    )


def expand_environment_references(
    data: Mapping[str, object],
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[dict[str, object], dict[ConfigValuePath, str]]:
    """Resolve exact ``$ENV:NAME`` strings and retain value-free metadata."""
    source_environment = os.environ if environment is None else environment
    references: dict[ConfigValuePath, str] = {}

    def visit(value: object, path: ConfigValuePath) -> object:
        if isinstance(value, str) and value.startswith("$ENV:"):
            match = _ENV_REFERENCE_RE.fullmatch(value)
            if match is None:
                raise ConfigResolutionError(
                    f"{format_setting_path(path)}: invalid environment reference; "
                    "expected $ENV:NAME"
                )
            name = match.group(1)
            if name not in source_environment:
                raise ConfigResolutionError(
                    f"{format_setting_path(path)}: environment variable {name} "
                    "is not set"
                )
            references[path] = name
            return source_environment[name]
        if isinstance(value, Mapping):
            return {
                str(key): visit(child, (*path, str(key)))
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [
                visit(child, (*path, index))
                for index, child in enumerate(value)
            ]
        if isinstance(value, tuple):
            return tuple(
                visit(child, (*path, index))
                for index, child in enumerate(value)
            )
        return copy.deepcopy(value)

    return visit(data, ()), references  # type: ignore[return-value]


__all__ = [
    "ConfigLayerSpec",
    "ConfigResolutionError",
    "ConfigSource",
    "ConfigValuePath",
    "LayeredConfigData",
    "SettingPath",
    "apply_resolved_value",
    "expand_config_compositions",
    "expand_environment_references",
    "format_setting_path",
    "iter_setting_leaves",
    "resolve_toml_layers",
]
