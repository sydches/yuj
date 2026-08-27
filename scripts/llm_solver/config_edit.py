"""Validated, comment-preserving edits of one public TOML setting."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ._config_layers import ConfigLayerSpec, ConfigValuePath
from ._config_redaction import REDACTED_VALUE, redact_config_value
from ._shared.paths import default_config_path, local_config_path
from ._shared.toml_compat import tomllib
from .config import ResolvedConfig, resolve_config


EDIT_SCHEMA = "yuj.config-edit"
EDIT_SCHEMA_VERSION = 1
_BARE_PATH = re.compile(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*\Z")
_ENV_REFERENCE = re.compile(r"\$ENV:([A-Za-z_][A-Za-z0-9_]*)\Z")
_ASSIGNMENT = re.compile(
    r"^(?P<prefix>\s*(?P<key>[A-Za-z0-9_-]+"
    r"(?:\s*\.\s*[A-Za-z0-9_-]+)*)\s*=\s*)(?P<rhs>.*)$"
)
_TABLE = re.compile(
    r"^\s*\[(?P<name>[A-Za-z0-9_-]+"
    r"(?:\s*\.\s*[A-Za-z0-9_-]+)*)\]\s*(?:#.*)?(?:\r?\n)?\Z"
)
_ARRAY_TABLE = re.compile(r"^\s*\[\[")
_MISSING = object()


class ConfigEditError(ValueError):
    """A requested persistent configuration edit is unsafe or invalid."""


@dataclass(frozen=True, slots=True)
class EditDestination:
    """One explicitly selected persistent configuration layer."""

    layer_id: str
    label: str
    path: Path
    user_layer_index: int | None
    create_parent: bool = False


@dataclass(frozen=True, slots=True)
class AssignmentSpan:
    start: int
    end: int
    prefix: str
    rhs: str


def parse_setting_path(value: str) -> tuple[str, ...]:
    """Parse the public bare dotted setting notation."""
    text = str(value).strip()
    if not text or not _BARE_PATH.fullmatch(text):
        raise ConfigEditError(
            "setting must be a bare dotted path such as loop.max_turns"
        )
    return tuple(text.split("."))


def select_destination(
    value: str,
    *,
    overlay_paths: Sequence[Path],
) -> EditDestination:
    """Resolve a writable layer name without accepting an arbitrary path."""
    name = str(value).strip()
    if name == "machine-local":
        return EditDestination(
            layer_id="machine-local",
            label="config.local.toml",
            path=local_config_path().expanduser().resolve(),
            user_layer_index=None,
            create_parent=True,
        )
    if name in {"checked-in-defaults", "base", "command-line"}:
        raise ConfigEditError(f"configuration layer {name!r} is read-only")
    match = re.fullmatch(r"overlay-([1-9][0-9]*)", name)
    if match is None:
        raise ConfigEditError(
            "layer must be machine-local or overlay-N from a supplied --config"
        )
    number = int(match.group(1))
    if number > len(overlay_paths):
        raise ConfigEditError(
            f"configuration layer {name!r} has no matching --config path"
        )
    requested = Path(overlay_paths[number - 1]).expanduser()
    path = Path(os.path.abspath(os.fspath(requested)))
    identity = path.resolve(strict=False)
    overlay_identities = [
        Path(os.path.abspath(os.fspath(Path(item).expanduser()))).resolve(
            strict=False
        )
        for item in overlay_paths
    ]
    if overlay_identities.count(identity) != 1:
        raise ConfigEditError(
            "the selected destination appears more than once in --config"
        )
    if identity == local_config_path().resolve(strict=False):
        raise ConfigEditError(
            "select machine-local instead of passing config.local.toml as an overlay"
        )
    protected = {
        default_config_path().resolve(),
    }
    if path in protected:
        raise ConfigEditError(f"configuration layer {name!r} is read-only")
    return EditDestination(
        layer_id=name,
        label=f"--config[{number}]",
        path=path,
        user_layer_index=number,
    )


def parse_typed_value(raw: str, *, expected: object) -> object:
    """Parse one CLI value using the checked-in setting's public type."""
    if isinstance(expected, str):
        return str(raw)
    try:
        value = tomllib.loads(f"value = {raw}\n")["value"]
    except tomllib.TOMLDecodeError as exc:
        raise ConfigEditError(
            "value is not valid TOML for this setting's type"
        ) from exc
    if not _same_public_type(value, expected):
        raise ConfigEditError(
            "value has the wrong type: expected "
            f"{_type_name(expected)}, got {_type_name(value)}"
        )
    return value


def _same_public_type(value: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return isinstance(value, bool)
    if isinstance(expected, int) and not isinstance(expected, bool):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(expected, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(expected, list):
        return isinstance(value, list)
    if isinstance(expected, dict):
        return isinstance(value, dict)
    return type(value) is type(expected)


def _type_name(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "table"
    return type(value).__name__


def _lookup(value: object, path: Sequence[str]) -> object:
    current = value
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            return _MISSING
        current = current[component]
    return current


def _parse_document(text: str, *, label: str) -> dict[str, object]:
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigEditError(f"{label} is malformed TOML ({exc})") from exc
    return dict(parsed)


def _documented_value(path: Sequence[str]) -> object:
    defaults = default_config_path().read_text(encoding="utf-8")
    parsed = _parse_document(defaults, label="config.toml")
    value = _lookup(parsed, path)
    if value is _MISSING or _find_assignment(defaults, path) is None:
        dotted = ".".join(path)
        raise ConfigEditError(f"unknown or non-editable setting {dotted!r}")
    return value


def _parts(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split("."))


def _assignment_complete(rhs: str) -> bool:
    try:
        tomllib.loads(f"value = {rhs}")
    except tomllib.TOMLDecodeError:
        return False
    return True


def _find_assignment(
    text: str,
    path: Sequence[str],
) -> AssignmentSpan | None:
    lines = text.splitlines(keepends=True)
    section: tuple[str, ...] | None = ()
    index = 0
    while index < len(lines):
        line = lines[index]
        if _ARRAY_TABLE.match(line):
            section = None
            index += 1
            continue
        table = _TABLE.match(line)
        if table is not None:
            section = _parts(table.group("name"))
            index += 1
            continue
        assignment = _ASSIGNMENT.match(line)
        if assignment is None:
            index += 1
            continue
        end = index + 1
        prefix = assignment.group("prefix")
        rhs = line[len(prefix) :]
        while not _assignment_complete(rhs) and end < len(lines):
            rhs += lines[end]
            end += 1
        if section is not None:
            candidate = (*section, *_parts(assignment.group("key")))
            if tuple(path) == candidate:
                return AssignmentSpan(
                    start=index,
                    end=end,
                    prefix=prefix,
                    rhs=rhs,
                )
        index = end
    return None


def _inline_comment(rhs: str) -> str:
    if "\n" in rhs or "\r" in rhs:
        return ""
    quote = ""
    escaped = False
    for index, character in enumerate(rhs):
        if escaped:
            escaped = False
            continue
        if quote == '"' and character == "\\":
            escaped = True
            continue
        if quote:
            if character == quote:
                quote = ""
            continue
        if character in {'"', "'"}:
            quote = character
            continue
        if character == "#":
            start = index
            while start > 0 and rhs[start - 1] in " \t":
                start -= 1
            return rhs[start:]
    return ""


def render_toml_value(value: object) -> str:
    """Render one parsed TOML value without changing surrounding text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigEditError("non-finite floating-point values are not supported")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, list):
        return "[" + ", ".join(render_toml_value(item) for item in value) + "]"
    if isinstance(value, Mapping):
        entries = []
        for key, child in value.items():
            rendered_key = (
                str(key)
                if re.fullmatch(r"[A-Za-z0-9_-]+", str(key))
                else json.dumps(str(key), ensure_ascii=False)
            )
            entries.append(f"{rendered_key} = {render_toml_value(child)}")
        return "{ " + ", ".join(entries) + " }"
    raise ConfigEditError(f"unsupported TOML value type {_type_name(value)}")


def edit_toml_text(
    text: str,
    *,
    path: Sequence[str],
    value: object = _MISSING,
    remove: bool = False,
) -> str:
    """Return one surgical assignment edit and preserve all other bytes."""
    parsed = _parse_document(text, label="destination layer")
    existing = _lookup(parsed, path)
    span = _find_assignment(text, path)
    lines = text.splitlines(keepends=True)
    if remove:
        if existing is _MISSING:
            raise ConfigEditError("setting is not present in the destination layer")
        if span is None:
            raise ConfigEditError(
                "setting is not stored as one standalone assignment"
            )
        return "".join((*lines[: span.start], *lines[span.end :]))
    if value is _MISSING:
        raise ConfigEditError("a set operation requires a value")
    rendered = render_toml_value(value)
    if existing is not _MISSING:
        if span is None:
            raise ConfigEditError(
                "setting is not stored as one standalone assignment"
            )
        ending = "\r\n" if lines[span.start].endswith("\r\n") else "\n"
        if not lines[span.start].endswith(("\n", "\r")):
            ending = ""
        comment = _inline_comment(span.rhs.rstrip("\r\n"))
        replacement = f"{span.prefix}{rendered}{comment}{ending}"
        return "".join(
            (*lines[: span.start], replacement, *lines[span.end :])
        )
    return _insert_assignment(text, path=path, rendered=rendered)


def _insert_assignment(text: str, *, path: Sequence[str], rendered: str) -> str:
    lines = text.splitlines(keepends=True)
    section = tuple(path[:-1])
    key = path[-1]
    newline = "\r\n" if "\r\n" in text else "\n"
    if not section:
        index = next(
            (i for i, line in enumerate(lines) if line.lstrip().startswith("[")),
            len(lines),
        )
        prefix = "".join(lines[:index])
        if prefix and not prefix.endswith(("\n", "\r")):
            prefix += newline
        return prefix + f"{key} = {rendered}{newline}" + "".join(lines[index:])

    section_start = None
    section_end = len(lines)
    for index, line in enumerate(lines):
        table = _TABLE.match(line)
        if table is None:
            continue
        current = _parts(table.group("name"))
        if section_start is None and current == section:
            section_start = index + 1
            continue
        if section_start is not None:
            section_end = index
            break
    assignment = f"{key} = {rendered}{newline}"
    if section_start is not None:
        return "".join((*lines[:section_end], assignment, *lines[section_end:]))

    prefix = text
    if prefix and not prefix.endswith(("\n", "\r")):
        prefix += newline
    if prefix and not prefix.endswith(newline * 2):
        prefix += newline
    return prefix + f"[{'.'.join(section)}]{newline}{assignment}"


def _contains_only_environment_references(value: object) -> bool:
    if isinstance(value, str):
        return _ENV_REFERENCE.fullmatch(value) is not None
    if isinstance(value, list):
        return bool(value) and all(
            _contains_only_environment_references(item) for item in value
        )
    if isinstance(value, Mapping):
        return bool(value) and all(
            _contains_only_environment_references(item)
            for item in value.values()
        )
    return False


def _guard_secret_value(value: object, path: Sequence[str]) -> None:
    _safe, sensitive, _reasons = redact_config_value(value, path=path)
    if sensitive and not _contains_only_environment_references(value):
        raise ConfigEditError(
            "literal values are not accepted for this secret-bearing setting; "
            "use $ENV:NAME or a dedicated credential command"
        )


def _safe_value(
    value: object,
    *,
    path: Sequence[str],
    environment_references: Mapping[ConfigValuePath, str] | None = None,
) -> tuple[object, str | None]:
    if value is _MISSING:
        return "<unset>", None
    safe, redacted, _reasons = redact_config_value(
        value,
        path=path,
        environment_references=environment_references,
    )
    reference = None
    if environment_references is not None:
        reference = environment_references.get(tuple(path))
    if reference:
        return REDACTED_VALUE, reference
    return safe if redacted else value, None


def _read_destination(destination: EditDestination) -> tuple[bytes | None, str]:
    path = destination.path
    if path.is_symlink():
        raise ConfigEditError("destination layer cannot be a symbolic link")
    if not path.exists():
        return None, ""
    if not path.is_file():
        raise ConfigEditError("destination layer is not a regular file")
    if path.stat().st_mode & 0o222 == 0:
        raise ConfigEditError("destination layer is not writable")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigEditError(
            f"cannot read destination layer ({exc.strerror or type(exc).__name__})"
        ) from exc
    try:
        return raw, raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigEditError("destination layer is not UTF-8 text") from exc


def _temporary_config(content: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(prefix="yuj-config-edit-", suffix=".toml")
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _atomic_save(
    destination: EditDestination,
    *,
    expected: bytes | None,
    proposed: bytes,
) -> None:
    path = destination.path
    if destination.create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.is_dir():
        raise ConfigEditError("destination parent directory does not exist")
    current = path.read_bytes() if path.exists() else None
    if current != expected:
        raise ConfigEditError(
            "destination layer changed after preview; no edit was saved"
        )
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(proposed)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            os.chmod(temporary, path.stat().st_mode & 0o777)
        else:
            os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ConfigEditError(
            f"cannot save destination layer ({exc.strerror or type(exc).__name__})"
        ) from exc


def _resolve_with_destination(
    *,
    specs: Sequence[ConfigLayerSpec],
    destination: EditDestination,
    destination_path: Path,
    overrides: Mapping[str, object],
) -> ResolvedConfig:
    adjusted = list(specs)
    local = local_config_path()
    if destination.user_layer_index is None:
        local = destination_path
    else:
        index = destination.user_layer_index
        selected = adjusted[index]
        adjusted[index] = ConfigLayerSpec(
            path=destination_path,
            layer_id=selected.layer_id,
            kind=selected.kind,
            label=selected.label,
        )
    return resolve_config(
        user_config=[spec.path for spec in adjusted],
        overrides=dict(overrides),
        layer_specs=adjusted,
        local_path=local,
    )


def _resolve_current(
    *,
    specs: Sequence[ConfigLayerSpec],
    destination: EditDestination,
    destination_exists: bool,
    overrides: Mapping[str, object],
) -> ResolvedConfig:
    current_specs = list(specs)
    if destination.user_layer_index is not None and not destination_exists:
        del current_specs[destination.user_layer_index]
    return resolve_config(
        user_config=[spec.path for spec in current_specs],
        overrides=dict(overrides),
        layer_specs=current_specs,
        local_path=(
            destination.path
            if destination.user_layer_index is None
            else local_config_path()
        ),
    )


def edit_configuration(
    *,
    operation: str,
    setting: str,
    raw_value: str | None,
    destination: EditDestination,
    specs: Sequence[ConfigLayerSpec],
    overrides: Mapping[str, object],
    apply: bool,
    validate: Callable[[ResolvedConfig], None] | None = None,
) -> dict[str, object]:
    """Validate, preview, and optionally atomically save one setting."""
    if operation not in {"set", "remove"}:
        raise ConfigEditError("operation must be set or remove")
    path = parse_setting_path(setting)
    expected_type = _documented_value(path)
    original_bytes, original_text = _read_destination(destination)
    original_data = _parse_document(original_text, label=destination.label)
    destination_before = _lookup(original_data, path)

    if operation == "set":
        if raw_value is None:
            raise ConfigEditError("set requires a value")
        value = parse_typed_value(raw_value, expected=expected_type)
        _guard_secret_value(value, path)
        proposed_text = edit_toml_text(
            original_text,
            path=path,
            value=value,
        )
        destination_after = value
    else:
        proposed_text = edit_toml_text(
            original_text,
            path=path,
            remove=True,
        )
        destination_after = _MISSING

    proposed_bytes = proposed_text.encode("utf-8")
    temporary = _temporary_config(proposed_bytes)
    try:
        current = _resolve_current(
            specs=specs,
            destination=destination,
            destination_exists=original_bytes is not None,
            overrides=overrides,
        )
        proposed = _resolve_with_destination(
            specs=specs,
            destination=destination,
            destination_path=temporary,
            overrides=overrides,
        )
        if validate is not None:
            validate(proposed)
        old_effective = _lookup(current.data, path)
        new_effective = _lookup(proposed.data, path)
        old_source = current.provenance.get(tuple(path))
        new_source = proposed.provenance.get(tuple(path))
        if old_effective is _MISSING or old_source is None:
            raise ConfigEditError("resolved configuration has no old setting value")
        if new_effective is _MISSING or new_source is None:
            raise ConfigEditError("resolved configuration has no resulting value")

        if apply:
            _atomic_save(
                destination,
                expected=original_bytes,
                proposed=proposed_bytes,
            )

        before_value, before_env = _safe_value(
            destination_before,
            path=path,
        )
        after_value, after_env = _safe_value(
            destination_after,
            path=path,
        )
        old_value, old_env = _safe_value(
            old_effective,
            path=path,
            environment_references=current.environment_references,
        )
        new_value, new_env = _safe_value(
            new_effective,
            path=path,
            environment_references=proposed.environment_references,
        )
        layers = []
        for layer in proposed.layers:
            item = layer.as_dict()
            item["destination"] = layer.layer_id == destination.layer_id
            layers.append(item)
        return {
            "schema": EDIT_SCHEMA,
            "schema_version": EDIT_SCHEMA_VERSION,
            "status": "applied" if apply else "preview",
            "success": True,
            "saved": bool(apply),
            "operation": operation,
            "setting": ".".join(path),
            "destination": {
                "id": destination.layer_id,
                "label": destination.label,
                "path": str(destination.path),
                "before": before_value,
                "before_environment_variable": before_env,
                "after": after_value,
                "after_environment_variable": after_env,
                "before_sha256": (
                    hashlib.sha256(original_bytes).hexdigest()
                    if original_bytes is not None
                    else None
                ),
                "after_sha256": hashlib.sha256(proposed_bytes).hexdigest(),
            },
            "old_effective": {
                "value": old_value,
                "environment_variable": old_env,
                "source_layer": old_source.layer_id,
            },
            "new_effective": {
                "value": new_value,
                "environment_variable": new_env,
                "source_layer": new_source.layer_id,
            },
            "layers": layers,
        }
    finally:
        temporary.unlink(missing_ok=True)


def build_edit_error(message: object) -> dict[str, object]:
    return {
        "schema": EDIT_SCHEMA,
        "schema_version": EDIT_SCHEMA_VERSION,
        "status": "error",
        "success": False,
        "saved": False,
        "diagnostics": [{"level": "error", "message": str(message)}],
    }


def _human_value(value: object, environment: object = None) -> str:
    if environment:
        return f"$ENV:{environment}"
    if value == "<unset>" or value == REDACTED_VALUE:
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def render_edit_human(document: Mapping[str, object]) -> str:
    if not document.get("success"):
        diagnostics = document.get("diagnostics") or []
        detail = "configuration edit failed"
        if isinstance(diagnostics, list) and diagnostics:
            item = diagnostics[0]
            if isinstance(item, Mapping):
                detail = str(item.get("message") or detail)
        return f"Yuj configuration edit: invalid\nError: {detail}\n"

    destination = document["destination"]
    old = document["old_effective"]
    new = document["new_effective"]
    assert isinstance(destination, Mapping)
    assert isinstance(old, Mapping)
    assert isinstance(new, Mapping)
    lines = [
        f"Yuj configuration edit: {document.get('status')}",
        f"Setting: {document.get('setting')}",
        f"Operation: {document.get('operation')}",
        "Destination: "
        f"{destination.get('id')} ({destination.get('path')})",
        "Destination before: "
        + _human_value(
            destination.get("before"),
            destination.get("before_environment_variable"),
        ),
        "Destination after: "
        + _human_value(
            destination.get("after"),
            destination.get("after_environment_variable"),
        ),
        "Old effective: "
        + _human_value(old.get("value"), old.get("environment_variable"))
        + f" [{old.get('source_layer')}]",
        "New effective: "
        + _human_value(new.get("value"), new.get("environment_variable"))
        + f" [{new.get('source_layer')}]",
        "Precedence (low to high):",
    ]
    layers = document.get("layers")
    if isinstance(layers, list):
        for layer in layers:
            if not isinstance(layer, Mapping):
                continue
            marker = " destination" if layer.get("destination") else ""
            applied = "applied" if layer.get("applied") else "not present"
            lines.append(
                f"  {layer.get('order')}: {layer.get('id')} "
                f"({applied}{marker})"
            )
    if not document.get("saved"):
        lines.append("Saved: no. Add --apply to save this validated preview.")
    else:
        lines.append("Saved: yes.")
    return "\n".join(lines) + "\n"


def render_edit_json(document: Mapping[str, object]) -> str:
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


__all__ = [
    "ConfigEditError",
    "EditDestination",
    "build_edit_error",
    "edit_configuration",
    "edit_toml_text",
    "parse_setting_path",
    "parse_typed_value",
    "render_edit_human",
    "render_edit_json",
    "render_toml_value",
    "select_destination",
]
