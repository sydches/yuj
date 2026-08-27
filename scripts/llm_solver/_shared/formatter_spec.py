"""Shared parsing and validation for explicit formatter declarations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


_FORMATTER_KEYS = frozenset({
    "name",
    "extensions",
    "command",
    "root_markers",
})
_MAX_NAME_CHARS = 80
_MAX_EXTENSIONS = 32
_MAX_COMMAND_PARTS = 64
_MAX_ROOT_MARKERS = 32


@dataclass(frozen=True, slots=True)
class FormatterSpec:
    """One validated formatter selection and command declaration."""

    name: str
    extensions: tuple[str, ...]
    command: tuple[str, ...]
    root_markers: tuple[str, ...]


def _string_list(
    value: object,
    *,
    field: str,
    maximum: int,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"formatter {field} must be an array of strings")
    if not value and not allow_empty:
        raise ValueError(f"formatter {field} must not be empty")
    if len(value) > maximum:
        raise ValueError(
            f"formatter {field} accepts at most {maximum} entries"
        )
    output: list[str] = []
    for item in value:
        if (
            not isinstance(item, str)
            or not item
            or not item.isprintable()
            or "\x00" in item
        ):
            raise ValueError(
                f"formatter {field} must contain nonempty printable strings"
            )
        output.append(item)
    return tuple(output)


def _validate_extension(value: str) -> str:
    normalized = value.lower()
    if (
        not normalized.startswith(".")
        or normalized in {".", ".."}
        or "/" in normalized
        or "\\" in normalized
        or len(normalized) > 32
    ):
        raise ValueError(
            f"formatter extension must be a short suffix such as '.py': {value!r}"
        )
    return normalized


def _validate_marker(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value in {".", ".."}
        or ".." in path.parts
        or "\\" in value
        or len(value) > 256
    ):
        raise ValueError(
            f"formatter root marker must be a safe relative path: {value!r}"
        )
    return value


def parse_formatter_spec(raw: Any) -> FormatterSpec:
    """Return one strict formatter declaration."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"formatter entry must be a table, got {type(raw).__name__}"
        )
    unknown = sorted(set(raw) - _FORMATTER_KEYS)
    if unknown:
        raise ValueError(f"formatter entry has unknown key {unknown[0]!r}")
    for key in ("name", "extensions", "command"):
        if key not in raw:
            raise ValueError(f"formatter entry is missing required key {key!r}")

    name = raw["name"]
    if (
        not isinstance(name, str)
        or not name.strip()
        or len(name) > _MAX_NAME_CHARS
        or not name.isprintable()
    ):
        raise ValueError("formatter name must be a short printable string")
    extensions = tuple(dict.fromkeys(
        _validate_extension(value)
        for value in _string_list(
            raw["extensions"],
            field="extensions",
            maximum=_MAX_EXTENSIONS,
        )
    ))
    command = _string_list(
        raw["command"],
        field="command",
        maximum=_MAX_COMMAND_PARTS,
    )
    if sum(part.count("{path}") for part in command) != 1:
        raise ValueError(
            f"formatter {name!r} command must contain exactly one "
            "'{path}' placeholder"
        )
    if "{path}" in command[0]:
        raise ValueError(
            f"formatter {name!r} executable cannot be the path placeholder"
        )
    markers = tuple(dict.fromkeys(
        _validate_marker(value)
        for value in _string_list(
            raw.get("root_markers", []),
            field="root_markers",
            maximum=_MAX_ROOT_MARKERS,
            allow_empty=True,
        )
    ))
    return FormatterSpec(
        name=name.strip(),
        extensions=extensions,
        command=command,
        root_markers=markers,
    )


def validate_formatter_specs(
    raw_specs: object,
    *,
    enabled: bool,
    timeout: object,
    max_output_chars: object,
) -> tuple[FormatterSpec, ...]:
    """Validate the complete formatter configuration."""
    if not isinstance(raw_specs, list):
        raise ValueError("formatter.formatters must be an array of tables")
    specs = tuple(parse_formatter_spec(raw) for raw in raw_specs)
    if enabled and not specs:
        raise ValueError(
            "formatter.enabled=true requires at least one formatter declaration"
        )
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("formatter names must be unique")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int)
        or not 1 <= timeout <= 300
    ):
        raise ValueError("formatter.timeout must be an integer from 1 to 300")
    if (
        isinstance(max_output_chars, bool)
        or not isinstance(max_output_chars, int)
        or not 256 <= max_output_chars <= 65_536
    ):
        raise ValueError(
            "formatter.max_output_chars must be an integer from 256 to 65536"
        )
    return specs


__all__ = [
    "FormatterSpec",
    "parse_formatter_spec",
    "validate_formatter_specs",
]
