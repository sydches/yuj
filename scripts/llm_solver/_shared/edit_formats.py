"""Canonical edit-dialect names shared by config, profiles, and tools."""
from __future__ import annotations

from types import MappingProxyType


EDIT_FORMATS: tuple[str, ...] = (
    "exact",
    "apply_patch",
    "udiff",
    "whole",
)

EDIT_FORMAT_TO_TOOL = MappingProxyType({
    "exact": "edit",
    "apply_patch": "apply_patch",
    "udiff": "udiff",
    "whole": "write",
})

EDIT_FORMAT_TOOL_NAMES: frozenset[str] = frozenset(
    EDIT_FORMAT_TO_TOOL.values()
)


def validate_edit_format(
    value: object,
    *,
    field: str,
    allow_inherit: bool = False,
) -> str:
    """Validate one public edit-format value and return it unchanged.

    The empty string is the settings-layer inheritance sentinel. Model
    profiles must always name a concrete dialect.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if allow_inherit and value == "":
        return value
    if value not in EDIT_FORMATS:
        choices = " | ".join(EDIT_FORMATS)
        suffix = " or an empty string to inherit the model profile" if allow_inherit else ""
        raise ValueError(f"{field} must be one of {choices}{suffix}; got {value!r}")
    return value


__all__ = [
    "EDIT_FORMATS",
    "EDIT_FORMAT_TO_TOOL",
    "EDIT_FORMAT_TOOL_NAMES",
    "validate_edit_format",
]
