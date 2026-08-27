"""Mechanical paths and operation kinds from model edit-tool arguments."""
from __future__ import annotations

import re
from collections.abc import Mapping


_APPLY_HEADER_RE = re.compile(
    r"^\*\*\* (?P<kind>Add|Update|Delete) File: (?P<path>.+?)\s*$",
    re.MULTILINE,
)


def _fallback_apply_patch(patch: str) -> list[tuple[str, str]]:
    kind_map = {"Add": "add", "Update": "update", "Delete": "delete"}
    return [
        (kind_map[match.group("kind")], match.group("path").strip())
        for match in _APPLY_HEADER_RE.finditer(patch)
        if match.group("path").strip()
    ]


def _plain_header_path(line: str, prefix: str) -> str:
    raw = line[len(prefix):].split("\t", 1)[0].strip()
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        raw = raw[1:-1]
    return raw


def _fallback_udiff(patch: str) -> list[tuple[str, str]]:
    lines = patch.splitlines()
    operations: list[tuple[str, str]] = []
    for index, line in enumerate(lines[:-1]):
        if not line.startswith("--- ") or not lines[index + 1].startswith("+++ "):
            continue
        old_path = _plain_header_path(line, "--- ")
        new_path = _plain_header_path(lines[index + 1], "+++ ")
        if old_path.startswith("a/") and (
            new_path.startswith("b/") or new_path == "/dev/null"
        ):
            old_path = old_path[2:]
        if new_path.startswith("b/") and (
            old_path.startswith("a/")
            or line.startswith("--- a/")
            or old_path == "/dev/null"
        ):
            new_path = new_path[2:]
        if old_path == "/dev/null" and new_path != "/dev/null":
            operations.append(("add", new_path))
        elif new_path == "/dev/null" and old_path != "/dev/null":
            operations.append(("delete", old_path))
        elif new_path and old_path == new_path:
            operations.append(("update", new_path))
    return operations


def _deduplicate(
    operations: list[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for kind, path in operations:
        normalized = str(path or "").strip()
        item = (str(kind or "update"), normalized)
        if not normalized or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def edit_operations(
    tool_name: str,
    arguments: Mapping[str, object] | None,
) -> tuple[tuple[str, str], ...]:
    """Return ordered ``(kind, path)`` operations without touching files.

    Valid patch payloads use the production parsers. Header-only fallbacks
    preserve useful attempted-path metadata when a malformed call is traced.
    """
    arguments = arguments or {}
    if tool_name in {
        "write", "edit", "notebook_edit", "str_replace", "create", "insert",
    }:
        path = arguments.get("path") or arguments.get("file_path")
        return _deduplicate([("update", str(path or ""))])

    patch = arguments.get("patch")
    if not isinstance(patch, str):
        return ()
    if tool_name == "apply_patch":
        from .apply_patch import PatchParseError, parse_patch
        try:
            operations = [
                (operation.kind, operation.path)
                for operation in parse_patch(patch)
            ]
        except (PatchParseError, TypeError, ValueError):
            operations = _fallback_apply_patch(patch)
        return _deduplicate(operations)
    if tool_name == "udiff":
        try:
            from .udiff import parse_unified_diff

            operations = [
                (operation.kind, operation.path)
                for operation in parse_unified_diff(patch)
            ]
        except (TypeError, ValueError):
            operations = _fallback_udiff(patch)
        return _deduplicate(operations)
    return ()


def edit_operation_paths(
    tool_name: str,
    arguments: Mapping[str, object] | None,
) -> tuple[str, ...]:
    """Return the distinct paths targeted by one edit-tool call."""
    return tuple(path for _kind, path in edit_operations(tool_name, arguments))


__all__ = ["edit_operation_paths", "edit_operations"]
