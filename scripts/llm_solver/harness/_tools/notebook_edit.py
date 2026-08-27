"""Deterministic source replacement for one Jupyter notebook cell."""
from __future__ import annotations

import copy
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ...config import Config
from ._common import (
    _is_external_readonly_path,
    _path_hint,
    _resolve,
)


class NotebookEditError(ValueError):
    """A notebook cannot be safely or unambiguously edited."""


@dataclass(frozen=True)
class NotebookEditProposal:
    """One validated raw-text replacement and its selected-cell identity."""

    text: str
    cell_index: int
    cell_id: str
    cell_type: str
    source_start: int
    source_end: int
    replacement: str
    source_form: str
    changed: bool


def _reject_constant(value: str):
    raise NotebookEditError(f"notebook contains invalid JSON constant {value!r}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise NotebookEditError(
                f"notebook contains duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _load_notebook(text: str) -> dict:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except NotebookEditError:
        raise
    except json.JSONDecodeError as exc:
        raise NotebookEditError(
            f"notebook is not valid JSON at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise NotebookEditError("notebook root must be a JSON object")
    nbformat = value.get("nbformat")
    minor = value.get("nbformat_minor")
    if isinstance(nbformat, bool) or not isinstance(nbformat, int) or nbformat < 4:
        raise NotebookEditError("notebook must use nbformat 4 or later")
    if isinstance(minor, bool) or not isinstance(minor, int) or minor < 0:
        raise NotebookEditError("notebook nbformat_minor must be a nonnegative integer")
    if not isinstance(value.get("metadata"), dict):
        raise NotebookEditError("notebook metadata must be a JSON object")
    cells = value.get("cells")
    if not isinstance(cells, list):
        raise NotebookEditError("notebook cells must be a JSON array")

    seen_ids: dict[str, int] = {}
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise NotebookEditError(f"cell {index} must be a JSON object")
        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown", "raw"}:
            raise NotebookEditError(
                f"cell {index} has unsupported cell_type {cell_type!r}"
            )
        if not isinstance(cell.get("metadata"), dict):
            raise NotebookEditError(f"cell {index} metadata must be a JSON object")
        _cell_source_text(cell, index=index)
        cell_id = cell.get("id")
        if cell_id is not None:
            if not isinstance(cell_id, str) or not cell_id:
                raise NotebookEditError(
                    f"cell {index} id must be a nonempty string"
                )
            if cell_id in seen_ids:
                raise NotebookEditError(
                    f"ambiguous duplicate cell id {cell_id!r} at indexes "
                    f"{seen_ids[cell_id]} and {index}"
                )
            seen_ids[cell_id] = index
        if cell_type == "code":
            outputs = cell.get("outputs")
            execution_count = cell.get("execution_count")
            if not isinstance(outputs, list):
                raise NotebookEditError(
                    f"code cell {index} outputs must be a JSON array"
                )
            if (
                execution_count is not None
                and (
                    isinstance(execution_count, bool)
                    or not isinstance(execution_count, int)
                )
            ):
                raise NotebookEditError(
                    f"code cell {index} execution_count must be an integer or null"
                )
        attachments = cell.get("attachments")
        if attachments is not None and not isinstance(attachments, dict):
            raise NotebookEditError(
                f"cell {index} attachments must be a JSON object"
            )
    return value


def _cell_source_text(cell: dict, *, index: int) -> str:
    source = cell.get("source")
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(item, str) for item in source):
        return "".join(source)
    raise NotebookEditError(
        f"cell {index} source must be a string or an array of strings"
    )


def _select_cell(
    notebook: dict,
    *,
    cell_index: int | None,
    cell_id: str | None,
) -> tuple[int, dict]:
    has_index = cell_index is not None
    has_id = cell_id is not None
    if has_index == has_id:
        raise NotebookEditError("provide exactly one of cell_index or cell_id")
    cells = notebook["cells"]
    if has_index:
        if isinstance(cell_index, bool) or not isinstance(cell_index, int):
            raise NotebookEditError("cell_index must be a nonnegative integer")
        if cell_index < 0 or cell_index >= len(cells):
            raise NotebookEditError(
                f"cell_index {cell_index} is outside the notebook cell range "
                f"0..{max(-1, len(cells) - 1)}"
            )
        return cell_index, cells[cell_index]

    if not isinstance(cell_id, str) or not cell_id:
        raise NotebookEditError("cell_id must be a nonempty string")
    matches = [
        index for index, cell in enumerate(cells) if cell.get("id") == cell_id
    ]
    if not matches:
        raise NotebookEditError(f"cell_id {cell_id!r} was not found")
    if len(matches) != 1:
        raise NotebookEditError(f"cell_id {cell_id!r} is ambiguous")
    index = matches[0]
    return index, cells[index]


def _skip_ws(text: str, position: int) -> int:
    while position < len(text) and text[position] in " \t\r\n":
        position += 1
    return position


def _string_end(text: str, position: int) -> int:
    if position >= len(text) or text[position] != '"':
        raise NotebookEditError("internal notebook JSON location mismatch")
    position += 1
    while position < len(text):
        char = text[position]
        if char == '"':
            return position + 1
        if char == "\\":
            position += 2
        else:
            position += 1
    raise NotebookEditError("internal unterminated notebook JSON string")


def _skip_value(text: str, position: int) -> int:
    position = _skip_ws(text, position)
    if position >= len(text):
        raise NotebookEditError("internal missing notebook JSON value")
    char = text[position]
    if char == '"':
        return _string_end(text, position)
    if char == "{":
        position = _skip_ws(text, position + 1)
        if position < len(text) and text[position] == "}":
            return position + 1
        while True:
            key_end = _string_end(text, position)
            position = _skip_ws(text, key_end)
            if position >= len(text) or text[position] != ":":
                raise NotebookEditError("internal malformed notebook JSON object")
            position = _skip_value(text, position + 1)
            position = _skip_ws(text, position)
            if position < len(text) and text[position] == "}":
                return position + 1
            if position >= len(text) or text[position] != ",":
                raise NotebookEditError("internal malformed notebook JSON object")
            position = _skip_ws(text, position + 1)
    if char == "[":
        position = _skip_ws(text, position + 1)
        if position < len(text) and text[position] == "]":
            return position + 1
        while True:
            position = _skip_value(text, position)
            position = _skip_ws(text, position)
            if position < len(text) and text[position] == "]":
                return position + 1
            if position >= len(text) or text[position] != ",":
                raise NotebookEditError("internal malformed notebook JSON array")
            position = _skip_ws(text, position + 1)
    position += 1
    while position < len(text) and text[position] not in " \t\r\n,]}":
        position += 1
    return position


def _object_value_span(
    text: str,
    object_start: int,
    member_name: str,
) -> tuple[int, int]:
    position = _skip_ws(text, object_start)
    if position >= len(text) or text[position] != "{":
        raise NotebookEditError("internal notebook object location mismatch")
    position = _skip_ws(text, position + 1)
    found: tuple[int, int] | None = None
    while position < len(text) and text[position] != "}":
        key_start = position
        key_end = _string_end(text, key_start)
        key = json.loads(text[key_start:key_end])
        position = _skip_ws(text, key_end)
        if position >= len(text) or text[position] != ":":
            raise NotebookEditError("internal malformed notebook object member")
        value_start = _skip_ws(text, position + 1)
        value_end = _skip_value(text, value_start)
        if key == member_name:
            if found is not None:
                raise NotebookEditError(
                    f"notebook object has ambiguous duplicate {member_name!r}"
                )
            found = (value_start, value_end)
        position = _skip_ws(text, value_end)
        if position < len(text) and text[position] == ",":
            position = _skip_ws(text, position + 1)
        elif position < len(text) and text[position] != "}":
            raise NotebookEditError("internal malformed notebook JSON object")
    if found is None:
        raise NotebookEditError(
            f"notebook object is missing required member {member_name!r}"
        )
    return found


def _array_item_spans(text: str, array_start: int) -> list[tuple[int, int]]:
    position = _skip_ws(text, array_start)
    if position >= len(text) or text[position] != "[":
        raise NotebookEditError("internal notebook cells location mismatch")
    position = _skip_ws(text, position + 1)
    items: list[tuple[int, int]] = []
    while position < len(text) and text[position] != "]":
        start = position
        end = _skip_value(text, start)
        items.append((start, end))
        position = _skip_ws(text, end)
        if position < len(text) and text[position] == ",":
            position = _skip_ws(text, position + 1)
        elif position < len(text) and text[position] != "]":
            raise NotebookEditError("internal malformed notebook cells array")
    return items


def _source_span(text: str, cell_index: int) -> tuple[int, int]:
    root_start = _skip_ws(text, 0)
    cells_start, _cells_end = _object_value_span(text, root_start, "cells")
    cells = _array_item_spans(text, cells_start)
    if cell_index >= len(cells):
        raise NotebookEditError("notebook cell locations do not match parsed cells")
    cell_start, _cell_end = cells[cell_index]
    return _object_value_span(text, cell_start, "source")


def _leading_indent(text: str, position: int) -> str:
    line_start = max(text.rfind("\n", 0, position), text.rfind("\r", 0, position))
    prefix = text[line_start + 1:position]
    return prefix[: len(prefix) - len(prefix.lstrip(" \t"))]


def _replacement_source(
    text: str,
    *,
    source_start: int,
    source_end: int,
    source_form: str,
    new_source: str,
) -> str:
    if source_form == "string":
        return json.dumps(new_source, ensure_ascii=False)
    lines = new_source.splitlines(keepends=True)
    original = text[source_start:source_end]
    if "\n" not in original and "\r" not in original:
        return json.dumps(lines, ensure_ascii=False)
    rendered = json.dumps(lines, ensure_ascii=False, indent=1)
    newline = "\r\n" if "\r\n" in text else "\n"
    indent = _leading_indent(text, source_start)
    return rendered.replace("\n", newline + indent)


def propose_notebook_edit(
    text: str,
    *,
    old_source: str,
    new_source: str,
    cell_index: int | None = None,
    cell_id: str | None = None,
) -> NotebookEditProposal:
    """Validate and construct one source-only notebook JSON replacement."""
    if not isinstance(old_source, str) or not isinstance(new_source, str):
        raise NotebookEditError("old_source and new_source must be strings")
    notebook = _load_notebook(text)
    selected_index, cell = _select_cell(
        notebook, cell_index=cell_index, cell_id=cell_id,
    )
    cell_type = str(cell["cell_type"])
    if cell_type not in {"code", "markdown"}:
        raise NotebookEditError(
            f"cell {selected_index} is {cell_type!r}; only code and markdown "
            "cell source can be edited"
        )
    current_source = _cell_source_text(cell, index=selected_index)
    if current_source != old_source:
        raise NotebookEditError(
            f"stale source for cell {selected_index}; read the current notebook "
            "and retry with its exact source"
        )
    source_value = cell["source"]
    source_form = "string" if isinstance(source_value, str) else "array"
    source_start, source_end = _source_span(text, selected_index)
    if old_source == new_source:
        return NotebookEditProposal(
            text=text,
            cell_index=selected_index,
            cell_id=str(cell.get("id") or ""),
            cell_type=cell_type,
            source_start=source_start,
            source_end=source_end,
            replacement=text[source_start:source_end],
            source_form=source_form,
            changed=False,
        )

    replacement = _replacement_source(
        text,
        source_start=source_start,
        source_end=source_end,
        source_form=source_form,
        new_source=new_source,
    )
    updated_text = text[:source_start] + replacement + text[source_end:]
    updated = _load_notebook(updated_text)
    updated_source = _cell_source_text(
        updated["cells"][selected_index], index=selected_index,
    )
    if updated_source != new_source:
        raise NotebookEditError("replacement did not preserve the requested source")
    expected = copy.deepcopy(notebook)
    expected["cells"][selected_index]["source"] = updated["cells"][selected_index][
        "source"
    ]
    if updated != expected:
        raise NotebookEditError("replacement changed unrelated notebook content")
    return NotebookEditProposal(
        text=updated_text,
        cell_index=selected_index,
        cell_id=str(cell.get("id") or ""),
        cell_type=cell_type,
        source_start=source_start,
        source_end=source_end,
        replacement=replacement,
        source_form=source_form,
        changed=True,
    )


def _atomic_write(target: Path, content: bytes) -> None:
    mode = stat.S_IMODE(target.stat().st_mode)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.yuj-", dir=target.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, mode)
        os.replace(temp_path, target)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


def notebook_edit(
    path: str,
    old_source: str,
    new_source: str,
    *,
    cwd: str,
    cell_index: int | None = None,
    cell_id: str | None = None,
    cfg: Config | None = None,
) -> str:
    """Replace one exact code or Markdown cell source without reserializing."""
    if cfg is not None and not bool(
        getattr(cfg, "tools_notebook_edit_enabled", False)
    ):
        return (
            "ERROR: notebook_edit tool is disabled "
            "(tools.notebook_edit_enabled=false)"
        )
    if "\n" in path or "\x00" in path:
        return "ERROR: path contains forbidden character (newline or NUL)"
    if Path(path).suffix.lower() != ".ipynb":
        return "ERROR: notebook_edit requires a .ipynb path"
    if cfg is not None and _is_external_readonly_path(
        cwd,
        path,
        readonly_roots=tuple(getattr(cfg, "skills_readable_dirs", ()) or ()),
    ):
        return f"ERROR: skill path is read-only: {path}"
    try:
        target = _resolve(cwd, path)
        try:
            previous_bytes = target.read_bytes()
        except FileNotFoundError:
            return f"ERROR: file not found: {path}" + _path_hint(cwd, path)
        if target.is_dir():
            return f"ERROR: {path} is a directory"
        try:
            text = previous_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return f"ERROR: notebook {path} is not valid UTF-8"
        proposal = propose_notebook_edit(
            text,
            old_source=old_source,
            new_source=new_source,
            cell_index=cell_index,
            cell_id=cell_id,
        )
        identity = (
            f"id={proposal.cell_id!r}"
            if proposal.cell_id
            else f"index={proposal.cell_index}"
        )
        if not proposal.changed:
            return (
                f"OK: {proposal.cell_type} cell {identity} in {path} already "
                "has the requested source; no file change"
            )
        _atomic_write(target, proposal.text.encode("utf-8"))
        from ..post_edit import run_post_edit_checks

        try:
            check = run_post_edit_checks(
                path, cwd=cwd, cfg=cfg, trigger="edit",
            )
        except BaseException:
            _atomic_write(target, previous_bytes)
            raise
        if check.action == "block":
            _atomic_write(target, previous_bytes)
            return (
                f"ERROR: notebook edit blocked by post-edit check "
                f"{check.check_name!r} for {path}{check.output}"
            )
        return (
            f"OK: updated {proposal.cell_type} cell {identity} in {path}"
            + check.output
        )
    except NotebookEditError as exc:
        return f"ERROR: cannot edit notebook {path}: {exc}"
    except FileNotFoundError:
        return f"ERROR: file not found: {path}" + _path_hint(cwd, path)
    except Exception as exc:
        return f"ERROR: cannot edit notebook {path}: {exc}"


__all__ = [
    "NotebookEditError",
    "NotebookEditProposal",
    "notebook_edit",
    "propose_notebook_edit",
]
