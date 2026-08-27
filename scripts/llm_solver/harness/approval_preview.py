"""Bounded, read-only previews for assistant approval requests."""
from __future__ import annotations

import difflib
import unicodedata
from collections.abc import Mapping
from pathlib import Path

from ._tools._common import _resolve


_MAX_DIFF_INPUT_BYTES = 1_000_000
_MAX_PREVIEW_CHARS = 16_000
_MAX_PREVIEW_LINES = 120
_MAX_PREVIEW_PATHS = 32
_FILE_MUTATION_TOOLS = frozenset({
    "write",
    "edit",
    "notebook_edit",
    "structural_edit",
    "apply_patch",
    "udiff",
})


class ApprovalPreviewError(ValueError):
    """A proposed mutation cannot be represented as a safe file preview."""


def _terminal_safe(text: object) -> str:
    value = str(text or "")
    out: list[str] = []
    for char in value:
        if char == "\n":
            out.append(char)
        elif unicodedata.category(char) in {"Cc", "Cf"}:
            codepoint = ord(char)
            if codepoint <= 0xFF:
                out.append(f"\\x{codepoint:02x}")
            elif codepoint <= 0xFFFF:
                out.append(f"\\u{codepoint:04x}")
            else:
                out.append(f"\\U{codepoint:08x}")
        else:
            out.append(char)
    return "".join(out)


def _bounded_content(text: str) -> tuple[str, bool, int, int]:
    safe = _terminal_safe(text)
    lines = safe.splitlines(keepends=True)
    bounded = "".join(lines[:_MAX_PREVIEW_LINES])
    if len(bounded) > _MAX_PREVIEW_CHARS:
        bounded = bounded[:_MAX_PREVIEW_CHARS]
    shown = len(bounded)
    total = len(safe)
    truncated = shown < total
    if truncated:
        if bounded and not bounded.endswith("\n"):
            bounded += "\n"
        bounded += (
            f"... [preview truncated: showing {shown} of {total} "
            "escaped characters]\n"
        )
    return bounded, truncated, shown, total


def _paths_payload(paths: list[str]) -> tuple[list[str], int]:
    safe = [_terminal_safe(path) for path in paths]
    return safe[:_MAX_PREVIEW_PATHS], max(0, len(safe) - _MAX_PREVIEW_PATHS)


def _available(
    *,
    format_name: str,
    paths: list[str],
    content: str,
    summary: str,
) -> dict[str, object]:
    bounded, truncated, shown, total = _bounded_content(content)
    shown_paths, omitted_paths = _paths_payload(paths)
    return {
        "schema_version": 1,
        "status": "available",
        "format": format_name,
        "summary": summary,
        "paths": shown_paths,
        "paths_omitted": omitted_paths,
        "content": bounded,
        "truncated": truncated,
        "shown_chars": shown,
        "original_chars": total,
    }


def _unavailable(message: str, *, paths: list[str] | None = None) -> dict[str, object]:
    shown_paths, omitted_paths = _paths_payload(paths or [])
    return {
        "schema_version": 1,
        "status": "unavailable",
        "format": "none",
        "summary": _terminal_safe(message)[:1000],
        "paths": shown_paths,
        "paths_omitted": omitted_paths,
        "content": "",
        "truncated": False,
        "shown_chars": 0,
        "original_chars": 0,
    }


def _not_applicable(tool_name: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "not_applicable",
        "format": "none",
        "summary": (
            f"The {_terminal_safe(tool_name)} request is not a file-mutation "
            "tool, so no proposed file diff applies."
        ),
        "paths": [],
        "paths_omitted": 0,
        "content": "",
        "truncated": False,
        "shown_chars": 0,
        "original_chars": 0,
    }


def _string_argument(arguments: Mapping[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ApprovalPreviewError(f"{name} is not text")
    return value


def _target(cwd: str, path: str) -> Path:
    if "\n" in path or "\x00" in path:
        raise ApprovalPreviewError("the target path contains a newline or NUL")
    try:
        return _resolve(cwd, path)
    except (OSError, ValueError) as exc:
        raise ApprovalPreviewError("the target path is outside the workspace") from exc


def _read_workspace_text(target: Path, display_path: str) -> tuple[str, bool]:
    if not target.exists():
        return "", False
    if not target.is_file():
        raise ApprovalPreviewError(
            f"{_terminal_safe(display_path)} is not a regular file"
        )
    try:
        with target.open("rb") as handle:
            raw = handle.read(_MAX_DIFF_INPUT_BYTES + 1)
    except OSError as exc:
        raise ApprovalPreviewError(
            f"the current {_terminal_safe(display_path)} file cannot be read"
        ) from exc
    if len(raw) > _MAX_DIFF_INPUT_BYTES:
        raise ApprovalPreviewError(
            f"the current {_terminal_safe(display_path)} file exceeds the "
            "safe diff input limit"
        )
    try:
        return raw.decode("utf-8"), True
    except UnicodeDecodeError as exc:
        raise ApprovalPreviewError(
            f"the current {_terminal_safe(display_path)} file is not UTF-8 text"
        ) from exc


def _render_diff_chunk(chunk: str) -> str:
    if chunk.endswith("\r\n"):
        return chunk[:-2] + "\n"
    if chunk.endswith("\n"):
        return chunk
    if chunk[:1] in {"+", "-", " "}:
        return chunk + "\n\\ No newline at end of file\n"
    return chunk + "\n"


def _line_ending_name(text: str) -> str:
    if "\r\n" in text:
        return "CRLF"
    if "\n" in text:
        return "LF"
    return "none"


def _unified_text_diff(
    *,
    path: str,
    workspace_text: str,
    proposed_text: str,
    existed: bool,
) -> str:
    if len(proposed_text.encode("utf-8")) > _MAX_DIFF_INPUT_BYTES:
        raise ApprovalPreviewError(
            "the proposed content exceeds the safe diff input limit"
        )
    from_name = (
        f"a/{path} (current workspace)"
        if existed
        else "/dev/null (current workspace)"
    )
    to_name = f"b/{path} (proposed, not applied)"
    chunks = difflib.unified_diff(
        workspace_text.splitlines(keepends=True),
        proposed_text.splitlines(keepends=True),
        fromfile=from_name,
        tofile=to_name,
        lineterm="\n",
    )
    rendered = "".join(_render_diff_chunk(chunk) for chunk in chunks)
    workspace_endings = _line_ending_name(workspace_text)
    proposed_endings = _line_ending_name(proposed_text)
    if workspace_endings != proposed_endings:
        rendered += (
            "[line endings: current workspace "
            f"{workspace_endings}; proposed {proposed_endings}]\n"
        )
    if not rendered:
        return "(the proposed operation makes no textual change)\n"
    return rendered


def _write_preview(cwd: str, arguments: Mapping[str, object]) -> dict[str, object]:
    path = _string_argument(arguments, "path")
    content = _string_argument(arguments, "content")
    target = _target(cwd, path)
    workspace_text, existed = _read_workspace_text(target, path)
    diff = _unified_text_diff(
        path=path,
        workspace_text=workspace_text,
        proposed_text=content,
        existed=existed,
    )
    return _available(
        format_name="unified_diff",
        paths=[path],
        content=diff,
        summary="Current workspace content compared with the proposed write.",
    )


def _edit_preview(cwd: str, arguments: Mapping[str, object]) -> dict[str, object]:
    path = _string_argument(arguments, "path")
    old_text = _string_argument(arguments, "old_str")
    new_text = _string_argument(arguments, "new_str")
    if not old_text:
        raise ApprovalPreviewError("old_str is empty")
    target = _target(cwd, path)
    workspace_text, existed = _read_workspace_text(target, path)
    if not existed:
        raise ApprovalPreviewError(f"{_terminal_safe(path)} does not exist")
    crlf = "\r\n" in workspace_text
    normalized = workspace_text.replace("\r\n", "\n") if crlf else workspace_text
    if old_text not in normalized:
        raise ApprovalPreviewError(
            "the exact old_str is not present in the current workspace file; "
            "a recovery match cannot be previewed safely"
        )
    proposed = normalized.replace(old_text, new_text, 1)
    if crlf:
        proposed = proposed.replace("\n", "\r\n")
    diff = _unified_text_diff(
        path=path,
        workspace_text=workspace_text,
        proposed_text=proposed,
        existed=True,
    )
    return _available(
        format_name="unified_diff",
        paths=[path],
        content=diff,
        summary="Current workspace content compared with the proposed exact edit.",
    )


def _notebook_edit_preview(
    cwd: str,
    arguments: Mapping[str, object],
) -> dict[str, object]:
    path = _string_argument(arguments, "path")
    old_source = _string_argument(arguments, "old_source")
    new_source = _string_argument(arguments, "new_source")
    target = _target(cwd, path)
    workspace_text, existed = _read_workspace_text(target, path)
    if not existed:
        raise ApprovalPreviewError(f"{_terminal_safe(path)} does not exist")
    from ._tools.notebook_edit import (
        NotebookEditError,
        propose_notebook_edit,
    )

    try:
        proposal = propose_notebook_edit(
            workspace_text,
            old_source=old_source,
            new_source=new_source,
            cell_index=arguments.get("cell_index"),
            cell_id=arguments.get("cell_id"),
        )
    except NotebookEditError as exc:
        raise ApprovalPreviewError(str(exc)) from exc
    diff = _unified_text_diff(
        path=path,
        workspace_text=workspace_text,
        proposed_text=proposal.text,
        existed=True,
    )
    return _available(
        format_name="unified_diff",
        paths=[path],
        content=diff,
        summary=(
            "Current notebook content compared with the proposed cell-source "
            "edit."
        ),
    )


def _structural_edit_preview(
    cwd: str,
    arguments: Mapping[str, object],
    *,
    cfg=None,
) -> dict[str, object]:
    path = _string_argument(arguments, "path")
    language = _string_argument(arguments, "language")
    query = _string_argument(arguments, "query")
    replacement = _string_argument(arguments, "replacement")
    expected_sha256 = _string_argument(arguments, "expected_sha256")
    from ._tools.structural_search import propose_structural_edit
    from .structural_patterns import StructuralPatternError

    try:
        plan = propose_structural_edit(
            path,
            language,
            query,
            replacement,
            cwd=cwd,
            cfg=cfg,
        )
    except StructuralPatternError as exc:
        raise ApprovalPreviewError(str(exc)) from exc
    if plan.preview_sha256 != expected_sha256:
        raise ApprovalPreviewError(
            "the preview hash does not match the current source and rewrite"
        )
    diff = _unified_text_diff(
        path=path,
        workspace_text=plan.original.decode("utf-8"),
        proposed_text=plan.proposed.decode("utf-8"),
        existed=True,
    )
    return _available(
        format_name="unified_diff",
        paths=[path],
        content=diff,
        summary=(
            f"Exact structural rewrite for {len(plan.matches)} matched "
            "location(s). It has not been applied."
        ),
    )


def _apply_patch_preview(
    cwd: str,
    arguments: Mapping[str, object],
) -> dict[str, object]:
    patch = _string_argument(arguments, "patch")
    if len(patch.encode("utf-8")) > _MAX_DIFF_INPUT_BYTES:
        raise ApprovalPreviewError(
            "the patch exceeds the safe preview input limit"
        )
    from .apply_patch import (
        PatchParseError,
        PatchVerifyError,
        _resolved_target,
        parse_patch,
    )

    try:
        operations = parse_patch(patch)
        for operation in operations:
            _resolved_target(Path(cwd), operation.path)
    except (PatchParseError, PatchVerifyError, OSError, ValueError) as exc:
        raise ApprovalPreviewError(
            "the apply_patch request is not a valid workspace patch"
        ) from exc
    return _available(
        format_name="apply_patch",
        paths=[operation.path for operation in operations],
        content=patch,
        summary="Exact apply_patch proposal. It has not been applied.",
    )


def _unified_diff_preview(
    cwd: str,
    arguments: Mapping[str, object],
) -> dict[str, object]:
    patch = _string_argument(arguments, "patch")
    if len(patch.encode("utf-8")) > _MAX_DIFF_INPUT_BYTES:
        raise ApprovalPreviewError(
            "the patch exceeds the safe preview input limit"
        )
    from .udiff import (
        UnifiedDiffApplyError,
        UnifiedDiffParseError,
        _resolved_target,
        parse_unified_diff,
    )

    try:
        operations = parse_unified_diff(patch)
        root = Path(cwd).resolve()
        for operation in operations:
            _resolved_target(root, operation.path)
    except (
        UnifiedDiffApplyError,
        UnifiedDiffParseError,
        OSError,
        ValueError,
    ) as exc:
        raise ApprovalPreviewError(
            "the udiff request is not a valid workspace patch"
        ) from exc
    return _available(
        format_name="unified_diff",
        paths=[operation.path for operation in operations],
        content=patch,
        summary="Exact unified-diff proposal. It has not been applied.",
    )


def build_approval_preview(
    *,
    cwd: str,
    tool_name: str,
    tool_args: Mapping[str, object],
    cfg=None,
) -> dict[str, object]:
    """Build a bounded proposal preview without invoking a tool handler."""
    try:
        if tool_name == "write":
            return _write_preview(cwd, tool_args)
        if tool_name == "edit":
            return _edit_preview(cwd, tool_args)
        if tool_name == "notebook_edit":
            return _notebook_edit_preview(cwd, tool_args)
        if tool_name == "structural_edit":
            return _structural_edit_preview(cwd, tool_args, cfg=cfg)
        if tool_name == "apply_patch":
            return _apply_patch_preview(cwd, tool_args)
        if tool_name == "udiff":
            return _unified_diff_preview(cwd, tool_args)
        if tool_name == "bash":
            return _unavailable(
                "Shell actions can change files dynamically, so Yuj cannot "
                "produce a reliable proposed file diff."
            )
        if tool_name not in _FILE_MUTATION_TOOLS:
            return _not_applicable(tool_name)
    except ApprovalPreviewError as exc:
        paths = []
        path = tool_args.get("path")
        if isinstance(path, str):
            paths.append(path)
        return _unavailable(f"Preview unavailable: {exc}", paths=paths)
    except Exception:
        return _unavailable(
            "Preview unavailable: the request could not be represented safely."
        )
    return _unavailable(
        "Preview unavailable: the request could not be represented safely."
    )


def render_approval_preview(preview: object) -> str:
    """Render a stored preview as terminal-safe, line-oriented text."""
    if not isinstance(preview, Mapping) or preview.get("schema_version") != 1:
        preview = _unavailable(
            "Preview unavailable: this request has no supported preview data."
        )
    status = str(preview.get("status") or "unavailable")
    if status not in {"available", "unavailable", "not_applicable"}:
        status = "unavailable"
    summary = _terminal_safe(preview.get("summary"))[:1000]
    lines = [f"approval_preview_status: {status}"]
    if status == "available":
        lines.append("approval_preview_state: proposed; not applied")
        lines.append(
            "approval_preview_format: "
            + _terminal_safe(preview.get("format"))[:80]
        )
    lines.append(f"approval_preview_summary: {summary}")
    paths = preview.get("paths")
    if isinstance(paths, list) and paths:
        lines.append("approval_preview_paths:")
        for path in paths[:_MAX_PREVIEW_PATHS]:
            lines.append(f"  {_terminal_safe(path)}")
        omitted = preview.get("paths_omitted")
        if isinstance(omitted, int) and omitted > 0:
            lines.append(f"  ... [{omitted} more paths omitted]")
    if status == "available":
        content, truncated, shown, total = _bounded_content(
            str(preview.get("content") or "")
        )
        stored_truncated = bool(preview.get("truncated"))
        lines.append(
            "approval_preview_truncated: "
            + ("yes" if stored_truncated or truncated else "no")
        )
        stored_shown = preview.get("shown_chars")
        stored_total = preview.get("original_chars")
        if isinstance(stored_shown, int) and isinstance(stored_total, int):
            lines.append(
                f"approval_preview_chars: {stored_shown}/{stored_total}"
            )
        else:
            lines.append(f"approval_preview_chars: {shown}/{total}")
        lines.append("approval_preview_content:")
        for line in content.splitlines():
            lines.append(f"  {line}")
    return "\n".join(lines)


__all__ = ["build_approval_preview", "render_approval_preview"]
