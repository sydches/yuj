"""Deterministic, policy-bounded Markdown prompt imports.

This module deliberately has no dependency on harness configuration, session
state, or the conversation loop.  Callers supply the filesystem policy and own
prompt assembly plus trace emission.
"""
from __future__ import annotations

from dataclasses import dataclass
import glob
import os
from pathlib import Path
import re
from typing import Sequence


DEFAULT_IMPORT_MAX_DEPTH = 5
MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
_GLOB_META = frozenset("*?[")
_IMPORT_LINE_RE = re.compile(
    r"^(?P<indent> {0,3})@(?P<path>\S+)[ \t]*$"
)
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})[^\r\n]*$")


@dataclass(frozen=True, slots=True)
class ImportTreeNode:
    """One import directive and the imports nested below it.

    ``path`` and ``source`` are policy-relative display labels.  They are safe
    to serialize in a public trace; absolute host paths are never retained.
    """

    request: str
    path: str
    source: str
    status: str
    depth: int
    byte_count: int = 0
    children: tuple[ImportTreeNode, ...] = ()

    def trace_record(self) -> dict[str, object]:
        """Return a JSON-ready representation for a session-start event."""
        return {
            "request": self.request,
            "path": self.path,
            "source": self.source,
            "status": self.status,
            "depth": self.depth,
            "bytes": self.byte_count,
            "children": [child.trace_record() for child in self.children],
        }


@dataclass(frozen=True, slots=True)
class ProcessedImports:
    """Resolved prompt content and deterministic import provenance."""

    content: str
    imports: tuple[ImportTreeNode, ...]
    imported_files: tuple[str, ...]
    imported_bytes: int

    def trace_tree(self) -> list[dict[str, object]]:
        """Return the import forest in encounter order."""
        return [node.trace_record() for node in self.imports]


@dataclass(frozen=True, slots=True)
class _Expansion:
    content: str
    nodes: tuple[ImportTreeNode, ...]
    loaded_paths: tuple[str, ...]
    loaded_bytes: int


class _UnreadableMatcher:
    """Resolve sandbox unreadable patterns once, before any prompt read."""

    def __init__(self, base_dir: Path, patterns: Sequence[str]) -> None:
        blocked: set[Path] = set()
        for original in patterns:
            pattern = str(original)
            if pattern.startswith("optional:"):
                pattern = pattern[len("optional:"):]
            expanded = os.path.expandvars(os.path.expanduser(pattern))
            candidate = Path(expanded)
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            candidate_text = str(candidate)
            if any(character in candidate_text for character in _GLOB_META):
                blocked.update(
                    Path(match).resolve(strict=False)
                    for match in glob.glob(
                        candidate_text,
                        recursive=True,
                        include_hidden=True,
                    )
                )
            else:
                blocked.add(candidate.resolve(strict=False))
        self._blocked = tuple(sorted(blocked, key=str))

    def blocks(self, path: Path) -> bool:
        resolved = path.resolve(strict=False)
        return any(
            resolved == blocked or blocked in resolved.parents
            for blocked in self._blocked
        )


class _ImportProcessor:
    def __init__(
        self,
        *,
        allowed_dirs: Sequence[Path],
        max_depth: int,
        unreadable: _UnreadableMatcher,
    ) -> None:
        self.allowed_dirs = tuple(allowed_dirs)
        self.max_depth = max_depth
        self.unreadable = unreadable

    def expand(
        self,
        text: str,
        *,
        current_dir: Path,
        source: str,
        depth: int,
        stack: tuple[Path, ...],
    ) -> _Expansion:
        output: list[str] = []
        nodes: list[ImportTreeNode] = []
        loaded_paths: list[str] = []
        loaded_bytes = 0
        fence: tuple[str, int] | None = None
        code_span_ticks: int | None = None

        for line in text.splitlines(keepends=True):
            body, ending = _split_line_ending(line)
            if fence is not None:
                output.append(line)
                if _closes_fence(body, fence):
                    fence = None
                continue

            if code_span_ticks is not None:
                output.append(line)
                code_span_ticks = _advance_code_span(body, code_span_ticks)
                continue

            opening = _FENCE_OPEN_RE.match(body)
            if opening is not None:
                marker = opening.group("fence")
                fence = (marker[0], len(marker))
                output.append(line)
                continue

            if body.startswith(("    ", "\t")):
                output.append(line)
                continue

            directive = _IMPORT_LINE_RE.match(body)
            if directive is None:
                output.append(line)
                code_span_ticks = _advance_code_span(body, code_span_ticks)
                continue

            request = directive.group("path")
            replacement, node, child_paths, child_bytes = self._load(
                request,
                current_dir=current_dir,
                source=source,
                depth=depth + 1,
                stack=stack,
            )
            output.append(_with_line_ending(replacement, ending))
            nodes.append(node)
            loaded_paths.extend(child_paths)
            loaded_bytes += child_bytes

        return _Expansion(
            content="".join(output),
            nodes=tuple(nodes),
            loaded_paths=tuple(loaded_paths),
            loaded_bytes=loaded_bytes,
        )

    def _load(
        self,
        request: str,
        *,
        current_dir: Path,
        source: str,
        depth: int,
        stack: tuple[Path, ...],
    ) -> tuple[str, ImportTreeNode, tuple[str, ...], int]:
        try:
            requested_path = Path(request).expanduser()
            candidate = (
                requested_path
                if requested_path.is_absolute()
                else current_dir / requested_path
            ).resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            node = self._error_node(
                request=request,
                path=_request_fallback(request),
                source=source,
                status="invalid_path",
                depth=depth,
            )
            return _error_comment(node), node, (), 0

        containing_root = _containing_root(candidate, self.allowed_dirs)
        if containing_root is None:
            safe_request = (
                _request_fallback(request)
                if requested_path.is_absolute()
                else request
            )
            node = self._error_node(
                request=safe_request,
                path=_outside_label(candidate),
                source=source,
                status="outside_allowed_dirs",
                depth=depth,
            )
            return _error_comment(node), node, (), 0

        display_path = candidate.relative_to(containing_root).as_posix() or "."
        safe_request = display_path if requested_path.is_absolute() else request
        if candidate.suffix.lower() not in MARKDOWN_SUFFIXES:
            node = self._error_node(
                request=safe_request,
                path=display_path,
                source=source,
                status="not_markdown",
                depth=depth,
            )
            return _error_comment(node), node, (), 0
        if self.unreadable.blocks(candidate):
            node = self._error_node(
                request=safe_request,
                path=display_path,
                source=source,
                status="unreadable",
                depth=depth,
            )
            return _error_comment(node), node, (), 0
        if candidate in stack:
            node = self._error_node(
                request=safe_request,
                path=display_path,
                source=source,
                status="cycle",
                depth=depth,
            )
            return _error_comment(node), node, (), 0
        if depth > self.max_depth:
            node = self._error_node(
                request=safe_request,
                path=display_path,
                source=source,
                status="depth_exceeded",
                depth=depth,
            )
            return _error_comment(node), node, (), 0
        if not candidate.is_file():
            node = self._error_node(
                request=safe_request,
                path=display_path,
                source=source,
                status="missing",
                depth=depth,
            )
            return _error_comment(node), node, (), 0

        try:
            raw = candidate.read_bytes()
        except OSError:
            node = self._error_node(
                request=safe_request,
                path=display_path,
                source=source,
                status="read_error",
                depth=depth,
            )
            return _error_comment(node), node, (), 0

        child = self.expand(
            raw.decode("utf-8-sig", errors="replace"),
            current_dir=candidate.parent,
            source=display_path,
            depth=depth,
            stack=(*stack, candidate),
        )
        node = ImportTreeNode(
            request=safe_request,
            path=display_path,
            source=source,
            status="loaded",
            depth=depth,
            byte_count=len(raw),
            children=child.nodes,
        )
        return (
            child.content,
            node,
            (display_path, *child.loaded_paths),
            len(raw) + child.loaded_bytes,
        )

    @staticmethod
    def _error_node(
        *,
        request: str,
        path: str,
        source: str,
        status: str,
        depth: int,
    ) -> ImportTreeNode:
        return ImportTreeNode(
            request=_safe_trace_value(request),
            path=_safe_trace_value(path),
            source=source,
            status=status,
            depth=depth,
        )


def process_imports(
    text: str,
    base_dir: str | Path,
    allowed_dirs: Sequence[str | Path],
    max_depth: int = DEFAULT_IMPORT_MAX_DEPTH,
    *,
    source_path: str | Path | None = None,
    unreadable_paths: Sequence[str] = (),
) -> ProcessedImports:
    """Inline standalone ``@path`` directives under an explicit path policy.

    Relative imports resolve from the importing file.  Absolute imports are
    accepted only when their resolved target remains under ``allowed_dirs``.
    Nested imports are bounded by ``max_depth`` and cycles are stopped using
    the resolved source-path stack.  Directives inside fenced or indented code,
    or inside inline code, remain literal.

    Failures produce a compact HTML comment in the resolved content and a
    structured import-tree node; prompt loading remains deterministic and
    debuggable without exposing absolute host paths.
    """
    if not isinstance(text, str):
        raise TypeError("prompt import text must be a string")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("imports_max_depth must be a non-negative integer")

    resolved_base = Path(base_dir).expanduser().resolve()
    if not resolved_base.is_dir():
        raise ValueError(f"prompt import base is not a directory: {base_dir}")
    roots: list[Path] = []
    for raw_root in allowed_dirs:
        root = Path(raw_root).expanduser().resolve(strict=False)
        if not root.is_dir():
            raise ValueError(f"allowed import directory is not a directory: {raw_root}")
        if root not in roots:
            roots.append(root)
    if not roots:
        raise ValueError("allowed_dirs must contain at least one directory")

    unreadable = _UnreadableMatcher(resolved_base, unreadable_paths)
    source = "<inline>"
    stack: tuple[Path, ...] = ()
    if source_path is not None:
        resolved_source = Path(source_path).expanduser().resolve(strict=False)
        source = _display_path(resolved_source, roots)
        stack = (resolved_source,)

    processor = _ImportProcessor(
        allowed_dirs=roots,
        max_depth=max_depth,
        unreadable=unreadable,
    )
    expanded = processor.expand(
        text,
        current_dir=resolved_base,
        source=source,
        depth=0,
        stack=stack,
    )
    return ProcessedImports(
        content=expanded.content,
        imports=expanded.nodes,
        imported_files=expanded.loaded_paths,
        imported_bytes=expanded.loaded_bytes,
    )


def _containing_root(path: Path, roots: Sequence[Path]) -> Path | None:
    for root in roots:
        if path == root or root in path.parents:
            return root
    return None


def _display_path(path: Path, roots: Sequence[Path]) -> str:
    root = _containing_root(path, roots)
    if root is None:
        return _outside_label(path)
    return path.relative_to(root).as_posix() or "."


def _outside_label(path: Path) -> str:
    name = path.name
    return f"<outside-allowed-dirs>/{name}" if name else "<outside-allowed-dirs>"


def _request_fallback(request: str) -> str:
    name = Path(request.replace("\x00", "")).name
    return name or "<invalid-path>"


def _safe_trace_value(value: str) -> str:
    return value.replace("\x00", "").replace("\r", "").replace("\n", "")


def _error_comment(node: ImportTreeNode) -> str:
    status = _safe_trace_value(node.status).replace("--", "-")
    path = _comment_attr(_safe_trace_value(node.path).replace("--", "-"))
    return f'<!-- yuj-import-error status="{status}" path="{path}" -->'


def _comment_attr(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _split_line_ending(line: str) -> tuple[str, str]:
    body = line.rstrip("\r\n")
    return body, line[len(body):]


def _with_line_ending(content: str, ending: str) -> str:
    if ending and not content.endswith(("\n", "\r")):
        return content + ending
    return content


def _closes_fence(line: str, fence: tuple[str, int]) -> bool:
    marker, minimum = fence
    return re.fullmatch(rf" {{0,3}}{re.escape(marker)}{{{minimum},}}[ \t]*", line) is not None


def _advance_code_span(line: str, active_ticks: int | None) -> int | None:
    """Track CommonMark-style backtick spans, including multiline spans."""
    offset = 0
    while offset < len(line):
        if line[offset] != "`":
            offset += 1
            continue
        run_end = offset + 1
        while run_end < len(line) and line[run_end] == "`":
            run_end += 1
        backslashes = 0
        before = offset - 1
        while before >= 0 and line[before] == "\\":
            backslashes += 1
            before -= 1
        if backslashes % 2 == 0:
            run_length = run_end - offset
            if active_ticks is None:
                active_ticks = run_length
            elif active_ticks == run_length:
                active_ticks = None
        offset = run_end
    return active_ticks


__all__ = [
    "DEFAULT_IMPORT_MAX_DEPTH",
    "ImportTreeNode",
    "MARKDOWN_SUFFIXES",
    "ProcessedImports",
    "process_imports",
]
