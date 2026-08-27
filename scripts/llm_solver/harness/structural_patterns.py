"""Deterministic Tree-sitter query search and single-file rewrite plans."""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Sequence

from .sandbox.ignore_policy import IgnorePolicy
from .structural_index import (
    SUPPORTED_STRUCTURAL_LANGUAGES,
    StructuralBackendUnavailable,
    _UnreadableMatcher,
    _normalize_match_captures,
    _query_matches,
    detect_structural_language,
    load_structural_language,
)


_DEFAULT_IGNORED_DIR_NAMES = frozenset({".git", ".hg", ".sl", ".svn"})
_TEMPLATE_CAPTURE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_.-]*)\}")


class StructuralPatternError(ValueError):
    """A structural query or rewrite cannot run safely."""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class PatternCapture:
    name: str
    text: str
    start_byte: int
    end_byte: int


@dataclass(frozen=True, slots=True)
class PatternMatch:
    path: str
    language: str
    line: int
    column: int
    start_byte: int
    end_byte: int
    text: str
    captures: tuple[PatternCapture, ...]


@dataclass(frozen=True, slots=True)
class PatternDiagnostic:
    path: str
    kind: str


@dataclass(frozen=True, slots=True)
class PatternSearchResult:
    matches: tuple[PatternMatch, ...]
    total: int
    files_scanned: int
    diagnostics: tuple[PatternDiagnostic, ...]
    capped: bool


@dataclass(frozen=True, slots=True)
class StructuralEditPlan:
    path: str
    language: str
    original: bytes
    proposed: bytes
    matches: tuple[PatternMatch, ...]
    source_sha256: str
    preview_sha256: str


def _parser(grammar):
    try:
        from tree_sitter import Parser
    except ImportError as exc:
        raise StructuralBackendUnavailable(
            "tree-sitter is required for structural pattern tools"
        ) from exc
    try:
        return Parser(grammar)
    except TypeError:  # tree-sitter < 0.25
        parser = Parser()
        if hasattr(parser, "set_language"):
            parser.set_language(grammar)
        else:
            parser.language = grammar
        return parser


def _compiled_query(language: str, source: str):
    if language not in SUPPORTED_STRUCTURAL_LANGUAGES:
        supported = ", ".join(SUPPORTED_STRUCTURAL_LANGUAGES)
        raise StructuralPatternError(
            "unsupported_language",
            f"language must be one of: {supported}",
        )
    if not isinstance(source, str) or not source.strip():
        raise StructuralPatternError("invalid_pattern", "query must not be empty")
    if "\x00" in source:
        raise StructuralPatternError("invalid_pattern", "query contains NUL")
    grammar = load_structural_language(language)
    try:
        from tree_sitter import Query
        query = Query(grammar, source)
    except StructuralBackendUnavailable:
        raise
    except Exception as exc:
        raise StructuralPatternError(
            "invalid_pattern",
            f"query is not valid for language {language!r}: {exc}",
        ) from exc
    names = {
        str(query.capture_name(index))
        for index in range(int(query.capture_count))
    }
    if "match" not in names:
        raise StructuralPatternError(
            "invalid_pattern", "query must capture one node as @match"
        )
    return grammar, query


def _glob_matches(path: str, pattern: str) -> bool:
    if not pattern:
        return True
    candidates = (pattern, pattern[3:]) if pattern.startswith("**/") else (pattern,)
    return any(fnmatchcase(path, candidate) for candidate in candidates)


def _candidate_files(
    *,
    workspace: Path,
    scope: Path,
    language: str,
    path_glob: str,
    unreadable_paths: Sequence[str],
    ignore_policy: IgnorePolicy | None,
    max_files: int,
) -> tuple[Path, ...]:
    unreadable = _UnreadableMatcher(workspace, unreadable_paths)

    def visible(path: Path, *, is_dir: bool) -> bool:
        if path.is_symlink() or unreadable.blocks(path):
            return False
        if ignore_policy is not None and ignore_policy.is_model_hidden(
            path, is_dir=is_dir
        ):
            return False
        return True

    candidates: list[Path] = []
    if scope.exists() and not visible(scope, is_dir=scope.is_dir()):
        raise StructuralPatternError("not_found", "path is not visible")
    if scope.is_file():
        if visible(scope, is_dir=False) and (
            detect_structural_language(scope) == language
        ) and _glob_matches(scope.name, path_glob):
            candidates.append(scope)
    elif scope.is_dir():
        for raw_dir, dir_names, file_names in os.walk(
            scope, topdown=True, followlinks=False
        ):
            directory = Path(raw_dir)
            dir_names[:] = [
                name
                for name in sorted(dir_names)
                if name not in _DEFAULT_IGNORED_DIR_NAMES
                and visible(directory / name, is_dir=True)
            ]
            for name in sorted(file_names):
                candidate = directory / name
                if not visible(candidate, is_dir=False):
                    continue
                if detect_structural_language(candidate) != language:
                    continue
                relative = candidate.relative_to(scope).as_posix()
                if _glob_matches(relative, path_glob):
                    candidates.append(candidate)
                    if len(candidates) > max_files:
                        raise StructuralPatternError(
                            "file_limit",
                            f"query exceeds structural_max_files={max_files}; "
                            "narrow path or glob",
                        )
    else:
        raise StructuralPatternError("not_found", "path does not exist")
    if not candidates:
        raise StructuralPatternError(
            "no_files",
            f"no visible {language} files match the requested path and glob",
        )
    return tuple(candidates)


def _read_source(path: Path, *, max_file_bytes: int) -> bytes:
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise StructuralPatternError(
            "read_error", f"could not read source ({type(exc).__name__})"
        ) from exc
    if len(source) > max_file_bytes:
        raise StructuralPatternError(
            "file_too_large",
            f"source exceeds structural_max_file_bytes={max_file_bytes}",
        )
    try:
        source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StructuralPatternError("invalid_utf8", "source is not valid UTF-8") from exc
    return source


def _node_capture(name: str, node, source: bytes) -> PatternCapture:
    start = int(node.start_byte)
    end = int(node.end_byte)
    return PatternCapture(
        name=name,
        text=source[start:end].decode("utf-8"),
        start_byte=start,
        end_byte=end,
    )


def _file_matches(
    *,
    display_path: str,
    language: str,
    source: bytes,
    grammar,
    query,
) -> tuple[PatternMatch, ...]:
    try:
        from tree_sitter import QueryCursor
    except ImportError as exc:
        raise StructuralBackendUnavailable(
            "tree-sitter is required for structural pattern tools"
        ) from exc
    tree = _parser(grammar).parse(source)
    if bool(getattr(tree.root_node, "has_error", False)):
        raise StructuralPatternError(
            "parse_error", f"tree-sitter could not parse {display_path} cleanly"
        )
    rows: list[PatternMatch] = []
    for raw_captures in _query_matches(query, QueryCursor, tree.root_node):
        captures = _normalize_match_captures(raw_captures)
        match_nodes = tuple(captures.get("match", ()))
        if len(match_nodes) != 1:
            raise StructuralPatternError(
                "ambiguous_match",
                "each query match must capture exactly one node as @match",
            )
        target = match_nodes[0]
        start = int(target.start_byte)
        end = int(target.end_byte)
        if end <= start:
            raise StructuralPatternError(
                "ambiguous_match", "@match must capture a nonempty source node"
            )
        normalized_captures = tuple(sorted(
            (
                _node_capture(name, node, source)
                for name, nodes in captures.items()
                for node in nodes
            ),
            key=lambda item: (
                item.name, item.start_byte, item.end_byte, item.text,
            ),
        ))
        rows.append(PatternMatch(
            path=display_path,
            language=language,
            line=int(target.start_point[0]) + 1,
            column=int(target.start_point[1]) + 1,
            start_byte=start,
            end_byte=end,
            text=source[start:end].decode("utf-8"),
            captures=normalized_captures,
        ))
    return tuple(sorted(
        set(rows),
        key=lambda item: (
            item.path,
            item.start_byte,
            item.end_byte,
            tuple(
                (capture.name, capture.start_byte, capture.end_byte, capture.text)
                for capture in item.captures
            ),
        ),
    ))


def search_structural_patterns(
    *,
    workspace: str | Path,
    scope: str | Path,
    language: str,
    query_source: str,
    path_glob: str = "",
    unreadable_paths: Sequence[str] = (),
    ignore_policy: IgnorePolicy | None = None,
    max_files: int = 1_000,
    max_matches: int = 100,
    max_file_bytes: int = 4_194_304,
) -> PatternSearchResult:
    """Search visible source files with one Tree-sitter query."""
    workspace_path = Path(workspace).resolve()
    scope_path = Path(scope).resolve()
    grammar, query = _compiled_query(language, query_source)
    candidates = _candidate_files(
        workspace=workspace_path,
        scope=scope_path,
        language=language,
        path_glob=path_glob,
        unreadable_paths=unreadable_paths,
        ignore_policy=ignore_policy,
        max_files=max_files,
    )
    matches: list[PatternMatch] = []
    diagnostics: list[PatternDiagnostic] = []
    total = 0
    files_scanned = 0
    for path in candidates:
        display = path.relative_to(workspace_path).as_posix()
        try:
            source = _read_source(path, max_file_bytes=max_file_bytes)
            file_matches = _file_matches(
                display_path=display,
                language=language,
                source=source,
                grammar=grammar,
                query=query,
            )
        except StructuralPatternError as exc:
            diagnostics.append(PatternDiagnostic(display, exc.kind))
            continue
        files_scanned += 1
        total += len(file_matches)
        remaining = max(0, max_matches - len(matches))
        matches.extend(file_matches[:remaining])
    return PatternSearchResult(
        matches=tuple(matches),
        total=total,
        files_scanned=files_scanned,
        diagnostics=tuple(diagnostics),
        capped=total > len(matches),
    )


def _render_replacement(template: str, match: PatternMatch) -> bytes:
    by_name: dict[str, list[str]] = {}
    for capture in match.captures:
        by_name.setdefault(capture.name, []).append(capture.text)

    def replace(found: re.Match[str]) -> str:
        name = found.group(1)
        values = by_name.get(name, [])
        if not values:
            raise StructuralPatternError(
                "missing_capture",
                f"replacement refers to missing capture {name!r}",
            )
        if len(values) != 1:
            raise StructuralPatternError(
                "ambiguous_capture",
                f"replacement capture {name!r} has {len(values)} values",
            )
        return values[0]

    return _TEMPLATE_CAPTURE_RE.sub(replace, template).encode("utf-8")


def build_structural_edit_plan(
    *,
    workspace: str | Path,
    target: str | Path,
    language: str,
    query_source: str,
    replacement: str,
    unreadable_paths: Sequence[str] = (),
    ignore_policy: IgnorePolicy | None = None,
    max_matches: int = 100,
    max_file_bytes: int = 4_194_304,
) -> StructuralEditPlan:
    """Build one validated source rewrite without changing the file."""
    if language not in SUPPORTED_STRUCTURAL_LANGUAGES:
        supported = ", ".join(SUPPORTED_STRUCTURAL_LANGUAGES)
        raise StructuralPatternError(
            "unsupported_language",
            f"language must be one of: {supported}",
        )
    workspace_path = Path(workspace).resolve()
    target_path = Path(target).resolve()
    if not target_path.is_file() or target_path.is_symlink():
        raise StructuralPatternError("not_found", "target must be a visible file")
    unreadable = _UnreadableMatcher(workspace_path, unreadable_paths)
    if unreadable.blocks(target_path) or (
        ignore_policy is not None
        and ignore_policy.is_model_hidden(target_path, is_dir=False)
    ):
        raise StructuralPatternError("not_found", "target must be a visible file")
    detected = detect_structural_language(target_path)
    if detected != language:
        raise StructuralPatternError(
            "language_mismatch",
            f"target language is {detected or 'unsupported'}, not {language}",
        )
    grammar, query = _compiled_query(language, query_source)
    original = _read_source(target_path, max_file_bytes=max_file_bytes)
    display = target_path.relative_to(workspace_path).as_posix()
    matches = _file_matches(
        display_path=display,
        language=language,
        source=original,
        grammar=grammar,
        query=query,
    )
    if not matches:
        raise StructuralPatternError("no_match", "query matched no source nodes")
    if len(matches) > max_matches:
        raise StructuralPatternError(
            "match_limit",
            f"query matches {len(matches)} nodes, above "
            f"structural_max_matches={max_matches}; narrow the query",
        )
    operations: list[tuple[PatternMatch, bytes]] = []
    for match in matches:
        rendered = _render_replacement(replacement, match)
        if rendered != original[match.start_byte:match.end_byte]:
            operations.append((match, rendered))
    if not operations:
        raise StructuralPatternError(
            "no_change", "replacement would not change any matched node"
        )
    ordered = sorted(operations, key=lambda item: item[0].start_byte)
    previous_end = -1
    for match, _rendered in ordered:
        if match.start_byte < previous_end:
            raise StructuralPatternError(
                "overlapping_matches",
                "query produced overlapping @match nodes; narrow the query",
            )
        previous_end = match.end_byte
    proposed = original
    for match, rendered in reversed(ordered):
        proposed = (
            proposed[:match.start_byte] + rendered + proposed[match.end_byte:]
        )
    updated_tree = _parser(grammar).parse(proposed)
    if bool(getattr(updated_tree.root_node, "has_error", False)):
        raise StructuralPatternError(
            "invalid_replacement",
            "replacement would leave the source with syntax errors",
        )
    digest = hashlib.sha256(b"yuj-structural-preview-v1\x00")
    for value in (display, language, query_source, replacement):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    digest.update(hashlib.sha256(original).digest())
    for match, rendered in ordered:
        digest.update(match.start_byte.to_bytes(8, "big"))
        digest.update(match.end_byte.to_bytes(8, "big"))
        digest.update(len(rendered).to_bytes(8, "big"))
        digest.update(rendered)
    return StructuralEditPlan(
        path=display,
        language=language,
        original=original,
        proposed=proposed,
        matches=tuple(item[0] for item in ordered),
        source_sha256=hashlib.sha256(original).hexdigest(),
        preview_sha256=digest.hexdigest(),
    )


__all__ = [
    "PatternCapture",
    "PatternDiagnostic",
    "PatternMatch",
    "PatternSearchResult",
    "StructuralEditPlan",
    "StructuralPatternError",
    "build_structural_edit_plan",
    "search_structural_patterns",
]
