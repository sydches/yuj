"""Deterministic repository-wide structural symbol indexing.

The module is deliberately independent of ``Config`` and the harness loop so
analysis tools can reuse it.  Tree-sitter dependencies are imported lazily;
importing this module never downloads or initializes a grammar.
"""
from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
import glob
import hashlib
import os
from pathlib import Path
from typing import Callable, Iterable, Literal, Protocol, Sequence


SymbolKind = Literal["def", "ref"]

_CORE_LANGUAGE_BY_SUFFIX = {
    ".cjs": "javascript",
    ".go": "go",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".py": "python",
    ".pyi": "python",
    ".rs": "rust",
    ".ts": "typescript",
    ".tsx": "tsx",
}
_DEFAULT_IGNORED_DIR_NAMES = frozenset({".git", ".hg", ".sl", ".svn"})
# Some upstream tags queries are intentionally additive to a parent grammar's
# query.  The language-pack API returns each query's source independently, so
# compose the known grammar inheritance here before compilation.
_DEFAULT_TAG_QUERY_PARENTS = {
    "tsx": ("javascript",),
    "typescript": ("javascript",),
}
_GLOB_META = frozenset("*?[")
_MAX_SIGNATURE_CHARS = 240


class StructuralIndexError(RuntimeError):
    """Base class for structural-index failures."""


class StructuralBackendUnavailable(StructuralIndexError):
    """Raised when the configured parser/query backend cannot run."""


class StructuralLanguageUnsupported(StructuralIndexError):
    """Raised when a language has no usable tags query."""


@dataclass(frozen=True, slots=True)
class StructuralRow:
    """One definition or reference discovered in a source file."""

    path: str
    line: int
    column: int
    kind: SymbolKind
    name: str
    signature: str
    language: str
    capture: str

    def render(self) -> str:
        """Render the public ``path:line kind name signature`` row shape."""
        suffix = f" {self.signature}" if self.signature else ""
        return f"{self.path}:{self.line} {self.kind} {self.name}{suffix}"


@dataclass(frozen=True, slots=True)
class IndexDiagnostic:
    """A file-local indexing failure that did not abort the repository scan."""

    path: str
    error_kind: str
    message: str


@dataclass(frozen=True, slots=True)
class IndexSnapshot:
    """Deterministically ordered result of one repository scan."""

    rows: tuple[StructuralRow, ...]
    diagnostics: tuple[IndexDiagnostic, ...]
    files_scanned: int
    cache_hits: int


@dataclass(frozen=True, slots=True)
class StructuralSearchPage:
    """One bounded page of symbol matches plus complete-scan metadata."""

    rows: tuple[StructuralRow, ...]
    total: int
    available: int
    page: int
    per_page: int
    next_page: int
    max_rows: int
    capped: bool
    files_scanned: int
    cache_hits: int
    diagnostics: tuple[IndexDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class FormattedRows:
    """Character-bounded rendered rows.

    ``shown`` counts complete rows.  No row is ever cut in the middle, which
    keeps every emitted line machine-parseable.
    """

    text: str
    shown: int
    char_limited: bool


class StructuralExtractor(Protocol):
    """Parser backend contract consumed by :class:`StructuralIndex`."""

    def detect_language(self, path: Path) -> str | None:
        """Return a tree-sitter language name for ``path`` when supported."""

    def extract(
        self,
        source: bytes,
        *,
        language: str,
        display_path: str,
    ) -> tuple[StructuralRow, ...]:
        """Extract definition/reference rows from one file."""


LanguageLoader = Callable[[str], object]
TagsQueryLoader = Callable[[str], str | None]
LanguageDetector = Callable[[str], str | None]


class TreeSitterTagExtractor:
    """Extract symbols by executing language-pack ``tags.scm`` queries.

    Loaders are injectable for offline deployments and tests.  With no
    loaders supplied, the implementation uses ``tree_sitter_language_pack``
    and the host ``tree_sitter`` package lazily on the first parsed file.
    """

    def __init__(
        self,
        *,
        language_loader: LanguageLoader | None = None,
        tags_query_loader: TagsQueryLoader | None = None,
        language_detector: LanguageDetector | None = None,
        query_parents: dict[str, Sequence[str]] | None = None,
    ) -> None:
        self._language_loader = language_loader
        self._tags_query_loader = tags_query_loader
        self._language_detector = language_detector
        parents = query_parents or _DEFAULT_TAG_QUERY_PARENTS
        self._query_parents = {
            str(child): tuple(str(parent) for parent in parent_names)
            for child, parent_names in parents.items()
        }

    def _load_tags_query(self, language: str) -> str | None:
        assert self._tags_query_loader is not None
        ordered: list[str] = []
        visiting: set[str] = set()

        def visit(current: str) -> None:
            if current in visiting:
                raise StructuralBackendUnavailable(
                    f"cycle in tags-query inheritance at {current!r}"
                )
            visiting.add(current)
            for parent in self._query_parents.get(current, ()):
                visit(parent)
            visiting.remove(current)
            query = self._tags_query_loader(current)
            if query and query not in ordered:
                ordered.append(query)

        visit(language)
        return "\n".join(ordered) or None

    def _ensure_pack_loaders(self) -> None:
        if self._language_loader is not None and self._tags_query_loader is not None:
            return
        try:
            import tree_sitter_language_pack as language_pack
        except ImportError as exc:
            raise StructuralBackendUnavailable(
                "tree-sitter-language-pack is required for structural indexing"
            ) from exc
        get_tags_query = getattr(language_pack, "get_tags_query", None)
        if get_tags_query is None:
            raise StructuralBackendUnavailable(
                "tree-sitter-language-pack does not expose get_tags_query(); "
                "install a release with bundled tags query support"
            )
        if self._language_loader is None:
            self._language_loader = language_pack.get_language
        if self._tags_query_loader is None:
            self._tags_query_loader = get_tags_query
        if self._language_detector is None:
            self._language_detector = getattr(language_pack, "detect_language", None)

    def detect_language(self, path: Path) -> str | None:
        core = _CORE_LANGUAGE_BY_SUFFIX.get(path.suffix.lower())
        if core is not None:
            return core
        if self._language_detector is None:
            try:
                self._ensure_pack_loaders()
            except StructuralBackendUnavailable:
                return None
        if self._language_detector is None:
            return None
        try:
            detected = self._language_detector(str(path))
        except Exception:
            return None
        return str(detected) if detected else None

    def extract(
        self,
        source: bytes,
        *,
        language: str,
        display_path: str,
    ) -> tuple[StructuralRow, ...]:
        self._ensure_pack_loaders()
        assert self._language_loader is not None
        assert self._tags_query_loader is not None
        try:
            query_source = self._load_tags_query(language)
        except Exception as exc:
            raise StructuralLanguageUnsupported(
                f"could not load tags query for {language}: {exc}"
            ) from exc
        if not query_source:
            raise StructuralLanguageUnsupported(
                f"language {language!r} has no bundled tags query"
            )

        try:
            from tree_sitter import Parser, Query, QueryCursor
        except ImportError as exc:
            raise StructuralBackendUnavailable(
                "tree-sitter is required to execute structural tag queries"
            ) from exc

        try:
            grammar = self._language_loader(language)
            try:
                parser = Parser(grammar)
            except TypeError:  # tree-sitter < 0.25
                parser = Parser()
                if hasattr(parser, "set_language"):
                    parser.set_language(grammar)
                else:
                    parser.language = grammar
            tree = parser.parse(source)
            try:
                query = Query(grammar, query_source)
            except TypeError:  # tree-sitter < 0.25
                query = grammar.query(query_source)
            matches = _query_matches(query, QueryCursor, tree.root_node)
        except StructuralIndexError:
            raise
        except Exception as exc:
            raise StructuralBackendUnavailable(
                f"tree-sitter could not parse {display_path} as {language}: {exc}"
            ) from exc

        rows: list[StructuralRow] = []
        for captures in matches:
            normalized = _normalize_match_captures(captures)
            names = normalized.get("name", ())
            typed = [
                (capture, node)
                for capture, nodes in normalized.items()
                if capture.startswith(("definition.", "reference."))
                for node in nodes
            ]
            for capture, target in typed:
                name_node = _nearest_name_node(target, names)
                if name_node is None:
                    continue
                name = _node_text(name_node, source).strip()
                if not name:
                    continue
                public_kind: SymbolKind = (
                    "def" if capture.startswith("definition.") else "ref"
                )
                row = StructuralRow(
                    path=display_path,
                    line=int(target.start_point[0]) + 1,
                    column=int(name_node.start_point[1]) + 1,
                    kind=public_kind,
                    name=name,
                    signature=_node_signature(target, source, name=name),
                    language=language,
                    capture=capture,
                )
                rows.append(row)

        # Query alternatives can yield duplicate captures for one node.  Use
        # the complete public identity so de-duplication is deterministic.
        unique = {row: None for row in rows}
        return tuple(sorted(unique, key=_row_sort_key))


def _query_matches(query: object, cursor_type: type, root: object) -> list[object]:
    """Run a query across supported py-tree-sitter cursor APIs."""
    try:
        cursor = cursor_type(query)
        raw = cursor.matches(root)
    except TypeError:
        try:
            cursor = cursor_type()
            raw = cursor.matches(query, root)
        except (AttributeError, TypeError):
            matches_method = getattr(query, "matches", None)
            if matches_method is None:
                raise StructuralBackendUnavailable(
                    "installed tree-sitter binding has no query matches API"
                )
            raw = matches_method(root)
    return [item[1] if isinstance(item, tuple) and len(item) == 2 else item for item in raw]


def _normalize_match_captures(captures: object) -> dict[str, tuple[object, ...]]:
    if isinstance(captures, dict):
        return {
            str(name): tuple(nodes if isinstance(nodes, (list, tuple)) else (nodes,))
            for name, nodes in captures.items()
        }
    normalized: dict[str, list[object]] = {}
    if isinstance(captures, (list, tuple)):
        for item in captures:
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            node, name = item
            normalized.setdefault(str(name), []).append(node)
    return {name: tuple(nodes) for name, nodes in normalized.items()}


def _nearest_name_node(target: object, names: Sequence[object]) -> object | None:
    if not names:
        return None
    start = int(target.start_byte)
    end = int(target.end_byte)
    contained = [
        node
        for node in names
        if start <= int(node.start_byte) and int(node.end_byte) <= end
    ]
    choices = contained or list(names)
    return min(
        choices,
        key=lambda node: (
            abs(int(node.start_byte) - start),
            int(node.start_byte),
            int(node.end_byte),
        ),
    )


def _node_text(node: object, source: bytes) -> str:
    return source[int(node.start_byte):int(node.end_byte)].decode(
        "utf-8", errors="replace"
    )


def _node_signature(target: object, source: bytes, *, name: str) -> str:
    start = int(target.start_byte)
    end = int(target.end_byte)
    child_by_field_name = getattr(target, "child_by_field_name", None)
    if child_by_field_name is not None:
        try:
            body = child_by_field_name("body")
        except Exception:
            body = None
        if body is not None and int(body.start_byte) > start:
            end = int(body.start_byte)
    raw = source[start:end].decode("utf-8", errors="replace")
    if end == int(target.end_byte) and "\n" in raw:
        raw = raw.splitlines()[0]
    compact = " ".join(raw.split())
    if compact == name:
        return ""
    if len(compact) > _MAX_SIGNATURE_CHARS:
        compact = compact[: _MAX_SIGNATURE_CHARS - 3].rstrip() + "..."
    return compact


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    digest: str
    language: str
    rows: tuple[StructuralRow, ...]


class _UnreadableMatcher:
    """Resolved unreadable paths with directory-descendant matching."""

    def __init__(self, root: Path, patterns: Sequence[str]) -> None:
        blocked: set[Path] = set()
        for original in patterns:
            pattern = str(original)
            if pattern.startswith("optional:"):
                pattern = pattern[len("optional:"):]
            expanded = os.path.expandvars(os.path.expanduser(pattern))
            candidate = Path(expanded)
            if not candidate.is_absolute():
                candidate = root / candidate
            candidate_text = str(candidate)
            if any(char in candidate_text for char in _GLOB_META):
                matches = glob.glob(
                    candidate_text,
                    recursive=True,
                    include_hidden=True,
                )
                blocked.update(Path(match).resolve(strict=False) for match in matches)
            else:
                blocked.add(candidate.resolve(strict=False))
        self._blocked = tuple(sorted(blocked, key=lambda item: str(item)))

    def blocks(self, path: Path) -> bool:
        resolved = path.resolve(strict=False)
        for blocked in self._blocked:
            if resolved == blocked or blocked in resolved.parents:
                return True
        return False


class StructuralIndex:
    """Content-cached structural index for one repository root."""

    def __init__(
        self,
        root: str | Path,
        *,
        extractor: StructuralExtractor | None = None,
        unreadable_paths: Sequence[str] = (),
        ignored_dir_names: Iterable[str] = _DEFAULT_IGNORED_DIR_NAMES,
        path_globs: Sequence[str] = (),
    ) -> None:
        resolved_root = Path(root).resolve()
        if not resolved_root.is_dir():
            raise ValueError(f"structural index root is not a directory: {root}")
        self.root = resolved_root
        self.extractor = extractor or TreeSitterTagExtractor()
        self._unreadable = _UnreadableMatcher(self.root, unreadable_paths)
        self._ignored_dir_names = frozenset(str(name) for name in ignored_dir_names)
        self._path_globs = tuple(str(pattern) for pattern in path_globs)
        self._cache: dict[Path, _CacheEntry] = {}

    def _is_readable_path(self, path: Path) -> bool:
        if path.is_symlink() or self._unreadable.blocks(path):
            return False
        return True

    def _accept_file(self, path: Path) -> bool:
        if not self._is_readable_path(path):
            return False
        if not self._path_globs:
            return True
        relative = path.relative_to(self.root).as_posix()
        return any(fnmatchcase(relative, pattern) for pattern in self._path_globs)

    def _candidate_paths(self) -> tuple[Path, ...]:
        candidates: list[Path] = []
        for raw_dir, dir_names, file_names in os.walk(
            self.root,
            topdown=True,
            followlinks=False,
        ):
            directory = Path(raw_dir)
            dir_names[:] = [
                name
                for name in sorted(dir_names)
                if name not in self._ignored_dir_names
                and self._is_readable_path(directory / name)
            ]
            for name in sorted(file_names):
                path = directory / name
                if self._accept_file(path):
                    candidates.append(path)
        return tuple(candidates)

    def scan(self) -> IndexSnapshot:
        rows: list[StructuralRow] = []
        diagnostics: list[IndexDiagnostic] = []
        files_scanned = 0
        cache_hits = 0
        for path in self._candidate_paths():
            language = self.extractor.detect_language(path)
            if not language:
                continue
            display_path = path.relative_to(self.root).as_posix()
            try:
                source = path.read_bytes()
            except OSError as exc:
                diagnostics.append(
                    IndexDiagnostic(
                        display_path,
                        "read_error",
                        _safe_os_error("could not read source file", exc),
                    )
                )
                continue
            digest = hashlib.sha256(source).hexdigest()
            cached = self._cache.get(path)
            if cached is not None and cached.digest == digest and cached.language == language:
                file_rows = cached.rows
                cache_hits += 1
            else:
                try:
                    file_rows = self.extractor.extract(
                        source,
                        language=language,
                        display_path=display_path,
                    )
                except StructuralLanguageUnsupported as exc:
                    diagnostics.append(
                        IndexDiagnostic(display_path, "unsupported_language", str(exc))
                    )
                    continue
                self._cache[path] = _CacheEntry(digest, language, file_rows)
            files_scanned += 1
            rows.extend(file_rows)
        return IndexSnapshot(
            rows=tuple(sorted(rows, key=_row_sort_key)),
            diagnostics=tuple(
                sorted(
                    diagnostics,
                    key=lambda item: (item.path, item.error_kind, item.message),
                )
            ),
            files_scanned=files_scanned,
            cache_hits=cache_hits,
        )

    def search(
        self,
        *,
        symbol: str | None = None,
        kind: SymbolKind | None = None,
        page: int = 1,
        per_page: int = 25,
        max_rows: int = 1_000,
    ) -> StructuralSearchPage:
        """Search exact symbol names and return one deterministic page."""
        if kind not in (None, "def", "ref"):
            raise ValueError("kind must be 'def', 'ref', or None")
        if page < 1:
            raise ValueError("page must be at least 1")
        if per_page < 1:
            raise ValueError("per_page must be at least 1")
        if max_rows < 1:
            raise ValueError("max_rows must be at least 1")
        snapshot = self.scan()
        matches = tuple(
            row
            for row in snapshot.rows
            if (symbol is None or row.name == symbol)
            and (kind is None or row.kind == kind)
        )
        total = len(matches)
        bounded = matches[:max_rows]
        start = (page - 1) * per_page
        page_rows = bounded[start:start + per_page]
        next_page = page + 1 if start + per_page < len(bounded) else 0
        return StructuralSearchPage(
            rows=page_rows,
            total=total,
            available=len(bounded),
            page=page,
            per_page=per_page,
            next_page=next_page,
            max_rows=max_rows,
            capped=total > len(bounded),
            files_scanned=snapshot.files_scanned,
            cache_hits=snapshot.cache_hits,
            diagnostics=snapshot.diagnostics,
        )


def format_rows(
    rows: Sequence[StructuralRow],
    *,
    max_output_chars: int | None = None,
) -> FormattedRows:
    """Render complete structural rows within an optional character cap."""
    if max_output_chars is not None and max_output_chars < 0:
        raise ValueError("max_output_chars must be non-negative or None")
    rendered: list[str] = []
    used = 0
    limited = False
    for row in rows:
        line = row.render()
        additional = len(line) + (1 if rendered else 0)
        if max_output_chars is not None and used + additional > max_output_chars:
            limited = True
            break
        rendered.append(line)
        used += additional
    if len(rendered) < len(rows):
        limited = True
    return FormattedRows("\n".join(rendered), len(rendered), limited)


def search_repository(
    root: str | Path,
    *,
    symbol: str | None = None,
    kind: SymbolKind | None = None,
    page: int = 1,
    per_page: int = 25,
    max_rows: int = 1_000,
    extractor: StructuralExtractor | None = None,
    unreadable_paths: Sequence[str] = (),
    ignored_dir_names: Iterable[str] = _DEFAULT_IGNORED_DIR_NAMES,
    path_globs: Sequence[str] = (),
) -> StructuralSearchPage:
    """One-shot convenience wrapper for analysis callers."""
    return StructuralIndex(
        root,
        extractor=extractor,
        unreadable_paths=unreadable_paths,
        ignored_dir_names=ignored_dir_names,
        path_globs=path_globs,
    ).search(
        symbol=symbol,
        kind=kind,
        page=page,
        per_page=per_page,
        max_rows=max_rows,
    )


def _row_sort_key(row: StructuralRow) -> tuple[object, ...]:
    return (
        row.path,
        row.line,
        row.column,
        row.kind,
        row.name,
        row.capture,
        row.signature,
        row.language,
    )


def _safe_os_error(action: str, error: OSError) -> str:
    """Describe an OS failure without retaining its absolute filename."""
    details = type(error).__name__
    if error.errno is not None:
        details += f" errno={error.errno}"
    return f"{action} ({details})"


__all__ = [
    "FormattedRows",
    "IndexDiagnostic",
    "IndexSnapshot",
    "StructuralBackendUnavailable",
    "StructuralExtractor",
    "StructuralIndex",
    "StructuralIndexError",
    "StructuralLanguageUnsupported",
    "StructuralRow",
    "StructuralSearchPage",
    "SymbolKind",
    "TreeSitterTagExtractor",
    "format_rows",
    "search_repository",
]
