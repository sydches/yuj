"""Deterministic, token-bounded repository map construction.

The module is a leaf over :mod:`structural_index`: it reuses the same
tree-sitter tags queries as ``list_definitions`` and has no dependency on the
harness loop or central ``Config`` object.  Cache files contain mechanical
symbol rows only.  Task personalization and token fitting are recomputed for
every solver session.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import html
import json
import logging
from pathlib import Path
from typing import Any

from .context import chars_div_4
from .structural_index import IndexSnapshot, StructuralIndex, StructuralRow


log = logging.getLogger(__name__)

REPO_MAP_REFRESH_POLICIES = frozenset({"auto", "always", "files", "manual"})
_CACHE_SCHEMA_VERSION = 1
_CACHE_FILE_NAME = "symbols.v1.json"
_CACHE_MAX_BYTES = 64 * 1024 * 1024
_CACHE_MAX_ROWS = 2_000_000
_PAGERANK_DAMPING = 0.85
_PAGERANK_ITERATIONS = 32
_MAP_OPENING = "<repo-map>\n# Ranked definitions; task-referenced symbols first"
_MAP_CLOSING = "</repo-map>"


def normalize_repo_map_refresh(value: object) -> str:
    """Validate and return one public refresh policy."""
    if not isinstance(value, str) or value not in REPO_MAP_REFRESH_POLICIES:
        choices = " | ".join(sorted(REPO_MAP_REFRESH_POLICIES))
        raise ValueError(
            "config error: context.repo_map_refresh must be one of "
            f"{choices}, got {value!r}."
        )
    return value


@dataclass(frozen=True, slots=True)
class RepoMapResult:
    """One session's immutable model block plus secret-free provenance."""

    content: str = ""
    tokens: int = 0
    files: int = 0
    symbols: int = 0
    cache_hit: bool = False
    refresh: str = "auto"
    sha256: str | None = None

    def trace_fields(self) -> dict[str, object]:
        return {
            "repo_map_tokens": self.tokens,
            "repo_map_refresh": self.refresh,
            "repo_map_files": self.files,
            "repo_map_symbols": self.symbols,
            "repo_map_cache_hit": self.cache_hit,
            "repo_map_sha256": self.sha256,
        }


def append_repo_map(task_message: str, result: RepoMapResult) -> str:
    """Append an enabled map without changing any existing task bytes."""
    if not result.content:
        return task_message
    return task_message + "\n\n" + result.content


def build_repo_map(
    root: str | Path,
    *,
    task_message: str,
    ranking_text: str | None = None,
    token_budget: int,
    refresh: str = "auto",
    cache_dir: str | Path | None = None,
    unreadable_paths: Sequence[str] = (),
    tokenizer: object | None = None,
    token_estimator: Callable[[list[dict]], int] | None = None,
    extractor: object | None = None,
) -> RepoMapResult:
    """Build one ranked ``<repo-map>`` block within ``token_budget``.

    Token cost is the incremental count of appending the block to the task
    user message.  A configured local tokenizer therefore measures the same
    message boundary that the model receives.  With no local tokenizer, the
    active profile estimator is used, then the documented chars/4 fallback.
    ``ranking_text`` lets the runtime rank from the original task statement
    while still fitting against pretest or resume framing in ``task_message``.
    """
    if isinstance(token_budget, bool) or not isinstance(token_budget, int):
        raise ValueError("repo-map token budget must be an integer")
    if token_budget < 0:
        raise ValueError("repo-map token budget must be non-negative")
    policy = normalize_repo_map_refresh(refresh)
    if token_budget == 0:
        return RepoMapResult(refresh=policy)
    count = _incremental_counter(
        task_message,
        tokenizer=tokenizer,
        token_estimator=token_estimator,
    )
    if count(_MAP_OPENING + "\n" + _MAP_CLOSING) > token_budget:
        return RepoMapResult(refresh=policy)

    index = StructuralIndex(
        root,
        extractor=extractor,
        unreadable_paths=unreadable_paths,
    )
    snapshot, cache_hit = _load_or_scan(
        index,
        refresh=policy,
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
        scope_identity=_cache_scope_identity(index, unreadable_paths),
    )
    ranked = _ranked_definitions(
        snapshot.rows,
        task_message if ranking_text is None else ranking_text,
    )
    if not ranked:
        return RepoMapResult(
            files=snapshot.files_scanned,
            cache_hit=cache_hit,
            refresh=policy,
        )

    content, tokens, symbols = _fit_ranked_rows(
        ranked,
        token_budget=token_budget,
        count=count,
    )
    return RepoMapResult(
        content=content,
        tokens=tokens,
        files=snapshot.files_scanned,
        symbols=symbols,
        cache_hit=cache_hit,
        refresh=policy,
        sha256=(
            hashlib.sha256(content.encode("utf-8")).hexdigest()
            if content
            else None
        ),
    )


def _cache_path(cache_dir: Path | None) -> Path | None:
    return None if cache_dir is None else cache_dir / _CACHE_FILE_NAME


def _root_identity(root: Path) -> str:
    return hashlib.sha256(
        str(root.resolve()).encode("utf-8", errors="surrogateescape")
    ).hexdigest()


def _cache_scope_identity(
    index: StructuralIndex,
    unreadable_paths: Sequence[str],
) -> str:
    """Bind even manual reuse to the current parser and visibility policy."""
    extractor_type = type(index.extractor)
    payload = {
        "extractor": (
            f"{extractor_type.__module__}.{extractor_type.__qualname__}"
        ),
        "unreadable_paths": sorted(str(item) for item in unreadable_paths),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_or_scan(
    index: StructuralIndex,
    *,
    refresh: str,
    cache_dir: Path | None,
    scope_identity: str,
) -> tuple[IndexSnapshot, bool]:
    cache_path = _cache_path(cache_dir)
    cached = _read_cache(
        cache_path,
        root=index.root,
        scope_identity=scope_identity,
    )
    fingerprint_kind = ""
    fingerprint = ""

    if refresh == "manual" and cached is not None:
        return cached[0], True
    if refresh in {"auto", "files"}:
        fingerprint_kind = refresh
        fingerprint = index.fingerprint(contents=refresh == "files")
        if (
            cached is not None
            and cached[1] == fingerprint_kind
            and cached[2] == fingerprint
        ):
            return cached[0], True

    snapshot = index.scan()
    _write_cache(
        cache_path,
        root=index.root,
        scope_identity=scope_identity,
        snapshot=snapshot,
        fingerprint_kind=fingerprint_kind,
        fingerprint=fingerprint,
    )
    return snapshot, False


def _read_cache(
    path: Path | None,
    *,
    root: Path,
    scope_identity: str,
) -> tuple[IndexSnapshot, str, str] | None:
    if path is None or not path.is_file():
        return None
    try:
        if path.stat().st_size > _CACHE_MAX_BYTES:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            return None
        if data.get("schema_version") != _CACHE_SCHEMA_VERSION:
            return None
        if data.get("root_identity") != _root_identity(root):
            return None
        if data.get("scope_identity") != scope_identity:
            return None
        raw_rows = data.get("rows")
        if not isinstance(raw_rows, list) or len(raw_rows) > _CACHE_MAX_ROWS:
            return None
        rows = tuple(_row_from_cache(item) for item in raw_rows)
        files_scanned = data.get("files_scanned", 0)
        if (
            isinstance(files_scanned, bool)
            or not isinstance(files_scanned, int)
            or files_scanned < 0
        ):
            return None
        fingerprint_kind = data.get("fingerprint_kind", "")
        fingerprint = data.get("fingerprint", "")
        if not isinstance(fingerprint_kind, str) or not isinstance(fingerprint, str):
            return None
        return (
            IndexSnapshot(
                rows=tuple(sorted(rows, key=_public_row_sort_key)),
                diagnostics=(),
                files_scanned=files_scanned,
                cache_hits=files_scanned,
            ),
            fingerprint_kind,
            fingerprint,
        )
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _row_from_cache(value: object) -> StructuralRow:
    if not isinstance(value, Mapping):
        raise ValueError("repo-map cache row must be an object")
    kind = value.get("kind")
    if kind not in {"def", "ref"}:
        raise ValueError("repo-map cache row has invalid kind")
    line = value.get("line")
    column = value.get("column")
    if (
        isinstance(line, bool)
        or not isinstance(line, int)
        or line < 1
        or isinstance(column, bool)
        or not isinstance(column, int)
        or column < 1
    ):
        raise ValueError("repo-map cache row has invalid location")
    strings = {}
    for name in ("path", "name", "signature", "language", "capture"):
        item = value.get(name)
        if not isinstance(item, str):
            raise ValueError(f"repo-map cache row has invalid {name}")
        strings[name] = item
    return StructuralRow(
        path=strings["path"],
        line=line,
        column=column,
        kind=kind,
        name=strings["name"],
        signature=strings["signature"],
        language=strings["language"],
        capture=strings["capture"],
    )


def _write_cache(
    path: Path | None,
    *,
    root: Path,
    scope_identity: str,
    snapshot: IndexSnapshot,
    fingerprint_kind: str,
    fingerprint: str,
) -> None:
    if path is None:
        return
    if snapshot.diagnostics:
        log.warning(
            "repo-map cache not updated after %d indexing diagnostic(s)",
            len(snapshot.diagnostics),
        )
        return
    if len(snapshot.rows) > _CACHE_MAX_ROWS:
        log.warning("repo-map cache row limit exceeded; cache not updated")
        return
    rows = [
        {
            "capture": row.capture,
            "column": row.column,
            "kind": row.kind,
            "language": row.language,
            "line": row.line,
            "name": row.name,
            "path": row.path,
            "signature": row.signature,
        }
        for row in sorted(snapshot.rows, key=_public_row_sort_key)
    ]
    payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "root_identity": _root_identity(root),
        "scope_identity": scope_identity,
        "fingerprint_kind": fingerprint_kind,
        "fingerprint": fingerprint,
        "files_scanned": snapshot.files_scanned,
        "rows": rows,
    }
    try:
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        if len(encoded) > _CACHE_MAX_BYTES:
            log.warning("repo-map cache byte limit exceeded; cache not updated")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(encoded)
        temporary.replace(path)
    except OSError as exc:
        log.warning("repo-map cache write failed: %s", type(exc).__name__)


def _task_mentions_path(task: str, candidate: str) -> bool:
    """Return whether ``candidate`` appears as one complete task path token.

    A final sentence period is punctuation, while the period in
    ``module.py.backup`` continues the path.  Checking those cases explicitly
    avoids both substring matches and the common ``src/module.py.`` miss.
    """
    start = 0
    while True:
        index = task.find(candidate, start)
        if index < 0:
            return False
        before = task[index - 1] if index else ""
        end = index + len(candidate)
        after = task[end] if end < len(task) else ""
        after_next = task[end + 1] if end + 1 < len(task) else ""
        left_continues = bool(
            before and (before.isalnum() or before in "_./\\-")
        )
        right_continues = bool(
            after
            and (
                after.isalnum()
                or after in "_/\\-"
                or (
                    after == "."
                    and after_next
                    and (after_next.isalnum() or after_next == "_")
                )
            )
        )
        if not left_continues and not right_continues:
            return True
        start = index + 1


def _mentioned_files(paths: Sequence[str], task: str) -> frozenset[str]:
    normalized = str(task).replace("\\", "/")
    basename_counts: dict[str, int] = defaultdict(int)
    for path in paths:
        basename_counts[Path(path).name] += 1
    mentioned: set[str] = set()
    for path in paths:
        candidates = [path]
        basename = Path(path).name
        if basename_counts[basename] == 1:
            candidates.append(basename)
        for candidate in candidates:
            if _task_mentions_path(normalized, candidate):
                mentioned.add(path)
                break
    return frozenset(mentioned)


def _ranked_definitions(
    rows: Sequence[StructuralRow],
    task: str,
) -> tuple[StructuralRow, ...]:
    definitions = tuple(row for row in rows if row.kind == "def")
    if not definitions:
        return ()
    paths = tuple(sorted({row.path for row in rows}))
    mentioned = _mentioned_files(paths, task)
    definitions_by_name: dict[str, set[str]] = defaultdict(set)
    references_by_name: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.kind == "def":
            definitions_by_name[row.name].add(row.path)
        else:
            references_by_name[row.name].append(row.path)

    edge_weights: dict[str, dict[str, float]] = {
        path: defaultdict(float) for path in paths
    }
    for name, sources in references_by_name.items():
        targets = tuple(sorted(definitions_by_name.get(name, ())))
        if not targets:
            continue
        share = 1.0 / len(targets)
        for source in sources:
            for target in targets:
                edge_weights[source][target] += share
    ranks = _personalized_pagerank(paths, edge_weights, mentioned)

    def score(row: StructuralRow) -> tuple[float, ...]:
        sources = references_by_name.get(row.name, ())
        personalized_refs = sum(1 for source in sources if source in mentioned)
        weighted_refs = sum(ranks.get(source, 0.0) for source in sources)
        return (
            float(personalized_refs),
            weighted_refs,
            ranks.get(row.path, 0.0),
            float(len(sources)),
        )

    return tuple(sorted(
        definitions,
        key=lambda row: (
            *(-item for item in score(row)),
            row.path,
            row.line,
            row.column,
            row.name,
            row.signature,
        ),
    ))


def _personalized_pagerank(
    paths: Sequence[str],
    edges: Mapping[str, Mapping[str, float]],
    mentioned: frozenset[str],
) -> dict[str, float]:
    if not paths:
        return {}
    uniform = 1.0 / len(paths)
    if mentioned:
        # Keep a small uniform floor so disconnected repository regions stay
        # rankable while most restart mass remains on files named by the task.
        personalization = {
            path: 0.10 * uniform
            + (0.90 / len(mentioned) if path in mentioned else 0.0)
            for path in paths
        }
    else:
        personalization = {path: uniform for path in paths}
    ranks = dict(personalization)
    for _ in range(_PAGERANK_ITERATIONS):
        updated = {
            path: (1.0 - _PAGERANK_DAMPING) * personalization[path]
            for path in paths
        }
        dangling_mass = 0.0
        for source in paths:
            outgoing = edges.get(source, {})
            total = sum(outgoing.values())
            if total > 0:
                for target in sorted(outgoing):
                    updated[target] += (
                        _PAGERANK_DAMPING
                        * ranks[source]
                        * outgoing[target]
                        / total
                    )
            else:
                dangling_mass += ranks[source]
        if dangling_mass:
            for target in paths:
                updated[target] += (
                    _PAGERANK_DAMPING
                    * dangling_mass
                    * personalization[target]
                )
        ranks = updated
    return ranks


def _incremental_counter(
    task_message: str,
    *,
    tokenizer: object | None,
    token_estimator: Callable[[list[dict]], int] | None,
) -> Callable[[str], int]:
    base_message = [{"role": "user", "content": task_message}]

    def count_messages(messages: list[dict]) -> int:
        if tokenizer is not None:
            method = getattr(tokenizer, "count", None)
            if not callable(method):
                raise TypeError("repo-map tokenizer must expose count(messages)")
            try:
                return int(method(messages, tools=None))
            except TypeError:
                return int(method(messages))
        estimator = token_estimator or chars_div_4
        return int(estimator(messages))

    base_tokens = count_messages(base_message)

    def incremental(block: str) -> int:
        combined = [{
            "role": "user",
            "content": task_message + "\n\n" + block,
        }]
        return max(0, count_messages(combined) - base_tokens)

    return incremental


def _fit_ranked_rows(
    rows: Sequence[StructuralRow],
    *,
    token_budget: int,
    count: Callable[[str], int],
) -> tuple[str, int, int]:
    if count(_MAP_OPENING + "\n" + _MAP_CLOSING) > token_budget:
        return "", 0, 0
    rendered: list[str] = []
    final_content = ""
    final_tokens = 0
    for row in rows:
        signature = row.signature or f"def {row.name}"
        full = _escape_map_line(f"{row.path}:{row.line} {signature}")
        short = _escape_map_line(f"{row.path}:{row.line} def {row.name}")
        selected = None
        for candidate_line in dict.fromkeys((full, short)):
            candidate = "\n".join(
                (_MAP_OPENING, *rendered, candidate_line, _MAP_CLOSING)
            )
            candidate_tokens = count(candidate)
            if candidate_tokens <= token_budget:
                selected = (candidate_line, candidate, candidate_tokens)
                break
        if selected is None:
            break
        line, final_content, final_tokens = selected
        rendered.append(line)
    if not rendered:
        return "", 0, 0
    return final_content, final_tokens, len(rendered)


def _escape_map_line(value: str) -> str:
    """Keep every mechanical row on one escaped XML-text line."""
    safe = value.encode("utf-8", errors="replace").decode("utf-8")
    return (
        html.escape(safe, quote=False)
        .replace("\r", r"\r")
        .replace("\n", r"\n")
        .replace("\t", r"\t")
    )


def _public_row_sort_key(row: StructuralRow) -> tuple[Any, ...]:
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


__all__ = [
    "REPO_MAP_REFRESH_POLICIES",
    "RepoMapResult",
    "append_repo_map",
    "build_repo_map",
    "normalize_repo_map_refresh",
]
