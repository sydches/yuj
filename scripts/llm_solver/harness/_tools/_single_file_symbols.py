"""Structural definitions from one already-resolved, permitted file."""
from pathlib import Path

from ...config import Config
from ..structural_index import (
    StructuralBackendUnavailable, StructuralLanguageUnsupported,
    StructuralSearchPage, TreeSitterTagExtractor,
)


def list_file_symbols(path: Path, display_path: str, cfg: Config) -> str:
    # Imported at call time because the public tool also owns its envelopes.
    from .list_definitions import _list_definitions_error, _render_repository_page

    extractor = TreeSitterTagExtractor()
    language = extractor.detect_language(path)
    if language is None:
        return _list_definitions_error(
            display_path, "unsupported_suffix",
            f"list_definitions does not support {path.suffix.lower()!r} files. "
            "Use read() for unsupported types.",
        )
    try:
        source = path.read_bytes()
        rows = tuple(row for row in extractor.extract(
            source, language=language, display_path=display_path,
        ) if row.kind == "def")
    except OSError:
        return _list_definitions_error(display_path, "os_error", "could not read requested file")
    except StructuralLanguageUnsupported:
        return _list_definitions_error(
            display_path, "unsupported_language", "no installed definition query for this language",
        )
    except StructuralBackendUnavailable:
        return _list_definitions_error(
            display_path, "backend_unavailable", "installed structural parser is unavailable",
        )
    max_rows = int(cfg.tools_ast_search_max_rows)
    selected = rows[:max_rows]
    page = StructuralSearchPage(
        rows=selected, total=len(rows), available=len(selected), page=1,
        per_page=max_rows, next_page=0, max_rows=max_rows,
        capped=len(rows) > len(selected), files_scanned=1, cache_hits=0,
        diagnostics=(),
    )
    envelope = _render_repository_page(
        page, path=display_path, max_chars=int(cfg.max_output_chars), mode="file",
    )
    from ..savings import get_ledger
    get_ledger().record_transform(
        bucket="outline_vs_read", layer="harness", mechanism="list_definitions",
        before=source.decode("utf-8", errors="replace"), after=envelope,
        surface="tool_output", ctx={"path": display_path, "suffix": path.suffix.lower()},
    )
    return envelope
