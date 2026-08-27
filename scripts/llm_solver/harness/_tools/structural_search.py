"""Tree-sitter query search and read-only structural rewrite previews."""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

from ...config import Config
from ..sandbox.ignore_policy import active_ignore_policy
from ..structural_index import StructuralBackendUnavailable
from ..structural_patterns import (
    PatternMatch,
    StructuralEditPlan,
    StructuralPatternError,
    build_structural_edit_plan,
    search_structural_patterns,
)
from ._common import _resolve


def _error(path: object, kind: str, reason: str) -> str:
    metadata = json.dumps(
        {"path": str(path), "error_kind": kind},
        ensure_ascii=False,
        sort_keys=True,
    )
    return f"ERROR: structural_search {metadata}: {reason}"


def _settings(cfg: Config | None) -> tuple[int, int, int, int]:
    return (
        int(getattr(cfg, "tools_structural_max_files", 1000)),
        int(getattr(cfg, "tools_structural_max_matches", 100)),
        int(getattr(cfg, "tools_structural_matches_per_page", 25)),
        int(getattr(cfg, "tools_structural_max_file_bytes", 4_194_304)),
    )


def _unreadable_paths(cfg: Config | None) -> tuple[str, ...]:
    return tuple(
        str(item) for item in (getattr(cfg, "unreadable_paths", ()) or ())
    )


def _match_row(match: PatternMatch) -> str:
    payload = {
        "path": match.path,
        "line": match.line,
        "column": match.column,
        "start_byte": match.start_byte,
        "end_byte": match.end_byte,
        "match_sha256": hashlib.sha256(match.text.encode("utf-8")).hexdigest(),
        "text_preview": match.text[:160],
        "captures": [
            {
                "name": capture.name,
                "start_byte": capture.start_byte,
                "end_byte": capture.end_byte,
            }
            for capture in match.captures
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _diff(plan: StructuralEditPlan) -> str:
    before = plan.original.decode("utf-8").splitlines(keepends=True)
    after = plan.proposed.decode("utf-8").splitlines(keepends=True)
    chunks = difflib.unified_diff(
        before,
        after,
        fromfile=f"a/{plan.path} (current workspace)",
        tofile=f"b/{plan.path} (proposed, not applied)",
        lineterm="\n",
    )
    return "".join(chunks)


def _bounded_preview(plan: StructuralEditPlan, *, max_chars: int) -> str:
    result_sha256 = hashlib.sha256(plan.proposed).hexdigest()
    metadata = {
        "path": plan.path,
        "language": plan.language,
        "matches": len(plan.matches),
        "source_sha256": plan.source_sha256,
        "result_sha256": result_sha256,
        "preview_sha256": plan.preview_sha256,
        "state": "not_applied",
    }
    lines = [
        "OK: structural_preview "
        + json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        *(_match_row(match) for match in plan.matches),
        "DIFF (proposed, not applied):",
    ]
    prefix = "\n".join(lines) + "\n"
    if len(prefix) >= max_chars:
        raise StructuralPatternError(
            "preview_too_large",
            "the location report exceeds max_output_chars; narrow the query",
        )
    diff = _diff(plan)
    remaining = max_chars - len(prefix)
    if len(diff) <= remaining:
        return prefix + diff
    marker = "\n... [diff preview truncated; preview hash covers the full change]\n"
    if len(marker) > remaining:
        raise StructuralPatternError(
            "preview_too_large",
            "the location report leaves no room for a diff marker; narrow the query",
        )
    budget = max(0, remaining - len(marker))
    shown: list[str] = []
    used = 0
    for line in diff.splitlines(keepends=True):
        if used + len(line) > budget:
            break
        shown.append(line)
        used += len(line)
    return prefix + "".join(shown) + marker


def propose_structural_edit(
    path: str,
    language: str,
    query: str,
    replacement: str,
    *,
    cwd: str,
    cfg: Config | None = None,
) -> StructuralEditPlan:
    """Build the exact preview shared by search, approval, and mutation."""
    if not all(
        isinstance(value, str)
        for value in (path, language, query, replacement)
    ):
        raise StructuralPatternError(
            "invalid_arguments",
            "path, language, query, and replacement must be text",
        )
    if "\n" in path or "\x00" in path:
        raise StructuralPatternError(
            "invalid_path", "path contains a newline or NUL"
        )
    target = _resolve(cwd, path)
    _max_files, max_matches, _per_page, max_file_bytes = _settings(cfg)
    return build_structural_edit_plan(
        workspace=cwd,
        target=target,
        language=language,
        query_source=query,
        replacement=replacement,
        unreadable_paths=_unreadable_paths(cfg),
        ignore_policy=active_ignore_policy(cwd),
        max_matches=max_matches,
        max_file_bytes=max_file_bytes,
    )


def structural_search(
    path: str,
    language: str,
    query: str,
    *,
    cwd: str,
    cfg: Config,
    glob: str = "",
    replacement: str | None = None,
    page: int = 1,
) -> str:
    """Search source structure or preview one exact single-file rewrite."""
    if not bool(getattr(cfg, "tools_structural_enabled", False)):
        return _error(
            path,
            "disabled",
            "tool is disabled (tools.structural_enabled=false)",
        )
    if not all(isinstance(value, str) for value in (path, language, query, glob)):
        return _error(
            path,
            "invalid_arguments",
            "path, language, query, and glob must be text",
        )
    if replacement is not None and not isinstance(replacement, str):
        return _error(path, "invalid_arguments", "replacement must be text")
    if "\n" in path or "\x00" in path:
        return _error(path, "invalid_path", "path contains a newline or NUL")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        return _error(path, "invalid_page", "page must be an integer >= 1")
    pagination = bool(getattr(cfg, "search_pagination_enabled", True))
    if not pagination and page != 1:
        return _error(
            path,
            "pagination_disabled",
            "page must be 1 when pagination is disabled",
        )
    try:
        scope = _resolve(cwd, path)
        if replacement is not None:
            if glob:
                return _error(
                    path,
                    "invalid_arguments",
                    "glob is not allowed when previewing a single-file rewrite",
                )
            if not scope.is_file():
                return _error(
                    path,
                    "single_file_required",
                    "replacement previews require path to name one source file",
                )
            plan = propose_structural_edit(
                path,
                language,
                query,
                replacement,
                cwd=cwd,
                cfg=cfg,
            )
            return _bounded_preview(plan, max_chars=int(cfg.max_output_chars))

        (
            max_files,
            max_matches,
            configured_per_page,
            max_file_bytes,
        ) = _settings(cfg)
        result = search_structural_patterns(
            workspace=cwd,
            scope=scope,
            language=language,
            query_source=query,
            path_glob=glob,
            unreadable_paths=_unreadable_paths(cfg),
            ignore_policy=active_ignore_policy(cwd),
            max_files=max_files,
            max_matches=max_matches,
            max_file_bytes=max_file_bytes,
        )
        if result.total == 0:
            diagnostic_kinds = sorted({item.kind for item in result.diagnostics})
            detail = (
                "; source diagnostics: " + ", ".join(diagnostic_kinds)
                if diagnostic_kinds
                else ""
            )
            kind = (
                diagnostic_kinds[0]
                if len(diagnostic_kinds) == 1
                else "no_match"
            )
            return _error(path, kind, "query matched no source nodes" + detail)
        per_page = configured_per_page if pagination else max_matches
        start = (page - 1) * per_page
        rows = result.matches[start:start + per_page]
        if not rows:
            return _error(
                path,
                "invalid_page",
                f"page {page} is outside the available structural results",
            )
        next_page = page + 1 if start + len(rows) < len(result.matches) else 0
        metadata = {
            "path": path,
            "language": language,
            "total": result.total,
            "available": len(result.matches),
            "shown": len(rows),
            "page": page,
            "next_page": next_page,
            "capped": result.capped,
            "files_scanned": result.files_scanned,
            "diagnostics": len(result.diagnostics),
        }
        lines = [
            "OK: structural_search "
            + json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            *(_match_row(match) for match in rows),
        ]
        for diagnostic in result.diagnostics:
            lines.append(json.dumps(
                {
                    "diagnostic": diagnostic.kind,
                    "path": diagnostic.path,
                },
                ensure_ascii=False,
                sort_keys=True,
            ))
        output = "\n".join(lines)
        if len(output) > int(cfg.max_output_chars):
            return _error(
                path,
                "output_limit",
                "complete result rows exceed max_output_chars; narrow the query",
            )
        return output
    except StructuralPatternError as exc:
        return _error(path, exc.kind, str(exc))
    except StructuralBackendUnavailable:
        return _error(
            path,
            "backend_unavailable",
            "structural backend unavailable; reinstall Yuj with its tree-sitter dependencies",
        )
    except ValueError as exc:
        return _error(path, "path_outside_cwd", str(exc))
    except OSError as exc:
        return _error(
            path,
            "os_error",
            f"could not inspect source ({type(exc).__name__})",
        )
    except Exception:
        return _error(path, "internal_error", "structural search failed")


__all__ = [
    "propose_structural_edit",
    "structural_search",
]
