"""Apply one preview-bound Tree-sitter structural rewrite."""
from __future__ import annotations

import hashlib
import json
import re

from ...config import Config
from ..structural_index import StructuralBackendUnavailable
from ..structural_patterns import StructuralPatternError
from ._common import _is_external_readonly_path, _resolve
from .notebook_edit import _atomic_write
from .structural_search import _match_row, propose_structural_edit


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _error(path: object, kind: str, reason: str) -> str:
    metadata = json.dumps(
        {"path": str(path), "error_kind": kind},
        ensure_ascii=False,
        sort_keys=True,
    )
    return f"ERROR: structural_edit {metadata}: {reason}"


def structural_edit(
    path: str,
    language: str,
    query: str,
    replacement: str,
    expected_sha256: str,
    *,
    cwd: str,
    cfg: Config,
) -> str:
    """Apply the exact rewrite identified by a structural preview hash."""
    if not bool(getattr(cfg, "tools_structural_enabled", False)):
        return _error(
            path,
            "disabled",
            "tool is disabled (tools.structural_enabled=false)",
        )
    if not all(
        isinstance(value, str)
        for value in (path, language, query, replacement, expected_sha256)
    ):
        return _error(
            path,
            "invalid_arguments",
            "path, language, query, replacement, and expected_sha256 must be text",
        )
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        return _error(
            path,
            "invalid_preview_hash",
            "expected_sha256 must be the lowercase hash from structural_search",
        )
    if "\n" in path or "\x00" in path:
        return _error(path, "invalid_path", "path contains a newline or NUL")
    if _is_external_readonly_path(
        cwd,
        path,
        readonly_roots=tuple(getattr(cfg, "skills_readable_dirs", ()) or ()),
    ):
        return _error(path, "read_only", "skill paths are read-only")
    try:
        target = _resolve(cwd, path)
        plan = propose_structural_edit(
            path,
            language,
            query,
            replacement,
            cwd=cwd,
            cfg=cfg,
        )
        if plan.preview_sha256 != expected_sha256:
            return _error(
                path,
                "stale_preview",
                "preview hash does not match current source and rewrite; run "
                "structural_search again and inspect the new preview",
            )
        result_sha256 = hashlib.sha256(plan.proposed).hexdigest()
        metadata = {
            "path": plan.path,
            "language": plan.language,
            "matches": len(plan.matches),
            "source_sha256": plan.source_sha256,
            "result_sha256": result_sha256,
            "preview_sha256": plan.preview_sha256,
        }
        head = (
            "OK: structural_edit "
            + json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        )
        evidence = "\n".join(
            [head, *(_match_row(match) for match in plan.matches)]
        )
        if len(evidence) > int(cfg.max_output_chars):
            return _error(
                path,
                "output_limit",
                "complete changed-location evidence exceeds max_output_chars; "
                "narrow the query",
            )
        try:
            current = target.read_bytes()
        except OSError as exc:
            return _error(
                path,
                "read_error",
                f"could not recheck source ({type(exc).__name__})",
            )
        if current != plan.original:
            return _error(
                path,
                "stale_preview",
                "source changed after preview validation; no change was applied",
            )
        _atomic_write(target, plan.proposed)
        from ..post_edit import run_post_edit_checks

        try:
            check = run_post_edit_checks(path, cwd=cwd, cfg=cfg, trigger="edit")
        except BaseException:
            _atomic_write(target, plan.original)
            raise
        if check.action == "block":
            _atomic_write(target, plan.original)
            return _error(
                path,
                "post_edit_blocked",
                f"post-edit check {check.check_name!r} blocked the rewrite"
                + check.output,
            )
        return evidence + check.output
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
            f"could not edit source ({type(exc).__name__})",
        )
    except Exception:
        return _error(path, "internal_error", "structural edit failed")


__all__ = ["structural_edit"]
