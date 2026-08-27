"""Model-facing unified-diff tool dispatcher."""
from __future__ import annotations

from ...config import Config


class AppliedUnifiedDiffResult(str):
    """String-compatible result carrying mechanically verified operations."""

    def __new__(cls, text: str, operations) -> "AppliedUnifiedDiffResult":
        value = super().__new__(cls, text)
        value.applied_operations = tuple(
            (str(operation.kind), str(operation.path)) for operation in operations
        )
        return value


def _runtime_edit_format(cfg: Config) -> str:
    effective = str(getattr(cfg, "effective_edit_format", "") or "")
    configured = str(getattr(cfg, "tools_edit_format", "") or "")
    if configured:
        return configured
    if bool(getattr(cfg, "tools_apply_patch_enabled", False)):
        return "apply_patch"
    if effective:
        return effective
    return "exact"


def udiff_tool(patch: str, *, cwd: str, cfg: Config) -> str:
    """Parse, pre-verify, and apply a standard unified diff."""
    from ..udiff import (
        UnifiedDiffApplyError,
        UnifiedDiffParseError,
        parse_unified_diff,
        verify_and_apply_unified_diff,
    )

    if _runtime_edit_format(cfg) != "udiff":
        return "ERROR: udiff is unavailable for the selected edit format"
    try:
        patches = parse_unified_diff(patch)
    except UnifiedDiffParseError as exc:
        return f"ERROR: udiff parse: {exc}"
    try:
        result, operations = verify_and_apply_unified_diff(
            patches,
            cwd,
            candidate_count=int(getattr(cfg, "edit_candidate_count", 3) or 3),
        )
    except UnifiedDiffApplyError as exc:
        return f"ERROR: udiff {exc.kind}: {exc}"

    from ..post_edit import run_post_edit_actions
    action_tail = ""
    for operation in operations:
        if operation.kind == "delete":
            continue
        check = run_post_edit_actions(
            operation.path, cwd=cwd, cfg=cfg, trigger="udiff"
        )
        if check.output:
            action_tail += check.output
    return AppliedUnifiedDiffResult(result + action_tail, operations)


__all__ = ["AppliedUnifiedDiffResult", "udiff_tool"]
