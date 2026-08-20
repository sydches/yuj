"""Focus + path-selection helpers for WorkingSetBaselineContext."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ._working_set_baseline_helpers import (
    _extract_action_target, _extract_test_target_from_action,
    _looks_like_test_path,
)

if TYPE_CHECKING:
    from ._working_set_baseline import WorkingSetBaselineContext


def focus_candidates(ctx: "WorkingSetBaselineContext") -> list[str]:
    focus = changed_paths(ctx)
    if focus:
        return focus
    focus = visible_paths(ctx)
    if focus:
        return focus
    return recent_action_targets(ctx)


def candidate_source_paths(ctx: "WorkingSetBaselineContext") -> list[str]:
    candidates: list[str] = []
    for path in focus_candidates(ctx):
        if (
            _looks_like_test_path(path)
            or path in candidates
            or not is_repo_file_candidate(ctx, path)
        ):
            continue
        candidates.append(path)
        if len(candidates) >= ctx._slot_max_candidates:
            break
    return candidates


def candidate_test_paths(ctx: "WorkingSetBaselineContext") -> list[str]:
    targets: list[str] = []
    for target in candidate_test_targets(ctx):
        if target in targets:
            continue
        targets.append(target)
        if len(targets) >= ctx._slot_max_candidates:
            break
    return targets


def candidate_test_targets(ctx: "WorkingSetBaselineContext") -> list[str]:
    targets: list[str] = []
    for rec in reversed(ctx._trace_records()):
        target = _extract_test_target_from_action(rec.action)
        if not target or target in targets:
            continue
        targets.append(target)
        if len(targets) >= max(2, ctx._slot_max_candidates):
            break
    return list(reversed(targets))


def focus_files_text(ctx: "WorkingSetBaselineContext") -> str:
    return ctx._format_path_list(focus_candidates(ctx), limit=3)


def changed_paths(ctx: "WorkingSetBaselineContext") -> list[str]:
    changed = [slot for slot in ctx._ws.files.values() if slot.epoch > 0]
    changed.sort(key=lambda slot: slot.last_access_turn, reverse=True)
    return [slot.path for slot in changed]


def visible_paths(ctx: "WorkingSetBaselineContext") -> list[str]:
    visible = sorted(
        ctx._ws.files.values(),
        key=lambda slot: slot.last_access_turn,
        reverse=True,
    )
    return [slot.path for slot in visible]


def recent_action_targets(ctx: "WorkingSetBaselineContext") -> list[str]:
    targets: list[str] = []
    for rec in reversed(ctx._trace_records()):
        target = _extract_action_target(rec.action)
        if not target or target in targets:
            continue
        targets.append(target)
        if len(targets) >= 4:
            break
    return targets


def test_target_text(ctx: "WorkingSetBaselineContext") -> str:
    return ctx._format_path_list(candidate_test_targets(ctx), limit=2)


def is_repo_file_candidate(ctx: "WorkingSetBaselineContext", path: str) -> bool:
    if not path or path in {".", ".."}:
        return False
    if path.endswith("/"):
        return False
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate.relative_to(ctx._cwd)
        except ValueError:
            return False
    else:
        candidate = (ctx._cwd / candidate).resolve(strict=False)
    try:
        if candidate.exists():
            return candidate.is_file()
    except OSError:
        return False
    name = candidate.name
    return bool(name) and "." in name and not name.startswith(".")


def needs_test_read(ctx: "WorkingSetBaselineContext") -> bool:
    target = test_target_text(ctx)
    if not target:
        return False
    seen_paths = changed_paths(ctx) + visible_paths(ctx)
    return not any(_looks_like_test_path(p) for p in seen_paths)
