"""State / phase / obligation text rendering for WorkingSetBaselineContext."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._working_set_baseline_helpers import _fit_lines

if TYPE_CHECKING:
    from ._working_set_baseline import WorkingSetBaselineContext


def state_text(ctx: "WorkingSetBaselineContext", max_chars: int) -> str:
    if ctx._contract == "slot":
        return slot_state_text(ctx, max_chars)
    lines: list[str] = []
    state_block = ctx._load_state_json().get("state") if ctx._has_solver_state() else None
    if isinstance(state_block, dict):
        for key in ("current_attempt", "last_verify", "next_action"):
            value = state_block.get(key)
            if value:
                label = key.replace("_", " ").capitalize()
                lines.append(f"{label}: {value}")

    lines.append("Working root: . (current directory already set)")
    lines.append(f"Phase: {phase_text(ctx)}")

    blocker = ctx._blocking_record()
    if blocker is not None:
        lines.append(f"Blocking command: {ctx._summary_line(blocker)}")

    focus = ctx._focus_files_text()
    if focus:
        lines.append(f"Focus files: {focus}")

    test_target = ctx._test_target_text()
    if test_target:
        lines.append(f"Test target: {test_target}")

    obligation = obligation_text(ctx)
    if obligation:
        lines.append(f"Next obligation: {obligation}")

    last_action = last_action_text(ctx)
    if last_action and not any(line.startswith("Current attempt:") for line in lines):
        lines.append(f"Last action: {last_action}")

    changed = ctx._format_path_list(ctx._changed_paths())
    if changed:
        lines.append(f"Files changed: {changed}")

    in_view = ctx._format_path_list(ctx._visible_paths())
    if in_view:
        lines.append(f"Files in view: {in_view}")

    if not lines or not any(line.startswith("Turn:") for line in lines):
        lines.append(f"Turn: {ctx._turn_count}")
    return _fit_lines(lines, max_chars)


def phase_text(ctx: "WorkingSetBaselineContext") -> str:
    recovery = ctx._recovery_state()
    if recovery is not None:
        return "recovery"
    changed = ctx._changed_paths()
    visible = ctx._focus_candidates()
    blocker = ctx._blocking_record()
    repeated = ctx._latest_repeated_trace_run()
    needs_test = ctx._needs_test_read()
    if changed:
        if blocker is not None and blocker.verdict.startswith("FAIL"):
            return "verify the latest change against the active blocker"
        return "verify or refine the latest change"
    if repeated is not None:
        return "leave inspection and choose a concrete file or check"
    if needs_test:
        return "read the focused test before another verification run"
    if blocker is not None and visible:
        return "prepare a targeted edit"
    if blocker is not None:
        return "investigate the active blocker"
    if visible:
        return "inspect files in view"
    return "orient"


def slot_state_text(ctx: "WorkingSetBaselineContext", max_chars: int) -> str:
    recovery = ctx._recovery_state()
    lines: list[str] = [f"repo_root: .", f"phase: {phase_text(ctx)}"]
    retained_thoughts = [
        entry.args_summary
        for entry in ctx._turn_entries
        if entry.tool_name == "think" and entry.args_summary
    ]
    if retained_thoughts:
        lines.append("scratchpad: " + " | ".join(retained_thoughts))

    candidate_source = ctx._format_path_list(ctx._candidate_source_paths(), limit=ctx._slot_max_candidates)
    candidate_test = ctx._format_path_list(ctx._candidate_test_paths(), limit=ctx._slot_max_candidates)
    edited = ctx._format_path_list(ctx._changed_paths(), limit=1)
    last_verdict = ctx._last_verdict_text()
    disallowed = ctx._disallowed_repeat_text()

    if recovery is not None:
        reason, target = recovery
        lines.append(f"stuck_reason: {reason}")
        if target:
            lines.append(f"focused_target: {target}")
        if candidate_source:
            lines.append(f"candidate_source: {candidate_source}")
        if candidate_test:
            lines.append(f"candidate_test: {candidate_test}")
        if last_verdict:
            lines.append(f"last_verdict: {last_verdict}")
        lines.append(
            "allowed_moves: read a concrete file | edit/write | run verification"
        )
        return _fit_lines(lines, max_chars)

    if candidate_source:
        lines.append(f"candidate_source: {candidate_source}")
    if candidate_test:
        lines.append(f"candidate_test: {candidate_test}")
    if edited:
        lines.append(f"edited_file: {edited}")
    if last_verdict:
        lines.append(f"last_verdict: {last_verdict}")
    if disallowed:
        lines.append(f"disallowed_repeat: {disallowed}")
    lines.append(f"next_action: {ctx._slot_next_action_text()}")
    return _fit_lines(lines, max_chars)


def obligation_text(ctx: "WorkingSetBaselineContext") -> str:
    changed = ctx._changed_paths()
    focus = ctx._focus_files_text()
    blocker = ctx._blocking_record()
    test_target = ctx._test_target_text()
    repeated = ctx._latest_repeated_trace_run()
    if changed and blocker is not None:
        return (
            f"verify or extend {focus} against the blocker before more exploration"
            if focus else
            "verify or extend the changed file before more exploration"
        )
    if changed:
        return (
            f"run the next focused verification on {focus}"
            if focus else
            "run the next focused verification"
        )
    if repeated is not None:
        repeated_target = ctx._repeated_target_text(repeated[0].action)
        if test_target and ctx._needs_test_read():
            return f"stop repeating {repeated_target}; read {test_target} before more checks"
        if focus:
            return f"stop repeating {repeated_target}; read or edit one focus target ({focus})"
        return f"stop repeating {repeated_target}; choose a new target or make an edit"
    if test_target and ctx._needs_test_read():
        return f"read {test_target} before another verification run"
    if blocker is not None and focus:
        return f"edit one focus file ({focus}) or change checks; do not repeat the same blocker unchanged"
    if blocker is not None:
        return "pick one concrete target before repeating checks"
    if focus:
        return f"choose one focus file ({focus}) and make the next concrete move"
    return "identify one concrete file or command target"


def last_action_text(ctx: "WorkingSetBaselineContext") -> str:
    if ctx._turn_entries:
        e = ctx._turn_entries[-1]
        return f"{e.tool_name}({e.args_summary}) → {e.outcome}"
    latest = ctx._latest_evidence_record()
    if latest is not None:
        return f"{latest.action} → {latest.verdict}"
    return ""
