"""Recovery + repeat-detection helpers for WorkingSetBaselineContext."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._working_set_baseline_helpers import (
    _extract_action_target, _extract_test_target_from_action,
    _is_inspection_action,
)

if TYPE_CHECKING:
    from ._working_set_baseline import TraceRecord, WorkingSetBaselineContext


def repeated_verify_run(
    ctx: "WorkingSetBaselineContext",
) -> tuple["TraceRecord", int, int] | None:
    threshold = int(ctx._recovery_verify_repeat_threshold or 0)
    if threshold <= 1:
        return None
    records = ctx._trace_records()
    i = 0
    latest = None
    while i < len(records):
        rec = records[i]
        j = i + 1
        while (
            j < len(records)
            and records[j].action == rec.action
            and records[j].outcome == rec.outcome
        ):
            j += 1
        run_len = j - i
        if run_len >= threshold and _extract_test_target_from_action(rec.action):
            latest = (rec, run_len, records[j - 1].turn)
        i = j
    return latest


def recovery_state(ctx: "WorkingSetBaselineContext") -> tuple[str, str] | None:
    verify_repeat = repeated_verify_run(ctx)
    if verify_repeat is not None:
        target = _extract_test_target_from_action(verify_repeat[0].action)
        return ("repeated verification without refinement", target)
    repeated = latest_repeated_trace_run(ctx)
    if repeated is None:
        return None
    if ctx._recovery_same_target_threshold > 0 and repeated[1] >= ctx._recovery_same_target_threshold:
        target = repeated_target_text(ctx, repeated[0].action)
        if target.startswith("/") or " under /" in target:
            return ("repeated inspection outside repo root", target)
        return ("repeated same-target inspection", target)
    return None


def latest_repeated_trace_run(
    ctx: "WorkingSetBaselineContext",
) -> tuple["TraceRecord", int, int] | None:
    threshold = max(
        int(ctx._inspect_repeat_threshold or 0),
        int(ctx._recovery_same_target_threshold or 0),
    )
    if threshold <= 1:
        return None
    records = ctx._trace_records()
    i = 0
    latest = None
    while i < len(records):
        rec = records[i]
        j = i + 1
        while (
            j < len(records)
            and records[j].action == rec.action
            and records[j].outcome == rec.outcome
        ):
            j += 1
        run_len = j - i
        if run_len >= threshold and _is_inspection_action(rec.action):
            latest = (rec, run_len, records[j - 1].turn)
        i = j
    return latest


def repeated_target_text(ctx: "WorkingSetBaselineContext", action: str) -> str:
    target = _extract_action_target(action)
    return target if target else action


def disallowed_repeat_text(ctx: "WorkingSetBaselineContext") -> str:
    recovery = recovery_state(ctx)
    if recovery is not None:
        reason, target = recovery
        return f"{reason}: {target}" if target else reason
    repeated = latest_repeated_trace_run(ctx)
    if repeated is None:
        return ""
    return repeated_target_text(ctx, repeated[0].action)


def last_verdict_text(ctx: "WorkingSetBaselineContext") -> str:
    rec = ctx._blocking_record() or ctx._latest_evidence_record()
    if rec is None:
        return ""
    return ctx._summary_line(rec)


def slot_next_action_text(ctx: "WorkingSetBaselineContext") -> str:
    candidate_source = ctx._format_path_list(
        ctx._candidate_source_paths(),
        limit=ctx._slot_max_candidates,
    )
    candidate_test = ctx._format_path_list(
        ctx._candidate_test_paths(),
        limit=ctx._slot_max_candidates,
    )
    changed = ctx._format_path_list(ctx._changed_paths(), limit=1)
    if changed:
        return f"run verification on {changed} or refine it"
    if candidate_test and ctx._needs_test_read():
        if candidate_source:
            return f"read {candidate_test} or change {candidate_source}"
        return f"read {candidate_test} or run verification"
    if candidate_source:
        if candidate_test:
            return f"change {candidate_source}, read {candidate_test}, or run verification"
        return f"change {candidate_source} or run verification"
    if candidate_test:
        return f"read {candidate_test} or run verification"
    return "read one concrete file, mutate a file, or run verification"
