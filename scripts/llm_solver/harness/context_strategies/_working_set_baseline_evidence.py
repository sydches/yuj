"""Trace + Evidence record-building and rendering for WorkingSetBaselineContext."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..._shared.classification import classify_outcome as _classify_outcome
from ._working_set_baseline_helpers import (
    _clean_reasoning, _fit_blocks, _fit_lines, _truncate_text,
)

if TYPE_CHECKING:
    from ._working_set_baseline import (
        EvidenceRecord, TraceRecord, WorkingSetBaselineContext,
    )


def trace_records(ctx: "WorkingSetBaselineContext") -> list["TraceRecord"]:
    from ._working_set_baseline import TraceRecord

    if ctx._has_solver_state():
        records: list[TraceRecord] = []
        for entry in ctx._load_state_json().get("trace", []):
            if not isinstance(entry, dict):
                continue
            turn = entry.get("turn")
            try:
                turn_num = int(turn)
            except (TypeError, ValueError):
                turn_num = ctx._turn_count
            result = str(entry.get("result", ""))
            gate_blocked = bool(entry.get("gate_blocked"))
            outcome = "BLOCKED" if gate_blocked else _classify_outcome(result)
            records.append(TraceRecord(
                turn=turn_num,
                reasoning=str(entry.get("reasoning", "") or ""),
                action=str(entry.get("action", "") or "?"),
                outcome=outcome,
            ))
        return records

    return [
        TraceRecord(
            turn=e.turn,
            reasoning=e.reasoning,
            action=f"{e.tool_name}({e.args_summary})",
            outcome=e.outcome,
        )
        for e in ctx._turn_entries
    ]


def evidence_records(ctx: "WorkingSetBaselineContext") -> list["EvidenceRecord"]:
    from ._working_set_baseline import EvidenceRecord

    if ctx._has_solver_state():
        latest_by_action: dict[str, EvidenceRecord] = {}
        for item in ctx._load_state_json().get("evidence", []):
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", "") or "?")
            step = item.get("step")
            try:
                turn = int(step)
            except (TypeError, ValueError):
                turn = ctx._turn_count
            latest_by_action[action] = EvidenceRecord(
                turn=turn,
                action=action,
                verdict=str(item.get("verdict", "") or "OK"),
                content=str(item.get("result", "") or ""),
                first_turn=turn,
                repeat_count=0,
            )
        return sorted(latest_by_action.values(), key=lambda rec: rec.turn)

    records: list[EvidenceRecord] = []
    for gate in ctx._ws.gate_latest.values():
        records.append(EvidenceRecord(
            turn=gate.turn,
            action=gate.cmd_display,
            verdict=gate.verdict,
            content=gate.content,
            first_turn=gate.first_turn,
            repeat_count=gate.repeat_count,
        ))
    records.sort(key=lambda rec: rec.turn)
    return records


def latest_evidence_record(ctx: "WorkingSetBaselineContext") -> "EvidenceRecord | None":
    records = evidence_records(ctx)
    if not records:
        return None
    return max(records, key=lambda rec: rec.turn)


def blocking_record(ctx: "WorkingSetBaselineContext") -> "EvidenceRecord | None":
    records = evidence_records(ctx)
    fails = [rec for rec in records if rec.verdict.startswith("FAIL")]
    if fails:
        return max(fails, key=lambda rec: rec.turn)
    if records:
        return max(records, key=lambda rec: rec.turn)
    return None


def gate_payload_text(ctx: "WorkingSetBaselineContext", max_chars: int) -> str:
    rec = blocking_record(ctx)
    if rec is None:
        return ""
    header = f"{rec.action} → {rec.verdict} (T{rec.turn})"
    if max_chars <= len(header):
        return header
    body_budget = max_chars - len(header) - 1
    body = _truncate_text(rec.content, body_budget)
    return f"{header}\n{body}" if body else header


def summary_line(rec: "EvidenceRecord") -> str:
    if rec.repeat_count > 0 and rec.first_turn is not None:
        suffix = " (unchanged)" if rec.repeat_count >= 2 else ""
        return (
            f"T{rec.first_turn}-{rec.turn}: {rec.action} → {rec.verdict} "
            f"×{rec.repeat_count + 1}{suffix}"
        )
    return f"T{rec.turn}: {rec.action} → {rec.verdict}"


def checks_text(ctx: "WorkingSetBaselineContext", max_chars: int) -> str:
    records = evidence_records(ctx)
    if not records:
        return ""
    fails = [summary_line(rec) for rec in records if rec.verdict.startswith("FAIL")]
    passes = [summary_line(rec) for rec in records if not rec.verdict.startswith("FAIL")]
    lines = fails + passes
    return _fit_lines(lines, max_chars, max_lines=ctx._evidence_lines)


def evidence_text(ctx: "WorkingSetBaselineContext", max_chars: int) -> str:
    records = evidence_records(ctx)
    if not records:
        return ""
    fails = [summary_line(rec) for rec in records if rec.verdict.startswith("FAIL")]
    passes = [summary_line(rec) for rec in records if not rec.verdict.startswith("FAIL")]
    lines: list[str] = []
    if fails:
        lines.append("-- unresolved --")
        lines.extend(fails[-ctx._evidence_lines:] if ctx._evidence_lines else fails)
    if passes:
        lines.append("-- resolved --")
        lines.extend(passes[-ctx._evidence_lines:] if ctx._evidence_lines else passes)
    return _fit_lines(lines, max_chars)


def trace_text(ctx: "WorkingSetBaselineContext", max_chars: int) -> str:
    records = trace_records(ctx)
    if not records:
        return ""

    blocks: list[str] = []
    prev_reasoning: str | None = None
    i = 0
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
        if ctx._style == "yuj":
            blocks.append(
                trace_block_yuj(
                    ctx, rec,
                    prev_reasoning=prev_reasoning,
                    run_len=run_len,
                    last_turn=records[j - 1].turn,
                )
            )
            reasoning = _clean_reasoning(rec.reasoning, ctx._trace_reasoning_chars)
            if reasoning:
                prev_reasoning = reasoning
        else:
            blocks.append(
                trace_block_generic(
                    ctx, rec,
                    run_len=run_len,
                    last_turn=records[j - 1].turn,
                )
            )
        i = j

    return _fit_blocks(blocks, max_chars, max_entries=ctx._trace_lines)


def trace_block_generic(
    ctx: "WorkingSetBaselineContext",
    rec: "TraceRecord",
    *,
    run_len: int,
    last_turn: int,
) -> str:
    reasoning = _clean_reasoning(rec.reasoning, ctx._trace_reasoning_chars)
    if run_len > 1:
        return f"- T{rec.turn}-{last_turn}: {rec.action} ×{run_len} → {rec.outcome}"
    if reasoning:
        return f"- T{rec.turn}: \"{reasoning}\" → {rec.action} {rec.outcome}"
    return f"- T{rec.turn}: {rec.action} {rec.outcome}"


def trace_block_yuj(
    ctx: "WorkingSetBaselineContext",
    rec: "TraceRecord",
    *,
    prev_reasoning: str | None,
    run_len: int,
    last_turn: int,
) -> str:
    lines: list[str] = []
    reasoning = _clean_reasoning(rec.reasoning, ctx._trace_reasoning_chars)
    if reasoning and reasoning != prev_reasoning:
        lines.append(f"T{rec.turn} [{reasoning}]")
    if run_len > 1:
        lines.append(
            f"    → {rec.action} ×{run_len} (T{rec.turn}-{last_turn}) → {rec.outcome}"
        )
    else:
        lines.append(f"    → {rec.action} → {rec.outcome}")
    return "\n".join(lines)
