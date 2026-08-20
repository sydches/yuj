"""Mechanical live detector that runs soft-tier trace nets in process.

Replaces the LLM detector's model call with field-equality nets computed
causally over ``session._trace_events``: recurring failing or passing output
hashes, identical-repeat plateaus, rereads after a gap, snippet
recurrence. Emits an ``LLMDetectorVerdict`` so everything downstream
(family lookup, executor apply, watch, restore, escalation) is reused
unchanged.

No text parsing: output_sha256 / output_snippet / args_summary equality
and pass_fail flags only. Warmup honored via
``guardrails_arm_after_turn`` (the same key the pre-dispatch guardrails
use): no fire at or before that turn.

Family vocabulary (must match the family lookup TSV rows):
  repeat_wall   — identical-repeat plateau or recurring failing hash
  reread_slump  — rereads / recurring passing hash / snippet recurrence
"""
from __future__ import annotations

from typing import Any

from ...trace_net_facts import (
    args_reread_after_gap,
    identical_repeat_plateau_start,
    same_failed_output_repeat,
    same_passing_output_recurrence,
)
from .llm_detector_core import LLMDetectorVerdict

# Default thresholds for the four facts. A runtime overlay may change them
# through [llm_hurdle_detector.trace_nets]. A non-positive value uses the
# default. The detector sets its window from the maximum reread gap.
_FAIL_MIN_STREAK = 4
_PASS_LOOKBACK = 20
_PASS_MIN_PRIOR = 2
_PASS_MIN_GAP = 2
_REREAD_MIN_ARGS_LEN = 20
_REREAD_MIN_GAP = 3
_REREAD_MAX_GAP = 30


def _p(cfg: Any, name: str, default: int) -> int:
    """Read a trace_nets threshold; use the default if it is non-positive."""
    v = getattr(cfg, f"trace_nets_{name}", None)
    try:
        v = int(v)
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


def _events_tail(session: Any, turn: int, window: int) -> list[dict]:
    events = []
    for event in getattr(session, "_trace_events", []) or []:
        if event.get("event") != "tool_call":
            continue
        try:
            event_turn = int(event.get("turn_number"))
        except (TypeError, ValueError):
            continue
        if event_turn <= int(turn):
            events.append(event)
    return events[-window:]


def evaluate_trace_nets(session: Any, turn: int) -> LLMDetectorVerdict:
    """Causal net evaluation at ``turn``; returns a closed verdict."""
    cfg = getattr(session, "cfg", None)
    arm_after = int(getattr(cfg, "guardrails_arm_after_turn", 0) or 0)
    # stop_resume delivery: a resumed segment inside an ACTIVE episode must
    # watch its first 5 turns (that IS the watch window — the knob rode in
    # at the resume). Warmup would blind exactly those turns, so an active
    # restored episode bypasses it. Fresh segments keep the quiet start.
    _active_episode = (
        getattr(cfg, "adaptive_control_delivery", "in_place") == "stop_resume"
        and getattr(session, "_llm_detector_pending_watch", None) is not None
    )
    if turn <= arm_after and not _active_episode:
        return _no_fire("warmup")
    p_fail_min_streak = _p(cfg, "fail_min_streak", _FAIL_MIN_STREAK)
    p_pass_lookback = _p(cfg, "pass_lookback", _PASS_LOOKBACK)
    p_pass_min_prior = _p(cfg, "pass_min_prior", _PASS_MIN_PRIOR)
    p_pass_min_gap = _p(cfg, "pass_min_gap", _PASS_MIN_GAP)
    p_reread_min_args_len = _p(cfg, "reread_min_args_len", _REREAD_MIN_ARGS_LEN)
    p_reread_min_gap = _p(cfg, "reread_min_gap", _REREAD_MIN_GAP)
    p_reread_max_gap = _p(cfg, "reread_max_gap", _REREAD_MAX_GAP)
    window = p_reread_max_gap + 1
    tail = _events_tail(session, turn, window)
    if len(tail) < 4:
        return _no_fire("too_few_turns")

    cur = tail[-1]
    idx = len(tail) - 1
    fires: list[tuple[str, str, str]] = []  # (family, net, evidence_ref)

    exact = identical_repeat_plateau_start(tail, idx)
    if exact is not None:
        prior_turn, current_turn = exact.evidence_turns
        fires.append((
            "repeat_wall",
            "identical_repeat_plateau_start",
            f"T{current_turn}:args+execution-sha == T{prior_turn}",
        ))

    failed = same_failed_output_repeat(
        tail,
        idx,
        min_streak=p_fail_min_streak,
    )
    if failed is not None:
        fires.append((
            "repeat_wall",
            "same_failed_output_repeat",
            f"T{cur.get('turn_number')}:same failing execution-sha repeated "
            f"{failed.occurrences} consecutive turns",
        ))

    passing = same_passing_output_recurrence(
        tail,
        idx,
        lookback=p_pass_lookback,
        min_prior=p_pass_min_prior,
        min_gap=p_pass_min_gap,
    )
    if passing is not None:
        fires.append((
            "reread_slump",
            "same_passing_output_recurrence",
            f"T{cur.get('turn_number')}:passing execution-sha recurred after "
            f"gap={passing.gap}",
        ))

    reread = args_reread_after_gap(
        tail,
        idx,
        min_args_len=p_reread_min_args_len,
        min_gap=p_reread_min_gap,
        max_gap=p_reread_max_gap,
    )
    if reread is not None:
        fires.append((
            "reread_slump",
            "args_reread_after_gap",
            f"T{cur.get('turn_number')}:args re-issued after gap={reread.gap} "
            "with no source write",
        ))

    if not fires:
        return _no_fire("")
    # walls outrank slumps (point evidence beats prefix, as in the routers)
    fires.sort(key=lambda f: 0 if f[0] == "repeat_wall" else 1)
    family, net, ref = fires[0]
    return LLMDetectorVerdict(
        hurdle_present="yes",
        hurdle_family=family,
        confidence="high" if len(fires) > 1 else "medium",
        evidence_refs=[ref] + [r for _, _, r in fires[1:3]],
        decision_summary=f"trace nets fired: {', '.join(n for _, n, _ in fires[:3])}",
        why_now=ref,
        new_facts_still_appearing=False,
        timing_basis="causal trace fields at the current turn",
        uncertainty="mechanical nets; no semantic reading",
    )


def _no_fire(reason: str) -> LLMDetectorVerdict:
    return LLMDetectorVerdict(
        hurdle_present="no",
        hurdle_family="",
        confidence="high",
        evidence_refs=[],
        abstain_reason=reason,
        decision_summary="no net fired" if not reason else f"no fire ({reason})",
        timing_basis="causal trace fields at the current turn",
    )
