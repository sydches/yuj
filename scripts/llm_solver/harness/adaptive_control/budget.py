"""Intervention budget and repeat guard.

Prevents the controller from looping on itself: caps total interventions, caps
per-signal interventions, forbids repeating the same intervention, and blocks a
new intervention while a watch is pending. Pure state + checks; the session owns
a BudgetState and calls check() before apply and record() after a real apply
(wired into the mutating apply path).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Budget block reasons.
WATCH_PENDING = "watch_pending"
TOTAL_EXHAUSTED = "intervention_budget_exhausted"
SAME_SIGNAL_EXHAUSTED = "same_signal_budget_exhausted"
REPEAT_DISALLOWED = "repeat_intervention_disallowed"
COOLDOWN_ACTIVE = "intervention_cooldown_active"


@dataclass
class BudgetState:
    interventions_applied_total: int = 0
    interventions_by_signal: dict = field(default_factory=dict)
    intervention_ids_used: set = field(default_factory=set)
    active_watch: bool = False
    last_intervention_slot: int | None = None


def check(state: BudgetState, signal_id: str, intervention_id: str, *,
          max_interventions: int, max_same_signal: int,
          disallow_repeat: bool, current_slot: int | None = None,
          cooldown_after_apply_slots: int = 0) -> tuple[bool, str]:
    """Return (allowed, blocked_reason). Order: pending watch, total cap,
    cooldown, total cap, per-signal cap, repeat."""
    if state.active_watch:
        return False, WATCH_PENDING
    if (
        state.last_intervention_slot is not None
        and current_slot is not None
        and cooldown_after_apply_slots > 0
        and current_slot - state.last_intervention_slot < cooldown_after_apply_slots
    ):
        return False, COOLDOWN_ACTIVE
    if state.interventions_applied_total >= max_interventions:
        return False, TOTAL_EXHAUSTED
    if state.interventions_by_signal.get(signal_id, 0) >= max_same_signal:
        return False, SAME_SIGNAL_EXHAUSTED
    if disallow_repeat and intervention_id in state.intervention_ids_used:
        return False, REPEAT_DISALLOWED
    return True, ""


def record(state: BudgetState, signal_id: str, intervention_id: str,
           *, current_slot: int | None = None) -> None:
    state.interventions_applied_total += 1
    state.interventions_by_signal[signal_id] = state.interventions_by_signal.get(signal_id, 0) + 1
    state.intervention_ids_used.add(intervention_id)
    if current_slot is not None:
        state.last_intervention_slot = current_slot
