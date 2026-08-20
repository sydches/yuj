"""Adaptive-control debug records.

One JSONL row per pause explaining the decision path, so a reviewer can tell why
a pause did or did not diagnose/apply without reading harness code. Off by
default; obeys the same no-peeking rule as live control (future_evidence_used is
always false). Stdlib-only so the loop can import it with no new deps.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

EVENT_TYPE = "adaptive_pause_debug"
SCHEMA_VERSION = "adaptive_debug_v0"
DEBUG_MODES = {"none", "summary", "verbose"}
DEFAULT_DEBUG_FILENAME = "adaptive_debug.jsonl"

# Fixed blocked vocabulary.
BLOCKED_REASONS = {
    "controller_stub", "missing_slot_stream", "missing_instance_id",
    "missing_attempt_id", "detector_not_wired", "detector_input_missing",
    "detector_no_fire", "detector_false_positive_gate_failed", "no_active_hurdle",
    "medicine_not_live_eligible", "missing_executor",
    "executor_planned_not_implemented", "executor_not_implemented",
    "executor_blocked", "apply_failed",
    # Lookup selection.
    "missing_lookup_row", "lookup_not_ready",
    # Budget and repeat guard.
    "intervention_budget_exhausted", "same_signal_budget_exhausted",
    "repeat_intervention_disallowed", "intervention_cooldown_active",
    "watch_pending",
    # Apply path.
    "missing_config_overlay", "missing_baseline_config",
    "config_overlay_invalid", "config_refresh_not_declared",
    "runtime_surface_refresh_failed", "stacking_forbidden",
}

DETECTOR_STATUSES = {"not_run", "blocked", "no_fire", "active_suspected", "active_confirmed"}
SELECTION_STATUSES = {"not_attempted", "blocked", "selected"}
APPLY_STATUSES = {"not_attempted", "applied", "blocked", "failed"}

# Ordered schema (spec "Summary Row Schema"). Booleans are real JSON booleans.
_DEFAULTS: dict = {
    # identity
    "event_type": EVENT_TYPE, "schema_version": SCHEMA_VERSION,
    "attempt_id": "", "instance_id": "", "cell_id": "", "wave_id": "",
    "observation_slot": 0, "boundary_type": "", "controller_version": "",
    "policy_version": "", "detector_version": "", "intervention_space_version": "",
    "evidence_regime": "",
    # target
    "source_hindsight_hurdle_mode": "", "online_signal_id": "",
    "intervention_target": "", "candidate_medicine_knob": "",
    "candidate_config_path": "", "source_static_cell_id": "",
    # prefix availability
    "prefix_snapshot_available": False, "recent_slot_count": 0,
    "recent_slot_tags": [], "mutation_state_available": False,
    "verification_state_available": False, "tool_error_state_available": False,
    "repeat_state_available": False, "prefix_blocked_reason": "",
    # detector path
    "detector_id": "", "detector_status": "not_run", "detector_blocked_reason": "",
    "detector_basis_refs": "", "future_evidence_used": False,
    # branch-bundle capture path
    "branch_point_id": "", "branch_bundle_path": "", "branch_bundle_status": "",
    "branch_bundle_reason": "",
    # medicine selection path
    "selection_status": "not_attempted", "selection_blocked_reason": "",
    "selected_intervention_id": "", "medicine_live_eligible": "unknown",
    "runtime_executor_id": "", "executor_status": "",
    # executor path
    "apply_attempted": False, "apply_status": "not_attempted",
    "executor_blocked_reason": "", "pre_control_digest": "", "post_control_digest": "",
    "baseline_config_paths": [], "active_config_basis": "",
    "applied_config_paths": [], "config_overlay_apply_status": "not_attempted",
    "config_overlay_blocked_reason": "", "refreshed_surfaces": [],
    "changed_config_fields": [], "blocked_config_fields": [],
    # final decision mirror
    "diagnosis_status": "", "active_hurdle_mode": "", "intervention_id": "",
}

DEBUG_ROW_FIELDS = list(_DEFAULTS.keys())


def build_debug_row(**overrides) -> dict:
    """One debug row with every field defaulted, then overrides applied. Unknown
    keys raise so the schema stays closed."""
    bad = set(overrides) - set(_DEFAULTS)
    if bad:
        raise KeyError(f"unknown debug fields: {sorted(bad)}")
    row = dict(_DEFAULTS)
    row.update(overrides)
    return row


def enabled(debug_mode: str) -> bool:
    return debug_mode in ("summary", "verbose")


def derive_debug_path(debug_ledger_path: str, ledger_path: str) -> str:
    """Explicit debug path wins; else sibling of the control ledger; else none."""
    if debug_ledger_path:
        return debug_ledger_path
    if ledger_path:
        return str(Path(ledger_path).parent / DEFAULT_DEBUG_FILENAME)
    return ""


def write_debug_row(path: str, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def emit(cfg, ledger_path: str, row: dict) -> None:
    """Write one debug row if debug is enabled. Never raises into the caller; a
    debug-writing failure logs a stderr warning but must not break the run."""
    mode = getattr(cfg, "adaptive_control_debug", "none")
    if not enabled(mode):
        return
    path = derive_debug_path(
        getattr(cfg, "adaptive_control_debug_ledger_path", ""), ledger_path)
    if not path:
        return
    try:
        if not getattr(cfg, "adaptive_control_debug_include_prefix", False) or mode != "verbose":
            row = {**row, "recent_slot_tags": []}
        write_debug_row(path, row)
    except Exception as exc:  # noqa: BLE001 - debug writing must not break a run
        print(f"adaptive_control debug write failed: {exc}", file=sys.stderr)
