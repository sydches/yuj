"""Typed control schemas.

Stdlib-only dataclasses so the harness loop can import this with no new deps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ControllerRequest:
    """Everything the controller may see at a pause boundary. Current-run prefix
    only — no future slots, terminal suffix, or scorer outcome."""
    attempt_id: str
    instance_id: str
    boundary_type: str
    observation_slot: int
    evidence_regime: str  # retrospective_atlas | prefix_replay | baseline_informed_live | causal_live
    online_signal_id: str = ""  # the live signal to watch for, if configured
    detector_mode: str = ""  # empty => controller/config default; otherwise manual | adaptive
    recent_slot_tags: list[str] = field(default_factory=list)
    source_mutation_state: bool = False
    verification_state: bool = False
    tool_error_state: bool = False
    repeat_state: bool = False
    active_config_digest: str = ""
    prefix_evidence_refs: list[str] = field(default_factory=list)


@dataclass
class ControllerDecision:
    diagnosis_status: str  # active_confirmed | active_suspected | no_active_hurdle | blocked_unobservable
    active_hurdle_mode: str = ""
    basis_refs: list[str] = field(default_factory=list)
    selected_intervention_id: str = ""
    payload: "InterventionPayload | None" = None
    future_evidence_used: bool = False  # must stay False for live decisions
    # Debug fields for the detector path.
    detector_id: str = ""
    detector_status: str = "not_run"  # not_run | blocked | no_fire | active_suspected | active_confirmed
    detector_blocked_reason: str = ""


@dataclass
class InterventionPayload:
    intervention_id: str
    executor_id: str
    timing_class: str
    fields: dict[str, Any] = field(default_factory=dict)
    candidate_config_path: str = ""
    baseline_config_paths: tuple[str, ...] = ()


@dataclass
class ExecutorResult:
    executor_id: str
    applied: bool
    pre_digest: str = ""
    post_digest: str = ""
    blocked_reason: str = ""
    baseline_config_paths: tuple[str, ...] = ()
    candidate_config_path: str = ""
    applied_config_paths: tuple[str, ...] = ()
    active_config_basis: str = ""
    refreshed_surfaces: tuple[str, ...] = ()
    changed_config_fields: tuple[str, ...] = ()
    blocked_config_fields: tuple[str, ...] = ()

    @property
    def apply_status(self) -> str:
        if self.applied:
            return "applied"
        if self.blocked_reason:
            return "blocked"
        return "not_attempted"


@dataclass
class ControlLedgerRow:
    """One record for each decision at an adaptive-control pause point."""
    attempt_id: str
    instance_id: str
    controller_version: str
    control_model: str  # in_process | segmented_resume
    observation_slot: int
    active_hurdle_mode: str
    diagnosis_status: str
    intervention_id: str = ""  # the SELECTED medicine (may be set even when apply blocked)
    intervention_basis: str = ""
    apply_status: str = "not_attempted"  # not_attempted | applied | blocked | failed
    intervention_slot: str = ""  # set only when apply_status == applied
    # Inclusive watch window: first included slot through last included slot.
    watch_window_start: str = ""
    watch_window_end: str = ""
    immediate_effect: str = ""  # cleared | unchanged | worsened | unknown
    next_hurdle_mode: str = ""
    final_outcome: str = ""
    evidence_refs: str = ""
    future_evidence_used: str = "false"
    branch_point_id: str = ""
    branch_bundle_path: str = ""
    branch_bundle_status: str = ""
    branch_bundle_reason: str = ""
    baseline_config_paths: str = ""
    candidate_config_path: str = ""
    active_config_basis: str = ""
    pre_intervention_config_digest: str = ""
    post_intervention_config_digest: str = ""
    refreshed_surfaces: str = ""
    changed_config_fields: str = ""
    blocked_config_fields: str = ""
    # Multi-intervention episode fields. They stay empty for a single apply.
    controller_state: str = ""  # MONITORING | ACTIVE_HURDLE | SELECT_MEDICINE | APPLYING | WATCHING | EPISODE_EXHAUSTED | STOPPED
    hurdle_episode_id: str = ""
    episode_online_signal_id: str = ""
    episode_attempt_index: str = ""  # 1-based intervention count within the episode
    candidate_rank_at_selection: str = ""
    candidate_exclusion_refs: str = ""  # why higher-ranked candidates were skipped
    previous_intervention_id: str = ""  # prior medicine in the same episode, if any
    same_hurdle_escalation: str = ""  # "true" when this apply follows an unchanged same-signal result
    episode_transition: str = ""  # cleared_to_progress | cleared_to_later_hurdle | unchanged | worsened | unknown | candidate_exhausted
    next_episode_id: str = ""  # later episode id when a different hurdle is detected
