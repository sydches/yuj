"""Intervention application and watch-window handling for LLM detector verdicts."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from .llm_detector_core import LLMDetectorVerdict

def _maybe_apply_detector_intervention(
    session: Any,
    turn: int,
    verdict: LLMDetectorVerdict,
    row: dict[str, Any],
) -> None:
    """Route a positive detector verdict through the ranked Atlas ladder."""
    cfg = getattr(session, "cfg", None)
    if (
        bool(getattr(cfg, "transformations_explicit", False))
        and not bool(getattr(cfg, "detector_activated_guardrails", True))
    ):
        if verdict.hurdle_present == "yes":
            row["intervention_selection"] = {
                "selection_status": "not_attempted",
                "selection_blocked_reason": (
                    "detector_activated_guardrails_disabled"
                ),
            }
        return
    pending = _pending_watch(session)
    if pending:
        try:
            _handle_pending_watch_verdict(session, turn, verdict, row, pending)
        except Exception as exc:  # noqa: BLE001 - detector apply path must fail open
            row["intervention_error"] = f"{type(exc).__name__}: {exc}"
        return

    if verdict.hurdle_present != "yes":
        return

    if not bool(getattr(cfg, "adaptive_control_enabled", False)):
        row["intervention_selection"] = {
            "selection_status": "not_attempted",
            "selection_blocked_reason": "adaptive_control_disabled",
        }
        return

    from . import episode

    machine = episode.machine(session)
    if machine.state == episode.EPISODE_EXHAUSTED:
        progress_refs = _material_progress_after_exhaustion(session, machine, turn)
        if not progress_refs:
            row["intervention_selection"] = {
                "selection_status": "not_attempted",
                "selection_blocked_reason": episode.CANDIDATE_EXHAUSTED,
            }
            return
        exhausted_slot = machine.exhausted_slot
        episode.resume_after_progress(machine)
        row["episode_resume_after_progress"] = {
            "prior_exhausted_slot": exhausted_slot,
            "material_progress_refs": progress_refs,
        }

    try:
        _select_and_apply_ranked_ladder(session, turn, verdict, row)
    except Exception as exc:  # noqa: BLE001 - detector apply path must fail open
        row["intervention_error"] = f"{type(exc).__name__}: {exc}"


def _handle_pending_watch_verdict(
    session: Any,
    turn: int,
    verdict: LLMDetectorVerdict,
    row: dict[str, Any],
    pending: dict[str, Any],
) -> None:
    """Observe or close the active LLM-detector watch budget."""
    from . import CONTROLLER_VERSION
    from . import episode, watch

    watched_family = str(pending.get("detector_family", "") or "")
    machine = episode.machine(session)
    transition = episode.UNKNOWN
    next_family = ""
    watch_end = int(pending.get("watch_window_end", turn) or turn)
    budget_exhausted = int(turn) >= watch_end
    post_slots = _post_intervention_slots(session, pending, turn)
    progress_slots = [slot for slot in post_slots if watch.material_progress(slot)]
    progress_refs = [str(slot.get("evidence_refs") or "") for slot in progress_slots]
    if verdict.hurdle_present == "yes" and verdict.hurdle_family == watched_family:
        transition = episode.UNCHANGED
    elif verdict.hurdle_present == "yes" and progress_slots:
        transition = episode.CLEARED_TO_LATER_HURDLE
        next_family = verdict.hurdle_family
    elif verdict.hurdle_present == "yes":
        # A family-name change without task progress is another symptom of the
        # same unresolved episode, not evidence that a later hurdle opened.
        transition = episode.UNCHANGED
        next_family = verdict.hurdle_family
    elif verdict.hurdle_present == "no" and progress_slots:
        transition = episode.CLEARED_TO_PROGRESS
    elif verdict.hurdle_present == "no" and post_slots:
        transition = episode.DISLODGED_NO_PROGRESS

    row["watch_transition"] = {
        "detector_family": watched_family,
        "verdict_family": verdict.hurdle_family,
        "episode_transition": transition,
        "next_hurdle_mode": next_family,
        "watch_window_start": pending.get("watch_window_start", ""),
        "watch_window_end": pending.get("watch_window_end", ""),
        "post_slot_count": len(post_slots),
        "material_progress": bool(progress_slots),
        "material_progress_refs": progress_refs,
        "budget_exhausted": budget_exhausted,
    }

    cleared = transition in {
        episode.CLEARED_TO_PROGRESS,
        episode.CLEARED_TO_LATER_HURDLE,
    }
    if not cleared and not budget_exhausted:
        row["watch_transition"]["watch_status"] = "continuing"
        row["watch_transition"]["episode_transition"] = ""
        return

    # Failing to prove an unlock within the fixed budget advances the ladder.
    # An uncertain detector result must not extend one candidate indefinitely.
    if transition == episode.UNKNOWN:
        transition = episode.UNCHANGED
        row["watch_transition"]["episode_transition"] = transition
    row["watch_transition"]["watch_status"] = "closed"

    closed_episode = episode.close_watch(machine, transition)
    _clear_pending_watch(session)

    if transition == episode.DISLODGED_NO_PROGRESS:
        routing_verdict = dataclasses.replace(
            verdict,
            hurdle_family=watched_family,
        )
        _select_and_apply_ranked_ladder(
            session,
            turn,
            routing_verdict,
            row,
            episode_transition=transition,
            same_hurdle_escalation=True,
        )
        return

    if transition in {episode.CLEARED_TO_PROGRESS, episode.CLEARED_TO_LATER_HURDLE}:
        restore_result = _restore_baseline_for_watch_close(session, row, transition)
        _append_detector_control_ledger(
            session=session,
            turn=turn,
            verdict=verdict,
            controller_version=CONTROLLER_VERSION,
            chosen={"intervention_id": "toml_overlay.restore_baseline"},
            apply_status=restore_result.apply_status,
            blocked_reason=restore_result.blocked_reason,
            result=restore_result,
            episode_transition=transition,
            controller_state=machine.state,
            immediate_effect="cleared",
            active_hurdle_mode=watched_family,
            next_hurdle_mode=next_family,
            watch_window_start=pending.get("watch_window_start", ""),
            watch_window_end=pending.get("watch_window_end", ""),
            hurdle_episode_id=getattr(closed_episode, "episode_id", "") or pending.get("episode_id", ""),
            episode_online_signal_id=getattr(closed_episode, "online_signal_id", "") or pending.get("signal_id", ""),
            episode_attempt_index=str(getattr(closed_episode, "attempt_index", "") or pending.get("episode_attempt_index", "")),
        )
        if (
            restore_result.applied
            and transition == episode.CLEARED_TO_LATER_HURDLE
            and verdict.hurdle_present == "yes"
        ):
            _select_and_apply_ranked_ladder(
                session,
                turn,
                verdict,
                row,
                episode_transition=transition,
                next_episode_from=pending.get("episode_id", ""),
                baseline_already_restored=True,
            )
        return

    if transition == episode.UNCHANGED:
        routing_verdict = verdict
        if verdict.hurdle_family != watched_family:
            routing_verdict = dataclasses.replace(verdict, hurdle_family=watched_family)
        _select_and_apply_ranked_ladder(
            session,
            turn,
            routing_verdict,
            row,
            episode_transition=transition,
            same_hurdle_escalation=True,
        )


def _post_intervention_slots(
    session: Any,
    pending: dict[str, Any],
    turn: int,
) -> list[dict[str, Any]]:
    """Project only the causal post-intervention trace window."""
    from .slot_recorder import recent_prefix_slots_from_events

    start = int(pending.get("watch_window_start", turn) or turn)
    slots = recent_prefix_slots_from_events(
        list(getattr(session, "_trace_events", []) or []),
        int(turn),
    )
    return [slot for slot in slots if start <= int(slot.get("slot_idx", -1)) <= int(turn)]


def _material_progress_after_exhaustion(
    session: Any,
    machine: Any,
    turn: int,
) -> list[str]:
    from . import watch
    from .slot_recorder import recent_prefix_slots_from_events

    exhausted_slot = int(getattr(machine, "exhausted_slot", -1))
    if exhausted_slot < 0:
        return []
    slots = recent_prefix_slots_from_events(
        list(getattr(session, "_trace_events", []) or []),
        int(turn),
    )
    return [
        str(slot.get("evidence_refs") or f"turn={slot.get('slot_idx', '')}")
        for slot in slots
        if exhausted_slot < int(slot.get("slot_idx", -1)) <= int(turn)
        and watch.material_progress(slot)
    ]


def _select_and_apply_ranked_ladder(
    session: Any,
    turn: int,
    verdict: LLMDetectorVerdict,
    row: dict[str, Any],
    *,
    episode_transition: str = "",
    same_hurdle_escalation: bool = False,
    next_episode_from: str = "",
    baseline_already_restored: bool = False,
) -> None:
    cfg = getattr(session, "cfg", None)
    from . import CONTROLLER_VERSION
    from . import episode, executors, lookup_runtime
    from .schema import InterventionPayload

    lookup_path = str(getattr(cfg, "adaptive_control_lookup_table_path", "") or "")
    lookup_rows = lookup_runtime.load_lookup(lookup_path)
    machine = episode.machine(session)
    ladder_size = lookup_runtime.ranked_ladder_size(lookup_rows)
    caps = _episode_caps_for_detector(cfg, ladder_size=ladder_size)

    chosen_preview, preview_status, preview_reason = lookup_runtime.select_ranked_ladder(
        lookup_rows,
        exclude_ids=(),
    )
    signal_id = (
        (chosen_preview or {}).get("online_signal_id", "")
        or verdict.hurdle_family
    )
    current = machine.current
    current_exclude_ids = ()
    if (
        current is not None
        and current.status == "open"
        and current.online_signal_id == signal_id
    ):
        current_exclude_ids = tuple(current.applied_intervention_ids)
    plan = episode.plan_apply(machine, caps, signal_id, turn)
    exclude_ids = plan.exclude_ids or current_exclude_ids
    chosen, selection_status, selection_reason = lookup_runtime.select_ranked_ladder(
        lookup_rows,
        exclude_ids=exclude_ids,
    )
    selection = {
        "lookup_table_path": lookup_path,
        "detector_family": verdict.hurdle_family,
        "selection_policy": "atlas_ranked_ladder",
        "selection_status": selection_status,
        "selection_blocked_reason": selection_reason or plan.block_reason,
        "excluded_intervention_ids": ";".join(exclude_ids),
    }
    if chosen:
        signal_id = chosen.get("online_signal_id", "") or signal_id
        selection.update({
            "selected_intervention_id": chosen.get("intervention_id", ""),
            "rank_within_ladder": chosen.get("rank_within_ladder", ""),
            "rank_within_family": chosen.get("rank_within_family", ""),
            "rank_within_hurdle": chosen.get("rank_within_hurdle", ""),
            "online_signal_id": signal_id,
            "candidate_config_path": chosen.get("candidate_config_path", ""),
            "primary_knob": chosen.get("primary_knob", ""),
        })
    if preview_status != "selected" and selection_status != "selected":
        selection["selection_blocked_reason"] = preview_reason or selection_reason
    row["intervention_selection"] = selection

    if selection_reason == "candidate_exhausted":
        row["intervention_apply"] = {
            "apply_status": "blocked",
            "blocked_reason": selection_reason,
        }
        exhausted = episode.mark_candidate_exhausted(machine, turn)
        _append_detector_control_ledger(
            session=session,
            turn=turn,
            verdict=verdict,
            controller_version=CONTROLLER_VERSION,
            chosen=None,
            apply_status="blocked",
            blocked_reason=selection_reason,
            result=None,
            episode_transition=episode.CANDIDATE_EXHAUSTED,
            controller_state=machine.state,
            candidate_exclusion_refs=";".join(exclude_ids),
            previous_intervention_id=plan.previous_intervention_id,
            same_hurdle_escalation="true" if plan.is_escalation or same_hurdle_escalation else "false",
            hurdle_episode_id=getattr(exhausted, "episode_id", "") if exhausted else "",
        )
        if not baseline_already_restored:
            restore_result = _restore_baseline_for_watch_close(session, row, selection_reason)
            row["baseline_restore_after_block"] = _result_dict(restore_result)
            _append_detector_control_ledger(
                session=session,
                turn=turn,
                verdict=verdict,
                controller_version=CONTROLLER_VERSION,
                chosen={"intervention_id": "toml_overlay.restore_baseline"},
                apply_status=restore_result.apply_status,
                blocked_reason=restore_result.blocked_reason,
                result=restore_result,
                episode_transition=episode.CANDIDATE_EXHAUSTED,
                controller_state=machine.state,
                immediate_effect=(
                    "baseline_restored_after_exhaustion"
                    if restore_result.applied
                    else "baseline_restore_failed_after_exhaustion"
                ),
                active_hurdle_mode=verdict.hurdle_family,
                hurdle_episode_id=getattr(exhausted, "episode_id", "") if exhausted else "",
                episode_online_signal_id=(
                    getattr(exhausted, "online_signal_id", "") if exhausted else ""
                ),
                episode_attempt_index=(
                    str(getattr(exhausted, "attempt_index", "")) if exhausted else ""
                ),
            )
        return

    if not plan.allowed:
        row["intervention_apply"] = {
            "apply_status": "blocked",
            "blocked_reason": plan.block_reason,
        }
        restore_result = None
        if not baseline_already_restored:
            restore_result = _restore_baseline_for_watch_close(session, row, plan.block_reason)
        _append_detector_control_ledger(
            session=session,
            turn=turn,
            verdict=verdict,
            controller_version=CONTROLLER_VERSION,
            chosen=chosen,
            apply_status="blocked",
            blocked_reason=plan.block_reason,
            result=None,
            episode_transition=episode_transition or plan.block_reason,
            controller_state=machine.state,
            candidate_exclusion_refs=";".join(exclude_ids),
            previous_intervention_id=plan.previous_intervention_id,
            same_hurdle_escalation="true" if plan.is_escalation or same_hurdle_escalation else "false",
        )
        if restore_result is not None:
            row["baseline_restore_after_block"] = _result_dict(restore_result)
        return

    if selection_status != "selected" or not chosen:
        blocked_reason = selection_reason or "missing_lookup_row"
        row["intervention_apply"] = {
            "apply_status": "blocked",
            "blocked_reason": blocked_reason,
        }
        restore_result = None
        if not baseline_already_restored:
            restore_result = _restore_baseline_for_watch_close(session, row, blocked_reason)
            row["baseline_restore_after_block"] = _result_dict(restore_result)
        _append_detector_control_ledger(
            session=session,
            turn=turn,
            verdict=verdict,
            controller_version=CONTROLLER_VERSION,
            chosen=None,
            apply_status="blocked",
            blocked_reason=blocked_reason,
            result=None,
            episode_transition=episode_transition,
            controller_state=machine.state,
            candidate_exclusion_refs=";".join(exclude_ids),
            previous_intervention_id=plan.previous_intervention_id,
            same_hurdle_escalation="true" if plan.is_escalation or same_hurdle_escalation else "false",
        )
        return

    intervention_id = chosen.get("intervention_id", "")
    payload = InterventionPayload(
        intervention_id=intervention_id,
        executor_id=chosen.get("runtime_executor_id", ""),
        timing_class=chosen.get("timing_class", ""),
        fields={"candidate_config_path": chosen.get("candidate_config_path", "")},
        candidate_config_path=chosen.get("candidate_config_path", ""),
    )
    # stop_resume hands off the decision through a graceful stop.
    # user_turn applies the overlay and adds a synthetic user message.
    # All delivery modes use the same episode and watch bookkeeping.
    _delivery = getattr(cfg, "adaptive_control_delivery", "in_place")
    if _delivery == "stop_resume":
        result = executors.stop_for_resume(
            session, payload,
            evidence=";".join(verdict.evidence_refs),
            rung=int(chosen.get("rank_within_ladder") or 0),
            hurdle_family=verdict.hurdle_family,
            episode_id=str(getattr(plan.episode, "episode_id", "") or ""),
            turn=turn,
        )
    elif _delivery == "user_turn":
        result = executors.user_turn_apply(
            session, payload,
            evidence=";".join(verdict.evidence_refs),
            rung=int(chosen.get("rank_within_ladder") or 0),
            hurdle_family=verdict.hurdle_family,
            turn=turn,
        )
    elif _delivery == "tool_result":
        result = executors.tool_result_apply(
            session, payload,
            evidence=";".join(verdict.evidence_refs),
            rung=int(chosen.get("rank_within_ladder") or 0),
            hurdle_family=verdict.hurdle_family,
            turn=turn,
        )
    elif _delivery == "user_turn_msg_only":
        result = executors.user_turn_msg_only_apply(
            session,
            evidence=";".join(verdict.evidence_refs),
            rung=int(chosen.get("rank_within_ladder") or 0),
            hurdle_family=verdict.hurdle_family,
            turn=turn,
        )
    else:
        result = executors.apply(session, payload)
    applied_episode = None
    if result.applied:
        attempt_id = str(getattr(session, "attempt_id", "") or "attempt")
        applied_episode = episode.record_apply(machine, caps, signal_id, intervention_id, turn, attempt_id)
        _set_pending_watch(
            session,
            turn,
            cfg,
            detector_family=verdict.hurdle_family,
            signal_id=signal_id,
            intervention_id=intervention_id,
            episode_id=applied_episode.episode_id,
            episode_attempt_index=applied_episode.attempt_index,
            previous_intervention_id=applied_episode.previous_intervention_id,
            same_hurdle_escalation=plan.is_escalation or same_hurdle_escalation,
        )
    result_dict = _result_dict(result)
    row["intervention_apply"] = result_dict
    pending = _pending_watch(session) or {}
    _append_detector_control_ledger(
        session=session,
        turn=turn,
        verdict=verdict,
        controller_version=CONTROLLER_VERSION,
        chosen=chosen,
        apply_status=result.apply_status,
        blocked_reason=result.blocked_reason,
        result=result,
        episode_transition=episode_transition,
        controller_state=machine.state,
        watch_window_start=pending.get("watch_window_start", ""),
        watch_window_end=pending.get("watch_window_end", ""),
        hurdle_episode_id=getattr(applied_episode, "episode_id", "") if applied_episode else "",
        episode_online_signal_id=signal_id,
        episode_attempt_index=str(getattr(applied_episode, "attempt_index", "") if applied_episode else ""),
        candidate_exclusion_refs=";".join(exclude_ids),
        previous_intervention_id=plan.previous_intervention_id,
        same_hurdle_escalation="true" if plan.is_escalation or same_hurdle_escalation else "false",
        next_episode_id="" if not next_episode_from else getattr(applied_episode, "episode_id", ""),
    )
    if not result.applied and not baseline_already_restored:
        restore_result = _restore_baseline_for_watch_close(
            session, row, "candidate_apply_failed",
        )
        row["baseline_restore_after_apply_failure"] = _result_dict(restore_result)
        active_episode = plan.episode
        _append_detector_control_ledger(
            session=session,
            turn=turn,
            verdict=verdict,
            controller_version=CONTROLLER_VERSION,
            chosen={"intervention_id": "toml_overlay.restore_baseline"},
            apply_status=restore_result.apply_status,
            blocked_reason=restore_result.blocked_reason,
            result=restore_result,
            episode_transition=episode_transition or result.blocked_reason,
            controller_state=machine.state,
            immediate_effect=(
                "baseline_restored_after_apply_failure"
                if restore_result.applied
                else "baseline_restore_failed_after_apply_failure"
            ),
            active_hurdle_mode=verdict.hurdle_family,
            hurdle_episode_id=getattr(active_episode, "episode_id", ""),
            episode_online_signal_id=signal_id,
            episode_attempt_index=str(
                getattr(active_episode, "attempt_index", "")
            ),
            previous_intervention_id=plan.previous_intervention_id,
            same_hurdle_escalation=(
                "true" if plan.is_escalation or same_hurdle_escalation else "false"
            ),
        )


def _episode_caps_for_detector(cfg: Any, *, ladder_size: int = 0):
    from . import episode

    caps = episode.caps_from_cfg(cfg)

    def _legacy(key: str, default: int) -> int:
        # same falsy-zero trap as episode.caps_from_cfg: `or default`
        # would eat an explicit 0
        value = getattr(cfg, key, None)
        if value is None or value == "":
            return default
        return int(value)

    legacy_attempt = _legacy("adaptive_control_max_interventions", 1)
    legacy_same = _legacy("adaptive_control_max_same_signal_interventions", 1)
    # Shadow mode (withheld branch): an explicit zero intervention
    # budget on BOTH the episode cap and the legacy cap means observe
    # only — no ladder floor, no applies. The detector keeps running
    # and recording; plan_apply refuses every apply.
    if caps.max_interventions_per_attempt == 0 and legacy_attempt == 0:
        return episode.Caps(
            max_interventions_per_attempt=0,
            max_interventions_per_hurdle_episode=0,
            max_distinct_hurdle_episodes_per_attempt=0,
            cooldown_after_apply_slots=caps.cooldown_after_apply_slots,
        )
    per_episode = max(
        caps.max_interventions_per_hurdle_episode,
        legacy_same,
        int(ladder_size),
    )
    distinct_episodes = max(
        caps.max_distinct_hurdle_episodes_per_attempt,
        legacy_attempt,
    )
    return episode.Caps(
        max_interventions_per_attempt=max(
            caps.max_interventions_per_attempt,
            legacy_attempt,
            per_episode * distinct_episodes,
        ),
        max_interventions_per_hurdle_episode=per_episode,
        max_distinct_hurdle_episodes_per_attempt=distinct_episodes,
        cooldown_after_apply_slots=caps.cooldown_after_apply_slots,
    )


def _watch_window(turn: int, cfg: Any) -> tuple[int, int]:
    width = max(1, int(getattr(cfg, "adaptive_control_watch_window_turns", 5) or 5))
    start = int(turn) + 1
    return start, int(turn) + width


def _pending_watch(session: Any) -> dict[str, Any] | None:
    pending = getattr(session, "_llm_detector_pending_watch", None)
    return pending if isinstance(pending, dict) else None


def _set_pending_watch(
    session: Any,
    turn: int,
    cfg: Any,
    *,
    detector_family: str,
    signal_id: str,
    intervention_id: str,
    episode_id: str,
    episode_attempt_index: int,
    previous_intervention_id: str,
    same_hurdle_escalation: bool,
) -> None:
    start, end = _watch_window(turn, cfg)
    setattr(session, "_llm_detector_pending_watch", {
        "detector_family": detector_family,
        "signal_id": signal_id,
        "intervention_id": intervention_id,
        "episode_id": episode_id,
        "episode_attempt_index": int(episode_attempt_index),
        "previous_intervention_id": previous_intervention_id,
        "same_hurdle_escalation": bool(same_hurdle_escalation),
        "intervention_slot": int(turn),
        "watch_window_start": start,
        "watch_window_end": end,
    })


def _clear_pending_watch(session: Any) -> None:
    if hasattr(session, "_llm_detector_pending_watch"):
        setattr(session, "_llm_detector_pending_watch", None)


def _restore_baseline_for_watch_close(session: Any, row: dict[str, Any], reason: str):
    from . import executors

    result = executors.restore_baseline(session)
    restore = _result_dict(result)
    restore["reason"] = reason
    row["baseline_restore"] = restore
    return result


def _result_dict(result: Any) -> dict[str, Any]:
    result_dict = dataclasses.asdict(result)
    result_dict["apply_status"] = result.apply_status
    return result_dict


def _append_detector_control_ledger(
    *,
    session: Any,
    turn: int,
    verdict: LLMDetectorVerdict,
    controller_version: str,
    chosen: dict[str, Any] | None,
    apply_status: str,
    blocked_reason: str,
    result: Any,
    episode_transition: str = "",
    controller_state: str = "",
    immediate_effect: str = "",
    active_hurdle_mode: str = "",
    next_hurdle_mode: str = "",
    watch_window_start: Any = "",
    watch_window_end: Any = "",
    hurdle_episode_id: str = "",
    episode_online_signal_id: str = "",
    episode_attempt_index: str = "",
    candidate_exclusion_refs: str = "",
    previous_intervention_id: str = "",
    same_hurdle_escalation: str = "",
    next_episode_id: str = "",
) -> None:
    cfg = getattr(session, "cfg", None)
    ledger_path = _control_ledger_path(session)
    if not ledger_path:
        return
    from . import ledger
    from .schema import ControlLedgerRow

    chosen = chosen or {}
    result = result or None
    basis = "atlas_ranked_ladder"
    if blocked_reason:
        basis += f":{blocked_reason}"
    row = ControlLedgerRow(
        attempt_id=str(getattr(session, "attempt_id", "") or ""),
        instance_id=str(getattr(session, "instance_id", "") or getattr(cfg, "adaptive_control_source_instance_id", "") or ""),
        controller_version=controller_version,
        control_model=str(getattr(cfg, "adaptive_control_model", "llm_detector_oscillation") or "llm_detector_oscillation"),
        observation_slot=int(turn),
        active_hurdle_mode=active_hurdle_mode or verdict.hurdle_family,
        diagnosis_status="active_confirmed",
        intervention_id=chosen.get("intervention_id", ""),
        intervention_basis=basis,
        apply_status=apply_status,
        intervention_slot=str(turn) if apply_status == "applied" else "",
        evidence_refs=";".join(verdict.evidence_refs),
        future_evidence_used="false",
        baseline_config_paths=";".join(getattr(result, "baseline_config_paths", ()) or ()),
        candidate_config_path=chosen.get("candidate_config_path", "") or getattr(result, "candidate_config_path", ""),
        active_config_basis=getattr(result, "active_config_basis", ""),
        pre_intervention_config_digest=getattr(result, "pre_digest", ""),
        post_intervention_config_digest=getattr(result, "post_digest", ""),
        refreshed_surfaces=";".join(getattr(result, "refreshed_surfaces", ()) or ()),
        changed_config_fields=";".join(getattr(result, "changed_config_fields", ()) or ()),
        blocked_config_fields=";".join(getattr(result, "blocked_config_fields", ()) or ()),
        watch_window_start=str(watch_window_start) if watch_window_start != "" else "",
        watch_window_end=str(watch_window_end) if watch_window_end != "" else "",
        immediate_effect=immediate_effect,
        next_hurdle_mode=next_hurdle_mode,
        controller_state=controller_state,
        hurdle_episode_id=hurdle_episode_id,
        episode_online_signal_id=episode_online_signal_id or chosen.get("online_signal_id", ""),
        episode_attempt_index=episode_attempt_index,
        candidate_rank_at_selection=(
            chosen.get("rank_within_ladder", "")
            or chosen.get("rank_within_family", "")
            or chosen.get("rank_within_hurdle", "")
        ),
        candidate_exclusion_refs=candidate_exclusion_refs,
        previous_intervention_id=previous_intervention_id,
        same_hurdle_escalation=same_hurdle_escalation,
        episode_transition=episode_transition,
        next_episode_id=next_episode_id,
    )
    ledger.append_row(ledger_path, row)


def _control_ledger_path(session: Any) -> str:
    cfg = getattr(session, "cfg", None)
    configured = str(getattr(cfg, "adaptive_control_ledger_path", "") or "")
    if not configured:
        return ""
    path = Path(configured)
    if path.is_absolute():
        return str(path)
    trace_path = getattr(session, "_trace_path", None)
    if trace_path:
        return str(Path(trace_path).parent / path)
    return configured
