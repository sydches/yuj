"""Multi-intervention episode state machine.

Per-attempt bookkeeping the pause hook uses to drive same-hurdle escalation and
sequential hurdle shepherding. Pure state + small helpers; the session owns one
EpisodeMachine. Stdlib-only so the harness loop imports it with no new deps.

The default caps allow one episode and one intervention.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Controller states.
MONITORING = "MONITORING"
ACTIVE_HURDLE = "ACTIVE_HURDLE"
SELECT_MEDICINE = "SELECT_MEDICINE"
APPLYING = "APPLYING"
WATCHING = "WATCHING"
EPISODE_EXHAUSTED = "EPISODE_EXHAUSTED"
STOPPED = "STOPPED"

# Keep cap block reasons distinct from intervention-budget reasons.
PER_ATTEMPT_EXHAUSTED = "interventions_per_attempt_exhausted"
PER_EPISODE_EXHAUSTED = "interventions_per_episode_exhausted"
DISTINCT_EPISODES_EXHAUSTED = "distinct_episodes_exhausted"
COOLDOWN_ACTIVE = "cooldown_active"

# Episode transition values (mirror watch.EPISODE_TRANSITIONS).
CLEARED_TO_PROGRESS = "cleared_to_progress"
CLEARED_TO_LATER_HURDLE = "cleared_to_later_hurdle"
UNCHANGED = "unchanged"
WORSENED = "worsened"
UNKNOWN = "unknown"
DISLODGED_NO_PROGRESS = "dislodged_no_progress"
CANDIDATE_EXHAUSTED = "candidate_exhausted"


@dataclass
class Caps:
    max_interventions_per_attempt: int = 1
    max_interventions_per_hurdle_episode: int = 1
    max_distinct_hurdle_episodes_per_attempt: int = 1
    cooldown_after_apply_slots: int = 5


def caps_from_cfg(cfg) -> Caps:
    def _i(key: str, default: int) -> int:
        # `or default` would eat an explicit 0 (0 or 1 == 1), making a
        # zero cap unsettable — the withheld-branch shadow mode needs
        # max_interventions_per_attempt = 0 to mean zero.
        value = getattr(cfg, key, None)
        if value is None or value == "":
            return default
        return int(value)

    cooldown_default = _i("adaptive_control_watch_window_turns", 5)
    return Caps(
        max_interventions_per_attempt=_i("adaptive_control_max_interventions_per_attempt", 1),
        max_interventions_per_hurdle_episode=_i("adaptive_control_max_interventions_per_hurdle_episode", 1),
        max_distinct_hurdle_episodes_per_attempt=_i("adaptive_control_max_distinct_hurdle_episodes_per_attempt", 1),
        cooldown_after_apply_slots=int(
            getattr(cfg, "adaptive_control_cooldown_after_apply_slots", cooldown_default) or cooldown_default),
    )


@dataclass
class Episode:
    episode_id: str
    online_signal_id: str
    open_slot: int
    attempt_index: int = 0  # interventions applied in this episode
    applied_intervention_ids: list = field(default_factory=list)
    previous_intervention_id: str = ""
    status: str = "open"  # open | cleared | exhausted | worsened


@dataclass
class ApplyPlan:
    allowed: bool
    block_reason: str = ""
    state: str = MONITORING
    is_escalation: bool = False
    is_new_episode: bool = False
    exclude_ids: tuple = ()
    previous_intervention_id: str = ""
    episode: "Episode | None" = None  # open episode to escalate, else None (new)


@dataclass
class EpisodeMachine:
    state: str = MONITORING
    interventions_total: int = 0
    episodes_opened: int = 0
    cooldown_until: int = -1
    last_intervention_slot: int = 0
    exhausted_slot: int = -1
    current: "Episode | None" = None
    episodes: list = field(default_factory=list)


def machine(session) -> EpisodeMachine:
    m = getattr(session, "_adaptive_control_episode_machine", None)
    if m is None:
        # stop_resume delivery: a prior segment may have saved controller
        # memory at its stop — restore it so the ladder continues instead
        # of starting a fresh attempt every segment.
        if getattr(getattr(session, "cfg", None),
                   "adaptive_control_delivery", "in_place") == "stop_resume":
            from .persistence import load_state
            load_state(session)
            m = getattr(session, "_adaptive_control_episode_machine", None)
    if m is None:
        m = EpisodeMachine()
        setattr(session, "_adaptive_control_episode_machine", m)
    return m


def plan_apply(m: EpisodeMachine, caps: Caps, signal: str, slot: int) -> ApplyPlan:
    """Decide whether and how to apply for `signal` at `slot`. Does not mutate
    counters — `record_apply` does that once the apply actually succeeds."""
    if m.state == EPISODE_EXHAUSTED:
        return ApplyPlan(False, CANDIDATE_EXHAUSTED, EPISODE_EXHAUSTED)
    if slot < m.cooldown_until:
        return ApplyPlan(False, COOLDOWN_ACTIVE, WATCHING)
    if m.interventions_total >= caps.max_interventions_per_attempt:
        return ApplyPlan(False, PER_ATTEMPT_EXHAUSTED, ACTIVE_HURDLE)
    cur = m.current
    if cur is not None and cur.status == "open" and cur.online_signal_id == signal:
        if cur.attempt_index >= caps.max_interventions_per_hurdle_episode:
            cur.status = "exhausted"
            m.state = EPISODE_EXHAUSTED
            return ApplyPlan(False, PER_EPISODE_EXHAUSTED, EPISODE_EXHAUSTED, episode=cur)
        prev = cur.applied_intervention_ids[-1] if cur.applied_intervention_ids else ""
        return ApplyPlan(
            True, "", SELECT_MEDICINE,
            is_escalation=cur.attempt_index >= 1,
            exclude_ids=tuple(cur.applied_intervention_ids),
            previous_intervention_id=prev, episode=cur)
    if m.episodes_opened >= caps.max_distinct_hurdle_episodes_per_attempt:
        return ApplyPlan(False, DISTINCT_EPISODES_EXHAUSTED, ACTIVE_HURDLE)
    return ApplyPlan(True, "", SELECT_MEDICINE, is_new_episode=True, episode=None)


def record_apply(m: EpisodeMachine, caps: Caps, signal: str, intervention_id: str,
                 slot: int, attempt_id: str) -> Episode:
    """Record an apply, start its cooldown, and return its episode."""
    cur = m.current
    if cur is None or cur.status != "open" or cur.online_signal_id != signal:
        m.episodes_opened += 1
        cur = Episode(episode_id=f"{attempt_id}#ep{m.episodes_opened}",
                      online_signal_id=signal, open_slot=int(slot))
        m.current = cur
        m.episodes.append(cur)
    cur.previous_intervention_id = cur.applied_intervention_ids[-1] if cur.applied_intervention_ids else ""
    cur.attempt_index += 1
    cur.applied_intervention_ids.append(intervention_id)
    m.interventions_total += 1
    m.last_intervention_slot = int(slot)
    m.cooldown_until = int(slot) + max(0, caps.cooldown_after_apply_slots)
    m.state = WATCHING
    return cur


def close_watch(m: EpisodeMachine, transition: str) -> "Episode | None":
    """Update the machine when the watch window closes with `transition`.
    Returns the episode that was being watched (may now be closed)."""
    ep = m.current
    if ep is None:
        return None
    if transition == CLEARED_TO_PROGRESS:
        ep.status = "cleared"
        m.current = None
        m.state = MONITORING
    elif transition == CLEARED_TO_LATER_HURDLE:
        ep.status = "cleared"
        m.current = None
        m.state = ACTIVE_HURDLE  # a new episode opens when the next apply lands
    elif transition == WORSENED:
        ep.status = "worsened"
        m.current = None
        m.state = STOPPED
    elif transition == UNCHANGED:
        m.state = ACTIVE_HURDLE  # escalate on the next active pause
    elif transition == DISLODGED_NO_PROGRESS:
        # The treated symptom went quiet, but no task progress followed. Keep
        # the episode and its candidate exclusions so a recurrence escalates.
        m.state = ACTIVE_HURDLE
    else:  # UNKNOWN: do not count as cleared; keep the episode open, keep monitoring
        m.state = MONITORING
    return ep


def mark_candidate_exhausted(m: EpisodeMachine, slot: int = -1) -> "Episode | None":
    ep = m.current
    if ep is not None:
        ep.status = "exhausted"
    m.current = None
    m.exhausted_slot = int(slot)
    m.state = EPISODE_EXHAUSTED
    return ep


def resume_after_progress(m: EpisodeMachine) -> bool:
    """Leave terminal exhaustion only after external task progress."""
    if m.state != EPISODE_EXHAUSTED:
        return False
    m.state = MONITORING
    m.exhausted_slot = -1
    return True
