"""Watch classifier: did an active online signal clear after intervention?

This is hurdle clearance, not final task resolution, and never reads scorer
output. It reuses the live detector's own rule and material-progress primitive on
the post-intervention window. The pure classifier is used by the pause hook when
the watch window closes.
"""
from __future__ import annotations

from . import detectors

CLEARED = "cleared"
UNCHANGED = "unchanged"
WORSENED = "worsened"  # reserved until a concrete harm detector is specified
BLOCKED = "blocked"
UNKNOWN = "unknown"

WATCH_RESULTS = {CLEARED, UNCHANGED, WORSENED, BLOCKED, UNKNOWN}


def _material_progress(slot: dict) -> bool:
    if slot.get("exclusion_context") == "true":
        return False
    if "effective_source_mutation" in slot:
        source_progress = slot.get("effective_source_mutation") == "true"
    else:
        source_progress = slot.get("source_mutation") == "true"
    return (
        source_progress
        or slot.get("test_execution_action") == "true"
        or slot.get("test_exit_status") in {"pass", "fail"}
        or slot.get("exec_outcome") in {"pass", "fail"}
        or slot.get("submit_like_action") == "true"
        or slot.get("slot_state") in {"test_pass", "test_fail", "submit", "done"}
    )


def material_progress(slot: dict) -> bool:
    """Public predicate shared by live and offline watch classifiers."""
    return _material_progress(slot)


def classify_watch(post_slots, online_signal_id: str, *, min_slots: int = 1) -> str:
    """Classify the watch window for `repeat_loop_without_new_evidence`.

    post_slots: prefix-only slots strictly after the intervention slot, within the
    watch window (or None if the stream is unavailable).
    """
    if post_slots is None:
        return BLOCKED
    fn = detectors.SIGNAL_DETECTORS.get(online_signal_id)
    if fn is None:
        return BLOCKED
    if len(post_slots) < min_slots:
        return UNKNOWN

    # the live detector returns status strings: "blocked" / "active_confirmed" / "no_fire"
    status, _refs, _blocked = fn(post_slots)
    if status == "blocked":
        return BLOCKED
    if status == "active_confirmed":
        return UNCHANGED  # the same signal is still active in the window
    # signal no longer active: cleared only if there is material progress evidence
    if any(_material_progress(s) for s in post_slots):
        return CLEARED
    return UNKNOWN  # gone but no progress evidence -> insufficient to call cleared


# Episode-level transition classes.
CLEARED_TO_PROGRESS = "cleared_to_progress"
CLEARED_TO_LATER_HURDLE = "cleared_to_later_hurdle"
CANDIDATE_EXHAUSTED = "candidate_exhausted"
EPISODE_TRANSITIONS = {
    CLEARED_TO_PROGRESS, CLEARED_TO_LATER_HURDLE, UNCHANGED, WORSENED, UNKNOWN, CANDIDATE_EXHAUSTED,
}

# transition -> short immediate_effect label kept for compact reports / back-compat
_SHORT = {
    CLEARED_TO_PROGRESS: CLEARED, CLEARED_TO_LATER_HURDLE: CLEARED,
    UNCHANGED: UNCHANGED, WORSENED: WORSENED, UNKNOWN: UNKNOWN, CANDIDATE_EXHAUSTED: UNCHANGED,
}


def short_effect(transition: str) -> str:
    """Map an episode transition to the short immediate_effect label."""
    return _SHORT.get(transition, UNKNOWN)


def classify_episode_transition(post_slots, online_signal_id: str, other_signal_ids=()):
    """Classify what happened to the treated hurdle episode after the watch window.

    Returns (transition, next_signal_id). `next_signal_id` is the later hurdle's
    online signal for `cleared_to_later_hurdle`, else "". Prefix-only; reuses the
    same detectors and material-progress primitive as the live diagnosis. Never
    reads scorer/terminal evidence.
    """
    if post_slots is None:
        return UNKNOWN, ""
    fn = detectors.SIGNAL_DETECTORS.get(online_signal_id)
    if fn is None:
        return UNKNOWN, ""
    status, _refs, _blocked = fn(post_slots)
    if status == "active_confirmed":
        return UNCHANGED, ""  # treated signal still active
    # treated signal no longer active in the window — did a different one appear?
    for other in other_signal_ids:
        if other == online_signal_id:
            continue
        ofn = detectors.SIGNAL_DETECTORS.get(other)
        if ofn is None:
            continue
        ostatus, _r, _b = ofn(post_slots)
        if ostatus == "active_confirmed":
            return CLEARED_TO_LATER_HURDLE, other
    if any(_material_progress(s) for s in post_slots):
        return CLEARED_TO_PROGRESS, ""
    return UNKNOWN, ""  # gone but no progress evidence -> insufficient


# Version 2 splits UNKNOWN into more exact states and adds an unwired-signal
# basis. The live path still uses version 1. Callers must opt in to version 2.
WATCH_CLASSIFIER_VERSION = "watch_classifier_v2"

INSUFFICIENT_WINDOW = "insufficient_window"   # too few post-slots: artifact, not verdict
DISLODGED_NO_PROGRESS = "dislodged_no_progress"  # stuck pattern broke; no material progress
INERT = "inert"                                # window repeats the pre-existing pattern
PROGRESSED = "progressed"                      # material progress, signal has no wired
                                               # detector -> cannot certify "cleared"

WATCH_RESULTS_V2 = WATCH_RESULTS | {
    INSUFFICIENT_WINDOW, DISLODGED_NO_PROGRESS, INERT, PROGRESSED,
}


def _signatures(slots, executed_only: bool = False) -> set:
    out = set()
    for s in (slots or []):
        if executed_only and s.get("executed") == "false":
            continue  # blocked/errored calls never ran: not "new actions"
        if s.get("repeat_signature"):
            out.add(s.get("repeat_signature"))
    return out


def classify_watch_v2(post_slots, online_signal_id: str, *, min_slots: int = 1,
                      pre_slots=None) -> str:
    """v2 watch classification (docs: the dislodged/inert/insufficient split).

    pre_slots: the slots leading INTO the branch point (the stuck pattern's
    reference window). Without it, dislodged-vs-inert cannot be claimed and
    the no-progress case stays UNKNOWN — we never assert "dislodged" without
    the reference.
    """
    if post_slots is None:
        return BLOCKED
    post_slots = list(post_slots)
    if len(post_slots) < min_slots:
        return INSUFFICIENT_WINDOW

    fn = detectors.SIGNAL_DETECTORS.get(online_signal_id)
    progressed = any(_material_progress(s) for s in post_slots)

    if fn is not None:
        status, _refs, _blocked = fn(post_slots)
        if status == "blocked":
            return BLOCKED
        if status == "active_confirmed":
            return UNCHANGED
        if progressed:
            return CLEARED
    else:
        # unwired signal (e.g. replay_stop_capture): no detector can certify
        # the hurdle gone; the honest ceiling is "progressed"
        if progressed:
            return PROGRESSED

    if pre_slots is None:
        return UNKNOWN  # no reference window: cannot split dislodged/inert
    # executed_only on POST: a window of gate-blocked novel commands did
    # not break the stuck pattern. Pre side keeps all
    # signatures — anything the model already tried, executed or not, is
    # not "new".
    new_actions = _signatures(post_slots, executed_only=True) - _signatures(pre_slots)
    return DISLODGED_NO_PROGRESS if new_actions else INERT
