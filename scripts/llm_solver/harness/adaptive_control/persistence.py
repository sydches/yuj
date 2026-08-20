"""Keep controller state across stop_resume segments.

The controller may stop at an application or escalation and resume in a
new session. Preserve the episode machine and pending watch so the new
segment continues the same attempt.

Save: called by ``executors.stop_for_resume`` at the stop.
Load: called lazily by ``episode.machine()`` on first touch in a new
session when the state file exists.

Watch windows use turn numbers from the stopping session. The overlay is
applied when the run resumes. Rebase the pending watch so it starts at turn
0 in the resumed session.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from ..._shared.telemetry_paths import ensure_telemetry_dir, telemetry_dir

STATE_NAME = "adaptive_controller_state.json"


def save_state(session) -> bool:
    from .episode import EpisodeMachine
    m = getattr(session, "_adaptive_control_episode_machine", None)
    watch = getattr(session, "_llm_detector_pending_watch", None)
    if m is None and watch is None:
        return False
    payload = {
        "machine": dataclasses.asdict(m) if m is not None else None,
        "pending_watch": dict(watch) if watch else None,
    }
    try:
        tdir = ensure_telemetry_dir(Path(getattr(session, "cwd", ".")))
        (tdir / STATE_NAME).write_text(json.dumps(payload, indent=1))
        return True
    except Exception:
        return False


def load_state(session) -> bool:
    """Restore machine + watch into the session; returns True on restore.

    The state file is consumed (renamed) so a later unrelated session in
    the same workspace cannot accidentally inherit stale state.
    """
    from .episode import Episode, EpisodeMachine
    try:
        p = telemetry_dir(Path(getattr(session, "cwd", "."))) / STATE_NAME
        if not p.exists():
            return False
        payload = json.loads(p.read_text())
        p.rename(p.with_suffix(".json.consumed"))
    except Exception:
        return False

    md = payload.get("machine")
    if md:
        episodes = [Episode(**e) for e in md.get("episodes", [])]
        cur = md.get("current")
        current = None
        if cur:
            current = next(
                (e for e in episodes if e.episode_id == cur.get("episode_id")),
                Episode(**cur),
            )
        m = EpisodeMachine(
            state=md.get("state", "monitoring"),
            interventions_total=int(md.get("interventions_total", 0)),
            episodes_opened=int(md.get("episodes_opened", 0)),
            # cooldown/window turns are session-local; rebased below
            cooldown_until=-1,
            last_intervention_slot=0,
            exhausted_slot=int(md.get("exhausted_slot", -1)),
            current=current,
            episodes=episodes,
        )
        setattr(session, "_adaptive_control_episode_machine", m)

    w = payload.get("pending_watch")
    if w:
        span = max(1, int(w.get("watch_window_end", 5)) - int(w.get("watch_window_start", 0)))
        w = dict(w)
        # rebase: the knob rides in at resume; watch the segment's first turns
        w["watch_window_start"] = 0
        w["watch_window_end"] = span
        setattr(session, "_llm_detector_pending_watch", w)
    return True


__all__ = ["save_state", "load_state", "STATE_NAME"]
