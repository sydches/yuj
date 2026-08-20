"""Replay handover: swap the replay client for the live model at the stop turn.

Spec: docs/replay_mode_spec.md "Handover (branching)". When the replay source
reports REPLAY_FINISH_REASON_STOP_TURN and a handover is armed, the session's
client becomes the live client and an optional overlay config (the knob under
test) is applied through the SAME canonical apply path live interventions use
(`adaptive_control.executors.apply` — toml_overlay_control_v1; no parallel
config mechanics). The watch budget bounds the live continuation.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

HANDOVER_INTERVENTION_ID = "replay_handover"


def arm(replay_client, *, live_client_factory, overlay_path: str = "",
        watch_turns: int = 0) -> None:
    """Attach handover intent to a ReplayClient (set in __main__)."""
    replay_client.handover = {
        "live_client_factory": live_client_factory,
        "overlay_path": overlay_path or "",
        "watch_turns": int(watch_turns or 0),
        "done": False,
    }


def maybe_handover(session, turn: int) -> bool:
    """If the session's client is a stopped replay with an armed handover,
    perform the swap. Returns True when the caller should retry the chat
    call against the (now live) client."""
    client = session.client
    handover = getattr(client, "handover", None)
    if not handover or handover.get("done"):
        return False

    from ...server.replay_client import ReplayClient  # local: avoid cycle
    if not isinstance(client, ReplayClient):
        return False

    live = handover["live_client_factory"]()
    # Keep writing the transcript that contains the replay prefix. Continue
    # its counter so turn markers stay in order.
    diary = getattr(client, "_transcript_path", None)
    if diary is not None and hasattr(live, "set_transcript"):
        client.close_transcript()
        live.set_transcript(diary, append=True)
        live._transcript_call_n = getattr(client, "_transcript_call_n", 0)
    session.client = live
    handover["done"] = True
    log.info("REPLAY HANDOVER at turn %d: live client engaged (diary=%s)",
             turn, diary or "none")

    overlay = handover.get("overlay_path") or ""
    if overlay:
        from ..adaptive_control.executors import TOML_OVERLAY_EXECUTOR_ID, apply
        from ..adaptive_control.schema import InterventionPayload
        result = apply(session, InterventionPayload(
            intervention_id=HANDOVER_INTERVENTION_ID,
            executor_id=TOML_OVERLAY_EXECUTOR_ID,
            timing_class="toml_overlay_reactive",
            candidate_config_path=overlay,
        ))
        log.info("REPLAY HANDOVER overlay=%s apply_status=%s reason=%s",
                 overlay, result.apply_status, result.blocked_reason or "-")
        if not result.applied:
            # End the handover when the required overlay cannot be applied.
            session._last_chat_error_reason = "replay_handover_overlay_blocked"
            return False
        # keep the live client's cfg in step with the applied overlay
        live.cfg = session.cfg

    watch = int(handover.get("watch_turns") or 0)
    if watch > 0:
        try:
            import dataclasses
            session.cfg = dataclasses.replace(
                session.cfg, max_turns=int(turn) + watch)
            live.cfg = session.cfg
            log.info("REPLAY HANDOVER watch window: turns %d..%d",
                     turn + 1, turn + watch)
        except Exception as e:  # noqa: BLE001 - watch bound is advisory
            log.warning("watch bound not applied: %s", e)
    return True


def maybe_capture_at_stop(session, entry: dict) -> None:
    """Capture the replay-stop bundle the moment the stop turn's tool_call
    event is recorded (state-complete: context already holds the result).
    This fires even when the session ENDS on that turn — the walked-turn ==
    final-turn edge (premature-done hurdles) that a next-chat trigger misses."""
    client = session.client
    stop = int(getattr(client, "stop_turn", 0) or 0)
    if not getattr(client, "is_replay", False) or stop <= 0:
        return
    if int(entry.get("turn_number", -1) or -1) != stop:
        return
    cfg = getattr(session, "cfg", None)
    if not getattr(cfg, "adaptive_control_branch_bundle_enabled", False):
        return
    from types import SimpleNamespace
    from ..adaptive_control import branch_bundle
    shim = SimpleNamespace(
        diagnosis_status="active_confirmed",
        active_hurdle_mode="replay_stop_capture",
        detector_id="replay_stop_capture",
        detector_status="replay_stop",
        basis_refs=[f"replay_stop_turn={stop}"],
    )
    row = branch_bundle.maybe_capture(session, shim, stop, "replay_stop")
    log.info("replay stop capture (at execution): status=%s path=%s reason=%s",
             row.get("status"), row.get("path"), row.get("reason") or "-")
