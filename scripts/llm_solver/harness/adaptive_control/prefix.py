"""Build a live prefix snapshot for adaptive control.

Project slot_state, op_kind, obs, contact, and repeat signature through the
observation slot only.

No-peeking: the adapter reads only what the session exposes for slots <= t. It
never reads future turns, terminal suffix, or scorer outcome.

The session exposes slot facts through
`session.recent_prefix_slots(observation_slot)`. If that hook has no data, return
`prefix_snapshot_available=false` with `missing_slot_stream`.
"""
from __future__ import annotations

_VERIFY_STATES = {"test_pass", "test_fail"}
_SUBMIT_STATES = {"submit", "done"}


def _truthy_field(slots: list[dict], key: str) -> bool:
    """A fact is 'available' when at least one recent slot carries the field."""
    return any(key in s and s.get(key) not in (None, "") for s in slots)


def get_prefix_slots(session, observation_slot: int):
    """Return the session's prefix-only slot facts for slots <= t, or None if the
    session exposes no such stream. Reads only the documented hook."""
    getter = getattr(session, "recent_prefix_slots", None)
    if not callable(getter):
        return None
    try:
        slots = getter(observation_slot)
    except Exception:
        return None
    return None if slots is None else list(slots)


def build_prefix_snapshot(session, observation_slot: int) -> dict:
    """Convenience: fetch slots from the session and project them."""
    return snapshot_from_slots(get_prefix_slots(session, observation_slot))


def snapshot_from_slots(slots) -> dict:
    """Return the debug 'Prefix Availability' fields for a list of prefix slots
    (or None). Keys are a subset of the debug row schema, ready to splat in."""
    blank = {
        "prefix_snapshot_available": False,
        "recent_slot_count": 0,
        "recent_slot_tags": [],
        "mutation_state_available": False,
        "verification_state_available": False,
        "tool_error_state_available": False,
        "repeat_state_available": False,
        "prefix_blocked_reason": "missing_slot_stream",
        "prefix_evidence_refs": "",
    }
    if slots is None:
        return blank

    slots = list(slots)
    tags = [s.get("slot_state", "") for s in slots]
    refs = ";".join(s.get("evidence_refs", "") for s in slots if s.get("evidence_refs"))
    return {
        "prefix_snapshot_available": True,
        "recent_slot_count": len(slots),
        "recent_slot_tags": tags,
        "mutation_state_available": _truthy_field(slots, "source_mutation")
        or _truthy_field(slots, "contact_state"),
        "verification_state_available": _truthy_field(slots, "test_like_action")
        or any(s.get("slot_state") in _VERIFY_STATES for s in slots),
        "tool_error_state_available": _truthy_field(slots, "obs_state")
        or any(s.get("slot_state") == "tool_error" for s in slots),
        "repeat_state_available": _truthy_field(slots, "repeat_signature"),
        "prefix_blocked_reason": "",
        "prefix_evidence_refs": refs,
    }
