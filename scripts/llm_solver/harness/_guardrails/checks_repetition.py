"""Post-turn decisions over completed observation receipts."""
from typing import Any
from .state import GuardrailState, Decision, PASS


def duplicate_guard(state: GuardrailState, cfg: Any, *,
                    tool_calls_sig: tuple, observations: tuple | None = None,
                    allow_intervention: bool = True, turn_number: int | None = None) -> Decision:
    """Limit repeated completed observations; unknown effects break the streak.

    The loop calls this only after the whole batch has completed and records
    the decision separately from the tool outputs. Counts are a declared
    repetition policy, not proof of task failure or lack of reasoning.
    """
    from ..repeated_observations import observation_key
    state.duplicate_evidence = {"eligible": False, "count": 0}
    keys = tuple(observation_key(item) for item in (observations or ()))
    if (not cfg.duplicate_guard_enabled or not keys
            or len(keys) != len(tool_calls_sig) or any(key is None for key in keys)):
        state.recent_calls.clear()
        state.duplicate_count = 0
        state.duplicate_warned = False
        return PASS
    tool_calls_sig = (tool_calls_sig, keys)
    if state.recent_calls and state.recent_calls[-1] == tool_calls_sig:
        state.duplicate_count += 1
    else:
        state.duplicate_count = 1
        state.duplicate_warned = False
    state.recent_calls.append(tool_calls_sig)
    tail = state.duplicate_count
    state.duplicate_evidence = {"eligible": True, "count": tail}
    # One observation is not a repetition, even with a declared limit of one.
    if tail < 2 or not allow_intervention:
        return PASS
    # Repetition is not a completion or failure verdict. The legacy abort
    # setting remains loadable, but never ends a session.
    if cfg.duplicate_warn_count > 0:
        if tail >= cfg.duplicate_warn_count and not state.duplicate_warned:
            state.duplicate_warned = True
            return Decision.warn(
                cfg.duplicate_warn.format(count=tail, abort="disabled",
                                          prior_turn=(turn_number - tail + 1 if turn_number is not None else 'earlier')),
                reason="duplicate_guard",
            )
    return PASS
