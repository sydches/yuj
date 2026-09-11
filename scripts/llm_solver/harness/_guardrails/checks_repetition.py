"""Post-turn decisions over completed observation receipts."""
from typing import Any
from .state import GuardrailState, Decision, PASS


def duplicate_guard(state: GuardrailState, cfg: Any, *,
                    tool_calls_sig: tuple, observations: tuple | None = None,
                    allow_intervention: bool = True) -> Decision:
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
        return PASS
    tool_calls_sig = (tool_calls_sig, keys)
    state.recent_calls.append(tool_calls_sig)
    tail = 0
    for observed in reversed(state.recent_calls):
        if observed != tool_calls_sig:
            break
        tail += 1
    state.duplicate_evidence = {"eligible": True, "count": tail}
    # One observation is not a repetition, even with a declared limit of one.
    if tail < 2 or not allow_intervention:
        return PASS
    # END — declared-disabled when duplicate_abort <= 0 (some baselines
    # zero it; previously that silently never matched the deque length and
    # the warn text printed "session ends at 0 identical").
    if (cfg.duplicate_abort > 0
            and len(state.recent_calls) >= cfg.duplicate_abort
            and len(set(list(state.recent_calls)[-cfg.duplicate_abort:])) == 1):
        return Decision.end("duplicate_abort")
    # WARN (optional, config-gated)
    if cfg.duplicate_warn_count > 0:
        if tail >= cfg.duplicate_warn_count:
            abort_disp = cfg.duplicate_abort if cfg.duplicate_abort > 0 else "disabled"
            return Decision.warn(
                cfg.duplicate_warn.format(count=tail, abort=abort_disp),
                reason="duplicate_guard",
            )
    return PASS
