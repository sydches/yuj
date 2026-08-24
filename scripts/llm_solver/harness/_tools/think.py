"""Side-effect-free scratchpad tool."""
from __future__ import annotations

from ..thoughts import EMPTY_THINK_RESULT


def think(thought: str, *, enabled: bool) -> str:
    """Acknowledge model reasoning without executing or persisting it here."""
    if not enabled:
        return "ERROR: think tool is disabled (tools.think_enabled=false)"
    if not isinstance(thought, str):
        raise TypeError("thought must be a string")
    return EMPTY_THINK_RESULT


__all__ = ["think"]
