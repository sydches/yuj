"""Dataclasses threaded through ``dispatch_one_tool_call``.

Split from ``_dispatch_tool_call.py`` so each file fits under the
project's 500-line cap. These types are stable (the per-turn state
shape doesn't drift) — keeping them in their own module makes the
dispatch helper itself smaller without further fragmenting the logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import Config
    from ..loop import Session


@dataclass
class TurnState:
    """Per-turn state visible to ``dispatch_one_tool_call``.

    All fields except ``turn_had_pressure`` are read-only inside the
    helper (they describe the turn's invariants). ``turn_had_pressure``
    is the cross-tc accumulator the caller reads back after the loop.
    """
    session: "Session"
    turn: int
    cfg: "Config"
    content: str | None
    prompt_tokens: int
    completion_tokens: int
    turn_warn_text: str
    phase_chat_ms: float
    phase_token_ms: float
    turn_t0: float
    preexecuted: dict[str, str]
    pre_tool_hooks: dict[str, Any]
    schema_validations: dict[str, Any]
    permission_resolutions: dict[str, Any]
    dispatch: Callable[..., str]
    log: Any
    tool_pre: dict
    tool_post: dict
    observers: dict
    turn_had_pressure: bool = False


@dataclass
class TCOutcome:
    """Result of one tool-call dispatch.

    ``end=False`` → caller advances to the next tool call in this turn.
    ``end=True``  → caller returns SessionResult with ``reason`` /
                   ``done`` and the turn-level total_prompt /
                   total_completion accumulators.
    """
    end: bool = False
    reason: str | None = None
    done: bool = False
