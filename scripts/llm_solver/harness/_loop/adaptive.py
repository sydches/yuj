"""Adaptive policy switching: phase-2 escalation based on pressure + test signal."""
from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..loop import Session

log = logging.getLogger(__name__)


def observe_test_signal(session: "Session", cmd: str, result: str) -> None:
    """Mark that we observed a non-trivial test command signal."""
    if result.startswith("ERROR:"):
        return
    cmd_s = (cmd or "").strip()
    if not cmd_s:
        return
    is_test = False
    if session.output_control is not None:
        try:
            from ...bash_quirks.transforms import _is_test_command
            is_test = _is_test_command(cmd_s, session.output_control)
        except Exception:
            is_test = False
    if not is_test:
        is_test = bool(re.search(
            r"\b(pytest|unittest|cargo test|go test|ctest|npm test|pnpm test|yarn test)\b",
            cmd_s,
        ))
    if is_test:
        exit_marker_present = "[exit code:" in result
        exit_ok = ("[exit code: 0]" in result) or not exit_marker_present
        if exit_ok:
            session._observed_test_signal = True


def maybe_switch_adaptive_phase(session: "Session", turn: int) -> None:
    """Switch from base to phase2 policy when configured conditions are met."""
    cfg = session.cfg
    if session._adaptive_switched or not getattr(cfg, "adaptive_policy_enabled", False):
        return
    if turn < int(getattr(cfg, "adaptive_switch_min_turn", 0) or 0):
        return
    if getattr(cfg, "adaptive_requires_mutation", True) and not session._guards.has_mutated:
        return
    if getattr(cfg, "adaptive_requires_test_signal", True) and not session._observed_test_signal:
        return

    window = int(getattr(cfg, "adaptive_low_pressure_window", 0) or 0)
    max_events = int(getattr(cfg, "adaptive_low_pressure_max_events", 0) or 0)
    if window > 0:
        if len(session._pressure_events) < window:
            return
        recent = list(session._pressure_events)[-window:]
        if sum(1 for x in recent if x) > max_events:
            return

    session.cfg = replace(
        cfg,
        done_guard_enabled=bool(getattr(cfg, "adaptive_phase2_done_guard_enabled", True)),
        bash_transforms_task_format_enabled=bool(
            getattr(cfg, "adaptive_phase2_bash_task_format_enabled", True)
        ),
        bash_transforms_structured_output_enabled=bool(
            getattr(cfg, "adaptive_phase2_bash_structured_output_enabled", True)
        ),
        bash_transforms_sink_threshold_chars=int(
            getattr(cfg, "adaptive_phase2_bash_sink_threshold_chars", 0) or 0
        ),
    )
    session._adaptive_phase = "phase2"
    session._adaptive_switched = True
    session._emit(
        "adaptive_phase_switch",
        session_number=session._session_number,
        turn_number=turn,
        phase=session._adaptive_phase,
        done_guard_enabled=session.cfg.done_guard_enabled,
        bash_task_format_enabled=session.cfg.bash_transforms_task_format_enabled,
        bash_structured_output_enabled=session.cfg.bash_transforms_structured_output_enabled,
        bash_sink_threshold_chars=session.cfg.bash_transforms_sink_threshold_chars,
    )
    log.info(
        "Adaptive policy switched to phase2 at turn %d (done_guard=%s, bash_task_format=%s, structured=%s)",
        turn,
        session.cfg.done_guard_enabled,
        session.cfg.bash_transforms_task_format_enabled,
        session.cfg.bash_transforms_structured_output_enabled,
    )
