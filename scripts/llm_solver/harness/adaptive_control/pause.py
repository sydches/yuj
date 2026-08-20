"""Retired adaptive-control live pause hook.

The former live detector/watch controller lived behind
``adaptive_control_enabled`` and could diagnose, select, apply, then watch an
intervention from inside the solve loop. That path is retired for live use.

Historical detector/controller/watch modules remain importable for offline
analysis and replay-only comparison, but this live entry point must not run
them. The separate HarnessObservation layer replaces the live path.
"""
from __future__ import annotations

from . import PAUSE_BOUNDARY


LIVE_ADAPTIVE_CONTROL_RETIRED = True
RETIREMENT_REASON = "live_detector_watch_retired_for_harness_observation"


def maybe_pause_for_adaptive_control(
    session,
    turn,
    boundary_type=PAUSE_BOUNDARY,
) -> None:
    """No-op compatibility shim for the retired live detector/watch hook.

    This function intentionally ignores ``adaptive_control_enabled``. Keeping a
    callable shim lets historical tests/imports fail closed while preventing the
    detector/watch mechanism from acting as live intervention authority.
    """
    return None
