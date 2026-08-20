"""Adaptive controller: fail-closed wrapper around live detector registry.

Reactive live signals are registered in `detectors.SIGNAL_DETECTORS`. That
registry may deliberately be empty while detector rules are rebuilt. In that
state the controller must not diagnose, select, or apply anything.
"""
from __future__ import annotations

from . import CONTROLLER_VERSION
from .detectors import SIGNAL_DETECTORS
from .schema import ControllerDecision, ControllerRequest


class Controller:
    version = CONTROLLER_VERSION
    has_live_detector = bool(SIGNAL_DETECTORS)

    def __init__(self, detector_mode: str = "manual", detector_registry: dict | None = None) -> None:
        self.detector_mode = detector_mode
        self.signal_detectors = SIGNAL_DETECTORS if detector_registry is None else detector_registry
        self.has_live_detector = bool(self.signal_detectors)

    def propose(self, request: ControllerRequest, slots=None) -> ControllerDecision:
        sig = request.online_signal_id
        mode = (request.detector_mode or self.detector_mode or "manual").strip().lower()
        if mode == "adaptive":
            return ControllerDecision(
                diagnosis_status="blocked_unobservable", detector_id=sig,
                detector_status="blocked", detector_blocked_reason="adaptive_detector_not_wired")
        if mode != "manual":
            return ControllerDecision(
                diagnosis_status="blocked_unobservable", detector_id=sig,
                detector_status="blocked", detector_blocked_reason="unknown_detector_mode")
        if not sig:
            # no target signal configured: the controller has no policy to run
            return ControllerDecision(
                diagnosis_status="no_active_hurdle",
                detector_status="not_run", detector_blocked_reason="controller_stub")

        fn = self.signal_detectors.get(sig)
        if fn is None:
            return ControllerDecision(
                diagnosis_status="no_active_hurdle", detector_id=sig,
                detector_status="blocked", detector_blocked_reason="detector_not_wired")

        status, refs, blocked = fn(slots)
        if status == "active_confirmed":
            return ControllerDecision(
                diagnosis_status="active_confirmed", active_hurdle_mode=sig,
                detector_id=sig, detector_status="active_confirmed",
                basis_refs=[refs] if refs else [])
        if status == "blocked":
            return ControllerDecision(
                diagnosis_status="blocked_unobservable", detector_id=sig,
                detector_status="blocked", detector_blocked_reason=blocked)
        return ControllerDecision(
            diagnosis_status="no_active_hurdle", detector_id=sig, detector_status="no_fire")
