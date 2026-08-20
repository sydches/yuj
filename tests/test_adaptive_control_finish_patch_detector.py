"""Retired finish-patch detector compatibility."""
from __future__ import annotations

import _ac_bootstrap  # noqa: F401
from llm_solver.harness.adaptive_control import detectors
from llm_solver.harness.adaptive_control.detectors import (
    detect_done_with_zero_source_mutations as det,
)


def test_finish_patch_detector_symbol_is_retired():
    assert det([{"slot_state": "done", "submit_like_action": "true"}]) == ("no_fire", "", "")
    assert "done_called_with_zero_source_mutations_in_session" not in detectors.SIGNAL_DETECTORS


def test_zero_registry_has_no_finish_patch_guard():
    assert detectors.SIGNAL_DETECTORS == {}
