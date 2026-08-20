"""Tests for the watch and clearance classifier.

Hurdle clearance from the post-intervention prefix; never reads scorer output.
Produces each of cleared / unchanged / blocked / unknown.
"""
from __future__ import annotations

import _ac_bootstrap  # noqa: F401  (stub-parent bootstrap; must precede harness import)
from llm_solver.harness.adaptive_control import watch

SIG = "repeat_loop_without_new_evidence"


def _loop(n):
    return [{"repeat_signature": "LOOP", "slot_state": "run"} for _ in range(n)]


def test_known_signal_blocks_when_detector_is_unwired():
    assert watch.classify_watch(_loop(6), SIG) == "blocked"


def test_progress_cannot_be_called_cleared_without_detector():
    post = [{"repeat_signature": "A", "slot_state": "run"},
            {"repeat_signature": "B", "slot_state": "edit", "source_mutation": "true"}]
    assert watch.classify_watch(post, SIG) == "blocked"


def test_blocked_when_no_stream():
    assert watch.classify_watch(None, SIG) == "blocked"


def test_unwired_detector_blocks_before_window_length_check():
    assert watch.classify_watch([], SIG, min_slots=1) == "blocked"


def test_unwired_detector_blocks_even_when_pattern_is_gone():
    post = [{"repeat_signature": f"S{i}", "slot_state": "read"} for i in range(3)]
    assert watch.classify_watch(post, SIG) == "blocked"


def test_unknown_signal_blocks():
    assert watch.classify_watch(_loop(6), "no_such_signal") == "blocked"


def test_exclusion_context_test_is_not_material_progress():
    slot = {
        "slot_idx": 103,
        "slot_state": "tool_error",
        "test_execution_action": "true",
        "test_exit_status": "fail",
        "exec_outcome": "fail",
        "exclusion_context": "true",
        "control_context": "git_stash",
    }

    assert watch.material_progress(slot) is False


if __name__ == "__main__":
    for n in list(globals()):
        if n.startswith("test_"):
            globals()[n]()
    print("Watch tests passed")
