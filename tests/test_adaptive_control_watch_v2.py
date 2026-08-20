"""classify_watch_v2: the dislodged/inert/insufficient/progressed split.

v1 classify_watch and classify_episode_transition are FROZEN (live path);
this file also pins that freeze.
"""
from __future__ import annotations

import _ac_bootstrap  # noqa: F401
from llm_solver.harness.adaptive_control import watch
from llm_solver.harness.adaptive_control.watch import classify_watch_v2 as v2

SIG = "repeat_loop_without_new_evidence"


def _slot(i, sig="", **kw):
    s = {"slot_idx": i, "slot_state": "run", "source_mutation": "false",
         "submit_like_action": "false", "repeat_signature": sig}
    s.update(kw)
    return s


PRE = [_slot(i, sig="bash|sed -n 100p x.py") for i in range(5)]  # the stuck pattern


def test_insufficient_window():
    assert v2([], SIG, pre_slots=PRE) == watch.INSUFFICIENT_WINDOW


def test_unwired_known_signal_uses_inert_fallback():
    post = [_slot(10 + i, sig="bash|sed -n 100p x.py") for i in range(5)]
    assert v2(post, SIG, pre_slots=PRE) == watch.INERT


def test_unwired_known_signal_progressed_not_cleared():
    post = [_slot(10, sig="bash|pytest x", source_mutation="true",
                  contact_state="source_write")]
    assert v2(post, SIG, pre_slots=PRE) == watch.PROGRESSED


def test_failed_effective_mutation_is_not_cleared():
    post = [_slot(10, sig="bash|cp /tmp/fix.py src/app.py",
                  source_mutation="true",
                  contact_state="source_write",
                  effective_source_mutation="false")]
    assert v2(post, SIG, pre_slots=PRE) == watch.DISLODGED_NO_PROGRESS


def test_dislodged_new_actions_no_progress():
    post = [_slot(10, sig="bash|cat README"), _slot(11, sig="bash|grep foo y.py")]
    assert v2(post, SIG, pre_slots=PRE) == watch.DISLODGED_NO_PROGRESS


def test_inert_same_pattern_below_detector_threshold():
    post = [_slot(10, sig="bash|sed -n 100p x.py"),
            _slot(11, sig="bash|sed -n 100p x.py")]  # 2 repeats: detector K=5 silent
    assert v2(post, SIG, pre_slots=PRE) == watch.INERT


def test_no_reference_window_stays_unknown():
    post = [_slot(10, sig="bash|cat README")]
    assert v2(post, SIG) == watch.UNKNOWN


def test_unwired_signal_progressed_not_cleared():
    post = [_slot(10, sig="bash|pytest", source_mutation="true",
                  contact_state="source_write")]
    assert v2(post, "replay_stop_capture", pre_slots=PRE) == watch.PROGRESSED


def test_unwired_signal_dislodged_and_inert():
    post_new = [_slot(10, sig="bash|cat README")]
    assert v2(post_new, "replay_stop_capture", pre_slots=PRE) == watch.DISLODGED_NO_PROGRESS
    post_same = [_slot(10, sig="bash|sed -n 100p x.py")]
    assert v2(post_same, "replay_stop_capture", pre_slots=PRE) == watch.INERT


def test_v1_frozen():
    """The live path must be untouched: same signatures, same labels."""
    import inspect
    sig = inspect.signature(watch.classify_watch)
    assert list(sig.parameters) == ["post_slots", "online_signal_id", "min_slots"]
    post = [_slot(10, sig="bash|cat README")]
    assert watch.classify_watch(post, SIG) == watch.BLOCKED
    assert watch.classify_watch(post, "replay_stop_capture") == watch.BLOCKED
