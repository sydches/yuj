"""Tests for the loop_detect guardrail — tight consecutive-identical
signature detector with a single recovery-inject before hard abort.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from _config_helpers import make_config
from llm_solver.harness.guardrails import (
    Action,
    Decision,
    GuardrailState,
    init_guardrail_state,
    loop_detect,
)
from llm_solver.harness._loop.run_step import (
    _defer_guard_end_during_active_watch,
    _run_post_turn_hooks,
)


def _state() -> GuardrailState:
    cfg = make_config(loop_detect_enabled=True, loop_detect_threshold=3)
    return init_guardrail_state(cfg), cfg


class TestLoopDetect:

    def test_disabled_passes(self):
        cfg = make_config(loop_detect_enabled=False, loop_detect_threshold=3)
        state = init_guardrail_state(cfg)
        sig = (("read", '{"path": "a"}'),)
        for _ in range(20):
            d = loop_detect(state, cfg, tool_calls_sig=sig)
            assert d.action == Action.PASS
        assert state.loop_detect_streak == 0

    def test_passes_below_threshold(self):
        state, cfg = _state()
        sig = (("read", '{"path": "a"}'),)
        for _ in range(2):  # threshold is 3
            d = loop_detect(state, cfg, tool_calls_sig=sig)
            assert d.action == Action.PASS
        assert state.loop_detect_streak == 2
        assert not state.loop_detect_warned

    def test_warns_at_threshold(self):
        state, cfg = _state()
        sig = (("read", '{"path": "a"}'),)
        loop_detect(state, cfg, tool_calls_sig=sig)       # streak 1
        loop_detect(state, cfg, tool_calls_sig=sig)       # streak 2
        d = loop_detect(state, cfg, tool_calls_sig=sig)   # streak 3 → WARN
        assert d.action == Action.WARN
        assert "Loop detected" in d.text
        assert state.loop_detect_warned

    def test_ends_on_next_repeat_after_warn(self):
        state, cfg = _state()
        sig = (("read", '{"path": "a"}'),)
        loop_detect(state, cfg, tool_calls_sig=sig)
        loop_detect(state, cfg, tool_calls_sig=sig)
        loop_detect(state, cfg, tool_calls_sig=sig)       # WARN
        d = loop_detect(state, cfg, tool_calls_sig=sig)   # END
        assert d.action == Action.END
        assert d.reason == "loop_detected"

    def test_reset_on_different_signature(self):
        state, cfg = _state()
        sig_a = (("read", '{"path": "a"}'),)
        sig_b = (("read", '{"path": "b"}'),)
        loop_detect(state, cfg, tool_calls_sig=sig_a)
        loop_detect(state, cfg, tool_calls_sig=sig_a)
        loop_detect(state, cfg, tool_calls_sig=sig_a)     # WARN
        assert state.loop_detect_warned
        d = loop_detect(state, cfg, tool_calls_sig=sig_b)
        assert d.action == Action.PASS
        assert state.loop_detect_streak == 1
        assert not state.loop_detect_warned

    def test_warn_then_break_then_warn_again(self):
        """A broken pattern resets fully; a new pattern must earn its
        own WARN before END can fire."""
        state, cfg = _state()
        sig_a = (("read", '{"path": "a"}'),)
        sig_b = (("read", '{"path": "b"}'),)
        for _ in range(3):
            loop_detect(state, cfg, tool_calls_sig=sig_a)  # WARN on 3rd
        loop_detect(state, cfg, tool_calls_sig=sig_b)      # break
        # Now build a second streak on sig_b; it should WARN not END.
        loop_detect(state, cfg, tool_calls_sig=sig_b)
        d = loop_detect(state, cfg, tool_calls_sig=sig_b)  # streak 3 on sig_b
        assert d.action == Action.WARN

    def test_registry_exposes_loop_detect(self):
        from llm_solver.harness.guardrails import (
            build_guardrail_registry,
            validate_guardrail_registry,
        )
        reg = build_guardrail_registry()
        assert "loop_detect" in reg.turn_pre_dispatch
        validate_guardrail_registry(reg)


def test_terminal_guard_is_deferred_during_adaptive_watch():
    events = []

    class Session:
        _session_number = 1
        _llm_detector_pending_watch = {
            "intervention_id": "toml_overlay.apply::loop.loop_detect_on_default",
            "episode_id": "attempt#ep1",
            "watch_window_end": 16,
        }

        def _emit(self, event, **fields):
            events.append((event, fields))

    deferred = _defer_guard_end_during_active_watch(
        Session(),
        Decision.end("loop_detected"),
        guard_name="loop_detect",
        turn=13,
    )

    assert deferred is True
    assert events == [(
        "adaptive_control_guard_end_deferred",
        {
            "session_number": 1,
            "turn_number": 13,
            "guard_name": "loop_detect",
            "guard_reason": "loop_detected",
            "intervention_id": "toml_overlay.apply::loop.loop_detect_on_default",
            "hurdle_episode_id": "attempt#ep1",
            "watch_window_end": 16,
        },
    )]


def test_terminal_guard_is_not_deferred_without_adaptive_watch():
    class Session:
        _session_number = 1

    assert _defer_guard_end_during_active_watch(
        Session(),
        Decision.end("loop_detected"),
        guard_name="loop_detect",
        turn=13,
    ) is False


def test_terminal_guard_is_not_deferred_after_watch_end():
    class Session:
        _session_number = 1
        _llm_detector_pending_watch = {"watch_window_end": 12}

    assert _defer_guard_end_during_active_watch(
        Session(),
        Decision.end("loop_detected"),
        guard_name="loop_detect",
        turn=13,
    ) is False


def test_blocked_turn_runs_all_post_turn_hooks():
    calls = []

    class Session:
        def _maybe_emit_harness_observation(self, turn):
            calls.append(("observation", turn))

        def _maybe_run_llm_hurdle_detector(self, turn):
            calls.append(("detector", turn))

        def _maybe_switch_adaptive_phase(self, turn):
            calls.append(("phase", turn))

        def _maybe_run_advisor(self, turn):
            calls.append(("advisor", turn))

    _run_post_turn_hooks(Session(), 36)

    assert calls == [
        ("observation", 36),
        ("detector", 36),
        ("phase", 36),
        ("advisor", 36),
    ]
