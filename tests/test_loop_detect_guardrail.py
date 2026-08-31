"""Tests for the loop_detect guardrail — tight consecutive-identical
signature detector with a single recovery-inject before hard abort.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from _config_helpers import make_config
from llm_solver.harness.guardrails import (
    Action,
    GuardrailState,
    init_guardrail_state,
    loop_detect,
)
from llm_solver.harness._loop.run_step import _run_post_turn_hooks
from llm_solver.server.types import ToolCall, TurnResult, Usage


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


def test_active_adaptive_watch_cannot_suppress_terminal_loop_guard(tmp_path):
    from llm_solver.harness.loop import Session

    cfg = make_config(
        max_turns=6,
        duplicate_abort=20,
        loop_detect_enabled=True,
        loop_detect_threshold=2,
        adaptive_control_enabled=False,
    )
    client = MagicMock()
    client.chat.return_value = TurnResult(
        content=None,
        tool_calls=[ToolCall(
            id="repeat",
            name="read",
            arguments={"path": "same.py"},
        )],
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )
    client.build_assistant_message.return_value = {
        "role": "assistant", "content": None,
    }
    session = Session(cfg, client, "sys", "prompt", str(tmp_path))
    session._llm_detector_pending_watch = {"watch_window_end": 99}

    with patch("llm_solver.harness.loop.dispatch", return_value="ok"):
        result = session.run()

    assert result.finish_reason == "loop_detected"
    assert result.done is False
    assert client.chat.call_count == 4


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
