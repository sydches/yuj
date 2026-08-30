"""Tests for harness loop, tools, solver, generate pipeline, config, and end-to-end integration."""
import json
import os
import subprocess as _subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import openai
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from llm_solver.server.types import TurnResult, Usage, ToolCall
from llm_solver.config import Config, load_config, MODEL_MAP, _deep_merge, get_sdk_config


# ──────────────────────────────────────────────
# Helper: build a Config without loading TOML
# ──────────────────────────────────────────────

from _config_helpers import make_config  # centralized defaults — see tests/_config_helpers.py


def make_turn_result(content=None, tool_calls=None, finish_reason="stop", prompt_tokens=10):
    return TurnResult(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=5),
    )


def _request_text(client) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for call in client.chat.call_args_list
        for message in call.args[0]
    )

class TestRuminationArmThresholdAbsolute:
    """rumination_gate_arm_threshold_abs decouples the gate from max_turns.
    When > 0, it overrides the percentage-of-max_turns calculation."""

    def test_abs_overrides_percentage(self):
        from llm_solver.harness.guardrails import init_guardrail_state
        cfg = make_config(
            max_turns=60, duplicate_abort=20,
            rumination_nudge_threshold=20,          # → nudge = 12 (20% of 60)
            rumination_gate_arm_threshold=30,        # → would-be arm = 18 (30% of 60)
            rumination_gate_arm_threshold_abs=30,    # overrides: arm = 30
        )
        state = init_guardrail_state(cfg)
        assert state.rumination_arm_threshold == 30
        assert state.rumination_nudge_threshold == 12

    def test_abs_zero_preserves_percentage_mode(self):
        from llm_solver.harness.guardrails import init_guardrail_state
        cfg = make_config(
            max_turns=90, duplicate_abort=20,
            rumination_nudge_threshold=20,
            rumination_gate_arm_threshold=30,
            rumination_gate_arm_threshold_abs=0,    # legacy mode
        )
        state = init_guardrail_state(cfg)
        assert state.rumination_arm_threshold == 27   # 30% of 90

    def test_abs_below_nudge_floor_clamps(self):
        """arm_abs must not shrink below the nudge threshold — the nudge→arm
        ordering is a precondition the rumination ladder relies on."""
        from llm_solver.harness.guardrails import init_guardrail_state
        cfg = make_config(
            max_turns=200, duplicate_abort=20,
            rumination_nudge_threshold=20,          # → nudge = 40
            rumination_gate_arm_threshold=30,
            rumination_gate_arm_threshold_abs=10,   # below nudge
        )
        state = init_guardrail_state(cfg)
        assert state.rumination_arm_threshold == 40   # clamped to nudge


class TestRuminationNudgeThresholdAbsolute:
    """rumination_nudge_threshold_abs decouples the nudge from max_turns.
    When > 0, it overrides the percentage-of-max_turns calculation."""

    def test_abs_overrides_percentage(self):
        from llm_solver.harness.guardrails import init_guardrail_state
        cfg = make_config(
            max_turns=90, duplicate_abort=20,
            rumination_nudge_threshold=20,           # → would-be nudge = 18
            rumination_nudge_threshold_abs=12,       # overrides: nudge = 12
            rumination_gate_arm_threshold=30,
        )
        state = init_guardrail_state(cfg)
        assert state.rumination_nudge_threshold == 12
        # Arm still computes from percentage unless arm_abs is also set.
        assert state.rumination_arm_threshold == 27  # 30% of 90

    def test_abs_zero_preserves_percentage_mode(self):
        from llm_solver.harness.guardrails import init_guardrail_state
        cfg = make_config(
            max_turns=90, duplicate_abort=20,
            rumination_nudge_threshold=20,
            rumination_nudge_threshold_abs=0,
            rumination_gate_arm_threshold=30,
        )
        state = init_guardrail_state(cfg)
        assert state.rumination_nudge_threshold == 18  # 20% of 90

    def test_abs_respects_min_threshold_floor(self):
        """nudge_abs must not drop below cfg.rumination_min_threshold."""
        from llm_solver.harness.guardrails import init_guardrail_state
        cfg = make_config(
            max_turns=90, duplicate_abort=20,
            rumination_nudge_threshold=20,
            rumination_nudge_threshold_abs=3,        # below min floor (default 6)
            rumination_gate_arm_threshold=30,
        )
        state = init_guardrail_state(cfg)
        assert state.rumination_nudge_threshold == 6  # clamped to min_threshold floor

    def test_both_abs_knobs_compose(self):
        """nudge_abs and arm_abs compose independently."""
        from llm_solver.harness.guardrails import init_guardrail_state
        cfg = make_config(
            max_turns=90, duplicate_abort=20,
            rumination_nudge_threshold=20,
            rumination_nudge_threshold_abs=12,
            rumination_gate_arm_threshold=30,
            rumination_gate_arm_threshold_abs=30,
        )
        state = init_guardrail_state(cfg)
        assert state.rumination_nudge_threshold == 12
        assert state.rumination_arm_threshold == 30


class TestRuminationNudgePostMutationThreshold:
    """rumination_nudge_threshold_abs_post_mutation sets a separate nudge
    threshold for after the model's first successful write/edit. Allows
    asymmetric nudge timing: aggressive pre-mutation push, baseline-like
    post-mutation nudge."""

    def _drive_ladder(self, non_writes, has_mutated_flag, pre_abs, post_abs):
        from llm_solver.harness.guardrails import (
            rumination_ladder, init_guardrail_state,
        )
        cfg = make_config(
            max_turns=90, duplicate_abort=20,
            rumination_nudge_threshold=20,
            rumination_nudge_threshold_abs=pre_abs,
            rumination_nudge_threshold_abs_post_mutation=post_abs,
            rumination_gate_arm_threshold=30,
        )
        state = init_guardrail_state(cfg)
        state.has_mutated = has_mutated_flag
        last = None
        for _ in range(non_writes):
            last = rumination_ladder(state, cfg, tc_name="bash",
                                     result="output",
                                     gate_blocked=False,
                                     already_blocked_this_turn=False)
        return state, last

    def test_post_abs_overrides_pre_after_mutation(self):
        """Pre-mut fires at 12. Post-mut threshold 18 delays it until 18 non-writes."""
        # Post-mutation, with 12 non-writes — should NOT fire (below post=18)
        state, d = self._drive_ladder(12, has_mutated_flag=True, pre_abs=12, post_abs=18)
        assert not d.text, "Post-mut nudge should not fire at count=12 when post_abs=18"

        # Post-mutation, with 18 non-writes — should fire
        state, d = self._drive_ladder(18, has_mutated_flag=True, pre_abs=12, post_abs=18)
        assert d.text, "Post-mut nudge should fire at count=18 when post_abs=18"

    def test_pre_abs_applies_before_mutation(self):
        """Pre-mutation, nudge fires at pre_abs (12) regardless of post_abs."""
        state, d = self._drive_ladder(12, has_mutated_flag=False, pre_abs=12, post_abs=18)
        assert d.text, "Pre-mut nudge should fire at pre_abs threshold"

    def test_post_abs_zero_defaults_to_pre(self):
        """When post_abs=0, post threshold equals pre threshold (symmetric)."""
        state, d = self._drive_ladder(12, has_mutated_flag=True, pre_abs=12, post_abs=0)
        assert d.text, "Post-mut nudge should fire at pre_abs when post_abs=0"

    def test_init_stores_both_thresholds(self):
        from llm_solver.harness.guardrails import init_guardrail_state
        cfg = make_config(
            max_turns=90, duplicate_abort=20,
            rumination_nudge_threshold=20,
            rumination_nudge_threshold_abs=12,
            rumination_nudge_threshold_abs_post_mutation=18,
            rumination_gate_arm_threshold=30,
        )
        state = init_guardrail_state(cfg)
        assert state.rumination_nudge_threshold == 12
        assert state.rumination_nudge_threshold_post_mutation == 18


class TestRuminationNudgeOnlyPreMutation:
    """When rumination_nudge_only_pre_mutation is True, the nudge fires
    only before the model's first successful write/edit. Post-mutation
    non-write streaks are left alone (they're productive exploration
    between edits, not stuck rumination)."""

    def _ladder(self, has_mutated_flag, only_pre_mut):
        from llm_solver.harness.guardrails import rumination_ladder, init_guardrail_state
        cfg = make_config(
            max_turns=90, duplicate_abort=20,
            rumination_nudge_threshold=20,  # nudge=18 at mt=90
            rumination_gate_arm_threshold=30,
            rumination_nudge_only_pre_mutation=only_pre_mut,
        )
        state = init_guardrail_state(cfg)
        state.has_mutated = has_mutated_flag
        # Drive 18 non-write calls to the threshold.
        last = None
        for _ in range(18):
            last = rumination_ladder(state, cfg, tc_name="bash",
                                     result="some output",
                                     gate_blocked=False,
                                     already_blocked_this_turn=False)
        return last

    def test_nudge_fires_pre_mutation_when_toggle_on(self):
        d = self._ladder(has_mutated_flag=False, only_pre_mut=True)
        assert d.text, "Nudge should fire when has_mutated=False"

    def test_nudge_suppressed_post_mutation_when_toggle_on(self):
        d = self._ladder(has_mutated_flag=True, only_pre_mut=True)
        assert not d.text, "Nudge should be suppressed when has_mutated=True"

    def test_nudge_fires_post_mutation_when_toggle_off(self):
        """Default behavior: nudge fires regardless of has_mutated state."""
        d = self._ladder(has_mutated_flag=True, only_pre_mut=False)
        assert d.text, "Nudge should fire in legacy (toggle-off) mode"


class TestRuminationSameTarget:
    """Repeated inspection of the same target should arm the existing
    rumination gate earlier than the coarse non-write threshold when the
    same-target knobs are configured."""

    def test_same_target_path_warns_and_arms_early(self):
        from llm_solver.harness.loop import Session

        cfg = make_config(
            max_turns=100,
            duplicate_abort=20,
            rumination_nudge_threshold=80,  # keep coarse nudge out of the way
            rumination_same_target_warn_count=3,
            rumination_same_target_arm_count=4,
            rumination_gate_grace_calls=1,
            loop_detect_enabled=False,  # exercise the rumination ladder, not loop_detect
        )
        client = MagicMock()

        call_count = [0]

        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 6:
                return make_turn_result(content="done", finish_reason="stop")
            tc = [ToolCall(id=f"c{call_count[0]}", name="read",
                          arguments={"path": "pkg/mod.py"})]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        dispatch_calls = []
        tool_results = []

        def tracking_dispatch(name, args, cwd, cfg, **kwargs):
            dispatch_calls.append((name, dict(args)))
            return "module contents"

        with patch("llm_solver.harness.loop.dispatch", side_effect=tracking_dispatch):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            orig_add = session.context.add_tool_result

            def capture(cid, result, **kwargs):
                tool_results.append(result)
                return orig_add(cid, result, **kwargs)

            session.context.add_tool_result = capture
            session.run()

        assert len(dispatch_calls) == 5, dispatch_calls
        request_text = _request_text(client)
        assert "same target hit 3 times" in request_text
        assert "pkg/mod.py" in request_text
        assert "Gate armed" in request_text
        assert all("same target hit" not in result for result in tool_results)
        assert "Gate armed" not in tool_results[4], tool_results
        assert tool_results[5].startswith("NOT EXECUTED."), tool_results
        assert "module contents" not in tool_results[5]

    def test_same_target_streak_resets_when_target_changes(self):
        from llm_solver.harness.loop import Session

        cfg = make_config(
            max_turns=100,
            duplicate_abort=20,
            rumination_nudge_threshold=80,
            rumination_same_target_warn_count=3,
            rumination_same_target_arm_count=4,
        )
        client = MagicMock()

        sequence = [
            {"path": "a.py"},
            {"path": "a.py"},
            {"path": "a.py"},
            {"path": "b.py"},
            {"path": "b.py"},
        ]
        call_count = [0]

        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > len(sequence):
                return make_turn_result(content="done", finish_reason="stop")
            tc = [ToolCall(id=f"c{call_count[0]}", name="read",
                          arguments=sequence[call_count[0] - 1])]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        tool_results = []
        with patch("llm_solver.harness.loop.dispatch", return_value="file contents"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            orig_add = session.context.add_tool_result

            def capture(cid, result, **kwargs):
                tool_results.append(result)
                return orig_add(cid, result, **kwargs)

            session.context.add_tool_result = capture
            session.run()

        final_request = client.chat.call_args_list[-1].args[0]
        assert sum(
            "same target hit" in str(message.get("content") or "")
            for message in final_request
        ) == 1
        assert not any("same target hit" in r for r in tool_results)
        assert not any("[harness gate]" in r for r in tool_results), tool_results


class TestTestReadGuard:

    def test_warns_when_running_tests_without_reading_test_file(self):
        from llm_solver.harness.guardrails import test_read_ladder, init_guardrail_state

        cfg = make_config(test_read_warn_after=1, test_read_nudge="read {target} after {count}")
        state = init_guardrail_state(cfg)
        decision = test_read_ladder(
            state,
            cfg,
            tc_name="bash",
            result="failed\n[exit code: 1]",
            gate_blocked=False,
            tc_args={"cmd": "pytest -q tests/test_app.py"},
        )
        assert decision.text == "read tests/test_app.py after 1"

    def test_resets_after_test_file_is_read(self):
        from llm_solver.harness.guardrails import (
            test_read_ladder,
            observe_test_file_read,
            init_guardrail_state,
        )

        cfg = make_config(test_read_warn_after=1, test_read_nudge="read {target} after {count}")
        state = init_guardrail_state(cfg)

        first = test_read_ladder(
            state,
            cfg,
            tc_name="bash",
            result="failed\n[exit code: 1]",
            gate_blocked=False,
            tc_args={"cmd": "pytest -q tests/test_app.py"},
        )
        assert first.text == "read tests/test_app.py after 1"

        observe_test_file_read(
            state,
            cfg,
            tc_name="read",
            result="def test_app(): ...",
            gate_blocked=False,
            tc_args={"path": "tests/test_app.py"},
            focus_key="file:tests/test_app.py",
            focus_display="tests/test_app.py",
        )

        second = test_read_ladder(
            state,
            cfg,
            tc_name="bash",
            result="failed\n[exit code: 1]",
            gate_blocked=False,
            tc_args={"cmd": "pytest -q tests/test_app.py"},
        )
        assert second.action.value == "pass"
