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

class TestRuminationNudge:
    """Counterpart to error_nudge: detects model stuck reading/grepping
    without committing to a write, and sends guidance in the next synthetic
    user turn so the model has an off-ramp from exploration."""

    def test_nudge_after_threshold_non_write_calls(self):
        """After rumination threshold non-write tool calls in a row,
        a nudge is sent with the next model request."""
        from llm_solver.harness.loop import Session
        threshold = 7  # at max_turns=100, 7% → 7 absolute (above floor of 6)
        cfg = make_config(max_turns=100, duplicate_abort=20,
                          rumination_nudge_threshold=threshold)
        client = MagicMock()

        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > threshold + 1:
                return make_turn_result(content="done", finish_reason="stop")
            tc = [ToolCall(id=f"c{call_count[0]}", name="bash",
                          arguments={"cmd": f"find . -name file{call_count[0]}.py"})]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        tool_results = []
        with patch("llm_solver.harness.loop.dispatch", return_value="some_output.py"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            orig_add = session.context.add_tool_result
            def capture(cid, result, **kwargs):
                tool_results.append(result)
                return orig_add(cid, result, **kwargs)
            session.context.add_tool_result = capture
            session.run()

        request_text = _request_text(client)
        assert "[HARNESS:" in request_text
        assert "non-mutation calls" in request_text
        assert all("[HARNESS:" not in result for result in tool_results)
        assert all("some_output.py" in result for result in tool_results)
        # Nudge text must mention the threshold count and the hard gate.
        assert f"{threshold}" in request_text
        assert "file-mutation tools" in request_text
        assert "Mutate a file now" in request_text

    def test_write_resets_rumination_counter(self):
        """A write/edit call resets the non-write counter — no nudge fires if
        the model regularly alternates reads with writes."""
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=100, duplicate_abort=20,
                          rumination_nudge_threshold=7)  # 7% of 100 → 7 absolute (above floor of 6)
        client = MagicMock()

        # Sequence: bash, bash, edit, bash, bash, stop
        # Each bash increments; the edit resets. Max streak = 2, below threshold 7.
        tool_sequence = [
            ("bash", {"cmd": "ls"}),
            ("bash", {"cmd": "cat f.py"}),
            ("edit", {"path": "f.py", "old_str": "x", "new_str": "y"}),
            ("bash", {"cmd": "ls"}),
            ("bash", {"cmd": "cat g.py"}),
        ]
        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > len(tool_sequence):
                return make_turn_result(content="done", finish_reason="stop")
            name, args = tool_sequence[call_count[0] - 1]
            tc = [ToolCall(id=f"c{call_count[0]}", name=name, arguments=args)]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        tool_results = []
        with patch("llm_solver.harness.loop.dispatch", return_value="ok"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            orig_add = session.context.add_tool_result
            def capture(cid, result, **kwargs):
                tool_results.append(result)
                return orig_add(cid, result, **kwargs)
            session.context.add_tool_result = capture
            session.run()

        nudge_results = [r for r in tool_results if "tool calls since your last write" in r]
        assert len(nudge_results) == 0, f"Unexpected rumination nudge: {nudge_results}"

    def test_rumination_nudge_fires_once_per_cycle(self):
        """Nudge text is one-shot per non-write cycle. Once fired, it does
        not re-fire until a successful write/edit resets the cycle —
        otherwise every subsequent non-write call would carry an identical
        append and become noise."""
        from llm_solver.harness.loop import Session
        threshold = 7  # 7% of 100 → 7 absolute (above floor of 6)
        cfg = make_config(max_turns=100, duplicate_abort=20,
                          rumination_nudge_threshold=threshold)
        client = MagicMock()

        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > threshold * 2:
                return make_turn_result(content="done", finish_reason="stop")
            tc = [ToolCall(id=f"c{call_count[0]}", name="read",
                          arguments={"path": f"f{call_count[0]}.py"})]
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
        # Nudge fires exactly once at the nudge threshold. The gate arms
        # (same turn under legacy config; a later turn when
        # rumination_gate_arm_threshold > rumination_nudge_threshold).
        # Without a successful write/edit between, the nudge does not
        # re-fire even though non-write calls continue to accumulate.
        assert sum(
            "non-mutation calls" in str(message.get("content") or "")
            for message in final_request
        ) == 1
        assert all("non-mutation calls" not in result for result in tool_results)

    def test_gate_blocks_non_write_after_nudge_fires(self):
        """Once the nudge fires, the gate is armed. The model gets one
        grace call (dispatched with a warning), then the SECOND non-write
        call is rejected WITHOUT invoking dispatch."""
        from llm_solver.harness.loop import Session
        threshold = 7  # 7% of 100 → 7 absolute (above floor of 6)
        cfg = make_config(max_turns=100, duplicate_abort=20,
                          rumination_nudge_threshold=threshold)
        client = MagicMock()

        # threshold reads to trip the gate, then two more reads:
        # one grace (dispatched), one blocked.
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > threshold + 2:
                return make_turn_result(content="done", finish_reason="stop")
            tc = [ToolCall(id=f"c{call_count[0]}", name="read",
                          arguments={"path": f"f{call_count[0]}.py"})]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        call_count = [0]
        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        dispatch_calls = []
        def tracking_dispatch(name, args, cwd, cfg, **kwargs):
            dispatch_calls.append((name, args))
            return "dispatched-output"

        tool_results = []
        with patch("llm_solver.harness.loop.dispatch", side_effect=tracking_dispatch):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            orig_add = session.context.add_tool_result
            def capture(cid, result, **kwargs):
                tool_results.append(result)
                return orig_add(cid, result, **kwargs)
            session.context.add_tool_result = capture
            session.run()

        # threshold reads dispatched + 1 grace read dispatched = threshold+1
        assert len(dispatch_calls) == threshold + 1, (
            f"Expected {threshold + 1} dispatches (incl grace), got {len(dispatch_calls)}: {dispatch_calls}"
        )
        # Grace output stays factual; the warning is a user-turn fragment.
        grace_result = tool_results[threshold]
        assert grace_result == "dispatched-output"
        assert "Gate armed" in _request_text(client)
        # The blocked call (threshold+2) must be the gate rejection
        assert len(tool_results) == threshold + 2
        gated = tool_results[-1]
        assert gated.startswith("NOT EXECUTED.")
        assert "Only a file-mutation tool" in gated
        assert "dispatched-output" not in gated

    def test_gate_does_not_clear_on_errored_write_or_edit(self):
        """A write/edit that returns ERROR: (e.g. old_str==new_str or
        not found) must NOT clear the gate — otherwise the model can
        game the gate with a no-op edit purely to resume exploration.
        Observed on attempt_009: the model submitted `old_str == new_str`
        at turn 14 which errored but cleared the gate, then went back
        to reads."""
        from llm_solver.harness.loop import Session
        threshold = 7  # 7% of 100 → 7 absolute (above floor of 6)
        cfg = make_config(max_turns=100, duplicate_abort=20,
                          rumination_nudge_threshold=threshold)
        client = MagicMock()

        # threshold reads → gate arms (grace=1)
        # errored edit → gate stays armed, grace still 1
        # first read after edit → grace consumed (dispatched with warning)
        # second read → must be blocked (grace=0, gate still armed)
        tool_sequence = (
            [("read", {"path": f"f{i}.py"}) for i in range(threshold)]
            + [("edit", {"path": "x.py", "old_str": "a", "new_str": "a"})]  # no-op
            + [("read", {"path": "grace.py"})]
            + [("read", {"path": "blocked.py"})]
        )
        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > len(tool_sequence):
                return make_turn_result(content="done", finish_reason="stop")
            name, args = tool_sequence[call_count[0] - 1]
            tc = [ToolCall(id=f"c{call_count[0]}", name=name, arguments=args)]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        dispatch_calls = []
        def errored_edit_dispatch(name, args, cwd, cfg, **kwargs):
            dispatch_calls.append(name)
            if name == "edit":
                return "ERROR: old_str not found"
            return "ok"

        tool_results = []
        with patch("llm_solver.harness.loop.dispatch", side_effect=errored_edit_dispatch):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            orig_add = session.context.add_tool_result
            def capture(cid, result, **kwargs):
                tool_results.append(result)
                return orig_add(cid, result, **kwargs)
            session.context.add_tool_result = capture
            session.run()

        # Dispatch should have received: threshold reads + 1 edit + 1 grace read.
        # The final read ("blocked.py") must NOT reach dispatch because
        # the errored edit kept the gate armed and grace was consumed.
        assert dispatch_calls == ["read"] * threshold + ["edit"] + ["read"], (
            f"Unexpected dispatches: {dispatch_calls}"
        )
        # Grace output stays factual; the warning reaches the next request.
        grace_result = tool_results[threshold + 1]  # after threshold reads + edit
        assert grace_result == "ok"
        assert "Gate armed" in _request_text(client)
        # Last tool result is the gated read, not dispatched output.
        gated = tool_results[-1]
        assert gated.startswith("NOT EXECUTED."), f"Expected gate block, got: {gated}"

    def test_gate_pauses_duplicate_abort(self):
        """While the rumination gate is armed, duplicate_abort must NOT
        fire — otherwise the gate's rejection messages (which the model
        may respond to with repeated identical calls) would trip
        duplicate_abort and end the session before the gate could
        redirect the model. Observed on attempt_010: duplicate_abort
        at turn 7 stole the gate's window."""
        from llm_solver.harness.loop import Session
        threshold = 7  # 7% of 100 → 7 absolute (above floor of 6)
        # duplicate_abort=9 > threshold so varied reads don't fill the
        # deque. The model makes threshold varied reads (arms the gate),
        # then 3 identical gated calls that would have tripped
        # duplicate_abort in the old behavior. The session must NOT end
        # on duplicate_abort; the gate should handle those repeated calls.
        cfg = make_config(max_turns=100, duplicate_abort=9,
                          rumination_nudge_threshold=threshold)
        client = MagicMock()

        # First threshold calls: varied reads (trip gate without tripping
        # duplicate_abort since they're not identical). Then 3 identical
        # finds (would trip duplicate_abort if it weren't paused).
        tool_sequence = (
            [("read", {"path": f"f{i}.py"}) for i in range(threshold)]
            + [("bash", {"cmd": "find . -name x.py"}) for _ in range(3)]
        )
        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > len(tool_sequence):
                return make_turn_result(content="done", finish_reason="stop")
            name, args = tool_sequence[call_count[0] - 1]
            tc = [ToolCall(id=f"c{call_count[0]}", name=name, arguments=args)]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        with patch("llm_solver.harness.loop.dispatch", return_value="ok"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            result = session.run()

        # Session must not end on duplicate_abort — the gate should be
        # handling those repeated calls.
        assert result.finish_reason != "duplicate_abort", (
            f"duplicate_abort fired while gate was armed: {result.finish_reason}"
        )

    def test_gate_clears_on_write_or_edit(self):
        """A write or edit call clears the gate and resets the counter,
        so subsequent non-write calls pass through normally until the
        next threshold crossing."""
        from llm_solver.harness.loop import Session
        threshold = 7  # 7% of 100 → 7 absolute (above floor of 6)
        cfg = make_config(max_turns=100, duplicate_abort=20,
                          rumination_nudge_threshold=threshold)
        client = MagicMock()

        # threshold reads → gate arms
        # then write → gate clears
        # then one read → dispatches normally (not gated)
        # then stop
        tool_sequence = (
            [("read", {"path": f"f{i}.py"}) for i in range(threshold)]
            + [("write", {"path": "out.py", "content": "pass"})]
            + [("read", {"path": "after.py"})]
        )
        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > len(tool_sequence):
                return make_turn_result(content="done", finish_reason="stop")
            name, args = tool_sequence[call_count[0] - 1]
            tc = [ToolCall(id=f"c{call_count[0]}", name=name, arguments=args)]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        dispatch_calls = []
        def tracking_dispatch(name, args, cwd, cfg, **kwargs):
            dispatch_calls.append(name)
            return "ok"

        with patch("llm_solver.harness.loop.dispatch", side_effect=tracking_dispatch):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            session.run()

        # Every call in the sequence should have reached dispatch:
        # threshold reads + 1 write + 1 read = threshold + 2
        assert dispatch_calls == ["read"] * threshold + ["write", "read"], (
            f"Gate didn't clear on write: {dispatch_calls}"
        )
