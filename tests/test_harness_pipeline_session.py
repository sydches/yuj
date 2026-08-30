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

class TestSessionRun:

    def test_session_stops_on_text_response(self):
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=10)
        client = MagicMock()
        client.chat.return_value = make_turn_result(content="done", finish_reason="stop")
        client.build_assistant_message.return_value = {"role": "assistant", "content": "done"}
        session = Session(cfg, client, "sys", "prompt", "/tmp")
        result = session.run()
        assert result.done is True
        assert result.finish_reason == "stop"
        assert client.chat.call_count == 1

    def test_session_duplicate_abort(self):
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=10, duplicate_abort=3)
        client = MagicMock()
        tc = [ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})]
        client.chat.return_value = make_turn_result(tool_calls=tc, finish_reason="tool_calls")
        client.build_assistant_message.return_value = {"role": "assistant", "content": None, "tool_calls": []}

        with patch("llm_solver.harness.loop.dispatch", return_value="output"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            result = session.run()

        assert result.finish_reason == "duplicate_abort"
        assert result.done is False

    def test_session_context_full(self):
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=10, context_size=100, context_fill_ratio=0.5)
        client = MagicMock()
        # Return a tool call so the session continues, but context estimate will be large
        tc = [ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})]
        client.chat.return_value = make_turn_result(tool_calls=tc, finish_reason="tool_calls", prompt_tokens=80)
        client.build_assistant_message.return_value = {"role": "assistant", "content": "x" * 1000}

        # Stub _get_server_ctx so the test's intentional context_size=100
        # isn't silently replaced when a live llama-server happens to be
        # running on :8080. Without this stub the test's tiny budget gets
        # overwritten to whatever the host server's n_ctx is (e.g. 65536),
        # the post-flight gate never trips, and the duplicate_abort fires
        # at turn 2 instead of context_full at turn 0.
        with patch("llm_solver.harness.loop.dispatch", return_value="y" * 1000), \
             patch.object(Session, "_get_server_ctx", return_value=0):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            result = session.run()

        assert result.finish_reason == "context_full"

    def test_session_context_full_preflight(self):
        """Pre-flight check ends session BEFORE the API call when context
        is already over budget at the top of a turn. Without this guard
        the request would 400 from the server (exceed_context_size_error)
        and the session would end with finish_reason='error' instead of
        'context_full'. Reproduces packaging.lv1 (Cfix run, T18 crash).
        """
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=10, context_size=100, context_fill_ratio=0.5)
        client = MagicMock()
        # If the pre-flight DOESN'T fire we'd hit chat — make that fatal.
        client.chat.side_effect = AssertionError("chat called despite over-budget context")
        # Same stub as test_session_context_full: prevent _get_server_ctx
        # from replacing cfg.context_size with the real server's n_ctx.
        with patch.object(Session, "_get_server_ctx", return_value=0):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            # Pre-load context past budget (>50 estimated tokens for
            # ctx=100, ratio=0.5). estimate_tokens() ≈ chars/4, so 1000
            # chars → ~250 tokens.
            session.context.add_user("x" * 1000)
            result = session.run()
        assert result.finish_reason == "context_full"
        assert client.chat.call_count == 0

    def test_session_context_full_preflight_uses_current_estimate_when_prior_pt_is_stale(self):
        """A large tool result can make the next request over budget even
        when the previous server-reported pt was safe.

        The pre-flight guard must look at the current context estimate too;
        relying only on the prior pt lets the next API call fail with an
        exceed-context error.
        """
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=10, context_size=100, context_fill_ratio=0.5)
        client = MagicMock()
        client.chat.side_effect = AssertionError("chat called despite stale pt plus over-budget context")

        with patch.object(Session, "_get_server_ctx", return_value=0):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            session._last_actual_prompt_tokens = 10
            session.context.add_tool_result("c1", "x" * 1000, tool_name="bash")
            result = session.run()

        assert result.finish_reason == "context_full"
        assert client.chat.call_count == 0

    def test_session_context_full_preflight_catches_underrun_via_live_pt_plus_delta(self):
        """The live count plus new-text estimate must catch an overrun."""
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=10, context_size=100, context_fill_ratio=0.5)
        client = MagicMock()
        client.chat.side_effect = AssertionError(
            "chat called despite live_pt+delta over budget")
        with patch.object(Session, "_get_server_ctx", return_value=0):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            # Last request: server said 45 (exact, gate at 50 passes).
            session._last_actual_prompt_tokens = 45
            # Estimator said 20 at that send (underrunning).
            session._preflight_prev_estimate = 20
            # New tool result grows the estimate to 35: still under the
            # gate on its own, but live 45 + delta 15 = 60 > 50.
            with patch.object(session.context, "estimate_tokens", return_value=35):
                result = session.run()
        assert result.finish_reason == "context_full"
        assert client.chat.call_count == 0

    def test_preflight_prompt_tokens_helper(self):
        from llm_solver.harness._loop.run_step import _preflight_prompt_tokens
        # no prior estimate: falls back to max(live, estimate)
        assert _preflight_prompt_tokens(45, 35, None) == 45
        # delta re-anchors the live term (density floor 0.25 = chars/4)
        assert _preflight_prompt_tokens(45, 35, 20) == 60
        # shrinking estimate (e.g. compaction) never subtracts
        assert _preflight_prompt_tokens(45, 15, 20) == 45
        # first turn (live_pt 0): estimate alone
        assert _preflight_prompt_tokens(0, 35, 20) == 35

    def test_preflight_density_calibration_catches_dense_output(self):
        """smoke20 django-11885: 37,652 new chars tokenized at 0.93
        tok/char (84,838 actual vs 65,536 n_ctx); the flat chars/4 delta
        admitted the request. With observed density the gate fires."""
        from llm_solver.harness._loop.run_step import (
            _observe_token_density, _preflight_prompt_tokens)
        from unittest.mock import MagicMock
        s = MagicMock(spec=[])
        # prior turn: gate saw live=38368, appended 12,248 chars (est delta
        # 3062); server then reported 49782 -> observed 0.93 tok/char.
        _observe_token_density(s, 38368, 12248, 49782)
        assert 0.9 < s._preflight_density < 1.0
        # fatal turn: live=49782, new est delta 9413 tokens (37,652 chars).
        got = _preflight_prompt_tokens(49782, 47000, 47000 - 9413,
                                       s._preflight_density)
        assert got > 62259  # 0.95 * 65536 -> guard fires
        # tiny appends never calibrate (template-overhead noise)
        s2 = MagicMock(spec=[])
        _observe_token_density(s2, 1000, 100, 1400)
        assert not hasattr(s2, "_preflight_density")

    def test_session_max_turns(self):
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=2, duplicate_abort=10)
        client = MagicMock()
        # Different tool calls each time to avoid duplicate_abort
        call_count = [0]
        def varying_chat(*args, **kwargs):
            call_count[0] += 1
            tc = [ToolCall(id=f"c{call_count[0]}", name="bash", arguments={"cmd": f"echo {call_count[0]}"})]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")
        client.chat.side_effect = varying_chat
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        with patch("llm_solver.harness.loop.dispatch", return_value="ok"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            result = session.run()

        assert result.finish_reason == "max_turns"
        assert result.turns == 2

    def test_session_length_response(self):
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=10)
        client = MagicMock()
        client.chat.return_value = make_turn_result(content="truncated...", finish_reason="length")
        client.build_assistant_message.return_value = {"role": "assistant", "content": "truncated..."}
        session = Session(cfg, client, "sys", "prompt", "/tmp")
        result = session.run()
        assert result.finish_reason == "length"
        assert result.done is False

    def test_session_uses_profile_token_estimator(self):
        from llm_solver.harness.loop import Session

        def estimate(_messages):
            return 42

        profile = type("Profile", (), {})()
        profile.estimate_tokens = estimate

        cfg = make_config(max_turns=10)
        client = MagicMock()
        client.__dict__["profile"] = profile
        client.chat.return_value = make_turn_result(content="done", finish_reason="stop")
        client.build_assistant_message.return_value = {"role": "assistant", "content": "done"}

        session = Session(cfg, client, "sys", "prompt", "/tmp")

        assert session.context._token_estimator is estimate
        assert session.context.estimate_tokens() == 42

    def test_local_tokenizer_drives_halflife_activation_with_tool_catalog(self):
        from llm_solver.harness.context_strategies import HalfLifeContext
        from llm_solver.harness.loop import Session

        def estimate(_messages):
            return 1

        class ExactTokenizer:
            def __init__(self):
                self.tool_counts = 0

            def count(self, messages, tools=None):
                if tools:
                    self.tool_counts += 1
                content_tokens = sum(
                    len(str(message.get("content", ""))) // 4
                    for message in messages
                )
                return content_tokens + (600 if tools else 0)

        profile = type("Profile", (), {})()
        profile.estimate_tokens = estimate
        client = MagicMock()
        client.__dict__["profile"] = profile
        tokenizer = ExactTokenizer()
        context = HalfLifeContext(
            context_size=1000,
            activation_ratio=0.50,
            verbatim_tool_results=1,
            cap_7_chars=40,
            token_estimator=estimate,
        )

        session = Session(
            make_config(context_size=1000),
            client,
            "sys",
            "prompt",
            "/tmp",
            context_manager=context,
            local_tokenizer=tokenizer,
        )
        session.context.add_assistant({"role": "assistant", "content": "one"})
        session.context.add_tool_result("call-1", "A" * 200)
        session.context.add_assistant({"role": "assistant", "content": "two"})
        session.context.add_tool_result("call-2", "B" * 200)

        tool_contents = [
            message["content"]
            for message in session.context.get_messages()
            if message.get("role") == "tool"
        ]
        assert tokenizer.tool_counts == 1
        assert "[halflife: omitted" in tool_contents[0]
        assert tool_contents[1] == "B" * 200

    def test_session_error_on_none_chat(self):
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=10)
        client = MagicMock()
        client.chat.side_effect = RuntimeError("fatal")
        client.build_assistant_message.return_value = {"role": "assistant", "content": ""}
        session = Session(cfg, client, "sys", "prompt", "/tmp")
        result = session.run()
        assert result.finish_reason == "error"

    def test_session_fails_fast_on_tool_surface_mismatch(self):
        from llm_solver.harness.loop import Session

        cfg = make_config(max_turns=10)
        client = MagicMock()

        bad = [{"type": "function", "function": {"name": "not_a_real_tool"}}]
        with patch("llm_solver.harness.loop.get_tool_schemas", return_value=bad):
            with pytest.raises(ValueError, match="Tool surface mismatch"):
                Session(cfg, client, "sys", "prompt", "/tmp")

    def test_session_applies_profile_max_tools_cap(self):
        """`done` is cap-immune.
        With cap=3, the result is `done` (immune) + the first 2 non-
        immune tools.
        """
        from llm_solver.harness.loop import Session
        from llm_solver.harness.schemas import get_tool_schemas

        cfg = make_config(max_turns=10)
        client = MagicMock()
        profile = type("Profile", (), {})()
        profile.max_tools = 3
        client.__dict__["profile"] = profile

        session = Session(cfg, client, "sys", "prompt", "/tmp")
        actual = [s["function"]["name"] for s in session._tool_schemas]
        assert "done" in actual, actual
        assert len(actual) == 3
        # Non-immune tools are taken from the head of the original
        # ordering, in original order.
        all_tools = [s["function"]["name"] for s in get_tool_schemas("minimal")]
        non_immune = [
            n for n in all_tools
            if n not in {"done", "bash_poll", "bash_kill"}
        ]
        assert [n for n in actual if n != "done"] == non_immune[:2]

    def test_session_applies_profile_simplify_schemas(self):
        from llm_solver.harness.loop import Session

        def _contains_description(value):
            if isinstance(value, dict):
                if "description" in value:
                    return True
                return any(_contains_description(v) for v in value.values())
            if isinstance(value, list):
                return any(_contains_description(v) for v in value)
            return False

        cfg = make_config(max_turns=10)
        client = MagicMock()
        profile = type("Profile", (), {})()
        profile.simplify_schemas = True
        client.__dict__["profile"] = profile

        session = Session(cfg, client, "sys", "prompt", "/tmp")
        assert all(not _contains_description(s) for s in session._tool_schemas)

    def test_session_uses_injected_tool_registry(self):
        from llm_solver.harness.loop import Session
        from llm_solver.harness.tools import build_tool_registry

        cfg = make_config(max_turns=1, duplicate_abort=10)
        client = MagicMock()
        tc = [ToolCall(id="c1", name="read", arguments={"path": "x.py"})]
        client.chat.return_value = make_turn_result(tool_calls=tc, finish_reason="tool_calls")
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}
        registry = build_tool_registry(
            overrides={"read": lambda _args, _cwd, _cfg: "REGISTRY_READ_OK"}
        )

        tool_results = []
        session = Session(
            cfg,
            client,
            "sys",
            "prompt",
            "/tmp",
            tool_registry=registry,
        )
        orig_add = session.context.add_tool_result

        def capture(cid, result, **kwargs):
            tool_results.append(result)
            return orig_add(cid, result, **kwargs)

        session.context.add_tool_result = capture
        session.run()
        assert any("REGISTRY_READ_OK" in r for r in tool_results)

    def test_session_fails_fast_when_injected_registry_missing_handler(self):
        from llm_solver.harness.loop import Session
        from llm_solver.harness.tools import ToolRegistry

        cfg = make_config(max_turns=10)
        client = MagicMock()
        bad_registry = ToolRegistry(handlers={"bash": lambda _a, _c, _f: "ok"})

        with pytest.raises(ValueError, match="Tool surface mismatch"):
            Session(cfg, client, "sys", "prompt", "/tmp", tool_registry=bad_registry)

    def test_adaptive_policy_switches_after_mutation(self):
        from llm_solver.harness.loop import Session

        cfg = make_config(
            max_turns=3,
            duplicate_abort=10,
            done_guard_enabled=False,
            adaptive_policy_enabled=True,
            adaptive_requires_mutation=True,
            adaptive_requires_test_signal=False,
            adaptive_phase2_done_guard_enabled=True,
        )
        client = MagicMock()
        turns = iter([
            make_turn_result(tool_calls=[ToolCall(id="c1", name="write", arguments={"path": "a.py", "content": "x"})], finish_reason="tool_calls"),
            make_turn_result(content="done", tool_calls=[], finish_reason="stop"),
        ])
        client.chat.side_effect = lambda *a, **k: next(turns)
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        with patch("llm_solver.harness.loop.dispatch", return_value="OK"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            result = session.run()

        assert result.finish_reason == "stop"
        assert session._adaptive_switched is True
        assert session.cfg.done_guard_enabled is True

    def test_adaptive_policy_respects_test_signal_gate(self):
        from llm_solver.harness.loop import Session

        cfg = make_config(
            max_turns=3,
            duplicate_abort=10,
            done_guard_enabled=False,
            adaptive_policy_enabled=True,
            adaptive_requires_mutation=True,
            adaptive_requires_test_signal=True,
            adaptive_phase2_done_guard_enabled=True,
        )
        client = MagicMock()
        turns = iter([
            make_turn_result(tool_calls=[ToolCall(id="c1", name="write", arguments={"path": "a.py", "content": "x"})], finish_reason="tool_calls"),
            make_turn_result(content="done", tool_calls=[], finish_reason="stop"),
        ])
        client.chat.side_effect = lambda *a, **k: next(turns)
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        with patch("llm_solver.harness.loop.dispatch", return_value="OK"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            session.run()

        assert session._adaptive_switched is False
        assert session.cfg.done_guard_enabled is False

    def test_adaptive_policy_accepts_successful_test_without_exit_marker(self):
        from llm_solver.harness.loop import Session

        cfg = make_config(
            max_turns=3,
            duplicate_abort=10,
            done_guard_enabled=False,
            adaptive_policy_enabled=True,
            adaptive_requires_mutation=True,
            adaptive_requires_test_signal=True,
            adaptive_phase2_done_guard_enabled=True,
        )
        client = MagicMock()
        turns = iter([
            make_turn_result(
                tool_calls=[ToolCall(id="c1", name="write", arguments={"path": "a.py", "content": "x"})],
                finish_reason="tool_calls",
            ),
            make_turn_result(
                tool_calls=[ToolCall(id="c2", name="bash", arguments={"cmd": "python3 -m pytest tests/test_app.py -v"})],
                finish_reason="tool_calls",
            ),
            make_turn_result(content="done", tool_calls=[], finish_reason="stop"),
        ])
        client.chat.side_effect = lambda *a, **k: next(turns)
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        def dispatch_stub(name, args, **kwargs):
            if name == "bash":
                return "============================== 1 passed ==============================\n"
            return "OK"

        with patch("llm_solver.harness.loop.dispatch", side_effect=dispatch_stub):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            result = session.run()

        assert result.finish_reason == "stop"
        assert session._observed_test_signal is True
        assert session._adaptive_switched is True
        assert session.cfg.done_guard_enabled is True

    def test_adaptive_policy_switches_task_format_transform_gating(self):
        from llm_solver.harness.loop import Session

        cfg = make_config(
            max_turns=3,
            duplicate_abort=10,
            bash_transforms_task_format_enabled=False,
            adaptive_policy_enabled=True,
            adaptive_requires_mutation=True,
            adaptive_requires_test_signal=False,
            adaptive_phase2_bash_task_format_enabled=True,
        )
        client = MagicMock()
        turns = iter([
            make_turn_result(tool_calls=[ToolCall(id="c1", name="write", arguments={"path": "a.py", "content": "x"})], finish_reason="tool_calls"),
            make_turn_result(tool_calls=[ToolCall(id="c2", name="bash", arguments={"cmd": "echo hi"})], finish_reason="tool_calls"),
            make_turn_result(content="done", tool_calls=[], finish_reason="stop"),
        ])
        client.chat.side_effect = lambda *a, **k: next(turns)
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        seen_output_control = []

        def _capture_dispatch(_name, _arguments, *, cwd, cfg, output_control=None,
                              universal_rewrites=None, tool_registry=None, **_unused):
            # This test inspects output_control and ignores other keywords.
            seen_output_control.append(output_control)
            return "OK"

        with patch("llm_solver.harness.loop.dispatch", side_effect=_capture_dispatch):
            session = Session(
                cfg,
                client,
                "sys",
                "prompt",
                "/tmp",
                output_control=object(),
            )
            session.run()

        assert seen_output_control[0] is None  # before switch
        assert seen_output_control[1] is not None  # after switch


# ──────────────────────────────────────────────
# 8. solve_task end-to-end (mock client)
# ──────────────────────────────────────────────
