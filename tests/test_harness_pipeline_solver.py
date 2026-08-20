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

class TestHarnessSolver:

    def test_collect_pending(self, tmp_path):
        from llm_solver.harness.solver import collect_pending
        repos = tmp_path / "repos"
        repos.mkdir()
        # Task 1: pending
        t1 = repos / "task1"
        t1.mkdir()
        (t1 / "prompt.txt").write_text("do something")
        # Task 2: completed
        t2 = repos / "task2"
        t2.mkdir()
        (t2 / "prompt.txt").write_text("do something else")
        (t2 / "checkpoint.json").write_text('{"status":"completed"}')
        # Task 3: no prompt
        t3 = repos / "task3"
        t3.mkdir()

        pending = collect_pending(tmp_path)
        assert len(pending) == 1
        assert pending[0].name == "task1"

    def test_collect_pending_no_repos_dir(self, tmp_path):
        from llm_solver.harness.solver import collect_pending
        with pytest.raises(FileNotFoundError):
            collect_pending(tmp_path)

    # check_done removed — completion is now model-signaled (stop with no tool calls)

    def test_write_checkpoint(self, tmp_path):
        from llm_solver.harness.solver import write_checkpoint
        write_checkpoint(tmp_path, "test-model", "completed")
        cp = json.loads((tmp_path / "checkpoint.json").read_text())
        assert cp["status"] == "completed"
        assert cp["model"] == "test-model"
        assert cp["solver"] == "llm_solver"

    def test_build_system_prompt_default(self):
        from llm_solver.harness.solver import build_system_prompt
        header = "You are a software engineering solver."
        prompt = build_system_prompt(header)
        assert "solver" in prompt.lower()
        assert prompt == header

    def test_build_system_prompt_with_file(self, tmp_path):
        from llm_solver.harness.solver import build_system_prompt
        header = "You are a software engineering solver."
        proto = tmp_path / "protocol.md"
        proto.write_text("Follow these rules.")
        prompt = build_system_prompt(header, proto)
        assert "Follow these rules." in prompt
        assert "solver" in prompt.lower()


# ──────────────────────────────────────────────
# 5. Cross-session learning (build_resume_prompt)
# ──────────────────────────────────────────────

class TestCrossSessionLearning:

    def _make_session(self, cfg=None):
        from llm_solver.harness.loop import Session, SessionResult
        cfg = cfg or make_config()
        client = MagicMock()
        client.build_assistant_message.return_value = {"role": "assistant", "content": "ok"}
        session = Session(cfg, client, "system", "initial", "/tmp")
        return session

    def test_resume_prompt_duplicate_abort(self):
        from llm_solver.harness.loop import build_resume_prompt, SessionResult
        cfg = make_config()
        session = self._make_session(cfg)
        session._tool_log = [("bash", "cmd='ls'"), ("bash", "cmd='ls'"), ("bash", "cmd='ls'")]
        result = SessionResult(turns=10, finish_reason="duplicate_abort", done=False, total_prompt_tokens=500)
        prompt = build_resume_prompt(result, session, cfg)
        assert "duplicate_abort" in prompt
        assert "identical" in prompt.lower()
        assert "bash" in prompt

    def test_resume_prompt_context_full(self):
        from llm_solver.harness.loop import build_resume_prompt, SessionResult
        cfg = make_config()
        session = self._make_session(cfg)
        session._last_fill = 0.92
        result = SessionResult(turns=30, finish_reason="context_full", done=False, total_prompt_tokens=8000)
        prompt = build_resume_prompt(result, session, cfg)
        assert "92%" in prompt
        assert "full" in prompt.lower()

    def test_resume_prompt_max_turns(self):
        from llm_solver.harness.loop import build_resume_prompt, SessionResult
        cfg = make_config()
        session = self._make_session(cfg)
        session._tool_log = [("read", "path='a.py'"), ("edit", "path='a.py'"), ("bash", "cmd='pytest'")]
        result = SessionResult(turns=60, finish_reason="max_turns", done=False, total_prompt_tokens=10000)
        prompt = build_resume_prompt(result, session, cfg)
        assert "max_turns" in prompt

    def test_resume_prompt_length(self):
        from llm_solver.harness.loop import build_resume_prompt, SessionResult
        cfg = make_config()
        session = self._make_session(cfg)
        result = SessionResult(turns=5, finish_reason="length", done=False, total_prompt_tokens=200)
        prompt = build_resume_prompt(result, session, cfg)
        assert "truncated" in prompt.lower()

    def test_resume_prompt_always_has_base(self):
        from llm_solver.harness.loop import build_resume_prompt, SessionResult
        cfg = make_config()
        session = self._make_session(cfg)
        result = SessionResult(turns=1, finish_reason="context_full", done=False)
        prompt = build_resume_prompt(result, session, cfg)
        assert cfg.resume_base in prompt


# ──────────────────────────────────────────────
# 6. Error taxonomy and transient retry
# ──────────────────────────────────────────────

class TestErrorTaxonomy:

    def test_transient_retry_succeeds(self):
        from llm_solver.harness.loop import Session
        cfg = make_config()
        client = MagicMock()
        tr = make_turn_result(content="ok")
        # Fail once, then succeed
        client.chat.side_effect = [
            openai.APIConnectionError(request=MagicMock()),
            tr,
        ]
        client.build_assistant_message.return_value = {"role": "assistant", "content": "ok"}
        session = Session(cfg, client, "sys", "user msg", "/tmp")
        with patch("llm_solver.harness.loop.time.sleep"):  # skip actual sleep
            result = session._chat_with_retry(0)
        assert result is not None
        assert result.content == "ok"

    def test_transient_retry_exhausted(self):
        from llm_solver.harness.loop import Session
        cfg = make_config()
        client = MagicMock()
        client.chat.side_effect = openai.APIConnectionError(request=MagicMock())
        client.build_assistant_message.return_value = {"role": "assistant", "content": ""}
        session = Session(cfg, client, "sys", "user msg", "/tmp")
        with patch("llm_solver.harness.loop.time.sleep"):
            result = session._chat_with_retry(0)
        assert result is None
        assert client.chat.call_count == cfg.max_transient_retries + 1

    def test_fatal_error_no_retry(self):
        from llm_solver.harness.loop import Session
        cfg = make_config()
        client = MagicMock()
        client.chat.side_effect = RuntimeError("unexpected")
        client.build_assistant_message.return_value = {"role": "assistant", "content": ""}
        session = Session(cfg, client, "sys", "user msg", "/tmp")
        result = session._chat_with_retry(0)
        assert result is None
        assert client.chat.call_count == 1  # no retry


# ──────────────────────────────────────────────
# 7. Session.run() integration
# ──────────────────────────────────────────────

