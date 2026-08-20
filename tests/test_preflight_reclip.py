"""Pre-flight overflow backstop + system log.

Pre-fix, a single oversized tool result appended at the tail of a turn
ended the session context_full at the next pre-flight ("single-turn
overflow death"). The backstop re-clips that one message in token
space (head+tail, ctx/2-token budget, visible notice), re-projects
once, and only ends the session when the projection STILL exceeds the
window. Every gate firing lands in the per-run system_log.jsonl — the
harness talking about itself, separate from .trace.jsonl.

state.json semantics are untouched: the backstop edits only the in-memory message list via
ContextManager.replace_all_messages, never the trace or the state
projection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from _config_helpers import make_config
from llm_solver.harness.context import FullTranscript
from llm_solver.harness.loop import Session
from llm_solver.harness.system_log import (
    close_system_log,
    command_shape,
    observe_tool_result,
    open_system_log,
)
from llm_solver.harness._loop.compaction import preflight_reclip_oversized
from llm_solver.server.types import TurnResult, Usage


@pytest.fixture()
def syslog(tmp_path):
    """Open a system log for the test; yield its path; always reset."""
    path = tmp_path / "system_log.jsonl"
    open_system_log(path)
    yield path
    close_system_log()


def _events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _stub_session(context_size: int, messages: list[dict]) -> Session:
    sess = Session.__new__(Session)
    sess.cfg = SimpleNamespace(context_size=context_size)
    sess.context = FullTranscript()
    sess.context.replace_all_messages(messages)
    return sess


def _make_turn_result(content="ok", finish_reason="stop"):
    return TurnResult(content=content, tool_calls=[],
                      finish_reason=finish_reason,
                      usage=Usage(prompt_tokens=10, completion_tokens=5))


# ── preflight_reclip_oversized (unit) ──────────────────────────────


def test_reclip_clips_largest_oversized_message_with_notice():
    huge = "line of output\n" * 2000  # ~30k chars ≈ 7.5k est tokens
    sess = _stub_session(1000, [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task prompt"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "tool_call_id": "c9", "content": huge},
    ])
    before = sess.context.estimate_tokens()
    info = preflight_reclip_oversized(sess)
    assert info is not None
    assert info["index"] == 3
    assert info["tool_call_id"] == "c9"
    assert info["orig_pt"] > 1000 // 2
    assert info["new_pt"] <= 1000 // 2
    msgs = sess.context.get_messages()
    clipped = msgs[3]["content"]
    # Visible notice where content was removed: original token size,
    # what is shown, and the advise-narrower-command line.
    assert "HARNESS re-clip" in clipped
    assert f"~{info['orig_pt']} tokens" in clipped
    assert "head and tail are shown" in clipped
    assert "Re-run a narrower command" in clipped
    # Head + tail survive around the notice.
    assert clipped.startswith("line of output")
    assert clipped.rstrip().endswith("]") or "line of output" in clipped[-200:]
    # Projection actually shrank; other messages untouched.
    assert sess.context.estimate_tokens() < before
    assert msgs[0]["content"] == "sys"
    assert msgs[1]["content"] == "task prompt"
    assert msgs[2]["role"] == "assistant"


def test_reclip_never_touches_initial_user_or_assistant():
    big_task = "t" * 8000  # initial task prompt — protected
    sess = _stub_session(1000, [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": big_task},
        {"role": "assistant", "content": "a" * 8000, "tool_calls": []},
    ])
    assert preflight_reclip_oversized(sess) is None
    msgs = sess.context.get_messages()
    assert msgs[1]["content"] == big_task
    assert len(msgs[2]["content"]) == 8000


def test_reclip_noop_when_no_message_exceeds_half_context():
    sess = _stub_session(100000, [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "tool", "tool_call_id": "c1", "content": "x" * 5000},
    ])
    assert preflight_reclip_oversized(sess) is None


def test_reclip_bails_when_strategy_cannot_replace():
    sess = _stub_session(1000, [])
    ctx = SimpleNamespace(
        get_messages=lambda: [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "tool", "tool_call_id": "c1", "content": "x" * 30000},
        ],
        replace_all_messages=lambda new: False,
    )
    sess.context = ctx
    assert preflight_reclip_oversized(sess) is None


# ── Loop-level: session survives / still ends (integration) ───────


def test_oversized_message_reclipped_and_session_survives(syslog):
    cfg = make_config(max_turns=5, context_size=8192, context_fill_ratio=0.85)
    client = MagicMock()
    client.chat.return_value = _make_turn_result()
    with patch.object(Session, "_get_server_ctx", return_value=0):
        session = Session(cfg, client, "sys", "task prompt", "/tmp")
        # One oversized tool result (~10k est tokens > 0.85*8192) whose
        # clipped form (ctx/2 = 4096 tokens) fits the projection again.
        session.context.add_tool_result("c1", "y" * 40000, tool_name="bash")
        result = session.run()
    # Pre-fix this ended context_full before any API call. Now: the
    # message is re-clipped, the retry projection fits, chat runs, and
    # the model's stop ends the session normally.
    assert result.finish_reason == "stop"
    assert result.done is True
    assert client.chat.call_count == 1
    clipped = [m for m in session.context.get_messages()
               if m.get("role") == "tool" and "HARNESS re-clip" in str(m.get("content"))]
    assert len(clipped) == 1
    events = _events(syslog)
    overflow = [e for e in events if e["type"] == "preflight_overflow"]
    assert len(overflow) == 1
    ev = overflow[0]
    assert ev["action"] == "reclipped"
    assert ev["ctx"] == 8192
    assert ev["preflight_pt"] <= int(8192 * 0.85)
    for field in ("ts", "turn", "live_pt", "estimate_pt", "density",
                  "command_shape", "quirk_hit"):
        assert field in ev


def test_still_too_big_ends_context_full_with_event(syslog):
    # Tiny window: even the clipped message (notice included) stays
    # over budget, so the legacy behavior is preserved: context_full
    # before any API call, with a session_end event in the system log.
    cfg = make_config(max_turns=5, context_size=100, context_fill_ratio=0.5)
    client = MagicMock()
    client.chat.side_effect = AssertionError("chat called despite over-budget context")
    with patch.object(Session, "_get_server_ctx", return_value=0):
        session = Session(cfg, client, "sys", "task", "/tmp")
        session.context.add_tool_result("c1", "x" * 1000, tool_name="bash")
        result = session.run()
    assert result.finish_reason == "context_full"
    assert client.chat.call_count == 0
    overflow = [e for e in _events(syslog) if e["type"] == "preflight_overflow"]
    assert len(overflow) == 1
    assert overflow[0]["action"] == "session_end"


def test_reclip_disabled_keeps_legacy_end(syslog):
    cfg = make_config(max_turns=5, context_size=8192, context_fill_ratio=0.85,
                      preflight_reclip_enabled=False)
    client = MagicMock()
    client.chat.side_effect = AssertionError("chat called despite over-budget context")
    with patch.object(Session, "_get_server_ctx", return_value=0):
        session = Session(cfg, client, "sys", "task", "/tmp")
        session.context.add_tool_result("c1", "y" * 40000, tool_name="bash")
        result = session.run()
    assert result.finish_reason == "context_full"
    assert client.chat.call_count == 0
    # No message was touched.
    assert not any("HARNESS re-clip" in str(m.get("content"))
                   for m in session.context.get_messages())
    overflow = [e for e in _events(syslog) if e["type"] == "preflight_overflow"]
    assert len(overflow) == 1
    assert overflow[0]["action"] == "session_end"


# ── Density blowout / oversized result observation ─────────────────


class _InflatingTokenizer:
    """Fake tokenizer whose real counts triple the chars/4 estimate."""

    def count(self, msgs):
        return sum(3 * (len(str(m.get("content", ""))) // 4) for m in msgs)


def test_density_blowout_event_even_when_it_fits(syslog):
    sess = Session.__new__(Session)
    sess.cfg = SimpleNamespace(context_size=10_000_000)  # plenty of room
    sess._tokenizer = _InflatingTokenizer()
    result = "z" * 8000  # est 2000, "real" 6000 → density 3.0 > 2.0
    observe_tool_result(sess, "c1", "bash",
                        {"cmd": "grep -rn pattern /repo/src"}, result,
                        quirk_hit=False, turn=7)
    events = _events(syslog)
    blowouts = [e for e in events if e["type"] == "density_blowout"]
    assert len(blowouts) == 1
    ev = blowouts[0]
    assert ev["turn"] == 7
    assert ev["density"] == 3.0
    assert ev["live_pt"] == 6000
    assert ev["estimate_pt"] == 2000
    assert ev["action"] == "none"
    assert ev["command_shape"] == "grep -rn"  # arguments stripped
    # Fits comfortably → no oversized_result alongside.
    assert not [e for e in events if e["type"] == "oversized_result"]


def test_oversized_result_event_when_over_half_context(syslog):
    sess = Session.__new__(Session)
    sess.cfg = SimpleNamespace(context_size=8000)  # half-ctx = 4000 tokens
    sess._tokenizer = _InflatingTokenizer()
    result = "z" * 8000  # "real" 6000 tokens > 4000
    observe_tool_result(sess, "c2", "bash", {"cmd": "cat /repo/big.log"},
                        result, quirk_hit=True, turn=3)
    oversized = [e for e in _events(syslog) if e["type"] == "oversized_result"]
    assert len(oversized) == 1
    assert oversized[0]["quirk_hit"] is True
    assert oversized[0]["command_shape"] == "cat"


def test_no_density_check_without_tokenizer(syslog):
    # chars/4 is the only count available — a per-result "real" count is
    # unknowable, so no event is fabricated.
    sess = Session.__new__(Session)
    sess.cfg = SimpleNamespace(context_size=100)
    observe_tool_result(sess, "c1", "bash", {"cmd": "ls"}, "z" * 8000,
                        quirk_hit=False, turn=1)
    assert _events(syslog) == []


# ── command_shape: binary + flags only, arguments stripped ─────────


def test_command_shape_strips_arguments_and_paths():
    shape = command_shape("bash", {
        "cmd": "grep -rn --max-count=100 'secret pattern' /home/user/repo | head -5"})
    assert shape == "grep -rn --max-count | head -5"
    assert "/" not in shape
    assert "secret" not in shape


def test_command_shape_env_prefix_and_basename():
    shape = command_shape("bash", {
        "cmd": "PYTHONPATH=/repo /usr/bin/python3 -m pytest tests/"})
    assert shape == "python3 -m"


def test_command_shape_non_bash_is_tool_name():
    assert command_shape("read", {"path": "/etc/passwd"}) == "read"


def test_system_log_events_carry_no_free_text(syslog):
    # The event schema is closed: nothing from the tool result body or
    # the command's arguments may appear in the file.
    sess = Session.__new__(Session)
    sess.cfg = SimpleNamespace(context_size=8000)
    sess._tokenizer = _InflatingTokenizer()
    observe_tool_result(sess, "c1", "bash",
                        {"cmd": "cat /repo/SENSITIVE_FILE.txt"},
                        "TOPSECRETPAYLOAD " * 1000, quirk_hit=False, turn=1)
    raw = Path(syslog).read_text()
    assert "TOPSECRETPAYLOAD" not in raw
    assert "SENSITIVE_FILE" not in raw
