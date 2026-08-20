"""Output dedup: byte-identical tool repeats collapsed.

The same read(P) on one file with no intervening mutation once appended the same bytes
verbatim N times to the rolling tool-result window — a major chunk
of context-fill thrash. Now the second and later identical results
collapse to a one-line back-reference and the chars saved land in
the savings ledger under bucket="output_dedup".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from _config_helpers import make_config
from llm_solver.harness.loop import Session
from llm_solver.server.types import ToolCall, TurnResult, Usage


def _turn(tool_calls=None, content=None):
    return TurnResult(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="stop",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


def _read_tc(path: str) -> ToolCall:
    return ToolCall(id=f"call_{path}", name="read", arguments={"path": path})


def test_identical_read_collapses_to_back_reference(tmp_path):
    cfg = make_config(max_turns=4, max_sessions=1, tools_output_dedup_enabled=True)
    client = MagicMock()
    # Turn 1: read foo.py. Turn 2: read foo.py again. Turn 3: stop.
    client.chat.side_effect = [
        _turn(tool_calls=[_read_tc("foo.py")]),
        _turn(tool_calls=[_read_tc("foo.py")]),
        _turn(content="done"),
    ]
    client.build_assistant_message.return_value = {"role": "assistant", "content": ""}

    captured = []

    def fake_dispatch(tc_name, tc_args, *, cwd, cfg, **_kw):
        return f"FILE_BODY {tc_args.get('path')}"

    def capture_add(cid, result, **kwargs):
        captured.append(result)

    sess = Session(cfg, client, "sys", "prompt", str(tmp_path))
    sess.context.add_tool_result = capture_add

    with patch("llm_solver.harness.loop.dispatch", side_effect=fake_dispatch), \
         patch("llm_solver.harness.loop._auto_commit"):
        sess.run()

    # Two read results captured: first is the real body, second is the
    # back-ref.
    read_results = [r for r in captured if "FILE_BODY" in r or "harness: identical" in r]
    assert len(read_results) == 2, captured
    assert "FILE_BODY foo.py" in read_results[0]
    assert "harness: identical to turn" in read_results[1]
    assert "foo.py" in read_results[1]


def test_disabled_knob_keeps_duplicates(tmp_path):
    cfg = make_config(max_turns=4, max_sessions=1, tools_output_dedup_enabled=False)
    client = MagicMock()
    client.chat.side_effect = [
        _turn(tool_calls=[_read_tc("foo.py")]),
        _turn(tool_calls=[_read_tc("foo.py")]),
        _turn(content="done"),
    ]
    client.build_assistant_message.return_value = {"role": "assistant", "content": ""}

    captured = []

    def fake_dispatch(tc_name, tc_args, *, cwd, cfg, **_kw):
        return f"FILE_BODY {tc_args.get('path')}"

    def capture_add(cid, result, **kwargs):
        captured.append(result)

    sess = Session(cfg, client, "sys", "prompt", str(tmp_path))
    sess.context.add_tool_result = capture_add

    with patch("llm_solver.harness.loop.dispatch", side_effect=fake_dispatch), \
         patch("llm_solver.harness.loop._auto_commit"):
        sess.run()

    read_results = [r for r in captured if "FILE_BODY" in r]
    # Both reads keep their full body — no collapse.
    assert read_results.count("FILE_BODY foo.py") == 2
    # No back-ref injected.
    assert all("harness: identical" not in r for r in captured)


def test_savings_ledger_records_dedup(tmp_path):
    cfg = make_config(max_turns=4, max_sessions=1, tools_output_dedup_enabled=True)
    client = MagicMock()
    client.chat.side_effect = [
        _turn(tool_calls=[_read_tc("foo.py")]),
        _turn(tool_calls=[_read_tc("foo.py")]),
        _turn(content="done"),
    ]
    client.build_assistant_message.return_value = {"role": "assistant", "content": ""}

    def fake_dispatch(tc_name, tc_args, *, cwd, cfg, **_kw):
        return "X" * 5000 + "\n" + tc_args.get("path", "")

    from llm_solver.harness.savings import open_ledger, close_ledger
    ledger_path = tmp_path / "_savings.jsonl"
    open_ledger(ledger_path)
    try:
        sess = Session(cfg, client, "sys", "prompt", str(tmp_path))
        with patch("llm_solver.harness.loop.dispatch", side_effect=fake_dispatch), \
             patch("llm_solver.harness.loop._auto_commit"):
            sess.run()
    finally:
        close_ledger()

    assert ledger_path.exists()
    rows = [json.loads(l) for l in ledger_path.read_text().splitlines() if l.strip()]
    dedup_rows = [r for r in rows if r.get("bucket") == "output_dedup"]
    assert len(dedup_rows) >= 1
    row = dedup_rows[0]
    assert row["mechanism"] == "read"
    assert row["input_chars"] > row["output_chars"]
