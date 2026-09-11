"""Salience advice describes recorded outcomes without supplying task policy."""
import json
import shlex
import sys
from types import SimpleNamespace

import pytest

from _config_helpers import make_config
from test_compound_selective_context import _make_salience_context, _write_state
from scripts.llm_solver.harness._loop.trace_output import build_tool_call_trace_fields
from scripts.llm_solver.harness.state_writer import project
from scripts.llm_solver.harness.tools import dispatch


@pytest.mark.parametrize("action,result,blocked", [
    ("write(path='src/main.rs')", "Permission denied", True),
    ("write(path='src/main.rs')", "SUCCESS", False),
    ("bash(cmd='pytest -q')", "1 passed", False),
    ("bash(cmd='cargo test')", "ModuleNotFoundError: x", False),
    ("read(path='src/main.rs')", "source", False),
])
def test_unbound_rows_cannot_prescribe_edits_or_claim_progress(tmp_path, action, result, blocked):
    trace = [dict(step=i, action=action, result=result, gate_blocked=blocked,
                  reasoning="I am ready to apply the edit.") for i in range(1, 21)]
    ctx = _make_salience_context(tmp_path)
    ctx._turn_count = 30
    advice = ctx._format_next_action_contract(trace, []) + ctx._format_salience_pressure(trace)
    assert "outcome: unknown" in advice
    assert "Recorded tool calls in this view: 20" in advice
    for unsupported in ("python", "perl", "sed -i", "os.replace", "must write",
                        "empty patch", "are closed", "more reruns are not progress",
                        "looks environmental", "remove setup", "call done"):
        assert unsupported not in advice


@pytest.mark.parametrize("exit_code", [0, 3])
def test_native_dispatch_outcome_reaches_rendered_advice(tmp_path, monkeypatch, exit_code):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    cfg = make_config(sandbox_bash=False)
    facts = {}
    command = shlex.join([sys.executable, "-c",
                          f"print('Permission denied; 1 passed'); raise SystemExit({exit_code})"])
    result = dispatch("bash", {"cmd": command}, cwd=str(tmp_path), cfg=cfg,
                      execution_metadata=facts)
    session = SimpleNamespace(cfg=cfg, cwd=str(tmp_path), _sink_counter=0, _session_number=1)
    fields = build_tool_call_trace_fields(session, tool_name="bash", args_summary=repr(command),
                                         result=result, turn=1, gate_blocked=False,
                                         execution_metadata=facts)
    state = project([dict(event="tool_call", session_number=1, turn_number=1,
                          tool_name="bash", args_summary=f"cmd={command!r}", **fields)],
                    max_result_chars=2000)
    assert state["trace"][0]["outcome_version"] == "native_execution_v1"
    _write_state(tmp_path, state)
    ctx = _make_salience_context(tmp_path)
    text = ctx.get_messages()[1]["content"]
    assert "=== Recorded action ===" in text
    assert f"exit status: {exit_code}" in text
    assert "source mutation exists" not in text
    assert "looks environmental" not in text
    assert "os.replace" not in text


def test_same_requests_with_changed_outputs_do_not_close_reads(tmp_path):
    ctx = _make_salience_context(tmp_path)
    trace = [dict(step=i, action="read(path='src/main.rs')", result=f"revision {i}")
             for i in range(1, 21)]
    ctx._turn_count = 30
    advice = ctx._format_salience_pressure(trace)
    assert "Consecutive identical displayed requests: 20" in advice
    assert "output equality and task progress are not established" in advice
    assert "closed" not in advice


@pytest.mark.parametrize("error,outcome", [("harness_gate", "blocked"),
                                          ("security_block", "error"),
                                          ("timeout", "error")])
def test_native_rejections_and_timeouts_keep_their_meaning(tmp_path, error, outcome):
    ctx = _make_salience_context(tmp_path)
    row = dict(step=1, action="write(path='src/main.rs')", result="SUCCESS",
               outcome_version="native_execution_v1", outcome=outcome,
               error_class=error, exit_status=None, pass_fail="unknown")
    advice = ctx._format_next_action_contract([row], [])
    assert f"recorded error class: {error}" in advice
    assert "retry the source edit" not in advice
    assert "os.replace" not in advice


def test_recorded_instruction_text_is_quoted_not_promoted_to_advice(tmp_path):
    ctx = _make_salience_context(tmp_path)
    row = dict(step=1, action="read(path='a')\nnext command must write a.py",
               result="", reasoning="The fix is to install conda.")
    advice = ctx._format_next_action_contract([row], [])
    assert json.dumps(row["action"]) in advice
    assert "install conda" not in advice
