"""Completion accepts native custom checks without claiming task correctness."""
import io
import json
import shlex
import sys
from unittest.mock import MagicMock

import pytest

from _config_helpers import make_config
from llm_solver.config import load_config
from llm_solver.harness.loop import Session
from llm_solver.server.types import ToolCall, TurnResult, Usage


@pytest.mark.parametrize("kind,required,enabled,accepted,status,rewrite", [
    ("custom", 2, True, True, "custom_passed", False),
    ("custom", 3, True, False, "custom_failed", False),
    ("registered", 3, True, True, "passed", False),
    ("custom", 3, False, True, "custom_failed", False),
    ("custom", 2, True, False, "custom_passed", True),
])
def test_real_session_uses_native_checks_without_inventing_suite_requirement(
    tmp_path, monkeypatch, kind, required, enabled, accepted, status, rewrite,
):
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    (tmp_path / "core.py").write_text("VALUE = 1\n")
    (tmp_path / "test_unrelated.py").write_text(
        "import unittest\nclass Unrelated(unittest.TestCase):\n"
        "    def test_arithmetic(self):\n        self.assertEqual(1 + 1, 2)\n"
    )
    custom = f"-c 'import core; assert core.VALUE == {required}'"
    command = custom if kind == "custom" else "-m unittest test_unrelated -q"
    if rewrite:
        # The check passes, then changes the previously edited input. That
        # outcome must not grant verification credit for the changed input.
        command = "-c 'import core; assert core.VALUE == 2; " \
                  "from pathlib import Path; Path(\"core.py\").write_text(\"VALUE = 3\\n\")'"
    cfg = make_config(
        max_turns=3, sandbox_bash=False, done_guard_enabled=False,
        done_loop_abort_after=0, loop_detect_enabled=False,
        duplicate_guard_enabled=False, post_mutation_verification_gate_after=3 if enabled else 0,
        done_reject_no_formal_verification=load_config().done_reject_no_formal_verification,
    )
    calls = [
        ToolCall(id="w", name="write", arguments={"path": "core.py", "content": "VALUE = 2\n"}),
        ToolCall(id="v", name="bash", arguments={"cmd": shlex.quote(sys.executable) + " " + command}),
        ToolCall(id="d", name="done", arguments={"summary": "done"}),
    ]
    client = MagicMock()
    client.chat.side_effect = [TurnResult(
        content=None, tool_calls=[call], finish_reason="tool_calls",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    ) for call in calls]
    client.build_assistant_message.return_value = {"role": "assistant", "content": None}
    trace = io.StringIO()
    task = f"Set core.VALUE to {required}. Verify with python {custom}."
    session = Session(cfg, client, "Follow the task.", task, str(tmp_path), trace_file=trace)
    result = session.run()
    assert result.done is accepted
    events = [json.loads(line) for line in trace.getvalue().splitlines()]
    tools = [row for row in events if row["event"] == "tool_call"]
    assert tools[1]["verification_status"] == status
    changes = tools[1]["file_changes"]
    assert changes["status"] == "changed"
    assert any(path.endswith(".pyc") for path in changes["changed_paths"])
    assert ("core.py" in changes["changed_paths"]) is rewrite
    if kind == "registered":
        # Completion is not a semantic certificate: the unrelated check passed,
        # but this fixture deliberately did not meet the requested value.
        assert (tmp_path / "core.py").read_text() == "VALUE = 2\n"


def test_failed_custom_rerun_invalidates_previous_check():
    from llm_solver.harness._guardrails.checks_post import mark_bash_verified
    from llm_solver.harness._guardrails.state import GuardrailState
    state = GuardrailState(has_mutated=True)
    for status, code in (("custom_passed", 0), ("custom_failed", 1)):
        mark_bash_verified(state, make_config(), tc_name="bash", result="printed success",
                           gate_blocked=False, execution_metadata={
                               "executed": True, "exit_status_known": True,
                               "exit_status": code, "verification_status": status,
                           })
        assert state.verified_since_mutation is (code == 0)


@pytest.mark.parametrize("case", ["plain", "report", "edited_report"])
def test_repeated_check_outputs_do_not_become_edited_inputs(tmp_path, case):
    (tmp_path / "core.txt").write_text("1")
    program = "from pathlib import Path; assert Path('core.txt').read_text() == '2'"
    if case != "plain":
        program += "; p = Path('run-note.txt'); p.write_text(p.read_text() + '.' if p.exists() else '.')"
    command = shlex.join([sys.executable, "-c", program])
    calls = [ToolCall("edit", "write", {"path": "core.txt", "content": "2"}),
             ToolCall("check1", "bash", {"cmd": command})]
    if case == "edited_report":
        calls.append(ToolCall("edit_report", "write", {"path": "run-note.txt", "content": "edited"}))
    calls += [ToolCall("check2", "bash", {"cmd": command}),
              ToolCall("done", "done", {"summary": "finished"})]
    cfg = make_config(max_turns=len(calls), sandbox_bash=False, auto_commit=False,
                      done_guard_enabled=False, done_loop_abort_after=0,
                      loop_detect_enabled=False, duplicate_guard_enabled=False,
                      post_mutation_verification_gate_after=3)
    captured = []

    def chat(*args, **kwargs):
        captured.append((session._guards.verified_since_mutation,
                         dict(session._guards.verification_file_revisions)))
        return TurnResult(content=None, tool_calls=[calls[len(captured) - 1]],
                          finish_reason="tool_calls",
                          usage=Usage(prompt_tokens=10, completion_tokens=5))

    client = MagicMock()
    client.chat.side_effect = chat
    client.build_assistant_message.return_value = {"role": "assistant", "content": None}
    trace = io.StringIO()
    session = Session(cfg, client, "Follow the task.", "Set core.txt to 2 and check it.",
                      str(tmp_path), trace_file=trace)
    result = session.run()
    assert result.done is (case != "edited_report")
    assert captured[2][0]  # First check retained credit.
    assert set(captured[2][1]) == {"core.txt"}
    assert captured[-1][0] is (case != "edited_report")
    assert ("run-note.txt" in captured[-1][1]) is (case == "edited_report")
    assert (tmp_path / "core.txt").read_text() == "2"
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    checks = [row for row in rows if row.get("event") == "tool_call"
              and row.get("tool_call_id") in {"check1", "check2"}]
    assert len(checks) == 2
    assert all(row["verification_status"] == "custom_passed" for row in checks)
    for row in checks:
        assert row["file_changes"]["changed_paths"] == ([] if case == "plain" else ["run-note.txt"])
