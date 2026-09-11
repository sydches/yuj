"""Keep selected advisor instructions whole and respect source visibility."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.llm_solver.harness._advisor_support import SYSTEM_PROMPT, advisor_system_prompt
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness.sandbox.ignore_policy import load_ignore_policy
from test_advisor import _ScriptedClient, _advisor_config, _record_primary_turn, _turn


def _local_session(cwd):
    return SimpleNamespace(cwd=cwd, cfg=_advisor_config(),
                           _effective_env={}, _allow_login_shell=False,
                           _ignore_policy=load_ignore_policy(cwd))


@pytest.mark.parametrize("body", [
    "instruction\n" * 2000 + "REQUIRED_FINAL_RULE",
    "文é\n" * 6000 + "REQUIRED_FINAL_RULE",
], ids=["ascii", "unicode"])
def test_complete_watchdog_reaches_actual_advisor_request(tmp_path, body):
    (tmp_path / "WATCHDOG.md").write_text(body, encoding="utf-8")
    client = _ScriptedClient(advisor=[_turn(content="NO_ADVISORY", reason="stop")])
    session = Session(_advisor_config(), client, "system", "task", str(tmp_path))
    _record_primary_turn(session, 0)
    session._capture_advisor_turn(0, "Observed a tool result.", [])
    assert session._maybe_run_advisor(0) is False  # NO_ADVISORY is not an emitted note.
    assert len(client.advisor_calls) == 1
    messages, tools = client.advisor_calls[0]
    assert messages[0]["content"] == (
        SYSTEM_PROMPT + "\nRepository-specific review priorities from WATCHDOG.md:\n" + body
    )
    assert {tool["function"]["name"] for tool in tools} == {"read", "grep", "glob", "advise"}


def test_hidden_watchdog_is_rejected_before_read(tmp_path, monkeypatch):
    source = tmp_path / "WATCHDOG.md"
    source.write_text("not permitted\n" * 2000)
    (tmp_path / ".yujignore").write_text("WATCHDOG.md\n")
    session = _local_session(tmp_path)
    original = Path.read_text

    def checked_read(path, *args, **kwargs):
        if path == source:
            raise AssertionError("hidden instruction source was read")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", checked_read)
    assert advisor_system_prompt(session, ignore_policy=session._ignore_policy) == SYSTEM_PROMPT


def test_missing_watchdog_keeps_fixed_protocol(tmp_path):
    session = _local_session(tmp_path)
    assert advisor_system_prompt(session, ignore_policy=session._ignore_policy) == SYSTEM_PROMPT


def test_external_watchdog_symlink_does_not_supply_priorities(tmp_path, monkeypatch):
    task = tmp_path / "task"
    task.mkdir()
    outside = tmp_path / "priorities.md"
    outside.write_text("external priority")
    (task / "WATCHDOG.md").symlink_to(outside)
    session = _local_session(task)
    original = Path.read_text

    def checked_read(path, *args, **kwargs):
        if path.resolve() == outside:
            raise AssertionError("external instruction source was read")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", checked_read)
    assert advisor_system_prompt(session, ignore_policy=session._ignore_policy) == SYSTEM_PROMPT


def test_internal_watchdog_symlink_preserves_complete_priorities(tmp_path):
    source = tmp_path / "guidance" / "review.md"
    source.parent.mkdir()
    body = "review priority\n" * 2000 + "FINAL_RULE"
    source.write_text(body)
    (tmp_path / "WATCHDOG.md").symlink_to(source)
    session = _local_session(tmp_path)
    assert advisor_system_prompt(session, ignore_policy=session._ignore_policy).endswith(body)


@pytest.mark.parametrize("reserved", [False, True])
def test_guidance_and_actual_advisor_tool_share_permissions(tmp_path, reserved):
    from scripts.llm_solver.server.types import ToolCall

    target = tmp_path / ("transcript.log" if reserved else "guidance/review.md")
    target.parent.mkdir(exist_ok=True)
    marker = "SYNTHETIC_SELECTED_PRIORITY"
    target.write_text(marker)
    (tmp_path / "WATCHDOG.md").symlink_to(target)
    client = _ScriptedClient(advisor=[
        _turn(calls=[ToolCall(id="read-priority", name="read", arguments={"path": str(target)})]),
        _turn(content="NO_ADVISORY", reason="stop"),
    ])
    session = Session(_advisor_config(turn_snapshots_enabled=False, auto_commit=False),
                      client, "system", "task", str(tmp_path))
    _record_primary_turn(session, 0)
    session._capture_advisor_turn(0, "Observed a result.", [])
    assert session._maybe_run_advisor(0) is False
    assert (marker in client.advisor_calls[0][0][0]["content"]) is not reserved
    assert (marker in client.advisor_calls[1][0][-1]["content"]) is not reserved
