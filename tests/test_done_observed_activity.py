"""Completion uses observed session activity, independently of Git HEAD."""
import shlex
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from tests._config_helpers import make_config
from scripts.llm_solver.harness._guardrails.checks_pre import done_guard
from scripts.llm_solver.harness._guardrails.state import Action
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


def _git(task, *args):
    return subprocess.run(["git", "-C", str(task), *args], check=True,
                          capture_output=True, text=True).stdout


@pytest.mark.parametrize("case,accepted", [
    ("plain_edit", True), ("initial_dirt", False), ("committed_edit", True),
    ("artifact_only", False), ("unavailable_observation", False),
    ("read_only", True), ("restored_edit", True),
])
def test_completion_after_real_dispatch(tmp_path, monkeypatch, case, accepted):
    task = tmp_path / "task"
    task.mkdir()
    (task / "source.txt").write_text("initial")
    if case in {"initial_dirt", "committed_edit", "artifact_only"}:
        _git(task, "init", "-q")
        _git(task, "config", "user.name", "Fixture")
        _git(task, "config", "user.email", "fixture@example.invalid")
        _git(task, "add", "source.txt")
        _git(task, "commit", "-qm", "initial")
    if case == "initial_dirt":
        (task / "source.txt").write_text("prepared before session")
    if case == "unavailable_observation":
        from scripts.llm_solver.harness import file_changes

        def unavailable(*args):
            raise subprocess.TimeoutExpired("native inventory", 0)

        monkeypatch.setattr(file_changes, "_inventory", unavailable)

    target = "metrics.json" if case == "artifact_only" else "source.txt"
    source = f"from pathlib import Path; Path({target!r}).write_text('changed')"
    commands = [shlex.join([sys.executable, "-c", source])]
    if case in {"initial_dirt", "read_only"}:
        commands = ["cat source.txt"]
    if case == "committed_edit":
        commands.append("git add source.txt && git commit -qm changed")
    if case == "restored_edit":
        commands.append(shlex.join([sys.executable, "-c",
            "from pathlib import Path; Path('source.txt').write_text('initial')"]))

    cfg = make_config(sandbox_bash=False, auto_commit=False,
        max_turns=len(commands) + 1, done_guard_enabled=True,
        done_require_mutation=case != "read_only", done_require_verify=False,
        done_require_pretest_parity=False, post_mutation_verification_gate_after=0,
        done_loop_abort_after=0)
    client = MagicMock()
    client.chat.side_effect = [
        TurnResult(content=None, tool_calls=[ToolCall(id=f"c{i}", name="bash",
                    arguments={"cmd": command})], finish_reason="tool_calls",
                   usage=Usage(prompt_tokens=10, completion_tokens=5))
        for i, command in enumerate(commands)
    ] + [TurnResult(content=None,
                    tool_calls=[ToolCall(id="done", name="done", arguments={"summary": "finished"})],
                    finish_reason="tool_calls",
                    usage=Usage(prompt_tokens=10, completion_tokens=5))]
    client.build_assistant_message.return_value = {"role": "assistant", "content": None}
    session = Session(cfg, client, "fixture", "run fixture actions", str(task))
    result = session.run()
    assert result.done is accepted
    decision = done_guard(session._guards, cfg, tc_name="done", cwd=str(task))
    assert (decision.action == Action.PASS) is accepted, decision.text
    if not accepted:
        assert not session._guards.has_mutated
        assert "No task-file change has been recorded" in decision.text
        assert "does not prove that files are unchanged" in decision.text
    if case == "committed_edit":
        assert not _git(task, "diff", "HEAD", "--", "source.txt")
        assert session._guards.has_mutated
    if case == "restored_edit":
        assert (task / "source.txt").read_text() == "initial"
        assert session._guards.has_mutated
    if case == "unavailable_observation":
        assert (task / "source.txt").read_text() == "changed"
    # Fresh sessions do not adopt old activity merely from the current tree.
    fresh = Session(cfg, client, "fixture", "new task", str(task))
    if case != "read_only":
        assert done_guard(fresh._guards, cfg, tc_name="done", cwd=str(task)).action == Action.BLOCK
