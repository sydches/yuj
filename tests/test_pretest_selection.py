"""Pretest execution must not be selected from a neighboring filename."""
import subprocess
from unittest.mock import patch

import pytest

from llm_solver.harness._loop.pretest_resume import run_pretest


def check(repo, script=None):
    return run_pretest(repo, pretest_script=script, pretest_timeout=1,
                       pretest_head_chars=100, pretest_tail_chars=100)


def test_undeclared_neighbor_is_not_executed(tmp_path):
    repo = tmp_path / "run/repos/task"
    repo.mkdir(parents=True)
    neighbor = tmp_path / "run/pretest/task.sh"
    neighbor.parent.mkdir()
    neighbor.write_text("exit 0")
    with patch("llm_solver.harness._loop.pretest_resume.subprocess.run") as execute:
        assert check(repo) == ""
    execute.assert_not_called()


@pytest.mark.parametrize("kind", ["missing", "directory", "broken_link"])
def test_explicit_unavailable_script_is_reported_without_execution(tmp_path, kind):
    script = tmp_path / "selected-check"
    if kind == "directory":
        script.mkdir()
    elif kind == "broken_link":
        script.symlink_to(tmp_path / "missing-target")
    with patch("llm_solver.harness._loop.pretest_resume.subprocess.run") as execute:
        block = check(tmp_path, script)
    assert "pretest not started" in block
    assert "exit code:" not in block
    execute.assert_not_called()


def test_explicit_selection_survives_task_relocation(tmp_path):
    script = tmp_path / "caller-check.sh"
    script.write_text("exit 1")
    response = subprocess.CompletedProcess([], 1, "check output", "")
    with patch("llm_solver.harness._loop.pretest_resume.subprocess.run", return_value=response) as execute:
        for location in (tmp_path / "old/task", tmp_path / "different/new-name"):
            location.mkdir(parents=True)
            assert "exit code: 1" in check(location, script)
            assert execute.call_args.args[0] == ["bash", str(script.resolve())]
            assert execute.call_args.kwargs["cwd"] == str(location)
    assert execute.call_count == 2
