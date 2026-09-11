"""Git exclusions and sink writes must follow creation, not directory names."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from _config_helpers import make_config
from test_worktree_runtime import _git, _make_repo
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness._loop.trace_output import _sink_trace_output
from scripts.llm_solver.harness._loop.state_projection import sink_to_disk


@pytest.mark.parametrize("subdir", ["", "src", "odd [name]"])
def test_session_keeps_preexisting_runtime_named_project_files_visible(tmp_path, subdir):
    repo = _make_repo(tmp_path)
    empty = tmp_path / "empty-excludes"
    empty.write_text("")
    _git(repo, "config", "core.excludesFile", str(empty))
    cwd = repo / subdir
    cwd.mkdir(exist_ok=True)
    for name in (".solver", ".tool_output"):
        (cwd / name).mkdir()
        (cwd / name / "project-notes.md").write_text("owned by the task")
    exclude = repo / ".git/info/exclude"
    prior_rules = exclude.read_bytes()
    before = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    session = Session(make_config(), MagicMock(), "system", "task", str(cwd))
    assert exclude.read_bytes() == prior_rules
    assert _git(repo, "status", "--porcelain=v1", "--untracked-files=all") == before
    session._sink_to_disk("new shell output", 1)
    created_trace = _sink_trace_output(session, "new trace output", 2)
    assert created_trace
    assert _git(repo, "status", "--porcelain=v1", "--untracked-files=all") == before
    for name in (".solver", ".tool_output"):
        assert (cwd / name / "project-notes.md").read_text() == "owned by the task"


@pytest.mark.parametrize("existing_link", [False, True])
def test_trace_sink_does_not_replace_existing_file(tmp_path, existing_link):
    output = tmp_path / ".tool_output"
    output.mkdir()
    existing = output / "1_0001_t2_trace.log"
    original = tmp_path / "original.txt"
    original.write_text("original")
    if existing_link:
        existing.symlink_to(original)
    else:
        existing.write_text("original")
    session = SimpleNamespace(cwd=str(tmp_path), _session_number=1, _sink_counter=0)
    path = _sink_trace_output(session, "fresh", 2)
    assert existing.read_text() == "original"
    assert original.read_text() == "original"
    assert path == ".tool_output/1_0002_t2_trace.log"
    assert (tmp_path / path).read_text() == "fresh"


@pytest.mark.parametrize("writer", [sink_to_disk, _sink_trace_output])
def test_sink_does_not_adopt_a_symlinked_output_directory(tmp_path, writer):
    task = tmp_path / "task"
    task.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    (task / ".tool_output").symlink_to(other, target_is_directory=True)
    session = SimpleNamespace(cwd=str(task), _session_number=1, _sink_counter=0, cfg=make_config())
    assert writer(session, "private output", 1) == ""
    assert list(other.iterdir()) == []


@pytest.mark.parametrize("expired", [False, True])
def test_exact_exclusion_uses_remaining_execution_allowance(tmp_path, monkeypatch, expired):
    from scripts.llm_solver.harness import time_budget, worktree_runtime

    repo = _make_repo(tmp_path)
    created = repo / "created.log"
    created.write_text("saved")
    exclude = repo / ".git/info/exclude"
    before = exclude.read_bytes()
    clock = [100.0]
    monkeypatch.setattr(time_budget.time, "monotonic", lambda: clock[0])
    run = worktree_runtime._run
    allowances = []

    def record_run(*args, **kwargs):
        allowances.append(kwargs["timeout"])
        return run(*args, **kwargs)

    monkeypatch.setattr(worktree_runtime, "_run", record_run)
    with time_budget.run_time_budget(20):
        with time_budget.command_time_budget(5):
            clock[0] += 6 if expired else 2
            worktree_runtime.exclude_created_runtime_file(repo, created)
    if expired:
        assert allowances == []
        assert exclude.read_bytes() == before
    else:
        assert allowances == [3.0]
        assert _git(repo, "check-ignore", "created.log") == "created.log"
