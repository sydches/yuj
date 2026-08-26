"""Canonical launcher, assistant-store, trace, and cleanup seams for #30."""
from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.llm_assist.__main__ import main as assist_main
from scripts.llm_assist.runner import create_session, run_session
from scripts.llm_assist.store import SessionStore
from scripts.llm_solver.__main__ import _prepare_task_worktree
from scripts.llm_solver._shared.telemetry_paths import trace_path
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness.loop import solve_task
from scripts.llm_solver.harness.worktree_runtime import (
    WorktreeRuntimeError,
    WorktreeRuntimeInfo,
    create_session_worktree,
)
from scripts.llm_solver.server.types import TurnResult, Usage

from _config_helpers import make_config


def _git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise AssertionError(proc.stderr)
    return proc.stdout.strip()


def _repo(tmp_path: Path, name: str = "repo") -> Path:
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "README.md").write_text("base\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-qm", "base")
    return repo


def test_runtime_worktree_config_default_overlay_and_type_validation(tmp_path):
    assert load_config().runtime_worktree == "off"
    overlay = tmp_path / "worktree.toml"
    overlay.write_text('[runtime]\nworktree = "auto"\n')
    assert load_config(user_config=overlay).runtime_worktree == "auto"
    overlay.write_text("[runtime]\nworktree = false\n")
    with pytest.raises(ValueError, match="runtime.worktree"):
        load_config(user_config=overlay)


def test_direct_wrapper_is_stable_reuses_and_maps_sandbox_cwd(tmp_path):
    repo = _repo(tmp_path)
    cfg = SimpleNamespace(runtime_worktree="auto")
    run_dir = tmp_path / "run"
    first = _prepare_task_worktree(
        cfg,
        run_dir=run_dir,
        source_cwd=repo,
        resume=False,
        multi_task=False,
    )
    assert first is not None
    (first.session_cwd / "README.md").write_text("isolated\n")
    resumed = _prepare_task_worktree(
        cfg,
        run_dir=run_dir,
        source_cwd=repo,
        resume=True,
        multi_task=False,
    )
    assert resumed is not None
    assert resumed.session_cwd == first.session_cwd
    assert (resumed.session_cwd / "README.md").read_text() == "isolated\n"
    assert (repo / "README.md").read_text() == "base\n"


def test_direct_multi_task_custom_branch_names_are_distinct(tmp_path):
    one = _repo(tmp_path, "one")
    two = _repo(tmp_path, "two")
    cfg = SimpleNamespace(runtime_worktree="review/session")
    run_dir = tmp_path / "run"
    infos = [
        _prepare_task_worktree(
            cfg,
            run_dir=run_dir,
            source_cwd=repo,
            resume=False,
            multi_task=True,
        )
        for repo in (one, two)
    ]
    assert all(info is not None for info in infos)
    assert infos[0].run_id != infos[1].run_id
    assert infos[0].branch != infos[1].branch
    assert all(info.branch.startswith("review/session-") for info in infos)


def test_session_start_traces_worktree_identity_only_when_enabled(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / ".git").mkdir()
    (task / "prompt.txt").write_text("finish")
    client = MagicMock()
    client.chat.return_value = TurnResult(
        content="done",
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(prompt_tokens=4, completion_tokens=1),
    )
    client.build_assistant_message.return_value = {
        "role": "assistant", "content": "done"
    }
    info = WorktreeRuntimeInfo(
        enabled=True,
        run_id="trace-run",
        source_cwd=task,
        repo_root=task,
        worktree_path=task,
        session_cwd=task,
        branch="worktree-trace-run",
        base_commit="a" * 40,
    )
    with patch("scripts.llm_solver.harness.loop._auto_commit"):
        assert solve_task(
            task,
            make_config(max_sessions=1),
            client,
            worktree_info=info,
        ) is True
    events = [json.loads(line) for line in trace_path(task).read_text().splitlines()]
    start = next(event for event in events if event["event"] == "session_start")
    assert start | info.session_start_fields() == start

    off_task = tmp_path / "off-task"
    off_task.mkdir()
    (off_task / ".git").mkdir()
    (off_task / "prompt.txt").write_text("finish")
    with patch("scripts.llm_solver.harness.loop._auto_commit"):
        assert solve_task(
            off_task, make_config(max_sessions=1), client
        ) is True
    off_events = [
        json.loads(line) for line in trace_path(off_task).read_text().splitlines()
    ]
    off_start = next(
        event for event in off_events if event["event"] == "session_start"
    )
    assert not {
        "worktree_path", "worktree_branch", "worktree_base_commit"
    } & off_start.keys()


def test_store_migrates_and_round_trips_worktree_identity(tmp_path):
    root = tmp_path / "assist"
    root.mkdir()
    db = root / "sessions.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            create table sessions (
                session_id text primary key, created_at text not null,
                updated_at text not null, cwd text not null,
                artifact_dir text not null, model text not null,
                status text not null, last_finish_reason text,
                prompt_text text not null, prompt_source text not null,
                context_mode text not null, system_prompt_path text,
                config_paths_json text not null
            )
            """
        )
    store = SessionStore(root)
    columns = {
        row[1]
        for row in sqlite3.connect(db).execute("pragma table_info(sessions)")
    }
    assert {
        "worktree_path", "worktree_branch", "worktree_base_commit"
    } <= columns

    record = store.create_session(
        cwd=tmp_path / "repo",
        model="model",
        prompt_text="task",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    store.update_session_worktree(
        record.session_id,
        path=tmp_path / "owned",
        branch="worktree-test",
        base_commit="b" * 40,
    )
    saved = store.get_session(record.session_id)
    assert saved is not None
    assert saved.worktree_path == str((tmp_path / "owned").resolve())
    assert saved.worktree_branch == "worktree-test"
    assert saved.worktree_base_commit == "b" * 40


def test_assistant_run_persists_and_strictly_reuses_worktree(tmp_path):
    repo = _repo(tmp_path)
    assist_root = tmp_path / "assist"
    overlay = tmp_path / "assistant-worktree.toml"
    overlay.write_text(
        '[runtime]\nworktree = "auto"\n'
        '\n[sandbox]\nbackend = "none"\n'
    )
    store = SessionStore(assist_root)
    record = create_session(
        store,
        cwd=repo,
        prompt_text="task",
        prompt_source="inline",
        model="test-model",
        config_paths=[overlay],
        system_prompt_path=None,
        context_mode="full",
    )
    client = MagicMock()
    client.query_server_context.return_value = None
    calls: list[tuple[Path, object]] = []

    def fake_solve(work_dir, _cfg, _client, **kwargs):
        calls.append((Path(work_dir), kwargs.get("worktree_info")))
        return True

    with (
        patch("scripts.llm_assist.runner._load_profile", return_value=None),
        patch("scripts.llm_assist.runner._make_client", return_value=client),
        patch("scripts.llm_assist.runner.build_model_role_runtime"),
        patch("scripts.llm_assist.runner.solve_task", side_effect=fake_solve),
    ):
        assert run_session(store, record, resume=False)[0] is True
        saved = store.get_session(record.session_id)
        assert saved is not None and saved.worktree_path
        worktree = Path(saved.worktree_path)
        (worktree / "README.md").write_text("resume state\n")
        assert run_session(store, saved, resume=True)[0] is True

    assert calls[0][0] == calls[1][0] == Path(saved.worktree_path)
    assert calls[1][1].reused is True
    assert (Path(saved.worktree_path) / "README.md").read_text() == "resume state\n"
    assert (repo / "README.md").read_text() == "base\n"
    assert store.get_active_session_id(repo) == record.session_id
    assert store.get_active_session_id(Path(saved.worktree_path)) is None
    metadata = json.loads((saved.artifact_path / "session.json").read_text())
    assert metadata["worktree_path"] == saved.worktree_path
    assert metadata["worktree_branch"] == saved.worktree_branch
    assert metadata["worktree_base_commit"] == saved.worktree_base_commit


def test_assistant_resume_fails_if_recorded_worktree_is_missing(tmp_path):
    repo = _repo(tmp_path)
    store = SessionStore(tmp_path / "assist")
    record = store.create_session(
        cwd=repo,
        model="model",
        prompt_text="task",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    store.update_session_worktree(
        record.session_id,
        path=repo / ".yuj_worktrees" / record.session_id,
        branch=f"worktree-{record.session_id}",
        base_commit=_git(repo, "rev-parse", "HEAD"),
    )
    saved = store.get_session(record.session_id)
    assert saved is not None
    from scripts.llm_assist.runner import _resolve_session_worktree

    with pytest.raises(WorktreeRuntimeError, match="no registered worktree"):
        _resolve_session_worktree(
            store,
            saved,
            cfg=SimpleNamespace(runtime_worktree="auto"),
            resume=True,
        )


def test_worktree_rm_cli_refuses_dirty_then_force_discards(tmp_path, capsys):
    repo = _repo(tmp_path)
    store = SessionStore(tmp_path / "assist")
    record = store.create_session(
        cwd=repo,
        model="model",
        prompt_text="task",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    info = create_session_worktree(
        repo, mode="auto", run_id=record.session_id
    )
    assert info is not None
    store.update_session_worktree(
        record.session_id,
        path=info.worktree_path,
        branch=info.branch,
        base_commit=info.base_commit,
    )
    (info.worktree_path / "README.md").write_text("discard me\n")

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        with pytest.raises(SystemExit, match="uncommitted"):
            assist_main(["worktree", "rm", record.session_id])
        assert assist_main(
            ["worktree", "rm", record.session_id, "--force"]
        ) == 0
    output = capsys.readouterr().out
    assert "removed_worktree:" in output
    assert "were discarded" in output
    assert not info.worktree_path.exists()
    assert _git(
        repo,
        "show-ref",
        "--verify",
        "--quiet",
        f"refs/heads/{info.branch}",
        check=False,
    ) == ""
