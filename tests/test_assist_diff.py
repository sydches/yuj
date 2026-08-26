"""Read-only session diff coverage for the installed assistant CLI."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from scripts.llm_assist.__main__ import main as assist_main
from scripts.llm_assist.session_diff import (
    SessionDiffError,
    build_session_worktree_diff,
)
from scripts.llm_assist.store import SessionStore
from scripts.llm_solver.harness.worktree_runtime import (
    create_session_worktree,
    remove_session_worktree,
)


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


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "README.md").write_text("base\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-qm", "base")
    return repo


def _record_with_worktree(tmp_path: Path):
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
    info = create_session_worktree(repo, mode="auto", run_id=record.session_id)
    assert info is not None
    store.update_session_worktree(
        record.session_id,
        path=info.worktree_path,
        branch=info.branch,
        base_commit=info.base_commit,
    )
    return repo, store, record, info


def test_diff_cli_renders_tracked_and_untracked_changes_without_mutation(
    tmp_path, capsys
):
    _repo_path, store, record, info = _record_with_worktree(tmp_path)
    store.set_session_label(record.session_id, "review-diff")
    (info.worktree_path / "README.md").write_text("changed\n")
    (info.worktree_path / "new file.txt").write_text("untracked\n")
    (info.worktree_path / "empty.txt").write_text("")
    before_head = _git(info.worktree_path, "rev-parse", "HEAD")
    before_status = _git(
        info.worktree_path, "status", "--porcelain=v1", "--untracked-files=all"
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert assist_main(["diff", "review-diff"]) == 0

    captured = capsys.readouterr()
    assert "diff --git a/README.md b/README.md" in captured.out
    assert "-base" in captured.out
    assert "+changed" in captured.out
    assert "new file.txt" in captured.out
    assert "+untracked" in captured.out
    assert "empty.txt" in captured.out
    assert "ownership: session-worktree" in captured.err
    assert "diff_state: changes" in captured.err
    assert "tracked_changes: yes" in captured.err
    assert "untracked_files: 2" in captured.err
    assert _git(info.worktree_path, "rev-parse", "HEAD") == before_head
    assert (
        _git(
            info.worktree_path,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        == before_status
    )


def test_diff_cli_reports_a_clean_owned_worktree(tmp_path, capsys):
    _repo_path, store, record, _info = _record_with_worktree(tmp_path)

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert assist_main(["diff", record.short_id]) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ownership: session-worktree" in captured.err
    assert "diff_state: clean" in captured.err
    assert "untracked_files: 0" in captured.err


def test_diff_cli_refuses_to_attribute_a_direct_session(tmp_path, capsys):
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
    (repo / "README.md").write_text("not attributable\n")

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert assist_main(["diff", record.session_id]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ownership: unknown" in captured.err
    assert "baseline: missing" in captured.err
    assert "diff_state: unavailable" in captured.err
    assert "cannot be attributed" in captured.err


def test_diff_cli_reports_a_removed_worktree(tmp_path, capsys):
    repo, store, record, _info = _record_with_worktree(tmp_path)
    remove_session_worktree(repo, record.session_id, force=True)

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert assist_main(["diff", record.session_id]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ownership: unverified" in captured.err
    assert "worktree: removed" in captured.err
    assert "diff_state: unavailable" in captured.err


def test_diff_builder_reports_a_missing_baseline(tmp_path):
    repo = _repo(tmp_path)

    try:
        build_session_worktree_diff(repo, "f" * 40)
    except SessionDiffError as exc:
        assert exc.code == "baseline_missing"
        assert "baseline is missing" in str(exc)
    else:
        raise AssertionError("missing baseline was accepted")
