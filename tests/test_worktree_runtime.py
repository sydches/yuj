"""Isolation, provenance, reuse, and cleanup tests for session worktrees."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from scripts.llm_solver.harness.worktree_runtime import (
    WORKTREE_DIR_NAME,
    WorktreeDirtyError,
    WorktreeExistsError,
    WorktreeRuntimeError,
    copy_workspace_to_worktree,
    create_session_worktree,
    inspect_session_worktree,
    remove_session_worktree,
    snapshot_workspace,
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


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "project"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "src").mkdir()
    (repo / "src" / "data.bin").write_bytes(bytes(range(256)) + b"\x00\xff")
    (repo / "README.md").write_text("base\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_auto_worktree_isolates_bytes_and_preserves_original_checkout(tmp_path):
    repo = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD")
    original_hash = _sha(repo / "src" / "data.bin")
    status_before = _git(repo, "status", "--porcelain=v1")

    info = create_session_worktree(repo, mode="auto", run_id="session-001")
    assert info is not None
    assert info.worktree_path == repo / WORKTREE_DIR_NAME / "session-001"
    assert info.session_cwd == info.worktree_path
    assert info.branch == "worktree-session-001"
    assert info.base_commit == base
    assert info.reused is False
    assert info.session_start_fields() == {
        "worktree_path": str(info.worktree_path),
        "worktree_branch": "worktree-session-001",
        "worktree_base_commit": base,
    }

    (info.worktree_path / "src" / "data.bin").write_bytes(b"isolated\x00change")
    (info.worktree_path / "new.txt").write_text("worktree only\n")
    assert _sha(repo / "src" / "data.bin") == original_hash
    assert not (repo / "new.txt").exists()
    assert _git(repo, "status", "--porcelain=v1") == status_before
    assert f"/{WORKTREE_DIR_NAME}/" in (repo / ".git" / "info" / "exclude").read_text()


def test_workspace_copy_preserves_dirty_endpoint_and_excludes_owned_children(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "README.md").write_text("dirty tracked\n")
    (repo / "new.txt").write_text("dirty untracked\n")
    before = snapshot_workspace(repo)

    info, copied = copy_workspace_to_worktree(
        repo, child_run_id="subagent-copy"
    )

    assert copied == before
    assert snapshot_workspace(repo) == before
    assert snapshot_workspace(info.worktree_path) == before
    assert (info.worktree_path / "README.md").read_text() == "dirty tracked\n"
    assert (info.worktree_path / "new.txt").read_text() == "dirty untracked\n"
    (info.worktree_path / "README.md").write_text("child only\n")
    assert (repo / "README.md").read_text() == "dirty tracked\n"
    remove_session_worktree(repo, "subagent-copy", force=True)


def test_subdirectory_invocation_maps_to_same_relative_cwd(tmp_path):
    repo = _make_repo(tmp_path)
    info = create_session_worktree(
        repo / "src", mode="auto", run_id="nested-session"
    )
    assert info is not None
    assert info.source_cwd == repo / "src"
    assert info.session_cwd == info.worktree_path / "src"
    assert (info.session_cwd / "data.bin").read_bytes() == bytes(range(256)) + b"\x00\xff"


def test_named_branch_and_explicit_base_commit(tmp_path):
    repo = _make_repo(tmp_path)
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "README.md").write_text("second\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-qm", "second")

    info = create_session_worktree(
        repo,
        mode="feature/session-runtime",
        run_id="named-session",
        base_commit=first,
    )
    assert info is not None
    assert info.branch == "feature/session-runtime"
    assert info.base_commit == first
    assert (info.worktree_path / "README.md").read_text() == "base\n"


def test_explicit_reuse_preserves_path_branch_and_original_base(tmp_path):
    repo = _make_repo(tmp_path)
    created = create_session_worktree(repo, mode="auto", run_id="resume-me")
    assert created is not None
    (created.worktree_path / "README.md").write_text("resume state\n")

    with pytest.raises(WorktreeExistsError, match="already exists"):
        create_session_worktree(repo, mode="auto", run_id="resume-me")
    reused = create_session_worktree(
        repo, mode="auto", run_id="resume-me", reuse=True
    )
    assert reused is not None
    assert reused.reused is True
    assert reused.worktree_path == created.worktree_path
    assert reused.base_commit == created.base_commit
    assert (reused.worktree_path / "README.md").read_text() == "resume state\n"


def test_inspect_uses_gitdir_metadata_not_workspace_file(tmp_path):
    repo = _make_repo(tmp_path)
    created = create_session_worktree(repo, mode="auto", run_id="inspect-me")
    assert created is not None
    assert not (created.worktree_path / "yuj-runtime.json").exists()

    inspected = inspect_session_worktree(repo, "inspect-me")
    assert inspected.worktree_path == created.worktree_path
    assert inspected.branch == created.branch
    assert inspected.base_commit == created.base_commit


def test_remove_deletes_clean_worktree_and_branch(tmp_path):
    repo = _make_repo(tmp_path)
    info = create_session_worktree(repo, mode="auto", run_id="remove-me")
    assert info is not None
    assert _git(repo, "show-ref", "--verify", f"refs/heads/{info.branch}")

    removed = remove_session_worktree(repo, "remove-me")
    assert removed.worktree_path == info.worktree_path
    assert removed.branch == info.branch
    assert removed.forced is False
    assert not info.worktree_path.exists()
    assert _git(
        repo,
        "show-ref",
        "--verify",
        "--quiet",
        f"refs/heads/{info.branch}",
        check=False,
    ) == ""
    assert str(info.worktree_path) not in _git(repo, "worktree", "list", "--porcelain")


def test_remove_refuses_dirty_or_unmerged_and_force_is_explicit(tmp_path):
    repo = _make_repo(tmp_path)
    info = create_session_worktree(repo, mode="auto", run_id="guarded-remove")
    assert info is not None
    (info.worktree_path / "README.md").write_text("dirty\n")
    with pytest.raises(WorktreeDirtyError, match="uncommitted"):
        remove_session_worktree(repo, "guarded-remove")

    _git(info.worktree_path, "add", "README.md")
    _git(info.worktree_path, "commit", "-qm", "session work")
    with pytest.raises(WorktreeDirtyError, match="unmerged"):
        remove_session_worktree(repo, "guarded-remove")
    removed = remove_session_worktree(repo, "guarded-remove", force=True)
    assert removed.forced is True
    assert not info.worktree_path.exists()


def test_off_mode_has_no_git_or_filesystem_side_effect(tmp_path):
    repo = _make_repo(tmp_path)
    exclude = repo / ".git" / "info" / "exclude"
    before = exclude.read_bytes()
    assert create_session_worktree(repo, mode="off", run_id="unused") is None
    assert exclude.read_bytes() == before
    assert not (repo / WORKTREE_DIR_NAME).exists()


def test_dirty_source_checkout_is_refused_before_branch_creation(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "README.md").write_text("dirty source\n")
    with pytest.raises(WorktreeDirtyError, match="source checkout"):
        create_session_worktree(repo, mode="auto", run_id="must-refuse")
    assert not (repo / WORKTREE_DIR_NAME / "must-refuse").exists()
    assert _git(
        repo,
        "show-ref",
        "--verify",
        "--quiet",
        "refs/heads/worktree-must-refuse",
        check=False,
    ) == ""


@pytest.mark.parametrize("run_id", ["../escape", "two/parts", ".", "bad.lock"])
def test_run_id_cannot_escape_worktree_root(tmp_path, run_id):
    repo = _make_repo(tmp_path)
    with pytest.raises(ValueError, match="safe"):
        create_session_worktree(repo, mode="auto", run_id=run_id)


def test_remove_refuses_unowned_registered_worktree(tmp_path):
    repo = _make_repo(tmp_path)
    path = repo / WORKTREE_DIR_NAME / "foreign"
    (repo / ".git" / "info" / "exclude").write_text(f"/{WORKTREE_DIR_NAME}/\n")
    _git(repo, "worktree", "add", "-q", "-b", "foreign-branch", str(path), "HEAD")

    with pytest.raises(WorktreeRuntimeError, match="metadata"):
        remove_session_worktree(repo, "foreign", force=True)
    assert path.exists()
    assert _git(repo, "show-ref", "--verify", "refs/heads/foreign-branch")
