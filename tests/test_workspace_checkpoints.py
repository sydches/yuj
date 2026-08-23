"""Byte-level and isolation contracts for workspace checkpoints."""
from __future__ import annotations

import dataclasses
import hashlib
import os
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness.sandbox._preflight import bwrap_preflight
from scripts.llm_solver.harness.tools import _bash_unreadable_paths, dispatch
from scripts.llm_solver.harness.workspace_checkpoints import (
    CheckpointNotFoundError,
    WorkspaceCheckpointStore,
    default_shadow_dir,
    restore_checkpoint,
    tool_call_needs_checkpoint,
)


NOW = datetime(2026, 8, 23, 13, 0, 0, tzinfo=timezone.utc)


def _clock() -> datetime:
    return NOW


def _git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if check and proc.returncode != 0:
        raise AssertionError(proc.stderr)
    return proc.stdout.strip()


def _make_project_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "task"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / ".gitignore").write_text("ignored.bin\ncache/\n")
    (repo / "alpha.bin").write_bytes(b"alpha\x00one\r\n")
    (repo / "delete.txt").write_bytes(b"delete me\n")
    executable = repo / "run.sh"
    executable.write_bytes(b"#!/bin/sh\nprintf ok\n")
    executable.chmod(0o755)
    os.symlink("alpha.bin", repo / "alpha.link")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


def _manifest(root: Path, *, skip: tuple[str, ...] = ()) -> dict[str, tuple]:
    result: dict[str, tuple] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if rel == ".git" or rel.startswith(".git/") or any(
            rel == name or rel.startswith(name + "/") for name in skip
        ):
            continue
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if stat.S_ISLNK(info.st_mode):
            result[rel] = ("symlink", os.readlink(path))
        else:
            data = path.read_bytes()
            result[rel] = (
                "file",
                bool(stat.S_IMODE(info.st_mode) & 0o111),
                hashlib.sha256(data).hexdigest(),
                data,
            )
    return result


def test_shadow_repo_is_external_and_project_git_is_untouched(tmp_path):
    repo = _make_project_repo(tmp_path)
    shadow = default_shadow_dir(repo)
    store = WorkspaceCheckpointStore(repo, clock=_clock)
    log_before = _git(repo, "log", "--format=%H")
    status_before = _git(repo, "status", "--porcelain=v1")
    index_before = (repo / ".git" / "index").read_bytes()

    checkpoint = store.capture(1)

    assert shadow == tmp_path / ".yuj_task" / ".shadow_git"
    assert shadow.is_dir()
    assert not shadow.is_relative_to(repo)
    assert checkpoint.commit == _git(shadow, "rev-parse", "HEAD")
    assert _git(repo, "log", "--format=%H") == log_before
    assert _git(repo, "status", "--porcelain=v1") == status_before
    assert (repo / ".git" / "index").read_bytes() == index_before


def test_capture_and_restore_created_modified_deleted_files_byte_exact(tmp_path):
    repo = _make_project_repo(tmp_path)
    store = WorkspaceCheckpointStore(repo, clock=_clock)
    first = store.capture(3)
    first_manifest = _manifest(repo)

    (repo / "alpha.bin").write_bytes(bytes(range(256)) + b"\x00\xff")
    (repo / "delete.txt").unlink()
    (repo / "created.dat").write_bytes(b"created\x00later")
    (repo / "run.sh").chmod(0o644)
    (repo / "alpha.link").unlink()
    os.symlink("created.dat", repo / "alpha.link")
    second = store.capture(8)
    second_manifest = _manifest(repo)

    (repo / "alpha.bin").write_bytes(b"corrupt")
    (repo / "created.dat").unlink()
    (repo / "after.txt").write_bytes(b"must disappear")

    restored_first = store.restore_checkpoint(3)
    assert restored_first.commit == first.commit
    assert _manifest(repo) == first_manifest
    assert not (repo / "created.dat").exists()
    assert (repo / "delete.txt").read_bytes() == b"delete me\n"

    restored_second = restore_checkpoint(repo, 8)
    assert restored_second.commit == second.commit
    assert _manifest(repo) == second_manifest
    assert not (repo / "delete.txt").exists()
    assert (repo / "alpha.bin").read_bytes() == bytes(range(256)) + b"\x00\xff"
    assert os.readlink(repo / "alpha.link") == "created.dat"


def test_gitignore_and_configured_excludes_are_preserved(tmp_path):
    repo = _make_project_repo(tmp_path)
    (repo / "ignored.bin").write_bytes(b"ignored-before")
    (repo / "cache").mkdir()
    (repo / "cache" / "data.bin").write_bytes(b"cache-before")
    (repo / "scratch").mkdir()
    (repo / "scratch" / "note.txt").write_bytes(b"scratch-before")
    store = WorkspaceCheckpointStore(repo, excludes=("scratch/**",), clock=_clock)
    checkpoint = store.capture(2)
    tree_paths = subprocess.run(
        [
            "git",
            f"--git-dir={store.shadow_dir}",
            "ls-tree",
            "-r",
            "--name-only",
            checkpoint.commit,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert "ignored.bin" not in tree_paths
    assert "cache/data.bin" not in tree_paths
    assert "scratch/note.txt" not in tree_paths

    (repo / "ignored.bin").write_bytes(b"ignored-after")
    (repo / "cache" / "data.bin").write_bytes(b"cache-after")
    (repo / "scratch" / "note.txt").write_bytes(b"scratch-after")
    (repo / "alpha.bin").write_bytes(b"changed")
    store.restore_checkpoint(2)
    assert (repo / "ignored.bin").read_bytes() == b"ignored-after"
    assert (repo / "cache" / "data.bin").read_bytes() == b"cache-after"
    assert (repo / "scratch" / "note.txt").read_bytes() == b"scratch-after"
    assert (repo / "alpha.bin").read_bytes() == b"alpha\x00one\r\n"


def test_empty_files_and_empty_workspace_round_trip(tmp_path):
    workspace = tmp_path / "empty-task"
    workspace.mkdir()
    store = WorkspaceCheckpointStore(workspace, clock=_clock)
    empty = store.capture(0)
    (workspace / "zero").write_bytes(b"")
    with_file = store.capture(1)
    (workspace / "zero").write_bytes(b"not empty")

    store.restore_checkpoint(1)
    assert (workspace / "zero").read_bytes() == b""
    store.restore_checkpoint(0)
    assert list(workspace.iterdir()) == []
    assert empty.commit != with_file.commit


def test_capture_reports_trace_fields_and_per_call_metrics(tmp_path):
    repo = _make_project_repo(tmp_path)
    store = WorkspaceCheckpointStore(repo, clock=_clock)
    first = store.capture(4)
    (repo / "alpha.bin").write_bytes(b"second")
    second = store.capture(5)

    assert first.trace_fields() == {
        "turn": 4,
        "commit": first.commit,
        "duration_ms": first.duration_ms,
        "file_count": 5,
        "byte_count": sum(
            len(value[-1]) if value[0] == "file" else len(value[1].encode())
            for value in _manifest(repo).values()
            if value[0] in {"file", "symlink"}
        )
        - len(b"second")
        + len(b"alpha\x00one\r\n"),
    }
    payload = store.metrics_payload()
    assert payload["enabled"] is True
    assert payload["count"] == 2
    assert [row["turn"] for row in payload["per_call"]] == [4, 5]
    assert [row["commit"] for row in payload["per_call"]] == [first.commit, second.commit]
    assert payload["total_duration_ms"] == round(first.duration_ms + second.duration_ms, 3)
    assert all(row["duration_ms"] >= 0 for row in payload["per_call"])


@pytest.mark.parametrize(
    ("name", "executed", "expected"),
    [
        ("write", True, True),
        ("edit", True, True),
        ("apply_patch", True, True),
        ("bash", True, True),
        ("bash", False, False),
        ("read", True, False),
        ("glob", True, False),
    ],
)
def test_mutating_tool_checkpoint_classification(name, executed, expected):
    assert tool_call_needs_checkpoint(name, executed=executed) is expected


def test_missing_turn_and_unsafe_layout_fail_loudly(tmp_path):
    repo = _make_project_repo(tmp_path)
    store = WorkspaceCheckpointStore(repo, clock=_clock)
    store.capture(1)
    with pytest.raises(CheckpointNotFoundError, match="turn 99"):
        store.restore_checkpoint(99)
    with pytest.raises(ValueError, match="outside"):
        WorkspaceCheckpointStore(repo, shadow_dir=repo / ".shadow_git")
    with pytest.raises(ValueError, match="safe relative"):
        WorkspaceCheckpointStore(repo, excludes=("../outside",))


def test_shadow_storage_is_hidden_from_read_glob_and_sandboxed_bash(tmp_path):
    repo = _make_project_repo(tmp_path)
    store = WorkspaceCheckpointStore(repo, clock=_clock)
    store.capture(1)
    cfg = dataclasses.replace(
        load_config(), sandbox_bash=False, sandbox_required=False
    )

    read_result = dispatch(
        "read", {"path": str(store.shadow_dir / "HEAD")}, cwd=str(repo), cfg=cfg
    )
    glob_result = dispatch(
        "glob", {"pattern": "**/*", "path": str(store.shadow_dir)}, cwd=str(repo), cfg=cfg
    )
    assert "ref: refs/heads/checkpoints" not in read_result
    assert "objects/" not in glob_result
    assert ".shadow_git" not in dispatch(
        "bash", {"cmd": "ls -A ."}, cwd=str(repo), cfg=cfg
    )
    masks = _bash_unreadable_paths(str(repo), cfg)
    assert f"optional:{store.shadow_dir.parent}" in masks

    sandbox_cfg = load_config()
    preflight_ok, _ = bwrap_preflight(sandbox_cfg.bwrap_bin)
    if not preflight_ok:
        pytest.skip("bwrap is required for absolute-path mask verification")
    bash_result = dispatch(
        "bash",
        {"cmd": f"ls -A {store.shadow_dir}"},
        cwd=str(repo),
        cfg=sandbox_cfg,
    )
    assert "objects" not in bash_result
    assert "checkpoint_metrics.jsonl" not in bash_result
