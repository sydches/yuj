"""Archived-session purge boundary, recovery, and resolver tests."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import stat
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.llm_assist.__main__ import main
from scripts.llm_assist.store import SessionPurgeInProgressError, SessionStore
from scripts.llm_solver.harness.clarifications import (
    create_clarification_request,
)
from scripts.llm_solver.harness.corrections import create_correction
from scripts.llm_solver.harness.worktree_runtime import create_session_worktree


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout.strip()


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


def _write_artifacts(record, *, secret: str = "artifact secret") -> None:
    root = record.artifact_path
    (root / ".solver").mkdir(parents=True)
    (root / "prompt.txt").write_text(secret)
    (root / "session.json").write_text(
        json.dumps(
            {
                "session_id": record.session_id,
                "parent_session_id": record.parent_session_id,
            },
            sort_keys=True,
        )
        + "\n"
    )
    (root / ".solver" / "state.json").write_text('{"state": {}}\n')
    (root / ".trace.jsonl").write_text(
        json.dumps({"event": "session_start", "session_number": 1})
        + "\n"
        + json.dumps(
            {
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "max_turns",
            }
        )
        + "\n"
    )


def _record(
    root: Path,
    cwd: Path,
    *,
    archived: bool = True,
    label: str | None = None,
    secret: str = "artifact secret",
):
    store = SessionStore(root)
    record = store.create_session(
        cwd=cwd,
        model="test-model",
        prompt_text="operator prompt secret",
        prompt_source="test",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
        credential_id="credential-secret-id",
    )
    _write_artifacts(record, secret=secret)
    store.update_session(
        record.session_id,
        status="paused",
        last_finish_reason="max_turns",
    )
    if label is not None:
        store.set_session_label(record.session_id, label)
    if archived:
        store.archive_session(record.session_id)
    refreshed = store.get_session(record.session_id)
    assert refreshed is not None
    return store, refreshed


def _tree_state(root: Path) -> dict[str, tuple[str, int, bytes]]:
    if not root.exists() and not root.is_symlink():
        return {}
    result: dict[str, tuple[str, int, bytes]] = {}

    def visit(directory: Path) -> None:
        for entry in sorted(os.scandir(directory), key=lambda item: item.name):
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            metadata = os.lstat(path)
            if stat.S_ISLNK(metadata.st_mode):
                result[relative] = (
                    "link",
                    stat.S_IMODE(metadata.st_mode),
                    os.fsencode(os.readlink(path)),
                )
            elif stat.S_ISDIR(metadata.st_mode):
                result[relative] = (
                    "dir",
                    stat.S_IMODE(metadata.st_mode),
                    b"",
                )
                visit(path)
            elif stat.S_ISREG(metadata.st_mode):
                result[relative] = (
                    "file",
                    stat.S_IMODE(metadata.st_mode),
                    path.read_bytes(),
                )
            else:
                result[relative] = (
                    "other",
                    stat.S_IMODE(metadata.st_mode),
                    b"",
                )

    visit(root)
    return result


def _logical_bytes(root: Path) -> int:
    return sum(
        os.lstat(path).st_size
        for path in root.rglob("*")
        if stat.S_ISREG(os.lstat(path).st_mode)
    )


def _confirm_args(record) -> list[str]:
    return ["purge", record.session_id, "--confirm", record.session_id]


def test_preview_is_exact_read_only_deterministic_and_secret_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    root = tmp_path / "assist"
    store, record = _record(
        root,
        tmp_path / "repo",
        label="cold-run",
        secret="DO-NOT-PRINT-THIS-SECRET",
    )
    nested = record.artifact_path / "nested"
    nested.mkdir()
    (nested / "b.bin").write_bytes(b"12345")
    (record.artifact_path / "a.txt").write_bytes(b"abc")
    before_tree = _tree_state(record.artifact_path)
    before_db = store.db_path.read_bytes()
    expected_bytes = _logical_bytes(record.artifact_path)

    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))
    assert main(["purge", record.session_id, "--preview"]) == 0
    first = capsys.readouterr().out
    assert main(["purge", record.session_id, "--preview"]) == 0
    second = capsys.readouterr().out

    assert first == second
    assert f"session_id: {record.session_id}" in first
    assert "purge_state: ready" in first
    assert f"estimated_bytes: {expected_bytes}" in first
    assert 'artifact: file "a.txt" bytes=3' in first
    assert 'artifact: dir "nested" bytes=0' in first
    assert 'artifact: file "nested/b.bin" bytes=5' in first
    assert "mutation: none" in first
    assert "DO-NOT-PRINT-THIS-SECRET" not in first
    assert "operator prompt secret" not in first
    assert "credential-secret-id" not in first
    assert _tree_state(record.artifact_path) == before_tree
    assert store.db_path.read_bytes() == before_db
    assert not (root / "purge-staging").exists()


@pytest.mark.parametrize("selector", ["latest", "last", "cold-run", "prefix"])
def test_preview_requires_one_full_immutable_session_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
):
    root = tmp_path / "assist"
    _store, record = _record(root, tmp_path / "repo", label="cold-run")
    if selector == "prefix":
        selector = record.session_id[:-2]
    before = _tree_state(record.artifact_path)
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))

    with pytest.raises(SystemExit, match="full immutable session ID"):
        main(["purge", selector, "--preview"])

    assert _tree_state(record.artifact_path) == before


@pytest.mark.parametrize(
    "args_kind,message",
    [
        ("missing", "requires --preview or --confirm"),
        ("implicit", "confirmation must be the same full immutable session ID"),
        ("prefix", "confirmation must be the same full immutable session ID"),
        ("other", "confirmation must be the same full immutable session ID"),
    ],
)
def test_purge_requires_exact_repeated_id_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args_kind: str,
    message: str,
):
    root = tmp_path / "assist"
    store, record = _record(root, tmp_path / "repo")
    before = _tree_state(record.artifact_path)
    before_db = store.db_path.read_bytes()
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))
    if args_kind == "missing":
        args = ["purge", record.session_id]
    elif args_kind == "implicit":
        args = ["purge", record.session_id, "--confirm", "yes"]
    elif args_kind == "prefix":
        args = ["purge", record.session_id, "--confirm", record.short_id]
    else:
        args = [
            "purge",
            record.session_id,
            "--confirm",
            "20260825_010203_deadbeef",
        ]

    with pytest.raises(SystemExit, match=message):
        main(args)

    assert _tree_state(record.artifact_path) == before
    assert store.db_path.read_bytes() == before_db


@pytest.mark.parametrize(
    "state,message",
    [
        ("ordinary", "must be archived"),
        ("running", "running session"),
        ("active", "active-session pointer"),
        ("locked", "locked session"),
        ("approval", "pending approval"),
        ("clarification", "pending clarification"),
        ("correction", "pending correction"),
    ],
)
def test_purge_refuses_unavailable_session_states_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    message: str,
):
    root = tmp_path / "assist"
    store, record = _record(
        root,
        tmp_path / "repo",
        archived=state != "ordinary",
    )
    if state == "active":
        store.set_active_session(record.cwd, record.session_id)
    elif state == "running":
        store.update_session(
            record.session_id,
            status="running",
            last_finish_reason=None,
        )
    elif state == "locked":
        store.acquire_session_lock(record.session_id)
    elif state == "approval":
        (record.artifact_path / "approval_request.json").write_text(
            json.dumps({"status": "pending"}) + "\n"
        )
    elif state == "clarification":
        create_clarification_request(
            record.artifact_path,
            request_id="request-purge",
            session_id=record.session_id,
            session_number=1,
            turn_number=1,
            tool_call_id="ask-purge",
            question="Which target?",
        )
    elif state == "correction":
        create_correction(
            record.artifact_path,
            correction_id="correction-purge",
            session_id=record.session_id,
            after_session_number=1,
            text="Use the exact target.",
        )
    before = _tree_state(record.artifact_path)
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))

    with pytest.raises(SystemExit, match=message):
        main(_confirm_args(record))

    assert store.get_session(record.session_id) is not None
    assert _tree_state(record.artifact_path) == before
    if state == "locked":
        store.release_session_lock(record.session_id)


def test_read_only_preview_ignores_but_does_not_remove_a_stale_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "assist"
    store, record = _record(root, tmp_path / "repo")
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            insert into session_locks (
                session_id, owner_host, owner_pid, acquired_at
            ) values (?, ?, ?, ?)
            """,
            (
                record.session_id,
                socket.gethostname(),
                424242,
                "2026-08-26T00:00:00+00:00",
            ),
        )
    before_db = store.db_path.read_bytes()
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))

    with patch(
        "scripts.llm_assist.store.os.kill",
        side_effect=ProcessLookupError,
    ):
        assert main(["purge", record.session_id, "--preview"]) == 0

    assert store.db_path.read_bytes() == before_db
    with sqlite3.connect(store.db_path) as connection:
        retained = connection.execute(
            "select 1 from session_locks where session_id = ?",
            (record.session_id,),
        ).fetchone()
    assert retained is not None


def test_success_removes_only_selected_row_and_owned_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    root = tmp_path / "assist"
    store, target = _record(root, tmp_path / "target", label="purge-me")
    _store, retained = _record(root, tmp_path / "retained", label="keep-me")
    retained_before = _tree_state(retained.artifact_path)
    credentials = root / "credentials" / "provider.json"
    credentials.parent.mkdir()
    credentials.write_text("host credential secret")
    measurement = tmp_path / "measurement" / "result.json"
    measurement.parent.mkdir()
    measurement.write_text('{"score": 1}\n')
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))

    assert main(_confirm_args(target)) == 0
    output = capsys.readouterr().out

    assert f"purged: {target.session_id}" in output
    assert "purge_state: completed" in output
    assert not target.artifact_path.exists()
    assert store.get_session(target.session_id) is None
    assert store.get_session(retained.session_id) is not None
    assert _tree_state(retained.artifact_path) == retained_before
    assert credentials.read_text() == "host credential secret"
    assert measurement.read_text() == '{"score": 1}\n'
    assert store.resolve_session_ref(target.session_id) is None
    assert store.resolve_session_ref(target.short_id) is None
    assert store.resolve_session_ref("purge-me") is None
    with sqlite3.connect(store.db_path) as connection:
        journal = connection.execute(
            """
            select phase, manifest_json, entry_count, estimated_bytes,
                   completed_at
            from session_purges where session_id = ?
            """,
            (target.session_id,),
        ).fetchone()
    assert journal is not None
    assert journal[0] == "completed"
    assert journal[1] == "[]"
    assert journal[2] > 0
    assert journal[3] > 0
    assert journal[4] is not None

    assert main(["sessions", "--archived"]) == 0
    listed = capsys.readouterr().out
    assert target.session_id not in listed
    assert retained.session_id in listed


@pytest.mark.parametrize(
    "command",
    [
        ["status"],
        ["show"],
        ["resume"],
        ["label", "new-label"],
        ["archive"],
        ["unarchive"],
        ["fork"],
        ["usage"],
        ["correct", "correction"],
        ["answer", "request-id", "answer"],
        ["rewind", "1"],
        ["approve"],
        ["reject"],
        ["worktree", "rm"],
    ],
)
def test_every_session_command_stops_resolving_a_purged_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: list[str],
):
    root = tmp_path / "assist"
    _store, record = _record(root, tmp_path / "repo")
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))
    assert main(_confirm_args(record)) == 0

    if command[:2] == ["worktree", "rm"]:
        argv = ["worktree", "rm", record.session_id]
    else:
        argv = [*command[:1], record.session_id, *command[1:]]
    with pytest.raises(SystemExit, match=f"unknown session: {record.session_id}"):
        main(argv)


@pytest.mark.parametrize(
    "defense,message",
    [
        ("file-symlink", "symbolic link"),
        ("directory-symlink", "symbolic link"),
        ("root-symlink", "symbolic link"),
        ("hard-link", "hard link"),
        ("fifo", "unsupported"),
        ("mount", "mount point"),
        ("file-mount", "mount point"),
        ("sessions-mount", "mount point"),
        ("row-path", "artifact boundary"),
        ("metadata", "session metadata"),
    ],
)
def test_preview_and_purge_fail_closed_on_unproved_artifact_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defense: str,
    message: str,
):
    root = tmp_path / "assist"
    store, record = _record(root, tmp_path / "repo")
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("outside secret")
    if defense == "file-symlink":
        (record.artifact_path / "unsafe").symlink_to(secret)
    elif defense == "directory-symlink":
        (record.artifact_path / "unsafe").symlink_to(outside, target_is_directory=True)
    elif defense == "root-symlink":
        moved = outside / "moved-session"
        record.artifact_path.rename(moved)
        record.artifact_path.symlink_to(moved, target_is_directory=True)
    elif defense == "hard-link":
        os.link(secret, record.artifact_path / "shared")
    elif defense == "fifo":
        os.mkfifo(record.artifact_path / "pipe")
    elif defense == "mount":
        (record.artifact_path / "mounted").mkdir()
        monkeypatch.setattr(
            "scripts.llm_assist.purge.os.path.ismount",
            lambda path: Path(path).name == "mounted",
        )
    elif defense == "file-mount":
        (record.artifact_path / "mounted-file").write_text("mounted data")
        monkeypatch.setattr(
            "scripts.llm_assist.purge.os.path.ismount",
            lambda path: Path(path).name == "mounted-file",
        )
    elif defense == "sessions-mount":
        sessions = root / "sessions"
        monkeypatch.setattr(
            "scripts.llm_assist.purge.os.path.ismount",
            lambda path: Path(path) == sessions,
        )
    elif defense == "row-path":
        with sqlite3.connect(store.db_path) as connection:
            connection.execute(
                "update sessions set artifact_dir = ? where session_id = ?",
                (str(outside), record.session_id),
            )
    else:
        (record.artifact_path / "session.json").write_text("not json\n")
    outside_before = _tree_state(outside)
    target_before = _tree_state(record.artifact_path)
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))

    with pytest.raises(SystemExit, match=message):
        main(["purge", record.session_id, "--preview"])
    with pytest.raises(SystemExit, match=message):
        main(_confirm_args(record))

    assert store.get_session(record.session_id) is not None
    assert _tree_state(record.artifact_path) == target_before
    assert _tree_state(outside) == outside_before


def test_directory_swap_to_symlink_never_reads_outside(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "assist"
    store, record = _record(root, tmp_path / "repo")
    nested = record.artifact_path / "nested"
    nested.mkdir()
    (nested / "inside.txt").write_text("inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside secret")
    outside_before = _tree_state(outside)
    real_open = os.open
    swapped = False

    def swap_before_open(
        path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if not swapped and path == "nested" and dir_fd is not None:
            moved = record.artifact_path / "nested-original"
            nested.rename(moved)
            nested.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))
    monkeypatch.setattr("scripts.llm_assist.purge.os.open", swap_before_open)

    with pytest.raises(SystemExit, match="cannot open saved artifact directory"):
        main(["purge", record.session_id, "--preview"])

    assert swapped is True
    assert store.get_session(record.session_id) is not None
    assert _tree_state(outside) == outside_before


def test_non_utf8_artifact_name_is_rejected_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "assist"
    store, record = _record(root, tmp_path / "repo")
    directory_fd = os.open(
        record.artifact_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        artifact_fd = os.open(
            b"\xff",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        os.close(artifact_fd)
        monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))

        with pytest.raises(SystemExit, match="valid UTF-8"):
            main(["purge", record.session_id, "--preview"])
        with pytest.raises(SystemExit, match="valid UTF-8"):
            main(_confirm_args(record))

        assert store.get_session(record.session_id) is not None
        assert os.stat(b"\xff", dir_fd=directory_fd).st_size == 0
    finally:
        os.close(directory_fd)


def test_managed_worktree_must_be_removed_by_its_separate_action_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "assist"
    store, record = _record(root, tmp_path / "repo")
    worktree = tmp_path / "repo" / ".yuj_worktrees" / record.session_id
    worktree.mkdir(parents=True)
    (worktree / "keep.txt").write_text("worktree data")
    store.update_session_worktree(
        record.session_id,
        path=worktree,
        branch=f"yuj/session-{record.session_id}",
        base_commit="a" * 40,
    )
    record = store.get_session(record.session_id)
    assert record is not None
    before = _tree_state(record.artifact_path)
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))

    with patch(
        "scripts.llm_assist.__main__.remove_session_worktree"
    ) as remove_worktree:
        with pytest.raises(SystemExit, match="yuj worktree rm"):
            main(_confirm_args(record))

    remove_worktree.assert_not_called()
    assert _tree_state(record.artifact_path) == before
    assert (worktree / "keep.txt").read_text() == "worktree data"
    assert store.get_session(record.session_id) is not None


def test_removed_worktree_identity_becomes_history_and_allows_purge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo = _repo(tmp_path)
    root = tmp_path / "assist"
    store, record = _record(root, repo, archived=False)
    worktree = create_session_worktree(
        repo,
        mode="auto",
        run_id=record.session_id,
    )
    assert worktree is not None
    store.update_session_worktree(
        record.session_id,
        path=worktree.worktree_path,
        branch=worktree.branch,
        base_commit=worktree.base_commit,
    )
    store.archive_session(record.session_id)
    record = store.get_session(record.session_id)
    assert record is not None
    saved_identity = (
        record.worktree_path,
        record.worktree_branch,
        record.worktree_base_commit,
    )
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))

    with pytest.raises(SystemExit, match="live managed worktree"):
        main(_confirm_args(record))
    assert main(["unarchive", record.session_id]) == 0
    assert main(["worktree", "rm", record.session_id]) == 0
    historical = store.get_session(record.session_id)
    assert historical is not None
    assert (
        historical.worktree_path,
        historical.worktree_branch,
        historical.worktree_base_commit,
    ) == saved_identity
    assert not worktree.worktree_path.exists()
    assert main(["archive", record.session_id]) == 0

    assert main(_confirm_args(record)) == 0
    assert store.get_session(record.session_id) is None


def test_malformed_empty_worktree_identity_is_not_treated_as_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "assist"
    store, record = _record(root, tmp_path / "repo")
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            update sessions
            set worktree_path = '', worktree_branch = '',
                worktree_base_commit = ''
            where session_id = ?
            """,
            (record.session_id,),
        )
    before = _tree_state(record.artifact_path)
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))

    with pytest.raises(SystemExit, match="worktree metadata is malformed"):
        main(_confirm_args(record))

    assert store.get_session(record.session_id) is not None
    assert _tree_state(record.artifact_path) == before


@pytest.mark.parametrize(
    "boundary",
    [
        "after_journal_prepare",
        "after_artifact_stage",
        "after_artifact_entry_remove",
        "after_artifact_root_remove",
        "before_session_row_remove",
    ],
)
def test_injected_failure_reports_remaining_state_and_retry_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    boundary: str,
):
    root = tmp_path / "assist"
    store, target = _record(root, tmp_path / "target")
    _store, retained = _record(root, tmp_path / "retained")
    retained_before = _tree_state(retained.artifact_path)
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))
    failed = False

    def inject(selected: str) -> None:
        nonlocal failed
        if selected == boundary and not failed:
            failed = True
            raise RuntimeError(f"injected at {boundary}")

    with patch("scripts.llm_assist.purge._purge_boundary", side_effect=inject):
        with pytest.raises(SystemExit, match="purge incomplete"):
            main(_confirm_args(target))
    failure_output = capsys.readouterr().out
    assert "purge_state:" in failure_output
    assert "remaining_entries:" in failure_output
    assert "remaining_bytes:" in failure_output
    remaining_entries = int(
        next(
            line.partition(":")[2].strip()
            for line in failure_output.splitlines()
            if line.startswith("remaining_entries:")
        )
    )
    assert sum(
        line.startswith("artifact:") for line in failure_output.splitlines()
    ) == remaining_entries
    assert store.get_session(target.session_id) is not None
    assert _tree_state(retained.artifact_path) == retained_before
    with pytest.raises(SessionPurgeInProgressError, match="purge is incomplete"):
        store.resolve_session_ref(target.session_id)
    assert target.session_id not in {
        item.session_id for item in store.list_sessions(archived=True)
    }
    with pytest.raises(SystemExit, match="session purge is incomplete"):
        main(["status", target.session_id])

    assert main(["purge", target.session_id, "--preview"]) == 0
    preview = capsys.readouterr().out
    assert "purge_state:" in preview
    assert f"last_failure: RuntimeError: injected at {boundary}" in preview
    assert "mutation: none" in preview

    assert main(_confirm_args(target)) == 0
    completed = capsys.readouterr().out
    assert "purge_state: completed" in completed
    assert store.get_session(target.session_id) is None
    assert not target.artifact_path.exists()
    assert _tree_state(retained.artifact_path) == retained_before


def test_session_row_change_before_finalization_is_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "assist"
    store, record = _record(root, tmp_path / "repo")
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))
    changed = False

    def change_row(boundary: str) -> None:
        nonlocal changed
        if boundary == "before_session_row_remove" and not changed:
            changed = True
            store.set_session_label(record.session_id, "changed-during-purge")

    with patch(
        "scripts.llm_assist.purge._purge_boundary",
        side_effect=change_row,
    ):
        with pytest.raises(
            SystemExit,
            match="session metadata changed before purge finalization",
        ):
            main(_confirm_args(record))

    retained = store.get_session(record.session_id)
    assert retained is not None
    assert retained.label == "changed-during-purge"
    assert not record.artifact_path.exists()
    journal = store.get_session_purge(record.session_id)
    assert journal is not None and journal.phase == "artifacts_removed"

    assert main(_confirm_args(record)) == 0
    assert store.get_session(record.session_id) is None


def test_keyboard_interrupt_is_journaled_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    root = tmp_path / "assist"
    store, record = _record(root, tmp_path / "repo")
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))

    def interrupt(boundary: str) -> None:
        if boundary == "after_artifact_stage":
            raise KeyboardInterrupt

    with patch("scripts.llm_assist.purge._purge_boundary", side_effect=interrupt):
        with pytest.raises(SystemExit, match="purge incomplete"):
            main(_confirm_args(record))
    output = capsys.readouterr().out
    assert "remaining_entries:" in output
    assert store.get_session(record.session_id) is not None

    assert main(_confirm_args(record)) == 0
    assert store.get_session(record.session_id) is None


def test_malformed_journal_path_is_inspectable_but_never_used_for_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "assist"
    store, record = _record(root, tmp_path / "repo")
    before = _tree_state(record.artifact_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("must remain")
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))

    def interrupt(boundary: str) -> None:
        if boundary == "after_journal_prepare":
            raise RuntimeError("stop before staging")

    with patch("scripts.llm_assist.purge._purge_boundary", side_effect=interrupt):
        with pytest.raises(SystemExit, match="purge incomplete"):
            main(_confirm_args(record))

    with sqlite3.connect(store.db_path) as connection:
        raw = connection.execute(
            "select manifest_json from session_purges where session_id = ?",
            (record.session_id,),
        ).fetchone()
        assert raw is not None
        manifest = json.loads(raw[0])
        manifest[0]["relative"] = "../outside.txt"
        malformed = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(malformed.encode("utf-8")).hexdigest()
        connection.execute(
            """
            update session_purges
            set manifest_json = ?, manifest_digest = ?
            where session_id = ?
            """,
            (malformed, digest, record.session_id),
        )

    with pytest.raises(SystemExit, match="journal path is malformed"):
        main(["purge", record.session_id, "--preview"])
    with pytest.raises(SystemExit, match="journal path is malformed"):
        main(_confirm_args(record))

    assert _tree_state(record.artifact_path) == before
    assert outside.read_text() == "must remain"


def test_interruption_after_final_transaction_reports_completed_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    root = tmp_path / "assist"
    store, record = _record(root, tmp_path / "repo")
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))

    def interrupt(boundary: str) -> None:
        if boundary == "after_session_row_remove":
            raise KeyboardInterrupt

    with patch("scripts.llm_assist.purge._purge_boundary", side_effect=interrupt):
        assert main(_confirm_args(record)) == 0

    output = capsys.readouterr().out
    assert "purge_state: completed" in output
    assert "remaining_entries: 0" in output
    assert "remaining_bytes: 0" in output
    assert store.get_session(record.session_id) is None
    assert not record.artifact_path.exists()


@pytest.mark.parametrize("purged", ["parent", "child"])
def test_parent_and_child_rows_and_artifacts_remain_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    purged: str,
):
    root = tmp_path / "assist"
    store, parent = _record(root, tmp_path / "repo", archived=False)
    store.acquire_session_lock(parent.session_id)
    child = store.prepare_forked_session(
        parent,
        config_paths=[],
        system_prompt_path=None,
    )
    store.insert_forked_session(child, expected_parent=parent)
    store.release_session_lock(parent.session_id)
    _write_artifacts(child, secret="child artifact")
    store.update_session(
        child.session_id,
        status="paused",
        last_finish_reason="forked",
    )
    parent = store.get_session(parent.session_id)
    child = store.get_session(child.session_id)
    assert parent is not None and child is not None
    target = parent if purged == "parent" else child
    survivor = child if purged == "parent" else parent
    store.archive_session(target.session_id)
    target = store.get_session(target.session_id)
    assert target is not None
    survivor_before = _tree_state(survivor.artifact_path)
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))

    assert main(_confirm_args(target)) == 0

    retained = store.get_session(survivor.session_id)
    assert retained is not None
    assert _tree_state(survivor.artifact_path) == survivor_before
    if purged == "parent":
        assert retained.parent_session_id == target.session_id
        assert store.resolve_session_ref(target.session_id) is None
    else:
        assert retained.parent_session_id is None


def test_purge_has_no_model_tool_worktree_or_measurement_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "assist"
    _store, record = _record(root, tmp_path / "repo")
    measurement = tmp_path / "measurement"
    measurement.mkdir()
    (measurement / ".trace.jsonl").write_text('{"event":"score"}\n')
    before = _tree_state(measurement)
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))

    with (
        patch("scripts.llm_assist.__main__._make_client") as model,
        patch("scripts.llm_assist.__main__.run_session") as run_session,
        patch("scripts.llm_assist.__main__.remove_session_worktree") as worktree,
        patch("scripts.llm_solver.__main__.main") as measurement_entrypoint,
    ):
        assert main(["purge", record.session_id, "--preview"]) == 0
        assert main(_confirm_args(record)) == 0

    model.assert_not_called()
    run_session.assert_not_called()
    worktree.assert_not_called()
    measurement_entrypoint.assert_not_called()
    assert _tree_state(measurement) == before


def test_preview_entry_count_is_bounded_and_does_not_mutate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "assist"
    store, record = _record(root, tmp_path / "repo")
    before = _tree_state(record.artifact_path)
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))
    monkeypatch.setattr("scripts.llm_assist.purge.MAX_ARTIFACT_ENTRIES", 2)

    with pytest.raises(SystemExit, match="too many artifact entries"):
        main(["purge", record.session_id, "--preview"])

    assert store.get_session(record.session_id) is not None
    assert _tree_state(record.artifact_path) == before


def test_completed_journal_with_a_live_row_never_claims_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    root = tmp_path / "assist"
    store, record = _record(root, tmp_path / "repo")
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            insert into session_purges (
                session_id, created_at, updated_at, phase, manifest_json,
                manifest_digest, entry_count, estimated_bytes, root_dev,
                root_ino, failure_detail, completed_at
            ) values (?, ?, ?, 'completed', '[]', ?, 1, 1, 1, 1, null, ?)
            """,
            (
                record.session_id,
                "2026-08-26T00:00:00+00:00",
                "2026-08-26T00:00:00+00:00",
                "0" * 64,
                "2026-08-26T00:00:00+00:00",
            ),
        )
    before = _tree_state(record.artifact_path)
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))

    with pytest.raises(SystemExit, match="completed purge journal still has"):
        main(["purge", record.session_id, "--preview"])
    with pytest.raises(SystemExit, match="completed purge journal still has"):
        main(_confirm_args(record))
    with pytest.raises(SystemExit, match="purge state is inconsistent"):
        main(["status", record.session_id])
    assert main(["sessions", "--archived"]) == 0
    assert record.session_id not in capsys.readouterr().out

    assert store.get_session(record.session_id) is not None
    assert _tree_state(record.artifact_path) == before


def test_completed_journal_with_a_reappeared_artifact_boundary_never_claims_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "assist"
    store, record = _record(root, tmp_path / "repo")
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(root))
    assert main(_confirm_args(record)) == 0
    record.artifact_path.mkdir()
    unexpected = record.artifact_path / "unexpected.txt"
    unexpected.write_text("must remain")

    with pytest.raises(SystemExit, match="still has an artifact boundary"):
        main(["purge", record.session_id, "--preview"])
    with pytest.raises(SystemExit, match="still has an artifact boundary"):
        main(_confirm_args(record))

    assert store.get_session(record.session_id) is None
    assert unexpected.read_text() == "must remain"
