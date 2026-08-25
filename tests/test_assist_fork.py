"""Saved-session fork isolation, lineage, refusal, and recovery tests."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import sqlite3
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.llm_assist.__main__ import main
from scripts.llm_assist._images import PendingImage, save_image_segment
from scripts.llm_assist.forking import ForkSessionError, fork_saved_session
from scripts.llm_assist.runner import (
    _resolve_session_worktree,
    _write_session_metadata,
    create_session,
    run_session,
    save_approval_request,
)
from scripts.llm_assist.store import SessionStore
from scripts.llm_solver.harness.clarifications import (
    create_clarification_request,
    record_clarification_answer,
)
from scripts.llm_solver.harness.corrections import (
    consume_correction,
    create_correction,
)
from scripts.llm_solver.harness.worktree_runtime import inspect_session_worktree
from scripts.llm_solver.harness.workspace_checkpoints import (
    WorkspaceCheckpointStore,
)


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4"
    "z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
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


def _row(store: SessionStore, session_id: str) -> dict[str, object]:
    with sqlite3.connect(store.db_path) as connection:
        connection.row_factory = sqlite3.Row
        value = connection.execute(
            "select * from sessions where session_id = ?",
            (session_id,),
        ).fetchone()
    assert value is not None
    return dict(value)


def _rows(store: SessionStore) -> list[dict[str, object]]:
    with sqlite3.connect(store.db_path) as connection:
        connection.row_factory = sqlite3.Row
        values = connection.execute(
            "select * from sessions order by session_id"
        ).fetchall()
    return [dict(value) for value in values]


def _tree_hashes(root: Path) -> dict[str, tuple[str, int]]:
    """Hash every entry without following links."""
    root = Path(root)
    if not root.exists() and not root.is_symlink():
        return {}
    result: dict[str, tuple[str, int]] = {}

    def visit(directory: Path) -> None:
        for entry in sorted(os.scandir(directory), key=lambda item: item.name):
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            metadata = os.lstat(path)
            if stat.S_ISLNK(metadata.st_mode):
                payload = os.fsencode(os.readlink(path))
                kind = "link"
            elif stat.S_ISDIR(metadata.st_mode):
                payload = b""
                kind = "dir"
            elif stat.S_ISREG(metadata.st_mode):
                payload = path.read_bytes()
                kind = "file"
            else:
                payload = f"mode:{metadata.st_mode}".encode()
                kind = "other"
            digest = hashlib.sha256(kind.encode() + b"\0" + payload).hexdigest()
            result[relative] = (digest, stat.S_IMODE(metadata.st_mode))
            if kind == "dir":
                visit(path)

    visit(root)
    return result


def _tree_digest(root: Path) -> str:
    payload = json.dumps(
        _tree_hashes(root), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_stopped_trace(artifact_dir: Path, *, finish_reason: str = "max_turns") -> bytes:
    rows = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 1,
            "tool_name": "read",
            "args_summary": "README.md",
            "result_summary": "base",
        },
        {
            "event": "session_end",
            "session_number": 1,
            "finish_reason": finish_reason,
            "turns": 1,
        },
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (artifact_dir / ".trace.jsonl").write_bytes(payload)
    (artifact_dir / "transcript.log").write_text(
        "=== turn 001 input ===\n"
        '{"messages":[{"role":"user","content":"task"}]}\n'
        "=== turn 001 output ===\n"
        '{"choices":[{"message":{"role":"assistant","content":"pause"}}]}\n'
    )
    return payload


def _source_session(
    store: SessionStore,
    cwd: Path,
    *,
    status: str = "paused",
    with_attachment: bool = False,
    with_checkpoint: bool = False,
    worktree_mode: str = "off",
):
    record = create_session(
        store,
        cwd=cwd,
        prompt_text="Complete the task.",
        prompt_source="test",
        model="test-model",
        config_paths=[],
        system_prompt_path=None,
        context_mode="full",
    )
    artifact_dir = record.artifact_path
    provider = artifact_dir / "provider.toml"
    provider.write_text(f'[runtime]\nworktree = "{worktree_mode}"\n')
    store.update_session_config_paths(record.session_id, [provider])
    record = store.get_session(record.session_id)
    assert record is not None
    _write_session_metadata(record)
    _write_stopped_trace(artifact_dir)
    (artifact_dir / ".solver" / "state.json").write_text(
        json.dumps({
            "state": {"current_attempt": "inspect", "last_verify": "tests"},
            "trace": [{"turn": 1}],
            "gates": [],
            "evidence": ["saved endpoint"],
            "inference": [],
        }, indent=2) + "\n"
    )
    (artifact_dir / "system_log.jsonl").write_text('{"event":"saved"}\n')
    (artifact_dir / "savings.jsonl").write_text('{"bucket":"test"}\n')
    (artifact_dir / "approval_decisions.json").write_text(
        json.dumps({"bash:sha256:test": "approved"}, indent=2) + "\n"
    )
    if with_attachment:
        image = PendingImage(
            display_name="pixel.png",
            media_type="image/png",
            data=PNG_1X1,
            size_bytes=len(PNG_1X1),
            sha256=hashlib.sha256(PNG_1X1).hexdigest(),
            width=1,
            height=1,
        )
        save_image_segment(
            artifact_dir,
            segment_number=1,
            prompt_text=record.prompt_text,
            images=[image],
        )
    if with_checkpoint:
        WorkspaceCheckpointStore(
            cwd,
            shadow_dir=artifact_dir / ".shadow_git",
        ).capture(1)
    store.update_session(
        record.session_id,
        status=status,
        last_finish_reason="max_turns",
    )
    store.clear_active_session(cwd, session_id=record.session_id)
    saved = store.get_session(record.session_id)
    assert saved is not None
    return saved


def _assert_no_partial_child(store: SessionStore, source_id: str) -> None:
    assert [row["session_id"] for row in _rows(store)] == [source_id]
    session_root = store.root / "sessions"
    assert {path.name for path in session_root.iterdir()} == {source_id}


def test_store_migrates_nullable_immutable_parent_identity(tmp_path: Path):
    root = tmp_path / "assist"
    root.mkdir()
    with sqlite3.connect(root / "sessions.sqlite3") as connection:
        connection.execute(
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
    record = store.create_session(
        cwd=tmp_path / "work",
        model="model",
        prompt_text="task",
        prompt_source="test",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    saved = store.get_session(record.session_id)
    assert saved is not None
    assert saved.parent_session_id is None
    with sqlite3.connect(store.db_path) as connection:
        columns = {
            row[1] for row in connection.execute("pragma table_info(sessions)")
        }
        assert "parent_session_id" in columns
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "update sessions set parent_session_id = ? where session_id = ?",
                ("different", record.session_id),
            )


def test_fork_copies_owned_history_records_exact_parent_and_resumes(tmp_path: Path):
    repo = _repo(tmp_path)
    store = SessionStore(tmp_path / "assist")
    source = _source_session(
        store,
        repo,
        with_attachment=True,
        with_checkpoint=True,
    )
    source_trace = (source.artifact_path / ".trace.jsonl").read_bytes()
    source_hashes = _tree_hashes(source.artifact_path)
    source_digest = _tree_digest(source.artifact_path)
    source_row = _row(store, source.session_id)

    result = fork_saved_session(store, source)
    child = result.child

    assert child.session_id != source.session_id
    assert child.parent_session_id == source.session_id
    assert child.status == "paused"
    assert child.last_finish_reason == "forked"
    assert child.label is None
    assert child.archived_at is None
    assert child.artifact_path != source.artifact_path
    assert child.config_paths == [str(child.artifact_path / "provider.toml")]
    assert _tree_hashes(source.artifact_path) == source_hashes
    assert _tree_digest(source.artifact_path) == source_digest
    assert _row(store, source.session_id) == source_row
    assert result.source_artifact_sha256 == source_digest

    child_trace = (child.artifact_path / ".trace.jsonl").read_bytes()
    assert child_trace.startswith(source_trace)
    fork_event = json.loads(child_trace[len(source_trace):])
    assert fork_event["event"] == "session_fork"
    assert fork_event["session_id"] == child.session_id
    assert fork_event["parent_session_id"] == source.session_id
    metadata = json.loads((child.artifact_path / "session.json").read_text())
    assert metadata["session_id"] == child.session_id
    assert metadata["parent_session_id"] == source.session_id
    assert metadata["config_paths"] == child.config_paths

    copied_names = {
        "prompt.txt",
        ".solver/state.json",
        "provider.toml",
        "attachments.json",
        "attachments/segment-0001/image-0001.png",
        ".shadow_git/HEAD",
        "system_log.jsonl",
        "savings.jsonl",
        "approval_decisions.json",
        "transcript.log",
    }
    for relative in copied_names:
        source_path = source.artifact_path / relative
        child_path = child.artifact_path / relative
        assert child_path.read_bytes() == source_path.read_bytes()
        assert child_path.stat().st_ino != source_path.stat().st_ino

    checkpoint = WorkspaceCheckpointStore(
        repo,
        shadow_dir=child.artifact_path / ".shadow_git",
    )
    assert len(checkpoint.checkpoint_for_turn(1)) >= 40

    client = MagicMock()
    client.query_server_context.return_value = None
    observed: dict[str, object] = {}

    def fake_solve(work_dir, _cfg, _client, **kwargs):
        observed["work_dir"] = Path(work_dir)
        observed.update(kwargs)
        return True

    with (
        patch("scripts.llm_assist.runner._load_profile", return_value=None),
        patch("scripts.llm_assist.runner._make_client", return_value=client),
        patch("scripts.llm_assist.runner._require_image_capability"),
        patch("scripts.llm_assist.runner.build_model_role_runtime"),
        patch("scripts.llm_assist.runner.solve_task", side_effect=fake_solve),
    ):
        success, _reason = run_session(store, child, resume=True)
    assert success is True
    assert observed["work_dir"] == repo
    assert observed["artifacts_dir"] == child.artifact_path
    assert observed["resume_from_artifacts"] is True
    client.set_session_id.assert_called_once_with(child.session_id)
    assert _tree_hashes(source.artifact_path) == source_hashes
    assert _row(store, source.session_id) == source_row


def test_fork_of_fork_keeps_lineage_config_and_consumed_correction(tmp_path: Path):
    repo = _repo(tmp_path)
    store = SessionStore(tmp_path / "assist")
    source = _source_session(store, repo)
    correction = create_correction(
        source.artifact_path,
        correction_id="ancestor-correction",
        session_id=source.session_id,
        after_session_number=1,
        text="Keep the verified ancestor decision.",
    )
    consumption = consume_correction(
        source.artifact_path,
        correction_id=correction["correction_id"],
        session_number=2,
        turn_number=0,
        delivery="resume",
    )
    with (source.artifact_path / ".trace.jsonl").open("a") as trace:
        trace.write(json.dumps({
            "event": "correction_created",
            "session_number": 1,
            "correction_id": correction["correction_id"],
            "text_sha256": correction["text_sha256"],
            "text_chars": len(correction["text"]),
        }) + "\n")
        trace.write(json.dumps({
            "event": "correction_consumed",
            "session_number": consumption["session_number"],
            "turn_number": consumption["turn_number"],
            "transcript_segment": consumption["transcript_segment"],
            "correction_id": correction["correction_id"],
            "text_sha256": correction["text_sha256"],
            "delivery": consumption["delivery"],
        }) + "\n")
    source_before = _tree_hashes(source.artifact_path)
    source_row = _row(store, source.session_id)

    child = fork_saved_session(store, source).child
    child_before = _tree_hashes(child.artifact_path)
    child_row = _row(store, child.session_id)
    grandchild = fork_saved_session(store, child).child

    assert child.parent_session_id == source.session_id
    assert grandchild.parent_session_id == child.session_id
    assert grandchild.config_paths == [
        str(grandchild.artifact_path / "provider.toml")
    ]
    metadata = json.loads(
        (grandchild.artifact_path / "session.json").read_text()
    )
    provider_path = grandchild.artifact_path / "provider.toml"
    assert metadata["config_paths"] == [str(provider_path)]
    assert metadata["config_path_hashes"] == {
        str(provider_path): hashlib.sha256(provider_path.read_bytes()).hexdigest()
    }
    events = [
        json.loads(line)
        for line in (
            grandchild.artifact_path / ".trace.jsonl"
        ).read_text().splitlines()
    ]
    assert [
        (event["session_id"], event["parent_session_id"])
        for event in events
        if event.get("event") == "session_fork"
    ] == [
        (child.session_id, source.session_id),
        (grandchild.session_id, child.session_id),
    ]

    client = MagicMock()
    client.query_server_context.return_value = None
    with (
        patch("scripts.llm_assist.runner._load_profile", return_value=None),
        patch("scripts.llm_assist.runner._make_client", return_value=client),
        patch("scripts.llm_assist.runner.build_model_role_runtime"),
        patch("scripts.llm_assist.runner.solve_task", return_value=True),
    ):
        success, _reason = run_session(store, grandchild, resume=True)
    assert success is True
    assert _tree_hashes(source.artifact_path) == source_before
    assert _row(store, source.session_id) == source_row
    assert _tree_hashes(child.artifact_path) == child_before
    assert _row(store, child.session_id) == child_row


def test_existing_child_artifact_collision_is_never_removed(tmp_path: Path):
    repo = _repo(tmp_path)
    store = SessionStore(tmp_path / "assist")
    source = _source_session(store, repo)
    collision_id = "20260825_000000_deadbeef"
    collision = store.root / "sessions" / collision_id
    collision.mkdir()
    sentinel = collision / "foreign-evidence.bin"
    sentinel.write_bytes(b"must survive\0")
    source_before = _tree_hashes(source.artifact_path)
    source_row = _row(store, source.session_id)

    with (
        patch(
            "scripts.llm_assist.store._new_session_id",
            return_value=collision_id,
        ),
        pytest.raises(ForkSessionError, match="already exists"),
    ):
        fork_saved_session(store, source)

    assert sentinel.read_bytes() == b"must survive\0"
    assert _tree_hashes(source.artifact_path) == source_before
    assert _row(store, source.session_id) == source_row
    assert [row["session_id"] for row in _rows(store)] == [source.session_id]


def test_cli_fork_is_explicit_prints_lineage_and_never_runs_a_model(
    tmp_path: Path, capsys,
):
    repo = _repo(tmp_path)
    store = SessionStore(tmp_path / "assist")
    source = _source_session(store, repo)

    with (
        patch("scripts.llm_assist.__main__.SessionStore", return_value=store),
        patch("scripts.llm_assist.__main__.run_session") as assistant_loop,
        patch("scripts.llm_solver.__main__.main") as measurement_entrypoint,
    ):
        assert main(["fork", source.short_id]) == 0
    output = capsys.readouterr().out
    child = next(
        record
        for record in store.list_sessions(limit=10)
        if record.session_id != source.session_id
    )
    assert f"forked: {child.session_id}" in output
    assert f"parent_session_id: {source.session_id}" in output
    assert f"resume with: yuj resume {child.short_id}" in output
    assistant_loop.assert_not_called()
    measurement_entrypoint.assert_not_called()

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert main(["status", child.session_id]) == 0
        assert main(["show", child.session_id]) == 0
    detail = capsys.readouterr().out
    assert detail.count(f"parent_session_id: {source.session_id}") == 2


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ("active", "active"),
        ("running", "running"),
        ("locked", "locked"),
        ("approval", "pending approval"),
        ("clarification", "pending clarification"),
        ("answered_clarification", "pending clarification"),
        ("correction", "pending correction"),
        ("archived", "unarchive"),
    ],
)
def test_fork_refuses_unresolved_or_unavailable_source_without_mutation(
    tmp_path: Path,
    state: str,
    message: str,
):
    repo = _repo(tmp_path)
    store = SessionStore(tmp_path / "assist")
    source = _source_session(store, repo)
    if state == "active":
        store.set_active_session(repo, source.session_id)
    elif state == "running":
        store.update_session(
            source.session_id,
            status="running",
            last_finish_reason=None,
        )
        source = store.get_session(source.session_id)
        assert source is not None
    elif state == "locked":
        store.acquire_session_lock(source.session_id)
    elif state == "approval":
        save_approval_request(source.artifact_path, {
            "status": "pending",
            "action_key": "bash:sha256:test",
            "tool_name": "bash",
            "args_summary": "rm build",
            "reason": "destructive",
            "requested_at": 1.0,
            "cmd": "rm build",
        })
    elif state in {"clarification", "answered_clarification"}:
        create_clarification_request(
            source.artifact_path,
            request_id="question-1",
            session_id=source.session_id,
            session_number=1,
            turn_number=1,
            tool_call_id="call-1",
            question="Which target?",
        )
        if state == "answered_clarification":
            record_clarification_answer(
                source.artifact_path,
                session_id=source.session_id,
                request_id="question-1",
                answer="The first target.",
            )
    elif state == "correction":
        correction = create_correction(
            source.artifact_path,
            correction_id="corr-1",
            session_id=source.session_id,
            after_session_number=1,
            text="Use the other file.",
        )
        with (source.artifact_path / ".trace.jsonl").open("a") as trace:
            trace.write(json.dumps({
                "event": "correction_created",
                "session_number": 1,
                "correction_id": correction["correction_id"],
                "text_sha256": correction["text_sha256"],
                "text_chars": len(correction["text"]),
            }) + "\n")
    elif state == "archived":
        archived, changed = store.archive_session(source.session_id)
        assert changed is True
        source = archived

    row_before = _row(store, source.session_id)
    tree_before = _tree_hashes(source.artifact_path)
    with pytest.raises(ForkSessionError, match=message):
        fork_saved_session(store, source)
    assert _row(store, source.session_id) == row_before
    assert _tree_hashes(source.artifact_path) == tree_before
    _assert_no_partial_child(store, source.session_id)
    if state == "locked":
        store.release_session_lock(source.session_id)


def test_archived_cli_refusal_names_the_exact_unarchive_action(tmp_path: Path):
    repo = _repo(tmp_path)
    store = SessionStore(tmp_path / "assist")
    source = _source_session(store, repo)
    store.archive_session(source.session_id)

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        with pytest.raises(
            SystemExit,
            match=rf"yuj unarchive {source.short_id}",
        ):
            main(["fork", source.short_id])
        with pytest.raises(SystemExit, match="explicit session reference"):
            main(["fork", "latest"])


@pytest.mark.parametrize("kind", ["file_symlink", "directory_symlink"])
def test_fork_rejects_any_artifact_symlink_without_reading_outside_boundary(
    tmp_path: Path,
    kind: str,
):
    repo = _repo(tmp_path)
    store = SessionStore(tmp_path / "assist")
    source = _source_session(store, repo)
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("do not copy\n")
    link = source.artifact_path / "unsafe"
    link.symlink_to(secret if kind == "file_symlink" else outside)
    before = _tree_hashes(source.artifact_path)
    row_before = _row(store, source.session_id)

    with pytest.raises(ForkSessionError, match="symbolic link"):
        fork_saved_session(store, source)

    assert secret.read_text() == "do not copy\n"
    assert _tree_hashes(source.artifact_path) == before
    assert _row(store, source.session_id) == row_before
    _assert_no_partial_child(store, source.session_id)


def test_fork_rejects_artifact_row_outside_owned_session_boundary(tmp_path: Path):
    repo = _repo(tmp_path)
    store = SessionStore(tmp_path / "assist")
    source = _source_session(store, repo)
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("outside bytes\n")
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "update sessions set artifact_dir = ? where session_id = ?",
            (str(outside), source.session_id),
        )
    malformed = store.get_session(source.session_id)
    assert malformed is not None
    row_before = _row(store, source.session_id)
    outside_before = _tree_hashes(outside)

    with pytest.raises(ForkSessionError, match="artifact boundary"):
        fork_saved_session(store, malformed)

    assert _tree_hashes(outside) == outside_before
    assert _row(store, source.session_id) == row_before
    assert len(_rows(store)) == 1


def test_fork_rejects_malformed_attachment_path_and_session_identity(tmp_path: Path):
    repo = _repo(tmp_path)
    store = SessionStore(tmp_path / "assist")
    source = _source_session(store, repo, with_attachment=True)
    manifest_path = source.artifact_path / "attachments.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["segments"][0]["images"][0]["relative_path"] = "../../secret.png"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    before = _tree_hashes(source.artifact_path)

    with pytest.raises(ForkSessionError, match="attachment saved path"):
        fork_saved_session(store, source)
    assert _tree_hashes(source.artifact_path) == before
    _assert_no_partial_child(store, source.session_id)

    manifest["segments"][0]["images"][0]["relative_path"] = (
        "attachments/segment-0001/image-0001.png"
    )
    manifest_path.write_text(json.dumps(manifest) + "\n")
    session_path = source.artifact_path / "session.json"
    metadata = json.loads(session_path.read_text())
    metadata["session_id"] = "20260825_000000_deadbeef"
    session_path.write_text(json.dumps(metadata) + "\n")
    before = _tree_hashes(source.artifact_path)
    with pytest.raises(ForkSessionError, match="session metadata identity"):
        fork_saved_session(store, source)
    assert _tree_hashes(source.artifact_path) == before
    _assert_no_partial_child(store, source.session_id)


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("injected commit interruption"), KeyboardInterrupt()],
    ids=["exception", "keyboard-interrupt"],
)
def test_failed_child_row_commit_removes_staged_and_published_artifacts(
    tmp_path: Path,
    failure: BaseException,
):
    repo = _repo(tmp_path)
    store = SessionStore(tmp_path / "assist")
    source = _source_session(store, repo, with_checkpoint=True)
    row_before = _row(store, source.session_id)
    tree_before = _tree_hashes(source.artifact_path)

    with (
        patch.object(
            store,
            "insert_forked_session",
            side_effect=failure,
        ),
        pytest.raises(ForkSessionError, match="before child publication"),
    ):
        fork_saved_session(store, source)

    assert _row(store, source.session_id) == row_before
    assert _tree_hashes(source.artifact_path) == tree_before
    _assert_no_partial_child(store, source.session_id)


def test_interruption_after_child_row_commit_removes_resolvable_child(
    tmp_path: Path,
):
    repo = _repo(tmp_path)
    store = SessionStore(tmp_path / "assist")
    source = _source_session(store, repo, with_checkpoint=True)
    row_before = _row(store, source.session_id)
    tree_before = _tree_hashes(source.artifact_path)
    insert = store.insert_forked_session

    def commit_then_interrupt(record, *, expected_parent):
        insert(record, expected_parent=expected_parent)
        raise KeyboardInterrupt()

    with (
        patch.object(
            store,
            "insert_forked_session",
            side_effect=commit_then_interrupt,
        ),
        pytest.raises(ForkSessionError, match="before child publication"),
    ):
        fork_saved_session(store, source)

    assert _row(store, source.session_id) == row_before
    assert _tree_hashes(source.artifact_path) == tree_before
    _assert_no_partial_child(store, source.session_id)


def test_new_child_inputs_events_and_attachments_never_enter_parent(
    tmp_path: Path,
):
    repo = _repo(tmp_path)
    store = SessionStore(tmp_path / "assist")
    source = _source_session(store, repo, with_attachment=True)
    child = fork_saved_session(store, source).child
    source_before = _tree_hashes(source.artifact_path)
    source_row = _row(store, source.session_id)

    save_approval_request(child.artifact_path, {
        "status": "pending",
        "action_key": "bash:sha256:child",
        "tool_name": "bash",
        "args_summary": "rm child-build",
        "reason": "destructive",
        "requested_at": 2.0,
        "cmd": "rm child-build",
    })
    request = create_clarification_request(
        child.artifact_path,
        request_id="child-question",
        session_id=child.session_id,
        session_number=2,
        turn_number=1,
        tool_call_id="child-call",
        question="Which child target?",
    )
    answer = record_clarification_answer(
        child.artifact_path,
        session_id=child.session_id,
        request_id=request["request_id"],
        answer="Only the child target.",
    )
    correction = create_correction(
        child.artifact_path,
        correction_id="child-correction",
        session_id=child.session_id,
        after_session_number=2,
        text="Change only the child.",
    )
    with (child.artifact_path / ".trace.jsonl").open("a") as trace:
        trace.write(json.dumps({
            "event": "tool_call",
            "session_number": 2,
            "turn_number": 1,
            "tool_name": "write",
            "args_summary": "child.txt",
            "result_summary": "ok",
        }) + "\n")
        trace.write(json.dumps({
            "event": "correction_created",
            "session_number": 2,
            "correction_id": correction["correction_id"],
            "text_sha256": correction["text_sha256"],
            "text_chars": len(correction["text"]),
        }) + "\n")
    image = PendingImage(
        display_name="child.png",
        media_type="image/png",
        data=PNG_1X1,
        size_bytes=len(PNG_1X1),
        sha256=hashlib.sha256(PNG_1X1).hexdigest(),
        width=1,
        height=1,
    )
    save_image_segment(
        child.artifact_path,
        segment_number=2,
        prompt_text="Child-only image.",
        images=[image],
    )

    assert request["session_id"] == child.session_id
    assert answer["session_id"] == child.session_id
    assert correction["session_id"] == child.session_id
    assert _tree_hashes(source.artifact_path) == source_before
    assert _row(store, source.session_id) == source_row
    assert not (source.artifact_path / "approval_request.json").exists()
    assert not (source.artifact_path / "clarification_request.json").exists()
    assert not (source.artifact_path / "correction.json").exists()
    assert not (
        source.artifact_path
        / "attachments"
        / "segment-0002"
        / "image-0001.png"
    ).exists()


def test_parent_and_child_mutation_archive_and_removal_are_independent(tmp_path: Path):
    repo = _repo(tmp_path)
    store = SessionStore(tmp_path / "assist")
    source = _source_session(store, repo, with_attachment=True)
    (source.artifact_path / "mutable-evidence.bin").write_bytes(b"shared\0")
    child = fork_saved_session(store, source).child
    source_evidence = source.artifact_path / "mutable-evidence.bin"
    child_evidence = child.artifact_path / "mutable-evidence.bin"
    source_bytes = source_evidence.read_bytes()
    child_bytes = child_evidence.read_bytes()

    child_evidence.write_bytes(b"child-only change\0")
    assert source_evidence.read_bytes() == source_bytes
    source_evidence.write_bytes(b"parent-only change\0")
    assert child_evidence.read_bytes() == b"child-only change\0"

    store.set_session_label(source.session_id, "parent-run")
    store.set_session_label(child.session_id, "child-run")
    archived_parent, changed = store.archive_session(source.session_id)
    assert changed is True
    unchanged_child = store.get_session(child.session_id)
    assert unchanged_child is not None
    assert unchanged_child.parent_session_id == source.session_id
    assert unchanged_child.archived_at is None
    assert unchanged_child.label == "child-run"
    store.unarchive_session(source.session_id)
    archived_child, changed = store.archive_session(child.session_id)
    assert changed is True
    unchanged_parent = store.get_session(source.session_id)
    assert unchanged_parent is not None
    assert unchanged_parent.archived_at is None
    assert unchanged_parent.label == "parent-run"
    assert archived_child.parent_session_id == source.session_id
    assert source_evidence.read_bytes() == b"parent-only change\0"
    assert child_evidence.read_bytes() == b"child-only change\0"
    assert source_bytes == child_bytes

    store.unarchive_session(child.session_id)
    survivor = fork_saved_session(store, source).child
    source_before_child_purge = _tree_hashes(source.artifact_path)
    source_row_before_child_purge = _row(store, source.session_id)
    for path in sorted(child.artifact_path.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    child.artifact_path.rmdir()
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "delete from sessions where session_id = ?",
            (child.session_id,),
        )
    assert _tree_hashes(source.artifact_path) == source_before_child_purge
    assert _row(store, source.session_id) == source_row_before_child_purge

    survivor_before_parent_purge = _tree_hashes(survivor.artifact_path)
    survivor_row_before_parent_purge = _row(store, survivor.session_id)
    for path in sorted(source.artifact_path.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    source.artifact_path.rmdir()
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "delete from sessions where session_id = ?",
            (source.session_id,),
        )
    assert _tree_hashes(survivor.artifact_path) == survivor_before_parent_purge
    assert _row(store, survivor.session_id) == survivor_row_before_parent_purge
    assert store.get_session(source.session_id) is None
    assert store.get_session(survivor.session_id).parent_session_id == source.session_id


def test_managed_worktree_fork_copies_endpoint_to_distinct_owned_identity(
    tmp_path: Path,
):
    repo = _repo(tmp_path)
    nested = repo / "package"
    nested.mkdir()
    (nested / "module.py").write_text("value = 'base'\n")
    _git(repo, "add", "package/module.py")
    _git(repo, "commit", "-qm", "add nested workspace")
    store = SessionStore(tmp_path / "assist")
    source = _source_session(store, nested, worktree_mode="auto")
    _info, source = _resolve_session_worktree(
        store,
        source,
        cfg=SimpleNamespace(runtime_worktree="auto"),
        resume=False,
    )
    assert source.worktree_path is not None
    source_identity = inspect_session_worktree(nested, source.session_id)
    source_worktree = source_identity.session_cwd
    (source_worktree / "module.py").write_text("value = 'saved endpoint'\n")
    (source_worktree / "untracked.bin").write_bytes(b"source endpoint\x00")
    checkpoint = WorkspaceCheckpointStore(
        source_worktree,
        shadow_dir=source.artifact_path / ".shadow_git",
    ).capture(1)
    source_status = _git(source_worktree, "status", "--porcelain=v1")
    source_artifacts = _tree_hashes(source.artifact_path)
    source_row = _row(store, source.session_id)

    child = fork_saved_session(store, source).child

    assert child.worktree_path is not None
    child_identity = inspect_session_worktree(nested, child.session_id)
    child_worktree = child_identity.session_cwd
    assert Path(child.worktree_path) != Path(source.worktree_path)
    assert child_worktree != source_worktree
    assert child.worktree_branch != source.worktree_branch
    assert child.worktree_base_commit == _git(source_worktree, "rev-parse", "HEAD")
    assert (child_worktree / "module.py").read_text() == "value = 'saved endpoint'\n"
    assert (child_worktree / "untracked.bin").read_bytes() == b"source endpoint\x00"
    assert _git(child_worktree, "status", "--porcelain=v1") == source_status
    assert inspect_session_worktree(nested, source.session_id) == source_identity
    assert _tree_hashes(source.artifact_path) == source_artifacts
    assert _row(store, source.session_id) == source_row

    child_checkpoint = WorkspaceCheckpointStore(
        child_worktree,
        shadow_dir=child.artifact_path / ".shadow_git",
    )
    assert child_checkpoint.checkpoint_for_turn(1) == checkpoint.commit
    (child_worktree / "module.py").write_text("value = 'child change'\n")
    assert (source_worktree / "module.py").read_text() == "value = 'saved endpoint'\n"


def test_managed_worktree_validation_failure_precedes_child_creation(tmp_path: Path):
    repo = _repo(tmp_path)
    store = SessionStore(tmp_path / "assist")
    source = _source_session(store, repo, worktree_mode="auto")
    _info, source = _resolve_session_worktree(
        store,
        source,
        cfg=SimpleNamespace(runtime_worktree="auto"),
        resume=False,
    )
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "update sessions set worktree_branch = ? where session_id = ?",
            ("wrong-branch", source.session_id),
        )
    malformed = store.get_session(source.session_id)
    assert malformed is not None
    artifacts_before = _tree_hashes(malformed.artifact_path)
    row_before = _row(store, malformed.session_id)
    registered_before = _git(repo, "worktree", "list", "--porcelain")

    with pytest.raises(ForkSessionError, match="worktree identity"):
        fork_saved_session(store, malformed)

    assert _tree_hashes(malformed.artifact_path) == artifacts_before
    assert _row(store, malformed.session_id) == row_before
    assert _git(repo, "worktree", "list", "--porcelain") == registered_before
    _assert_no_partial_child(store, malformed.session_id)


def test_managed_worktree_is_cleaned_if_child_row_commit_fails(tmp_path: Path):
    repo = _repo(tmp_path)
    store = SessionStore(tmp_path / "assist")
    source = _source_session(store, repo, worktree_mode="auto")
    _info, source = _resolve_session_worktree(
        store,
        source,
        cfg=SimpleNamespace(runtime_worktree="auto"),
        resume=False,
    )
    worktrees_before = _git(repo, "worktree", "list", "--porcelain")
    branches_before = _git(repo, "for-each-ref", "--format=%(refname)", "refs/heads")
    source_artifacts = _tree_hashes(source.artifact_path)
    source_row = _row(store, source.session_id)

    with (
        patch.object(
            store,
            "insert_forked_session",
            side_effect=RuntimeError("database stopped"),
        ),
        pytest.raises(ForkSessionError, match="database stopped"),
    ):
        fork_saved_session(store, source)

    assert _git(repo, "worktree", "list", "--porcelain") == worktrees_before
    assert _git(repo, "for-each-ref", "--format=%(refname)", "refs/heads") == branches_before
    assert _tree_hashes(source.artifact_path) == source_artifacts
    assert _row(store, source.session_id) == source_row
    _assert_no_partial_child(store, source.session_id)


def test_managed_worktree_is_cleaned_if_copy_is_interrupted(tmp_path: Path):
    repo = _repo(tmp_path)
    store = SessionStore(tmp_path / "assist")
    source = _source_session(store, repo, worktree_mode="auto")
    _info, source = _resolve_session_worktree(
        store,
        source,
        cfg=SimpleNamespace(runtime_worktree="auto"),
        resume=False,
    )
    worktrees_before = _git(repo, "worktree", "list", "--porcelain")
    branches_before = _git(repo, "for-each-ref", "--format=%(refname)", "refs/heads")
    source_artifacts = _tree_hashes(source.artifact_path)
    source_row = _row(store, source.session_id)

    with (
        patch(
            "scripts.llm_solver.harness.worktree_runtime._replace_worktree_contents",
            side_effect=KeyboardInterrupt(),
        ),
        pytest.raises(ForkSessionError, match="before child publication"),
    ):
        fork_saved_session(store, source)

    assert _git(repo, "worktree", "list", "--porcelain") == worktrees_before
    assert _git(repo, "for-each-ref", "--format=%(refname)", "refs/heads") == branches_before
    assert _tree_hashes(source.artifact_path) == source_artifacts
    assert _row(store, source.session_id) == source_row
    _assert_no_partial_child(store, source.session_id)


def test_foreign_live_lock_is_refused_even_when_source_row_says_paused(tmp_path: Path):
    repo = _repo(tmp_path)
    store = SessionStore(tmp_path / "assist")
    source = _source_session(store, repo)
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            insert into session_locks (
                session_id, owner_host, owner_pid, acquired_at
            ) values (?, ?, ?, ?)
            """,
            (source.session_id, socket.gethostname(), 1, "2026-08-25T00:00:00Z"),
        )
    before = _tree_hashes(source.artifact_path)
    row_before = _row(store, source.session_id)
    with pytest.raises(ForkSessionError, match="locked"):
        fork_saved_session(store, source)
    assert _tree_hashes(source.artifact_path) == before
    assert _row(store, source.session_id) == row_before
    _assert_no_partial_child(store, source.session_id)
