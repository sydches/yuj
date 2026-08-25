"""Saved-session archive lifecycle acceptance tests."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.llm_assist.__main__ import main
from scripts.llm_assist.runner import save_approval_request
from scripts.llm_assist.store import SessionStore
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness.clarifications import (
    create_clarification_request,
    record_clarification_answer,
)
from scripts.llm_solver.harness.corrections import create_correction
from scripts.llm_solver.server.replay_client import parse_transcript_turns


def _record(
    store: SessionStore,
    cwd: Path,
    *,
    status: str = "paused",
    label: str | None = None,
):
    record = store.create_session(
        cwd=cwd,
        model="test-model",
        prompt_text="Complete the task.",
        prompt_source="test",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    if status != "created":
        store.update_session(
            record.session_id,
            status=status,
            last_finish_reason="test-boundary",
        )
    if label is not None:
        store.set_session_label(record.session_id, label)
    refreshed = store.get_session(record.session_id)
    assert refreshed is not None
    return refreshed


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _session_row(store: SessionStore, session_id: str) -> dict[str, object]:
    with sqlite3.connect(store.db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "select * from sessions where session_id = ?",
            (session_id,),
        ).fetchone()
    assert row is not None
    return dict(row)


def _without_archive(record) -> dict[str, object]:
    values = asdict(record)
    values.pop("archived_at")
    return values


def _create_pre_archive_database(root: Path, cwd: Path) -> str:
    root.mkdir()
    session_id = "20260824_120000_deadbeef"
    with sqlite3.connect(root / "sessions.sqlite3") as connection:
        connection.execute(
            """
            create table sessions (
                session_id text primary key,
                created_at text not null,
                updated_at text not null,
                cwd text not null,
                artifact_dir text not null,
                model text not null,
                status text not null,
                last_finish_reason text,
                prompt_text text not null,
                prompt_source text not null,
                context_mode text not null,
                system_prompt_path text,
                config_paths_json text not null,
                worktree_path text,
                worktree_branch text,
                worktree_base_commit text,
                provider text,
                auth_method text,
                credential_id text,
                label text,
                parent_session_id text
            )
            """
        )
        connection.execute(
            """
            create unique index sessions_label_unique
            on sessions(label) where label is not null
            """
        )
        connection.execute(
            """
            insert into sessions (
                session_id, created_at, updated_at, cwd, artifact_dir,
                model, status, last_finish_reason, prompt_text,
                prompt_source, context_mode, system_prompt_path,
                config_paths_json, worktree_path, worktree_branch,
                worktree_base_commit, provider, auth_method, credential_id,
                label, parent_session_id
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                "2026-08-24T12:00:00+00:00",
                "2026-08-24T12:01:00+00:00",
                str(cwd.resolve()),
                str(root / "sessions" / session_id),
                "legacy-model",
                "paused",
                "max_turns",
                "Legacy task.",
                "inline",
                "full",
                None,
                "[]",
                str((cwd / ".yuj-worktrees" / session_id).resolve()),
                "legacy-worktree",
                "b" * 40,
                "openai-compatible",
                "api-key-env",
                None,
                "legacy-run",
                "20260823_120000_feedface",
            ),
        )
    return session_id


def test_new_store_has_idempotent_nullable_archive_schema(tmp_path: Path):
    root = tmp_path / "assist"

    first = SessionStore(root)
    second = SessionStore(root)
    record = _record(second, tmp_path / "work")

    with sqlite3.connect(first.db_path) as connection:
        columns = {
            row[1]: row[3]
            for row in connection.execute("pragma table_info(sessions)")
        }
        assert connection.execute("pragma integrity_check").fetchone()[0] == "ok"

    assert columns["archived_at"] == 0
    assert record.archived_at is None


def test_existing_store_migrates_once_and_preserves_label_and_lineage(
    tmp_path: Path,
):
    root = tmp_path / "assist"
    session_id = _create_pre_archive_database(root, tmp_path / "work")

    first = SessionStore(root)
    migrated = first.get_session(session_id)
    second = SessionStore(root)
    migrated_again = second.get_session(session_id)

    assert migrated is not None
    assert migrated.archived_at is None
    assert migrated.label == "legacy-run"
    assert migrated_again == migrated
    before = _session_row(second, session_id)

    archived, changed = second.archive_session(session_id)

    after = _session_row(second, session_id)
    assert changed is True
    assert archived.archived_at is not None
    assert after["parent_session_id"] == "20260823_120000_feedface"
    assert after["label"] == "legacy-run"
    assert {
        key: value for key, value in after.items() if key != "archived_at"
    } == {
        key: value for key, value in before.items() if key != "archived_at"
    }


def test_archive_column_migration_rolls_back_with_a_later_schema_failure(
    tmp_path: Path,
):
    root = tmp_path / "assist"
    session_id = _create_pre_archive_database(root, tmp_path / "work")
    with sqlite3.connect(root / "sessions.sqlite3") as connection:
        connection.execute("drop index sessions_label_unique")
        connection.execute(
            """
            insert into sessions (
                session_id, created_at, updated_at, cwd, artifact_dir,
                model, status, prompt_text, prompt_source, context_mode,
                config_paths_json, label
            )
            select ?, created_at, updated_at, cwd, ?, model, status,
                   prompt_text, prompt_source, context_mode,
                   config_paths_json, label
            from sessions where session_id = ?
            """,
            (
                "20260824_120001_cafebabe",
                str(root / "sessions" / "20260824_120001_cafebabe"),
                session_id,
            ),
        )

    with pytest.raises(sqlite3.IntegrityError):
        SessionStore(root)

    with sqlite3.connect(root / "sessions.sqlite3") as connection:
        columns = {
            row[1] for row in connection.execute("pragma table_info(sessions)")
        }
    assert "archived_at" not in columns


def test_read_only_pre_archive_store_treats_legacy_rows_as_unarchived(
    tmp_path: Path,
):
    root = tmp_path / "assist"
    session_id = _create_pre_archive_database(root, tmp_path / "work")

    store = SessionStore(root, read_only=True)

    assert [record.session_id for record in store.list_sessions()] == [session_id]
    assert store.list_sessions(archived=True) == []
    assert store.resolve_session_ref("legacy-run").session_id == session_id


def test_archive_and_unarchive_change_only_archive_metadata_and_no_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    store = SessionStore(tmp_path / "assist")
    record = _record(
        store,
        tmp_path / "work",
        status="paused",
        label="evidence-run",
    )
    worktree = tmp_path / "owned-worktree"
    worktree.mkdir()
    (worktree / "state.bin").write_bytes(b"owned\x00worktree\xff")
    store.update_session_worktree(
        record.session_id,
        path=worktree,
        branch="yuj-session-evidence",
        base_commit="c" * 40,
    )
    record = store.get_session(record.session_id)
    assert record is not None
    record.artifact_path.mkdir(parents=True)
    evidence = {
        "prompt.txt": b"Complete the task.\n",
        ".trace.jsonl": (
            b'{"event":"session_start","session_number":1}\n'
            b'{"event":"session_end","session_number":1,'
            b'"finish_reason":"max_turns","turns":2}\n'
        ),
        "transcript.log": (
            b"=== turn 1 input ===\n"
            b'[{"role":"user","content":"Complete the task."}]\n'
            b"=== turn 1 output ===\n"
            b'{"content":"Continue.","tool_calls":[]}\n'
        ),
        ".solver/state.json": b'{"state":{"phase":"stopped"}}\n',
    }
    for name, contents in evidence.items():
        path = record.artifact_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)

    before_record = store.get_session(record.session_id)
    before_artifacts = _artifact_bytes(record.artifact_path)
    before_worktree = _artifact_bytes(worktree)
    replay_before = parse_transcript_turns(
        record.artifact_path / "transcript.log"
    )
    measurement_before = load_config(overrides={"runtime_mode": "measurement"})

    with (
        patch("scripts.llm_assist.__main__.SessionStore", return_value=store),
        patch("scripts.llm_assist.__main__._make_client") as make_client,
        patch("scripts.llm_assist.__main__.run_session") as run_session,
        patch("scripts.llm_assist.__main__.rewind_session") as rewind_session,
        patch(
            "scripts.llm_assist.__main__.remove_session_worktree"
        ) as remove_worktree,
    ):
        assert main(["archive", "evidence-run"]) == 0
        archived_output = capsys.readouterr().out
        archived = store.get_session(record.session_id)
        assert archived is not None
        assert archived.archived_at is not None
        assert _without_archive(archived) == _without_archive(before_record)
        assert "archive: archived" in archived_output
        assert "changed: yes" in archived_output

        assert main(["unarchive", record.short_id]) == 0
        unarchived_output = capsys.readouterr().out

    unarchived = store.get_session(record.session_id)
    assert unarchived == before_record
    assert "archive: unarchived" in unarchived_output
    assert "changed: yes" in unarchived_output
    assert _artifact_bytes(record.artifact_path) == before_artifacts
    assert _artifact_bytes(worktree) == before_worktree
    assert (
        parse_transcript_turns(record.artifact_path / "transcript.log")
        == replay_before
    )
    assert (
        load_config(overrides={"runtime_mode": "measurement"})
        == measurement_before
    )
    make_client.assert_not_called()
    run_session.assert_not_called()
    rewind_session.assert_not_called()
    remove_worktree.assert_not_called()


def test_archive_refuses_active_pointer_running_and_lock_without_mutation(
    tmp_path: Path,
):
    store = SessionStore(tmp_path / "assist")
    active = _record(store, tmp_path / "active")
    running = _record(store, tmp_path / "running", status="running")
    locked = _record(store, tmp_path / "locked")
    store.set_active_session(active.cwd, active.session_id)
    store.acquire_session_lock(locked.session_id)
    before = {
        record.session_id: _session_row(store, record.session_id)
        for record in (active, running, locked)
    }

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        with pytest.raises(SystemExit, match="active session"):
            main(["archive", active.session_id])
        with pytest.raises(SystemExit, match="running session"):
            main(["archive", running.session_id])
        with pytest.raises(SystemExit, match="locked session"):
            main(["archive", locked.session_id])

    for record in (active, running, locked):
        assert _session_row(store, record.session_id) == before[record.session_id]


@pytest.mark.parametrize(
    "pending_kind",
    ["approval", "clarification", "answer", "correction"],
)
def test_archive_refuses_pending_input_without_mutation(
    tmp_path: Path,
    pending_kind: str,
):
    store = SessionStore(tmp_path / "assist")
    record = _record(store, tmp_path / pending_kind)
    record.artifact_path.mkdir(parents=True)
    if pending_kind == "approval":
        save_approval_request(
            record.artifact_path,
            {
                "status": "pending",
                "tool_name": "bash",
                "cmd": "make test",
                "reason": "operator decision",
            },
        )
        message = "pending approval"
    elif pending_kind in {"clarification", "answer"}:
        request = create_clarification_request(
            record.artifact_path,
            request_id=f"request-{pending_kind}",
            session_id=record.session_id,
            session_number=1,
            turn_number=2,
            tool_call_id=f"ask-{pending_kind}",
            question="Which target?",
        )
        if pending_kind == "answer":
            record_clarification_answer(
                record.artifact_path,
                session_id=record.session_id,
                request_id=request["request_id"],
                answer="Use PostgreSQL.",
            )
        message = "pending clarification"
    else:
        correction = create_correction(
            record.artifact_path,
            correction_id="corr-archive-test",
            session_id=record.session_id,
            after_session_number=1,
            text="Use the exact target.",
        )
        (record.artifact_path / ".trace.jsonl").write_text(
            json.dumps(
                {
                    "event": "correction_created",
                    "session_number": 1,
                    "correction_id": correction["correction_id"],
                    "text_sha256": correction["text_sha256"],
                    "text_chars": len(correction["text"]),
                }
            )
            + "\n"
        )
        message = "pending correction"
    before_row = _session_row(store, record.session_id)
    before_artifacts = _artifact_bytes(record.artifact_path)

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        with pytest.raises(SystemExit, match=message):
            main(["archive", record.session_id])

    assert _session_row(store, record.session_id) == before_row
    assert _artifact_bytes(record.artifact_path) == before_artifacts


def test_repeated_archive_and_unarchive_report_stable_current_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    store = SessionStore(tmp_path / "assist")
    record = _record(store, tmp_path / "work")

    with (
        patch("scripts.llm_assist.__main__.SessionStore", return_value=store),
        patch(
            "scripts.llm_assist.store._utc_now",
            return_value="2026-08-25T22:00:00+00:00",
        ),
    ):
        with pytest.raises(SystemExit, match="explicit session reference"):
            main(["archive", "latest"])
        with pytest.raises(SystemExit, match="explicit session reference"):
            main(["unarchive", "last"])
        assert main(["archive", record.session_id]) == 0
        first_output = capsys.readouterr().out
        first = store.get_session(record.session_id)
        assert first is not None
        assert main(["archive", record.session_id]) == 0
        second_output = capsys.readouterr().out
        second = store.get_session(record.session_id)

        assert second == first
        assert "changed: yes" in first_output
        assert "changed: no" in second_output
        assert first_output.replace("changed: yes", "changed: no") == second_output

        assert main(["unarchive", record.session_id]) == 0
        first_unarchive = capsys.readouterr().out
        assert main(["unarchive", record.session_id]) == 0
        second_unarchive = capsys.readouterr().out

    assert store.get_session(record.session_id).archived_at is None
    assert "changed: yes" in first_unarchive
    assert "changed: no" in second_unarchive
    assert first_unarchive.replace("changed: yes", "changed: no") == second_unarchive


def test_default_lists_and_latest_current_resume_exclude_archived_sessions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    store = SessionStore(tmp_path / "assist")
    cwd = tmp_path / "work"
    cwd.mkdir()
    ordinary = _record(store, cwd, label="ordinary-run")
    archived = _record(store, cwd, label="archived-run")
    store.archive_session(archived.session_id)
    # A stale active pointer must not make an archived record implicit.
    store.set_active_session(cwd, archived.session_id)
    monkeypatch.chdir(cwd)

    selected: list[str] = []

    def fake_run_session(store_obj, record, *, resume, resume_prompt_text=None):
        selected.append(record.session_id)
        store_obj.update_session(
            record.session_id,
            status="completed",
            last_finish_reason="stop",
        )
        return True, "stop"

    with (
        patch("scripts.llm_assist.__main__.SessionStore", return_value=store),
        patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session),
    ):
        assert main(["sessions"]) == 0
        listed = capsys.readouterr().out
        assert ordinary.session_id in listed
        assert archived.session_id not in listed

        assert main(["status"]) == 0
        assert f"session_id: {ordinary.session_id}" in capsys.readouterr().out
        assert main(["show", "--turns", "0", "--trace-lines", "0"]) == 0
        assert f"session_id: {ordinary.session_id}" in capsys.readouterr().out
        assert main(["current"]) == 0
        assert f"session_id: {ordinary.session_id}" in capsys.readouterr().out
        assert main(["resume"]) == 0

    assert selected == [ordinary.session_id]


def test_pending_approval_selector_excludes_archived_sessions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    store = SessionStore(tmp_path / "assist")
    cwd = tmp_path / "work"
    cwd.mkdir()
    ordinary = _record(store, cwd)
    archived = _record(store, cwd)
    store.archive_session(archived.session_id)
    for record in (ordinary, archived):
        save_approval_request(
            record.artifact_path,
            {
                "status": "pending",
                "tool_name": "bash",
                "cmd": "make test",
                "reason": "operator decision",
            },
        )
    store.set_active_session(cwd, archived.session_id)
    monkeypatch.chdir(cwd)

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert main(["approve"]) == 0

    assert f"approved: {ordinary.session_id}" in capsys.readouterr().out
    archived_request = json.loads(
        (archived.artifact_path / "approval_request.json").read_text()
    )
    assert archived_request["status"] == "pending"


def test_explicit_archived_listing_status_show_and_usage_are_read_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    store = SessionStore(tmp_path / "assist")
    ordinary = _record(store, tmp_path / "ordinary")
    archived = _record(store, tmp_path / "archived", label="cold-run")
    archived.artifact_path.mkdir(parents=True)
    (archived.artifact_path / ".trace.jsonl").write_text(
        json.dumps(
            {
                "event": "session_usage",
                "session_number": 1,
                "scope": "all_model_responses",
                "input_tokens": 10,
                "output_tokens": 2,
                "cached_tokens": 4,
                "cost": None,
                "quota": None,
            }
        )
        + "\n"
    )
    archived, _ = store.archive_session(archived.session_id)
    before = _artifact_bytes(archived.artifact_path)

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert main(["sessions", "--archived"]) == 0
        archived_list = capsys.readouterr().out
        assert archived.session_id in archived_list
        assert ordinary.session_id not in archived_list
        assert "archived" in archived_list
        assert f"archived_at={archived.archived_at}" in archived_list

        assert main(["status", "cold-run"]) == 0
        status_output = capsys.readouterr().out
        assert "archived: yes" in status_output
        assert f"archived_at: {archived.archived_at}" in status_output
        assert f"next: yuj unarchive {archived.short_id}" in status_output

        assert main(
            ["show", archived.short_id, "--turns", "0", "--trace-lines", "0"]
        ) == 0
        show_output = capsys.readouterr().out
        assert "archived: yes" in show_output
        assert f"archived_at: {archived.archived_at}" in show_output

        assert main(["usage", archived.session_id]) == 0
        usage_output = capsys.readouterr().out
        assert f"session_id: {archived.session_id}" in usage_output
        assert "input_tokens: 10" in usage_output

    assert _artifact_bytes(archived.artifact_path) == before


@pytest.mark.parametrize(
    "command",
    [
        ["resume"],
        ["label", "replacement-label"],
        ["correct", "Use the exact target."],
        ["answer", "request-1", "Use PostgreSQL."],
        ["rewind", "1"],
        ["approve"],
        ["reject"],
        ["worktree", "rm"],
    ],
)
def test_mutating_commands_refuse_archived_session_with_unarchive_instruction(
    tmp_path: Path,
    command: list[str],
):
    store = SessionStore(tmp_path / "assist")
    record = _record(store, tmp_path / "work", label="frozen-run")
    archived, _ = store.archive_session(record.session_id)
    before_row = _session_row(store, record.session_id)
    before_artifacts = _artifact_bytes(record.artifact_path)
    argv = [command[0]]
    if command[0] == "worktree":
        argv.extend(["rm", "frozen-run"])
    else:
        argv.append("frozen-run")
        argv.extend(command[1:])

    message = (
        "session is archived; run "
        f"yuj unarchive {archived.short_id} first"
    )
    with (
        patch("scripts.llm_assist.__main__.SessionStore", return_value=store),
        patch("scripts.llm_assist.__main__.run_session") as run_session,
        patch("scripts.llm_assist.__main__.rewind_session") as rewind_session,
        patch(
            "scripts.llm_assist.__main__.remove_session_worktree"
        ) as remove_worktree,
    ):
        with pytest.raises(SystemExit, match=message):
            main(argv)

    assert _session_row(store, record.session_id) == before_row
    assert _artifact_bytes(record.artifact_path) == before_artifacts
    run_session.assert_not_called()
    rewind_session.assert_not_called()
    remove_worktree.assert_not_called()


def test_unarchive_restores_original_selection_order_label_and_worktree(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    store = SessionStore(tmp_path / "assist")
    cwd = tmp_path / "work"
    cwd.mkdir()
    older = _record(store, cwd, label="older-run")
    newer = _record(store, cwd, label="newer-run")
    store.update_session_worktree(
        newer.session_id,
        path=tmp_path / "owned-worktree",
        branch="yuj-session-newer",
        base_commit="d" * 40,
    )
    newer = store.get_session(newer.session_id)
    assert newer is not None
    identity = (
        newer.session_id,
        newer.label,
        newer.worktree_path,
        newer.worktree_branch,
        newer.worktree_base_commit,
    )
    store.archive_session(newer.session_id)
    monkeypatch.chdir(cwd)

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert main(["status"]) == 0
        assert f"session_id: {older.session_id}" in capsys.readouterr().out
        assert main(["unarchive", "newer-run"]) == 0
        capsys.readouterr()
        assert main(["status"]) == 0
        assert f"session_id: {newer.session_id}" in capsys.readouterr().out

    restored = store.get_session(newer.session_id)
    assert restored is not None
    assert restored.archived_at is None
    assert (
        restored.session_id,
        restored.label,
        restored.worktree_path,
        restored.worktree_branch,
        restored.worktree_base_commit,
    ) == identity


def test_archived_label_remains_unique_and_resolves_for_inspection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    store = SessionStore(tmp_path / "assist")
    archived = _record(store, tmp_path / "archived", label="reserved-run")
    ordinary = _record(store, tmp_path / "ordinary")
    store.archive_session(archived.session_id)

    with pytest.raises(ValueError, match="already assigned"):
        store.set_session_label(ordinary.session_id, "reserved-run")

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert main(["status", "reserved-run"]) == 0

    assert f"session_id: {archived.session_id}" in capsys.readouterr().out
