"""Manual saved-session label acceptance tests."""
from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.llm_assist.__main__ import main
from scripts.llm_assist.runner import (
    _write_session_metadata,
    save_approval_request,
)
from scripts.llm_assist.store import AmbiguousSessionRefError, SessionStore
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness.clarifications import (
    create_clarification_request,
    load_clarification_answer,
)


def _record(store: SessionStore, cwd: Path, *, status: str = "created"):
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
        record = store.get_session(record.session_id)
        assert record is not None
    return record


def _insert_session(store: SessionStore, session_id: str, cwd: Path) -> None:
    now = "2026-08-25T00:00:00+00:00"
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            insert into sessions (
                session_id, created_at, updated_at, cwd, artifact_dir,
                model, status, last_finish_reason, prompt_text,
                prompt_source, context_mode, system_prompt_path,
                config_paths_json
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                now,
                now,
                str(cwd.resolve()),
                str(store.root / "sessions" / session_id),
                "test-model",
                "created",
                None,
                "Complete the task.",
                "test",
                "full",
                None,
                "[]",
            ),
        )


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _without_label(record) -> dict:
    values = asdict(record)
    values.pop("label", None)
    return values


def _create_pre_label_database(root: Path, cwd: Path) -> str:
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
                credential_id text
            )
            """
        )
        connection.execute(
            """
            insert into sessions values (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?
            )
            """,
            (
                session_id,
                "2026-08-24T12:00:00+00:00",
                "2026-08-24T12:00:00+00:00",
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
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        )
    return session_id


def test_new_store_has_idempotent_nullable_unique_label_schema(tmp_path: Path):
    root = tmp_path / "assist"

    SessionStore(root)
    SessionStore(root)

    with sqlite3.connect(root / "sessions.sqlite3") as connection:
        columns = {
            row[1] for row in connection.execute("pragma table_info(sessions)")
        }
        indexes = {
            row[1]: row[2]
            for row in connection.execute("pragma index_list(sessions)")
        }
        label_index_columns = [
            row[2]
            for row in connection.execute(
                "pragma index_info(sessions_label_unique)"
            )
        ]

    assert "label" in columns
    assert indexes["sessions_label_unique"] == 1
    assert label_index_columns == ["label"]


def test_existing_pre_label_store_migrates_once_and_preserves_rows(
    tmp_path: Path,
):
    root = tmp_path / "assist"
    session_id = _create_pre_label_database(root, tmp_path / "work")

    first = SessionStore(root)
    migrated = first.get_session(session_id)
    second = SessionStore(root)
    migrated_again = second.get_session(session_id)

    assert migrated is not None
    assert migrated.label is None
    assert migrated_again == migrated
    second.set_session_label(session_id, "legacy-run")
    assert SessionStore(root).get_session(session_id).label == "legacy-run"


def test_set_replace_and_clear_preserve_every_other_index_field(
    tmp_path: Path,
):
    store = SessionStore(tmp_path / "assist")
    record = _record(store, tmp_path / "work", status="archived")
    store.update_session_worktree(
        record.session_id,
        path=tmp_path / "owned-worktree",
        branch="yuj-session-test",
        base_commit="b" * 40,
    )
    store.set_active_session(record.cwd, record.session_id)
    before = store.get_session(record.session_id)
    assert before is not None

    store.set_session_label(record.session_id, "First.Label")
    first = store.get_session(record.session_id)
    assert first is not None
    assert first.label == "First.Label"
    assert _without_label(first) == _without_label(before)

    store.set_session_label(record.session_id, "replacement-label")
    replacement = store.get_session(record.session_id)
    assert replacement is not None
    assert replacement.label == "replacement-label"
    assert _without_label(replacement) == _without_label(before)

    store.clear_session_label(record.session_id)
    cleared = store.get_session(record.session_id)
    assert cleared == before
    assert store.get_active_session_id(record.cwd) == record.session_id


@pytest.mark.parametrize(
    ("label", "message"),
    [
        ("", "session label must not be empty"),
        (" leading", "must start with an ASCII letter"),
        ("two words", "must start with an ASCII letter"),
        ("9lives", "must start with an ASCII letter"),
        ("éclair", "must start with an ASCII letter"),
        ("a" * 65, "must be at most 64 characters"),
        ("latest", "session label 'latest' is reserved"),
        ("LAST", "session label 'LAST' is reserved"),
        ("deadbeef", "must not look like a session ID"),
        (
            "20260825_120000_deadbeef",
            "must not look like a session ID",
        ),
    ],
)
def test_invalid_reserved_and_id_like_labels_fail_without_mutation(
    tmp_path: Path,
    label: str,
    message: str,
):
    store = SessionStore(tmp_path / "assist")
    record = _record(store, tmp_path / "work")
    before = store.get_session(record.session_id)

    with pytest.raises(ValueError, match=message):
        store.set_session_label(record.session_id, label)

    assert store.get_session(record.session_id) == before


def test_label_boundary_and_case_are_preserved_without_normalization(
    tmp_path: Path,
):
    store = SessionStore(tmp_path / "assist")
    upper = _record(store, tmp_path / "upper")
    lower = _record(store, tmp_path / "lower")
    boundary = "A" + "b" * 63

    store.set_session_label(upper.session_id, boundary)
    store.set_session_label(lower.session_id, boundary.lower())

    assert store.resolve_session_ref(boundary).session_id == upper.session_id
    assert store.resolve_session_ref(boundary.lower()).session_id == lower.session_id
    assert store.resolve_session_ref("a" + "b" * 62 + "B") is None


def test_selector_conflicting_label_is_rejected_before_update(tmp_path: Path):
    store = SessionStore(tmp_path / "assist")
    target = _record(store, tmp_path / "target")
    _insert_session(
        store,
        "20260825_120000_abc12345",
        tmp_path / "selector-owner",
    )

    with pytest.raises(
        ValueError,
        match=(
            "session label 'abc' conflicts with an existing "
            "session ID selector"
        ),
    ):
        store.set_session_label(target.session_id, "abc")

    assert store.get_session(target.session_id).label is None


def test_duplicate_replacement_rolls_back_to_the_previous_label(tmp_path: Path):
    store = SessionStore(tmp_path / "assist")
    first = _record(store, tmp_path / "first")
    second = _record(store, tmp_path / "second")
    store.set_session_label(first.session_id, "shared-label")
    store.set_session_label(second.session_id, "keep-me")

    with pytest.raises(
        ValueError,
        match="session label 'shared-label' is already assigned to another session",
    ):
        store.set_session_label(second.session_id, "shared-label")

    assert store.get_session(first.session_id).label == "shared-label"
    assert store.get_session(second.session_id).label == "keep-me"


def test_duplicate_writers_are_serialized_by_the_unique_index(tmp_path: Path):
    root = tmp_path / "assist"
    first_store = SessionStore(root)
    second_store = SessionStore(root)
    first = _record(first_store, tmp_path / "first")
    second = _record(first_store, tmp_path / "second")
    barrier = threading.Barrier(2)

    def assign(store: SessionStore, session_id: str) -> str:
        barrier.wait()
        try:
            store.set_session_label(session_id, "race-label")
        except ValueError as exc:
            return str(exc)
        return "ok"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda pair: assign(*pair),
                (
                    (first_store, first.session_id),
                    (second_store, second.session_id),
                ),
            )
        )

    assert sorted(results) == [
        "ok",
        "session label 'race-label' is already assigned to another session",
    ]
    assigned = [
        record.label
        for record in first_store.list_sessions()
        if record.label is not None
    ]
    assert assigned == ["race-label"]


def test_resolver_preserves_full_and_unique_prefix_selection(tmp_path: Path):
    store = SessionStore(tmp_path / "assist")
    record = _record(store, tmp_path / "work")
    store.set_session_label(record.session_id, "named-run")

    assert store.resolve_session_ref(record.session_id) == store.get_session(
        record.session_id
    )
    assert store.resolve_session_ref(record.short_id) == store.get_session(
        record.session_id
    )
    assert store.resolve_session_ref(record.short_id[:6]) == store.get_session(
        record.session_id
    )
    assert store.resolve_session_ref("named-run") == store.get_session(
        record.session_id
    )


def test_future_label_and_id_prefix_collision_fails_instead_of_selecting(
    tmp_path: Path,
):
    store = SessionStore(tmp_path / "assist")
    labeled = _record(store, tmp_path / "labeled")
    store.set_session_label(labeled.session_id, "abc")
    _insert_session(
        store,
        "20260825_120000_abc12345",
        tmp_path / "later-session",
    )

    with pytest.raises(
        AmbiguousSessionRefError,
        match=(
            "session ref 'abc' is ambiguous between an exact label "
            "and a session ID prefix; use the full session ID"
        ),
    ):
        store.resolve_session_ref("abc")


def test_existing_ambiguous_prefix_error_remains_stable(tmp_path: Path):
    store = SessionStore(tmp_path / "assist")
    _insert_session(
        store,
        "20260825_120000_abc11111",
        tmp_path / "first",
    )
    _insert_session(
        store,
        "20260825_120001_abc22222",
        tmp_path / "second",
    )

    with pytest.raises(
        AmbiguousSessionRefError,
        match=(
            "session ref 'abc' matches multiple sessions; "
            "use a longer prefix or the full id"
        ),
    ):
        store.resolve_session_ref("abc")


def test_label_cli_sets_replaces_and_clears_without_a_model_call(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    store = SessionStore(tmp_path / "assist")
    record = _record(store, tmp_path / "work")

    with (
        patch("scripts.llm_assist.__main__.SessionStore", return_value=store),
        patch("scripts.llm_assist.__main__._make_client") as make_client,
        patch("scripts.llm_assist.__main__.resolve_served_model") as resolve_model,
        patch("scripts.llm_assist.__main__.run_session") as run_session,
    ):
        assert main(["label", record.session_id, "First.Label"]) == 0
        assert main(["label", "First.Label", "replacement-label"]) == 0
        assert main(["label", "replacement-label", "--clear"]) == 0

    output = capsys.readouterr().out
    assert f"labeled: {record.session_id}" in output
    assert "label: First.Label" in output
    assert "label: replacement-label" in output
    assert f"label_cleared: {record.session_id}" in output
    assert "label: -" in output
    assert store.get_session(record.session_id).label is None
    make_client.assert_not_called()
    resolve_model.assert_not_called()
    run_session.assert_not_called()


def test_label_cli_requires_exactly_one_update_action(tmp_path: Path):
    store = SessionStore(tmp_path / "assist")
    record = _record(store, tmp_path / "work")

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        with pytest.raises(SystemExit, match="provide a label or --clear"):
            main(["label", record.session_id])
        with pytest.raises(
            SystemExit,
            match="label value and --clear are mutually exclusive",
        ):
            main(["label", record.session_id, "value", "--clear"])

    assert store.get_session(record.session_id).label is None


def test_sessions_status_show_and_current_render_label_or_dash(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    store = SessionStore(tmp_path / "assist")
    cwd = tmp_path / "work"
    cwd.mkdir()
    labeled = _record(store, cwd)
    unlabeled = _record(store, tmp_path / "other")
    store.set_session_label(labeled.session_id, "visible-label")
    store.set_active_session(cwd, labeled.session_id)
    monkeypatch.chdir(cwd)

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert main(["sessions"]) == 0
        sessions_output = capsys.readouterr().out
        assert "status     label" in sessions_output
        assert f"{labeled.session_id}  created    visible-label" in sessions_output
        assert f"{unlabeled.session_id}  created    -" in sessions_output

        assert main(["status", "visible-label"]) == 0
        status_output = capsys.readouterr().out
        assert (
            f"session_id: {labeled.session_id}\n"
            "label: visible-label\n"
            f"session_ref: {labeled.short_id}"
        ) in status_output

        assert main(["show", "visible-label"]) == 0
        show_output = capsys.readouterr().out
        assert (
            f"session_id: {labeled.session_id}\n"
            "label: visible-label\n"
            f"session_ref: {labeled.short_id}"
        ) in show_output

        assert main(["current"]) == 0
        assert "label: visible-label" in capsys.readouterr().out

        assert main(["status", unlabeled.session_id]) == 0
        assert "label: -" in capsys.readouterr().out


def test_resume_correct_and_answer_resolve_exact_labels(
    tmp_path: Path,
):
    store = SessionStore(tmp_path / "assist")
    resume_record = _record(store, tmp_path / "resume", status="paused")
    correct_record = _record(store, tmp_path / "correct", status="paused")
    answer_record = _record(
        store,
        tmp_path / "answer",
        status="input_required",
    )
    for record, label in (
        (resume_record, "resume-run"),
        (correct_record, "correct-run"),
        (answer_record, "answer-run"),
    ):
        store.set_session_label(record.session_id, label)
        record.artifact_path.mkdir(parents=True, exist_ok=True)

    request = create_clarification_request(
        answer_record.artifact_path,
        request_id="request-label-test",
        session_id=answer_record.session_id,
        session_number=1,
        turn_number=2,
        tool_call_id="ask-label-test",
        question="Which target?",
    )
    correction = {
        "correction_id": "corr-label-test",
        "session_id": correct_record.session_id,
        "after_session_number": 1,
        "text": "Use the exact target.",
        "text_sha256": "a" * 64,
    }

    with (
        patch("scripts.llm_assist.__main__.SessionStore", return_value=store),
        patch(
            "scripts.llm_assist.__main__.run_session",
            return_value=(True, "stop"),
        ) as run_session,
    ):
        assert main(["resume", "resume-run"]) == 0
    assert run_session.call_args.args[1].session_id == resume_record.session_id

    with (
        patch("scripts.llm_assist.__main__.SessionStore", return_value=store),
        patch(
            "scripts.llm_assist.__main__._correction_state_for_record",
            return_value=SimpleNamespace(phase="none"),
        ),
        patch(
            "scripts.llm_assist.__main__.derive_live_state",
            return_value=SimpleNamespace(status="paused", session_number=1),
        ),
        patch(
            "scripts.llm_assist.__main__.create_correction",
            return_value=correction,
        ) as create_correction,
        patch("scripts.llm_assist.__main__.append_trace_event_fsync"),
    ):
        assert main(["correct", "correct-run", correction["text"]]) == 0
    assert (
        create_correction.call_args.kwargs["session_id"]
        == correct_record.session_id
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert main(
            [
                "answer",
                "answer-run",
                request["request_id"],
                "Use PostgreSQL.",
            ]
        ) == 0
    answer = load_clarification_answer(answer_record.artifact_path)
    assert answer is not None
    assert answer["session_id"] == answer_record.session_id


def test_approval_rejection_and_rewind_resolve_exact_labels(tmp_path: Path):
    store = SessionStore(tmp_path / "assist")
    approved = _record(store, tmp_path / "approved", status="approval_pending")
    rejected = _record(store, tmp_path / "rejected", status="approval_pending")
    rewound = _record(store, tmp_path / "rewound", status="paused")
    for record, label in (
        (approved, "approve-run"),
        (rejected, "reject-run"),
        (rewound, "rewind-run"),
    ):
        store.set_session_label(record.session_id, label)
        record.artifact_path.mkdir(parents=True, exist_ok=True)
    for record in (approved, rejected):
        save_approval_request(
            record.artifact_path,
            {
                "status": "pending",
                "tool_name": "bash",
                "cmd": "make test",
                "args_summary": "cmd='make test'",
                "reason": "operator decision",
            },
        )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert main(["approve", "approve-run"]) == 0
        assert main(["reject", "reject-run"]) == 0

    event = {
        "from_turn": 2,
        "to_turn": 1,
        "commit": "c" * 40,
        "reason": "operator_cli",
    }
    with (
        patch("scripts.llm_assist.__main__.SessionStore", return_value=store),
        patch(
            "scripts.llm_assist.__main__._correction_state_for_record",
            return_value=SimpleNamespace(phase="none"),
        ),
        patch(
            "scripts.llm_assist.__main__.rewind_session",
            return_value=event,
        ) as rewind_session,
    ):
        assert main(["rewind", "rewind-run", "1"]) == 0
    assert rewind_session.call_args.args[1].session_id == rewound.session_id


def test_usage_and_worktree_removal_resolve_exact_labels(tmp_path: Path):
    store = SessionStore(tmp_path / "assist")
    usage_record = _record(store, tmp_path / "usage")
    worktree_record = _record(store, tmp_path / "repo")
    store.set_session_label(usage_record.session_id, "usage-run")
    store.set_session_label(worktree_record.session_id, "worktree-run")
    store.update_session_worktree(
        worktree_record.session_id,
        path=tmp_path / "owned-worktree",
        branch="yuj-worktree-label-test",
        base_commit="d" * 40,
    )
    worktree_record = store.get_session(worktree_record.session_id)
    assert worktree_record is not None

    with (
        patch("scripts.llm_assist.__main__.SessionStore", return_value=store),
        patch(
            "scripts.llm_assist.__main__.aggregate_session_usage",
            return_value=object(),
        ) as aggregate,
        patch(
            "scripts.llm_assist.__main__.render_session_usage",
            return_value=["usage: exact"],
        ),
    ):
        assert main(["usage", "usage-run"]) == 0
    assert aggregate.call_args.args[0] == [
        usage_record.artifact_path / ".trace.jsonl"
    ]

    inspection = SimpleNamespace(
        worktree_path=Path(worktree_record.worktree_path),
        branch=worktree_record.worktree_branch,
        base_commit=worktree_record.worktree_base_commit,
    )
    removed = SimpleNamespace(
        worktree_path=inspection.worktree_path,
        branch=inspection.branch,
        forced=False,
    )
    with (
        patch("scripts.llm_assist.__main__.SessionStore", return_value=store),
        patch(
            "scripts.llm_assist.__main__.inspect_session_worktree",
            return_value=inspection,
        ),
        patch(
            "scripts.llm_assist.__main__.remove_session_worktree",
            return_value=removed,
        ) as remove_worktree,
    ):
        assert main(["worktree", "rm", "worktree-run"]) == 0
    assert remove_worktree.call_args.args[1] == worktree_record.session_id


def test_label_changes_only_index_metadata_and_stays_out_of_evidence(
    tmp_path: Path,
):
    store = SessionStore(tmp_path / "assist")
    record = _record(store, tmp_path / "work", status="archived")
    artifact_dir = record.artifact_path
    artifact_dir.mkdir(parents=True)
    evidence = {
        "prompt.txt": b"Complete the task.\n",
        ".trace.jsonl": b'{"event":"session_end","finish_reason":"stop"}\n',
        "approval_request.json": b'{"status":"pending"}\n',
        "clarification_request.json": b'{"status":"pending"}\n',
        "correction.json": b'{"status":"pending"}\n',
    }
    for name, contents in evidence.items():
        (artifact_dir / name).write_bytes(contents)
    store.set_active_session(record.cwd, record.session_id)
    lock = store.acquire_session_lock(record.session_id)
    before_record = store.get_session(record.session_id)
    before_artifacts = _artifact_bytes(artifact_dir)
    measurement_before = load_config(overrides={"runtime_mode": "measurement"})

    store.set_session_label(record.session_id, "metadata-only")
    labeled = store.get_session(record.session_id)
    measurement_after = load_config(overrides={"runtime_mode": "measurement"})

    assert labeled is not None
    assert labeled.label == "metadata-only"
    assert _without_label(labeled) == _without_label(before_record)
    assert _artifact_bytes(artifact_dir) == before_artifacts
    assert store.get_active_session_id(record.cwd) == record.session_id
    assert store.get_session_lock(record.session_id) == lock
    assert measurement_after == measurement_before

    _write_session_metadata(labeled)
    metadata = json.loads((artifact_dir / "session.json").read_text())
    assert "label" not in metadata
    assert "metadata-only" not in json.dumps(metadata)
    store.release_session_lock(record.session_id)
