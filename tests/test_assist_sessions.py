"""Compact listing, filters, and explicit session selection."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.llm_assist.__main__ import main
from scripts.llm_assist.store import SessionStore


def _record(
    store: SessionStore,
    cwd: Path,
    *,
    model: str,
    status: str,
    label: str | None = None,
    finish_reason: str | None = None,
):
    cwd.mkdir(parents=True, exist_ok=True)
    record = store.create_session(
        cwd=cwd,
        model=model,
        prompt_text="task",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    store.update_session(
        record.session_id,
        status=status,
        last_finish_reason=finish_reason,
    )
    if label is not None:
        store.set_session_label(record.session_id, label)
    saved = store.get_session(record.session_id)
    assert saved is not None
    return saved


def test_default_listing_is_bounded_and_full_listing_keeps_exact_fields(
    tmp_path, capsys
):
    store = SessionStore(tmp_path / "assist")
    cwd = tmp_path / ("very-long-directory-name-" * 4) / "repository"
    record = _record(
        store,
        cwd,
        model="provider/" + "very-long-model-name-" * 5,
        status="approval_pending",
        label="release-" + "long-label-" * 5,
        finish_reason="approval_required",
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert main(["sessions"]) == 0
        compact = capsys.readouterr().out
        assert main(["sessions", "--full"]) == 0
        full = capsys.readouterr().out

    assert all(len(line) <= 80 for line in compact.splitlines())
    assert record.short_id in compact
    assert record.session_id not in compact
    assert record.model not in compact
    assert record.cwd.endswith("repository")
    assert record.session_id in full
    assert f"label: {record.label}" in full
    assert f"status: {record.status}" in full
    assert "finish_reason: approval_required" in full
    assert f"model: {record.model}" in full
    assert f"cwd: {record.cwd}" in full


def test_listing_filters_before_limit_and_supports_all_archive_states(
    tmp_path, capsys
):
    store = SessionStore(tmp_path / "assist")
    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    paused = _record(
        store,
        first_cwd,
        model="model-a",
        status="paused",
        label="paused-one",
    )
    archived = _record(
        store,
        first_cwd,
        model="model-b",
        status="completed",
        label="archived-one",
    )
    store.archive_session(archived.session_id)
    newest = _record(
        store,
        second_cwd,
        model="model-c",
        status="running",
        label="running-one",
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert main(["sessions", "--limit", "1", "--status", "paused"]) == 0
        status_output = capsys.readouterr().out
        assert main(
            [
                "sessions",
                "--all",
                "--status",
                "paused",
                "--cwd",
                str(first_cwd),
                "--label",
                "paused-one",
            ]
        ) == 0
        combined_output = capsys.readouterr().out
        assert main(["sessions", "--archived"]) == 0
        archived_output = capsys.readouterr().out
        assert main(
            [
                "sessions",
                "--all",
                "--status",
                "paused",
                "--status",
                "running",
            ]
        ) == 0
        repeated_status_output = capsys.readouterr().out
        assert main(["sessions", "--all"]) == 0
        all_output = capsys.readouterr().out

    assert paused.short_id in status_output
    assert newest.short_id not in status_output
    assert paused.short_id in combined_output
    assert archived.short_id not in combined_output
    assert newest.short_id not in combined_output
    assert archived.short_id in archived_output
    assert paused.short_id not in archived_output
    assert paused.short_id in repeated_status_output
    assert newest.short_id in repeated_status_output
    assert archived.short_id not in repeated_status_output
    assert {paused.short_id, archived.short_id, newest.short_id} <= set(
        all_output.split()
    )


def test_interactive_selector_returns_an_immutable_identity_without_mutation(
    tmp_path, capsys
):
    store = SessionStore(tmp_path / "assist")
    _record(
        store,
        tmp_path / "one",
        model="model-a",
        status="paused",
        label="first-choice",
    )
    _record(
        store,
        tmp_path / "two",
        model="model-b",
        status="paused",
        label="second-choice",
    )
    eligible = store.list_sessions(statuses=("paused",))
    expected = eligible[1]
    before = store.list_sessions(archived=None)

    with (
        patch("scripts.llm_assist.__main__.SessionStore", return_value=store),
        patch("scripts.llm_assist.__main__._is_interactive", return_value=True),
        patch("builtins.input", return_value="2") as prompt,
    ):
        assert main(["sessions", "--status", "paused", "--select"]) == 0

    output = capsys.readouterr().out
    prompt.assert_called_once()
    assert f"selected_session_id: {expected.session_id}" in output
    assert f"selected_session_ref: {expected.short_id}" in output
    assert f"selected_label: {expected.label}" in output
    assert f"next: yuj show {expected.session_id}" in output
    assert store.list_sessions(archived=None) == before


def test_noninteractive_selector_never_prompts(tmp_path):
    store = SessionStore(tmp_path / "assist")
    _record(
        store,
        tmp_path / "one",
        model="model",
        status="paused",
    )

    with (
        patch("scripts.llm_assist.__main__.SessionStore", return_value=store),
        patch("scripts.llm_assist.__main__._is_interactive", return_value=False),
        patch("builtins.input") as prompt,
        pytest.raises(SystemExit, match="interactive terminal"),
    ):
        main(["sessions", "--select"])

    prompt.assert_not_called()


def test_listing_rejects_a_nonpositive_limit(tmp_path):
    store = SessionStore(tmp_path / "assist")
    with (
        patch("scripts.llm_assist.__main__.SessionStore", return_value=store),
        pytest.raises(SystemExit, match="at least 1"),
    ):
        main(["sessions", "--limit", "0"])
