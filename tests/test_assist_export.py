"""Redacted, deterministic assistant-session export."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.llm_assist.__main__ import main
from scripts.llm_assist.store import SessionStore


def _write_trace(path: Path, events: list[dict]) -> None:
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def _transcript_turn(turn: int, request: dict, response: dict) -> str:
    return (
        f"=== turn {turn:03d} input ===\n"
        + json.dumps(request)
        + "\n"
        + f"=== turn {turn:03d} output ===\n"
        + json.dumps(response)
        + "\n"
    )


def _response(content: str, *, tool_calls=None) -> dict:
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(
            candidate for candidate in root.rglob("*") if candidate.is_file()
        )
    }


def _record(tmp_path: Path, *, prompt: str = "Fix the task"):
    assist_root = tmp_path / "assist"
    workspace = tmp_path / "private-host" / "repository"
    workspace.mkdir(parents=True)
    private_config = tmp_path / "private-config.toml"
    private_config.write_text("[model]\n")
    private_prompt = tmp_path / "private-system.md"
    private_prompt.write_text("private system prompt\n")
    store = SessionStore(assist_root)
    record = store.create_session(
        cwd=workspace,
        model="test-model",
        prompt_text=prompt,
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=private_prompt,
        config_paths=[private_config],
        provider="openai",
        auth_method="api-key",
        credential_id="private-credential-id",
    )
    record.artifact_path.mkdir(parents=True)
    return store, record, workspace, private_config, private_prompt


def test_export_is_stable_read_only_and_redacts_private_evidence(
    tmp_path, capsys
):
    secret = "sk-" + "A" * 40
    expected_workspace = tmp_path / "private-host" / "repository"
    store, record, workspace, private_config, private_prompt = _record(
        tmp_path,
        prompt=f"Fix {expected_workspace}/file.py with {secret}",
    )
    assert workspace == expected_workspace
    store.update_session(
        record.session_id,
        status="completed",
        last_finish_reason="stop",
    )
    store.set_session_label(record.session_id, "share-me")
    saved = store.get_session(record.session_id)
    assert saved is not None
    prompt = saved.prompt_text
    store.archive_session(saved.session_id)
    saved = store.get_session(saved.session_id)
    assert saved is not None

    trace = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 1,
            "tool_name": "read",
            "args_summary": f"path={saved.cwd}/file.py password=hunter2",
            "result_summary": f"token={secret}",
            "reasoning": "PRIVATE MODEL REASONING",
            "outcome": "success",
        },
        {
            "event": "session_usage",
            "session_number": 1,
            "scope": "all_model_responses",
            "input_tokens": 100,
            "output_tokens": 20,
            "cached_tokens": 25,
            "cost": None,
            "quota": None,
        },
        {
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "stop",
            "turns": 2,
        },
    ]
    _write_trace(saved.artifact_path / ".trace.jsonl", trace)
    tool_call = [{"id": "call-1", "function": {"name": "read", "arguments": "{}"}}]
    transcript = _transcript_turn(
        1,
        {
            "messages": [
                {"role": "system", "content": "RAW PRIVATE SYSTEM CONTENT"},
                {"role": "user", "content": prompt},
            ]
        },
        _response("PRIVATE MODEL REASONING", tool_calls=tool_call),
    )
    transcript += _transcript_turn(
        2,
        {
            "messages": [
                {"role": "system", "content": "RAW PRIVATE SYSTEM CONTENT"},
                {"role": "user", "content": prompt},
                {"role": "tool", "content": f"token={secret}"},
            ]
        },
        _response(f"Completed the requested change. Authorization: Bearer {secret}"),
    )
    (saved.artifact_path / "transcript.log").write_text(transcript)
    before = _tree_bytes(store.root)

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert main(["export", "share-me", "--no-pager"]) == 0
        first = capsys.readouterr().out
        assert main(["export", saved.short_id, "--no-pager"]) == 0
        second = capsys.readouterr().out

    assert first == second
    assert "# Yuj session report" in first
    assert "`yuj-session-report-v1`" in first
    assert "## Task" in first
    assert "## Conversation" in first
    assert "Completed the requested change" in first
    assert "## Tool activity" in first
    assert "password=[REDACTED:assigned_secret]" in first
    assert "[REDACTED:openai_key]" in first
    assert "[REDACTED:authorization]" in first
    assert "## Usage" in first
    assert "input_tokens: 100" in first
    assert "## Provenance" in first
    assert "PRIVATE MODEL REASONING" not in first
    assert "RAW PRIVATE SYSTEM CONTENT" not in first
    assert secret not in first
    assert saved.cwd not in first
    assert saved.artifact_dir not in first
    assert str(private_config.resolve()) not in first
    assert str(private_prompt.resolve()) not in first
    assert "private-credential-id" not in first
    assert _tree_bytes(store.root) == before


def test_export_reconstructs_digest_bound_followups_and_final_responses(
    tmp_path, capsys
):
    store, record, _workspace, _config, _prompt = _record(
        tmp_path, prompt="Initial task"
    )
    store.update_session(record.session_id, status="completed", last_finish_reason="stop")
    saved = store.get_session(record.session_id)
    assert saved is not None
    followup = "Also check the empty-input case."
    _write_trace(
        saved.artifact_path / ".trace.jsonl",
        [
            {"event": "session_start", "session_number": 1},
            {
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "max_turns",
                "turns": 1,
            },
            {
                "event": "operator_followup",
                "session_number": 2,
                "prompt_source": "inline",
                "text_sha256": hashlib.sha256(followup.encode()).hexdigest(),
                "text_chars": len(followup),
            },
            {"event": "session_start", "session_number": 2},
            {
                "event": "session_end",
                "session_number": 2,
                "finish_reason": "stop",
                "turns": 1,
            },
        ],
    )
    (saved.artifact_path / "transcript.pre_seg_1.log").write_text(
        _transcript_turn(
            1,
            {"messages": [{"role": "user", "content": "Initial task"}]},
            _response("First response"),
        )
    )
    (saved.artifact_path / "transcript.log").write_text(
        _transcript_turn(
            1,
            {
                "messages": [
                    {"role": "user", "content": "Initial task"},
                    {"role": "assistant", "content": "First response"},
                    {"role": "user", "content": followup},
                ]
            },
            _response("Second response"),
        )
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        assert main(["export", saved.session_id, "--no-pager"]) == 0

    output = capsys.readouterr().out
    assert output.count(followup) == 1
    assert "### Operator follow-up (session 2, turn 0)" in output
    assert "### Assistant response (session 1, turn 1)" in output
    assert "First response" in output
    assert "### Assistant response (session 2, turn 1)" in output
    assert "Second response" in output
    assert "Transcript segments: `2`" in output


def test_export_refuses_linked_raw_evidence_without_reading_it(tmp_path):
    store, record, _workspace, _config, _prompt = _record(tmp_path)
    saved = store.get_session(record.session_id)
    assert saved is not None
    outside_secret = tmp_path / "outside-secret"
    outside_secret.write_text("DO-NOT-READ-THIS")
    (saved.artifact_path / "transcript.log").symlink_to(outside_secret)

    with (
        patch("scripts.llm_assist.__main__.SessionStore", return_value=store),
        pytest.raises(SystemExit, match="regular owned file") as exc_info,
    ):
        main(["export", saved.session_id, "--no-pager"])

    assert "DO-NOT-READ-THIS" not in str(exc_info.value)


def test_export_help_describes_stdout_markdown(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["export", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "redacted Markdown session report" in output
    assert "--pager" in output
    assert "--no-pager" in output
