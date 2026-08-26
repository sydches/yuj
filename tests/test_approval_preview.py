"""Read-only file previews for assistant approval requests."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.llm_assist.__main__ import main
from scripts.llm_assist.runner import load_approval_request
from scripts.llm_assist.store import SessionStore
from scripts.llm_solver.harness.approval_preview import (
    build_approval_preview,
    render_approval_preview,
)
from scripts.llm_solver.harness.approvals import approval_decision


def test_write_preview_compares_workspace_with_proposal_without_mutation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before\n")

    preview = build_approval_preview(
        cwd=str(tmp_path),
        tool_name="write",
        tool_args={"path": "notes.txt", "content": "after\n"},
    )

    assert preview["status"] == "available"
    assert preview["format"] == "unified_diff"
    assert preview["paths"] == ["notes.txt"]
    assert "--- a/notes.txt (current workspace)" in preview["content"]
    assert "+++ b/notes.txt (proposed, not applied)" in preview["content"]
    assert "-before" in preview["content"]
    assert "+after" in preview["content"]
    assert target.read_text() == "before\n"


def test_exact_edit_preview_matches_first_replacement_without_mutation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("same\nsame\n")

    preview = build_approval_preview(
        cwd=str(tmp_path),
        tool_name="edit",
        tool_args={
            "path": "notes.txt",
            "old_str": "same",
            "new_str": "changed",
        },
    )

    assert preview["status"] == "available"
    assert str(preview["content"]).count("+changed") == 1
    assert target.read_text() == "same\nsame\n"


@pytest.mark.parametrize(
    ("tool_name", "proposal", "format_name"),
    [
        (
            "apply_patch",
            "*** Begin Patch\n"
            "*** Update File: notes.txt\n"
            "@@\n"
            "-before\n"
            "+after\n"
            "*** End Patch\n",
            "apply_patch",
        ),
        (
            "udiff",
            "--- a/notes.txt\n"
            "+++ b/notes.txt\n"
            "@@ -1 +1 @@\n"
            "-before\n"
            "+after\n",
            "unified_diff",
        ),
    ],
)
def test_patch_preview_preserves_exact_proposal_without_applying_it(
    tmp_path: Path,
    tool_name: str,
    proposal: str,
    format_name: str,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before\n")

    preview = build_approval_preview(
        cwd=str(tmp_path),
        tool_name=tool_name,
        tool_args={"patch": proposal},
    )

    assert preview["status"] == "available"
    assert preview["format"] == format_name
    assert preview["paths"] == ["notes.txt"]
    assert preview["content"] == proposal
    assert target.read_text() == "before\n"


def test_large_preview_is_bounded_and_marks_omitted_content(
    tmp_path: Path,
) -> None:
    proposal = "".join(f"line {index}\n" for index in range(2_000))

    preview = build_approval_preview(
        cwd=str(tmp_path),
        tool_name="write",
        tool_args={"path": "large.txt", "content": proposal},
    )

    assert preview["status"] == "available"
    assert preview["truncated"] is True
    assert preview["shown_chars"] < preview["original_chars"]
    assert "preview truncated: showing" in preview["content"]
    assert not (tmp_path / "large.txt").exists()


def test_unrepresentable_requests_say_why_without_misleading_content(
    tmp_path: Path,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("current\n")

    shell = build_approval_preview(
        cwd=str(tmp_path),
        tool_name="bash",
        tool_args={"cmd": "rm notes.txt"},
    )
    stale_edit = build_approval_preview(
        cwd=str(tmp_path),
        tool_name="edit",
        tool_args={
            "path": "notes.txt",
            "old_str": "missing",
            "new_str": "replacement",
        },
    )
    malformed_patch = build_approval_preview(
        cwd=str(tmp_path),
        tool_name="apply_patch",
        tool_args={"patch": "not a patch"},
    )

    assert shell["status"] == "unavailable"
    assert "dynamically" in shell["summary"]
    assert stale_edit["status"] == "unavailable"
    assert "cannot be previewed safely" in stale_edit["summary"]
    assert malformed_patch["status"] == "unavailable"
    assert malformed_patch["content"] == ""
    assert target.read_text() == "current\n"


def test_preview_escapes_terminal_control_content(tmp_path: Path) -> None:
    preview = build_approval_preview(
        cwd=str(tmp_path),
        tool_name="write",
        tool_args={
            "path": "notes.txt",
            "content": "safe\n\x1b]0;title\x07unsafe\n",
        },
    )
    rendered = render_approval_preview(preview)

    assert "\x1b" not in rendered
    assert "\x07" not in rendered
    assert "\\x1b" in rendered
    assert "\\x07" in rendered


def test_approval_request_saves_preview_and_show_renders_it_before_rejection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    work = tmp_path / "work"
    work.mkdir()
    target = work / "notes.txt"
    target.write_text("before\n")
    record = store.create_session(
        cwd=work,
        model="test-model",
        prompt_text="private task text",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    record.artifact_path.mkdir(parents=True)
    trace_path = record.artifact_path / ".trace.jsonl"
    trace_path.write_text(json.dumps({
        "event": "session_end",
        "session_number": 1,
        "finish_reason": "approval_required",
        "turns": 1,
    }) + "\n")

    allowed, _reason = approval_decision(
        runtime_mode="assistant",
        cwd=str(work),
        trace_path=trace_path,
        tool_name="write",
        tool_args={"path": "notes.txt", "content": "after\n"},
        args_summary="path='notes.txt', content='after'",
        required_reason="permission rule requires approval",
        permission_rule="*",
    )
    store.update_session(
        record.session_id,
        status="approval_pending",
        last_finish_reason="approval_required",
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        show_rc = main([
            "show",
            record.session_id,
            "--turns",
            "0",
            "--no-trace",
            "--no-pager",
        ])
    shown = capsys.readouterr().out

    assert allowed is False
    assert show_rc == 0
    assert "approval_preview_state: proposed; not applied" in shown
    assert "approval_preview_paths:\n  notes.txt" in shown
    assert "-before" in shown
    assert "+after" in shown
    assert target.read_text() == "before\n"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        reject_rc = main([
            "reject",
            record.session_id,
            "--reason",
            "keep the current file",
        ])

    request = load_approval_request(record.artifact_path)
    assert reject_rc == 0
    assert request is not None
    assert request["status"] == "rejected"
    assert target.read_text() == "before\n"
