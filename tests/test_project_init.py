"""Safe project-instruction initialization through the assistant CLI."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.llm_assist.__main__ import (
    _project_init_overrides,
    _render_provider_overlay,
    _validate_project_init_destination,
    main,
)
from scripts.llm_assist.runner import load_approval_request
from scripts.llm_assist.store import SessionStore
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._loop.profile_resolution import build_tool_surface
from scripts.llm_solver.harness.approval_preview import build_approval_preview
from scripts.llm_solver.harness.approvals import approval_decision
from scripts.llm_solver.harness.sandbox.ignore_policy import (
    PROJECT_INIT_PRIVATE_RULES,
    load_ignore_policy,
)
from scripts.llm_solver.harness.tool_policy import PermissionPolicy
from scripts.llm_solver.harness.tool_validation import ToolSchemaSet


def _init_config(destination: str = "AGENTS.md"):
    base = load_config(overrides={"runtime_mode": "assistant", "max_sessions": 1})
    overrides = _project_init_overrides(destination, base)
    return load_config(overrides={
        "runtime_mode": "assistant",
        "max_sessions": 1,
        **overrides,
    }), overrides


def test_init_parser_builds_a_fixed_bounded_task(tmp_path: Path) -> None:
    captured = {}

    def fake_run(args):
        captured.update(vars(args))
        return 0

    with patch("scripts.llm_assist.__main__.cmd_run", side_effect=fake_run):
        assert main([
            "init", "-C", str(tmp_path), "--output", "AGENTS.md",
        ]) == 0

    assert captured["cwd"] == tmp_path.resolve()
    assert captured["project_init_destination"] == "AGENTS.md"
    assert captured["context"] == "full"
    assert captured["edit_format"] == "whole"
    assert captured["plan_mode"] == "off"
    assert "within 80 lines and 8,000 characters" in captured["prompt_text"]
    assert "Do not copy secrets" in captured["prompt_text"]
    assert "call write once" in captured["prompt_text"]


@pytest.mark.parametrize(
    "name",
    ("nested/AGENTS.md", "/tmp/AGENTS.md", "AGENTS.txt", "AG?NTS.md"),
)
def test_init_rejects_ambiguous_destinations(
    tmp_path: Path, name: str,
) -> None:
    with pytest.raises(SystemExit):
        main(["init", "-C", str(tmp_path), "--output", name])


def test_init_rejects_unconfigured_and_ignored_destinations(
    tmp_path: Path,
) -> None:
    cfg = load_config(overrides={"runtime_mode": "assistant", "max_sessions": 1})
    with pytest.raises(SystemExit, match="configured instruction filename"):
        _validate_project_init_destination(tmp_path, "RULES.md", cfg)

    (tmp_path / ".yujignore").write_text("/AGENTS.md\n")
    with pytest.raises(SystemExit, match="configured ignore file"):
        _validate_project_init_destination(tmp_path, "AGENTS.md", cfg)


def test_init_rejects_a_new_gitignored_destination(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("/AGENTS.md\n")
    cfg = load_config(overrides={"runtime_mode": "assistant", "max_sessions": 1})

    with pytest.raises(SystemExit, match="ignored by Git"):
        _validate_project_init_destination(tmp_path, "AGENTS.md", cfg)


def test_init_overlay_is_durable_and_preserves_user_permissions(
    tmp_path: Path,
) -> None:
    user = tmp_path / "user.toml"
    user.write_text(
        "[permissions.rules.read]\n"
        '"private/*" = "deny"\n'
        "[permissions.rules.write]\n"
        '"CLAUDE.md" = "deny"\n'
    )
    base = load_config(
        user_config=user,
        overrides={"runtime_mode": "assistant", "max_sessions": 1},
    )
    overrides = _project_init_overrides("AGENTS.md", base)
    overlay = tmp_path / "provider.toml"
    overlay.write_text(_render_provider_overlay(overrides))
    restored = load_config(
        user_config=[user, overlay],
        overrides={"runtime_mode": "assistant", "max_sessions": 1},
    )

    assert restored.assistant_project_init_destination == "AGENTS.md"
    assert restored.assistant_project_init_max_chars == 8000
    assert restored.assistant_project_init_max_lines == 80
    assert restored.tools_lazy_loading_enabled is False
    assert restored.tools_schema_validation == "reject"
    assert restored.tools_constrained_decoding == "off"
    assert restored.tools_edit_format == "whole"
    assert restored.runtime_worktree == "off"
    assert restored.state_ignore_file_names[0] == ".gitignore"

    policy = PermissionPolicy.from_rule_tables(restored.permissions_rules)
    allowed_read = policy.evaluate(
        tool_name="read",
        arguments={"path": "README.md"},
        runtime_mode="assistant",
    )
    private_read = policy.evaluate(
        tool_name="read",
        arguments={"path": "private/notes.md"},
        runtime_mode="assistant",
    )
    selected_write = policy.evaluate(
        tool_name="write",
        arguments={"path": "AGENTS.md", "content": "# Project\n"},
        runtime_mode="assistant",
    )
    wrong_write = policy.evaluate(
        tool_name="write",
        arguments={"path": "CLAUDE.md", "content": "# Wrong\n"},
        runtime_mode="assistant",
    )
    allowed, reason = approval_decision(
        runtime_mode="assistant",
        cwd=str(tmp_path),
        trace_path=tmp_path / ".trace.jsonl",
        tool_name="write",
        tool_args={"path": "AGENTS.md", "content": "# Project\n"},
        args_summary="path='AGENTS.md'",
        cfg=restored,
    )
    assert allowed_read.allowed
    assert private_read.denied
    assert selected_write.allowed
    assert wrong_write.denied
    assert allowed is False
    assert reason == "project instruction writes require operator approval"
    assert base.permissions_rules == restored.permissions_rules

    request_path = tmp_path / "approval_request.json"
    request = json.loads(request_path.read_text())
    request["status"] = "approved"
    request_path.write_text(json.dumps(request) + "\n")
    approved, _reason = approval_decision(
        runtime_mode="assistant",
        cwd=str(tmp_path),
        trace_path=tmp_path / ".trace.jsonl",
        tool_name="write",
        tool_args={"path": "AGENTS.md", "content": "# Project\n"},
        args_summary="path='AGENTS.md'",
        cfg=restored,
    )
    changed, _reason = approval_decision(
        runtime_mode="assistant",
        cwd=str(tmp_path),
        trace_path=tmp_path / ".trace.jsonl",
        tool_name="write",
        tool_args={"path": "AGENTS.md", "content": "# Changed\n"},
        args_summary="path='AGENTS.md'",
        cfg=restored,
    )
    assert approved is True
    assert changed is False


def test_init_write_schema_pins_path_size_and_line_count() -> None:
    cfg, _overrides = _init_config()
    client = SimpleNamespace()
    surface = build_tool_surface(cfg, client)
    names = set(surface.active_names)
    schemas = ToolSchemaSet.from_openai_tools(surface.active_schemas)

    assert names == {
        "read", "write", "glob", "grep", "ask_user", "done",
    }
    assert schemas.validate(
        "write", {"path": "AGENTS.md", "content": "# Project\n"}
    ).valid
    assert not schemas.validate(
        "write", {"path": "CLAUDE.md", "content": "# Project\n"}
    ).valid
    assert not schemas.validate(
        "write", {"path": "AGENTS.md", "content": "x" * 8001}
    ).valid
    assert schemas.validate(
        "write",
        {"path": "AGENTS.md", "content": "".join(["x\n"] * 80)},
    ).valid
    assert not schemas.validate(
        "write",
        {"path": "AGENTS.md", "content": "\n".join(["x"] * 81)},
    ).valid


def test_init_preview_shows_the_complete_proposal_without_writing(
    tmp_path: Path,
) -> None:
    target = tmp_path / "AGENTS.md"
    target.write_text("".join(f"old {index}\n" for index in range(500)))
    proposal = "".join(f"new {index}\n" for index in range(70))
    cfg, _overrides = _init_config()

    preview = build_approval_preview(
        cwd=str(tmp_path),
        tool_name="write",
        tool_args={"path": "AGENTS.md", "content": proposal},
        cfg=cfg,
    )

    assert preview["status"] == "available"
    assert preview["format"] == "complete_file"
    assert preview["paths"] == ["AGENTS.md"]
    assert preview["content"] == proposal
    assert preview["truncated"] is False
    assert target.read_text().startswith("old 0\n")


def test_init_private_paths_are_hidden_from_repository_tools(
    tmp_path: Path,
) -> None:
    for name in (".git", ".internal", ".solver", ".tool_output", ".procs"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "private.txt").write_text("private\n")
    (tmp_path / "README.md").write_text("public\n")

    policy = load_ignore_policy(
        tmp_path,
        builtin_rules=PROJECT_INIT_PRIVATE_RULES,
    )

    assert policy.is_model_hidden(".internal", is_dir=True)
    assert policy.is_ignored(".git/private.txt", is_dir=False)
    assert not policy.is_ignored("README.md", is_dir=False)
    assert policy.source_names[0] == "<project-init-private-paths>"


def test_init_cli_prints_exact_pending_proposal_and_changes_no_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = SessionStore(tmp_path / "assist-home")
    work = tmp_path / "work"
    work.mkdir()
    subprocess.run(["git", "init", "-q", str(work)], check=True)
    proposal = "# Project instructions\n\nRun `pytest -q` before submitting.\n"

    def fake_run_session(store_obj, record, *, resume):
        assert resume is False
        cfg = load_config(
            user_config=[Path(path) for path in record.config_paths],
            overrides={"runtime_mode": "assistant", "max_sessions": 1},
        )
        trace = record.artifact_path / ".trace.jsonl"
        allowed, _reason = approval_decision(
            runtime_mode="assistant",
            cwd=str(work),
            trace_path=trace,
            tool_name="write",
            tool_args={"path": "AGENTS.md", "content": proposal},
            args_summary="path='AGENTS.md'",
            cfg=cfg,
        )
        assert allowed is False
        with trace.open("a") as handle:
            handle.write(json.dumps({
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "approval_required",
                "turns": 1,
            }) + "\n")
        store_obj.update_session(
            record.session_id,
            status="approval_pending",
            last_finish_reason="approval_required",
        )
        return False, "approval_required"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch("scripts.llm_assist.__main__.preflight_assistant_startup"), \
            patch(
                "scripts.llm_assist.__main__.resolve_served_model",
                return_value=("test-model", ["test-model"]),
            ), \
            patch(
                "scripts.llm_assist.__main__.run_session",
                side_effect=fake_run_session,
            ):
        rc = main([
            "init", "-C", str(work), "--output", "AGENTS.md",
        ])

    output = capsys.readouterr().out
    record = store.list_sessions(limit=1)[0]
    approval = load_approval_request(record.artifact_path)
    assert rc == 1
    assert record.context_mode == "full"
    assert approval is not None and approval["status"] == "pending"
    assert approval["preview"]["content"] == proposal
    assert f"instruction_destination: {work / 'AGENTS.md'}" in output
    assert "approval_preview_format: complete_file" in output
    assert "Run `pytest -q` before submitting." in output
    assert f"approve_with: yuj approve {record.short_id}" in output
    assert not (work / "AGENTS.md").exists()
