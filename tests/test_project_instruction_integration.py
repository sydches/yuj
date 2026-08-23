"""Runtime integration tests for opt-in project instruction files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._loop._driver_setup import (
    load_system_prompt_and_provenance,
)
from scripts.llm_solver.server.types import TurnResult, Usage

from _config_helpers import make_config


def test_canonical_project_doc_defaults_load() -> None:
    cfg = load_config()

    assert cfg.project_docs_enabled is False
    assert cfg.project_doc_names == ("AGENTS.md", "CLAUDE.md")
    assert cfg.project_doc_max_bytes == 32768
    assert cfg.project_root_markers == (".git", ".hg", ".sl")
    assert cfg.project_doc_global_dir == "~/.config/yuj"
    assert cfg.imports_enabled is True
    assert cfg.imports_max_depth == 5


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('project_docs_enabled = "yes"', "project_docs_enabled must be a boolean"),
        ('project_doc_names = ["../AGENTS.md"]', "project_doc_names entries"),
        ("project_doc_max_bytes = -1", "project_doc_max_bytes"),
        ('project_root_markers = ["nested/.git"]', "project_root_markers entries"),
        ("project_doc_global_dir = 4", "project_doc_global_dir must be a string"),
        ('imports_enabled = "yes"', "imports_enabled must be a boolean"),
        ("imports_max_depth = -1", "imports_max_depth must be a non-negative"),
        ("imports_max_depth = true", "imports_max_depth must be a non-negative"),
    ],
)
def test_project_doc_config_rejects_invalid_values(
    tmp_path: Path, body: str, message: str,
) -> None:
    overlay = tmp_path / "invalid.toml"
    overlay.write_text(f"[prompts]\n{body}\n")

    with pytest.raises(ValueError, match=message):
        load_config(user_config=overlay)


def test_prompt_assembly_order_provenance_and_default_off_identity(
    tmp_path: Path,
) -> None:
    work = tmp_path / "repo"
    global_dir = tmp_path / "global"
    work.mkdir()
    global_dir.mkdir()
    (work / ".git").mkdir()
    (work / "AGENTS.md").write_text("PROJECT")
    (global_dir / "AGENTS.md").write_text("GLOBAL")
    arm = tmp_path / "arm.md"
    arm.write_text("ARM\n")
    client = SimpleNamespace(profile=SimpleNamespace(preamble="PROFILE"))

    disabled = make_config(system_header="HEADER")
    prompt, _provenance, _contract, metadata = (
        load_system_prompt_and_provenance(
            disabled, client, work, arm, None, None, None,
        )
    )
    assert prompt == "PROFILE\n\nARM\n\nHEADER"
    assert metadata.trace_fields() == {
        "project_instruction_files": [],
        "project_instruction_bytes": 0,
        "project_instruction_imported_bytes": 0,
        "project_instruction_resolved_bytes": 0,
        "project_instructions_truncated": False,
        "prompt_import_tree": [{
            "owner": "system_prompt",
            "source": "arm.md",
            "source_bytes": len("ARM\n"),
            "imported_bytes": 0,
            "imports": [],
        }],
    }

    enabled = make_config(
        system_header="HEADER",
        project_docs_enabled=True,
        project_doc_global_dir=str(global_dir),
    )
    prompt, provenance, _contract, metadata = load_system_prompt_and_provenance(
        enabled, client, work, arm, None, None, None,
    )

    global_block = (
        '<project-instructions path="global/AGENTS.md">\n'
        "GLOBAL\n</project-instructions>"
    )
    project_block = (
        '<project-instructions path="AGENTS.md">\n'
        "PROJECT\n</project-instructions>"
    )
    assert prompt == (
        f"PROFILE\n\nARM\n\n{global_block}\n\n{project_block}\n\nHEADER"
    )
    assert provenance["system_prompt_sha256"] == hashlib.sha256(
        prompt.encode()
    ).hexdigest()[:16]
    assert provenance["system_prompt_chars"] == len(prompt)
    assert [
        item["path"] for item in metadata.project_instruction_files
    ] == ["global/AGENTS.md", "AGENTS.md"]
    assert [
        item["owner"] for item in metadata.prompt_import_tree
    ] == ["system_prompt", "project_instruction", "project_instruction"]
    assert str(tmp_path) not in json.dumps(metadata.trace_fields())


def test_solve_task_traces_project_docs_and_costs_resolved_blocks(
    tmp_path: Path,
) -> None:
    from scripts.llm_solver._shared.telemetry_paths import trace_path
    from scripts.llm_solver.harness.loop import solve_task

    work = tmp_path / "task"
    work.mkdir()
    (work / ".git").mkdir()
    (work / "AGENTS.md").write_text("TASK RULE")
    (work / "prompt.txt").write_text("finish")
    savings_dir = tmp_path / "savings"
    client = MagicMock()
    client.chat.return_value = TurnResult(
        content="done",
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(prompt_tokens=10, completion_tokens=2),
    )
    client.build_assistant_message.return_value = {
        "role": "assistant",
        "content": "done",
    }
    cfg = make_config(
        max_sessions=1,
        project_docs_enabled=True,
        project_doc_global_dir="",
    )

    with patch("scripts.llm_solver.harness.loop._auto_commit"):
        assert solve_task(work, cfg, client, savings_dir=savings_dir) is True

    events = [
        json.loads(line)
        for line in trace_path(work).read_text().splitlines()
        if line.strip()
    ]
    start = next(event for event in events if event["event"] == "session_start")
    assert start["project_instruction_files"] == [
        {
            "path": "AGENTS.md",
            "bytes": len("TASK RULE"),
            "scope": "project",
            "truncated": False,
        }
    ]
    assert start["project_instruction_bytes"] == len("TASK RULE")
    assert start["project_instruction_imported_bytes"] == 0
    assert start["project_instruction_resolved_bytes"] == len("TASK RULE")
    assert start["project_instructions_truncated"] is False
    assert str(tmp_path) not in json.dumps(start)

    ledger = [
        json.loads(line)
        for line in (savings_dir / f"{work.name}.jsonl").read_text().splitlines()
    ]
    project_cost = next(
        row for row in ledger if row["bucket"] == "project_instructions"
    )
    expected_block = (
        '<project-instructions path="AGENTS.md">\n'
        "TASK RULE\n</project-instructions>"
    )
    assert project_cost["output_chars"] == len(expected_block)
    assert project_cost["ctx"]["files"] == ["AGENTS.md"]
    assert str(tmp_path) not in json.dumps(project_cost)

    state_path = work / ".solver" / "state.json"
    if state_path.exists():
        assert "project_instruction" not in state_path.read_text()
