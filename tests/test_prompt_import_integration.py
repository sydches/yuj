"""Runtime acceptance tests for instruction-file ``@imports``."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from scripts.llm_solver.harness._loop._driver_setup import (
    load_system_prompt_and_provenance,
)
from scripts.llm_solver.harness.injections import load_injections_with_metadata
from scripts.llm_solver.server.types import TurnResult, Usage

from _config_helpers import make_config


def _client(*, preamble: str = "") -> SimpleNamespace:
    return SimpleNamespace(profile=SimpleNamespace(preamble=preamble))


def test_arm_import_switch_order_allowed_roots_and_final_hash(tmp_path: Path) -> None:
    work = tmp_path / "repo"
    arms = tmp_path / "arms"
    work.mkdir()
    arms.mkdir()
    (work / ".git").mkdir()
    (work / "shared.md").write_text("ARM IMPORT")
    (tmp_path / "secret.md").write_text("HOST SECRET")
    arm = arms / "arm.md"
    arm.write_text(f"ARM START\n@{work / 'shared.md'}\n@../secret.md\nARM END\n")

    enabled = make_config(system_header="HEADER", imports_enabled=True)
    prompt, provenance, _contract, metadata = load_system_prompt_and_provenance(
        enabled, _client(preamble="PROFILE"), work, arm, None, None, None,
    )

    assert prompt.startswith("PROFILE\n\nARM START\nARM IMPORT")
    assert prompt.endswith("ARM END\n\nHEADER")
    assert "HOST SECRET" not in prompt
    assert 'status="outside_allowed_dirs"' in prompt
    assert provenance["system_prompt_sha256"] == hashlib.sha256(
        prompt.encode()
    ).hexdigest()[:16]
    assert metadata.prompt_import_tree[0]["owner"] == "system_prompt"
    assert [
        node["status"] for node in metadata.prompt_import_tree[0]["imports"]
    ] == ["loaded", "outside_allowed_dirs"]
    assert str(tmp_path) not in json.dumps(metadata.trace_fields())

    disabled = make_config(system_header="HEADER", imports_enabled=False)
    literal, _provenance, _contract, literal_metadata = (
        load_system_prompt_and_provenance(
            disabled, _client(), work, arm, None, None, None,
        )
    )
    assert f"@{work / 'shared.md'}" in literal
    assert "@../secret.md" in literal
    assert literal_metadata.prompt_import_tree[0]["imports"] == []


def test_project_imports_expand_before_utf8_cap_and_count_bytes(tmp_path: Path) -> None:
    work = tmp_path / "repo"
    work.mkdir()
    (work / ".git").mkdir()
    (work / "AGENTS.md").write_text("@shared.md\n")
    shared = work / "shared.md"
    shared.write_text("ééééé")
    cfg = make_config(
        system_header="HEADER",
        project_docs_enabled=True,
        project_doc_global_dir="",
        project_doc_max_bytes=7,
    )

    prompt, _provenance, _contract, metadata = (
        load_system_prompt_and_provenance(
            cfg, _client(), work, None, None, None, None,
        )
    )

    assert "ééé" in prompt
    assert "éééé" not in prompt
    assert metadata.project_instruction_bytes == len("@shared.md\n")
    assert metadata.project_instruction_imported_bytes == len(shared.read_bytes())
    assert metadata.project_instruction_resolved_bytes == 6
    assert metadata.project_instructions_truncated is True
    project_source = metadata.prompt_import_tree[0]
    assert project_source["source"] == "AGENTS.md"
    assert project_source["imports"][0]["path"] == "shared.md"


def test_global_document_import_cannot_escape_global_root(tmp_path: Path) -> None:
    work = tmp_path / "repo"
    global_dir = tmp_path / "global"
    work.mkdir()
    global_dir.mkdir()
    (work / ".git").mkdir()
    (global_dir / "AGENTS.md").write_text("@../secret.md\n")
    (tmp_path / "secret.md").write_text("GLOBAL ESCAPE")
    cfg = make_config(
        system_header="HEADER",
        project_docs_enabled=True,
        project_doc_global_dir=str(global_dir),
    )

    prompt, _provenance, _contract, metadata = (
        load_system_prompt_and_provenance(
            cfg, _client(), work, None, None, None, None,
        )
    )

    assert "GLOBAL ESCAPE" not in prompt
    assert 'status="outside_allowed_dirs"' in prompt
    assert metadata.prompt_import_tree[0]["source"] == "global/AGENTS.md"
    assert str(tmp_path) not in json.dumps(metadata.trace_fields())


def test_injection_is_parsed_after_expansion_and_disabled_mode_is_literal(
    tmp_path: Path,
) -> None:
    work = tmp_path / "repo"
    injection_dir = work / ".harness" / "injections"
    injection_dir.mkdir(parents=True)
    shared = work / "shared.md"
    shared.write_text("SHARED BODY")
    source = injection_dir / "a.md"
    source.write_text(
        '+++\nname = "hint"\ntrigger = "always"\n+++\n'
        "@../../shared.md\n"
    )

    enabled = load_injections_with_metadata(
        injection_dir,
        imports_enabled=True,
        allowed_dirs=(work,),
    )
    assert enabled.injections[0].body == "SHARED BODY"
    assert enabled.prompt_import_tree[0]["source"] == ".harness/injections/a.md"
    assert enabled.prompt_import_tree[0]["imported_bytes"] == len(
        shared.read_bytes()
    )

    disabled = load_injections_with_metadata(
        injection_dir,
        imports_enabled=False,
        allowed_dirs=(work,),
    )
    assert disabled.injections[0].body == "@../../shared.md"
    assert disabled.prompt_import_tree[0]["imports"] == []


def test_session_start_aggregates_safe_import_tree_without_state_projection(
    tmp_path: Path,
) -> None:
    from scripts.llm_solver._shared.telemetry_paths import trace_path
    from scripts.llm_solver.harness.loop import solve_task

    work = tmp_path / "task"
    injection_dir = work / ".harness" / "injections"
    injection_dir.mkdir(parents=True)
    (work / ".git").mkdir()
    (work / "prompt.txt").write_text("finish")
    (work / "arm-part.md").write_text("ARM PART")
    (work / "project-part.md").write_text("PROJECT PART")
    (work / "injection-part.md").write_text("INJECTION PART")
    arm = work / "arm.md"
    arm.write_text("@arm-part.md\n")
    (work / "AGENTS.md").write_text("@project-part.md\n")
    (injection_dir / "hint.md").write_text(
        '+++\nname = "hint"\ntrigger = "always"\n+++\n'
        "@../../injection-part.md\n"
    )
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
        injections_enabled=True,
        turn_snapshots_enabled=False,
    )

    with patch("scripts.llm_solver.harness.loop._auto_commit"):
        assert solve_task(work, cfg, client, system_prompt_file=arm) is True

    events = [
        json.loads(line)
        for line in trace_path(work).read_text().splitlines()
        if line.strip()
    ]
    start = next(event for event in events if event["event"] == "session_start")
    assert [entry["owner"] for entry in start["prompt_import_tree"]] == [
        "system_prompt",
        "project_instruction",
        "injection",
    ]
    assert all(entry["imports"] for entry in start["prompt_import_tree"])
    assert str(tmp_path) not in json.dumps(start["prompt_import_tree"])
    state = json.loads((work / ".solver" / "state.json").read_text())
    assert "prompt_import_tree" not in json.dumps(state)
