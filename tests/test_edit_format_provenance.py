"""Acceptance coverage for CLI override and run-artifact provenance."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _config_helpers import make_config
from llm_solver._shared.telemetry_paths import trace_path
from llm_solver.harness._loop.trace_schema import TRACE_EVENT_REQUIRED_FIELDS
from llm_solver.harness.action_metadata import action_metadata
from llm_solver.harness.loop import solve_task
from llm_solver.harness.state_writer import project
from llm_solver.server.types import TurnResult, Usage
from scripts.llm_assist.__main__ import main as assist_main
from scripts.llm_assist.store import SessionStore
from scripts.llm_solver.__main__ import (
    _model_log_tag,
    main as measurement_main,
)


def test_measurement_log_tag_keeps_exact_model_paths_out_of_filenames() -> None:
    exact_model = "/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"

    tag = _model_log_tag(exact_model)

    assert "/" not in tag
    assert tag.startswith("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf-")
    assert len(tag) <= 77
    assert _model_log_tag("qwen3.6-35b-a3b") == "qwen3.6-35b-a3b"


def test_session_start_and_state_project_effective_profile_format(
    tmp_path: Path,
) -> None:
    (tmp_path / "prompt.txt").write_text("Fix it.")
    cfg = make_config(max_turns=1, max_sessions=1)
    client = MagicMock()
    client.__dict__["profile"] = SimpleNamespace(
        name="fixture",
        edit_format="whole",
        max_tools=99,
        simplify_schemas=False,
    )
    client.chat.return_value = TurnResult(
        content="Done.",
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(prompt_tokens=10, completion_tokens=2),
    )
    client.build_assistant_message.return_value = {
        "role": "assistant",
        "content": "Done.",
    }

    with (
        patch("llm_solver.harness.loop._auto_commit"),
        patch("llm_solver.harness.loop.Session._get_server_ctx", return_value=8192),
    ):
        assert solve_task(tmp_path, cfg, client) is True

    events = [
        json.loads(line)
        for line in trace_path(tmp_path).read_text().splitlines()
        if line.strip()
    ]
    start = next(event for event in events if event["event"] == "session_start")
    assert start["edit_format"] == "whole"
    state = json.loads((tmp_path / ".solver" / "state.json").read_text())
    assert state["meta"]["edit_format"] == "whole"
    assert "edit_format" in TRACE_EVENT_REQUIRED_FIELDS["session_start"]


def test_udiff_operation_projects_canonical_source_path() -> None:
    patch_text = (
        "--- a/src/app.py\n+++ b/src/app.py\n"
        "@@ -1 +1 @@\n-old\n+new"
    )
    metadata = action_metadata("udiff", {"patch": patch_text})
    state = project([
        {
            "event": "session_start",
            "session_number": 1,
            "edit_format": "udiff",
        },
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 0,
            "tool_name": "udiff",
            "args_summary": "patch=<unified diff>",
            "result_summary": "OK: applied unified diff",
            **metadata,
        },
    ], max_result_chars=2000, imperative_projection=True)

    assert metadata["source_write_paths"] == ["src/app.py"]
    assert state["meta"]["edit_format"] == "udiff"
    assert state["trace"][0]["source_write_paths"] == ["src/app.py"]
    assert state["process"]["phase"] == "post_mutation_unverified"


def test_measurement_cli_maps_edit_format_override(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    cfg = make_config(runtime_mode="measurement", tools_edit_format="udiff")

    with (
        patch("scripts.llm_solver.__main__.load_config", return_value=cfg) as load,
        patch(
            "scripts.llm_solver.__main__.load_profile",
            return_value=SimpleNamespace(name="test", inherits="_base"),
        ),
        patch("scripts.llm_solver.__main__._build_run_metadata", return_value={}),
        patch("scripts.llm_solver.__main__._write_session_json"),
    ):
        assert measurement_main([
            str(run_dir),
            "--task", str(task_dir),
            "--dry-run",
            "--edit-format", "udiff",
        ]) == 0

    assert load.call_args.kwargs["overrides"]["tools_edit_format"] == "udiff"


def test_installed_cli_persists_edit_format_override(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "assist-home")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    seen_config_paths: list[str] = []

    def _run(store_obj, record, *, resume):
        seen_config_paths.extend(record.config_paths)
        record.artifact_path.mkdir(parents=True, exist_ok=True)
        (record.artifact_path / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "stop",
                "turns": 1,
            }) + "\n"
        )
        store_obj.update_session(
            record.session_id, status="completed", last_finish_reason="stop"
        )
        return True, "stop"

    with (
        patch("scripts.llm_assist.__main__.SessionStore", return_value=store),
        patch(
            "scripts.llm_assist.__main__.resolve_served_model",
            return_value=("served", ["served"]),
        ) as resolve,
        patch("scripts.llm_assist.__main__.run_session", side_effect=_run),
    ):
        assert assist_main([
            "run",
            "--cwd", str(work_dir),
            "--prompt-text", "Do it.",
            "--edit-format", "whole",
        ]) == 0

    assert resolve.call_args.kwargs["config_overrides"] == {
        "tools_edit_format": "whole"
    }
    overlay = Path(seen_config_paths[-1])
    assert overlay.name == "provider.toml"
    assert overlay.read_text() == '[tools]\nedit_format = "whole"\n'
