"""Runtime acceptance tests for per-request thinking-level control."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _config_helpers import make_config
from llm_solver._shared.telemetry_paths import trace_path
from llm_solver.config import load_config
from llm_solver.harness.loop import solve_task
from llm_solver.server.client import LlamaClient
from llm_solver.server.profile_loader import load_profile
from llm_solver.server.request_controls import resolve_thinking_level
from llm_solver.server.security import validate_profile
from llm_solver.server.types import TurnResult, Usage
from scripts.llm_assist.__main__ import main as assist_main
from scripts.llm_assist.store import SessionStore
from scripts.llm_solver.__main__ import main as measurement_main


def _response():
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Done.", tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=5),
        model_dump_json=lambda: "{}",
    )


def test_profile_mapped_thinking_is_applied_per_request_and_side_stays_off(
    caplog: pytest.LogCaptureFixture,
) -> None:
    profile = load_profile("_base", PROJECT_ROOT / "profiles")
    cfg = make_config(thinking_level="high")

    with caplog.at_level(logging.WARNING):
        client = LlamaClient(cfg, profile=profile)
    captured: list[dict] = []
    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **payload: captured.append(payload) or _response()
            )
        )
    )

    client.chat([{"role": "user", "content": "work"}], [])
    client.complete_side_request(
        {
            "model": cfg.model,
            "messages": [{"role": "user", "content": "summarize"}],
            "max_tokens": 100,
        }
    )

    assert client.thinking_resolution.requested_level == "high"
    assert client.thinking_resolution.effective_level == "on"
    assert client.thinking_resolution.clamped is True
    assert "thinking level high is unsupported" in caplog.text
    assert captured[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True},
        "cache_prompt": False,
    }
    assert captured[1]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_prompt": False,
    }


def test_profile_reasoning_levels_inherit_and_validate_at_load(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles"
    base = profiles / "_base"
    child = profiles / "child"
    base.mkdir(parents=True)
    child.mkdir()
    (base / "profile.toml").write_text(
        """\
[profile]
name = "_base"
inherits = ""
[reasoning_levels.off]
enable_thinking = false
[reasoning_levels.on]
enable_thinking = true
"""
    )
    (child / "profile.toml").write_text(
        """\
[profile]
name = "child"
inherits = "_base"
[reasoning_levels.on]
reasoning_effort = "high"
"""
    )

    profile = load_profile("child", profiles)

    assert profile.reasoning_levels == {
        "off": {"enable_thinking": False},
        "on": {"reasoning_effort": "high"},
    }

    bad = profiles / "bad"
    bad.mkdir()
    (bad / "profile.toml").write_text(
        """\
[profile]
name = "bad"
inherits = "_base"
[reasoning_levels.turbo]
reasoning_effort = "turbo"
"""
    )
    violations = validate_profile(bad)
    assert len(violations) == 1
    assert "unknown profile reasoning level" in violations[0]


def test_effective_thinking_level_is_in_session_trace_and_provenance(
    tmp_path: Path,
) -> None:
    (tmp_path / "prompt.txt").write_text("Fix it.")
    cfg = make_config(
        thinking_level="xhigh",
        max_turns=1,
        max_sessions=1,
        context_size=16_000,
    )
    resolution = resolve_thinking_level(
        "xhigh",
        {
            "off": {"enable_thinking": False},
            "on": {"enable_thinking": True},
        },
    )
    client = MagicMock()
    client.__dict__["_thinking_resolution"] = resolution
    client.chat.return_value = TurnResult(
        content="Done.",
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(prompt_tokens=100, completion_tokens=5),
    )
    client.build_assistant_message.return_value = {
        "role": "assistant",
        "content": "Done.",
    }

    with (
        patch("llm_solver.harness.loop._auto_commit"),
        patch("llm_solver.harness.loop.Session._get_server_ctx", return_value=16_000),
    ):
        assert solve_task(tmp_path, cfg, client) is True

    events = [
        json.loads(line)
        for line in trace_path(tmp_path).read_text().splitlines()
        if line.strip()
    ]
    start = next(event for event in events if event["event"] == "session_start")
    assert start["thinking_level"] == "on"
    assert start["thinking_level_requested"] == "xhigh"

    provenance = json.loads((tmp_path / "metrics.json").read_text())["provenance"]
    assert provenance["thinking_level_requested"] == "xhigh"
    assert provenance["thinking_level_effective"] == "on"
    assert provenance["thinking_level_clamped"] is True


def test_thinking_config_default_and_invalid_value(tmp_path: Path) -> None:
    assert load_config().thinking_level == "off"
    overlay = tmp_path / "bad-thinking.toml"
    overlay.write_text('[model]\nthinking_level = "turbo"\n')
    with pytest.raises(ValueError, match="thinking_level"):
        load_config(overlay)


def test_measurement_cli_maps_thinking_override(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    cfg = make_config(runtime_mode="measurement", thinking_level="high")

    with (
        patch("scripts.llm_solver.__main__.load_config", return_value=cfg) as load,
        patch(
            "scripts.llm_solver.__main__.load_profile",
            return_value=SimpleNamespace(name="test", inherits="_base"),
        ),
        patch("scripts.llm_solver.__main__._build_run_metadata", return_value={}),
        patch("scripts.llm_solver.__main__._write_session_json"),
    ):
        assert measurement_main(
            [
                str(run_dir),
                "--task", str(task_dir),
                "--dry-run",
                "--thinking", "high",
            ]
        ) == 0

    assert load.call_args.kwargs["overrides"]["thinking_level"] == "high"


def test_replay_refuses_treatment_changing_thinking_override(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as exc:
        measurement_main([
            str(tmp_path / "run"),
            "--replay-from", str(tmp_path / "source"),
            "--thinking", "high",
        ])
    assert exc.value.code == 2


def test_installed_cli_persists_thinking_in_session_overlay(
    tmp_path: Path,
) -> None:
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
        patch("scripts.llm_assist.__main__.preflight_assistant_startup"),
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
            "--thinking", "high",
        ]) == 0

    assert resolve.call_args.kwargs["config_overrides"] == {
        "thinking_level": "high"
    }
    overlay = Path(seen_config_paths[-1])
    assert overlay.name == "provider.toml"
    assert overlay.read_text() == '[model]\nthinking_level = "high"\n'
