"""Runtime acceptance tests for llama-server prompt-cache controls."""
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
from scripts.llm_assist.__main__ import _print_run_compact_summary
from scripts.llm_assist.runner import run_session, session_compact_summary
from scripts.llm_assist.store import SessionStore
from llm_solver._shared.telemetry_paths import trace_path
from llm_solver.config import load_config
from llm_solver.harness.loop import solve_task
from llm_solver.harness.state_writer import project
from llm_solver.server._streaming import assemble_stream
from llm_solver.server.client import LlamaClient
from llm_solver.server.request_controls import (
    derive_cache_slot,
    extract_cache_observation,
)
from llm_solver.server.types import SideRequestResult, ToolCall, TurnResult, Usage
from scripts.llm_solver._main_helpers import _build_run_metadata


def _response(*, prompt: int, cached: int, completion: int = 5):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Done.", tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        ),
        model_dump_json=lambda: "{}",
    )


def test_llama_client_applies_affinity_retention_and_usage_from_fake_transport() -> None:
    cfg = make_config(
        cache_affinity=8,
        cache_retention="session",
        server_request_extra={"seed": 7},
    )
    client = LlamaClient(cfg, profile=None)
    client.set_session_id("stable-session")
    captured: list[dict] = []
    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **payload: captured.append(payload)
                or _response(prompt=1_000, cached=750)
            )
        )
    )

    result = client.chat(
        [{"role": "system", "content": "system"}],
        [],
    )

    assert captured[0]["extra_body"] == {
        "seed": 7,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_prompt": True,
        "id_slot": derive_cache_slot("stable-session", 8),
    }
    assert result.usage == Usage(
        prompt_tokens=1_000,
        completion_tokens=5,
        cached_tokens=750,
        cache_hit_ratio=0.75,
    )


def test_side_request_forces_cache_false_and_removes_slot() -> None:
    cfg = make_config(
        cache_affinity=4,
        cache_retention="session",
        server_request_extra={
            "seed": 9,
            "cache_prompt": True,
            "id_slot": 3,
        },
    )
    client = LlamaClient(cfg, profile=None)
    client.set_session_id("stable-session")
    captured: list[dict] = []
    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **payload: captured.append(payload)
                or _response(prompt=100, cached=0)
            )
        )
    )

    result = client.complete_side_request(
        {
            "model": cfg.model,
            "messages": [{"role": "user", "content": "summarize"}],
            "max_tokens": 100,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
                "cache_prompt": True,
                "id_slot": 2,
            },
        }
    )

    assert isinstance(result, SideRequestResult)
    assert captured[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "seed": 9,
        "cache_prompt": False,
    }


def test_stream_assembly_preserves_cache_usage_and_llama_timings() -> None:
    chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                delta=SimpleNamespace(content="Done.", tool_calls=[]),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=1_000,
            completion_tokens=10,
            prompt_tokens_details=SimpleNamespace(cached_tokens=800),
        ),
        model_extra={"timings": {"prompt_n": 200, "cache_n": 800}},
    )

    response = assemble_stream([chunk])
    observation = extract_cache_observation(response)

    assert observation is not None
    assert observation.prompt_tokens == 1_000
    assert observation.cached_tokens == 800
    assert observation.hit_ratio == 0.8
    assert observation.source == "usage+timings"
    dumped = json.loads(response.model_dump_json())
    assert dumped["usage"]["prompt_tokens_details"]["cached_tokens"] == 800
    assert dumped["timings"] == {"prompt_n": 200, "cache_n": 800}


def test_turn_trace_metrics_warning_and_state_boundary(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    (tmp_path / "prompt.txt").write_text("Fix it.")
    cfg = make_config(
        max_turns=2,
        max_sessions=1,
        context_size=16_000,
        cache_miss_warn_ratio=0.8,
    )
    client = MagicMock()
    client.chat.side_effect = [
        TurnResult(
            content="Inspect.",
            tool_calls=[
                ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
            ],
            finish_reason="tool_calls",
            usage=Usage(
                prompt_tokens=100,
                completion_tokens=5,
                cached_tokens=0,
                cache_hit_ratio=0.0,
            ),
        ),
        TurnResult(
            content="Done.",
            tool_calls=[],
            finish_reason="stop",
            usage=Usage(
                prompt_tokens=200,
                completion_tokens=5,
                cached_tokens=100,
                cache_hit_ratio=0.5,
            ),
        ),
    ]
    client.build_assistant_message.return_value = {
        "role": "assistant",
        "content": "Inspect.",
    }

    with (
        caplog.at_level(logging.WARNING),
        patch("llm_solver.harness.loop._auto_commit"),
        patch("llm_solver.harness.loop.dispatch", return_value="README"),
        patch("llm_solver.harness.loop.Session._get_server_ctx", return_value=16_000),
    ):
        assert solve_task(tmp_path, cfg, client) is True

    turns = [
        json.loads(line)
        for line in trace_path(tmp_path).read_text().splitlines()
        if '"event":"turn"' in line
    ]
    assert len(turns) == 2
    expected_first = {
        "session_number": 1,
        "turn_number": 0,
        "role": "main",
        "prompt_tokens": 100,
        "cached_tokens": 0,
        "cache_hit_ratio": 0.0,
    }
    assert {key: turns[0][key] for key in expected_first} == expected_first
    assert turns[1]["cached_tokens"] == 100
    assert turns[1]["cache_hit_ratio"] == 0.5
    assert caplog.text.count("prompt cache hit ratio") == 1

    metrics = json.loads((tmp_path / "metrics.json").read_text())["metrics"]
    assert metrics["prompt_cache"] == {
        "prompt_tokens": 300,
        "cached_tokens": 100,
        "cache_hit_ratio": 0.333333,
        "requests_observed": 2,
        "requests_unobserved": 0,
    }
    state_text = (tmp_path / ".solver" / "state.json").read_text()
    assert "cache_hit_ratio" not in state_text
    assert '"event": "turn"' not in state_text


def test_turn_cache_event_does_not_change_model_facing_state_projection() -> None:
    start = {"event": "session_start", "session_number": 1}
    turn = {
        "event": "turn",
        "session_number": 1,
        "turn_number": 0,
        "role": "main",
        "prompt_tokens": 100,
        "cached_tokens": 80,
        "cache_hit_ratio": 0.8,
    }

    before = project([start], max_result_chars=20_000)
    after = project([start, turn], max_result_chars=20_000)

    for section in ("state", "trace", "gates", "evidence", "inference"):
        assert after[section] == before[section]


def test_config_defaults_and_invalid_cache_controls(tmp_path: Path) -> None:
    cfg = load_config()
    assert cfg.server_request_extra == {}
    assert cfg.cache_affinity is False
    assert cfg.cache_retention == "off"
    assert cfg.cache_miss_warn_ratio == 0.0

    overlay = tmp_path / "bad-cache.toml"
    overlay.write_text("[server]\ncache_miss_warn_ratio = 1.1\n")
    with pytest.raises(ValueError, match="cache_miss_warn_ratio"):
        load_config(overlay)


def test_installed_runner_binds_client_to_session_record_id(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "assist")
    record = store.create_session(
        cwd=tmp_path / "work",
        model="test-model",
        prompt_text="Fix it.",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    cfg = make_config(runtime_mode="assistant", max_sessions=1)
    client = MagicMock()

    with (
        patch("scripts.llm_assist.runner.load_config", return_value=cfg),
        patch("scripts.llm_assist.runner._load_profile", return_value=None),
        patch("scripts.llm_assist.runner._make_client", return_value=client),
        patch("scripts.llm_assist.runner._apply_effective_context", return_value=cfg),
        patch("scripts.llm_assist.runner.solve_task", return_value=False),
        patch("scripts.llm_assist.runner.last_finish_reason", return_value="max_turns"),
    ):
        run_session(store, record, resume=False)

    client.set_session_id.assert_called_once_with(record.session_id)


def test_measurement_metadata_persists_stable_session_id(tmp_path: Path) -> None:
    cfg = make_config()
    args = SimpleNamespace(
        config=[],
        context="halflife",
        system_prompt=None,
    )
    kwargs = {
        "run_dir": tmp_path,
        "cfg": cfg,
        "args": args,
        "overrides": {},
        "started_at": "2026-08-23T12:00:00+00:00",
    }

    first = _build_run_metadata(**kwargs)
    second = _build_run_metadata(**kwargs)

    assert first["session_id"] == second["session_id"]
    assert len(first["session_id"]) == 32


def test_operator_summary_reports_latest_cache_ratio(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / ".trace.jsonl").write_text("")
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "prompt_cache": {
                        "prompt_tokens": 1_000,
                        "cached_tokens": 750,
                        "cache_hit_ratio": 0.75,
                        "requests_observed": 2,
                        "requests_unobserved": 0,
                    }
                }
            }
        )
    )

    summary = session_compact_summary(tmp_path)
    assert summary["cache_hit_ratio"] == 0.75
    record = SimpleNamespace(artifact_path=tmp_path)
    _print_run_compact_summary(record)
    assert "cache_hit_ratio: 75.0%" in capsys.readouterr().out
