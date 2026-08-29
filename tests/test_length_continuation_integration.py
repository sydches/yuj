"""Runtime/config/profile acceptance coverage for length continuation."""
from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import openai
import pytest

from _config_helpers import make_config
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._loop.chat_io import _aggregate_usage
from scripts.llm_solver.harness._loop.trace_schema import (
    TRACE_EVENT_REQUIRED_FIELDS,
)
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.server.client import LlamaClient
from scripts.llm_solver.server.profile_loader import load_profile
from scripts.llm_solver.server.security import validate_profile
from scripts.llm_solver.server.types import Usage


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILES = PROJECT_ROOT / "profiles"


def _response(
    content: str | None,
    reason: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    return {
        "choices": [{
            "message": {"content": content, "tool_calls": []},
            "finish_reason": reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def _profile(*, supports_prefill: bool, normalize=None):
    base = load_profile("_base", PROFILES)
    modules = [] if normalize is None else [normalize]
    return replace(
        base,
        supports_prefill=supports_prefill,
        _normalize_rules=[],
        _normalize_modules=modules,
    )


def _client(cfg, *, supports_prefill: bool, normalize=None) -> LlamaClient:
    return LlamaClient(
        cfg,
        profile=_profile(
            supports_prefill=supports_prefill,
            normalize=normalize,
        ),
    )


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_length_continue_config_default_overlay_and_rejection(
    tmp_path: Path,
) -> None:
    assert load_config().length_continue_max == 0
    overlay = tmp_path / "length.toml"
    overlay.write_text("[loop]\nlength_continue_max = 3\n")
    assert load_config(user_config=overlay).length_continue_max == 3

    for invalid in ("-1", "true", '"2"', "1.5"):
        overlay.write_text(
            f"[loop]\nlength_continue_max = {invalid}\n"
        )
        with pytest.raises(ValueError, match="loop.length_continue_max"):
            load_config(user_config=overlay)


def test_usage_aggregation_preserves_unknown_cache_counts() -> None:
    unknown = _aggregate_usage([
        Usage(10, 2, cached_tokens=4, cache_hit_ratio=0.4),
        Usage(20, 3),
    ])
    assert unknown == Usage(
        prompt_tokens=30,
        completion_tokens=5,
        cached_tokens=None,
        cache_hit_ratio=None,
    )

    observed = _aggregate_usage([
        Usage(10, 2, cached_tokens=4, cache_hit_ratio=0.4),
        Usage(20, 3, cached_tokens=6, cache_hit_ratio=0.3),
    ])
    assert observed.prompt_tokens == 30
    assert observed.completion_tokens == 5
    assert observed.cached_tokens == 10
    assert observed.cache_hit_ratio == pytest.approx(1 / 3)
    assert {
        "session_number", "turn_number", "attempt", "tokens",
    } <= TRACE_EVENT_REQUIRED_FIELDS["length_continue"]


def test_profile_capability_inheritance_override_and_validation(
    tmp_path: Path,
) -> None:
    assert load_profile("_base", PROFILES).supports_prefill is False
    assert load_profile(
        "qwen3.6-35b-a3b", PROFILES
    ).supports_prefill is True
    assert load_profile("qwen38-27b", PROFILES).supports_prefill is True

    shutil.copytree(PROFILES / "_base", tmp_path / "_base")
    child = tmp_path / "child"
    child.mkdir()
    (child / "profile.toml").write_text(
        """
[profile]
format_version = 1
name = "child"
inherits = "_base"

[model]
supports_prefill = true
""".strip()
        + "\n"
    )
    assert load_profile("child", tmp_path).supports_prefill is True

    (child / "profile.toml").write_text(
        """
[profile]
format_version = 1
name = "child"
inherits = "_base"

[model]
supports_prefill = "yes"
""".strip()
        + "\n"
    )
    violations = validate_profile(child)
    assert len(violations) == 1
    assert "supports_prefill must be a boolean" in violations[0]


@pytest.mark.parametrize(
    ("limit", "supported"),
    [(0, True), (2, False)],
)
def test_off_or_unsupported_keeps_one_call_and_length_fallback(
    tmp_path: Path,
    limit: int,
    supported: bool,
) -> None:
    cfg = make_config(
        max_turns=2,
        length_continue_max=limit,
        sandbox_bash=False,
    )
    normalize_calls = []

    def normalize(response):
        normalize_calls.append(dict(response))
        return response

    client = _client(
        cfg, supports_prefill=supported, normalize=normalize
    )
    client._call_api = MagicMock(
        return_value=_response(
            "truncated", "length", prompt_tokens=10, completion_tokens=4
        )
    )
    trace = tmp_path / ".trace.jsonl"
    with open(trace, "a") as trace_file, patch.object(
        Session, "_get_server_ctx", return_value=cfg.context_size
    ):
        session = Session(
            cfg,
            client,
            "system",
            "task",
            str(tmp_path),
            trace_file=trace_file,
            trace_path=trace,
            session_number=1,
        )
        result = session.run()

    assert result.finish_reason == "length"
    assert client._call_api.call_count == 1
    initial_request = client._call_api.call_args.args[0]
    assert "continue_final_message" not in initial_request["extra_body"]
    assert "add_generation_prompt" not in initial_request["extra_body"]
    assert len(normalize_calls) == 1
    assert session._length_continuation_count == 0
    assert not any(
        event["event"] == "length_continue" for event in _events(trace)
    )


def test_split_thinking_tool_call_joins_once_in_the_real_session(
    tmp_path: Path,
) -> None:
    cfg = make_config(
        max_turns=1,
        length_continue_max=2,
        parallel_readonly_enabled=False,
        sandbox_bash=False,
    )
    normalized_inputs = []

    def normalize(response):
        text = str(response.get("content") or "")
        normalized_inputs.append(text)
        payload = json.loads(text.split("</think>", 1)[1])
        response["content"] = None
        response["tool_calls"] = [{
            "id": "",
            "type": "function",
            "function": {
                "name": payload["name"],
                "arguments": json.dumps(payload["arguments"]),
            },
        }]
        response["finish_reason"] = "tool_calls"
        return response

    client = _client(cfg, supports_prefill=True, normalize=normalize)
    client._call_api = MagicMock(side_effect=[
        _response(
            "<think>inspect</thi",
            "length",
            prompt_tokens=10,
            completion_tokens=2,
        ),
        _response(
            'nk>{"name":"read","arguments":{"path":"docs/conf',
            "length",
            prompt_tokens=12,
            completion_tokens=3,
        ),
        _response(
            'configuration.md"}}',
            "length",
            prompt_tokens=15,
            completion_tokens=4,
        ),
    ])
    trace = tmp_path / ".trace.jsonl"
    with (
        open(trace, "a") as trace_file,
        patch.object(Session, "_get_server_ctx", return_value=cfg.context_size),
        patch("scripts.llm_solver.harness.loop.dispatch", return_value="read"),
    ):
        session = Session(
            cfg,
            client,
            "system",
            "task",
            str(tmp_path),
            trace_file=trace_file,
            trace_path=trace,
            session_number=2,
        )
        result = session.run()

    assert result.finish_reason == "max_turns"
    assert result.total_prompt_tokens == 37
    assert result.total_completion_tokens == 9
    assert len(normalized_inputs) == 1
    assert normalized_inputs[0].endswith(
        '{"name":"read","arguments":{"path":"docs/configuration.md"}}'
    )
    requests = [call.args[0] for call in client._call_api.call_args_list]
    assert requests[1]["messages"][:-1] == requests[0]["messages"]
    assert requests[1]["messages"][-1]["content"] == "<think>inspect</thi"
    assert requests[2]["messages"][:-1] == requests[0]["messages"]
    assert requests[2]["messages"][-1]["content"].endswith("docs/conf")
    for request in requests[1:]:
        assert request["extra_body"]["continue_final_message"] is True
        assert request["extra_body"]["add_generation_prompt"] is False

    assistants = [
        message for message in session.context.get_messages()
        if message.get("role") == "assistant"
    ]
    assert len(assistants) == 1
    assert assistants[0]["content"] is None
    assert json.loads(
        assistants[0]["tool_calls"][0]["function"]["arguments"]
    ) == {"path": "docs/configuration.md"}

    continuation_events = [
        event for event in _events(trace)
        if event["event"] == "length_continue"
    ]
    assert [event["attempt"] for event in continuation_events] == [1, 2]
    assert [event["tokens"] for event in continuation_events] == [3, 4]
    assert all("content" not in event for event in continuation_events)
    assert all("messages" not in event for event in continuation_events)
    assert session._length_continuation_count == 2


def test_exhausted_attempts_feed_existing_metrics_and_length_rollover(
    tmp_path: Path,
) -> None:
    from scripts.llm_solver._shared.telemetry_paths import trace_path
    from scripts.llm_solver.harness.loop import solve_task

    (tmp_path / "prompt.txt").write_text("Finish the task")
    cfg = make_config(
        max_sessions=1,
        max_turns=2,
        length_continue_max=1,
        sandbox_bash=False,
        state_writer_enabled=True,
    )
    client = _client(cfg, supports_prefill=True, normalize=lambda value: value)
    client._call_api = MagicMock(side_effect=[
        _response(
            "one ", "length", prompt_tokens=20, completion_tokens=2
        ),
        _response(
            "two", "length", prompt_tokens=25, completion_tokens=3
        ),
    ])

    with (
        patch("scripts.llm_solver.harness.loop._auto_commit"),
        patch.object(Session, "_get_server_ctx", return_value=cfg.context_size),
    ):
        assert solve_task(tmp_path, cfg, client) is False

    metrics = json.loads((tmp_path / "metrics.json").read_text())["metrics"]
    assert metrics["length_continuations"] == 1
    assert metrics["total_prompt_tokens"] == 45
    assert metrics["total_completion_tokens"] == 5
    events = _events(trace_path(tmp_path))
    continuation = next(
        event for event in events if event["event"] == "length_continue"
    )
    assert continuation["tokens"] == 3
    assert not ({"content", "messages", "request"} & set(continuation))
    end = next(event for event in events if event["event"] == "session_end")
    assert end["finish_reason"] == "length"
    exit_event = next(
        event for event in events if event["event"] == "session_exit"
    )
    assert exit_event["kind"] == "truncated"
    assert exit_event["reason"] == "length"
    state = json.loads((tmp_path / ".solver" / "state.json").read_text())
    assert "length_continue" not in json.dumps(state)


def test_continuation_error_rebuilds_request_after_fallback(
    tmp_path: Path,
) -> None:
    cfg = make_config(
        max_transient_retries=0,
        length_continue_max=1,
        sandbox_bash=False,
    )
    primary = _client(cfg, supports_prefill=True, normalize=lambda value: value)
    primary._call_api = MagicMock(side_effect=[
        _response(
            "partial", "length", prompt_tokens=10, completion_tokens=2
        ),
        openai.APIConnectionError(request=MagicMock()),
    ])
    replacement_cfg = replace(cfg, model="replacement-model")
    replacement = _client(
        replacement_cfg,
        supports_prefill=True,
        normalize=lambda value: value,
    )
    replacement._call_api = MagicMock(return_value=_response(
        "replacement complete",
        "stop",
        prompt_tokens=8,
        completion_tokens=2,
    ))
    session = Session(
        cfg, primary, "system", "task", str(tmp_path), session_number=1
    )

    def activate(active_session, _turn, *, reason):
        assert reason == "transient_exhausted"
        active_session.client = replacement
        active_session.cfg = replacement_cfg
        return True

    with patch(
        "scripts.llm_solver.harness._loop.chat_io.activate_next_fallback",
        side_effect=activate,
    ):
        result = session._chat_with_retry(0)

    assert result.content == "replacement complete"
    assert primary._call_api.call_count == 2
    replacement_request = replacement._call_api.call_args.args[0]
    assert replacement_request["model"] == "replacement-model"
    assert replacement_request["messages"][-1]["role"] == "user"
    assert "continue_final_message" not in replacement_request["extra_body"]


def test_streamed_raw_pieces_use_the_same_normalize_once_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def chunk(*, content=None, finish_reason=None, usage=None):
        delta = SimpleNamespace(
            content=content,
            tool_calls=None,
            role=None,
            function_call=None,
            refusal=None,
        )
        choice = SimpleNamespace(
            delta=delta, finish_reason=finish_reason, index=0
        )
        return SimpleNamespace(choices=[choice], usage=usage)

    def usage(prompt_tokens, completion_tokens):
        return SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    monkeypatch.setenv("YUJ_STREAMING", "1")
    cfg = make_config(length_continue_max=1, sandbox_bash=False)
    seen = []

    def normalize(response):
        seen.append(response["content"])
        return response

    client = _client(cfg, supports_prefill=True, normalize=normalize)
    streams = [
        iter([
            chunk(content="partial "),
            chunk(
                finish_reason="length", usage=usage(10, 2)
            ),
        ]),
        iter([
            chunk(content="tail"),
            chunk(finish_reason="stop", usage=usage(12, 1)),
        ]),
    ]
    payloads = []

    def create(**payload):
        payloads.append(payload)
        return streams.pop(0)

    with patch.object(
        client.client.chat.completions, "create", side_effect=create
    ):
        session = Session(
            cfg, client, "system", "task", str(tmp_path), session_number=1
        )
        result = session._chat_with_retry(0)

    assert result.content == "partial tail"
    assert result.usage.prompt_tokens == 22
    assert result.usage.completion_tokens == 3
    assert seen == ["partial tail"]
    assert len(payloads) == 2
    assert payloads[1]["extra_body"]["continue_final_message"] is True
    assert payloads[1]["stream"] is True
