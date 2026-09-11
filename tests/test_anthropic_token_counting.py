"""Native Messages budgeting uses the converted request and provider precision."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import requests

from _config_helpers import make_config
from scripts.llm_assist._anthropic import AnthropicClient
from scripts.llm_solver.harness.request_counting import resolve_counter
from scripts.llm_solver.server.profile_loader import load_profile


MESSAGES = [
    {"role": "system", "content": "Use tools."},
    {"role": "user", "content": "Read the file."},
    {"role": "assistant", "content": "Reading.", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"path":"a.py"}'}},
    ]},
    {"role": "tool", "tool_call_id": "c1", "content": "source"},
]
TOOLS = [{"type": "function", "function": {
    "name": "read", "description": "Read source.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
}}]


def response(body, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(body).encode()
    return result


def cfg(**changes):
    return make_config(**{
        "tokenizer_id": "auto", "base_url": "http://native.invalid/v1",
        "api_key": "fixture-key", "model": "chosen-model", "request_dialect": "openai",
        "context_size": 200, "max_tokens": 128, **changes,
    })


def generated(usage=None):
    return response({"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                     "usage": usage or {"input_tokens": 10, "cache_creation_input_tokens": 50,
                                        "cache_read_input_tokens": 100, "output_tokens": 1}})


@pytest.mark.parametrize("profile_enabled", [False, True])
def test_native_count_matches_sent_input_and_complete_cached_usage(tmp_path, monkeypatch, profile_enabled):
    profile = load_profile("_base", Path("profiles")) if profile_enabled else None
    if profile:
        def denormalize(self, messages):
            copied = copy.deepcopy(messages)
            copied[0]["content"] += " [wire]"
            return copied
        monkeypatch.setattr(type(profile), "denormalize_messages", denormalize)
    client = AnthropicClient(cfg(), profile)
    posts, events = [], []
    def post(url, **kwargs):
        posts.append((url, copy.deepcopy(kwargs)))
        return response({"input_tokens": 160}) if url.endswith("/count_tokens") else generated()
    monkeypatch.setattr(client._http, "post", post)
    counter = resolve_counter(client, client.cfg, event_sink=lambda event, **row: events.append((event, row)))
    assert counter.count(MESSAGES, TOOLS) == 160
    diary = tmp_path / "diary.log"
    client.set_transcript(diary)
    result = client.chat(MESSAGES, TOOLS, 1)
    client.close_transcript()
    counted, recounted, sent = [call[1]["json"] for call in posts]
    assert counted == recounted
    assert sent == {**counted, "max_tokens": 39}
    assert counted["model"] == "chosen-model"
    assert counted["tools"][0]["input_schema"] == TOOLS[0]["function"]["parameters"]
    assert counted["messages"][-1]["content"][0] == {
        "type": "tool_result", "tool_use_id": "c1", "content": "source",
    }
    if profile_enabled:
        assert counted["system"].count("[wire]") == 1
    assert result.usage.prompt_tokens == 160
    assert result.usage.cached_tokens == 100
    assert events[-1][1]["count_delta"] == 0
    assert counter.last["count_precision"] == "estimate"
    assert counter.last["count_reason"] == "provider_reports_estimate"
    assert posts[-1][1]["headers"] == posts[0][1]["headers"]
    assert posts[0][1]["headers"]["X-Api-Key"] == "fixture-key"
    assert posts[0][0] == "http://native.invalid/v1/messages/count_tokens"
    assert posts[-1][0] == "http://native.invalid/v1/messages"
    assert posts[0][1]["timeout"] == (10, 10)
    assert '"max_tokens": 39' in diary.read_text()


def test_native_side_requests_are_counted_without_advancing_the_diary(tmp_path, monkeypatch):
    client = AnthropicClient(cfg())
    posts = []
    def post(url, **kwargs):
        posts.append((url, kwargs["json"]))
        return response({"input_tokens": 160}) if url.endswith("/count_tokens") else generated()
    monkeypatch.setattr(client._http, "post", post)
    diary = tmp_path / "diary.log"
    client.set_transcript(diary)
    client.complete_side_request({"messages": MESSAGES, "max_tokens": 80})
    client.complete_tool_side_request(MESSAGES, TOOLS)
    client.close_transcript()
    assert diary.read_text() == ""
    assert client._transcript_call_n == 0
    assert len(posts) == 4
    assert "tools" not in posts[0][1]
    assert posts[1][1]["max_tokens"] == posts[3][1]["max_tokens"] == 39


def test_native_stream_counts_input_without_streaming_the_count_request(monkeypatch):
    from test_stream_rules_integration import _NativeResponse, _message_events

    client = AnthropicClient(cfg(context_size=43008))
    client._narration_streaming = True
    posts = []
    stream = _NativeResponse(_message_events("ok"))
    def post(url, **kwargs):
        posts.append((url, kwargs))
        return response({"input_tokens": 1200}) if url.endswith("/count_tokens") else stream
    monkeypatch.setattr(client._http, "post", post)
    result = client.chat(MESSAGES, TOOLS, 1)
    assert result.content == "ok"
    assert "stream" not in posts[0][1] and "stream" not in posts[0][1]["json"]
    assert posts[1][1]["stream"] and posts[1][1]["json"]["stream"]
    assert stream.closed


def test_provider_estimate_disagreement_stays_visible_without_false_exactness(monkeypatch):
    client = AnthropicClient(cfg())
    events = []
    monkeypatch.setattr(client._http, "post", lambda url, **kwargs:
                        response({"input_tokens": 162}) if url.endswith("/count_tokens") else generated())
    counter = resolve_counter(client, client.cfg, event_sink=lambda event, **row: events.append((event, row)))
    client.chat(MESSAGES, TOOLS, 1)
    assert events[-1][1]["count_delta"] == -2
    assert counter.count(MESSAGES, TOOLS) == 162
    assert counter.last["count_basis"] == "backend_input_tokens"
    assert counter.last["count_precision"] == "estimate"


@pytest.mark.parametrize("status", [400, 401, 404, 429])
def test_native_errors_are_explicit_and_model_switch_rebinds(status, monkeypatch, caplog):
    client = AnthropicClient(cfg())
    seen = []
    def post(url, **kwargs):
        model = kwargs["json"]["model"]
        seen.append(model)
        return (response({"error": "private response"}, status) if model == "chosen-model"
                else response({"input_tokens": 71}))
    monkeypatch.setattr(client._http, "post", post)
    counter = client.get_request_token_counter()
    assert counter.count(MESSAGES, TOOLS) > 0
    assert counter.last["count_basis"] == "character_estimate"
    assert counter.last["count_reason"] == f"HTTPError:{status}"
    assert "private response" not in caplog.text
    client.cfg = replace(client.cfg, model="replacement-model")
    assert counter.count(MESSAGES, TOOLS) == 71
    assert seen == ["chosen-model", "replacement-model"]


def test_native_count_projection_preserves_all_declared_input_controls(monkeypatch):
    client = AnthropicClient(cfg())
    seen = []
    def post(url, **kwargs):
        seen.append(kwargs["json"])
        return response({"input_tokens": 29})
    monkeypatch.setattr(client._http, "post", post)
    native = {"model": "chosen-model", "messages": [{"role": "user", "content": "task"}],
              "max_tokens": 100, "stream": True, "temperature": 0.4,
              "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"},
              "cache_control": {"type": "ephemeral"}, "tools": [{"name": "echo", "input_schema": {}}]}
    assert client.get_request_token_counter().count_payload(native) == 29
    assert seen == [{key: value for key, value in native.items()
                     if key not in {"max_tokens", "stream", "temperature"}}]


def test_unknown_input_is_explicit_and_does_not_suppress_the_next_valid_count(monkeypatch):
    client = AnthropicClient(cfg())
    posts = []
    monkeypatch.setattr(client._http, "post",
                        lambda *args, **kwargs: posts.append(kwargs) or response({"input_tokens": 33}))
    counter = client.get_request_token_counter()
    native = client.prepare_anthropic_request(client.prepare_chat_request(MESSAGES, TOOLS))
    assert counter.count_payload({**native, "unknown_input_control": {}}) > 0
    assert counter.last["count_basis"] == "character_estimate"
    assert counter.last["count_reason"] == "UncountableRequest"
    assert not posts and counter.last["counting_calls"] == 0
    assert counter.count_payload(native) == 33
    assert len(posts) == counter.calls == 1
