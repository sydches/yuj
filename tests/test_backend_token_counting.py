"""Request-bound counting across the real SDK and harness budget decisions."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from io import StringIO
from pathlib import Path

import httpx
import openai
import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness.context import FullTranscript
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness.request_counting import resolve_counter
from scripts.llm_solver.server.client import LlamaClient
from scripts.llm_solver.server.profile_loader import load_profile
from scripts.llm_solver.server.types import ContextBudgetExceeded


MESSAGES = [{"role": "user", "content": "Count this sentence."}]
TOOLS = [{"type": "function", "function": {
    "name": "echo", "parameters": {"type": "object", "properties": {}},
}}]


def attach_transport(client, handler):
    client.client.close()
    client.client = openai.OpenAI(
        base_url=client.cfg.base_url, api_key=client.cfg.api_key, max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def completion(prompt_tokens):
    return httpx.Response(200, json={
        "id": "reply", "object": "chat.completion", "model": "served",
        "created": 0, "choices": [{"index": 0, "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1,
                  "total_tokens": prompt_tokens + 1},
    })


@pytest.mark.parametrize("with_profile", [False, True])
def test_count_uses_generation_preparation_and_authenticated_sdk_transport(with_profile, monkeypatch):
    cfg = make_config(tokenizer_id="auto", model="served-alias", context_size=100,
                      max_tokens=80, server_request_extra={"temperature": 0.2})
    profile = load_profile("_base", Path("profiles")) if with_profile else None
    if profile:
        def denormalize(messages):
            result = copy.deepcopy(messages)
            result[0]["content"] += " [wire]"
            return result
        monkeypatch.setattr(type(profile), "denormalize_messages", lambda self, messages: denormalize(messages))
    client = LlamaClient(cfg, profile)
    requests, events = [], []

    def handler(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        assert request.headers["authorization"] == "Bearer local"
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 27})
        assert request.url.path == "/v1/chat/completions"
        return completion(27)

    attach_transport(client, handler)
    counter = resolve_counter(client, cfg, event_sink=lambda event, **row: events.append((event, row)))
    original = copy.deepcopy(MESSAGES)
    assert counter.count(MESSAGES, TOOLS) == 27
    client.chat(MESSAGES, TOOLS, turn=0)
    assert MESSAGES == original
    counted, recounted, generated = [row[1] for row in requests]
    assert counted == recounted
    assert generated == {**counted, "max_tokens": 72}
    assert counted["model"] == "served-alias"
    assert counted["tools"] == TOOLS
    assert counted["temperature"] == 0.2
    assert "extra_body" not in counted
    assert counted["chat_template_kwargs"]["enable_thinking"] is False
    if with_profile:
        assert counted["messages"][0]["content"].count("[wire]") == 1
    assert events[-1][0] == "request_token_count_usage"
    assert events[-1][1]["count_delta"] == 0
    assert counter.calls == 2
    assert counter.elapsed_seconds > 0


def test_side_and_continued_requests_count_their_own_controls_and_messages():
    cfg = make_config(tokenizer_id="auto", thinking_level="high")
    client = LlamaClient(cfg)
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        return (httpx.Response(200, json={"input_tokens": 20})
                if request.url.path.endswith("/input_tokens") else completion(20))
    attach_transport(client, handler)
    client.chat(MESSAGES, TOOLS, turn=0)
    client.complete_side_request({"messages": MESSAGES, "max_tokens": 10})
    client.complete_tool_side_request(MESSAGES, TOOLS)
    continued = client.prepare_chat_request(
        MESSAGES + [{"role": "assistant", "content": "continuing"}], TOOLS,
    )
    client._call_api(continued)
    for index in range(0, len(requests), 2):
        assert requests[index][1] == requests[index + 1][1]
    assert requests[0][1]["chat_template_kwargs"]["enable_thinking"] is True
    assert requests[2][1]["chat_template_kwargs"]["enable_thinking"] is False
    assert requests[4][1]["chat_template_kwargs"]["enable_thinking"] is False
    assert "tools" not in requests[2][1]
    assert requests[6][1]["messages"][-1]["content"] == "continuing"


@pytest.mark.parametrize("entry", ["chat", "side", "tool_side"])
@pytest.mark.parametrize("prompt_tokens", [100, 101])
@pytest.mark.parametrize("with_profile", [False, True])
def test_exhausted_backend_count_refuses_before_generation(entry, prompt_tokens, with_profile):
    cfg = make_config(tokenizer_id="auto", model="served", context_size=100, max_tokens=15)
    client = LlamaClient(cfg, load_profile("_base", Path("profiles")) if with_profile else None)
    requests = []
    def handler(request):
        requests.append(request.url.path)
        assert request.url.path.endswith("/input_tokens"), "exhausted request reached generation"
        return httpx.Response(200, json={"input_tokens": prompt_tokens})
    attach_transport(client, handler)
    original = copy.deepcopy(MESSAGES)
    with pytest.raises(ContextBudgetExceeded) as failure:
        if entry == "chat":
            client.chat(MESSAGES, TOOLS, turn=0)
        elif entry == "side":
            client.complete_side_request({"messages": MESSAGES, "max_tokens": 15})
        else:
            client.complete_tool_side_request(MESSAGES, TOOLS)
    assert len(requests) == 1
    assert failure.value.context_size == 100
    assert failure.value.count_record["prompt_tokens"] == prompt_tokens
    assert failure.value.count_record["count_basis"] == "backend_input_tokens"
    assert failure.value.count_record["count_precision"] == "backend_reported"
    assert failure.value.count_record["request_sha256"]
    assert client._transcript_call_n == 0
    assert MESSAGES == original


@pytest.mark.parametrize("prompt_tokens,output_tokens", [(90, 9), (99, 1)])
def test_remaining_output_space_preserves_usable_requests(prompt_tokens, output_tokens):
    client = LlamaClient(make_config(tokenizer_id="auto", model="served", context_size=100, max_tokens=15))
    sent = []
    def handler(request):
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": prompt_tokens})
        sent.append(json.loads(request.content))
        return completion(prompt_tokens)
    attach_transport(client, handler)
    assert client.chat(MESSAGES, TOOLS, turn=0).content == "ok"
    assert sent[0]["max_tokens"] == output_tokens


def test_spare_backend_capacity_does_not_expand_prepared_request_output():
    from scripts.llm_solver.context_allocation import allocate_context
    cfg = allocate_context(make_config(tokenizer_id="auto", model="served",
        context_size=100, max_tokens=25), observed=1000)
    client = LlamaClient(cfg)
    sent = []
    def handler(request):
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 90})
        sent.append(json.loads(request.content))
        return completion(90)
    attach_transport(client, handler)
    assert client.chat(MESSAGES, TOOLS, turn=0).content == "ok"
    assert sent[0]["max_tokens"] == 9
    assert client.cfg.context_allocation["observed_capacity"] == 1000
    assert client.cfg.context_size == 100


def test_streaming_exhaustion_refuses_before_opening_a_stream(monkeypatch):
    monkeypatch.setenv("YUJ_STREAMING", "1")
    client = LlamaClient(make_config(tokenizer_id="auto", model="served", context_size=100, max_tokens=15))
    def handler(request):
        assert request.url.path.endswith("/input_tokens")
        return httpx.Response(200, json={"input_tokens": 100})
    attach_transport(client, handler)
    with pytest.raises(ContextBudgetExceeded):
        client.chat(MESSAGES, TOOLS, turn=0)


def test_side_request_without_output_cap_still_refuses_known_exhaustion():
    client = LlamaClient(make_config(tokenizer_id="auto", model="served", context_size=100))
    def handler(request):
        assert request.url.path.endswith("/input_tokens")
        return httpx.Response(200, json={"input_tokens": 100})
    attach_transport(client, handler)
    with pytest.raises(ContextBudgetExceeded):
        client.complete_side_request({"messages": MESSAGES})


@pytest.mark.parametrize("response", [
    httpx.Response(404, json={"error": "unsupported"}),
    httpx.Response(401, json={"error": "private credential must not enter logs"}),
    httpx.Response(200, json={"input_tokens": True}),
    httpx.Response(200, json={"input_tokens": -1}),
    httpx.Response(200, json={"input_tokens": "42"}),
    httpx.Response(200, json={}),
])
def test_unavailable_counts_are_explicit_and_model_change_reprobes(response, caplog):
    client = LlamaClient(make_config(tokenizer_id="auto", model="first"))
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body["model"])
        return response if body["model"] == "first" else httpx.Response(200, json={"input_tokens": 91})
    attach_transport(client, handler)
    counter = client.get_request_token_counter()
    assert counter.count(MESSAGES) > 0
    assert counter.last["count_basis"] == "character_estimate"
    assert counter.last["count_reason"]
    counter.count(MESSAGES)
    assert seen == ["first"]  # no repeated failed probes inside one budget pass
    assert "private credential" not in caplog.text
    client.cfg = replace(client.cfg, model="second")
    assert counter.count(MESSAGES) == 91
    assert seen == ["first", "second"]
    assert counter.last["count_basis"] == "backend_input_tokens"


def test_standard_compatible_dialect_does_not_assume_llama_extension():
    client = LlamaClient(make_config(tokenizer_id="auto", request_dialect="openai"))
    def no_request(request):
        pytest.fail("unsupported dialect must not probe a llama extension")
    attach_transport(client, no_request)
    counter = client.get_request_token_counter()
    assert counter.count(MESSAGES, TOOLS) > counter.count(MESSAGES)
    assert counter.last["count_reason"] == "unsupported_request_dialect"
    assert counter.calls == 0


def test_counts_are_not_cached_across_server_changes_at_the_same_alias():
    client = LlamaClient(make_config(tokenizer_id="auto"))
    answers = iter([31, 49])
    attach_transport(client, lambda request: httpx.Response(200, json={"input_tokens": next(answers)}))
    counter = client.get_request_token_counter()
    assert [counter.count(MESSAGES), counter.count(MESSAGES)] == [31, 49]


def test_rejected_fragment_does_not_disable_counting_the_next_complete_request():
    client = LlamaClient(make_config(tokenizer_id="auto"))
    def handler(request):
        messages = json.loads(request.content)["messages"]
        return (httpx.Response(400, json={"error": "incomplete tool history"})
                if messages[0]["role"] == "tool"
                else httpx.Response(200, json={"input_tokens": 39}))
    attach_transport(client, handler)
    counter = client.get_request_token_counter()
    assert counter.count([{"role": "tool", "content": "result"}]) > 0
    assert counter.last["count_basis"] == "character_estimate"
    assert counter.count(MESSAGES) == 39
    assert counter.calls == 2


def test_session_context_and_preflight_use_current_request_counter(tmp_path):
    from scripts.llm_solver.harness._loop.run_step import _preflight_estimate

    cfg = make_config(tokenizer_id="auto")
    client = LlamaClient(cfg)
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"input_tokens": 719})
    attach_transport(client, handler)
    context = FullTranscript(token_estimator=lambda messages: 1)
    trace = StringIO()
    session = Session(cfg, client, "system", "task", str(tmp_path),
                      context_manager=context, trace_file=trace)
    assert context.estimate_tokens() == 719
    assert _preflight_estimate(session) == 719
    assert all(body["tools"] == session.model_tool_schemas for body in seen)
    assert '"event":"request_token_count"' in trace.getvalue()
    assert session._tokenizer is client.get_request_token_counter()
    session._plan_tool_schemas = TOOLS
    session._plan_mode.active = True
    assert _preflight_estimate(session) == 719
    assert seen[-1]["tools"] == TOOLS


def test_backend_usage_disagreement_is_visible():
    client = LlamaClient(make_config(tokenizer_id="auto"))
    attach_transport(client, lambda request: (
        httpx.Response(200, json={"input_tokens": 50})
        if request.url.path.endswith("/input_tokens") else completion(57)))
    events = []
    counter = resolve_counter(client, client.cfg, event_sink=lambda event, **row: events.append((event, row)))
    client.chat(MESSAGES, [], turn=0)
    assert events[-1][1]["count_delta"] == 7
    assert counter.last["prompt_tokens"] == 50
    assert counter.count(MESSAGES) == 50
    assert counter.last["count_basis"] == "backend_input_tokens_unverified"
    assert counter.last["count_reason"] == "response_usage_disagreement"
    client.cfg = replace(client.cfg, model="another")
    assert counter.count(MESSAGES) == 50
    assert counter.last["count_basis"] == "backend_input_tokens"


def test_fallback_fit_and_subsequent_context_use_replacement_backend(tmp_path):
    from test_model_fallback_integration import _cfg, _session, FIXTURE_PROFILES
    from scripts.llm_solver.harness._loop.model_fallback_runtime import activate_next_fallback
    from scripts.llm_solver.harness._loop.model_role_runtime import build_model_role_runtime

    cfg = _cfg(tokenizer_id="auto")
    main = LlamaClient(cfg, load_profile("_base", FIXTURE_PROFILES))
    attach_transport(main, lambda request: httpx.Response(200, json={"input_tokens": 7000}))
    replacements, seen = [], []
    def factory(role_cfg, role_profile):
        client = LlamaClient(role_cfg, role_profile)
        client.query_server_context = lambda: 4096
        def handler(request):
            seen.append((str(request.url), json.loads(request.content)))
            return httpx.Response(200, json={"input_tokens": 113})
        attach_transport(client, handler)
        replacements.append(client)
        return client
    runtime = build_model_role_runtime(
        cfg=cfg, main_client=main, profiles_dir=FIXTURE_PROFILES, client_factory=factory,
    )
    session = _session(tmp_path, cfg, main, runtime)
    assert session.context.estimate_tokens() == 7000
    session._last_actual_prompt_tokens = 7000
    session._preflight_density = 1.0
    assert activate_next_fallback(session, 1, reason="transient_exhausted")
    assert session.client is replacements[0]
    assert session._tokenizer is replacements[0].get_request_token_counter()
    assert session._tokenizer is not main.get_request_token_counter()
    assert session.context.estimate_tokens() == 113
    assert session._last_actual_prompt_tokens == 0
    assert session._preflight_prev_estimate is None
    assert session._preflight_density == 0.25
    assert all(url.startswith("http://127.0.0.1:8182/") for url, body in seen)
    assert seen[-1][1]["tools"] == session.model_tool_schemas


def test_compaction_does_not_skip_backend_count_for_short_dense_input(tmp_path):
    from test_session_compaction import _make_session, _seed_trace_jsonl

    _seed_trace_jsonl(tmp_path, 30)
    session = _make_session(tmp_path, trace_events=[{"event": "tool_call", "tool_name": "write"}])
    cfg = make_config(tokenizer_id="auto")
    client = LlamaClient(cfg)
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        count = 98000 if body["messages"] == messages else len(request.content) // 4
        return httpx.Response(200, json={"input_tokens": count})
    attach_transport(client, handler)
    session._tokenizer = client.get_request_token_counter()
    messages = [{"role": "system", "content": "short"}, *MESSAGES]
    session._maybe_compact_messages(messages)
    assert seen and seen[0]["messages"] == messages
    assert session._compacted


def test_repo_map_increment_uses_backend_request_boundaries():
    from scripts.llm_solver.harness.repo_map import _incremental_counter

    client = LlamaClient(make_config(tokenizer_id="auto"))
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"input_tokens": 99 + len(body["messages"][0]["content"])})
    attach_transport(client, handler)
    counter = _incremental_counter("task", tokenizer=client.get_request_token_counter(), token_estimator=None)
    assert counter("map") == 5  # the two separating newlines plus the map
    assert [row["messages"][0]["content"] for row in seen] == ["task", "task\n\nmap"]


def test_replay_handover_binds_live_backend_counter(tmp_path):
    from scripts.llm_solver.harness._loop import replay_handover
    from scripts.llm_solver.server.replay_client import ReplayClient

    transcript = tmp_path / "replay.log"
    transcript.write_text('=== turn 001 input ===\n{}\n=== turn 001 output ===\n'
                          '{"choices":[{"message":{"role":"assistant"},"finish_reason":"stop"}]}')
    replay = ReplayClient(transcript, stop_turn=1)
    cfg = make_config(tokenizer_id="auto")
    live = LlamaClient(cfg)
    attach_transport(live, lambda request: httpx.Response(200, json={"input_tokens": 219}))
    session = Session(cfg, replay, "system", "task", str(tmp_path), context_manager=FullTranscript())
    assert session._tokenizer is None
    replay_handover.arm(replay, live_client_factory=lambda: live)
    assert replay_handover.maybe_handover(session, 1)
    assert session._tokenizer is live.get_request_token_counter()
    assert session.context.estimate_tokens() == 219


def test_current_backend_count_supersedes_stale_usage_in_the_session_loop(tmp_path):
    cfg = make_config(tokenizer_id="auto", max_turns=1, reply_mode="conversation")
    client = LlamaClient(cfg)
    generated = []
    def handler(request):
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 100})
        generated.append(json.loads(request.content))
        return completion(100)
    attach_transport(client, handler)
    session = Session(cfg, client, "system", "task", str(tmp_path), context_manager=FullTranscript())
    session._get_server_ctx = lambda: cfg.context_size
    session._last_actual_prompt_tokens = 9000
    session._preflight_prev_estimate = 40
    session._preflight_density = 2.0
    session.run()
    assert len(generated) == 1
    assert not getattr(session, "_compacted", False)
    assert any(message["content"] == "task" for message in generated[0]["messages"])


def test_profile_without_tool_support_counts_only_the_sent_surface():
    profile = replace(load_profile("_base", Path("profiles")), supports_tool_calls=False)
    client = LlamaClient(make_config(tokenizer_id="auto"), profile)
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        return (httpx.Response(200, json={"input_tokens": 17})
                if request.url.path.endswith("/input_tokens") else completion(17))
    attach_transport(client, handler)
    assert client.get_request_token_counter().count(MESSAGES, TOOLS) == 17
    client.chat(MESSAGES, TOOLS, turn=0)
    assert len(bodies) == 3 and bodies[0] == bodies[1] == bodies[2]
    assert "tools" not in bodies[0] and "tool_choice" not in bodies[0]


def test_stream_count_and_generation_share_the_final_payload(monkeypatch):
    monkeypatch.setenv("YUJ_STREAMING", "1")
    client = LlamaClient(make_config(tokenizer_id="auto"))
    bodies, events = [], []
    def handler(request):
        bodies.append(json.loads(request.content))
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 17})
        chunks = [
            {"id": "reply", "object": "chat.completion.chunk", "created": 0, "model": "served",
             "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]},
            {"id": "reply", "object": "chat.completion.chunk", "created": 0, "model": "served",
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 17, "completion_tokens": 1, "total_tokens": 18}},
        ]
        data = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=data)
    attach_transport(client, handler)
    resolve_counter(client, client.cfg, event_sink=lambda event, **row: events.append((event, row)))
    result = client.chat(MESSAGES, TOOLS, turn=0)
    assert result.content == "ok"
    assert len(bodies) == 2 and bodies[0] == bodies[1]
    assert bodies[0]["stream"] is True
    assert bodies[0]["stream_options"] == {"include_usage": True}
    assert events[-1][1]["count_delta"] == 0


def test_replacement_transport_cannot_inherit_a_false_counting_capability():
    class AnotherTransport(LlamaClient):
        def _call_api(self, payload):
            raise AssertionError("no generation in this check")
    client = AnotherTransport(make_config(tokenizer_id="auto"))
    assert resolve_counter(client, client.cfg) is None
