"""Visible prose uses the serving tokenizer without requesting generation."""
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from _config_helpers import make_config
from test_backend_token_counting import attach_transport
from scripts.llm_solver.server.client import LlamaClient
from scripts.llm_solver.server._streaming import StreamRuleInterrupt
from scripts.llm_solver.harness.stream_rules import NarrationBudget
from scripts.llm_solver.harness import time_budget


def rig(handler, **settings):
    client = LlamaClient(make_config(model="active-model", tokenizer_id="auto", **settings))
    attach_transport(client, handler)
    budget = NarrationBudget(context_size=400, fraction=.01, message="Act.",
                             text_counter=client.get_text_token_counter())
    budget.prepare_request(completion_tokens=40, model="active-model")
    return client, budget


def text(value):
    return SimpleNamespace(source="text", delta=value)


@pytest.mark.parametrize("value, token_count, interrupts", [("界" * 4, 12, True), ("a" * 17, 3, False)])
def test_native_density_changes_decision_and_uses_actual_route(value, token_count, interrupts):
    requests = []
    def handler(request):
        requests.append(request)
        assert request.url.path == "/proxy/tokenize"
        assert request.headers["authorization"] == "Bearer local"
        assert json.loads(request.content) == {"model": "active-model", "content": value,
                                             "add_special": False, "parse_special": False}
        return httpx.Response(200, json={"tokens": list(range(token_count))})
    _, budget = rig(handler, base_url="http://fixture/proxy/v1")
    if interrupts:
        with pytest.raises(StreamRuleInterrupt) as caught:
            budget.observe(text(value))
        record = caught.value.matches[0]
        assert record["measurement_basis"] == "backend_text_tokens"
        assert record["observed_text_tokens"] == token_count
    else:
        budget.observe(text(value))
    assert len(requests) == 1
    assert budget.tokens == token_count


def test_chunks_and_continuations_keep_full_text_and_request_allowance():
    seen = []
    def handler(request):
        value = json.loads(request.content)["content"]
        seen.append(value)
        return httpx.Response(200, json={"tokens": list(range(len(value)))})
    _, budget = rig(handler)
    budget.observe(SimpleNamespace(source="reasoning", delta="ignored" * 100))
    budget.observe(SimpleNamespace(source="tool", delta="ignored" * 100))
    for value in "abc":
        budget.observe(text(value))
    assert seen == []
    budget.prepare_request(completion_tokens=1, model="active-model")
    assert seen == ["abc"]
    budget.observe(text("d"))
    with pytest.raises(StreamRuleInterrupt):
        budget.observe(text("e"))
    assert seen == ["abc", "abcd", "abcde"]
    assert budget.chars == 5


@pytest.mark.parametrize("response", [httpx.Response(404), httpx.Response(200, json={"tokens": [True]})])
def test_unavailable_measurement_falls_back_once(response):
    seen = []
    def handler(request):
        seen.append(request)
        return response
    _, budget = rig(handler)
    budget.observe(text("aaaa"))
    with pytest.raises(StreamRuleInterrupt) as caught:
        budget.observe(text("b" * 13))
    assert len(seen) == 1
    assert caught.value.matches[0]["measurement_basis"] == "character_estimate"
    assert caught.value.matches[0]["text_count"]["count_basis"] == "unavailable"


def test_counting_wait_uses_remaining_run_and_shared_count_cost(monkeypatch):
    now, waits = [0.0], []
    monkeypatch.setattr(time_budget.time, "monotonic", lambda: now[0])
    def handler(request):
        waits.append(request.extensions["timeout"]["read"])
        now[0] += 1
        return httpx.Response(200, json={"tokens": [1]})
    _, budget = rig(handler, timeout_connect=1)
    with time_budget.run_time_budget(.5):
        budget.observe(text("abcd"))
        budget.observe(text("efgh"))
    assert waits == [.5]
    assert budget.text_counter is None


def test_model_change_and_unsupported_dialect_do_not_guess_tokenizer():
    def forbidden(request):
        raise AssertionError("no HTTP expected")
    client, budget = rig(forbidden)
    client.cfg = replace(client.cfg, model="changed-model")
    budget.observe(text("abcd"))
    assert budget.text_counter is None
    client.cfg = replace(client.cfg, request_dialect="openai")
    assert client.get_text_token_counter() is None


def test_real_session_binds_native_counter_and_replays_interrupt(tmp_path):
    import io
    from unittest.mock import MagicMock
    from test_stream_rules_integration import _ClosableStream, _chunk
    from scripts.llm_solver.harness.loop import Session
    from test_stream_rules_integration import ReplayClient

    cfg = make_config(tokenizer_id="auto", context_size=4096, stream_rules_enabled=False)
    client = LlamaClient(cfg)
    def handler(request):
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 100})
        assert request.url.path == "/tokenize"
        return httpx.Response(200, json={"tokens": list(range(50))})
    attach_transport(client, handler)
    stream = _ClosableStream([_chunk(content="界" * 41), _chunk(content="unread")])
    client.client.chat.completions.create = MagicMock(return_value=stream)
    transcript = tmp_path / "narration.log"
    client.set_transcript(transcript)
    trace = io.StringIO()
    session = Session(cfg, client, "system", "task", str(tmp_path), trace_file=trace)
    assert session._chat_with_retry(1).finish_reason == "narration_discarded"
    client.close_transcript()
    assert stream.closed
    assert '"measurement_basis": "backend_text_tokens"' in transcript.read_text()
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    measurement = next(row["narration_measurement"] for row in rows if "narration_measurement" in row)
    assert measurement["observed_text_tokens"] == 50
    assert measurement["text_count"]["count_basis"] == "backend_text_tokens"
    replay = ReplayClient(transcript, strict_fidelity=False)
    replay_session = Session(cfg, replay, "system", "task", str(tmp_path))
    assert replay_session._chat_with_retry(1).finish_reason == "narration_discarded"
