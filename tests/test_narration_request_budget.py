"""Narration estimates follow the prepared request without extra model calls."""
import io
import json
from unittest.mock import MagicMock

import pytest

from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness.stream_rules import NarrationBudget
from scripts.llm_solver.server._streaming import StreamDelta, StreamRuleInterrupt
from scripts.llm_solver.server.client import LlamaClient
from tests._config_helpers import make_config
from tests.test_stream_rules_integration import _chunk, _ClosableStream


@pytest.mark.parametrize("text", ["abcdefghi", "甲乙丙丁戊己庚辛壬", "😀" * 9])
@pytest.mark.parametrize("chunk_size", [1, 3, 9])
def test_different_text_densities_remain_explicit_estimates(text, chunk_size):
    budget = NarrationBudget(context_size=100, fraction=0.5, message="Act.")
    budget.prepare_request(completion_tokens=2, model="active-model")
    with pytest.raises(StreamRuleInterrupt) as caught:
        for start in range(0, len(text), chunk_size):
            budget.observe(StreamDelta("text", text[start:start + chunk_size]))
    record = caught.value.matches[0]
    assert record["offset"] == 8
    assert record["observed_chars"] == 9
    assert record["measurement_basis"] == "character_estimate"
    assert record["characters_per_estimated_token"] == 4
    assert record["request_completion_tokens"] == 2
    assert record["request_model"] == "active-model"


def test_continuations_share_policy_but_bind_each_request_allowance():
    budget = NarrationBudget(context_size=100, fraction=0.05, message="Act.")
    budget.prepare_request(completion_tokens=2)
    budget.observe(StreamDelta("text", "a" * 8))
    budget.prepare_request(completion_tokens=1)
    assert (budget.chars, budget.limit_chars) == (8, 12)
    budget.observe(StreamDelta("text", "b" * 4))
    budget.prepare_request(completion_tokens=100)
    assert (budget.chars, budget.limit_chars) == (12, 20)
    budget.observe(StreamDelta("thinking", "x" * 100))
    budget.observe(StreamDelta("tool", "x" * 100))
    budget.observe(StreamDelta("text", "c" * 8))
    with pytest.raises(StreamRuleInterrupt):
        budget.observe(StreamDelta("text", "d"))


@pytest.mark.parametrize("limit", [None, 0, -1, True, 1.5, "2"])
def test_missing_or_invalid_request_allowance_stays_unknown(limit):
    budget = NarrationBudget(context_size=100, fraction=0.01, message="Act.")
    budget.prepare_request(completion_tokens=limit)
    with pytest.raises(StreamRuleInterrupt) as caught:
        budget.observe(StreamDelta("text", "abcde"))
    assert caught.value.matches[0]["request_budget_basis"] == "unknown"
    assert caught.value.matches[0]["request_completion_tokens"] is None


def test_transport_binds_after_prompt_count_and_preserves_replay(tmp_path):
    cfg = make_config(context_size=100, max_tokens=90)
    client = LlamaClient(cfg, profile=None)
    client._request_token_counter = lambda messages, tools: 97
    budget = NarrationBudget(context_size=100, fraction=0.5, message="Act.")
    def observe(delta):
        budget.observe(delta)
    observe.prepare_request = budget.prepare_request
    client._stream_observer = observe
    client._narration_streaming = True
    stream = _ClosableStream([_chunk(content="abcdefghi"), _chunk(content="unread")])
    client.client.chat.completions.create = MagicMock(return_value=stream)
    transcript = tmp_path / "narration.log"
    client.set_transcript(transcript)
    with pytest.raises(StreamRuleInterrupt) as caught:
        client._call_api({"model": "wire-model", "max_tokens": 90,
                          "messages": [{"role": "user", "content": "task"}]})
    client.close_transcript()
    assert stream.closed
    assert client.client.chat.completions.create.call_count == 1
    wire = client.client.chat.completions.create.call_args.kwargs
    assert wire["max_tokens"] == 2
    assert "logprobs" not in wire
    record = caught.value.matches[0]
    assert record["request_model"] == "wire-model"
    assert record["offset"] == 8
    assert record["request_budget_basis"] == "prepared_request_max_tokens"
    replay = StreamRuleInterrupt.from_transcript(json.loads(caught.value.model_dump_json()))
    assert replay.matches == caught.value.matches
    assert '"measurement_basis": "character_estimate"' in transcript.read_text()


def test_session_wires_budget_and_records_estimate(tmp_path, monkeypatch):
    monkeypatch.setenv("YUJ_STREAMING", "0")
    cfg = make_config(max_tokens=2, stream_rules_enabled=False)
    client = LlamaClient(cfg, profile=None)
    stream = _ClosableStream([_chunk(content="abcdefghi"), _chunk(content="unread")])
    client.client.chat.completions.create = MagicMock(return_value=stream)
    trace = io.StringIO()
    session = Session(cfg, client, "system", "task", str(tmp_path), trace_file=trace)
    result = session._chat_with_retry(1)
    assert result.finish_reason == "narration_discarded"
    assert client.client.chat.completions.create.call_count == 1
    assert client._stream_observer is None
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    trigger = next(row for row in rows if row["event"] == "stream_rule_triggered")
    assert trigger["offset"] == 8
    assert trigger["narration_measurement"]["request_completion_tokens"] == 2
    assert trigger["narration_measurement"]["measurement_basis"] == "character_estimate"
