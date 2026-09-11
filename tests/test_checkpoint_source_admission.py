"""Checkpoint serialization preserves source before producer request admission."""
import copy
import json

import httpx
import pytest

from tests._config_helpers import make_config
from tests.test_backend_token_counting import attach_transport, completion
from tests.test_checkpoint_summary import CharTokenizer, _messages, _valid_summary
from scripts.llm_solver.harness._loop.checkpoint_summary import generate_checkpoint
from scripts.llm_solver.server.client import LlamaClient


@pytest.mark.parametrize("producer_capacity,accepted", [(512, False), (32768, True)])
def test_complete_source_reaches_producer_count_before_generation(producer_capacity, accepted):
    messages = _messages(3)
    diagnostic = "HEAD" + "x" * 6000 + "MIDDLE_DIAGNOSTIC" + "y" * 6000 + "TAIL"
    messages[3]["content"] = diagnostic
    original = copy.deepcopy(messages)
    client = LlamaClient(make_config(tokenizer_id="auto", context_size=producer_capacity,
                                    max_tokens=2000, model="summary-producer"))
    counted, generated = [], []

    def handler(request):
        body = json.loads(request.content)
        size = len(json.dumps(body["messages"])) // 4
        assert diagnostic in body["messages"][-1]["content"]
        assert "tools" not in body and body["model"] == "summary-producer"
        if request.url.path.endswith("/input_tokens"):
            counted.append(body)
            return httpx.Response(200, json={"input_tokens": size})
        generated.append(body)
        response = completion(size).json()
        response["choices"][0]["message"]["content"] = _valid_summary()
        return httpx.Response(200, json=response)

    attach_transport(client, handler)
    try:
        result = generate_checkpoint(
            model="receiver", messages=messages, trace_events=[], tokenizer=CharTokenizer(),
            keep_recent_tokens=1, max_summary_tokens=2000, budget=5000,
            call_model=lambda payload: client.complete_side_request(payload).content,
        )
    finally:
        client.client.close()
    assert counted and bool(generated) is accepted
    assert result.valid is accepted
    assert messages == original
    if accepted:
        assert generated[0]["messages"] == counted[-1]["messages"]
        assert result.tokens_after < 5000
    else:
        assert result.fallback == "digest"
        assert "ContextBudgetExceeded" in result.reason
