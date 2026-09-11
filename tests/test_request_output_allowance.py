"""Final requests cannot expand the client's resolved output allowance."""
import copy
import json
from pathlib import Path

import httpx
import pytest

from _config_helpers import make_config
from test_backend_token_counting import attach_transport, completion
from scripts.llm_solver.server.client import LlamaClient
from scripts.llm_solver.server.profile_loader import load_profile
from scripts.llm_solver.server.request_controls import RequestControlError


@pytest.mark.parametrize("with_profile", [False, True])
@pytest.mark.parametrize("requested,expected", [(None, 25), (-1, 25), (900, 25), (7, 7)])
def test_side_output_is_bounded_before_counting_and_generation(with_profile, requested, expected):
    cfg = make_config(context_size=1000, max_tokens=25, tokenizer_id="auto")
    profile = load_profile("_base", Path("profiles")) if with_profile else None
    client = LlamaClient(cfg, profile)
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        assert body["max_tokens"] == expected
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 20})
        return completion(20)

    attach_transport(client, handler)
    payload = {"messages": [{"role": "user", "content": "Keep useful guidance."}]}
    if requested is not None:
        payload["max_tokens"] = requested
    original = copy.deepcopy(payload)
    assert client.complete_side_request(payload).content == "ok"
    assert payload == original
    assert len(bodies) == 2 and bodies[0] == bodies[1]


@pytest.mark.parametrize("requested", [0, -2, True, 2.5, "100"])
def test_invalid_request_output_refuses_before_counting(requested):
    client = LlamaClient(make_config(max_tokens=25, tokenizer_id="auto"))
    attach_transport(client, lambda request: pytest.fail("invalid output reached HTTP"))
    with pytest.raises(RequestControlError):
        client.complete_side_request({"messages": [], "max_tokens": requested})


@pytest.mark.parametrize("limit", [0, -1, True, 2.5])
def test_unresolved_client_allowance_refuses_before_counting(limit):
    client = LlamaClient(make_config(max_tokens=limit, tokenizer_id="auto"))
    attach_transport(client, lambda request: pytest.fail("invalid allowance reached HTTP"))
    with pytest.raises(RequestControlError):
        client.complete_side_request({"messages": [], "max_tokens": 10})


@pytest.mark.parametrize("override", [
    {"extra_body": {"max_tokens": 900}},
    {"extra_body": {"n_predict": 900}},
    {"extra_body": {"max_completion_tokens": 900}},
    {"n_predict": 900},
    {"max_completion_tokens": 900},
])
def test_final_transport_rejects_output_limit_overrides(override):
    client = LlamaClient(make_config(max_tokens=25, tokenizer_id="auto"))
    attach_transport(client, lambda request: pytest.fail("output override reached HTTP"))
    with pytest.raises(RequestControlError):
        client._call_api({"model": "served", "messages": [], "max_tokens": 10, **override})
