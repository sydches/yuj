"""Counting uses declared time and propagates caller exhaustion before HTTP."""
import httpx
import pytest

from _config_helpers import make_config
from test_backend_token_counting import attach_transport, MESSAGES, TOOLS
from test_anthropic_token_counting import cfg as native_config, response
from scripts.llm_assist._anthropic import AnthropicClient
from scripts.llm_solver.server.client import LlamaClient
from scripts.llm_solver.harness.request_counting import resolve_counter, observe_role_counter
from scripts.llm_solver.harness.time_budget import run_time_budget, command_time_budget, BudgetExhausted
from types import SimpleNamespace


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('cap', [0.02, 2.0])
def test_transport_count_timeout_uses_remaining_allowance(monkeypatch, native, cap):
    now = [0.0]
    monkeypatch.setattr('scripts.llm_solver.harness.time_budget.time.monotonic', lambda: now[0])
    cfg = native_config(timeout_connect=cap) if native else make_config(tokenizer_id='auto', timeout_connect=cap)
    client = AnthropicClient(cfg) if native else LlamaClient(cfg)
    timeouts, events = [], []
    if native:
        def post(url, **kwargs):
            assert url.endswith('/messages/count_tokens')
            timeouts.append(kwargs['timeout'])
            return response({'input_tokens': 12})
        monkeypatch.setattr(client._http, 'post', post)
    else:
        def http(request):
            assert request.url.path.endswith('/input_tokens')
            timeouts.append(tuple(request.extensions['timeout'].values()))
            return httpx.Response(200, json={'input_tokens': 12})
        attach_transport(client, http)
    counter = resolve_counter(client, cfg, event_sink=lambda event, **row: events.append(row))
    with run_time_budget(1):
        assert counter.count(MESSAGES, TOOLS) == 12
        assert set(timeouts[-1]) == {min(cap, 1)}
        now[0] = 0.99
        assert counter.count(MESSAGES, TOOLS) == 12
        assert all(value == pytest.approx(0.01) for value in timeouts[-1])
        now[0] = 1
        with pytest.raises(BudgetExhausted):
            client.chat(MESSAGES, TOOLS, turn=0)
        assert len(timeouts) == 2
        assert counter.last['count_reason'] == 'BudgetExhausted'
        assert counter.last['count_basis'] == 'character_estimate'
        assert counter.last['counting_calls'] == 0
        assert counter.last['counting_timeout_seconds'] == 0
        assert events[-1] == counter.last
    # Exhaustion belongs to the caller, not a cached backend capability failure.
    assert counter.count(MESSAGES, TOOLS) == 12
    assert set(timeouts[-1]) == {cap}


def test_role_counter_uses_parent_command_deadline(monkeypatch):
    now = [0.0]
    monkeypatch.setattr('scripts.llm_solver.harness.time_budget.time.monotonic', lambda: now[0])
    client = LlamaClient(make_config(tokenizer_id='auto', timeout_connect=2))
    events = []
    owner = SimpleNamespace(_session_number=1, _current_turn=2,
        _emit=lambda event, **row: events.append(row))
    observe_role_counter(owner, client, 'side')
    attach_transport(client, lambda request: pytest.fail('expired count must not launch HTTP'))
    with run_time_budget(5), command_time_budget(1):
        now[0] = 1
        with pytest.raises(BudgetExhausted):
            client.get_request_token_counter().count(MESSAGES, TOOLS)
    assert events[-1]['role'] == 'side'
    assert events[-1]['counting_calls'] == 0


def test_zero_configured_count_allowance_does_not_launch(monkeypatch):
    client = LlamaClient(make_config(tokenizer_id='auto', timeout_connect=0))
    attach_transport(client, lambda request: pytest.fail('zero allowance must not launch HTTP'))
    counter = resolve_counter(client, client.cfg)
    assert counter.count(MESSAGES, TOOLS) > 0
    assert counter.last['count_reason'] == 'counting_allowance_exhausted'
    assert counter.last['counting_calls'] == 0
