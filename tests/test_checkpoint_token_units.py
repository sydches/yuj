"""Receiving-model reserve is not a producing-model token allowance."""
import pytest

from scripts.llm_solver.harness._loop.checkpoint_summary import summary_token_limit
from scripts.llm_solver.server.request_controls import bound_completion_budget


@pytest.mark.parametrize("reserve", [1, 1000, 10000])
@pytest.mark.parametrize("declared", [300, 6000])
def test_positive_receiver_space_does_not_convert_the_producer_allowance(reserve, declared):
    assert summary_token_limit(
        reserve_tokens=reserve, configured_max_tokens=declared,
    ) == declared


@pytest.mark.parametrize("reserve,declared", [(0, 4000), (-1, 4000), (1000, 0), (1000, -1)])
def test_no_receiver_space_or_no_generation_permission_prevents_request(reserve, declared):
    assert summary_token_limit(
        reserve_tokens=reserve, configured_max_tokens=declared,
    ) == 0


@pytest.mark.parametrize("producer_input,expected", [(100, 3000), (3000, 999)])
def test_producer_capacity_is_applied_in_its_own_units(producer_input, expected):
    payload = {
        "model": "producer", "messages": [{"role": "user", "content": "history"}],
        "max_tokens": summary_token_limit(reserve_tokens=1, configured_max_tokens=3000),
    }
    seen = []

    def producer_count(request):
        seen.append(request)
        return producer_input

    bounded = bound_completion_budget(payload, 4000, payload_counter=producer_count)
    assert seen == [payload]
    assert bounded["max_tokens"] == expected
    assert payload["max_tokens"] == 3000
