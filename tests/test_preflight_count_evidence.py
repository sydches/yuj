"""Preflight must distinguish backend estimates from matching reported counts."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.llm_solver.harness._loop.run_step import _preflight_estimate


def _session(value, record):
    counter = SimpleNamespace(count=Mock(return_value=value), last=record)
    return SimpleNamespace(
        _tokenizer=counter,
        context=SimpleNamespace(get_messages=lambda: [{"role": "user", "content": "task"}],
                                estimate_tokens=lambda: 777),
    )


@pytest.mark.parametrize("record,authoritative", [
    ({"count_basis": "backend_input_tokens", "count_precision": "backend_reported", "prompt_tokens": 100}, True),
    ({"count_basis": "backend_input_tokens", "count_precision": "estimate", "prompt_tokens": 100}, False),
    ({"count_basis": "backend_input_tokens", "count_precision": "unverified", "prompt_tokens": 100}, False),
    ({"count_basis": "backend_input_tokens", "count_precision": "backend_reported", "prompt_tokens": 99}, False),
    ({"count_basis": "backend_input_tokens"}, False),
    ({"count_basis": "backend_input_tokens_unverified", "count_precision": "unverified", "prompt_tokens": 100}, False),
])
def test_record_basis_precision_and_value_must_agree(monkeypatch, record, authoritative):
    monkeypatch.setattr("scripts.llm_solver.harness._loop.run_step.effective_model_tool_schemas", lambda _: [])
    session = _session(100, record)
    assert _preflight_estimate(session) == 100
    assert session._preflight_count_authoritative is authoritative


@pytest.mark.parametrize("value", [True, -1, 3.5, "100"])
def test_invalid_counter_value_does_not_become_an_integer_fact(monkeypatch, value):
    monkeypatch.setattr("scripts.llm_solver.harness._loop.run_step.effective_model_tool_schemas", lambda _: [])
    session = _session(value, {"count_basis": "backend_input_tokens",
                              "count_precision": "backend_reported", "prompt_tokens": value})
    assert _preflight_estimate(session) == 777
    assert session._preflight_count_authoritative is False


def test_counting_recovery_reassesses_precision_each_time(monkeypatch):
    monkeypatch.setattr("scripts.llm_solver.harness._loop.run_step.effective_model_tool_schemas", lambda _: [])
    record = {"count_basis": "backend_input_tokens", "count_precision": "backend_reported", "prompt_tokens": 100}
    session = _session(100, record)
    for precision, expected in (("backend_reported", True), ("estimate", False), ("backend_reported", True)):
        session._tokenizer.last = {**record, "count_precision": precision}
        assert _preflight_estimate(session) == 100
        assert session._preflight_count_authoritative is expected
    session._tokenizer.count.side_effect = OSError("count unavailable")
    assert _preflight_estimate(session) == 777
    assert session._preflight_count_authoritative is False
