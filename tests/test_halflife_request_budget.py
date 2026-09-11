"""HalfLife pressure must follow the current request evidence and allowance."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.llm_solver.harness.context_strategies.halflife_context import HalfLifeContext
from scripts.llm_solver.harness.request_counting import bind_session_counter


class Counter:
    def __init__(self, value=600, precision="backend_reported", basis="backend_input_tokens"):
        self.value, self.precision, self.basis = value, precision, basis
        self.calls = []
        self.last = {}

    def count(self, messages, tools=None):
        self.calls.append((deepcopy(messages), deepcopy(tools)))
        self.last = dict(prompt_tokens=self.value, count_precision=self.precision,
                         count_basis=self.basis)
        return self.value


def make_session(counter, *, context=1000, output=200, reference=0, mode="auto"):
    ctx = HalfLifeContext(context_size=context, context_limit_tokens=reference,
                          verbatim_tool_results=0, cap_7_chars=200)
    ctx.add_system("system")
    ctx.add_user("task")
    ctx.add_assistant({"role": "assistant", "content": "inspect"})
    ctx.add_tool_result("call", "HEAD" + "x" * 992 + "TAIL")
    cfg = SimpleNamespace(context_size=context, max_tokens=output,
                          context_fill_ratio=.9, tokenizer_id=mode)
    session = SimpleNamespace(context=ctx, cfg=cfg, client=object(),
                              _session_number=1, _current_turn=0,
                              model_tool_schemas=[], _emit=lambda *a, **k: None)
    bind_session_counter(session, counter)
    return session


def content(session):
    return session.context.get_messages()[-1]["content"]


@pytest.mark.parametrize("precision,basis", [
    ("estimate", "backend_input_tokens"),
    ("unverified", "backend_input_tokens_unverified"),
    ("estimate", "character_estimate"),
])
def test_unverified_count_preserves_undecayed_source(precision, basis):
    session = make_session(Counter(precision=precision, basis=basis))
    assert len(content(session)) == 1000


def test_unknown_backend_preserves_source_even_at_explicit_activation():
    session = make_session(None, reference=1)
    assert len(content(session)) == 1000


def test_response_reserve_limits_activation_reference():
    session = make_session(Counter(value=450), output=200)
    assert len(content(session)) == 200  # allowance 800, declared half = 400


def test_explicit_reference_cannot_exceed_current_allowance():
    session = make_session(Counter(), reference=10000)
    assert len(content(session)) == 200


def test_current_allocation_change_rebuilds_cached_view_and_preserves_raw():
    counter = Counter()
    session = make_session(counter, context=4000, output=1000)
    raw = session.context.snapshot_messages()
    assert len(content(session)) == 1000
    session.cfg.context_size, session.cfg.max_tokens = 1000, 200
    assert len(content(session)) == 200
    session.cfg.context_size, session.cfg.max_tokens = 4000, 1000
    assert len(content(session)) == 1000
    assert session.context.snapshot_messages() == raw


def test_tool_surface_change_recounts_full_source_without_counting_cached_reads():
    counter = Counter(value=300)
    session = make_session(counter)
    assert len(content(session)) == 1000
    assert len(content(session)) == 1000
    assert len(counter.calls) == 1
    counter.value = 600
    session.model_tool_schemas = [{"type": "function", "function": {"name": "read"}}]
    assert len(content(session)) == 200
    assert counter.calls[-1][1] == session.model_tool_schemas
    assert counter.calls[-1][0][-1]["content"] == "HEAD" + "x" * 992 + "TAIL"


def test_rebinding_new_counter_reconsiders_full_source():
    session = make_session(Counter())
    assert len(content(session)) == 200
    bind_session_counter(session, Counter(value=300), reset_observations=True)
    assert len(content(session)) == 1000


def test_estimate_refreshes_projection_when_the_tool_surface_changes():
    counter = Counter(value=300)
    session = make_session(counter)
    assert session.context.estimate_tokens() == 300
    counter.value = 600
    session.model_tool_schemas = [{"type": "function", "function": {"name": "read"}}]
    assert session.context.estimate_tokens() == 600
    assert len(content(session)) == 200


def test_omission_does_not_claim_unverified_artifacts_exist():
    session = make_session(Counter())
    result = content(session)
    assert "trace/transcript artifacts" not in result
    assert "retrieval unverified" in result


def test_explicit_legacy_opt_out_retains_declared_estimate_policy():
    session = make_session(None, reference=1, mode="")
    assert len(content(session)) == 200


@pytest.mark.parametrize("value", [True, -1, 600.5, "600"])
def test_invalid_count_never_authorizes_decay(value):
    assert len(content(make_session(Counter(value=value), reference=1))) == 1000


def test_mismatched_count_evidence_and_counter_failure_preserve_source():
    class Mismatch(Counter):
        def count(self, messages, tools=None):
            super().count(messages, tools)
            self.last["prompt_tokens"] += 1
            return self.value

    class Unavailable(Counter):
        def count(self, messages, tools=None):
            raise OSError("count unavailable")

    for counter in (Mismatch(), Unavailable()):
        assert len(content(make_session(counter, reference=1))) == 1000


def test_prompt_policy_also_bounds_reference_and_exhaustion_skips_count():
    counter = Counter(value=350)
    session = make_session(counter)
    session.cfg.context_fill_ratio = .6
    assert len(content(session)) == 200
    calls = len(counter.calls)
    session.cfg.max_tokens = session.cfg.context_size
    assert len(content(session)) == 1000
    assert len(counter.calls) == calls


@pytest.mark.parametrize("status", [200, 503])
def test_real_session_uses_prepared_request_with_mock_backend(tmp_path, status):
    import json
    import httpx
    from _config_helpers import make_config
    from scripts.llm_solver.harness.loop import Session
    from scripts.llm_solver.server.client import LlamaClient
    from test_backend_token_counting import attach_transport

    cfg = make_config(tokenizer_id="auto", context_size=4000, max_tokens=1000,
                      context_fill_ratio=.9, model="observed-model")
    client = LlamaClient(cfg)
    requests = []

    def handler(request):
        assert request.url.path.endswith("/input_tokens")
        requests.append(json.loads(request.content))
        return httpx.Response(status, json={"input_tokens": 1600})

    attach_transport(client, handler)
    ctx = HalfLifeContext(context_size=4000, verbatim_tool_results=0, cap_7_chars=200)
    session = Session(cfg, client, "system", "task", str(tmp_path), context_manager=ctx)
    source = "HEAD" + "x" * 5992 + "TAIL"
    ctx.add_assistant({"role": "assistant", "content": None, "tool_calls": [
        {"id": "read-1", "type": "function", "function": {
            "name": "read", "arguments": '{"path":"x"}'}}]})
    ctx.add_tool_result("read-1", source)
    try:
        assert len(ctx.get_messages()[-1]["content"]) == (200 if status == 200 else 6000)
        assert requests[-1]["messages"][-1]["content"] == source
        assert requests[-1]["tools"] == session.model_tool_schemas
        assert requests[-1]["model"] == "observed-model"
        assert ctx.snapshot_messages()[-1]["content"] == source
        if status == 200:
            assert ctx._retention_count_evidence["request_sha256"]
            assert ctx._retention_count_evidence["prompt_tokens"] == 1600
    finally:
        client.client.close()
