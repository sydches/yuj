"""Capacity constrains caller permission across startup and replacement."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests._config_helpers import make_config
from scripts.llm_solver.context_allocation import allocate_context
from scripts.llm_solver.harness._loop.compaction import get_server_ctx
from scripts.llm_solver.harness._loop.model_role_runtime import _target_config
from scripts.llm_solver.harness._loop.model_roles import ModelRoleResolver, ModelTarget
from scripts.llm_solver.server.profile_loader import load_profile


@pytest.mark.parametrize("declared, observed, expected", [
    (1024, 8192, 1024), (8192, 1024, 1024), (1024, None, 1024),
    (1024, True, 1024), (1024, -1, 1024), (1024, "8192", 1024),
])
def test_capacity_cannot_grant_permission(declared, observed, expected):
    cfg = make_config(context_size=declared, max_tokens=0)
    effective = allocate_context(cfg, observed=observed)
    assert effective.context_size == expected
    assert effective.context_allocation["declared_context"] == declared
    assert effective.context_allocation["observed_capacity"] == (
        observed if type(observed) is int and observed > 0 else None)
    assert effective.max_tokens == int(expected * cfg.max_tokens_fraction)
    assert cfg.context_size == declared and cfg.context_allocation is None


def test_smaller_capacity_does_not_destroy_the_original_declaration_or_output_cap():
    cfg = make_config(context_size=8192, max_tokens=700)
    small = allocate_context(cfg, observed=1024)
    large = allocate_context(replace(small, model="replacement"), observed=16384)
    assert large.context_size == 8192 and large.max_tokens == 700
    assert large.context_allocation["declared_context"] == 8192
    assert large.context_allocation["model"] == "replacement"


def test_role_policy_is_distinct_from_inherited_and_profile_capacity():
    profile = SimpleNamespace(context_capacity=16384, context_capacity_source="fixture")
    resolver = ModelRoleResolver(
        main_target=ModelTarget("main", "main", "http://fixture/v1", context_size=8192),
        role_specs={"inherited": "large", "explicit": {"profile": "large", "context_size": 12000}},
        profile_loader=lambda _: profile,
    )
    narrowed = allocate_context(make_config(context_size=8192, max_tokens=0), observed=1024)
    inherited = _target_config(narrowed, resolver.resolve("inherited"))
    explicit = _target_config(narrowed, resolver.resolve("explicit"))
    assert inherited.context_size == 8192
    assert inherited.context_allocation["declaration_source"] == "inherited_caller"
    assert explicit.context_size == 12000
    assert explicit.context_allocation["declaration_source"] == "explicit_role"
    assert inherited.context_allocation["observed_capacity"] is None
    assert explicit.context_allocation["profile_capacity"] == 16384


def test_profile_loader_default_is_not_capacity_and_inherited_source_is_retained(tmp_path):
    base = tmp_path / "base"
    child = tmp_path / "child"
    base.mkdir()
    child.mkdir()
    base_text = '[profile]\nname="base"\n[reasoning_levels.off]\nchat_template_kwargs={enable_thinking=false}\n'
    (base / "profile.toml").write_text(base_text)
    (child / "profile.toml").write_text('[profile]\nname="child"\ninherits="base"\n')
    profile = load_profile("child", tmp_path)
    assert profile.context_size == 40960 and profile.context_capacity is None
    assert allocate_context(make_config(context_size=60000), profile=profile).context_size == 60000
    (base / "profile.toml").write_text(base_text + '[model]\ncontext_size=2000\n')
    profile = load_profile("child", tmp_path)
    assert profile.context_capacity == 2000
    assert profile.context_capacity_source == str(base / "profile.toml")
    assert allocate_context(make_config(context_size=60000), profile=profile).context_size == 2000


def test_compaction_uses_shared_allocation_and_does_not_requery_bound_startup():
    cfg = allocate_context(make_config(context_size=1024), observed=8192)
    client = SimpleNamespace(cfg=cfg, profile=None, query_server_context=Mock(side_effect=AssertionError))
    session = SimpleNamespace(cfg=cfg, client=client, _emit=Mock())
    assert get_server_ctx(session) == 1024
    assert get_server_ctx(session) == 1024
    assert client.cfg is session.cfg
    assert session._emit.call_count == 1


@pytest.mark.parametrize("declared, observed", [(1024, 8192), (8192, 1024), (1024, None)])
@pytest.mark.parametrize("ratio, boundary", [(0.95, 972), (0.5, 512)])
@pytest.mark.parametrize("excess", [0, 1])
def test_compaction_ratio_uses_effective_permission_at_the_boundary(
    tmp_path, declared, observed, ratio, boundary, excess,
):
    from tests.test_session_compaction import _make_session, _seed_trace_jsonl

    _seed_trace_jsonl(tmp_path, 30)
    session = _make_session(
        tmp_path, [{"event": "tool_call", "tool_name": "write"}],
        cfg_extra={"context_size": declared, "context_fill_ratio": ratio},
    )
    session.client.query_server_context = Mock(return_value=observed)
    messages = [{"role": "system", "content": "system"},
                {"role": "user", "content": "task"}]
    counter = SimpleNamespace(
        count=Mock(side_effect=lambda value, **kwargs: boundary + excess if value == messages else 100),
        last={"count_basis": "backend_input_tokens"},
    )
    session._tokenizer = counter

    session._maybe_compact_messages(messages)

    assert session.cfg.context_size == 1024
    assert session.cfg.context_fill_ratio == ratio
    assert session.cfg.context_allocation["declared_context"] == declared
    assert session.cfg.context_allocation["observed_capacity"] == observed
    assert session.cfg.context_allocation["prompt_limit"] == boundary
    assert session._compacted is bool(excess)
    assert counter.count.call_args_list[0].args[0] == messages
    session.client.query_server_context.assert_called_once_with()


@pytest.mark.parametrize("same_endpoint", [False, True])
def test_client_replacement_invalidates_capacity_and_count_observations(monkeypatch, same_endpoint):
    from scripts.llm_solver.harness import request_counting
    reset = Mock()
    monkeypatch.setattr(request_counting, "bind_session_counter", reset)
    cfg = make_config(context_size=8192)
    first = SimpleNamespace(profile=None, query_server_context=Mock(return_value=1024))
    session = SimpleNamespace(cfg=cfg, client=first)
    assert get_server_ctx(session) == 1024
    # Preserve the declaration but change the configured model and endpoint.
    if not same_endpoint:
        session.cfg = replace(session.cfg, model="other", base_url="http://other/v1")
    session.client = SimpleNamespace(profile=None, query_server_context=Mock(return_value=16384))
    assert get_server_ctx(session) == 8192
    assert session.cfg.context_allocation["observed_capacity"] == 16384
    reset.assert_called_once_with(session, reset_observations=True)


@pytest.mark.parametrize("props, slots, expected", [
    ({"n_ctx_per_slot": 2048, "default_generation_settings": {"n_ctx": 8192}}, [], 2048),
    ({"default_generation_settings": {"n_ctx": True}}, [{"n_ctx": 4096}], 4096),
    ({}, [{"n_ctx": 4096}, {"n_ctx": 2048}], 2048),
    ({}, [{"n_ctx": 4096}, {"n_ctx": None}], None),
    ({"default_generation_settings": {"n_ctx": -1}}, [{"n_ctx": "8192"}], None),
])
def test_native_capacity_rejects_invalid_values_and_respects_slot_limits(monkeypatch, props, slots, expected):
    from scripts.llm_solver.server.client import LlamaClient
    import requests
    client = SimpleNamespace(_server_root=lambda: "http://fixture")
    monkeypatch.setattr(requests, "get", lambda url, **kw: SimpleNamespace(
        ok=True, json=lambda: props if url.endswith("/props") else slots))
    assert LlamaClient.query_server_context(client) == expected


def test_assistant_startup_uses_same_rule():
    from scripts.llm_assist.runner import _apply_effective_context
    cfg = make_config(context_size=1024, max_tokens=0)
    client = SimpleNamespace(profile=None, query_server_context=lambda: 8192)
    assert _apply_effective_context(cfg, client).context_allocation == allocate_context(cfg, observed=8192).context_allocation


def test_session_preflight_uses_the_newly_constrained_config(tmp_path):
    from unittest.mock import MagicMock
    from scripts.llm_solver.harness.loop import Session
    cfg = make_config(context_size=8192, max_turns=1,
        preflight_reclip_enabled=False, digest_compaction_safety_margin=-10.0)
    client = MagicMock()
    client.query_server_context.return_value = 1024
    client.chat.side_effect = AssertionError("oversized prompt reached model")
    session = Session(cfg, client, "system", "x" * 8192, str(tmp_path))
    result = session.run()
    assert result.finish_reason == "context_full"
    assert client.chat.call_count == 0
    assert session.cfg.context_size == 1024
    assert session.cfg.context_allocation["declared_context"] == 8192
