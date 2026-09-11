"""Capacity observations must not rewrite declared character policy."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from _config_helpers import make_config
from scripts.llm_solver.context_allocation import allocate_context
from scripts.llm_solver.harness._loop.model_fallback_runtime import _apply_context_size
from scripts.llm_solver.harness._loop.model_role_runtime import _target_config
from scripts.llm_solver.harness._loop.model_roles import ModelRoleResolver, ModelTarget
from scripts.llm_assist.runner import _apply_effective_context


def _config():
    return make_config(context_size=8192, max_output_chars=567, recent_tool_results_chars=1234)


def _assert_policy(cfg):
    assert cfg.max_output_chars == 567
    assert cfg.recent_tool_results_chars == 1234
    assert cfg.context_allocation["output_character_limits"] == {
        "max_output_chars": 567, "recent_tool_results_chars": 1234,
        "basis": "resolved_output_policy",
    }


@pytest.mark.parametrize("observed", [1024, 8192, 16384, None])
@pytest.mark.parametrize("entry", ["allocation", "assistant", "fallback"])
def test_shared_entries_preserve_policy_at_each_capacity(observed, entry):
    cfg = _config()
    client = SimpleNamespace(cfg=cfg, profile=None, query_server_context=lambda: observed)
    if entry == "allocation":
        effective = allocate_context(cfg, observed=observed)
    elif entry == "assistant":
        effective = _apply_effective_context(cfg, client)
    else:
        _apply_context_size(client, observed)
        effective = client.cfg
    _assert_policy(effective)
    assert effective.context_size == min(8192, observed or 8192)
    assert cfg.context_allocation is None


def test_role_change_and_repeated_capacity_binding_preserve_character_limits():
    profile = SimpleNamespace(context_capacity=16384, context_capacity_source="fixture")
    resolver = ModelRoleResolver(
        main_target=ModelTarget("main", "main", "http://fixture/v1", context_size=8192),
        role_specs={"secondary": "replacement"}, profile_loader=lambda _: profile,
    )
    cfg = allocate_context(_config(), observed=1024)
    replacement = _target_config(cfg, resolver.resolve("secondary"))
    _assert_policy(replacement)
    client = SimpleNamespace(cfg=replacement, profile=profile)
    for capacity in (512, 16384, 8192):
        _apply_context_size(client, capacity)
        _assert_policy(client.cfg)


def test_explicit_policy_revision_is_not_overruled_by_an_old_allocation_record():
    cfg = allocate_context(_config(), observed=1024)
    revised = replace(cfg, max_output_chars=111, recent_tool_results_chars=222)
    effective = allocate_context(revised, observed=16384)
    assert effective.max_output_chars == 111
    assert effective.recent_tool_results_chars == 222
    assert effective.context_allocation["output_character_limits"]["max_output_chars"] == 111
