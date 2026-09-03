"""Focused tests for per-request reasoning and cache policy helpers."""
import json
import logging
from types import SimpleNamespace

import pytest

from scripts.llm_solver.server.request_controls import (
    CACHE_RETENTION_LEVELS,
    REQUEST_DIALECTS,
    CacheUsageAccumulator,
    THINKING_LEVELS,
    RequestControlError,
    apply_request_controls,
    attach_request_extra,
    derive_cache_slot,
    extract_cache_observation,
    normalize_request_dialect,
    resolve_cache_request,
    resolve_thinking_level,
    validate_cache_miss_warn_ratio,
    validate_reasoning_levels,
    warn_on_cache_miss,
)


def test_thinking_ladder_is_the_public_seven_level_order():
    assert THINKING_LEVELS == (
        "off",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    )


def test_exact_thinking_level_maps_to_profile_request_body():
    levels = {
        "off": {"chat_template_kwargs": {"enable_thinking": False}},
        "high": {"reasoning_effort": "high", "thinking_budget": 8192},
    }

    resolution = resolve_thinking_level("high", levels)

    assert resolution.effective_level == "high"
    assert resolution.clamped is False
    assert resolution.request_extra == {
        "reasoning_effort": "high",
        "thinking_budget": 8192,
    }
    assert resolution.trace_fields() == {"thinking_level": "high"}
    assert resolution.provenance_fields() == {
        "thinking_level_requested": "high",
        "thinking_level_effective": "high",
        "thinking_level_clamped": False,
    }


def test_generic_on_profile_clamps_positive_effort_and_logs_warning(caplog):
    levels = {
        "off": {"enable_thinking": False},
        "on": {"enable_thinking": True},
    }

    with caplog.at_level(logging.WARNING):
        resolution = resolve_thinking_level("xhigh", levels)

    assert resolution.effective_level == "on"
    assert resolution.request_extra == {"enable_thinking": True}
    assert resolution.trace_fields() == {
        "thinking_level": "on",
        "thinking_level_requested": "xhigh",
    }
    assert "thinking level xhigh is unsupported" in caplog.text


def test_clamp_prefers_nearest_supported_effort_without_overspending():
    levels = {
        "off": {"enable_thinking": False},
        "low": {"reasoning_effort": "low"},
        "high": {"reasoning_effort": "high"},
    }

    assert resolve_thinking_level("medium", levels).effective_level == "low"
    assert resolve_thinking_level("max", levels).effective_level == "high"
    # No positive level exists at or below minimal, so keep thinking enabled
    # with the lowest positive capability instead of silently choosing off.
    assert resolve_thinking_level("minimal", levels).effective_level == "low"


def test_off_clamps_to_lowest_supported_level_when_profile_cannot_disable():
    resolution = resolve_thinking_level(
        "off",
        {"medium": {"reasoning_effort": "medium"}},
    )
    assert resolution.effective_level == "medium"
    assert resolution.clamped is True


def test_request_extra_is_attached_without_mutating_profile_or_payload():
    levels = {"high": {"chat_template_kwargs": {"enable_thinking": True}}}
    resolution = resolve_thinking_level("high", levels)
    payload = {"model": "served-model", "messages": []}

    sdk_kwargs = attach_request_extra(payload, resolution.request_extra)
    sdk_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] = False

    assert payload == {"model": "served-model", "messages": []}
    assert levels["high"]["chat_template_kwargs"]["enable_thinking"] is True


@pytest.mark.parametrize(
    ("levels", "match"),
    [
        ({}, "non-empty"),
        ({"turbo": {"reasoning_effort": "high"}}, "unknown profile reasoning"),
        ({"off": {}}, "at least one request field"),
        ({"off": {"model": "replacement"}}, "transport-owned"),
        ({"off": {"thinking_budget": float("nan")}}, "finite JSON number"),
        ({"off": {"chat_template_kwargs": object()}}, "JSON-compatible"),
    ],
)
def test_reasoning_level_profile_validation_rejects_invalid_maps(levels, match):
    with pytest.raises(RequestControlError, match=match):
        validate_reasoning_levels(levels)


def test_user_cannot_request_profile_only_on_alias():
    with pytest.raises(RequestControlError, match="must be one of"):
        resolve_thinking_level("on", {"on": {"enable_thinking": True}})


def test_cache_request_pins_session_to_slot_and_enables_retention():
    resolution = resolve_cache_request(
        session_id="session-018f",
        cache_affinity=8,
        cache_retention="session",
    )

    assert 0 <= resolution.slot_id < 8
    assert resolution.slot_id == derive_cache_slot("session-018f", 8)
    assert resolution.request_extra == {
        "cache_prompt": True,
        "id_slot": resolution.slot_id,
    }
    assert resolution.trace_fields() == {
        "cache_retention": "session",
        "cache_prompt": True,
        "cache_slot": resolution.slot_id,
    }


def test_cache_affinity_hash_is_stable_and_uses_configured_slot_range():
    first = [derive_cache_slot(f"session-{index}", 16) for index in range(64)]
    second = [derive_cache_slot(f"session-{index}", 16) for index in range(64)]

    assert first == second
    assert all(slot is not None and 0 <= slot < 16 for slot in first)
    assert len(set(first)) > 1
    assert derive_cache_slot("single-slot", True) == 0
    assert derive_cache_slot("disabled", False) is None


def test_side_request_disables_cache_and_does_not_claim_main_session_slot():
    resolution = resolve_cache_request(
        session_id="session-main",
        cache_affinity=4,
        cache_retention="session",
        side_request=True,
    )

    assert resolution.slot_id is None
    assert resolution.request_extra == {"cache_prompt": False}
    assert resolution.trace_fields() == {
        "cache_retention": "session",
        "cache_prompt": False,
        "cache_side_request": True,
    }


def test_fake_openai_client_receives_llama_cache_fields_in_request_body():
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)

    cache = resolve_cache_request(
        session_id="session-wire",
        cache_affinity=2,
        cache_retention="session",
    )
    payload = attach_request_extra(
        {"model": "served", "messages": [], "max_tokens": 32},
        cache.request_extra,
    )

    FakeCompletions().create(**payload)

    assert calls[0]["extra_body"] == {
        "cache_prompt": True,
        "id_slot": derive_cache_slot("session-wire", 2),
    }


def test_openai_dialect_strips_llama_fields_from_every_extra_layer():
    payload = {
        "model": "served",
        "messages": [],
        "extra_body": {
            "cache_prompt": True,
            "id_slot": 1,
            "chat_template_kwargs": {"enable_thinking": True},
            "payload_field": "kept",
        },
    }

    request = apply_request_controls(
        payload,
        session_id="session-wire",
        server_request_extra={
            "cache_prompt": True,
            "id_slot": 2,
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_effort": "low",
        },
        cache_affinity=4,
        cache_retention="session",
        side_request=False,
        policy_extra={
            "chat_template_kwargs": {"enable_thinking": False},
            "policy_field": "kept",
        },
        request_dialect="openai",
    )

    assert request["extra_body"] == {
        "payload_field": "kept",
        "reasoning_effort": "low",
        "policy_field": "kept",
    }
    assert payload["extra_body"]["cache_prompt"] is True


def test_openai_dialect_omits_empty_extra_body():
    request = apply_request_controls(
        {"model": "served", "messages": [], "extra_body": {}},
        session_id="",
        server_request_extra={},
        cache_affinity=False,
        cache_retention="session",
        side_request=True,
        policy_extra={"chat_template_kwargs": {"enable_thinking": False}},
        request_dialect="openai",
    )

    assert "extra_body" not in request


def test_extract_cache_observation_from_openai_usage_details():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=1000,
            prompt_tokens_details=SimpleNamespace(cached_tokens=750),
        )
    )

    observation = extract_cache_observation(response)

    assert observation is not None
    assert observation.prompt_tokens == 1000
    assert observation.cached_tokens == 750
    assert observation.hit_ratio == 0.75
    assert observation.source == "usage"
    assert observation.trace_fields() == {
        "prompt_tokens": 1000,
        "cached_tokens": 750,
        "cache_hit_ratio": 0.75,
    }


def test_extract_cache_observation_from_llama_timings_processed_suffix():
    observation = extract_cache_observation(
        {"timings": {"prompt_n": 250, "cache_n": 750}}
    )

    assert observation is not None
    assert observation.prompt_tokens == 1000
    assert observation.cached_tokens == 750
    assert observation.hit_ratio == 0.75
    assert observation.source == "timings"


def test_missing_cache_telemetry_stays_unknown_not_a_false_miss():
    observation = extract_cache_observation(
        {"usage": {"prompt_tokens": 1000, "completion_tokens": 10}}
    )

    assert observation is not None
    assert observation.cached_tokens is None
    assert observation.hit_ratio is None
    assert observation.trace_fields()["cached_tokens"] is None
    assert warn_on_cache_miss(observation, warn_ratio=0.8, prior_turns=2) is False


def test_low_cache_ratio_warns_only_after_first_turn(caplog):
    observation = extract_cache_observation(
        {"usage": {"prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 100}}}
    )
    assert observation is not None

    with caplog.at_level(logging.WARNING):
        assert warn_on_cache_miss(
            observation,
            warn_ratio=0.5,
            prior_turns=0,
        ) is False
        assert warn_on_cache_miss(
            observation,
            warn_ratio=0.5,
            prior_turns=1,
        ) is True

    assert caplog.text.count("prompt cache hit ratio") == 1


def test_session_cache_ratio_is_weighted_by_prompt_tokens():
    accumulator = CacheUsageAccumulator()
    accumulator.record(
        extract_cache_observation(
            {"timings": {"prompt_n": 100, "cache_n": 900}}
        )
    )
    accumulator.record(
        extract_cache_observation(
            {"timings": {"prompt_n": 100, "cache_n": 0}}
        )
    )
    accumulator.record(
        extract_cache_observation({"usage": {"prompt_tokens": 50}})
    )

    assert accumulator.metrics_fields() == {
        "prompt_cache": {
            "prompt_tokens": 1100,
            "cached_tokens": 900,
            "cache_hit_ratio": 0.818182,
            "requests_observed": 2,
            "requests_unobserved": 1,
        }
    }


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"session_id": "s", "cache_affinity": -1, "cache_retention": "session"},
            "cache_affinity",
        ),
        (
            {"session_id": "", "cache_affinity": 2, "cache_retention": "session"},
            "session_id",
        ),
        (
            {"session_id": "s", "cache_affinity": 1, "cache_retention": "forever"},
            "cache_retention",
        ),
    ],
)
def test_invalid_cache_request_settings_fail_closed(kwargs, match):
    with pytest.raises(RequestControlError, match=match):
        resolve_cache_request(**kwargs)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("inf"), True, "0.5"])
def test_invalid_cache_warning_ratios_fail_closed(value):
    with pytest.raises(RequestControlError, match="between 0 and 1"):
        validate_cache_miss_warn_ratio(value)


def test_cache_retention_modes_are_explicit_and_llama_scoped():
    assert CACHE_RETENTION_LEVELS == ("off", "session")


def test_request_dialects_are_explicit_and_invalid_values_fail_closed():
    assert REQUEST_DIALECTS == ("llama", "openai")
    assert normalize_request_dialect(" OPENAI ") == "openai"
    with pytest.raises(RequestControlError, match="request_dialect"):
        normalize_request_dialect("google")


def test_default_halflife_model_prefix_is_byte_identical_across_turns():
    from scripts.llm_solver.harness.context_strategies.halflife_context import (
        HalfLifeContext,
    )

    context = HalfLifeContext(context_size=40960)
    context.add_system("stable system")
    context.add_user("stable task")
    context.add_assistant(
        {
            "role": "assistant",
            "content": "inspect",
            "tool_calls": [
                {
                    "id": "call_0_0",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"a.py"}'},
                }
            ],
        }
    )
    context.add_tool_result("call_0_0", "first result")
    prefix_message_count = 4
    first_prefix = json.dumps(
        context.get_messages()[:prefix_message_count],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    context.add_assistant(
        {
            "role": "assistant",
            "content": "edit next",
            "tool_calls": [],
        }
    )
    context.add_tool_result("call_1_0", "second result")
    second_prefix = json.dumps(
        context.get_messages()[:prefix_message_count],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    assert second_prefix == first_prefix
