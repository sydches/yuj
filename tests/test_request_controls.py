"""Focused tests for per-request reasoning and cache policy helpers."""
import logging

import pytest

from scripts.llm_solver.server.request_controls import (
    THINKING_LEVELS,
    RequestControlError,
    attach_request_extra,
    resolve_thinking_level,
    validate_reasoning_levels,
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
