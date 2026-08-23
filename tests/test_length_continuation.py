"""Focused contract tests for overlap-safe length continuation."""
from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from typing import Any

import pytest

from scripts.llm_solver.harness._loop.length_continuation import (
    FALLBACK_ATTEMPT_LIMIT,
    FALLBACK_DISABLED,
    FALLBACK_NO_PARTIAL_CONTENT,
    FALLBACK_PREFILL_UNSUPPORTED,
    build_length_continuation_request,
    continue_length_response,
    exact_overlap_length,
    join_exact_overlap,
)
from scripts.llm_solver.server.types import Usage


class FakeSplitClient:
    """Return raw split outputs while retaining exact request snapshots."""

    def __init__(self, responses: list[Mapping[str, Any]]):
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def chat(self, request: dict[str, Any]) -> Mapping[str, Any]:
        self.requests.append(copy.deepcopy(request))
        if not self._responses:
            raise AssertionError("unexpected continuation request")
        return self._responses.pop(0)


def _base_request() -> dict[str, Any]:
    return {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Read the configuration guide."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 4_096,
        "temperature": 0,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
            "cache_prompt": True,
        },
    }


def _raw(
    content: str | None,
    finish_reason: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict[str, Any]:
    return {
        "content": content,
        "tool_calls": [],
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


@pytest.mark.parametrize(
    ("existing", "continuation", "overlap", "joined"),
    [
        ("", "new", 0, "new"),
        ("old", "", 0, "old"),
        ("alpha", "beta", 0, "alphabeta"),
        ("alpha beta", "beta gamma", 4, "alpha beta gamma"),
        ("aaaa", "aaab", 3, "aaaab"),
        ("same", "same", 4, "same"),
        ("Case", "case", 0, "Casecase"),
        ("line\n", "\nnext", 1, "line\nnext"),
    ],
)
def test_exact_overlap_join_is_longest_and_content_preserving(
    existing: str,
    continuation: str,
    overlap: int,
    joined: str,
) -> None:
    assert exact_overlap_length(existing, continuation) == overlap
    assert join_exact_overlap(existing, continuation) == joined


def test_overlap_join_handles_long_repetitive_text_without_guessing() -> None:
    existing = ("ab" * 20_000) + "XYZ"
    continuation = "XYZ" + ("cd" * 20_000)

    assert exact_overlap_length(existing, continuation) == 3
    assert join_exact_overlap(existing, continuation) == (
        existing + continuation[3:]
    )


def test_follow_up_request_preserves_base_and_adds_assistant_prefill() -> None:
    base = _base_request()
    original = copy.deepcopy(base)

    request = build_length_continuation_request(
        base,
        "partial assistant output",
    )

    assert base == original
    assert request is not base
    assert request["messages"][:-1] == original["messages"]
    assert request["messages"][-1] == {
        "role": "assistant",
        "content": "partial assistant output",
    }
    assert request["model"] == original["model"]
    assert request["tools"] == original["tools"]
    assert request["tool_choice"] == original["tool_choice"]
    assert request["max_tokens"] == original["max_tokens"]
    assert request["temperature"] == original["temperature"]
    assert request["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_prompt": True,
        "continue_final_message": True,
        "add_generation_prompt": False,
    }


def test_continuation_template_controls_override_conflicting_extras() -> None:
    base = _base_request()
    base["extra_body"].update({
        "continue_final_message": False,
        "add_generation_prompt": True,
    })

    request = build_length_continuation_request(base, "partial")

    assert request["extra_body"]["continue_final_message"] is True
    assert request["extra_body"]["add_generation_prompt"] is False
    assert base["extra_body"]["continue_final_message"] is False
    assert base["extra_body"]["add_generation_prompt"] is True


def test_each_follow_up_uses_original_messages_plus_full_joined_prefill() -> None:
    base = _base_request()
    client = FakeSplitClient([
        _raw(
            "gamma delta END-",
            "length",
            prompt_tokens=30,
            completion_tokens=4,
        ),
        _raw(
            "END-POINT",
            "stop",
            prompt_tokens=34,
            completion_tokens=2,
        ),
    ])
    normalize_calls: list[dict[str, Any]] = []

    def normalize(response: dict[str, Any]) -> dict[str, Any]:
        normalize_calls.append(copy.deepcopy(response))
        return response

    result = continue_length_response(
        base_request=base,
        initial_response=_raw("alpha beta gamma", "length"),
        max_attempts=3,
        supports_prefill=True,
        call_model=client.chat,
        normalize=normalize,
    )

    assert len(client.requests) == 2
    assert client.requests[0]["messages"] == base["messages"] + [
        {"role": "assistant", "content": "alpha beta gamma"}
    ]
    assert client.requests[1]["messages"] == base["messages"] + [
        {
            "role": "assistant",
            "content": "alpha beta gamma delta END-",
        }
    ]
    assert result.joined_content == "alpha beta gamma delta END-POINT"
    assert result.response["content"] == result.joined_content
    assert result.fallback_reason is None
    assert result.exhausted is False
    assert result.continuation_count == 2
    assert result.continuation_tokens == 6
    assert result.continuation_prompt_tokens == 64
    assert [attempt.overlap_chars for attempt in result.attempts] == [5, 4]
    assert [attempt.attempt for attempt in result.attempts] == [1, 2]
    assert len(normalize_calls) == 1
    assert normalize_calls[0]["content"] == result.joined_content


def test_split_thinking_and_tool_call_normalize_once_after_join() -> None:
    initial = "<think>inspect the owner before acting</thi"
    client = FakeSplitClient([
        _raw(
            'nk>{"name":"read","arguments":{"path":"docs/conf',
            "length",
            prompt_tokens=72,
            completion_tokens=9,
        ),
        _raw(
            "configuration.md\"}}",
            "length",
            prompt_tokens=80,
            completion_tokens=8,
        )
    ])
    normalized_inputs: list[str] = []

    def profile_normalize(response: dict[str, Any]) -> dict[str, Any]:
        content = str(response.get("content") or "")
        normalized_inputs.append(content)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        parsed = json.loads(content)
        response["content"] = None
        response["tool_calls"] = [
            {
                "id": "",
                "type": "function",
                "function": {
                    "name": parsed["name"],
                    "arguments": json.dumps(parsed["arguments"]),
                },
            }
        ]
        response["finish_reason"] = "tool_calls"
        return response

    result = continue_length_response(
        base_request=_base_request(),
        initial_response=_raw(initial, "length"),
        max_attempts=2,
        supports_prefill=True,
        call_model=client.chat,
        normalize=profile_normalize,
    )

    assert len(normalized_inputs) == 1
    assert normalized_inputs[0].startswith("<think>")
    assert normalized_inputs[0].endswith(
        '{"name":"read","arguments":{"path":"docs/configuration.md"}}'
    )
    assert result.response["finish_reason"] == "tool_calls"
    tool_call = result.response["tool_calls"][0]["function"]
    assert tool_call["name"] == "read"
    assert json.loads(tool_call["arguments"]) == {
        "path": "docs/configuration.md"
    }
    # The raw server still reported length, but the joined normalize pass
    # recovered a complete call, so the session must not take length fallback.
    assert result.raw_response["finish_reason"] == "length"
    assert result.fallback_reason is None
    assert result.exhausted is False


def test_attempt_limit_keeps_incomplete_join_for_existing_length_fallback() -> None:
    client = FakeSplitClient([
        _raw("two ", "length", completion_tokens=2),
        _raw("three", "length", completion_tokens=1),
    ])
    normalize_calls = 0

    def normalize(response: dict[str, Any]) -> dict[str, Any]:
        nonlocal normalize_calls
        normalize_calls += 1
        return response

    result = continue_length_response(
        base_request=_base_request(),
        initial_response=_raw("one ", "length"),
        max_attempts=2,
        supports_prefill=True,
        call_model=client.chat,
        normalize=normalize,
    )

    assert normalize_calls == 1
    assert result.joined_content == "one two three"
    assert result.response["finish_reason"] == "length"
    assert result.response["tool_calls"] == []
    assert result.fallback_reason == FALLBACK_ATTEMPT_LIMIT
    assert result.exhausted is True
    assert result.continuation_count == 2
    assert result.continuation_tokens == 3


@pytest.mark.parametrize(
    ("max_attempts", "supports_prefill", "content", "fallback"),
    [
        (0, True, "partial", FALLBACK_DISABLED),
        (2, False, "partial", FALLBACK_PREFILL_UNSUPPORTED),
        (2, True, None, FALLBACK_NO_PARTIAL_CONTENT),
        (2, True, "", FALLBACK_NO_PARTIAL_CONTENT),
    ],
)
def test_off_unsupported_and_missing_partial_make_no_follow_up(
    max_attempts: int,
    supports_prefill: bool,
    content: str | None,
    fallback: str,
) -> None:
    client = FakeSplitClient([])
    normalize_calls = 0

    def normalize(response: dict[str, Any]) -> dict[str, Any]:
        nonlocal normalize_calls
        normalize_calls += 1
        return response

    result = continue_length_response(
        base_request=_base_request(),
        initial_response=_raw(content, "length"),
        max_attempts=max_attempts,
        supports_prefill=supports_prefill,
        call_model=client.chat,
        normalize=normalize,
    )

    assert client.requests == []
    assert normalize_calls == 1
    assert result.attempts == ()
    assert result.fallback_reason == fallback
    assert result.exhausted is False


def test_non_length_response_is_only_normalized_and_never_re_requested() -> None:
    client = FakeSplitClient([])
    seen: list[dict[str, Any]] = []

    result = continue_length_response(
        base_request=_base_request(),
        initial_response=_raw("complete", "stop"),
        max_attempts=4,
        supports_prefill=True,
        call_model=client.chat,
        normalize=lambda response: seen.append(response) or response,
    )

    assert client.requests == []
    assert len(seen) == 1
    assert result.response["content"] == "complete"
    assert result.fallback_reason is None


def test_usage_objects_are_recorded_for_trace_and_metrics_accounting() -> None:
    client = FakeSplitClient([
        {
            "content": " done",
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": Usage(prompt_tokens=120, completion_tokens=7),
        }
    ])

    result = continue_length_response(
        base_request=_base_request(),
        initial_response=_raw("nearly", "length"),
        max_attempts=1,
        supports_prefill=True,
        call_model=client.chat,
        normalize=lambda response: response,
    )

    assert result.attempts[0].tokens == 7
    assert result.attempts[0].prompt_tokens == 120
    assert result.continuation_tokens == 7
    assert result.continuation_prompt_tokens == 120


def test_invalid_contract_inputs_fail_before_model_call() -> None:
    client = FakeSplitClient([])

    with pytest.raises(TypeError, match="max_attempts"):
        continue_length_response(
            base_request=_base_request(),
            initial_response=_raw("partial", "length"),
            max_attempts=True,
            supports_prefill=True,
            call_model=client.chat,
            normalize=lambda response: response,
        )
    with pytest.raises(ValueError, match="must not be negative"):
        continue_length_response(
            base_request=_base_request(),
            initial_response=_raw("partial", "length"),
            max_attempts=-1,
            supports_prefill=True,
            call_model=client.chat,
            normalize=lambda response: response,
        )
    with pytest.raises(ValueError, match="messages must be a list"):
        build_length_continuation_request(
            {"messages": "not-a-list"},
            "partial",
        )
    with pytest.raises(ValueError, match="extra_body must be a mapping"):
        build_length_continuation_request(
            {"messages": [], "extra_body": "bad"},
            "partial",
        )
    assert client.requests == []
