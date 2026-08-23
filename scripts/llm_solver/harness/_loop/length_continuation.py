"""Raw-response kernel for overlap-safe length continuation.

This leaf owns only continuation request construction, exact text joining, and
the normalize-once boundary.  It does not call the trace writer, update run
metrics, or decide whether a session rolls over.  The loop integration passes
it raw (pre-normalize) response mappings and keeps the existing length-session
fallback whenever :class:`LengthContinuationResult` reports one.

The ``base_request`` is the exact first-call payload after profile
denormalization and ordinary request controls.  Every follow-up reuses that
payload, appends one assistant prefill containing the complete joined partial,
and applies the two template controls needed to continue the final message.
"""
from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


LENGTH_FINISH_REASON = "length"

FALLBACK_DISABLED = "disabled"
FALLBACK_PREFILL_UNSUPPORTED = "prefill_unsupported"
FALLBACK_NO_PARTIAL_CONTENT = "no_partial_content"
FALLBACK_ATTEMPT_LIMIT = "attempt_limit"

_CONTINUATION_REQUEST_EXTRA: Mapping[str, object] = {
    "continue_final_message": True,
    "add_generation_prompt": False,
}

RawResponse = Mapping[str, Any]
ContinuationCall = Callable[[dict[str, Any]], RawResponse]
NormalizeCall = Callable[[dict[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class LengthContinuationAttempt:
    """Measured facts for one follow-up model request.

    ``tokens`` is the server-reported completion-token count for this
    follow-up.  It is the value intended for the ``length_continue`` trace
    event.  ``prompt_tokens`` is retained separately for aggregate accounting.
    Missing usage telemetry is represented by zero.
    """

    attempt: int
    tokens: int
    prompt_tokens: int
    overlap_chars: int
    received_chars: int
    finish_reason: str


@dataclass(frozen=True)
class LengthContinuationResult:
    """Normalized result plus bounded continuation telemetry."""

    response: Mapping[str, Any]
    raw_response: Mapping[str, Any]
    joined_content: str | None
    attempts: tuple[LengthContinuationAttempt, ...]
    fallback_reason: str | None
    exhausted: bool

    @property
    def continuation_count(self) -> int:
        return len(self.attempts)

    @property
    def continuation_tokens(self) -> int:
        return sum(attempt.tokens for attempt in self.attempts)

    @property
    def continuation_prompt_tokens(self) -> int:
        return sum(attempt.prompt_tokens for attempt in self.attempts)


def exact_overlap_length(existing: str, continuation: str) -> int:
    """Return the longest exact suffix/prefix overlap in linear time.

    Matching is deliberately byte-for-code-point exact: no whitespace,
    newline, case, or Unicode normalization is performed.  That prevents a
    heuristic join from silently changing model output.  The prefix-function
    implementation avoids quadratic behavior on repetitive long responses.
    """
    if not isinstance(existing, str) or not isinstance(continuation, str):
        raise TypeError("overlap inputs must be strings")
    limit = min(len(existing), len(continuation))
    if limit == 0:
        return 0

    pattern = continuation[:limit]
    prefix = [0] * len(pattern)
    matched = 0
    for index in range(1, len(pattern)):
        while matched and pattern[index] != pattern[matched]:
            matched = prefix[matched - 1]
        if pattern[index] == pattern[matched]:
            matched += 1
        prefix[index] = matched

    matched = 0
    for char in existing[-limit:]:
        while matched and char != pattern[matched]:
            matched = prefix[matched - 1]
        if char == pattern[matched]:
            matched += 1
    return matched


def join_exact_overlap(existing: str, continuation: str) -> str:
    """Append ``continuation`` after trimming only its exact repeated prefix."""
    overlap = exact_overlap_length(existing, continuation)
    return existing + continuation[overlap:]


def build_length_continuation_request(
    base_request: Mapping[str, Any],
    partial_content: str,
) -> dict[str, Any]:
    """Return one non-mutating assistant-prefill follow-up request.

    All first-call fields (model, tools, sampling, token cap, cache controls,
    and provider extras) are preserved.  Continuation-owned template flags
    override conflicting caller values because generating a new assistant
    message would violate the continuation contract.
    """
    if not isinstance(base_request, Mapping):
        raise TypeError("base_request must be a mapping")
    if not isinstance(partial_content, str):
        raise TypeError("partial_content must be a string")
    if not partial_content:
        raise ValueError("partial_content must not be empty")

    request = copy.deepcopy(dict(base_request))
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise ValueError("base_request.messages must be a list")
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(
                f"base_request.messages[{index}] must be a mapping"
            )
    request["messages"] = [
        copy.deepcopy(dict(message)) for message in messages
    ]
    request["messages"].append(
        {"role": "assistant", "content": partial_content}
    )

    existing_extra = request.get("extra_body")
    if existing_extra is None:
        extra_body: dict[str, Any] = {}
    elif isinstance(existing_extra, Mapping):
        extra_body = copy.deepcopy(dict(existing_extra))
    else:
        raise ValueError("base_request.extra_body must be a mapping")
    extra_body.update(_CONTINUATION_REQUEST_EXTRA)
    request["extra_body"] = extra_body
    return request


def continue_length_response(
    *,
    base_request: Mapping[str, Any],
    initial_response: RawResponse,
    max_attempts: int,
    supports_prefill: bool,
    call_model: ContinuationCall,
    normalize: NormalizeCall,
) -> LengthContinuationResult:
    """Join bounded raw continuations, then invoke ``normalize`` once.

    ``call_model`` receives complete HTTP/SDK request payloads and must return
    raw response mappings with ``content`` and ``finish_reason`` fields.  The
    callback may raise; transport retry/fallback policy remains owned by the
    chat loop.  ``normalize`` receives only the final joined raw mapping.

    A false capability flag, a zero attempt limit, or missing partial text
    makes no follow-up call and leaves the raw length response available for
    the existing fresh-session fallback.  If the last response still reports
    ``length``, a truthy normalized ``tool_calls`` value counts as recovered;
    otherwise the attempt limit is exhausted.
    """
    _validate_attempt_limit(max_attempts)
    if not isinstance(supports_prefill, bool):
        raise TypeError("supports_prefill must be a bool")
    if not callable(call_model):
        raise TypeError("call_model must be callable")
    if not callable(normalize):
        raise TypeError("normalize must be callable")

    current = _copy_response(initial_response, field="initial_response")
    initial_reason = _finish_reason(current)
    joined_content = _content(current)
    attempts: list[LengthContinuationAttempt] = []
    fallback_reason: str | None = None

    if initial_reason == LENGTH_FINISH_REASON:
        if max_attempts == 0:
            fallback_reason = FALLBACK_DISABLED
        elif not supports_prefill:
            fallback_reason = FALLBACK_PREFILL_UNSUPPORTED
        elif not joined_content:
            fallback_reason = FALLBACK_NO_PARTIAL_CONTENT
        else:
            for attempt_number in range(1, max_attempts + 1):
                request = build_length_continuation_request(
                    base_request,
                    joined_content,
                )
                received = _copy_response(
                    call_model(request),
                    field=f"continuation response {attempt_number}",
                )
                piece = _content(received) or ""
                overlap = exact_overlap_length(joined_content, piece)
                joined_content = joined_content + piece[overlap:]
                received["content"] = joined_content
                current = received
                prompt_tokens, completion_tokens = _usage_tokens(received)
                reason = _finish_reason(received)
                attempts.append(
                    LengthContinuationAttempt(
                        attempt=attempt_number,
                        tokens=completion_tokens,
                        prompt_tokens=prompt_tokens,
                        overlap_chars=overlap,
                        received_chars=len(piece),
                        finish_reason=reason,
                    )
                )
                if reason != LENGTH_FINISH_REASON:
                    break

    candidate = copy.deepcopy(current)
    if joined_content is not None:
        candidate["content"] = joined_content
    normalized_value = normalize(candidate)
    if not isinstance(normalized_value, Mapping):
        raise TypeError("normalize must return a mapping")
    normalized = copy.deepcopy(dict(normalized_value))

    exhausted = False
    if initial_reason == LENGTH_FINISH_REASON:
        has_complete_tool_call = bool(normalized.get("tool_calls"))
        if has_complete_tool_call:
            fallback_reason = None
        elif attempts and _finish_reason(current) == LENGTH_FINISH_REASON:
            fallback_reason = FALLBACK_ATTEMPT_LIMIT
            exhausted = len(attempts) >= max_attempts

    return LengthContinuationResult(
        response=normalized,
        raw_response=copy.deepcopy(current),
        joined_content=joined_content,
        attempts=tuple(attempts),
        fallback_reason=fallback_reason,
        exhausted=exhausted,
    )


def _validate_attempt_limit(max_attempts: int) -> None:
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise TypeError("max_attempts must be an integer")
    if max_attempts < 0:
        raise ValueError("max_attempts must not be negative")


def _copy_response(response: RawResponse, *, field: str) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise TypeError(f"{field} must be a mapping")
    copied = copy.deepcopy(dict(response))
    _content(copied)
    _finish_reason(copied)
    return copied


def _content(response: Mapping[str, Any]) -> str | None:
    content = response.get("content")
    if content is not None and not isinstance(content, str):
        raise TypeError("response content must be a string or None")
    return content


def _finish_reason(response: Mapping[str, Any]) -> str:
    reason = response.get("finish_reason")
    if not isinstance(reason, str) or not reason:
        raise ValueError("response finish_reason must be a non-empty string")
    return reason


def _usage_tokens(response: Mapping[str, Any]) -> tuple[int, int]:
    usage = response.get("usage")
    if isinstance(usage, Mapping):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
    else:
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
    return _token_count(prompt), _token_count(completion)


def _token_count(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, count)


__all__ = [
    "FALLBACK_ATTEMPT_LIMIT",
    "FALLBACK_DISABLED",
    "FALLBACK_NO_PARTIAL_CONTENT",
    "FALLBACK_PREFILL_UNSUPPORTED",
    "LENGTH_FINISH_REASON",
    "LengthContinuationAttempt",
    "LengthContinuationResult",
    "build_length_continuation_request",
    "continue_length_response",
    "exact_overlap_length",
    "join_exact_overlap",
]
