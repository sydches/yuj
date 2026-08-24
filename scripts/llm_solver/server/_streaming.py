"""Streaming chat completion: assemble Stream[ChatCompletionChunk] into
a single non-stream-shaped response object.

Why streaming saves wall-clock latency on the harness side:
  - llama-server with stream=True emits SSE frames as tokens are
    generated; the client receives bytes overlapping with generation
    instead of waiting for the server to render+ship the entire
    completion at the end.
  - Network transfer + JSON serialize cost of the final blob is
    amortized across the generation window. Measured savings on
    qwen36-A3B at ~80-token completions: 100–300 ms/turn.

What streaming does NOT change:
  - Normal tool-call parsing semantics — we BUFFER the whole response into
    a synthesized non-stream-shaped object, then downstream parses it exactly
    as today. An optional harness observer may inspect accumulated argument
    snapshots and abort, but incomplete calls never reach ordinary dispatch.
  - Transport retry semantics — exceptions raised mid-stream propagate up to
    chat_with_retry which classifies them via _TRANSIENT_ERRORS. A validated
    stream-rule match uses its own typed same-turn retry signal.

Failure modes handled here:
  - Mid-stream APIConnectionError / APITimeoutError / InternalServerError:
    raised during iteration; let them propagate so chat_with_retry
    retries the entire call (same semantics as non-streaming).
  - Stream ends WITHOUT a finish_reason: the server hung up partway
    through. Treat as transient — raise APIConnectionError so the
    retry layer kicks in. A complete response always includes a
    finish_reason in the last chunk.
  - Partial JSON in accumulated tool-call arguments: handed to the
    existing parse_args() in client.py which already handles
    JSONDecodeError by returning {} (same behavior as non-streaming
    when the model emits malformed JSON).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

import openai


# ── Synthetic response shape ─────────────────────────────────────────
#
# The non-stream OpenAI SDK response is a pydantic model with attribute
# access: resp.choices[0].message.content / .tool_calls,
# resp.choices[0].finish_reason, resp.usage.prompt_tokens, etc.
# The dataclasses below mirror that shape so both _chat_with_profile and
# _chat_legacy can consume the streamed result without branching.
# resp.model_dump_json() (used by the verbatim transcript) is provided
# as a method on _StreamedResponse.


@dataclass
class _StreamedFunction:
    name: str
    arguments: str


@dataclass
class _StreamedToolCall:
    id: str
    type: str
    function: _StreamedFunction


@dataclass
class _StreamedMessage:
    content: str | None
    tool_calls: list[_StreamedToolCall] | None = None


@dataclass
class _StreamedChoice:
    message: _StreamedMessage
    finish_reason: str
    index: int = 0


@dataclass
class _StreamedPromptTokenDetails:
    cached_tokens: int | None = None


@dataclass
class _StreamedUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_details: _StreamedPromptTokenDetails | None = None


@dataclass
class _StreamedTimings:
    prompt_n: int | None = None
    cache_n: int | None = None


@dataclass
class _StreamedResponse:
    """Quacks like the non-stream openai response. Attribute-access only."""
    choices: list[_StreamedChoice]
    usage: _StreamedUsage = field(default_factory=_StreamedUsage)
    timings: _StreamedTimings | None = None

    def model_dump_json(self) -> str:
        """Render to JSON for the verbatim transcript log."""
        import json
        choices_dump = []
        for c in self.choices:
            tcs = None
            if c.message.tool_calls:
                tcs = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in c.message.tool_calls
                ]
            choices_dump.append({
                "index": c.index,
                "message": {
                    "role": "assistant",
                    "content": c.message.content,
                    "tool_calls": tcs,
                },
                "finish_reason": c.finish_reason,
            })
        usage_dump = {
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
        }
        if self.usage.prompt_tokens_details is not None:
            usage_dump["prompt_tokens_details"] = {
                "cached_tokens": self.usage.prompt_tokens_details.cached_tokens,
            }
        body = {
            "choices": choices_dump,
            "usage": usage_dump,
            "_streamed": True,
        }
        if self.timings is not None:
            body["timings"] = {
                "prompt_n": self.timings.prompt_n,
                "cache_n": self.timings.cache_n,
            }
        return json.dumps(body)


@dataclass(frozen=True, slots=True)
class StreamDelta:
    """One model-stream surface exposed to a harness-owned observer."""

    source: str  # text | thinking | tool
    delta: str
    tool_index: int = -1
    tool_name: str = ""
    tool_arguments: str = ""


class StreamRuleInterrupt(RuntimeError):
    """Control signal raised after a rule match closes the live stream.

    ``matches`` is transcript-serializable and includes the rule body because
    replay must reproduce the exact hidden retry injection without consulting
    future trace rows or mutable rule files.  The ordinary trace event omits
    that body.
    """

    def __init__(
        self,
        matches: Iterable[Mapping[str, object]],
        partial_response: _StreamedResponse | None = None,
    ) -> None:
        self.matches = tuple(dict(match) for match in matches)
        self.partial_response = partial_response
        names = ", ".join(str(match.get("rule") or "?") for match in self.matches)
        super().__init__(f"stream rule interrupted response: {names}")

    def attach_partial(self, response: _StreamedResponse) -> None:
        self.partial_response = response

    def model_dump_json(self) -> str:
        """Return a valid transcript response that ReplayClient can re-raise."""
        import json

        if self.partial_response is None:
            partial = _StreamedResponse(
                choices=[_StreamedChoice(
                    message=_StreamedMessage(content=None),
                    finish_reason="stream_rule_interrupted",
                )]
            )
        else:
            partial = self.partial_response
        body = json.loads(partial.model_dump_json())
        body["_stream_rule_interrupt"] = {"matches": list(self.matches)}
        return json.dumps(body)

    @classmethod
    def from_transcript(cls, body: Mapping[str, object]) -> "StreamRuleInterrupt":
        """Reconstruct the control signal from one saved JSON response."""
        marker = body.get("_stream_rule_interrupt")
        if not isinstance(marker, Mapping):
            raise ValueError("response is not a stream-rule interrupt record")
        matches = marker.get("matches")
        if not isinstance(matches, list) or any(
            not isinstance(item, Mapping) for item in matches
        ):
            raise ValueError("stream-rule interrupt record has invalid matches")
        choices = body.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        choice = choice if isinstance(choice, Mapping) else {}
        message = choice.get("message")
        message = message if isinstance(message, Mapping) else {}
        tool_calls: list[_StreamedToolCall] = []
        raw_tool_calls = message.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            for raw_tool in raw_tool_calls:
                if not isinstance(raw_tool, Mapping):
                    continue
                function = raw_tool.get("function")
                function = function if isinstance(function, Mapping) else {}
                tool_calls.append(_StreamedToolCall(
                    id=str(raw_tool.get("id") or ""),
                    type=str(raw_tool.get("type") or "function"),
                    function=_StreamedFunction(
                        name=str(function.get("name") or ""),
                        arguments=str(function.get("arguments") or ""),
                    ),
                ))
        usage = body.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        details = usage.get("prompt_tokens_details")
        details = details if isinstance(details, Mapping) else {}
        cached = details.get("cached_tokens")
        partial = _StreamedResponse(
            choices=[_StreamedChoice(
                message=_StreamedMessage(
                    content=(
                        str(message.get("content"))
                        if message.get("content") is not None else None
                    ),
                    tool_calls=tool_calls or None,
                ),
                finish_reason=str(
                    choice.get("finish_reason") or "stream_rule_interrupted"
                ),
            )],
            usage=_StreamedUsage(
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                prompt_tokens_details=(
                    _StreamedPromptTokenDetails(cached_tokens=int(cached))
                    if isinstance(cached, int) and cached >= 0 else None
                ),
            ),
        )
        return cls(matches, partial)


def _field(value, name: str):
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    direct = getattr(value, name, None)
    if direct is not None:
        return direct
    extra = getattr(value, "model_extra", None)
    return extra.get(name) if isinstance(extra, dict) else None


# ── Stream assembly ──────────────────────────────────────────────────


def _build_response(
    *,
    content_parts: list[str],
    tool_calls_by_index: dict[int, dict],
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int | None,
    timing_prompt_n: int | None,
    timing_cache_n: int | None,
) -> _StreamedResponse:
    """Build the SDK-shaped response used for complete and aborted streams."""
    tool_calls_list = None
    if tool_calls_by_index:
        tool_calls_list = []
        for idx in sorted(tool_calls_by_index):
            slot = tool_calls_by_index[idx]
            tool_calls_list.append(_StreamedToolCall(
                id=slot["id"],
                type=slot["type"] or "function",
                function=_StreamedFunction(
                    name=slot["function"]["name"],
                    arguments=slot["function"]["arguments"],
                ),
            ))
    content = "".join(content_parts) if content_parts else None
    return _StreamedResponse(
        choices=[_StreamedChoice(
            message=_StreamedMessage(
                content=content, tool_calls=tool_calls_list,
            ),
            finish_reason=finish_reason,
        )],
        usage=_StreamedUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=(
                _StreamedPromptTokenDetails(cached_tokens=cached_tokens)
                if cached_tokens is not None
                else None
            ),
        ),
        timings=(
            _StreamedTimings(
                prompt_n=timing_prompt_n,
                cache_n=timing_cache_n,
            )
            if timing_prompt_n is not None or timing_cache_n is not None
            else None
        ),
    )


def assemble_stream(
    chunks: Iterable,
    observer: Callable[[StreamDelta], None] | None = None,
) -> _StreamedResponse:
    """Consume a Stream[ChatCompletionChunk] and build a _StreamedResponse.

    Raises openai.APIConnectionError if the stream ends without a
    finish_reason (incomplete response). Other openai exceptions
    raised during iteration propagate up unchanged so the retry
    layer in chat_with_retry can classify them.

    Tool calls are accumulated by index. Each chunk's delta.tool_calls
    item carries a partial function name and/or argument segment;
    we string-concat into the index slot so the final assembled
    arguments string parses identically to the non-streaming case.
    Partial JSON (e.g. on a clean stream that produced malformed
    arguments) is handled downstream by parse_args which returns {}
    on JSONDecodeError — same behavior as today's non-stream path.
    """
    content_parts: list[str] = []
    tool_calls_by_index: dict[int, dict] = {}
    finish_reason: str | None = None
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens: int | None = None
    timing_prompt_n: int | None = None
    timing_cache_n: int | None = None
    received_any_chunk = False

    def notify(delta: StreamDelta) -> None:
        if observer is None:
            return
        try:
            observer(delta)
        except Exception as exc:
            if isinstance(exc, StreamRuleInterrupt):
                exc.attach_partial(_build_response(
                    content_parts=content_parts,
                    tool_calls_by_index=tool_calls_by_index,
                    finish_reason="stream_rule_interrupted",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                    timing_prompt_n=timing_prompt_n,
                    timing_cache_n=timing_cache_n,
                ))
            close = getattr(chunks, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    # The semantic cancellation signal is the original
                    # exception. A close failure must not turn an intentional
                    # interruption into an unrelated transport error.
                    pass
            raise

    for chunk in chunks:
        received_any_chunk = True
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            details = _field(usage, "prompt_tokens_details")
            if details is None:
                details = _field(usage, "input_tokens_details")
            observed_cached = _field(details, "cached_tokens")
            if isinstance(observed_cached, int) and observed_cached >= 0:
                cached_tokens = observed_cached
        timings = _field(chunk, "timings")
        observed_prompt_n = _field(timings, "prompt_n")
        observed_cache_n = _field(timings, "cache_n")
        if isinstance(observed_prompt_n, int) and observed_prompt_n >= 0:
            timing_prompt_n = observed_prompt_n
        if isinstance(observed_cache_n, int) and observed_cache_n >= 0:
            timing_cache_n = observed_cache_n
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        choice = choices[0]
        if getattr(choice, "finish_reason", None) is not None:
            finish_reason = choice.finish_reason
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        delta_content = getattr(delta, "content", None)
        if delta_content:
            content_parts.append(delta_content)
            notify(StreamDelta(source="text", delta=str(delta_content)))
        delta_thinking = (
            _field(delta, "reasoning_content")
            or _field(delta, "thinking")
            or _field(delta, "reasoning")
        )
        if isinstance(delta_thinking, str) and delta_thinking:
            notify(StreamDelta(source="thinking", delta=delta_thinking))
        delta_tool_calls = getattr(delta, "tool_calls", None) or []
        for tc_chunk in delta_tool_calls:
            idx = getattr(tc_chunk, "index", 0)
            slot = tool_calls_by_index.setdefault(idx, {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            })
            tc_id = getattr(tc_chunk, "id", None)
            if tc_id:
                slot["id"] = tc_id
            tc_type = getattr(tc_chunk, "type", None)
            if tc_type:
                slot["type"] = tc_type
            tc_func = getattr(tc_chunk, "function", None)
            if tc_func is not None:
                fn_name = getattr(tc_func, "name", None)
                if fn_name:
                    slot["function"]["name"] += fn_name
                fn_args = getattr(tc_func, "arguments", None)
                if fn_args:
                    slot["function"]["arguments"] += fn_args
            notify(StreamDelta(
                source="tool",
                delta=str(
                    getattr(tc_func, "arguments", "")
                    if tc_func is not None else ""
                ),
                tool_index=int(idx),
                tool_name=str(slot["function"]["name"]),
                tool_arguments=str(slot["function"]["arguments"]),
            ))

    # Incomplete response detection. A complete chat completion stream
    # always emits a finish_reason in the last content choice. Missing
    # finish_reason ⇒ the server cut us off mid-completion; classify
    # as transient so the retry layer kicks in.
    if finish_reason is None:
        if not received_any_chunk:
            msg = "stream produced no chunks (server returned empty stream)"
        else:
            msg = "stream ended without finish_reason — incomplete response"
        raise openai.APIConnectionError(
            message=msg,
            request=None,  # type: ignore[arg-type]
        )

    return _build_response(
        content_parts=content_parts,
        tool_calls_by_index=tool_calls_by_index,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        timing_prompt_n=timing_prompt_n,
        timing_cache_n=timing_cache_n,
    )
