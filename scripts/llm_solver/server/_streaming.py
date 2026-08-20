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
  - Tool-call parsing semantics — we BUFFER the whole response into
    a synthesized non-stream-shaped object, then downstream parses
    it exactly as today. The harness loop never sees an incomplete
    tool call. (Early-fire on tool_call finish_reason is explicitly
    OUT OF SCOPE — too invasive for too little marginal gain.)
  - Retry semantics — exceptions raised mid-stream propagate up to
    chat_with_retry which classifies them via _TRANSIENT_ERRORS.

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
from typing import Iterable

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
class _StreamedUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class _StreamedResponse:
    """Quacks like the non-stream openai response. Attribute-access only."""
    choices: list[_StreamedChoice]
    usage: _StreamedUsage = field(default_factory=_StreamedUsage)

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
        return json.dumps({
            "choices": choices_dump,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
            },
            "_streamed": True,
        })


# ── Stream assembly ──────────────────────────────────────────────────


def assemble_stream(chunks: Iterable) -> _StreamedResponse:
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
    received_any_chunk = False

    for chunk in chunks:
        received_any_chunk = True
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
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

    # Build the synthesized response. Sort tool_calls by stream-order
    # index so multi-tool turns retain the model's ordering.
    tool_calls_list = None
    if tool_calls_by_index:
        tool_calls_list = []
        for idx in sorted(tool_calls_by_index.keys()):
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
        ),
    )
