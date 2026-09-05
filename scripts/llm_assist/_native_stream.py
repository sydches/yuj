"""Expose native provider text to the existing stream observer.

Adapters still own authentication, payloads, and successful response parsing.
The shared assembler owns interrupted-response snapshots and stream closure.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace as Obj

from ..llm_solver.server._streaming import assemble_stream


def _events(response):
    """Read JSON SSE data, including multi-line data fields."""
    data = []
    for line in response.iter_lines(decode_unicode=True):
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not line:
            if data:
                body = "\n".join(data)
                data.clear()
                if body != "[DONE]":
                    yield json.loads(body)
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data and "\n".join(data) != "[DONE]":
        yield json.loads("\n".join(data))


def _chunk(*, text=None, thinking=None, tool=None, usage=None, finish=None):
    return Obj(
        choices=[Obj(delta=Obj(content=text, thinking=thinking,
                               tool_calls=[tool] if tool else None),
                     finish_reason=finish)],
        usage=usage,
    )


def read_anthropic_stream(response, observer):
    """Assemble Messages events without changing provider-owned fields."""
    raw = {}
    arguments = {}

    def chunks():
        try:
            for event in _events(response):
                kind = event.get("type")
                if kind == "message_start":
                    raw.update(copy.deepcopy(event["message"]))
                    usage = raw.get("usage", {})
                    prompt = sum(int(usage.get(k, 0) or 0) for k in (
                        "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"
                    ))
                    # The opening output count is not the eventual completion
                    # count. Leave it unknown if generation is interrupted.
                    yield _chunk(usage=Obj(prompt_tokens=prompt, completion_tokens=0))
                elif kind == "content_block_start":
                    index = event["index"]
                    block = copy.deepcopy(event["content_block"])
                    blocks = raw.setdefault("content", [])
                    if index != len(blocks):
                        raise ValueError("out-of-order content block")
                    blocks.append(block)
                    if block["type"] == "text":
                        yield _chunk(text=block.get("text", ""))
                    elif block["type"] == "tool_use":
                        arguments[index] = ""
                        yield _chunk(tool=Obj(index=index, id=block["id"], type="function",
                                             function=Obj(name=block["name"], arguments="")))
                elif kind == "content_block_delta":
                    index = event["index"]
                    block, delta = raw["content"][index], event["delta"]
                    field = {"text_delta": "text", "thinking_delta": "thinking",
                             "signature_delta": "signature"}.get(delta["type"])
                    if field:
                        value = delta.get(field, "")
                        block[field] = block.get(field, "") + value
                        if field == "text":
                            yield _chunk(text=value)
                        elif field == "thinking":
                            yield _chunk(thinking=value)
                    elif delta["type"] == "input_json_delta":
                        value = delta.get("partial_json", "")
                        arguments[index] += value
                        yield _chunk(tool=Obj(index=index, id=None, type=None,
                                             function=Obj(name=None, arguments=value)))
                elif kind == "content_block_stop":
                    index = event["index"]
                    if arguments.get(index):
                        raw["content"][index]["input"] = json.loads(arguments[index])
                elif kind == "message_delta":
                    raw.update(event.get("delta", {}))
                    raw.setdefault("usage", {}).update(event.get("usage", {}))
                elif kind == "message_stop":
                    yield _chunk(finish="stop")
                    return
                elif kind == "error":
                    raise ValueError("native message stream failed")
        finally:
            response.close()

    assemble_stream(chunks(), observer=observer)
    return raw


def read_responses_stream(response, observer):
    """Observe Responses deltas; retain the provider's final response verbatim."""
    completed = None

    def chunks():
        nonlocal completed
        try:
            for event in _events(response):
                kind = event.get("type")
                if kind == "response.output_text.delta":
                    yield _chunk(text=event.get("delta", ""))
                elif kind in {"response.reasoning_text.delta", "response.reasoning_summary_text.delta"}:
                    yield _chunk(thinking=event.get("delta", ""))
                elif kind == "response.output_item.added":
                    item = event.get("item", {})
                    if item.get("type") == "function_call":
                        yield _chunk(tool=Obj(index=event["output_index"],
                                             id=item.get("call_id") or item.get("id"),
                                             type="function", function=Obj(
                                                 name=item.get("name", ""),
                                                 arguments=item.get("arguments", ""))))
                elif kind == "response.function_call_arguments.delta":
                    yield _chunk(tool=Obj(index=event["output_index"], id=None, type=None,
                                         function=Obj(name=None, arguments=event.get("delta", ""))))
                elif kind in {"response.completed", "response.done", "response.incomplete"}:
                    completed = copy.deepcopy(event["response"])
                    completed.setdefault("status", "incomplete" if kind == "response.incomplete" else "completed")
                    yield _chunk(finish="stop")
                    return
                elif kind in {"response.failed", "error"}:
                    raise ValueError("native response stream failed")
        finally:
            response.close()

    assemble_stream(chunks(), observer=observer)
    if completed is None:
        raise ValueError("native response stream did not complete")
    return completed
