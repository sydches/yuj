"""Anthropic Messages API adapter for the yuj CLI.

This module wraps LlamaClient at its HTTP boundary. It keeps the profile
normalization, transcript handling, and chat flow unchanged.
"""
from __future__ import annotations

import json

import requests

from ..llm_solver.server.client import LlamaClient, parse_args

_ANTHROPIC_VERSION = "2023-06-01"


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _CompatResponse:
    """Anthropic response reshaped to the OpenAI SDK response surface."""

    def __init__(self, raw: dict, *, message: _Obj, finish_reason: str, usage: _Obj):
        self._raw = raw
        self.choices = [_Obj(message=message, finish_reason=finish_reason)]
        self.usage = usage

    def model_dump_json(self) -> str:
        return json.dumps(self._raw, default=str)


def _to_anthropic_payload(payload: dict) -> dict:
    system_parts: list[str] = []
    messages: list[dict] = []
    pending_tool_results: list[dict] = []

    def flush_tool_results() -> None:
        nonlocal pending_tool_results
        if pending_tool_results:
            _append_anthropic_message(messages, "user", pending_tool_results)
            pending_tool_results = []

    for msg in payload.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            if content:
                system_parts.append(str(content))
            continue
        if role == "tool":
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": str(content or ""),
            })
            continue

        flush_tool_results()
        if role == "assistant":
            blocks = []
            if content:
                blocks.append({"type": "text", "text": str(content)})
            for tc in msg.get("tool_calls") or []:
                func = tc.get("function", {})
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "input": parse_args(func.get("arguments", "{}")),
                })
            _append_anthropic_message(messages, "assistant", blocks or [{"type": "text", "text": ""}])
        elif role == "user":
            _append_anthropic_message(
                messages,
                "user",
                [{"type": "text", "text": str(content or "")}],
            )

    flush_tool_results()
    out = {
        "model": payload["model"],
        "messages": messages,
        "max_tokens": payload["max_tokens"],
    }
    if system_parts:
        out["system"] = "\n\n".join(system_parts)
    tools = payload.get("tools") or []
    if tools:
        out["tools"] = [_to_anthropic_tool(tool) for tool in tools]
    return out


def _append_anthropic_message(messages: list[dict], role: str, content: list[dict]) -> None:
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].extend(content)
        return
    messages.append({"role": role, "content": content})


def _to_anthropic_tool(tool: dict) -> dict:
    func = tool.get("function", {})
    return {
        "name": func.get("name", ""),
        "description": func.get("description", ""),
        "input_schema": func.get("parameters") or {"type": "object", "properties": {}},
    }


def _anthropic_to_openai_response(raw: dict) -> _CompatResponse:
    text_parts: list[str] = []
    tool_calls = []
    for block in raw.get("content", []):
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            tool_calls.append(_Obj(
                id=block.get("id", ""),
                function=_Obj(
                    name=block.get("name", ""),
                    arguments=json.dumps(block.get("input") or {}),
                ),
            ))
    stop_reason = raw.get("stop_reason") or "end_turn"
    finish_reason = {
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "end_turn": "stop",
        "stop_sequence": "stop",
    }.get(stop_reason, stop_reason)
    usage_raw = raw.get("usage") or {}
    message = _Obj(
        content="\n".join(part for part in text_parts if part) or None,
        tool_calls=tool_calls,
    )
    usage = _Obj(
        prompt_tokens=int(usage_raw.get("input_tokens") or 0),
        completion_tokens=int(usage_raw.get("output_tokens") or 0),
    )
    return _CompatResponse(raw, message=message, finish_reason=finish_reason, usage=usage)


class AnthropicClient(LlamaClient):
    """LlamaClient with the HTTP boundary swapped to Anthropic Messages."""

    def _call_api(self, payload: dict):
        # Mirrors LlamaClient._call_api's transcript contract, minus the
        # streaming branch — the Messages adapter is request/response only.
        self._transcript_call_n += 1
        n = self._transcript_call_n
        self._write_transcript(
            f"turn {n:03d} input",
            json.dumps(payload, default=str),
        )
        try:
            resp = self._call_anthropic_api(payload)
        except Exception as e:
            self._write_transcript(
                f"turn {n:03d} output", f"{type(e).__name__}: {e}"
            )
            raise
        self._write_transcript(f"turn {n:03d} output", resp.model_dump_json())
        return resp

    def _call_anthropic_api(self, payload: dict) -> _CompatResponse:
        """Call Anthropic Messages and adapt the response to the OpenAI SDK shape."""
        anthropic_payload = _to_anthropic_payload(payload)
        resp = requests.post(
            f"{self.cfg.base_url.rstrip('/')}/messages",
            headers={
                "x-api-key": self.cfg.api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json=anthropic_payload,
            timeout=(self.cfg.timeout_connect, self.cfg.timeout_read),
        )
        resp.raise_for_status()
        return _anthropic_to_openai_response(resp.json())

    def health_check(self) -> list[str]:
        resp = requests.get(
            f"{self.cfg.base_url.rstrip('/')}/models",
            headers={
                "x-api-key": self.cfg.api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
            },
            timeout=(self.cfg.timeout_connect, self.cfg.timeout_read),
        )
        resp.raise_for_status()
        data = resp.json()
        return [str(m["id"]) for m in data.get("data", []) if "id" in m]

    def query_server_context(self) -> int | None:
        # No /props or /slots on a hosted API; cfg.context_size stands.
        return None
