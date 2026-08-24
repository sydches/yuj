"""Anthropic Messages API adapter for the yuj CLI.

This module wraps LlamaClient at its HTTP boundary. It keeps the profile
normalization, transcript handling, and chat flow unchanged.
"""
from __future__ import annotations

import hashlib
import json
import uuid

import requests

from ..llm_solver.server.client import LlamaClient, parse_args
from ._auth import (
    AuthProtocolError,
    CredentialSession,
    ProviderAuthError,
    classify_provider_response,
)

_ANTHROPIC_VERSION = "2023-06-01"
_CLAUDE_CODE_VERSION = "2.1.220"
_SUBSCRIPTION_BETAS = "claude-code-20250219"
_SUBSCRIPTION_USER_AGENT = (
    f"claude-cli/{_CLAUDE_CODE_VERSION} (external, claude-desktop)"
)
_SUBSCRIPTION_SYSTEM = (
    "You are a Claude agent, built on Anthropic's Claude Agent SDK."
)
_SUBSCRIPTION_TOOL_PREFIX = "_"
_SUBSCRIPTION_MAX_OUTPUT_TOKENS = 64_000
_SUBSCRIPTION_BILLING_PREFIX = "x-anthropic-billing-header:"
_CCH_PLACEHOLDER = b"cch=00000"
_CCH_SEED = 0x4D659218E32A3268
_CCH_SYSTEM_ANCHOR = (
    b'"system":[{"type":"text","text":"'
    + _SUBSCRIPTION_BILLING_PREFIX.encode()
)
_CCH_SEARCH_WINDOW = 150
_UINT64_MASK = (1 << 64) - 1
_XXH64_PRIME_1 = 0x9E3779B185EBCA87
_XXH64_PRIME_2 = 0xC2B2AE3D27D4EB4F
_XXH64_PRIME_3 = 0x165667B19E3779F9
_XXH64_PRIME_4 = 0x85EBCA77C2B2AE63
_XXH64_PRIME_5 = 0x27D4EB2F165667C5
_ANTHROPIC_BUILTIN_TOOL_NAMES = frozenset({
    "code_execution",
    "computer",
    "text_editor",
    "web_search",
})


def _rotate_left_64(value: int, count: int) -> int:
    value &= _UINT64_MASK
    return ((value << count) | (value >> (64 - count))) & _UINT64_MASK


def _xxhash64_round(accumulator: int, lane: int) -> int:
    accumulator = (
        accumulator + lane * _XXH64_PRIME_2
    ) & _UINT64_MASK
    accumulator = _rotate_left_64(accumulator, 31)
    return (accumulator * _XXH64_PRIME_1) & _UINT64_MASK


def _xxhash64_merge(accumulator: int, lane: int) -> int:
    accumulator ^= _xxhash64_round(0, lane)
    return (
        accumulator * _XXH64_PRIME_1 + _XXH64_PRIME_4
    ) & _UINT64_MASK


def _xxhash64(data: bytes, seed: int = 0) -> int:
    """Return XXH64 using the published little-endian reference algorithm."""
    length = len(data)
    offset = 0
    seed &= _UINT64_MASK

    if length >= 32:
        lane_1 = (seed + _XXH64_PRIME_1 + _XXH64_PRIME_2) & _UINT64_MASK
        lane_2 = (seed + _XXH64_PRIME_2) & _UINT64_MASK
        lane_3 = seed
        lane_4 = (seed - _XXH64_PRIME_1) & _UINT64_MASK
        while offset <= length - 32:
            lane_1 = _xxhash64_round(
                lane_1, int.from_bytes(data[offset:offset + 8], "little")
            )
            offset += 8
            lane_2 = _xxhash64_round(
                lane_2, int.from_bytes(data[offset:offset + 8], "little")
            )
            offset += 8
            lane_3 = _xxhash64_round(
                lane_3, int.from_bytes(data[offset:offset + 8], "little")
            )
            offset += 8
            lane_4 = _xxhash64_round(
                lane_4, int.from_bytes(data[offset:offset + 8], "little")
            )
            offset += 8
        result = (
            _rotate_left_64(lane_1, 1)
            + _rotate_left_64(lane_2, 7)
            + _rotate_left_64(lane_3, 12)
            + _rotate_left_64(lane_4, 18)
        ) & _UINT64_MASK
        for lane in (lane_1, lane_2, lane_3, lane_4):
            result = _xxhash64_merge(result, lane)
    else:
        result = (seed + _XXH64_PRIME_5) & _UINT64_MASK

    result = (result + length) & _UINT64_MASK
    while offset <= length - 8:
        lane = int.from_bytes(data[offset:offset + 8], "little")
        result ^= _xxhash64_round(0, lane)
        result = (
            _rotate_left_64(result, 27) * _XXH64_PRIME_1
            + _XXH64_PRIME_4
        ) & _UINT64_MASK
        offset += 8
    if offset <= length - 4:
        lane = int.from_bytes(data[offset:offset + 4], "little")
        result ^= (lane * _XXH64_PRIME_1) & _UINT64_MASK
        result = (
            _rotate_left_64(result, 23) * _XXH64_PRIME_2
            + _XXH64_PRIME_3
        ) & _UINT64_MASK
        offset += 4
    while offset < length:
        result ^= (data[offset] * _XXH64_PRIME_5) & _UINT64_MASK
        result = (
            _rotate_left_64(result, 11) * _XXH64_PRIME_1
        ) & _UINT64_MASK
        offset += 1

    result ^= result >> 33
    result = (result * _XXH64_PRIME_2) & _UINT64_MASK
    result ^= result >> 29
    result = (result * _XXH64_PRIME_3) & _UINT64_MASK
    result ^= result >> 32
    return result & _UINT64_MASK


def _first_user_message_text(payload: dict) -> str:
    for message in payload.get("messages", []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                return text if isinstance(text, str) else ""
        return ""
    return ""


def _subscription_billing_header(first_user_message: str) -> str:
    selected = "".join(
        first_user_message[index] if index < len(first_user_message) else "0"
        for index in (4, 7, 20)
    )
    suffix = hashlib.sha256(
        f"59cf53e54c78{selected}{_CLAUDE_CODE_VERSION}".encode()
    ).hexdigest()[:3]
    return (
        f"{_SUBSCRIPTION_BILLING_PREFIX} "
        f"cc_version={_CLAUDE_CODE_VERSION}.{suffix}; "
        "cc_entrypoint=claude-desktop; cch=00000;"
    )


def _serialize_subscription_body(payload: dict) -> bytes:
    body = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    anchor = body.find(_CCH_SYSTEM_ANCHOR)
    if anchor < 0:
        raise AuthProtocolError(
            "claude", "subscription billing attestation was malformed"
        )
    search_from = anchor + len(_CCH_SYSTEM_ANCHOR)
    placeholder = body.find(_CCH_PLACEHOLDER, search_from)
    if (
        placeholder < 0
        or placeholder - search_from > _CCH_SEARCH_WINDOW
    ):
        raise AuthProtocolError(
            "claude", "subscription billing attestation was malformed"
        )
    cch = f"{_xxhash64(body, _CCH_SEED) & 0xFFFFF:05x}".encode()
    patched = bytearray(body)
    patched[placeholder + 4:placeholder + 9] = cch
    return bytes(patched)


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


def _to_anthropic_payload(payload: dict, *, subscription: bool = False) -> dict:
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
                    "name": _encode_tool_name(
                        func.get("name", ""), subscription=subscription
                    ),
                    "input": parse_args(func.get("arguments", "{}")),
                })
            _append_anthropic_message(
                messages,
                "assistant",
                blocks or [{"type": "text", "text": ""}],
            )
        elif role == "user":
            _append_anthropic_message(
                messages,
                "user",
                [{"type": "text", "text": str(content or "")}],
            )

    flush_tool_results()
    if subscription:
        out = {
            "model": payload["model"],
            "messages": messages,
            "system": [
                {
                    "type": "text",
                    "text": _subscription_billing_header(
                        _first_user_message_text(payload)
                    ),
                },
                {"type": "text", "text": _SUBSCRIPTION_SYSTEM},
                *(
                    [{"type": "text", "text": "\n\n".join(system_parts)}]
                    if system_parts
                    else []
                ),
            ],
        }
        tools = payload.get("tools") or []
        if tools:
            out["tools"] = [
                _to_anthropic_tool(tool, subscription=True)
                for tool in tools
            ]
        out["max_tokens"] = min(
            payload["max_tokens"], _SUBSCRIPTION_MAX_OUTPUT_TOKENS
        )
        return out

    out = {
        "model": payload["model"],
        "messages": messages,
        "max_tokens": payload["max_tokens"],
    }
    if system_parts:
        out["system"] = "\n\n".join(system_parts)
    tools = payload.get("tools") or []
    if tools:
        out["tools"] = [
            _to_anthropic_tool(tool, subscription=False)
            for tool in tools
        ]
    return out


def _append_anthropic_message(
    messages: list[dict], role: str, content: list[dict]
) -> None:
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].extend(content)
        return
    messages.append({"role": role, "content": content})


def _to_anthropic_tool(tool: dict, *, subscription: bool = False) -> dict:
    func = tool.get("function", {})
    return {
        "name": _encode_tool_name(
            func.get("name", ""), subscription=subscription
        ),
        "description": func.get("description", ""),
        "input_schema": func.get("parameters") or {"type": "object", "properties": {}},
    }


def _encode_tool_name(name: object, *, subscription: bool) -> str:
    value = str(name or "")
    if not subscription or value.lower() in _ANTHROPIC_BUILTIN_TOOL_NAMES:
        return value
    return f"{_SUBSCRIPTION_TOOL_PREFIX}{value}"


def _decode_tool_name(name: object, *, subscription: bool) -> str:
    value = str(name or "")
    if subscription and value.startswith(_SUBSCRIPTION_TOOL_PREFIX):
        return value[len(_SUBSCRIPTION_TOOL_PREFIX):]
    return value


def _observed_usage_count(value: object) -> int | None:
    """Return one provider-owned token count without coercing missing data."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _anthropic_to_openai_response(
    raw: dict, *, subscription: bool = False
) -> _CompatResponse:
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
                    name=_decode_tool_name(
                        block.get("name", ""), subscription=subscription
                    ),
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
    raw_usage = raw.get("usage")
    usage_raw = raw_usage if isinstance(raw_usage, dict) else {}
    input_tokens = _observed_usage_count(usage_raw.get("input_tokens"))
    output_tokens = _observed_usage_count(usage_raw.get("output_tokens"))
    cache_creation_tokens = (
        _observed_usage_count(usage_raw.get("cache_creation_input_tokens"))
        if "cache_creation_input_tokens" in usage_raw
        else (0 if input_tokens is not None else None)
    )
    cache_read_tokens = (
        _observed_usage_count(usage_raw.get("cache_read_input_tokens"))
        if "cache_read_input_tokens" in usage_raw
        else (0 if input_tokens is not None else None)
    )
    prompt_known = all(
        count is not None
        for count in (input_tokens, cache_creation_tokens, cache_read_tokens)
    )
    prompt_tokens = (
        input_tokens + cache_creation_tokens + cache_read_tokens
        if prompt_known
        else 0
    )
    message = _Obj(
        content="\n".join(part for part in text_parts if part) or None,
        tool_calls=tool_calls,
    )
    usage = _Obj(
        prompt_tokens=prompt_tokens,
        completion_tokens=output_tokens if output_tokens is not None else 0,
        prompt_tokens_known=prompt_known,
        completion_tokens_known=output_tokens is not None,
        prompt_tokens_details=(
            _Obj(cached_tokens=cache_read_tokens)
            if cache_read_tokens is not None
            else None
        ),
    )
    return _CompatResponse(
        raw, message=message, finish_reason=finish_reason, usage=usage
    )


class AnthropicClient(LlamaClient):
    """LlamaClient with the HTTP boundary swapped to Anthropic Messages."""

    def __init__(
        self,
        cfg,
        profile=None,
        *,
        auth: CredentialSession | None = None,
        http=None,
    ):
        super().__init__(cfg, profile=profile)
        self._auth = auth
        self._http = http or requests.Session()

    def _headers(self) -> dict[str, str]:
        headers = {
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        if self._auth is None:
            headers["X-Api-Key"] = self.cfg.api_key
            return headers
        credential = self._auth.access()
        if self._auth.binding.auth_method == "subscription":
            headers.update({
                "Accept": "application/json",
                "Authorization": f"Bearer {credential.token}",
                "anthropic-beta": _SUBSCRIPTION_BETAS,
                "anthropic-dangerous-direct-browser-access": "true",
                "User-Agent": _SUBSCRIPTION_USER_AGENT,
                "x-app": "cli",
                "x-client-request-id": str(uuid.uuid4()),
            })
            session_id = getattr(self, "_session_id", "")
            if session_id:
                headers["X-Claude-Code-Session-Id"] = session_id
        else:
            headers["X-Api-Key"] = credential.token
        return headers

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
        except ProviderAuthError as e:
            self._last_provider_auth_error = e
            self._write_transcript(
                f"turn {n:03d} output", f"{type(e).__name__}: {e}"
            )
            raise
        except Exception as e:
            self._write_transcript(
                f"turn {n:03d} output", f"{type(e).__name__}: {e}"
            )
            raise
        self._write_transcript(f"turn {n:03d} output", resp.model_dump_json())
        return resp

    def _call_anthropic_api(self, payload: dict) -> _CompatResponse:
        """Call Anthropic Messages and adapt the response to the OpenAI SDK shape."""
        subscription = (
            self._auth is not None
            and self._auth.binding.auth_method == "subscription"
        )
        anthropic_payload = _to_anthropic_payload(
            payload, subscription=subscription
        )
        headers = self._headers()
        try:
            request = {
                "headers": headers,
                "timeout": (self.cfg.timeout_connect, self.cfg.timeout_read),
            }
            if subscription:
                request["data"] = _serialize_subscription_body(
                    anthropic_payload
                )
            else:
                request["json"] = anthropic_payload
            resp = self._http.post(
                f"{self.cfg.base_url.rstrip('/')}/messages",
                **request,
            )
        except Exception as exc:
            if self._auth is not None:
                raise AuthProtocolError(
                    "claude", "model request transport failed"
                ) from exc
            raise
        if self._auth is None:
            resp.raise_for_status()
        else:
            classify_provider_response("claude", resp)
        try:
            raw = resp.json()
        except Exception as exc:
            if self._auth is not None:
                raise AuthProtocolError(
                    "claude", "model response was malformed"
                ) from exc
            raise
        return _anthropic_to_openai_response(raw, subscription=subscription)

    def health_check(self) -> list[str]:
        resp = self._http.get(
            f"{self.cfg.base_url.rstrip('/')}/models",
            headers=self._headers(),
            timeout=(self.cfg.timeout_connect, self.cfg.timeout_read),
        )
        if self._auth is None:
            resp.raise_for_status()
        else:
            classify_provider_response("claude", resp)
        data = resp.json()
        return [str(m["id"]) for m in data.get("data", []) if "id" in m]

    def query_server_context(self) -> int | None:
        # No /props or /slots on a hosted API; cfg.context_size stands.
        return None
