"""ChatGPT subscription transport for the Yuj assistant shell."""
from __future__ import annotations

import json

import requests

from ..llm_solver.server.client import LlamaClient
from ._anthropic import _CompatResponse, _Obj, _observed_usage_count
from ._auth import (
    AuthProtocolError,
    CredentialSession,
    ProviderAuthError,
    classify_provider_response,
)

_RESPONSES_BETA = "responses=experimental"
_YUJ_CLIENT_VERSION = "0.1.0"


def _to_responses_payload(payload: dict, *, session_id: str) -> dict:
    instructions: list[str] = []
    input_items: list[dict] = []

    for message in payload.get("messages", []):
        role = message.get("role")
        content = message.get("content")
        if role in {"system", "developer"}:
            if content:
                instructions.append(str(content))
            continue
        if role == "tool":
            input_items.append({
                "type": "function_call_output",
                "call_id": message.get("tool_call_id", ""),
                "output": str(content or ""),
            })
            continue
        if role not in {"user", "assistant"}:
            continue

        text_type = "input_text" if role == "user" else "output_text"
        if content is not None:
            wire_content = (
                _to_responses_user_content(content)
                if role == "user"
                else [{"type": text_type, "text": str(content)}]
            )
            input_items.append({
                "role": role,
                "content": wire_content,
            })
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                arguments = function.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, separators=(",", ":"))
                input_items.append({
                    "type": "function_call",
                    "call_id": call.get("id", ""),
                    "name": function.get("name", ""),
                    "arguments": arguments,
                })

    request: dict = {
        "model": payload["model"],
        "instructions": "\n\n".join(instructions),
        "input": input_items,
        "store": False,
        "stream": True,
        "parallel_tool_calls": True,
    }
    tools = payload.get("tools") or []
    if tools:
        request["tools"] = [_to_responses_tool(tool) for tool in tools]
        request["tool_choice"] = "auto"
    if session_id:
        request["prompt_cache_key"] = session_id
    return request


def _to_responses_user_content(content: object) -> list[dict]:
    if not isinstance(content, list):
        return [{"type": "input_text", "text": str(content)}]
    blocks: list[dict] = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("user content part must be an object")
        part_type = part.get("type")
        if part_type == "text":
            blocks.append({
                "type": "input_text",
                "text": str(part.get("text") or ""),
            })
            continue
        if part_type != "image_url":
            raise ValueError(f"unsupported user content part: {part_type!r}")
        image_url = part.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        if not isinstance(url, str) or not url.startswith("data:image/"):
            raise ValueError("image content requires a data URL")
        blocks.append({"type": "input_image", "image_url": url})
    return blocks


def _to_responses_tool(tool: dict) -> dict:
    function = tool.get("function") or {}
    return {
        "type": "function",
        "name": function.get("name", ""),
        "description": function.get("description", ""),
        "parameters": function.get("parameters")
        or {"type": "object", "properties": {}},
    }


def _responses_to_openai(raw: dict) -> _CompatResponse:
    text_parts: list[str] = []
    tool_calls: list[_Obj] = []
    for item in raw.get("output") or []:
        item_type = item.get("type")
        if item_type == "message":
            for content in item.get("content") or []:
                if content.get("type") in {"output_text", "text"}:
                    text_parts.append(str(content.get("text") or ""))
        elif item_type in {"function_call", "tool_call"}:
            tool_calls.append(_Obj(
                id=item.get("call_id") or item.get("id") or "",
                type="function",
                function=_Obj(
                    name=item.get("name") or "",
                    arguments=item.get("arguments") or "{}",
                ),
            ))

    status = str(raw.get("status") or "completed")
    finish_reason = "tool_calls" if tool_calls else {
        "completed": "stop",
        "incomplete": "length",
    }.get(status, status)
    raw_usage = raw.get("usage")
    usage_raw = raw_usage if isinstance(raw_usage, dict) else {}
    input_tokens = _observed_usage_count(usage_raw.get("input_tokens"))
    output_tokens = _observed_usage_count(usage_raw.get("output_tokens"))
    raw_input_details = usage_raw.get("input_tokens_details")
    input_details = (
        raw_input_details if isinstance(raw_input_details, dict) else {}
    )
    cached_tokens = _observed_usage_count(input_details.get("cached_tokens"))
    return _CompatResponse(
        raw,
        message=_Obj(
            content="\n".join(part for part in text_parts if part) or None,
            tool_calls=tool_calls,
        ),
        finish_reason=finish_reason,
        usage=_Obj(
            prompt_tokens=input_tokens if input_tokens is not None else 0,
            completion_tokens=output_tokens if output_tokens is not None else 0,
            prompt_tokens_known=input_tokens is not None,
            completion_tokens_known=output_tokens is not None,
            prompt_tokens_details=(
                _Obj(cached_tokens=cached_tokens)
                if cached_tokens is not None
                else None
            ),
        ),
    )


class CodexSubscriptionClient(LlamaClient):
    """LlamaClient with a pinned ChatGPT subscription HTTP transport."""

    def __init__(
        self,
        cfg,
        profile=None,
        *,
        auth: CredentialSession,
        http=None,
    ):
        super().__init__(cfg, profile=profile)
        self._auth = auth
        self._http = http or requests.Session()

    def _headers(self) -> dict[str, str]:
        credential = self._auth.access()
        if not credential.account_id:
            raise AuthProtocolError(
                "codex", "subscription credential has no account identity"
            )
        headers = {
            "Authorization": f"Bearer {credential.token}",
            "chatgpt-account-id": credential.account_id,
            "originator": "yuj",
            "version": _YUJ_CLIENT_VERSION,
            "User-Agent": "yuj",
            "OpenAI-Beta": _RESPONSES_BETA,
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        session_id = getattr(self, "_session_id", "")
        if session_id:
            headers.update({
                "conversation_id": session_id,
                "session_id": session_id,
                "x-client-request-id": session_id,
            })
        return headers

    def _call_api(self, payload: dict, *, record_transcript: bool = True):
        n = 0
        if record_transcript:
            self._transcript_call_n += 1
            n = self._transcript_call_n
            self._write_transcript(
                f"turn {n:03d} input", json.dumps(payload, default=str)
            )
        try:
            response = self._call_responses_api(payload)
        except ProviderAuthError as exc:
            self._last_provider_auth_error = exc
            if record_transcript:
                self._write_transcript(
                    f"turn {n:03d} output", f"{type(exc).__name__}: {exc}"
                )
            raise
        except Exception as exc:
            if record_transcript:
                self._write_transcript(
                    f"turn {n:03d} output", f"{type(exc).__name__}: {exc}"
                )
            raise
        if record_transcript:
            self._write_transcript(
                f"turn {n:03d} output", response.model_dump_json()
            )
        return response

    def _call_responses_api(self, payload: dict) -> _CompatResponse:
        request = _to_responses_payload(
            payload, session_id=getattr(self, "_session_id", "")
        )
        headers = self._headers()
        try:
            response = self._http.post(
                f"{self.cfg.base_url.rstrip('/')}/responses",
                headers=headers,
                json=request,
                stream=True,
                timeout=(self.cfg.timeout_connect, self.cfg.timeout_read),
            )
        except Exception as exc:
            raise AuthProtocolError(
                "codex", "subscription request transport failed"
            ) from exc
        classify_provider_response("codex", response)
        completed = _completed_response(response)
        return _responses_to_openai(completed)

    def health_check(self) -> list[str]:
        headers = self._headers()
        try:
            response = self._http.get(
                f"{self.cfg.base_url.rstrip('/')}/models",
                headers=headers,
                timeout=(self.cfg.timeout_connect, self.cfg.timeout_read),
            )
        except Exception as exc:
            raise AuthProtocolError(
                "codex", "model-list transport failed"
            ) from exc
        classify_provider_response("codex", response)
        try:
            raw = response.json()
        except Exception as exc:
            raise AuthProtocolError("codex", "model list was malformed") from exc
        models = raw.get("models") if isinstance(raw, dict) else None
        if not isinstance(models, list):
            models = raw.get("data", []) if isinstance(raw, dict) else []
        return [
            str(item.get("slug") or item.get("id"))
            for item in models
            if isinstance(item, dict) and (item.get("slug") or item.get("id"))
        ]

    def query_server_context(self) -> int | None:
        return None


def _completed_response(response) -> dict:
    completed: dict | None = None
    try:
        lines = response.iter_lines(decode_unicode=True)
        for line in lines:
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            event_type = event.get("type") if isinstance(event, dict) else None
            if event_type in {
                "response.completed",
                "response.done",
                "response.incomplete",
            }:
                candidate = event.get("response")
                if isinstance(candidate, dict):
                    completed = dict(candidate)
                    completed.setdefault(
                        "status",
                        "incomplete"
                        if event_type == "response.incomplete"
                        else "completed",
                    )
            elif event_type in {"response.failed", "error"}:
                raise AuthProtocolError("codex", "subscription response failed")
    except AuthProtocolError:
        raise
    except Exception as exc:
        raise AuthProtocolError("codex", "subscription response was malformed") from exc
    if completed is None:
        raise AuthProtocolError("codex", "subscription response did not complete")
    return completed


__all__ = ["CodexSubscriptionClient"]
