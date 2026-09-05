"""LlamaClient — OpenAI SDK wrapper with profile-based normalize/denormalize."""
from __future__ import annotations

import base64
import copy
import json
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import openai

from ..config import Config

if TYPE_CHECKING:
    from .profile_loader import Profile

from ._streaming import StreamRuleInterrupt, assemble_stream
from . import request_controls
from .types import ImageInput, SideRequestResult, ToolCall, TurnResult, Usage


def _streaming_enabled() -> bool:
    """Return whether ``YUJ_STREAMING=1`` enables streaming for this process.

    Streaming is off by default.
    """
    return os.environ.get("YUJ_STREAMING", "0") == "1"

log = logging.getLogger(__name__)


# Legacy helpers — kept for backward compatibility, superseded by profile pipelines.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(content: str | None) -> str | None:
    """Remove <think>...</think> blocks from model output."""
    if not content:
        return content
    cleaned = _THINK_RE.sub("", content).strip()
    return cleaned or None


def parse_args(raw) -> dict:
    """Handle arguments as dict (llama-server bug #20198) or JSON string."""
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        log.warning("Failed to parse tool arguments: %r", raw)
        return {}


def _member(value, name: str, default=None):
    """Read one field from an SDK object or a primitive response mapping."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _tool_call_extra_content(tool_call) -> dict | None:
    """Copy an opaque provider mapping attached to one tool call."""
    extra_content = _member(tool_call, "extra_content")
    if extra_content is None:
        model_extra = _member(tool_call, "model_extra", {}) or {}
        if isinstance(model_extra, Mapping):
            extra_content = model_extra.get("extra_content")
    if not isinstance(extra_content, Mapping):
        return None
    return copy.deepcopy(dict(extra_content))


class LlamaClient:
    """OpenAI SDK wrapper with profile-driven normalize/denormalize pipelines."""

    def __init__(self, cfg: Config, profile: Profile | None = None):
        self.cfg = cfg
        self.profile = profile
        self.client = openai.OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            max_retries=0,
            timeout=openai.Timeout(
                connect=cfg.timeout_connect,
                read=cfg.timeout_read,
                write=cfg.timeout_read,
                pool=cfg.timeout_connect,
            ),
        )
        # Write the verbatim transcript at the HTTP boundary.
        # Set per task via set_transcript(); each HTTP call is one input/output
        # pair tagged only with a monotonic call counter and direction.
        # Held for the task lifetime to avoid 4 syscalls/turn; close_transcript releases it.
        self._transcript_path: Path | None = None
        self._transcript_file = None
        self._transcript_call_n: int = 0
        self._session_id: str = ""
        self._thinking_resolution = None
        self._thinking_signature = None
        # Bound by chat_io for one logical solver response.  Keeping the
        # observer out of the request payload preserves provider/profile
        # behavior while exposing SSE deltas to the owning harness layer.
        self._stream_observer = None
        self._last_call_streamed = False
        self._image_inputs: tuple[ImageInput, ...] = ()
        self._image_target_correction: str | None = None
        _ = self.thinking_resolution

    @property
    def thinking_resolution(self):
        requested = getattr(self.cfg, "thinking_level", "off")
        signature = (requested, id(self.profile))
        if getattr(self, "_thinking_signature", None) != signature:
            levels = getattr(self.profile, "reasoning_levels", None)
            self._thinking_resolution = request_controls.resolve_thinking_level(
                requested, levels or request_controls.DEFAULT_REASONING_LEVELS
            )
            self._thinking_signature = signature
        return self._thinking_resolution

    def set_session_id(self, session_id: str) -> None:
        """Bind subsequent requests to one stable product session identity."""
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        self._session_id = session_id

    def set_image_inputs(
        self, images: list[ImageInput] | tuple[ImageInput, ...]
    ) -> None:
        """Bind validated images to assistant conversation requests."""
        selected = tuple(images)
        for image in selected:
            if not isinstance(image, ImageInput):
                raise TypeError("image inputs must be ImageInput values")
        self._image_inputs = selected

    def preserve_image_target_before_correction(self, text: str) -> None:
        """Keep one correction from becoming the image-bearing user turn."""
        if not isinstance(text, str) or not text:
            raise ValueError("correction text must be a non-empty string")
        self._image_target_correction = text

    def _messages_with_image_inputs(self, messages: list[dict]) -> list[dict]:
        """Attach images to the latest eligible text user turn."""
        images = self._image_inputs
        if not images:
            return messages
        correction_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if messages[index].get("role") == "user"
                and messages[index].get("content")
                == self._image_target_correction
            ),
            None,
        )
        selected_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if messages[index].get("role") == "user"
                and isinstance(messages[index].get("content"), str)
                and index != correction_index
            ),
            None,
        )
        if selected_index is None:
            raise ValueError("image inputs require a text user message")
        wire_messages = list(messages)
        selected_message = dict(wire_messages[selected_index])
        content: list[dict] = []
        for image in images:
            encoded = base64.b64encode(image.data).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image.media_type};base64,{encoded}",
                },
            })
        content.append({"type": "text", "text": selected_message["content"]})
        selected_message["content"] = content
        wire_messages[selected_index] = selected_message
        return wire_messages

    def _attach_request_controls(
        self,
        payload: dict,
        *,
        side_request: bool,
        policy_extra: dict | None = None,
    ) -> dict:
        """Apply configured extras and the selected request-body policy."""
        if side_request:
            policy_extra = {"chat_template_kwargs": {"enable_thinking": False}}
        elif policy_extra is None:
            policy_extra = dict(self.thinking_resolution.request_extra)
        return request_controls.apply_request_controls(
            payload,
            session_id=getattr(self, "_session_id", ""),
            server_request_extra=getattr(
                self.cfg, "server_request_extra", {}
            ) or {},
            cache_affinity=getattr(self.cfg, "cache_affinity", False),
            cache_retention=getattr(self.cfg, "cache_retention", "session"),
            side_request=side_request,
            policy_extra=policy_extra,
            request_dialect=getattr(self.cfg, "request_dialect", "llama"),
        )

    def set_transcript(self, path: Path | None, append: bool = False) -> None:
        """Enable verbatim transcript at `path`. Truncates and resets counter.

        append=True continues an existing diary. During replay handover,
        the live client keeps writing the file that contains the replay
        prefix. This leaves one complete conversation record.

        Pass None to disable. Each HTTP call writes one input block and one
        output block, separated only by `=== turn NNN input ===` markers.
        Raw bytes — no pretty-printing, no transformation, no per-call tags
        beyond the counter and direction. File handle is opened here and
        held open until close_transcript() or a subsequent set_transcript().
        """
        self.close_transcript()
        self._transcript_path = path
        self._transcript_call_n = 0
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Truncate on open (append=True continues); handle kept open.
            self._transcript_file = open(path, "a" if append else "w")

    def close_transcript(self) -> None:
        """Release the transcript file handle, if any."""
        if self._transcript_file is not None:
            try:
                self._transcript_file.close()
            except OSError:
                pass
            self._transcript_file = None

    def _write_transcript(self, marker: str, body: str) -> None:
        if self._transcript_file is None:
            return
        self._transcript_file.write(f"=== {marker} ===\n")
        self._transcript_file.write(body)
        if not body.endswith("\n"):
            self._transcript_file.write("\n")
        self._transcript_file.flush()

    @property
    def supports_stream_observer(self) -> bool:
        # An adapter replacing this transport must explicitly provide its own
        # observer support before autonomous execution can rely on it.
        return type(self)._call_api is LlamaClient._call_api

    def _call_api(self, payload: dict, *, record_transcript: bool = True):
        """Send one HTTP request and save its payload and response.

        Routes to streaming when YUJ_STREAMING is on. The
        streaming path assembles SSE chunks into a synthesized
        non-stream-shaped response so downstream parsing in
        _chat_with_profile / _chat_legacy is unchanged. Errors raised
        mid-stream propagate up identically to the non-stream path
        and are classified by chat_with_retry's _TRANSIENT_ERRORS
        tuple. See server/_streaming.py for the assembly contract.
        """
        n = 0
        if record_transcript:
            self._transcript_call_n += 1
            n = self._transcript_call_n
            self._write_transcript(
                f"turn {n:03d} input",
                json.dumps(payload, default=str),
            )
        if _streaming_enabled() or getattr(self, "_narration_streaming", False):
            self._last_call_streamed = True
            stream_payload = dict(payload)
            stream_payload["stream"] = True
            # include_usage on the final chunk gives us prompt_tokens
            # / completion_tokens — same fields the non-stream usage
            # block carries. Required for the harness's
            # _last_actual_prompt_tokens signal.
            stream_payload["stream_options"] = {"include_usage": True}
            try:
                stream = self.client.chat.completions.create(**stream_payload)
                resp = assemble_stream(
                    stream, observer=getattr(self, "_stream_observer", None)
                )
            except StreamRuleInterrupt as e:
                # This is an intentional, replayable response outcome, not a
                # transport failure.  Store valid JSON under the ordinary
                # output marker so ReplayClient can reproduce the retry.
                if record_transcript:
                    self._write_transcript(
                        f"turn {n:03d} output", e.model_dump_json()
                    )
                raise
            except Exception as e:
                if record_transcript:
                    self._write_transcript(
                        f"turn {n:03d} output (stream error)",
                        f"{type(e).__name__}: {e}",
                    )
                raise
            if record_transcript:
                self._write_transcript(f"turn {n:03d} output", resp.model_dump_json())
            return resp
        self._last_call_streamed = False
        try:
            resp = self.client.chat.completions.create(**payload)
        except Exception as e:
            if record_transcript:
                self._write_transcript(
                    f"turn {n:03d} output", f"{type(e).__name__}: {e}"
                )
            raise
        try:
            body = resp.model_dump_json()
        except AttributeError:
            body = json.dumps(resp, default=str)
        if record_transcript:
            self._write_transcript(f"turn {n:03d} output", body)
        return resp

    def complete_side_request(self, payload: dict) -> SideRequestResult:
        """Send one harness-owned no-tool completion without a solver turn.

        Side requests share this client's endpoint, model profile, and HTTP
        transport, but they never add ``tools``/``tool_choice`` and never
        write ``=== turn N ===`` transcript blocks. This keeps transcript
        resume parsing scoped to the actual solver conversation.
        """
        if "tools" in payload or "tool_choice" in payload:
            raise ValueError("side requests must omit tools and tool_choice")
        request = dict(payload)
        messages = request.get("messages")
        if not isinstance(messages, list):
            raise ValueError("side request messages must be a list")
        if self.profile is not None:
            request["messages"] = self.profile.denormalize_messages(messages)
        request["model"] = self.cfg.model
        request = self._attach_request_controls(request, side_request=True)
        response = self._call_api(request, record_transcript=False)
        message = response.choices[0].message
        if getattr(message, "tool_calls", None):
            raise ValueError("side request returned tool calls")
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            raise ValueError("side request returned no text content")
        return SideRequestResult(
            content=content,
            usage=request_controls.usage_from_response(response),
        )

    def complete_tool_side_request(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        turn: int = 0,
    ) -> TurnResult:
        """Run an isolated harness-owned tool conversation step.

        Unlike a primary solver turn, this request never writes the primary
        transcript and always applies side-request cache/thinking controls.
        Callers must still enforce their own tool allowlist before dispatch.
        """
        if not isinstance(messages, list):
            raise ValueError("tool side-request messages must be a list")
        if not isinstance(tools, list):
            raise ValueError("tool side-request tools must be a list")

        if self.profile is not None:
            request: dict = {
                "model": self.cfg.model,
                "messages": self.profile.denormalize_messages(messages),
                "max_tokens": self.cfg.max_tokens,
            }
            if self.profile.supports_tool_calls:
                request["tools"] = tools
                request["tool_choice"] = "auto"
            request = self._attach_request_controls(request, side_request=True)
            raw_response = self._call_raw_profile_request(
                request, record_transcript=False
            )
            normalized = self.profile.normalize(dict(raw_response))
            return self._turn_result_from_normalized(
                normalized,
                raw_response["usage"],
                turn,
                fallback_finish_reason=raw_response["finish_reason"],
            )

        request = self._attach_request_controls(
            {
                "model": self.cfg.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": self.cfg.max_tokens,
            },
            side_request=True,
        )
        response = self._call_api(request, record_transcript=False)
        return self._legacy_turn_result_from_response(response, turn)

    def health_check(self) -> list[str]:
        """Verify server is reachable via /v1/models. Raises on connection failure."""
        resp = self.client.models.list()
        return [m.id for m in resp.data]

    def _server_root(self) -> str:
        """Return the HTTP root for llama-compatible side endpoints."""
        base = self.cfg.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return base.rstrip("/")

    def query_server_metadata(self) -> dict:
        """Snapshot available server metadata endpoints.

        This is provenance, not control flow. The harness cannot infer every
        launch flag from the OpenAI API, but llama.cpp-compatible runtimes
        expose enough via /props, /slots, and /v1/models to pin model/template/
        sampling facts when the server reports them. Callers hash the returned
        body and store it once per run.
        """
        import requests

        root = self._server_root()
        snapshot: dict = {
            "base_url": self.cfg.base_url,
            "root_url": root,
            "endpoints": {},
        }
        for endpoint in ("/props", "/slots", "/v1/models"):
            record: dict
            try:
                resp = requests.get(f"{root}{endpoint}", timeout=5)
                record = {
                    "ok": bool(resp.ok),
                    "status_code": resp.status_code,
                }
                if resp.ok:
                    try:
                        record["json"] = resp.json()
                    except Exception as e:
                        record["json_error"] = f"{type(e).__name__}: {e}"
                        record["text_head"] = resp.text[:20000]
                else:
                    record["text_head"] = resp.text[:2000]
            except Exception as e:
                record = {
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                }
            snapshot["endpoints"][endpoint] = record

        if any(r.get("ok") for r in snapshot["endpoints"].values()):
            return snapshot
        return {}

    def query_server_context(self) -> int | None:
        """Query the server's effective n_ctx. Returns None if unavailable.

        Tries /props, /slots in order. Works on any platform — it's an HTTP
        call, not a hardware query. The server already resolved VRAM constraints
        when it started.
        """
        import requests

        base = self._server_root()
        for endpoint in ("/props", "/slots"):
            try:
                resp = requests.get(f"{base}{endpoint}", timeout=5)
                if not resp.ok:
                    continue
                data = resp.json()
                if endpoint == "/props":
                    n_ctx = data.get("default_generation_settings", {}).get("n_ctx")
                    if n_ctx:
                        return int(n_ctx)
                elif endpoint == "/slots":
                    if isinstance(data, list) and data:
                        n_ctx = data[0].get("n_ctx")
                        if n_ctx:
                            return int(n_ctx)
            except Exception:
                continue
        return None

    def chat(
        self, messages: list[dict], tools: list[dict], turn: int = 0
    ):
        """Single API call. Returns TurnResult (iterable as 4-tuple for backward compat).

        When a profile is loaded: denormalize before HTTP, normalize after.
        Without profile: legacy ad-hoc quirk handling.
        """

        if self.profile:
            return self._chat_with_profile(messages, tools, turn)
        return self._chat_legacy(messages, tools, turn)

    def _chat_with_profile(
        self, messages: list[dict], tools: list[dict], turn: int
    ):
        """Profile-driven chat: denormalize → HTTP → normalize → TurnResult."""
        request = self._prepare_profile_chat_request(messages, tools)
        raw_response = self._call_raw_profile_request(request)
        normalized = self.profile.normalize(dict(raw_response))
        return self._turn_result_from_normalized(
            normalized,
            raw_response["usage"],
            turn,
            fallback_finish_reason=raw_response["finish_reason"],
        )

    def _prepare_profile_chat_request(
        self, messages: list[dict], tools: list[dict]
    ) -> dict:
        """Build the exact profile-denormalized first-call request."""
        profile = self.profile
        if profile is None:
            raise RuntimeError("profile request preparation requires a profile")
        log.debug("DENORM_IN messages=%d tools=%d", len(messages), len(tools))
        wire_messages = profile.denormalize_messages(messages)
        wire_messages = self._messages_with_image_inputs(wire_messages)
        log.debug("DENORM_OUT messages=%d", len(wire_messages))
        request = {
            "model": self.cfg.model,
            "messages": wire_messages,
            "max_tokens": self.cfg.max_tokens,
        }
        if profile.supports_tool_calls:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        elif tools:
            log.info("Profile %s reports supports_tool_calls=false; omitting tool schema payload", profile.name)
        request = self._attach_constrained_tool_decoding(request)
        return self._attach_request_controls(request, side_request=False)

    def _attach_constrained_tool_decoding(self, request: dict) -> dict:
        """Attach a profile-approved constraint to a normal tool-call request."""
        tools = request.get("tools")
        mode = getattr(self.cfg, "tools_constrained_decoding", "off")
        if mode == "off" or not isinstance(tools, list) or not tools:
            return request
        from ..harness.tool_validation import (
            ToolSchemaSet,
            attach_constrained_decoding,
            resolve_constrained_decoding,
        )

        resolution = resolve_constrained_decoding(
            mode=mode,
            schemas=ToolSchemaSet.from_openai_tools(tools),
            supports_constrained_tools=getattr(
                self.profile, "supports_constrained_tools", False
            ),
        )
        if resolution.fallback_reason:
            log.warning(
                "constrained decoding disabled for profile %s: %s",
                getattr(self.profile, "name", ""),
                resolution.fallback_reason,
            )
        return attach_constrained_decoding(request, resolution)

    def _call_raw_profile_request(
        self, request: dict, *, record_transcript: bool = True
    ) -> dict:
        """Make one profile request and return its pre-normalize response."""
        resp = self._call_api(request, record_transcript=record_transcript)
        choices = _member(resp, "choices", ()) or ()
        if not choices:
            raise ValueError("model response contains no choices")
        choice = choices[0]
        msg = _member(choice, "message")
        if msg is None:
            raise ValueError("model response choice contains no message")
        reason = _member(choice, "finish_reason") or "stop"
        usage = request_controls.usage_from_response(resp)
        raw_tool_calls = []
        for tc in (_member(msg, "tool_calls", ()) or ()):
            function = _member(tc, "function", {}) or {}
            raw_tool_call = {
                "id": _member(tc, "id", "") or "",
                "type": "function",
                "function": {
                    "name": _member(function, "name", "") or "",
                    "arguments": _member(function, "arguments", "{}"),
                },
            }
            extra_content = _tool_call_extra_content(tc)
            if extra_content is not None:
                raw_tool_call["extra_content"] = extra_content
            raw_tool_calls.append(raw_tool_call)

        raw_response = {
            "content": _member(msg, "content"),
            "tool_calls": raw_tool_calls,
            "finish_reason": reason,
            "usage": usage,
        }
        log.debug(
            "NORM_IN content=%s tool_calls=%d finish_reason=%s",
            repr(raw_response["content"][:100])
            if raw_response["content"] else None,
            len(raw_tool_calls),
            reason,
        )
        return raw_response

    def _turn_result_from_normalized(
        self,
        normalized: Mapping,
        usage: Usage,
        turn: int,
        *,
        fallback_finish_reason: str = "stop",
    ) -> TurnResult:
        """Build the canonical turn without invoking normalize again."""
        if not isinstance(normalized, Mapping):
            raise TypeError("normalized response must be a mapping")
        if not isinstance(usage, Usage):
            raise TypeError("response usage must be canonical Usage")
        content = normalized.get("content")
        if content == "":
            content = None
        norm_reason = (
            normalized.get("finish_reason") or fallback_finish_reason
        )
        norm_tool_calls_raw = normalized.get("tool_calls", [])

        tool_calls: list[ToolCall] = []
        if norm_tool_calls_raw and norm_reason in ("tool_calls", "tool"):
            for i, tc in enumerate(norm_tool_calls_raw):
                if isinstance(tc, Mapping):
                    tc_id = f"call_{turn}_{i}"
                    func = tc.get("function", {})
                    name = _member(func, "name", "") or ""
                    arguments = parse_args(_member(func, "arguments", "{}"))
                    tool_calls.append(
                        ToolCall(
                            id=tc_id,
                            name=name,
                            arguments=arguments,
                            extra_content=_tool_call_extra_content(tc),
                        )
                    )

        log.debug(
            "NORM_OUT content=%s tool_calls=%d finish_reason=%s",
            repr(content[:100]) if content else None,
            len(tool_calls),
            norm_reason,
        )
        return TurnResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason=norm_reason,
            usage=usage,
        )

    def _chat_legacy(
        self, messages: list[dict], tools: list[dict], turn: int
    ):
        """Legacy chat without profile — ad-hoc quirk handling."""
        from .types import TurnResult

        payload = self._attach_request_controls({
            "model": self.cfg.model,
            "messages": self._messages_with_image_inputs(messages),
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": self.cfg.max_tokens,
        }, side_request=False)
        resp = self._call_api(payload)
        return self._legacy_turn_result_from_response(resp, turn)

    def _legacy_turn_result_from_response(self, resp, turn: int) -> TurnResult:
        """Normalize one legacy response without making or recording a call."""

        msg = resp.choices[0].message
        reason = resp.choices[0].finish_reason or "stop"
        usage = request_controls.usage_from_response(resp)

        # Strip thinking blocks from content
        raw_content = getattr(msg, "content", None)
        if raw_content and "<think>" in raw_content:
            log.debug("Stripped thinking block (%d chars)", len(raw_content))
        content = strip_thinking(raw_content)

        # Parse tool calls with quirk handling
        tool_calls: list[ToolCall] = []
        if msg.tool_calls and reason in ("tool_calls", "tool"):
            for i, tc in enumerate(msg.tool_calls):
                tc_id = f"call_{turn}_{i}"  # deterministic; server IDs are random
                name = tc.function.name
                arguments = parse_args(tc.function.arguments)
                tool_calls.append(ToolCall(
                    id=tc_id,
                    name=name,
                    arguments=arguments,
                    extra_content=_tool_call_extra_content(tc),
                ))

        return TurnResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason=reason,
            usage=usage,
        )

    def build_assistant_message(
        self, content: str | None, tool_calls: list[ToolCall]
    ) -> dict:
        """Build a history-safe assistant message dict."""
        msg: dict = {"role": "assistant", "content": content}
        if tool_calls:
            wire_tool_calls = []
            for tc in tool_calls:
                wire_tool_call = {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                if tc.extra_content is not None:
                    wire_tool_call["extra_content"] = copy.deepcopy(
                        tc.extra_content
                    )
                wire_tool_calls.append(wire_tool_call)
            msg["tool_calls"] = wire_tool_calls
        return msg
