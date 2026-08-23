"""LlamaClient — OpenAI SDK wrapper with profile-based normalize/denormalize."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import openai

from ..config import Config

if TYPE_CHECKING:
    from .profile_loader import Profile

from ._streaming import assemble_stream
from .request_controls import apply_request_controls, usage_from_response
from .types import SideRequestResult, ToolCall


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
        # File handle held for the task lifetime to avoid 4 syscalls/turn
        # (2x open + close for input + output) — close_transcript() releases it.
        self._transcript_path: Path | None = None
        self._transcript_file = None
        self._transcript_call_n: int = 0
        self._session_id: str = ""

    def set_session_id(self, session_id: str) -> None:
        """Bind subsequent requests to one stable product session identity."""
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        self._session_id = session_id

    def _attach_request_controls(
        self,
        payload: dict,
        *,
        side_request: bool,
        policy_extra: dict | None = None,
    ) -> dict:
        """Apply configured extras and final llama cache policy."""
        return apply_request_controls(
            payload,
            session_id=getattr(self, "_session_id", ""),
            server_request_extra=getattr(
                self.cfg, "server_request_extra", {}
            ) or {},
            cache_affinity=getattr(self.cfg, "cache_affinity", False),
            cache_retention=getattr(self.cfg, "cache_retention", "off"),
            side_request=side_request,
            policy_extra=policy_extra,
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
        if _streaming_enabled():
            stream_payload = dict(payload)
            stream_payload["stream"] = True
            # include_usage on the final chunk gives us prompt_tokens
            # / completion_tokens — same fields the non-stream usage
            # block carries. Required for the harness's
            # _last_actual_prompt_tokens signal.
            stream_payload["stream_options"] = {"include_usage": True}
            try:
                stream = self.client.chat.completions.create(**stream_payload)
                resp = assemble_stream(stream)
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
            usage=usage_from_response(response),
        )

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
        from .types import TurnResult

        profile = self.profile

        # DENORM_IN: canonical messages as harness sent them
        log.debug("DENORM_IN messages=%d tools=%d", len(messages), len(tools))

        # Denormalize messages for wire format
        wire_messages = profile.denormalize_messages(messages)

        # DENORM_OUT: wire-format messages after denormalization
        log.debug("DENORM_OUT messages=%d", len(wire_messages))

        payload = {
            "model": self.cfg.model,
            "messages": wire_messages,
            "max_tokens": self.cfg.max_tokens,
        }
        if profile.supports_tool_calls:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        elif tools:
            log.info("Profile %s reports supports_tool_calls=false; omitting tool schema payload", profile.name)

        payload = self._attach_request_controls(payload, side_request=False)
        resp = self._call_api(payload)

        msg = resp.choices[0].message
        reason = resp.choices[0].finish_reason or "stop"
        usage = usage_from_response(resp)

        # Build raw response dict for normalize pipeline
        raw_tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                raw_tool_calls.append({
                    "id": getattr(tc, "id", None) or "",
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        raw_response = {
            "content": getattr(msg, "content", None),
            "tool_calls": raw_tool_calls,
            "finish_reason": reason,
        }

        # NORM_IN: raw model response
        log.debug("NORM_IN content=%s tool_calls=%d finish_reason=%s",
                   repr(raw_response["content"][:100]) if raw_response["content"] else None,
                   len(raw_tool_calls), reason)

        # Normalize
        normalized = profile.normalize(raw_response)

        # NORM_OUT: canonical TurnResult
        content = normalized.get("content")
        if content == "":
            content = None
        norm_reason = normalized.get("finish_reason", reason)
        norm_tool_calls_raw = normalized.get("tool_calls", [])

        # Build canonical ToolCall list
        tool_calls: list[ToolCall] = []
        if norm_tool_calls_raw and norm_reason in ("tool_calls", "tool"):
            for i, tc in enumerate(norm_tool_calls_raw):
                if isinstance(tc, dict):
                    tc_id = f"call_{turn}_{i}"  # deterministic; server IDs are random
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    arguments = parse_args(func.get("arguments", "{}"))
                    tool_calls.append(ToolCall(id=tc_id, name=name, arguments=arguments))

        log.debug("NORM_OUT content=%s tool_calls=%d finish_reason=%s",
                   repr(content[:100]) if content else None,
                   len(tool_calls), norm_reason)

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
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": self.cfg.max_tokens,
        }, side_request=False)
        resp = self._call_api(payload)

        msg = resp.choices[0].message
        reason = resp.choices[0].finish_reason or "stop"
        usage = usage_from_response(resp)

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
                tool_calls.append(ToolCall(id=tc_id, name=name, arguments=arguments))

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
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in tool_calls
            ]
        return msg
