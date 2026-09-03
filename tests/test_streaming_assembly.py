"""Streaming chat-completion assembly + failure-mode regression tests.

The streaming path (server/_streaming.py) consumes a Stream of
ChatCompletionChunk objects from the OpenAI SDK and assembles them
into a synthesized non-stream-shaped response. Downstream parsing in
client.py / chat_io.py is unchanged, so these tests pin the
assembler's contract directly:

  - basic content-only stream → correct concatenation
  - single tool call across multiple chunks → assembled correctly
  - multiple tool calls (parallel) → preserved by index, ordered
  - usage from include_usage final chunk → propagated
  - finish_reason from last content choice → propagated
  - mid-stream API errors → propagate (chat_with_retry handles)
  - stream ends without finish_reason → APIConnectionError (transient)
  - empty stream (no chunks) → APIConnectionError
  - malformed tool-call arguments JSON → handed to parse_args (returns {})
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import openai
import pytest

from scripts.llm_solver.server._streaming import (
    _StreamedChoice, _StreamedFunction, _StreamedMessage, _StreamedResponse,
    _StreamedToolCall, _StreamedUsage, assemble_stream,
)


# ── Chunk builders ────────────────────────────────────────────────────


def _chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    """Build a fake ChatCompletionChunk for the assembler.

    Uses SimpleNamespace so attribute access matches what the SDK
    returns (the assembler uses getattr with default None).
    """
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        role=None,
        function_call=None,
        refusal=None,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason, index=0)
    return SimpleNamespace(choices=[choice], usage=usage)


def _tc_chunk(
    index, *, id=None, name=None, arguments=None, type=None,
    extra_content=None,
):
    """Build a ChoiceDeltaToolCall fragment."""
    func = None
    if name is not None or arguments is not None:
        func = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(
        index=index,
        id=id,
        type=type,
        function=func,
        extra_content=extra_content,
    )


def _usage(prompt_tokens, completion_tokens):
    return SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
    )


# ── Basic assembly ────────────────────────────────────────────────────


def test_content_only_stream():
    chunks = [
        _chunk(content="hello "),
        _chunk(content="world"),
        _chunk(finish_reason="stop", usage=_usage(120, 2)),
    ]
    resp = assemble_stream(chunks)
    assert resp.choices[0].message.content == "hello world"
    assert resp.choices[0].message.tool_calls is None
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.prompt_tokens == 120
    assert resp.usage.completion_tokens == 2


def test_empty_content_returns_none():
    """Stream with no content deltas (only finish + usage) yields content=None."""
    chunks = [
        _chunk(finish_reason="stop", usage=_usage(50, 0)),
    ]
    resp = assemble_stream(chunks)
    assert resp.choices[0].message.content is None
    assert resp.choices[0].message.tool_calls is None


# ── Tool calls ────────────────────────────────────────────────────────


def test_single_tool_call_assembled_across_chunks():
    """Real-world chunk shape: tool call name + arguments arrive in
    multiple delta fragments. Assembler must concat into one call."""
    chunks = [
        _chunk(tool_calls=[_tc_chunk(0, id="call_abc", type="function", name="bash")]),
        _chunk(tool_calls=[_tc_chunk(0, arguments='{"cm')]),
        _chunk(tool_calls=[_tc_chunk(0, arguments='d":"l')]),
        _chunk(tool_calls=[_tc_chunk(0, arguments='s -la"}')]),
        _chunk(finish_reason="tool_calls", usage=_usage(2400, 30)),
    ]
    resp = assemble_stream(chunks)
    tcs = resp.choices[0].message.tool_calls
    assert tcs is not None and len(tcs) == 1
    assert tcs[0].id == "call_abc"
    assert tcs[0].type == "function"
    assert tcs[0].function.name == "bash"
    assert tcs[0].function.arguments == '{"cmd":"ls -la"}'
    assert resp.choices[0].finish_reason == "tool_calls"


def test_tool_call_extra_content_survives_stream_assembly():
    extra_content = {
        "google": {"thought_signature": "opaque-provider-signature"},
    }
    chunks = [
        _chunk(tool_calls=[_tc_chunk(
            0,
            id="call_abc",
            type="function",
            name="read",
            arguments='{"path":"x.py"}',
            extra_content=extra_content,
        )]),
        _chunk(finish_reason="tool_calls", usage=_usage(100, 10)),
    ]
    response = assemble_stream(chunks)
    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.extra_content == extra_content
    dumped = json.loads(response.model_dump_json())
    assert dumped["choices"][0]["message"]["tool_calls"][0][
        "extra_content"
    ] == extra_content


def test_multiple_tool_calls_preserve_index_order():
    """Parallel tool calls arrive interleaved by index; final list is index-sorted."""
    chunks = [
        _chunk(tool_calls=[_tc_chunk(0, id="c0", name="read")]),
        _chunk(tool_calls=[_tc_chunk(1, id="c1", name="grep")]),
        _chunk(tool_calls=[_tc_chunk(0, arguments='{"path":"a"}')]),
        _chunk(tool_calls=[_tc_chunk(1, arguments='{"pattern":"x"}')]),
        _chunk(tool_calls=[_tc_chunk(2, id="c2", name="bash", arguments='{"cmd":"echo"}')]),
        _chunk(finish_reason="tool_calls", usage=_usage(100, 20)),
    ]
    resp = assemble_stream(chunks)
    tcs = resp.choices[0].message.tool_calls
    assert tcs is not None and len(tcs) == 3
    assert [tc.id for tc in tcs] == ["c0", "c1", "c2"]
    assert tcs[0].function.name == "read"
    assert tcs[0].function.arguments == '{"path":"a"}'
    assert tcs[2].function.arguments == '{"cmd":"echo"}'


def test_tool_call_index_arrives_out_of_order():
    """Index 1's chunks come before index 0's — final list is still ordered."""
    chunks = [
        _chunk(tool_calls=[_tc_chunk(1, id="c1", name="b", arguments='{}')]),
        _chunk(tool_calls=[_tc_chunk(0, id="c0", name="a", arguments='{}')]),
        _chunk(finish_reason="tool_calls", usage=_usage(50, 10)),
    ]
    resp = assemble_stream(chunks)
    tcs = resp.choices[0].message.tool_calls
    assert [tc.id for tc in tcs] == ["c0", "c1"]


# ── Failure modes — chat_with_retry will see these ───────────────────


def test_mid_stream_internal_server_error_propagates():
    """A 500 partway through iteration must propagate as
    InternalServerError so chat_with_retry's _TRANSIENT_ERRORS catches it."""
    def gen():
        yield _chunk(content="partial ")
        raise openai.InternalServerError(
            message="upstream 500",
            response=MagicMock(status_code=500, request=MagicMock()),
            body=None,
        )
    with pytest.raises(openai.InternalServerError):
        assemble_stream(gen())


def test_mid_stream_connection_drop_propagates():
    """APIConnectionError raised mid-stream must propagate (transient)."""
    def gen():
        yield _chunk(content="some ")
        yield _chunk(content="more text ")
        raise openai.APIConnectionError(
            message="connection reset", request=MagicMock(),
        )
    with pytest.raises(openai.APIConnectionError):
        assemble_stream(gen())


def test_stream_without_finish_reason_raises_transient():
    """Server hung up partway through — no finish_reason in any chunk.
    The assembler must raise APIConnectionError so chat_with_retry retries
    instead of returning a malformed-looking complete response."""
    chunks = [
        _chunk(content="hello"),
        _chunk(content=" world"),
        # No final chunk with finish_reason — server cut us off.
    ]
    with pytest.raises(openai.APIConnectionError, match="finish_reason"):
        assemble_stream(chunks)


def test_empty_stream_raises_transient():
    """Server returned no chunks at all — also transient."""
    with pytest.raises(openai.APIConnectionError, match="no chunks"):
        assemble_stream([])


def test_partial_arguments_json_left_for_parse_args():
    """If the stream completes with a finish_reason but the model emitted
    malformed JSON in the tool-call arguments, the assembler hands the
    raw string through unchanged. parse_args() in client.py handles
    JSONDecodeError by returning {} — same behavior as the non-stream
    path on a malformed response. We only verify the assembler's
    contract: pass-through, no exception."""
    chunks = [
        _chunk(tool_calls=[_tc_chunk(0, id="c0", name="bash", arguments='{"cmd":"l')]),
        # finish_reason present, but arguments is incomplete JSON.
        _chunk(finish_reason="tool_calls", usage=_usage(100, 5)),
    ]
    resp = assemble_stream(chunks)
    tcs = resp.choices[0].message.tool_calls
    assert tcs is not None and len(tcs) == 1
    # Raw bytes preserved — parse_args downstream will return {}.
    assert tcs[0].function.arguments == '{"cmd":"l'


# ── Usage propagation edge cases ──────────────────────────────────────


def test_usage_only_in_final_chunk():
    """include_usage emits usage in the FINAL chunk (after content choices
    are done). The assembler must capture it from any chunk that has it."""
    chunks = [
        _chunk(content="hi"),
        _chunk(finish_reason="stop"),
        _chunk(usage=_usage(80, 1)),  # usage in trailing chunk
    ]
    resp = assemble_stream(chunks)
    assert resp.usage.prompt_tokens == 80
    assert resp.usage.completion_tokens == 1
    assert resp.choices[0].finish_reason == "stop"


def test_usage_zero_when_server_omits_it():
    """Some servers don't honor stream_options=include_usage. Assembler
    must default to 0 rather than crash on missing usage."""
    chunks = [
        _chunk(content="ok"),
        _chunk(finish_reason="stop"),
    ]
    resp = assemble_stream(chunks)
    assert resp.usage.prompt_tokens == 0
    assert resp.usage.completion_tokens == 0


# ── _StreamedResponse shape contract — quacks like the SDK response ──


def test_streamed_response_attribute_access_matches_sdk_shape():
    """Downstream code does:
      msg = resp.choices[0].message
      reason = resp.choices[0].finish_reason
      msg.content; msg.tool_calls
      tc.function.name; tc.function.arguments; tc.id
      resp.usage.prompt_tokens / .completion_tokens
    The synthesized response must support every one of these."""
    chunks = [
        _chunk(tool_calls=[_tc_chunk(0, id="c0", name="bash", arguments='{}')]),
        _chunk(content="hi"),
        _chunk(finish_reason="tool_calls", usage=_usage(10, 1)),
    ]
    resp = assemble_stream(chunks)
    assert resp.choices[0].finish_reason == "tool_calls"
    msg = resp.choices[0].message
    assert msg.content == "hi"
    assert msg.tool_calls is not None
    tc = msg.tool_calls[0]
    assert tc.id == "c0"
    assert tc.function.name == "bash"
    assert tc.function.arguments == "{}"
    assert resp.usage.prompt_tokens == 10
    assert resp.usage.completion_tokens == 1


def test_streamed_response_model_dump_json_for_transcript():
    """Verbatim transcript uses resp.model_dump_json() — ensure the
    streamed response provides one that round-trips a sane shape."""
    import json
    resp = _StreamedResponse(
        choices=[_StreamedChoice(
            message=_StreamedMessage(
                content="hi",
                tool_calls=[_StreamedToolCall(
                    id="x", type="function",
                    function=_StreamedFunction(name="bash", arguments='{"cmd":"ls"}'),
                )],
            ),
            finish_reason="tool_calls",
        )],
        usage=_StreamedUsage(prompt_tokens=10, completion_tokens=2),
    )
    d = json.loads(resp.model_dump_json())
    assert d["choices"][0]["message"]["content"] == "hi"
    assert d["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "bash"
    assert d["choices"][0]["finish_reason"] == "tool_calls"
    assert d["usage"]["prompt_tokens"] == 10
    assert d["_streamed"] is True


# ── End-to-end: streaming routed through _call_api ────────────────────


def test_call_api_streaming_path_uses_stream_true(monkeypatch):
    """When YUJ_STREAMING=1, _call_api passes stream=True + stream_options."""
    monkeypatch.setenv("YUJ_STREAMING", "1")
    from scripts.llm_solver.server.client import LlamaClient
    from scripts.llm_solver.config import Config
    from tests._config_helpers import make_config
    cfg = make_config()
    client = LlamaClient(cfg, profile=None)

    captured_payload = {}

    def fake_create(**payload):
        captured_payload.update(payload)
        # Return an iterator of chunks like the SDK would.
        return iter([
            _chunk(content="ok"),
            _chunk(finish_reason="stop", usage=_usage(40, 1)),
        ])

    with patch.object(client.client.chat.completions, "create",
                       side_effect=fake_create):
        resp = client._call_api({"model": "test", "messages": [], "max_tokens": 100})

    assert captured_payload.get("stream") is True
    assert captured_payload.get("stream_options") == {"include_usage": True}
    assert resp.choices[0].message.content == "ok"
    assert resp.usage.prompt_tokens == 40


def test_call_api_non_streaming_when_disabled(monkeypatch):
    """When YUJ_STREAMING is unset (default OFF), _call_api uses the
    legacy non-stream path — no stream= or stream_options in payload."""
    monkeypatch.delenv("YUJ_STREAMING", raising=False)
    from scripts.llm_solver.server.client import LlamaClient
    from tests._config_helpers import make_config
    cfg = make_config()
    client = LlamaClient(cfg, profile=None)

    captured_payload = {}

    def fake_create(**payload):
        captured_payload.update(payload)
        m = MagicMock()
        m.model_dump_json.return_value = "{}"
        return m

    with patch.object(client.client.chat.completions, "create",
                       side_effect=fake_create):
        client._call_api({"model": "test", "messages": [], "max_tokens": 100})

    assert "stream" not in captured_payload
    assert "stream_options" not in captured_payload


def test_call_api_off_switch_with_yuj_streaming_zero(monkeypatch):
    """YUJ_STREAMING=0 explicitly disables streaming."""
    monkeypatch.setenv("YUJ_STREAMING", "0")
    from scripts.llm_solver.server.client import LlamaClient
    from tests._config_helpers import make_config
    cfg = make_config()
    client = LlamaClient(cfg, profile=None)

    captured_payload = {}

    def fake_create(**payload):
        captured_payload.update(payload)
        m = MagicMock()
        m.model_dump_json.return_value = "{}"
        return m

    with patch.object(client.client.chat.completions, "create",
                       side_effect=fake_create):
        client._call_api({"model": "test", "messages": [], "max_tokens": 100})

    assert "stream" not in captured_payload
