"""Runtime, artifact, replay, and startup proofs for mid-stream rules."""
from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._loop._driver_setup import load_session_stream_rules
from scripts.llm_solver.harness.context import FullTranscript
from scripts.llm_solver.harness.context_strategies.compact_transcript import (
    CompactTranscript,
)
from scripts.llm_solver.harness.injections import Injection
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.harness.state_writer import project
from scripts.llm_solver.harness.stream_rules import StreamRuleError, parse_stream_rule
from scripts.llm_solver.server.client import LlamaClient
from scripts.llm_solver.server.replay_client import ReplayClient
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage
from tests._config_helpers import make_config


def _rule(
    *,
    condition: str = "forbidden",
    scope: str = "text",
    interrupt_mode: str = "always",
    body: str = "Choose a supported operation.",
):
    return parse_stream_rule(
        "+++\n"
        'name = "correct-response"\n'
        f'condition = "{condition}"\n'
        f'scope = "{scope}"\n'
        f'interruptMode = "{interrupt_mode}"\n'
        "+++\n"
        f"{body}\n",
        source_path=".harness/stream_rules/correct-response.md",
    )


def _chunk(*, content=None, tool_calls=None, finish_reason=None, usage=None):
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        role=None,
        function_call=None,
        refusal=None,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason, index=0)
    return SimpleNamespace(choices=[choice], usage=usage)


def _usage(prompt_tokens, completion_tokens):
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


class _ClosableStream:
    def __init__(self, chunks):
        self._chunks = iter(chunks)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)

    def close(self):
        self.closed = True


def _trace_rows(trace: io.StringIO):
    return [json.loads(line) for line in trace.getvalue().splitlines() if line]


def _compact_context() -> CompactTranscript:
    return CompactTranscript(
        "task",
        recent_results_chars=20_000,
        trace_reasoning_chars=200,
        min_turns=0,
        args_summary_chars=80,
    )


@pytest.mark.parametrize("context_factory", [FullTranscript, _compact_context])
def test_autonomous_narration_stops_early_recovers_and_replays(tmp_path, monkeypatch, context_factory):
    monkeypatch.setenv("YUJ_STREAMING", "0")
    cfg = make_config(context_size=43008, stream_rules_enabled=False)
    limit = int(cfg.context_size * cfg.narration_context_fraction * 4)
    first = _ClosableStream([_chunk(content="x" * (limit + 1)), _chunk(content="unread" * 8000)])
    call = SimpleNamespace(index=0, id="call_1_0", type="function", function=SimpleNamespace(
        name="write", arguments=json.dumps({"path": "out.py", "content": "x" * 40000})
    ))
    second = _ClosableStream([
        _chunk(content="Write the fix.", tool_calls=[call]),
        _chunk(finish_reason="tool_calls", usage=_usage(300, 10000)),
    ])
    client = LlamaClient(cfg, profile=None)
    client.client.chat.completions.create = MagicMock(side_effect=[first, second])
    transcript = tmp_path / "narration.log"
    client.set_transcript(transcript)
    trace = io.StringIO()
    session = Session(cfg, client, "system", "task", str(tmp_path),
                      context_manager=context_factory(), trace_file=trace)
    result = session._chat_with_retry(1)
    client.close_transcript()
    assert first.closed
    assert next(first._chunks).choices[0].delta.content.startswith("unread")
    assert result.tool_calls[0].name == "write"
    assert len(result.tool_calls[0].arguments["content"]) == 40000
    assert client.client.chat.completions.create.call_count == 2
    retry = client.client.chat.completions.create.call_args_list[1].kwargs
    assert retry["stream"] is True
    assert retry["model"] == client.client.chat.completions.create.call_args_list[0].kwargs["model"]
    assert "Take the next concrete coding action" in json.dumps(retry["messages"])
    assert "x" * (limit + 1) not in json.dumps(retry["messages"])
    assert result.usage.completion_tokens > 10000
    assert result.usage.completion_tokens_known is False
    assert result.last_prompt_tokens == 300
    events = _trace_rows(trace)
    assert [e["action"] for e in events if e["event"] == "narration_limit"] == ["retry"]

    replay = ReplayClient(transcript, strict_fidelity=False)
    replay_session = Session(cfg, replay, "system", "task", str(tmp_path),
                             context_manager=context_factory())
    replay_result = replay_session._chat_with_retry(1)
    assert replay_result.tool_calls == result.tool_calls
    assert replay_result.usage == result.usage


def test_second_narration_breach_ends_task_without_session_restart(tmp_path, monkeypatch):
    from scripts.llm_solver.harness.loop import solve_task
    from scripts.llm_solver._shared.telemetry_paths import trace_path

    monkeypatch.setenv("YUJ_STREAMING", "0")
    cfg = make_config(max_sessions=3, max_turns=5, context_size=43008, sandbox_bash=False)
    limit = int(cfg.context_size * cfg.narration_context_fraction * 4)
    streams = [_ClosableStream([_chunk(content="x" * (limit + 1))]) for _ in range(2)]
    client = LlamaClient(cfg, profile=None)
    client.client.chat.completions.create = MagicMock(side_effect=streams)
    (tmp_path / "prompt.txt").write_text("Fix the issue")
    with patch("scripts.llm_solver.harness.loop._auto_commit"), patch.object(Session, "_get_server_ctx", return_value=43008):
        assert solve_task(tmp_path, cfg, client) is False
    assert client.client.chat.completions.create.call_count == 2
    assert all(stream.closed for stream in streams)
    metrics = json.loads((tmp_path / "metrics.json").read_text())["metrics"]
    assert metrics["sessions_used"] == 1
    assert metrics["usage_estimated"] is True
    assert metrics["total_completion_tokens"] == 2 * ((limit + 4) // 4)
    events = [json.loads(line) for line in trace_path(tmp_path).read_text().splitlines()]
    assert [e["action"] for e in events if e["event"] == "narration_limit"] == ["retry", "end"]
    assert [e["finish_reason"] for e in events if e["event"] == "session_end"] == ["narration_limit"]


def test_reply_contract_defaults_and_validation(tmp_path):
    assert load_config().reply_mode == "autonomous"
    assert load_config(overrides={"runtime_mode": "assistant"}).reply_mode == "conversation"
    assert load_config(overrides={"runtime_mode": "assistant", "reply_mode": "auto"}).reply_mode == "conversation"
    assert load_config(overrides={"runtime_mode": "assistant", "reply_mode": "autonomous"}).reply_mode == "autonomous"
    for setting in ['reply_mode="unknown"', 'narration_context_fraction=0', 'narration_context_fraction=true']:
        config = tmp_path / "reply.toml"
        config.write_text("[loop]\n" + setting + "\n")
        with pytest.raises(ValueError):
            load_config(user_config=config)


@pytest.mark.parametrize("context_size", [20000, 43008, 262144])
def test_narration_allowance_scales_and_excludes_other_surfaces(context_size):
    from scripts.llm_solver.harness.stream_rules import NarrationBudget
    from scripts.llm_solver.server._streaming import StreamDelta, StreamRuleInterrupt

    budget = NarrationBudget(context_size=context_size, fraction=0.01, message="Act.")
    assert budget.limit_chars == int(context_size * 0.01 * 4)
    budget.observe(StreamDelta("thinking", "x" * 40000))
    budget.observe(StreamDelta("tool", "x" * 40000))
    budget.observe(StreamDelta("text", "x" * budget.limit_chars))
    with pytest.raises(StreamRuleInterrupt):
        budget.observe(StreamDelta("text", "x"))


def test_conversation_keeps_long_prose_and_does_not_force_streaming(tmp_path, monkeypatch):
    from scripts.llm_solver.server._streaming import assemble_stream

    monkeypatch.setenv("YUJ_STREAMING", "0")
    cfg = make_config(reply_mode="conversation")
    response = assemble_stream([
        _chunk(content="x" * 40000),
        _chunk(finish_reason="stop", usage=_usage(100, 10000)),
    ])
    client = LlamaClient(cfg, profile=None)
    client.client.chat.completions.create = MagicMock(return_value=response)
    session = Session(cfg, client, "system", "task", str(tmp_path))
    result = session._chat_with_retry(1)
    assert result.content == "x" * 40000
    assert client.client.chat.completions.create.call_count == 1
    assert not client.client.chat.completions.create.call_args.kwargs.get("stream")


def test_autonomous_rejects_transport_without_observer_before_request(tmp_path):
    class NonStreamingAdapter(LlamaClient):
        def _call_api(self, payload, **kwargs):
            pytest.fail("unsupported adapter must not start generation")

    client = NonStreamingAdapter(make_config(), profile=None)
    session = Session(client.cfg, client, "system", "task", str(tmp_path))
    assert session._chat_with_retry(1) is None


def test_interrupted_continuation_counts_completed_and_interrupted_calls(tmp_path):
    from scripts.llm_solver.harness._loop.chat_io import _record_narration_usage
    from scripts.llm_solver.server._streaming import StreamRuleInterrupt, assemble_stream

    cfg = make_config()
    session = Session(cfg, _StaticClient(None), "system", "task", str(tmp_path))
    session._abandoned_chat_usage = None
    partial = assemble_stream([
        _chunk(content="too much text"),
        _chunk(finish_reason="stop", usage=_usage(150, 8)),
    ])
    exc = StreamRuleInterrupt([{"observed_chars": 13}], partial)
    exc.prior_usages = (Usage(100, 10),)
    _record_narration_usage(session, [], exc, 1, 1)
    assert session._abandoned_chat_usage.prompt_tokens == 250
    assert session._abandoned_chat_usage.completion_tokens == 18
    assert session._narration_usage_estimated is False


def test_fake_stream_is_closed_discard_omits_partial_and_replay_retries(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("YUJ_STREAMING", "1")
    cfg = make_config(
        max_transient_retries=0,
        stream_rules_enabled=True,
        stream_rules_context_mode="discard",
    )
    rule = _rule()
    first_stream = _ClosableStream([
        _chunk(content="partial prefix "),
        _chunk(content="forbidden"),
    ])
    second_stream = _ClosableStream([
        _chunk(content="corrected response"),
        _chunk(finish_reason="stop", usage=_usage(20, 2)),
    ])
    live_client = LlamaClient(cfg, profile=None)
    live_client.client.chat.completions.create = MagicMock(
        side_effect=[first_stream, second_stream]
    )
    transcript = tmp_path / "live.log"
    live_client.set_transcript(transcript)
    live_trace = io.StringIO()
    live_context = _compact_context()
    live_session = Session(
        cfg,
        live_client,
        "system",
        "task",
        str(tmp_path),
        context_manager=live_context,
        trace_file=live_trace,
        session_number=3,
        stream_rules=[rule],
    )
    live_session._get_server_ctx = lambda: cfg.context_size

    live_result = live_session._chat_with_retry(0)
    live_client.close_transcript()

    assert live_result is not None and live_result.content == "corrected response"
    assert first_stream.closed is True
    messages = live_context.get_messages()
    assert not any(message["role"] == "assistant" for message in messages)
    assert not any(
        "partial prefix forbidden" in str(message.get("content") or "")
        for message in messages
    )
    assert "<injected-fragment" in messages[-1]["content"]
    assert (
        "<injected-fragment"
        in live_client.client.chat.completions.create.call_args_list[1]
        .kwargs["messages"][-1]["content"]
    )

    rows = _trace_rows(live_trace)
    trigger = next(row for row in rows if row["event"] == "stream_rule_triggered")
    injection = next(row for row in rows if row["event"] == "stream_rule_injection")
    assert (trigger["rule"], trigger["scope"], trigger["offset"]) == (
        "correct-response",
        "text",
        15,
    )
    assert "body" not in trigger
    assert injection["rules"] == ["correct-response"]
    assert injection["delivery"] == "retry"
    projected = project(rows, max_result_chars=20_000)
    assert [gate["event"] for gate in projected["gates"]] == [
        "stream_rule_triggered",
        "stream_rule_injection",
    ]
    assert all("body" not in gate for gate in projected["gates"])
    assert projected["meta"]["last_turn"] == 0

    transcript_text = transcript.read_text()
    assert '"_stream_rule_interrupt"' in transcript_text
    assert "partial prefix forbidden" in transcript_text

    replay_client = ReplayClient(transcript, strict_fidelity=False)
    replay_trace = io.StringIO()
    replay_context = FullTranscript()
    replay_session = Session(
        cfg,
        replay_client,
        "system",
        "task",
        str(tmp_path),
        context_manager=replay_context,
        trace_file=replay_trace,
        session_number=4,
        stream_rules=[rule],
    )
    replay_session._get_server_ctx = lambda: cfg.context_size

    replay_result = replay_session._chat_with_retry(0)

    assert replay_result is not None
    assert replay_result.content == "corrected response"
    assert replay_client.served_turns == 1
    assert "<injected-fragment" in replay_context.get_messages()[-1]["content"]
    replay_rows = _trace_rows(replay_trace)
    assert [row["event"] for row in replay_rows] == [
        "stream_rule_triggered",
        "stream_rule_injection",
    ]
    assert {row["turn_number"] for row in replay_rows} == {0}


class _StaticClient:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def chat(self, messages, tools, turn=0):
        self.calls += 1
        return self.result

    @staticmethod
    def build_assistant_message(content, tool_calls):
        return {"role": "assistant", "content": content}


class _SequenceClient:
    def __init__(self, results):
        self.results = iter(results)
        self.requests = []

    def chat(self, messages, tools, turn=0):
        self.requests.append([dict(message) for message in messages])
        return next(self.results)

    @staticmethod
    def build_assistant_message(content, tool_calls):
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in tool_calls
            ]
        return message


def test_nonstream_response_is_posthoc_and_noninterrupting(tmp_path):
    cfg = make_config(stream_rules_enabled=True)
    client = _StaticClient(TurnResult("forbidden", [], "stop", Usage(2, 1)))
    context = _compact_context()
    trace = io.StringIO()
    session = Session(
        cfg,
        client,
        "system",
        "task",
        str(tmp_path),
        context_manager=context,
        trace_file=trace,
        stream_rules=[_rule()],
    )
    session._get_server_ctx = lambda: cfg.context_size

    result = session._chat_with_retry(0)

    assert result is not None and result.content == "forbidden"
    assert client.calls == 1
    assert [row["event"] for row in _trace_rows(trace)] == [
        "stream_rule_triggered"
    ]
    session._apply_pending_stream_rule_injections(1)
    assert "<injected-fragment" in context.get_messages()[-1]["content"]
    assert _trace_rows(trace)[-1]["delivery"] == "next_turn"


def test_noninterrupt_tool_match_prepends_reminder_in_real_dispatch(tmp_path):
    cfg = make_config(stream_rules_enabled=True, max_turns=2)
    client = _SequenceClient([
        TurnResult(
            None,
            [ToolCall("call_0_0", "bash", {"cmd": "printf tool-output"})],
            "tool_calls",
            Usage(2, 1),
        ),
        TurnResult("finished", [], "stop", Usage(3, 1)),
    ])
    trace = io.StringIO()
    session = Session(
        cfg,
        client,
        "system",
        "task",
        str(tmp_path),
        context_manager=FullTranscript(),
        trace_file=trace,
        stream_rules=[
            _rule(
                condition="printf tool-output",
                scope="tool:bash",
                interrupt_mode="never",
            )
        ],
    )
    session._get_server_ctx = lambda: cfg.context_size

    result = session.run()

    assert result.finish_reason == "stop"
    tool_message = next(
        message for message in client.requests[1] if message["role"] == "tool"
    )
    assert tool_message["content"].startswith(
        '<system-reminder reason="rule_violation"'
    )
    assert "tool-output" in tool_message["content"]
    rows = _trace_rows(trace)
    assert any(row["event"] == "stream_rule_triggered" for row in rows)
    injection = next(
        row for row in rows if row["event"] == "stream_rule_injection"
    )
    assert injection["delivery"] == "tool_result"


def test_stream_reminder_keeps_tool_result_while_path_injection_uses_user_turn(
    tmp_path,
):
    target = tmp_path / "src" / "main.py"
    target.parent.mkdir()
    target.write_text("VALUE = 1\n")
    cfg = make_config(
        stream_rules_enabled=True,
        injections_enabled=True,
        injections_path_rules_enabled=True,
        max_turns=2,
    )
    client = _SequenceClient([
        TurnResult(
            None,
            [ToolCall("read-1", "read", {"path": "src/main.py"})],
            "tool_calls",
            Usage(2, 1),
        ),
        TurnResult("finished", [], "stop", Usage(3, 1)),
    ])
    trace = io.StringIO()
    session = Session(
        cfg,
        client,
        "system",
        "task",
        str(tmp_path),
        context_manager=FullTranscript(),
        trace_file=trace,
        stream_rules=[
            _rule(
                condition="src/main[.]py",
                scope="tool:read",
                interrupt_mode="never",
            )
        ],
        injections=[Injection(
            name="python-path",
            trigger="path",
            keywords=(),
            fire_once=True,
            body="Use the repository's Python conventions.",
            source_path=".harness/injections/python-path.md",
            paths=("src/*.py",),
        )],
    )
    session._get_server_ctx = lambda: cfg.context_size

    result = session.run()

    assert result.finish_reason == "stop"
    tool_result = next(
        message["content"]
        for message in client.requests[1]
        if message["role"] == "tool"
    )
    assert tool_result.startswith('<system-reminder reason="rule_violation"')
    assert '<injected-fragment rule="python-path"' not in tool_result
    next_request = "\n".join(
        str(message.get("content") or "")
        for message in client.requests[1]
    )
    assert '<injected-fragment rule="python-path" trigger="path"' in next_request
    rows = _trace_rows(trace)
    assert any(
        row.get("event") == "stream_rule_injection"
        and row.get("delivery") == "tool_result"
        for row in rows
    )
    assert any(
        row.get("event") == "injection"
        and row.get("trigger") == "path"
        for row in rows
    )
    assert any(
        row.get("event") == "user_turn_injection"
        and row.get("mechanism") == "python-path"
        and row.get("tool_call_id") == "read-1"
        for row in rows
    )


def test_noninterrupt_rule_decorates_inactive_tool_rejection(tmp_path):
    cfg = make_config(
        stream_rules_enabled=True,
        tools_lazy_loading_enabled=True,
        tools_active_default=("bash", "read", "glob", "grep", "done"),
        tools_edit_format="whole",
        error_nudge_threshold=99,
        max_turns=2,
    )
    client = _SequenceClient([
        TurnResult(
            None,
            [ToolCall(
                "write-1",
                "write",
                {"path": "created.txt", "content": "not executed\n"},
            )],
            "tool_calls",
            Usage(2, 1),
        ),
        TurnResult("finished", [], "stop", Usage(3, 1)),
    ])
    session = Session(
        cfg,
        client,
        "system",
        "task",
        str(tmp_path),
        context_manager=FullTranscript(),
        trace_file=io.StringIO(),
        stream_rules=[
            _rule(
                condition="not executed",
                scope="tool:write",
                interrupt_mode="never",
            )
        ],
    )
    session._get_server_ctx = lambda: cfg.context_size

    result = session.run()

    assert result.finish_reason == "stop"
    assert not (tmp_path / "created.txt").exists()
    tool_result = next(
        message["content"]
        for message in client.requests[1]
        if message["role"] == "tool"
    )
    assert tool_result.startswith('<system-reminder reason="rule_violation"')
    assert '"type":"tool_not_active"' in tool_result


def test_config_knobs_load_and_invalid_values_fail_clearly(tmp_path):
    defaults = load_config()
    assert defaults.stream_rules_enabled is False
    assert defaults.stream_rules_dir == ".harness/stream_rules"
    assert defaults.stream_rules_context_mode == "discard"
    assert defaults.stream_rules_repeat_gap == 10

    overlay = tmp_path / "stream-rules.toml"
    overlay.write_text(
        "[loop]\n"
        "stream_rules_enabled = true\n"
        'stream_rules_dir = ".harness/custom-rules"\n'
        'stream_rules_context_mode = "keep"\n'
        "stream_rules_repeat_gap = 7\n"
    )
    cfg = load_config(user_config=overlay)
    assert cfg.stream_rules_enabled is True
    assert cfg.stream_rules_dir == ".harness/custom-rules"
    assert cfg.stream_rules_context_mode == "keep"
    assert cfg.stream_rules_repeat_gap == 7

    overlay.write_text(
        '[loop]\nstream_rules_context_mode = "erase"\n'
    )
    with pytest.raises(ValueError, match="stream_rules_context_mode"):
        load_config(user_config=overlay)


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ('stream_rules_enabled = "yes"', "stream_rules_enabled"),
        ('stream_rules_dir = "../outside"', "stream_rules_dir"),
        ("stream_rules_repeat_gap = 0", "stream_rules_repeat_gap"),
    ],
)
def test_invalid_stream_rule_knob_types_and_bounds_fail(tmp_path, setting, message):
    overlay = tmp_path / "invalid-stream-rule-knob.toml"
    overlay.write_text(f"[loop]\n{setting}\n")

    with pytest.raises(ValueError, match=message):
        load_config(user_config=overlay)


def test_invalid_rule_fails_during_startup_load_before_model_call(tmp_path):
    rule_dir = tmp_path / ".harness" / "stream_rules"
    rule_dir.mkdir(parents=True)
    (rule_dir / "bad.md").write_text(
        '+++\nname = "bad"\ncondition = "["\n+++\nCorrect it.\n'
    )
    cfg = make_config(stream_rules_enabled=True)

    with pytest.raises(StreamRuleError, match=r"bad\.md: invalid condition regex"):
        load_session_stream_rules(cfg, tmp_path)


def test_startup_rejects_rule_symlink_outside_task_repository(tmp_path):
    repository = tmp_path / "repository"
    rule_dir = repository / ".harness" / "stream_rules"
    rule_dir.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text(
        '+++\nname = "outside"\ncondition = "x"\n+++\nDo not load me.\n'
    )
    (rule_dir / "outside.md").symlink_to(outside)
    cfg = make_config(stream_rules_enabled=True)

    with pytest.raises(StreamRuleError, match="escapes the task repository"):
        load_session_stream_rules(cfg, repository)
