"""API failures must include their details in the trace."""

from unittest.mock import MagicMock

import openai
import pytest

from scripts.llm_solver.harness._loop.chat_io import chat_with_retry


def _session_with_chat(side_effect):
    session = MagicMock()
    session.cfg.max_transient_retries = 1
    session.cfg.retry_backoff = [0]
    session._compaction_count = 0
    session.context.get_messages.return_value = []
    session._tool_schemas = []
    session.client.chat.side_effect = side_effect
    return session


def test_context_overflow_emits_typed_trace_event(monkeypatch):
    monkeypatch.setattr(
        "scripts.llm_solver.harness._loop.chat_io.maybe_compact_messages",
        lambda session, msgs, **kwargs: msgs,
    )
    boom = openai.BadRequestError(
        "request (108639 tokens) exceeds the available context size (65536)",
        response=MagicMock(status_code=400), body=None,
    )
    session = _session_with_chat(boom)
    assert chat_with_retry(session, 82) is None
    calls = [c for c in session._emit.call_args_list if c.args[0] == "api_error"]
    assert len(calls) == 1
    kw = calls[0].kwargs
    assert kw["turn_number"] == 82
    assert kw["error_type"] == "BadRequestError"
    assert kw["error_kind"] == "context_overflow"
    assert "108639 tokens" in kw["detail"]
    assert session._last_chat_error_reason == "context_full"


def test_other_bad_request_remains_fatal(monkeypatch):
    monkeypatch.setattr(
        "scripts.llm_solver.harness._loop.chat_io.maybe_compact_messages",
        lambda session, msgs, **kwargs: msgs,
    )
    boom = openai.BadRequestError(
        "invalid tool schema",
        response=MagicMock(status_code=400),
        body=None,
    )
    session = _session_with_chat(boom)
    assert chat_with_retry(session, 12) is None
    calls = [c for c in session._emit.call_args_list if c.args[0] == "api_error"]
    assert len(calls) == 1
    assert calls[0].kwargs["error_kind"] == "fatal"
    assert "_last_chat_error_reason" not in vars(session)


def test_transient_exhausted_emits_trace_event(monkeypatch):
    monkeypatch.setattr(
        "scripts.llm_solver.harness._loop.chat_io.maybe_compact_messages",
        lambda session, msgs, **kwargs: msgs,
    )
    err = openai.APIConnectionError(request=MagicMock())
    session = _session_with_chat(err)
    assert chat_with_retry(session, 5) is None
    calls = [c for c in session._emit.call_args_list if c.args[0] == "api_error"]
    assert len(calls) == 1
    assert calls[0].kwargs["error_kind"] == "transient_exhausted"


def test_success_emits_nothing(monkeypatch):
    monkeypatch.setattr(
        "scripts.llm_solver.harness._loop.chat_io.maybe_compact_messages",
        lambda session, msgs, **kwargs: msgs,
    )
    ok = MagicMock(finish_reason="stop")
    session = _session_with_chat([ok])
    assert chat_with_retry(session, 1) is ok
    assert not [c for c in session._emit.call_args_list if c.args[0] == "api_error"]


@pytest.mark.parametrize("second_rejected", [False, True])
def test_wire_overflow_retries_same_client_once(monkeypatch, second_rejected):
    boom = openai.BadRequestError(
        "request exceeds the available context size",
        response=MagicMock(status_code=400),
        body={"type": "exceed_context"},
    )
    ok = MagicMock(finish_reason="stop")
    session = _session_with_chat([boom, boom if second_rejected else ok])
    original_client = session.client
    compacted = [{"role": "user", "content": "bounded digest"}]
    calls = []

    def compact(active_session, messages, *, force=False):
        calls.append(force)
        if force:
            active_session._compaction_count += 1
            active_session.context.get_messages.return_value = compacted
            return compacted
        return messages

    monkeypatch.setattr(
        "scripts.llm_solver.harness._loop.chat_io.maybe_compact_messages",
        compact,
    )
    monkeypatch.setattr(
        "scripts.llm_solver.harness._loop.chat_io.activate_next_fallback",
        lambda *args, **kwargs: pytest.fail("wire overflow must not switch models"),
    )

    result = chat_with_retry(session, 36)

    assert session.client is original_client
    assert session.client.chat.call_count == 2
    assert calls == [False, True, False]
    if second_rejected:
        assert result is None
        assert session._last_chat_error_reason == "context_full"
    else:
        assert result is ok
