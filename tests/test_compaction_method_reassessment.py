"""Fallback decisions use the current candidate, never checkpoint frequency."""
from unittest.mock import MagicMock

import pytest

from scripts.llm_solver.harness._loop.compaction import maybe_compact_messages
from scripts.llm_solver.harness import time_budget
from scripts.llm_solver.server.types import SideRequestResult, Usage
from tests.test_compaction_integration import _session, _summary, _messages


@pytest.mark.parametrize("bad_summary", ["missing sections", _summary() + "X" * 13000])
def test_current_invalid_candidate_falls_back_then_reassesses_next_request(tmp_path, bad_summary):
    session, calls, events = _session(tmp_path, bad_summary)
    before = session.context.get_messages()
    first = maybe_compact_messages(session, before)
    assert len(calls) == 1
    assert events[-1]["fallback"] == "digest"
    reason = events[-1]["checkpoint_validation_reason"]
    assert reason and reason != "ok"
    if bad_summary.startswith("##"):
        assert "does not fit budget" in reason
    assert first[:2] == before[:2]
    assert first[-2:] == before[-2:]
    assert not getattr(session, "_compaction_method_override", "")
    session.client.complete_side_request = MagicMock(
        return_value=SideRequestResult(_summary(), Usage(100, 30)))
    grown = first + _messages(pairs=18)[2:]
    session.context.replace_all_messages(grown)
    session._compaction_turn += 1
    maybe_compact_messages(session, grown)
    assert session.client.complete_side_request.call_count == 1
    assert events[-1]["method"] == "checkpoint"
    assert events[-1]["fallback"] == ""
    assert events[-1]["checkpoint_validation_reason"] == "ok"


def test_legacy_private_digest_override_cannot_replace_current_method(tmp_path):
    session, calls, events = _session(tmp_path, _summary())
    session._compaction_method_override = "digest"
    maybe_compact_messages(session, session.context.get_messages())
    assert len(calls) == 1
    assert events[-1]["method"] == "checkpoint"


@pytest.mark.parametrize("expires", ["before_check", "before_summary", "during_summary"])
def test_exhausted_time_does_not_start_or_apply_checkpoint_replacement(tmp_path, monkeypatch, expires):
    session, calls, events = _session(tmp_path, _summary())
    session._role_token_ledger = MagicMock()
    before = session.context.get_messages()
    remaining = [0 if expires == "before_check" else 1]
    monkeypatch.setattr(time_budget, "remaining_run_seconds", lambda: remaining[0])
    if expires == "before_summary":
        count = session._tokenizer.count
        def count_and_expire(messages, tools=None):
            remaining[0] = 0
            return count(messages, tools)
        session._tokenizer.count = count_and_expire
    if expires == "during_summary":
        complete = session.client.complete_side_request
        def complete_and_expire(payload):
            remaining[0] = 0
            return complete(payload)
        session.client.complete_side_request = complete_and_expire
    result = maybe_compact_messages(session, before)
    assert result is before
    assert session.context.get_messages() == before
    assert len(calls) == (1 if expires == "during_summary" else 0)
    assert session._role_token_ledger.record_usage.call_count == len(calls)
    if calls:
        usage = session._role_token_ledger.record_usage.call_args.args[1]
        assert (usage.prompt_tokens, usage.completion_tokens) == (100, 30)
    assert events == []
    assert not getattr(session, "_checkpoint_previous_summary", "")
