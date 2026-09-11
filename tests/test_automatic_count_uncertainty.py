"""Automatic uncertainty cannot justify discarding context or declaring it full."""
import io
import json
import copy
from unittest.mock import MagicMock, Mock

import pytest

from tests._config_helpers import make_config
from tests.test_compaction_integration import _session, _summary
from tests.test_salience_section_limits import _context
from scripts.llm_solver.harness._loop.compaction import maybe_compact_messages
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.server.types import TurnResult, Usage


class UncertainCounter:
    def __init__(self, precision):
        self.precision = precision
        self.last = {}

    def count(self, messages, tools=None):
        if self.precision == "error":
            raise OSError("counter unavailable")
        value = len(json.dumps([messages, tools])) // 4
        self.last = {"count_basis": "backend_input_tokens", "count_precision": self.precision,
                     "prompt_tokens": value}
        return value


@pytest.mark.parametrize("precision", ["estimate", "unverified", "error"])
def test_automatic_preflight_and_chat_preserve_source_with_uncertain_counts(tmp_path, precision):
    cfg = make_config(tokenizer_id="auto", context_size=8192, max_turns=1)
    client = MagicMock()
    client.query_server_context.return_value = 8192
    client.chat.return_value = TurnResult(content="observed", tool_calls=[],
                                         finish_reason="stop", usage=Usage(10, 5))
    trace = io.StringIO()
    session = Session(cfg, client, "system", "task", str(tmp_path), trace_file=trace,
                      local_tokenizer=UncertainCounter(precision))
    source = "source evidence " * 4000
    session.context.add_tool_result("c1", source, tool_name="read")
    session._last_actual_prompt_tokens = 90000
    session._preflight_density = 2.0
    session._preflight_prev_estimate = 10
    result = session.run()
    assert result.finish_reason == "stop"
    sent = client.chat.call_args.args[0]
    assert any(m.get("content") == source for m in sent)
    assert session._preflight_gate_chars_new == 0
    events = list(map(json.loads, trace.getvalue().splitlines()))
    assert any(e["event"] == "context_count_unverified" for e in events)
    assert not any(e["event"] == "compaction" for e in events)


@pytest.mark.parametrize("force,expected", [(False, False), (True, True)])
def test_compaction_requires_reported_count_or_explicit_overflow_evidence(tmp_path, force, expected):
    session, calls, _ = _session(tmp_path, _summary())
    session.cfg.tokenizer_id = "auto"
    session._tokenizer = UncertainCounter("unverified")
    messages = session.context.get_messages()
    result = maybe_compact_messages(session, messages, force=force)
    assert bool(calls) is expected
    if not expected:
        assert result == messages


@pytest.mark.parametrize("precision", ["estimate", "unverified", "error", "backend_reported"])
def test_salience_pressure_requires_current_evidence_in_auto_mode(tmp_path, monkeypatch, precision):
    ctx = _context(tmp_path, trace=50, unresolved=30, tool_chars=30000)
    cfg = make_config(tokenizer_id="auto", context_size=1024, max_tokens=100)
    counter = UncertainCounter(precision)
    ctx.set_projection_request(lambda: (counter, cfg, []))
    build = Mock(wraps=ctx._build_parts)
    monkeypatch.setattr(ctx, "_build_parts", build)
    state = "retained evidence " * 1000
    result = ctx._bounded_projection(state_text=state, suffix_text="", trace=[], evidence=[])
    assert state in result[-1]["content"]
    assert ctx.projection_pressure["actionable"] is (precision == "backend_reported")
    assert (build.call_count > 1) is (precision == "backend_reported")


@pytest.mark.parametrize("candidate_precision", ["backend_reported", "estimate", "error", "after_clip"])
def test_inner_compaction_preserves_tail_when_recount_loses_authority(tmp_path, candidate_precision):
    from tests.test_session_compaction import (
        _make_session, _make_fake_context, _seed_trace_jsonl, _heavy_messages,
    )

    _seed_trace_jsonl(tmp_path, 8)
    session = _make_session(tmp_path, [], server_ctx_value=4096,
                            cfg_extra={"tokenizer_id": "auto"})
    session.context = _make_fake_context()
    messages = _heavy_messages(3, payload_chars=2000)
    messages[-1]["content"] = "x" * 25000 + "MIDDLE_DIAGNOSTIC" + "y" * 25000
    original = copy.deepcopy(messages)

    class Counter:
        last = {}
        candidate_calls = 0

        def count(self, current, tools=None):
            initial = current == messages
            # Digest-only counts are separate; vary the candidate request.
            candidate = not initial and len(current) > 1
            self.candidate_calls += int(candidate)
            if candidate and candidate_precision == "error":
                raise OSError("candidate counter unavailable")
            reported = not candidate or candidate_precision == "backend_reported"
            if candidate and candidate_precision == "after_clip":
                reported = self.candidate_calls == 1
            value = 6000 if initial else 3000 if candidate and reported else len(json.dumps(current)) // 4
            if candidate and candidate_precision == "after_clip":
                value = 12000
            self.last = {"count_basis": "backend_input_tokens" if reported else "character_estimate",
                         "count_precision": "backend_reported" if reported else "estimate",
                         "prompt_tokens": value}
            return value

    session._tokenizer = Counter()
    result = session._maybe_compact_messages(messages)
    assert session._compacted
    if candidate_precision == "after_clip":
        assert result[-1]["content"] != original[-1]["content"]
        assert session._tokenizer.candidate_calls == 2
    else:
        assert result[-1]["content"] == original[-1]["content"]
    assert messages == original
    assert session.context._all_messages == result
    event = next(row for row in session.emitted if row["event"] == "compaction")
    assert event["tokens_after_count_precision"] == (
        "backend_reported" if candidate_precision == "backend_reported" else "estimate")
