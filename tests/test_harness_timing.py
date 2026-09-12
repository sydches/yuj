"""Raw dispatch timing stays separate from queueing and overlapping workers."""
from datetime import datetime
import threading
import time
from unittest.mock import MagicMock

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness import loop
from scripts.llm_solver.harness._loop import timing
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


def _client(calls):
    client = MagicMock()
    client.chat.side_effect = [
        TurnResult(content="inspect", tool_calls=calls, finish_reason="tool_calls", usage=Usage(10, 2)),
        TurnResult(content="done", tool_calls=[], finish_reason="stop", usage=Usage(10, 2)),
    ]
    client.build_assistant_message.return_value = {"role": "assistant", "content": ""}
    return client


def test_parallel_dispatch_wall_counts_overlap_once(tmp_path, monkeypatch):
    both_running = threading.Barrier(2)
    def dispatch(*args, **kwargs):
        both_running.wait(timeout=5)
        time.sleep(0.04)
        return "task bytes"
    monkeypatch.setattr(loop, "dispatch", dispatch)
    session = loop.Session(make_config(max_turns=2, parallel_readonly_enabled=True),
        _client([ToolCall(str(i), "read", {"path": f"{i}.py"}) for i in range(2)]),
        "system", "task", str(tmp_path))
    assert session.run().done
    assert session._tool_dispatch_timings == {}
    ends = [row for row in session._trace_events if row["event"] == "tool_end"]
    turn = next(row for row in session._trace_events if row["event"] == "turn_timing")
    summed = sum(row["duration_ms"] for row in ends)
    assert summed - turn["tool_ms"] >= 20
    assert max(row["duration_ms"] for row in ends) <= turn["tool_ms"] + 0.02
    assert turn["model_to_boundary_ms"] == pytest.approx(
        turn["chat_call_ms"] + turn["tool_ms"] + turn["harness_ms"], abs=0.02)
    assert all(datetime.fromisoformat(row["ended_at"]).tzinfo is not None for row in ends)


def test_model_tool_and_post_boundaries_use_monotonic_clock(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(timing.time, "perf_counter", lambda: clock[0])
    client = _client([ToolCall("read", "read", {"path": "a.py"})])
    responses = iter(client.chat.side_effect)
    def chat(*args, **kwargs):
        clock[0] += 19
        return next(responses)
    client.chat.side_effect = chat
    def dispatch(*args, **kwargs):
        clock[0] += 3
        return "task bytes"
    monkeypatch.setattr(loop, "dispatch", dispatch)
    session = loop.Session(make_config(max_turns=2), client, "system", "task", str(tmp_path))
    original_hook = session._run_hook
    def hook(event, **kwargs):
        if event == "pre_tool":
            clock[0] += 2
        return original_hook(event, **kwargs)
    monkeypatch.setattr(session, "_run_hook", hook)
    original_add = session.context.add_tool_result
    def add(*args, **kwargs):
        clock[0] += 5
        return original_add(*args, **kwargs)
    monkeypatch.setattr(session.context, "add_tool_result", add)
    assert session.run().done
    rows = session._trace_events
    first = next(row for row in rows if row["event"] == "turn_timing")
    assert first["chat_call_ms"] == 19000
    assert first["tool_ms"] == 3000
    assert first["harness_ms"] == 7000
    assert first["post_ms"] == 5000
    assert first["model_to_boundary_ms"] == 29000
    end = next(row for row in rows if row["event"] == "tool_end")
    result = next(row for row in rows if row["event"] == "tool_call")
    assert rows.index(end) < rows.index(result)
    assert end["duration_ms"] == result["duration_ms"] == 3000


def test_union_excludes_idle_gaps_and_nested_intervals():
    records = [timing.DispatchTiming("read", start, end, "", True)
               for start, end in [(1, 5), (2, 3), (4, 6), (9, 10)]]
    assert timing.dispatch_union_ms(records) == 6000


def test_missing_call_id_cannot_inherit_previous_dispatch_time():
    session = MagicMock()
    session._tool_dispatch_timings = {"earlier": timing.DispatchTiming("read", 1, 5, "", True)}
    assert timing.dispatch_trace_fields(session, {}) == {"duration_ms": 0, "dispatch_executed": False}


@pytest.mark.parametrize("branch", ["narration_discarded", "intent_block"])
def test_early_continue_records_each_model_turn_without_inventing_dispatch(tmp_path, monkeypatch, branch):
    from scripts.llm_solver.harness.guardrails import build_guardrail_registry, Decision, PASS
    registry = build_guardrail_registry(turn_pre_overrides={
        "intent_gate": lambda state, cfg, **kwargs: (
            Decision.block("fixture", reason="fixture")
            if branch == "intent_block" and kwargs["turn"] == 1 else PASS)
    })
    calls = [ToolCall("blocked", "read", {"path": "a.py"})] if branch == "intent_block" else []
    client = _client(calls)
    if branch == "narration_discarded":
        client.chat.side_effect = [
            TurnResult(content=None, tool_calls=[], finish_reason=branch, usage=Usage(10, 2)),
            TurnResult(content="done", tool_calls=[], finish_reason="stop", usage=Usage(10, 2)),
        ]
    session = loop.Session(make_config(max_turns=2), client, "system", "task", str(tmp_path),
                           guardrail_registry=registry)
    session._turn_start_offset = 1
    monkeypatch.setattr(loop, "dispatch", lambda *a, **k: pytest.fail("blocked call dispatched"))
    assert session.run().done
    rows = [row for row in session._trace_events if row["event"] == "turn_timing"]
    assert [row["turn_number"] for row in rows] == [1, 2]
    assert [row["boundary"] for row in rows] == ["next_model_call_entry", "session_end"]
    assert all(row["tool_ms"] == 0 and row["post_ms"] is None and not row["dispatch_executed"] for row in rows)
    if branch == "intent_block":
        result = next(row for row in session._trace_events if row["event"] == "tool_call")
        assert result["tool_call_id"] == "blocked"
        assert result["duration_ms"] == 0 and not result["dispatch_executed"]


def test_parallel_dispatch_failure_records_every_finished_worker(tmp_path, monkeypatch):
    from scripts.llm_solver.harness._tools._run_in_sandbox import SandboxUnavailableError
    both_running = threading.Barrier(2)
    def dispatch(*args, **kwargs):
        both_running.wait(timeout=5)
        raise SandboxUnavailableError("fixture unavailable")
    monkeypatch.setattr(loop, "dispatch", dispatch)
    session = loop.Session(make_config(max_turns=1, parallel_readonly_enabled=True),
        _client([ToolCall(str(i), "read", {"path": f"{i}.py"}) for i in range(2)]),
        "system", "task", str(tmp_path))
    assert session.run().finish_reason == "sandbox_unavailable"
    ends = [row for row in session._trace_events if row["event"] == "tool_end"]
    assert len(ends) == 2 and all(not row["completed"] for row in ends)
    terminal = next(row for row in session._trace_events if row["event"] == "turn_timing")
    assert terminal["boundary"] == "session_end" and terminal["dispatch_executed"]
    assert all(terminal[field] >= 0 for field in ("duration_ms", "tool_phase_ms", "post_turn_ms", "post_ms", "tool_ms"))
