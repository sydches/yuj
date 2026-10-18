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
    from types import SimpleNamespace
    from scripts.llm_solver.harness.operation_timing import take_operations
    session._tokenizer = SimpleNamespace(count=lambda messages, tools=None: 100)
    session._turn_start_offset = 1
    monkeypatch.setattr(loop, "dispatch", lambda *a, **k: pytest.fail("blocked call dispatched"))
    assert session.run().done
    rows = [row for row in session._trace_events if row["event"] == "turn_timing"]
    assert [row["turn_number"] for row in rows] == [1, 2]
    assert [row["boundary"] for row in rows] == ["next_model_call_entry", "session_end"]
    assert all(row["tool_ms"] == 0 and row["post_ms"] is None and not row["dispatch_executed"] for row in rows)
    preparation = [item for row in rows for item in row['operations']
                   if item['operation'] in (['request.initial_preflight'], ['request.preflight'])]
    assert [(item['operation'], item['calls']) for item in preparation] == [
        (['request.initial_preflight'], 1), (['request.preflight'], 1)]
    assert take_operations() == [] and session._pending_turn_timing is None
    if branch == "intent_block":
        result = next(row for row in session._trace_events if row["event"] == "tool_call")
        assert result["tool_call_id"] == "blocked"
        assert result["duration_ms"] == 0 and not result["dispatch_executed"]


def test_preflight_and_post_hooks_close_before_each_turn_drain(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.llm_solver.harness.operation_timing import take_operations

    clock, counts = [10.0], []
    client = _client([ToolCall('read', 'read', {'path': 'a.py'})])
    monkeypatch.setattr(loop, 'dispatch', lambda *args, **kwargs: 'unchanged task bytes')
    session = loop.Session(make_config(max_turns=2), client, 'system', 'task', str(tmp_path))
    monkeypatch.setattr(timing.time, 'perf_counter', lambda: clock[0])
    original_messages = session.context.get_messages
    def messages():
        clock[0] += 0.002
        return original_messages()
    def count(messages, tools=None):
        counts.append((messages, tools))
        clock[0] += 0.003
        return 100
    monkeypatch.setattr(session.context, 'get_messages', messages)
    session._tokenizer = SimpleNamespace(count=count)
    monkeypatch.setattr(session, '_maybe_switch_adaptive_phase', lambda turn: clock.__setitem__(0, clock[0] + 0.005))
    assert session.run().done
    rows = [row for row in session._trace_events if row['event'] == 'turn_timing']
    first = {tuple(item['operation']): item for item in rows[0]['operations']}
    for label in ('request.initial_preflight', 'request.preflight'):
        assert first[(label,)]['calls'] == 1
        assert first[(label, 'request.messages')]['elapsed_ms'] == 2
        assert first[(label, 'request.count')]['elapsed_ms'] == 3
        assert first[(label,)]['elapsed_ms'] == 5 and first[(label,)]['self_ms'] == 0
    assert first[('turn.post_hooks',)]['calls'] == 1
    assert first[('turn.post_hooks',)]['elapsed_ms'] == 5
    assert not any(item['operation'][0].startswith('request.') for item in rows[1]['operations'])
    assert client.chat.call_count == 2 and len(counts) >= 2
    assert len([row for row in session._trace_events if row['event'] == 'tool_call']) == 1
    assert take_operations() == [] and session._pending_turn_timing is None


@pytest.mark.parametrize('abort_at', [1, 2])
def test_preflight_exception_keeps_completed_spans_without_inventing_initial_turn(tmp_path, monkeypatch, abort_at):
    from types import SimpleNamespace
    from scripts.llm_solver.harness import operation_timing

    class PreflightAbort(BaseException):
        pass

    client = _client([])
    client.chat.side_effect = [TurnResult(content=None, tool_calls=[],
        finish_reason='narration_discarded', usage=Usage(10, 2))]
    session = loop.Session(make_config(max_turns=2), client, 'system', 'task', str(tmp_path))
    # This fixture targets preflight; client-side preparation has its own count.
    monkeypatch.setattr(session, '_chat_with_retry', lambda turn: client.chat())
    calls = []
    def count(messages, tools=None):
        calls.append(messages)
        if len(calls) == abort_at:
            raise PreflightAbort('fixture abort')
        return 100
    session._tokenizer = SimpleNamespace(count=count)
    with pytest.raises(PreflightAbort, match='fixture abort'):
        session.run()
    rows = [row for row in session._trace_events if row['event'] == 'turn_timing']
    assert len(rows) == abort_at - 1 and client.chat.call_count == abort_at - 1
    if rows:
        assert rows[0]['boundary'] == 'session_end'
        operations = {tuple(item['operation']): item for item in rows[0]['operations']}
        assert operations[('request.initial_preflight',)]['raised'] == 0
        assert operations[('request.preflight',)]['raised'] == 1
        assert operations[('request.preflight', 'request.count')]['raised'] == 1
    assert session._pending_turn_timing is None and operation_timing.take_operations() == []
    assert operation_timing._collector.get() is None and operation_timing._frame.get() is None


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


@pytest.mark.parametrize("parallel", [False, True])
def test_operation_breakdown_reaches_trace_with_call_identity(tmp_path, monkeypatch, parallel):
    from scripts.llm_solver.harness.operation_timing import measured, take_operations
    @measured("fixture.read")
    def dispatch(*args, **kwargs):
        return "task bytes"
    monkeypatch.setattr(loop, "dispatch", dispatch)
    session = loop.Session(make_config(max_turns=2, parallel_readonly_enabled=parallel),
        _client([ToolCall(str(i), "read", {"path": f"{i}.py"}) for i in range(2)]),
        "system", "task", str(tmp_path))
    assert session.run().done
    rows = [row for row in session._trace_events if row["event"] == "turn_timing"]
    assert all(row["operation_timing_version"] == 1 for row in rows)
    reads = [row for row in rows[0]["operations"] if row["operation"][-1] == "fixture.read"]
    assert {row["tool_call_id"] for row in reads} == {"0", "1"}
    assert all(row["calls"] == 1 for row in reads)
    assert not any(row["operation"][-1] == "fixture.read" for row in rows[1]["operations"])
    assert take_operations() == []


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("require_mutation", [False, True])
def test_completion_guard_has_cost_even_without_tool_dispatch(tmp_path, explicit, require_mutation):
    client = _client([])
    client.chat.side_effect = [TurnResult(content="finished",
        tool_calls=[ToolCall("finish", "done", {"message": "done"})] if explicit else [],
        finish_reason="tool_calls" if explicit else "stop", usage=Usage(10, 2))]
    session = loop.Session(make_config(max_turns=1, done_guard_enabled=True,
                           done_require_mutation=require_mutation, done_require_verify=False),
                           client, "system", "task", str(tmp_path))
    assert session.run().done is not require_mutation
    turn = next(row for row in session._trace_events if row["event"] == "turn_timing")
    guards = [row for row in turn["operations"] if row["operation"][-1] == "verification.done_guard"]
    assert len(guards) == 1
    assert guards[0]["tool_call_id"] == ("finish" if explicit else None)
    assert not turn["dispatch_executed"] and turn["tool_ms"] == 0
