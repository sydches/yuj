"""Duplicate decisions require completed observations, not request equality."""
import io
import json
from unittest.mock import MagicMock

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness.guardrails import Action, duplicate_guard, init_guardrail_state
from scripts.llm_solver.harness.repeated_observations import read_observation
from scripts.llm_solver.harness.tools import dispatch, build_tool_registry
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.server.types import TurnResult, ToolCall, Usage
from test_process_manager import make_manager


def config(**changes):
    return make_config(duplicate_guard_enabled=True, duplicate_abort=2,
                       duplicate_warn_count=0, loop_detect_enabled=False,
                       max_turns=4, **changes)


def test_missing_pending_changed_and_mixed_evidence_reset_the_streak():
    cfg = config()
    state = init_guardrail_state(cfg)
    same = (read_observation("same"),)
    def check(receipts, sig=("query",)):
        return duplicate_guard(state, cfg, tool_calls_sig=sig, observations=receipts)
    assert check(same).action == Action.PASS
    assert check(None).action == Action.PASS
    assert check(same).action == Action.PASS
    assert check((read_observation("changed"),)).action == Action.PASS
    assert check((dict(same[0], pending=True),)).action == Action.PASS
    assert check((same[0], None), ("query", "unknown")).action == Action.PASS
    assert check(same).action == Action.PASS
    assert check(same).action == Action.END


def test_dispatch_receipt_precedes_output_clipping_and_does_not_trust_overrides(tmp_path):
    cfg = config(max_output_chars=100)
    path = tmp_path / "large.txt"
    path.write_text("prefix" * 100 + "one" + "suffix" * 100)
    facts = {}
    dispatch("read", {"path": "large.txt"}, cwd=str(tmp_path), cfg=cfg, execution_metadata=facts)
    first = facts["observation_receipt"]
    path.write_text("prefix" * 100 + "two" + "suffix" * 100)
    dispatch("read", {"path": "large.txt"}, cwd=str(tmp_path), cfg=cfg, execution_metadata=facts)
    assert first != facts["observation_receipt"]
    facts = {}
    registry = build_tool_registry(overrides={"read": lambda *args: "same"})
    dispatch("read", {"path": "large.txt"}, cwd=str(tmp_path), cfg=cfg,
             tool_registry=registry, execution_metadata=facts)
    assert "observation_receipt" not in facts


@pytest.mark.parametrize("changing,parallel", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("arm_after", [0, 2])
def test_actual_session_observes_results_before_deciding(tmp_path, changing, parallel, arm_after):
    cfg = config(parallel_readonly_enabled=parallel, guardrails_arm_after_turn=arm_after)
    client = MagicMock()
    calls = []
    def turn(*args, **kwargs):
        number = len(calls)
        calls.append(number)
        (tmp_path / "file.txt").write_text(str(number) if changing else "same")
        tools = [ToolCall(id=f"r{number}", name="read", arguments={"path": "file.txt"})]
        if parallel:
            tools.append(ToolCall(id=f"s{number}", name="read", arguments={"path": "other.txt"}))
        return TurnResult(content=None, tool_calls=tools, finish_reason="tool_calls",
                          usage=Usage(prompt_tokens=10, completion_tokens=5))
    (tmp_path / "other.txt").write_text("other")
    client.chat.side_effect = turn
    client.build_assistant_message.return_value = {"role": "assistant", "content": None}
    trace = io.StringIO()
    session = Session(cfg, client, "system", "task", str(tmp_path), trace_file=trace)
    result = session.run()
    assert result.finish_reason == ("max_turns" if changing else "duplicate_abort")
    assert len(calls) == (4 if changing else arm_after + 2)
    assert result.turns == len(calls)
    events = [json.loads(line) for line in trace.getvalue().splitlines()]
    checks = [row for row in events if row["event"] == "duplicate_observation_check"]
    assert len(checks) == len(calls)
    assert checks[0]["interventions_allowed"] is False
    assert checks[0]["count"] == 1
    assert all(row["action"] == "pass" and not row["interventions_allowed"]
               for row in checks[:arm_after + 1])
    if not changing:
        assert checks[-1]["action"] == "end"
        assert checks[-1]["count"] == 2
    observed = [row for row in events if row["event"] == "tool_call"]
    assert len(observed) == len(calls) * (2 if parallel else 1)
    assert all("observation_receipt" in row for row in observed)


def test_running_polls_with_or_without_new_bytes_never_exhaust_duplicate_policy(tmp_path):
    manager, factory, clock, events = make_manager(tmp_path)
    started = manager.start("fixture")
    cfg = config()
    state = init_guardrail_state(cfg)
    try:
        for i in range(90):
            if i % 2:
                factory.processes[0].write(b"new bytes")
            result = manager.poll(started.proc_id, timeout_s=0)
            receipt = result.result.observation_receipt
            assert receipt["pending"] is True
            assert manager.has_pending_observations()
            assert duplicate_guard(state, cfg, tool_calls_sig=("poll",),
                                   observations=(receipt,)).action == Action.PASS
        factory.processes[0].finish()
        assert manager.has_pending_observations()
        manager.poll(started.proc_id, timeout_s=0)
        assert not manager.has_pending_observations()
        from scripts.llm_solver.harness.process_manager import ReplayProcessManager
        replay = ReplayProcessManager(events)
        replay.start("fixture")
        assert replay.has_pending_observations()
        for event in events:
            if event["event"] == "proc_poll":
                poll = replay.poll(started.proc_id)
                assert poll.result.observation_receipt == event["execution_metadata"]["observation_receipt"]
        assert not replay.has_pending_observations()
    finally:
        manager.close()


def test_stable_reads_do_not_abort_while_background_outcome_is_unobserved(tmp_path):
    manager, factory, clock, events = make_manager(tmp_path)
    manager.start("fixture")
    cfg = config()
    client = MagicMock()
    client.chat.return_value = TurnResult(content=None,
        tool_calls=[ToolCall(id="r", name="read", arguments={"path":"file.txt"})],
        finish_reason="tool_calls", usage=Usage(prompt_tokens=10, completion_tokens=5))
    client.build_assistant_message.return_value = {"role":"assistant", "content":None}
    (tmp_path / "file.txt").write_text("unchanged")
    session = Session(cfg, client, "system", "task", str(tmp_path), process_manager=manager)
    try:
        result = session.run()
        assert result.finish_reason == "max_turns"
        assert client.chat.call_count == 4
    finally:
        manager.close()


def test_one_observation_is_not_a_duplicate_even_with_limit_one():
    from dataclasses import replace
    cfg = replace(config(), duplicate_abort=1, duplicate_warn_count=1)
    state = init_guardrail_state(cfg)
    args = dict(tool_calls_sig=("query",), observations=(read_observation("same"),))
    assert duplicate_guard(state, cfg, **args).action == Action.PASS
    assert duplicate_guard(state, cfg, **args).action == Action.END


def test_identical_shell_output_does_not_hide_repeated_effects(tmp_path):
    cfg = config(sandbox_bash=False)
    client = MagicMock()
    client.chat.return_value = TurnResult(content=None,
        tool_calls=[ToolCall(id="effect", name="bash",
                            arguments={"cmd":"printf x >> changes.txt"})],
        finish_reason="tool_calls", usage=Usage(prompt_tokens=10, completion_tokens=5))
    client.build_assistant_message.return_value = {"role":"assistant", "content":None}
    session = Session(cfg, client, "system", "task", str(tmp_path))
    result = session.run()
    assert result.finish_reason == "max_turns"
    assert (tmp_path / "changes.txt").read_text() == "xxxx"


def test_intervening_blocked_turn_breaks_the_observation_streak(tmp_path):
    from scripts.llm_solver.harness.guardrails import build_guardrail_registry, Decision, PASS
    registry = build_guardrail_registry(turn_pre_overrides={
        "intent_gate": lambda state, cfg, **kw: (
            Decision.block("fixture block", reason="fixture")
            if kw["turn"] == 2 else PASS
        )
    })
    client = MagicMock()
    client.chat.return_value = TurnResult(content=None,
        tool_calls=[ToolCall(id="r", name="read", arguments={"path":"file.txt"})],
        finish_reason="tool_calls", usage=Usage(prompt_tokens=10, completion_tokens=5))
    client.build_assistant_message.return_value = {"role":"assistant", "content":None}
    (tmp_path / "file.txt").write_text("same")
    session = Session(config(guardrails_arm_after_turn=1), client, "system", "task", str(tmp_path),
                      guardrail_registry=registry)
    assert session.run().finish_reason == "max_turns"
    assert client.chat.call_count == 4
