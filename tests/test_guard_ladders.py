"""Ladder actions preserve execution and have bounded advisory episodes."""
import io
import json
import shlex
import sys
from unittest.mock import MagicMock

import pytest

from tests._config_helpers import make_config
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness.guardrails import Action, init_guardrail_state
from scripts.llm_solver.harness._guardrails.checks_pre import intent_gate, rumination_gate
from scripts.llm_solver.harness._guardrails.checks_post import record_mutation
from scripts.llm_solver.harness.loop import Session
from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage


@pytest.mark.parametrize("policy,blocks", [({}, 3), ({"rungs": {"2": 1}}, 0),
    ({"rungs": {"2": 1, "4": 3}, "release_after": 2}, 2)])
def test_silent_episode_releases_and_resets(policy, blocks):
    cfg = make_config(require_intent=True, intent_grace_turns=0,
                      intent_abort_threshold=1, guard_ladders={"silent_call": policy})
    state = init_guardrail_state(cfg)
    def step(content=""):
        return intent_gate(state, cfg, turn=5, content=content, tool_calls=[object()]).action
    outcomes = [step() for _ in range(250)]
    assert outcomes.count(Action.BLOCK) == blocks
    assert outcomes.count(Action.WARN) == (2 if blocks else 1)
    assert Action.END not in outcomes
    assert outcomes[-1] == Action.PASS
    assert step("I will inspect the declaration") == Action.PASS
    assert step() == Action.WARN


def test_inspection_release_lasts_until_real_mutation():
    cfg = make_config(rumination_enabled=True, rumination_gate_max_blocks=1)
    state = init_guardrail_state(cfg)
    state.rumination_gate = True
    results = [rumination_gate(state, cfg, tc_name="read").action for _ in range(250)]
    assert results.count(Action.BLOCK) == 3
    assert results.count(Action.WARN) == 1
    assert results[-1] == Action.PASS
    record_mutation(state)
    assert not state.rumination_released


@pytest.mark.parametrize("bad", ['rungs = { 5 = 2 }', 'rungs = { 2 = 0 }',
                                  'release_after = 0', 'end = 3'])
def test_invalid_ladder_fails_config_load(tmp_path, bad):
    path = tmp_path / "bad.toml"
    path.write_text('[loop.guard_ladders.silent_call]\n' + bad + '\n')
    with pytest.raises(ValueError):
        load_config(user_config=[path])


def test_long_silent_session_executes_after_bounded_blocks(tmp_path):
    (tmp_path / "source.txt").write_text("contents")
    cfg = make_config(max_turns=30, require_intent=True, intent_grace_turns=0,
                      rumination_enabled=False, loop_detect_enabled=False,
                      duplicate_guard_enabled=False, auto_commit=False)
    client = MagicMock()
    client.chat.return_value = TurnResult(content=None,
        tool_calls=[ToolCall(id="read", name="read", arguments={"path": "source.txt"})],
        finish_reason="tool_calls", usage=Usage(prompt_tokens=10, completion_tokens=5))
    client.build_assistant_message.return_value = {"role": "assistant", "content": None}
    trace = io.StringIO()
    session = Session(cfg, client, "system", "inspect", str(tmp_path), trace_file=trace)
    session.run()
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    calls = [r for r in rows if r.get("event") == "tool_call"]
    assert len(calls) == 30
    assert sum(bool(r.get("gate_blocked")) for r in calls) == 3
    assert not calls[-1].get("gate_blocked")


@pytest.mark.parametrize("changing", [False, True])
def test_request_repeat_reuses_only_current_retained_read(tmp_path, changing):
    cfg = make_config(max_turns=6, loop_detect_enabled=True, loop_detect_threshold=2,
                      duplicate_guard_enabled=False, rumination_enabled=False,
                      tools_output_dedup_enabled=False, auto_commit=False)
    requests = []
    client = MagicMock()
    client.is_replay = False
    def chat(messages, tools, **kwargs):
        requests.append(str(messages))
        if changing or len(requests) == 1:
            (tmp_path / "status.txt").write_text(str(len(requests)))
        return TurnResult(content=None, tool_calls=[ToolCall(id=str(len(requests)), name="read",
            arguments={"path": "status.txt"})], finish_reason="tool_calls",
            usage=Usage(prompt_tokens=10, completion_tokens=5))
    client.chat.side_effect = chat
    client.build_assistant_message.return_value = {"role": "assistant", "content": None}
    session = Session(cfg, client, "system", "inspect", str(tmp_path))
    session.run()
    assert ("unchanged since turn" in requests[-1]) is (not changing)
    assert "identical name and arguments" in requests[-1]
    if changing:
        assert "5" in requests[-1]


@pytest.mark.parametrize("passes", [True, False, "mutates"])
def test_done_runs_actual_available_component_once(tmp_path, passes):
    marker = tmp_path.parent / (tmp_path.name + "-check-ran")
    (tmp_path / "core.py").write_text("VALUE = 0\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_core.py").write_text(
        'from pathlib import Path\n'
        'def test_value():\n'
        f'    Path({str(marker)!r}).open("a").write("x")\n'
        + ('    Path("core.py").write_text("VALUE = 9\\n")\n' if passes == "mutates" else '') +
        f'    assert {passes!r}\n')
    cfg = make_config(max_turns=4, sandbox_bash=False, auto_commit=False,
        done_guard_enabled=True, done_require_verify=True,
        post_mutation_verification_gate_after=0, analysis_task_format="pytest",
        done_loop_abort_after=0)
    turns = [ToolCall(id="write", name="write", arguments={"path": "core.py", "content": "VALUE = 1\n"})]
    turns += [ToolCall(id=f"done{i}", name="done", arguments={"message": "finished"}) for i in range(3)]
    client = MagicMock()
    client.chat.side_effect = [TurnResult(content=None, tool_calls=[tc], finish_reason="tool_calls",
        usage=Usage(prompt_tokens=10, completion_tokens=5)) for tc in turns]
    client.build_assistant_message.return_value = {"role": "assistant", "content": None}
    trace = io.StringIO()
    result = Session(cfg, client, "system", "change value", str(tmp_path), trace_file=trace).run()
    assert marker.read_text() == "x"
    assert result.done is (passes is True), str(client.chat.call_args_list[-1])
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    checks = [r for r in rows if r.get("gate_reason") == "done_verification"]
    assert len(checks) == 1, str(client.chat.call_args_list[-1])
    assert checks[0]["tool_dispatch_ms"] > 0


@pytest.mark.parametrize("enabled", [True, False])
def test_done_reports_missing_target_once_without_inventing_pass(tmp_path, enabled):
    cfg = make_config(max_turns=4, sandbox_bash=False, auto_commit=False,
                      done_guard_enabled=True, done_require_verify=True,
                      analysis_task_format="pytest", done_loop_abort_after=0,
                      guard_ladders={} if enabled else {"done_without_check": {"rungs": {}}})
    calls = [ToolCall(id="write", name="write", arguments={"path": "core.py", "content": "VALUE = 1\n"})]
    calls += [ToolCall(id=f"done{i}", name="done", arguments={}) for i in range(3)]
    client = MagicMock()
    client.chat.side_effect = [TurnResult(content=None, tool_calls=[tc], finish_reason="tool_calls",
        usage=Usage(prompt_tokens=10, completion_tokens=5)) for tc in calls]
    client.build_assistant_message.return_value = {"role": "assistant", "content": None}
    trace = io.StringIO()
    session = Session(cfg, client, "system", "edit", str(tmp_path), trace_file=trace)
    assert not session.run().done
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    assert sum(r.get("gate_reason") == "done_verification" for r in rows) == int(enabled)
    assert not session._guards.verified_since_mutation


def _completion_session(tmp_path, calls, **overrides):
    cfg = make_config(max_turns=len(calls), sandbox_bash=False, auto_commit=False,
        done_guard_enabled=True, done_require_mutation=True, done_require_verify=True,
        done_require_pretest_parity=False, done_loop_abort_after=0,
        allow_implicit_done=True, reply_mode="conversation", analysis_task_format="pytest",
        **overrides)
    client = MagicMock()
    client.is_replay = False
    client.chat.side_effect = [TurnResult(
        content=call if isinstance(call, str) else None,
        tool_calls=[] if isinstance(call, str) else [call],
        finish_reason="stop" if isinstance(call, str) else "tool_calls",
        usage=Usage(prompt_tokens=10, completion_tokens=5)) for call in calls]
    client.build_assistant_message.side_effect = lambda content, tools, **kwargs: {
        "role": "assistant", "content": content,
        **({"tool_calls": [{"id": tc.id, "type": "function", "function": {
            "name": tc.name, "arguments": json.dumps(tc.arguments)}} for tc in tools]} if tools else {})}
    session = Session(cfg, client, "system", "Repair the implementation", str(tmp_path))
    return session, client


@pytest.mark.parametrize("implicit", [False, True])
def test_both_finishes_require_observed_work(tmp_path, implicit):
    finish = "All tests pass" if implicit else ToolCall(
        id="done", name="done", arguments={"message": "All tests pass"})
    session, _ = _completion_session(tmp_path, [finish])
    assert not session.run().done
    assert not session._guards.has_mutated
    assert session._guards.done_blocked_count == 1


@pytest.mark.parametrize("passes", [False, True])
def test_implicit_finish_runs_component_and_delivers_result(tmp_path, passes):
    (tmp_path / "core.py").write_text("VALUE = 0\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_core.py").write_text(f"def test_core():\n    assert {passes}\n")
    calls = [ToolCall(id="edit", name="write", arguments={
        "path": "core.py", "content": "VALUE = 1\n"}), "Finished", "Finished"]
    session, client = _completion_session(tmp_path, calls)
    assert session.run().done is passes
    request = client.chat.call_args_list[-1].args[0]
    assert "automatic_verification" in str(request)
    # Only the actual edit is recorded as an assistant tool request.
    assert all(tc["function"]["name"] != "done" for msg in request
               for tc in msg.get("tool_calls", []))
    assert session._guards.formal_verification_passed_since_mutation is passes


@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("probe", ["print(1)", "import core; print(1)"])
def test_custom_probe_cannot_clear_failed_component(tmp_path, implicit, probe):
    (tmp_path / "core.py").write_text("VALUE = 0\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_core.py").write_text("def test_core():\n    assert False\n")
    finish = "All tests pass" if implicit else ToolCall(
        id="done", name="done", arguments={"message": "All tests pass"})
    calls = [ToolCall(id="edit", name="write", arguments={
        "path": "core.py", "content": "VALUE = 1\n"}), finish,
        ToolCall(id="probe", name="bash", arguments={
            "cmd": shlex.join([sys.executable, "-c", probe])}), finish, finish]
    session, _ = _completion_session(tmp_path, calls,
        post_mutation_verification_gate_after=3)
    assert not session.run().done
    assert session._guards.formal_verification_failure_pending
    assert not session._guards.verified_since_mutation


def test_formal_failure_survives_custom_probe_and_unavailable_runner():
    from scripts.llm_solver.harness._guardrails.checks_post import mark_bash_verified
    from scripts.llm_solver.harness._guardrails.verification import observe_post_mutation_verification
    cfg = make_config()
    state = init_guardrail_state(cfg)
    record_mutation(state)
    for status, code in [("failed", 1), ("custom_passed", 0), ("runner_unavailable", 127), ("passed", 0)]:
        metadata = {"executed": True, "exit_status_known": True,
                    "exit_status": code, "verification_status": status}
        mark_bash_verified(state, cfg, tc_name="bash", result="", gate_blocked=False,
                           execution_metadata=metadata)
        observe_post_mutation_verification(state, cfg, tc_name="bash", result="",
            gate_blocked=False, execution_metadata=metadata)
        assert state.verified_since_mutation is (status == "passed")
        assert state.formal_verification_failure_pending is (status != "passed")
        if status == "failed":
            record_mutation(state)
            assert state.formal_verification_failure_pending


@pytest.mark.parametrize("done_enabled,post_gate", [(True, 0), (False, 3)])
def test_unavailable_runner_does_not_release_an_existing_failure(done_enabled, post_gate):
    from scripts.llm_solver.harness._guardrails.checks_pre import done_guard
    cfg = make_config(done_guard_enabled=done_enabled,
                      post_mutation_verification_gate_after=post_gate,
                      done_loop_abort_after=0)
    state = init_guardrail_state(cfg)
    record_mutation(state)
    state.formal_verification_failure_pending = True
    state.post_mutation_automatic_verification_unavailable = True
    assert done_guard(state, cfg, tc_name="done").action == Action.BLOCK


@pytest.mark.parametrize("name,args", [("run_tests", {"path": "tests"}),
                                      ("bash", {"cmd": "pytest tests/test_core.py"})])
def test_no_edit_gates_allow_verification_without_releasing_inspection(name, args):
    from scripts.llm_solver.harness._guardrails.checks_pre import pre_mutation_gate
    cfg = make_config(rumination_enabled=True, pre_mutation_turn_cap=1)
    state = init_guardrail_state(cfg)
    state.rumination_gate = True
    state.rumination_gate_grace = 0
    assert pre_mutation_gate(state, cfg, tc_name=name, tc_args=args,
                             turn_number=20).action == Action.PASS
    assert rumination_gate(state, cfg, tc_name=name, tc_args=args).action == Action.PASS
    assert state.gate_block_count == 0
    assert rumination_gate(state, cfg, tc_name="read", tc_args={"path": "core.py"}).action == Action.BLOCK
