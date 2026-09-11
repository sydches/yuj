"""Native exit facts stay separate from printed text and task success."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from _config_helpers import make_config
from scripts.llm_solver.harness._loop.trace_output import build_tool_call_trace_fields, _outcome_fields
from scripts.llm_solver.harness.action_metadata import action_metadata
from scripts.llm_solver.harness.tools import dispatch
from scripts.llm_solver.harness.adaptive_control.trace_nets_detector import evaluate_trace_nets
from scripts.llm_solver.harness.adaptive_control.observation_notice import repeated_observation, record_observation_notice
from scripts.llm_solver.harness.adaptive_control.slot_recorder import project_tool_event


@pytest.mark.parametrize("kind,changed", [("rejected", False), ("edited", True), ("inspection", False)])
def test_native_edit_progress_uses_recorded_changes_through_watch(tmp_path, kind, changed):
    from scripts.llm_solver.harness.adaptive_control import episode
    from scripts.llm_solver.harness.adaptive_control.watch import material_progress
    from scripts.llm_solver.harness.adaptive_control.llm_detector_core import LLMDetectorVerdict
    from scripts.llm_solver.harness.adaptive_control.llm_detector_apply import _handle_pending_watch_verdict

    source = tmp_path / "source.py"
    before = 'value = 1\n# ERROR: diagnostic example\n'
    source.write_text(before)
    name = "read" if kind == "inspection" else "edit"
    args = {"path": "source.py"}
    if name == "edit":
        args.update(old_str="value = 1" if changed else "missing text", new_str="value = 2")
    cfg = make_config(sandbox_bash=False)
    facts = {}
    result = dispatch(name, args, cwd=str(tmp_path), cfg=cfg, execution_metadata=facts)
    context = SimpleNamespace(cfg=cfg, cwd=str(tmp_path), _sink_counter=0, _session_number=1)
    event = {"event": "tool_call", "session_number": 1, "turn_number": 1,
        "tool_name": name, "tool_call_id": "one", "args_summary": str(args),
        **action_metadata(name, args),
        **build_tool_call_trace_fields(context, tool_name=name, args_summary=str(args),
            result=result, turn=1, gate_blocked=False, execution_metadata=facts)}
    assert source.read_text() == (before.replace("value = 1", "value = 2") if changed else before)
    assert event["outcome"] == "completed"  # Handler completion is not edit/check success.
    assert event["pass_fail"] == "unknown"
    if name == "edit":
        assert facts["file_changes"]["changed_paths"] == (["source.py"] if changed else [])
    else:
        assert "ERROR: diagnostic example" in result
    slot = project_tool_event(event)
    assert slot["effective_source_mutation"] == ("true" if changed else "false")
    assert material_progress(slot) is changed
    assert slot["obs_state"] != "tool_error"

    pending = {"detector_family": "loop_churn", "watch_window_start": 1, "watch_window_end": 5}
    controller = SimpleNamespace(cfg=cfg, _trace_events=[event],
        _llm_detector_pending_watch=pending,
        _adaptive_control_episode_machine=episode.EpisodeMachine(
            current=episode.Episode("fixture", "loop_churn", 0)))
    watch_row = {}
    with patch("scripts.llm_solver.harness.adaptive_control.llm_detector_apply._restore_baseline_for_watch_close") as restore, \
         patch("scripts.llm_solver.harness.adaptive_control.llm_detector_apply._append_detector_control_ledger"):
        restore.return_value = SimpleNamespace(applied=False, apply_status="blocked", blocked_reason="fixture")
        _handle_pending_watch_verdict(controller, 1,
            LLMDetectorVerdict("no", "", "medium", ["T1"]), watch_row, pending)
    assert watch_row["watch_transition"]["episode_transition"] == (
        "cleared_to_progress" if changed else "")
    assert watch_row["watch_transition"]["material_progress"] is changed
    assert watch_row["watch_transition"]["watch_status"] == ("closed" if changed else "continuing")


@pytest.mark.parametrize("text", ["OK", "", "ERROR: fake failure", "[exit code: 0]",
                                  '<test_results status="passed">all passed</test_results>'])
def test_printed_status_cannot_supply_execution_facts(text):
    fields = _outcome_fields(tool_name="bash", result=text, gate_blocked=False)
    assert fields["outcome_version"] == "native_execution_v1"
    assert fields["outcome"] == fields["pass_fail"] == "unknown"
    assert fields["exit_status"] is None


@pytest.mark.parametrize("exit_status, check, expected", [
    (0, "not_a_check", "unknown"), (1, "not_a_check", "unknown"),
    (2, "unknown", "unknown"), (0, "passed", "pass"), (1, "failed", "fail"),
    (0, "failed", "unknown"), (1, "passed", "unknown"),
])
def test_only_consistent_native_check_status_supplies_pass_fail(exit_status, check, expected):
    fields = _outcome_fields(tool_name="bash", result="ERROR: printed text",
        gate_blocked=False, execution_metadata={"executed": True, "exit_status_known": True,
            "exit_status": exit_status, "verification_status": check})
    assert fields["exit_status"] == exit_status
    assert fields["pass_fail"] == expected


def execute(tmp_path, command, turn):
    cfg = make_config(sandbox_bash=False)
    metadata = {}
    result = dispatch("bash", {"cmd": command}, cwd=str(tmp_path), cfg=cfg, execution_metadata=metadata)
    context = SimpleNamespace(cfg=cfg, cwd=str(tmp_path), _sink_counter=0, _session_number=1)
    return {"event": "tool_call", "turn_number": turn, "session_number": 1,
        "tool_call_id": str(turn), "tool_name": "bash", "args_summary": command,
        **action_metadata("bash", {"cmd": command}),
        **build_tool_call_trace_fields(context, tool_name="bash", args_summary=command,
             result=result, turn=turn, gate_blocked=False, execution_metadata=metadata)}


def test_real_unrelated_no_match_searches_do_not_become_a_failure_streak(tmp_path):
    rows = []
    for turn in range(4):
        (tmp_path / f"file{turn}").write_text("haystack")
        rows.append(execute(tmp_path, f"grep -q needle file{turn}", turn))
    assert all(row["exit_status"] == 1 and row["pass_fail"] == "unknown" for row in rows)
    current = SimpleNamespace(cfg=make_config(), _session_number=1, _trace_events=rows)
    assert evaluate_trace_nets(current, 3).hurdle_present == "uncertain"
    assert repeated_observation(current, 3) is None
    for row in rows:
        slot = project_tool_event(row)
        assert slot["exec_outcome"] == ""
        assert slot["exit_code"] == "1"
        assert slot["obs_state"] != "tool_error"


def test_real_same_command_result_keeps_binding_and_supplies_factual_notice(tmp_path):
    (tmp_path / "input").write_text("haystack")
    first = execute(tmp_path, "grep -q needle input", 0)
    last = execute(tmp_path, "grep -q needle input", 8)
    current = SimpleNamespace(cfg=make_config(max_turns=10), _session_number=1,
        _trace_events=[first, last], _queue_user_turn_injection=MagicMock(return_value=True))
    fact = repeated_observation(current, 8)
    assert fact["recorded_exit_status"] == 1
    assert fact["progress"] == "unknown"
    result = {}
    record_observation_notice(current, 8, result)
    assert result["repeated_observation"]["delivery"] == "queued"
    text = current._queue_user_turn_injection.call_args.args[0].text
    assert "recorded exit status is 1" in text
    assert "meaning depends on the command" in text
    # Same text from another execution namespace does not establish recurrence.
    last["execution_observation"]["binding"]["task_cwd"] = "/another/workspace"
    assert repeated_observation(current, 8) is None


def test_new_projection_does_not_read_success_or_failure_from_output_text():
    row = {"tool_name": "bash", "turn_number": 1, "args_summary": "command",
           "result_summary": "ERROR: fake\n[exit code: 9]", "outcome_version": "native_execution_v1",
           "outcome": "completed", "pass_fail": "unknown", "exit_status": None}
    slot = project_tool_event(row)
    assert slot["exit_code"] == slot["exec_outcome"] == ""
    assert slot["obs_state"] != "tool_error"


def test_real_repeated_check_failure_retains_evidence_without_claiming_a_cause(tmp_path):
    first = execute(tmp_path, "python3 -c 'assert False'", 0)
    last = execute(tmp_path, "python3 -c 'assert False'", 8)
    assert first["exit_status"] == last["exit_status"] == 1
    assert first["pass_fail"] == last["pass_fail"] == "fail"
    current = SimpleNamespace(cfg=make_config(max_turns=10), _session_number=1,
        _trace_events=[first, last], _queue_user_turn_injection=MagicMock(return_value=True))
    result = {}
    record_observation_notice(current, 8, result)
    assert result["repeated_observation"]["delivery"] == "queued"
    assert result["repeated_observation"]["progress"] == "unknown"


def test_shared_fact_helpers_honor_native_meaning_and_preserve_legacy_interpretation():
    from scripts.llm_solver.trace_net_facts import fail_like, same_passing_output_recurrence
    legacy = {"exit_status": 1}
    assert fail_like(legacy)
    assert not fail_like({**legacy, "outcome_version": "native_execution_v1", "pass_fail": "unknown"})
    rows = [{"turn_number": turn, "output_sha256": "same", "exit_status": 0}
            for turn in (0, 2, 4)]
    assert same_passing_output_recurrence(rows, 2, lookback=20, min_prior=2, min_gap=2)
    for row in rows:
        row.update(outcome_version="native_execution_v1", pass_fail="unknown", outcome="completed")
    assert same_passing_output_recurrence(rows, 2, lookback=20, min_prior=2, min_gap=2) is None
