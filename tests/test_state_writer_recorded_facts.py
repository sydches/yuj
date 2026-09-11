"""Replay path and check evidence without borrowing live facts or clipped text."""
from dataclasses import asdict
import json

import pytest

from scripts.llm_solver.harness import state_writer, task_environment


def start(session, root):
    environment = task_environment.TaskEnvironment('/host/' + str(session), root, (root,))
    return {"event": "session_start", "session_number": session,
            "task_environment": asdict(environment)}


def tool(command, *, session=1, **fields):
    return {"event": "tool_call", "tool_name": "bash", "session_number": session,
            "turn_number": 1, "args_summary": "cmd=" + repr(command),
            "result_summary": "", **fields}


def project(events):
    return state_writer.project(events, max_result_chars=40, imperative_projection=True)


@pytest.mark.parametrize("ambient", [None, "/observed/task", "/unrelated/task"])
def test_each_session_uses_its_own_recorded_mapping_and_restores_live_context(ambient):
    active = task_environment.TaskEnvironment("/host/live", ambient, (ambient,)) if ambient else None
    token = task_environment._ACTIVE.set(active)
    try:
        events = [start(1, "/observed/task"),
                  tool("sed -i 's/a/b/' /observed/task/src/one.py", pass_fail="pass"),
                  start(2, "/another/task"),
                  tool("sed -i 's/a/b/' /another/task/src/two.py", session=2, pass_fail="pass"),
                  tool("cat /observed/task/outside.py", session=2)]
        result = project(events)
        assert result["process"]["last_mutation_step"] == 2
        assert result["process"]["target_paths"] == ["src/one.py", "src/two.py"]
        assert [row["path_binding"] for row in result["trace"]] == ["recorded"] * 3
        assert task_environment._ACTIVE.get() is active
    finally:
        task_environment._ACTIVE.reset(token)


@pytest.mark.parametrize("record", [None, {}, {"host_root": "/task", "working_directory": "/task", "aliases": [None]}])
def test_missing_or_malformed_binding_does_not_borrow_live_paths(record):
    token = task_environment._ACTIVE.set(task_environment.TaskEnvironment("/task", "/task", ("/task",)))
    try:
        result = project([{"event": "session_start", "session_number": 1, "task_environment": record},
                          tool("sed -i 's/a/b/' /task/source.py", source_write_like=True,
                               source_write_paths=["/task/source.py"]),
                          tool("cat src/relative.py")])
        assert result["process"]["phase"] == "pre_mutation_discovery"
        assert result["process"]["target_paths"] == ["src/relative.py"]
        assert result["trace"][0]["path_binding"] == "unknown"
    finally:
        task_environment._ACTIVE.reset(token)


def test_session_without_mapping_cannot_reuse_previous_session_mapping():
    result = project([start(1, "/task"), {"event": "session_start", "session_number": 2},
                      tool("sed -i 's/a/b/' /task/source.py", session=2)])
    assert result["process"]["phase"] == "pre_mutation_discovery"
    assert result["trace"][0]["path_binding"] == "unknown"


def test_classification_precedes_display_limits_without_expanding_model_state(monkeypatch):
    events = [start(1, "/task"), tool("sed -i 's/a/b/' /task/source.py"),
              tool("SETTING=" + "x" * 180 + " pytest tests/", verification_status="passed", pass_fail="pass"),
              tool("SETTING=" + "x" * 180 + " pytest tests/")]
    before = project(events)
    monkeypatch.setattr(state_writer, "_MAX_ACTION_CHARS", 8)
    after = project(events)
    assert after["process"] == before["process"]
    assert after["trace"][-2]["check_kind"] == "completed_test_check"
    assert after["trace"][-1]["check_kind"] == "check_attempt"
    assert len(after["trace"][-1]["action"]) <= len("bash()") + 8
    assert "x" * 180 not in json.dumps(after)
    assert after["process"]["target_paths"] == ["source.py"]


@pytest.mark.parametrize("status,kind,completed", [
    ("passed", "completed_test_check", True),
    ("failed", "completed_test_check", True),
    ("process_running", "check_attempt", False),
    ("timed_out", "check_attempt", False),
    ("collection_error", "check_attempt", False),
    ("shell_unresolved", "check_attempt", False),
    ("error", "check_attempt", False),
    ("", "check_attempt", False),
])
def test_only_recorded_completed_test_status_advances_phase(status, kind, completed):
    result = project([tool("sed -i 's/a/b/' source.py"),
                      tool("pytest tests/", verification_status=status, pass_fail="pass")])
    assert (result["process"]["phase"] == "post_verification") is completed
    assert result["evidence"][-1]["kind"] == kind
    assert "call done" not in result["process"]["suggested_next_action"]


@pytest.mark.parametrize("command,status", [
    ('python -c "print(1)"', ""),
    ('python -c "assert 1 == 1"', "custom_passed"),
    ('python -c "assert False"', "custom_failed"),
])
def test_custom_probes_remain_visible_without_becoming_task_verification(command, status):
    result = project([tool("sed -i 's/a/b/' source.py"),
                      tool(command, verification_status=status)])
    assert result["process"]["phase"] == "post_mutation_unverified"
    assert result["evidence"][-1]["kind"] == "custom_probe"
    assert result["trace"][-1]["action"].startswith("bash(")


def test_unparseable_legacy_command_does_not_invent_completed_check():
    result = project([tool("sed -i 's/a/b/' source.py"),
                      {**tool("pytest"), "args_summary": "cmd='pytest tests/..."}])
    assert result["process"]["phase"] == "post_mutation_unverified"
    assert result["trace"][-1]["check_kind"] == "command_result"


def test_recorded_scope_restores_context_after_exception():
    original = task_environment._ACTIVE.get()
    with pytest.raises(RuntimeError):
        with task_environment.recorded_task_environment(start(1, "/task")["task_environment"]):
            raise RuntimeError("fixture")
    assert task_environment._ACTIVE.get() is original
