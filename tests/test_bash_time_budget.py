"""Shell execution shares the caller deadline without benchmark durations."""
import importlib
import io
import shlex
import sys
import time
from types import SimpleNamespace

import pytest

from tests._config_helpers import make_config
from scripts.llm_solver.harness import time_budget as budgets
from scripts.llm_solver.harness import tools
from scripts.llm_solver.harness._loop.trace_output import build_tool_call_trace_fields


def call(tmp_path, *, ceiling=0, command="printf result", sandbox=False):
    cfg = make_config(bash_timeout=ceiling, sandbox_bash=sandbox)
    facts = {}
    result = tools.dispatch("bash", {"cmd": command}, cwd=str(tmp_path),
                            cfg=cfg, execution_metadata=facts)
    return result, facts, cfg


@pytest.mark.parametrize("ceiling, expected", [(0, [7, 5]), (4, [4, 4]), (20, [7, 5])])
def test_calls_use_remaining_run_time_and_optional_ceiling(tmp_path, monkeypatch, ceiling, expected):
    clock = [0.0]
    monkeypatch.setattr(budgets.time, "monotonic", lambda: clock[0])
    limits = []
    def execute(*args, **kwargs):
        limits.append(budgets.command_timeout(kwargs["timeout"]))
        clock[0] += 2
        return "result", 0, False
    monkeypatch.setattr(tools, "_run_in_sandbox", execute)
    with budgets.run_time_budget(10):
        clock[0] = 3
        for limit in expected:
            _, facts, _ = call(tmp_path, ceiling=ceiling)
            assert facts["execution_budget"]["effective_seconds"] == limit
            assert facts["execution_budget"]["declared_run_seconds"] == 10
    assert limits == expected
    assert not budgets.command_is_scoped()


@pytest.mark.parametrize("ceiling", [0, 7])
def test_absent_run_deadline_is_not_invented(tmp_path, monkeypatch, ceiling):
    observed = []
    def execute(*args, **kwargs):
        observed.append(kwargs["timeout"])
        return "result", 0, False
    monkeypatch.setattr(tools, "_run_in_sandbox", execute)
    _, facts, _ = call(tmp_path, ceiling=ceiling)
    assert observed == [ceiling or None]
    assert facts["execution_budget"]["declared_run_seconds"] is None
    assert facts["execution_budget"]["status"] == ("allocated" if ceiling else "unbounded")


def test_exhaustion_prevents_even_the_inprocess_read_and_is_recorded(tmp_path, monkeypatch):
    bash_module = importlib.import_module("scripts.llm_solver.harness._tools.bash")
    monkeypatch.setattr(bash_module, "_try_inproc_trivial_read",
                        lambda *a, **k: pytest.fail("read after exhaustion"))
    monkeypatch.setattr(tools, "_run_in_sandbox",
                        lambda *a, **k: pytest.fail("launch after exhaustion"))
    clock = [0.0]
    monkeypatch.setattr(budgets.time, "monotonic", lambda: clock[0])
    with budgets.run_time_budget(1):
        clock[0] = 1
        result, facts, cfg = call(tmp_path, command="cat x", sandbox=True)
    assert facts["executed"] is False and facts["exit_status"] is None
    assert not facts["timed_out"]
    assert facts["execution_budget"]["status"] == "exhausted"
    session = SimpleNamespace(cfg=cfg, cwd=str(tmp_path), _sink_counter=0, _session_number=1)
    trace = build_tool_call_trace_fields(session, tool_name="bash", args_summary="cat x",
        result=result, turn=1, gate_blocked=False, execution_metadata=facts)
    assert trace["execution_budget"] == facts["execution_budget"]


def test_backend_setup_consumes_the_same_allowance(tmp_path, monkeypatch):
    from scripts.llm_solver.harness._tools import _run_in_sandbox as backend
    clock = [0.0]
    monkeypatch.setattr(budgets.time, "monotonic", lambda: clock[0])
    def setup(*args, **kwargs):
        clock[0] = 2
        return ["must-not-run"]
    monkeypatch.setattr(backend, "build_bash_argv", setup)
    monkeypatch.setattr(backend.subprocess, "Popen", lambda *a, **k: pytest.fail("late launch"))
    monkeypatch.setattr(backend.subprocess, "run", lambda *a, **k: pytest.fail("late launch"))
    with budgets.run_time_budget(1):
        _, facts, _ = call(tmp_path)
    assert not facts["executed"] and facts["verification_status"] == "budget_exhausted"


def test_parent_command_allowance_cannot_be_extended(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(budgets.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(tools, "_run_in_sandbox", lambda *a, **k: ("result", 0, False))
    with budgets.command_time_budget(3):
        clock[0] = 1
        _, facts, _ = call(tmp_path, ceiling=20)
        assert facts["execution_budget"]["effective_seconds"] == 2
        assert facts["execution_budget"]["limiting_source"] == "parent_command"


def test_real_shell_times_out_under_remaining_run_time(tmp_path):
    command = shlex.join([sys.executable, "-c", "import time; time.sleep(2)"])
    started = time.monotonic()
    with budgets.run_time_budget(0.2):
        result, facts, _ = call(tmp_path, command=command)
    assert time.monotonic() - started < 0.9
    assert facts["executed"] and facts["timed_out"] and facts["exit_status"] is None
    assert 0 < facts["execution_budget"]["effective_seconds"] <= 0.2
    assert "timed out" in result


def test_expiry_after_process_start_is_timeout_not_unexecuted(tmp_path, monkeypatch):
    from scripts.llm_solver.harness._tools import _run_in_sandbox as backend
    # Isolate the submitted Bash process from the earlier file-inventory probe.
    monkeypatch.setattr("scripts.llm_solver.harness.file_changes._inventory",
                        lambda *args: ({}, []))
    clock = [0.0]
    monkeypatch.setattr(budgets.time, "monotonic", lambda: clock[0])
    killed = []
    process = SimpleNamespace(pid=123, stdout=io.StringIO(), stderr=io.StringIO(),
                              wait=lambda: None,
                              communicate=lambda **kwargs: pytest.fail("communicated after expiry"))
    def start(*args, **kwargs):
        clock[0] = 2
        return process
    monkeypatch.setattr(backend.subprocess, "Popen", start)
    monkeypatch.setattr(backend.os, "killpg", lambda pid, signal: killed.append(pid))
    with budgets.run_time_budget(1):
        _, facts, _ = call(tmp_path)
    assert killed == [123]
    assert facts["executed"] and facts["timed_out"] and facts["exit_status"] is None


@pytest.mark.parametrize("ceiling", [False, True, -1, float("inf"), float("nan")])
def test_invalid_shell_allowance_is_not_treated_as_unlimited(tmp_path, ceiling):
    with pytest.raises(ValueError):
        tools.bash("printf unused", cwd=str(tmp_path), timeout=ceiling, sandbox=False)
