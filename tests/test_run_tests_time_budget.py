"""Run tests use declared time, never a benchmark-derived duration estimate."""
from pathlib import Path
import shlex
import sys
import time
from types import SimpleNamespace

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness import time_budget as budgets
from scripts.llm_solver.harness.tools import dispatch
from scripts.llm_solver.harness._loop.trace_output import build_tool_call_trace_fields


def config(**changes):
    return make_config(tools_run_tests_enabled=True, analysis_task_format="pytest",
                       sandbox_bash=False, **changes)


def test_allocation_uses_the_smaller_declared_limit_and_does_not_reset(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(budgets.time, "monotonic", lambda: clock[0])
    with budgets.run_time_budget(10):
        clock[0] += 3
        first = budgets.command_allowance(4)
        assert first.record["effective_seconds"] == 4
        assert first.record["limiting_source"] == "per_call_limit"
        clock[0] += 5
        second = budgets.command_allowance(4)
        assert second.record["effective_seconds"] == 2
        assert second.record["limiting_source"] == "run_remaining"
        with budgets.run_time_budget(100):
            assert budgets.remaining_run_seconds() == 2
            assert budgets.command_allowance().record["declared_run_seconds"] == 10
    assert budgets.remaining_run_seconds() is None


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True])
def test_invalid_limits_do_not_become_unlimited(value):
    with pytest.raises(ValueError):
        budgets.command_allowance(value)


def test_no_declared_deadline_is_explicitly_unbounded():
    allocation = budgets.command_allowance()
    assert allocation.remaining() is None
    assert allocation.record["limiting_source"] == "no_declared_limit"
    assert allocation.record["status"] == "unbounded"


def test_actual_solve_entry_scopes_budget_even_when_setup_fails(tmp_path, monkeypatch):
    from scripts.llm_solver.harness._loop import driver
    seen = []
    def stop(*args):
        seen.append(budgets.remaining_run_seconds())
        raise RuntimeError("fixture stops before any task setup")
    monkeypatch.setattr(driver, "resolve_run_paths", stop)
    with pytest.raises(RuntimeError, match="fixture stops"):
        driver.solve_task(tmp_path, make_config(task_wall_clock_limit_s=10), object())
    assert 0 < seen[0] <= 10
    assert budgets.remaining_run_seconds() is None


@pytest.mark.parametrize("scope", ["run", "parent_command"])
def test_exhausted_budget_refuses_before_discovery_and_survives_trace(tmp_path, monkeypatch, scope):
    from scripts.llm_solver.harness import tools
    def must_not_execute(*args, **kwargs):
        pytest.fail("exhausted run launched execution")
    monkeypatch.setattr(tools, "_run_in_sandbox", must_not_execute)
    cfg = config()
    clock = [0.0]
    monkeypatch.setattr(budgets.time, "monotonic", lambda: clock[0])
    budget = budgets.run_time_budget if scope == "run" else budgets.command_time_budget
    with budget(1):
        clock[0] = 1
        facts = {}
        result = dispatch("run_tests", {}, cwd=str(tmp_path), cfg=cfg, execution_metadata=facts)
    assert facts["executed"] is False
    assert '<test_results status="budget_exhausted">' in result
    assert facts["verification_status"] == "budget_exhausted"
    assert facts["exit_status"] is None and not facts["timed_out"]
    assert facts["execution_budget"]["effective_seconds"] == 0
    assert facts["execution_budget"]["limiting_source"] == ("run_remaining" if scope == "run" else scope)
    session = SimpleNamespace(cfg=cfg, cwd=str(tmp_path), _sink_counter=0, _session_number=1)
    fields = build_tool_call_trace_fields(session, tool_name="run_tests", args_summary="", result=result,
                                         turn=1, gate_blocked=False, execution_metadata=facts)
    assert fields["execution_budget"] == facts["execution_budget"]


def test_backend_rechecks_after_sandbox_setup_before_launch(tmp_path, monkeypatch):
    from scripts.llm_solver.harness._tools import _run_in_sandbox as backend
    clock = [0.0]
    monkeypatch.setattr(budgets.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(backend, "get_persistent_runner", lambda: None)
    monkeypatch.setattr(backend, "container_mode", lambda: None)
    def build(*args, **kwargs):
        clock[0] = 2
        return ["must-not-run"]
    monkeypatch.setattr(backend, "_build_bwrap_argv", build)
    monkeypatch.setattr(backend.subprocess, "run", lambda *a, **k: pytest.fail("launched after deadline"))
    monkeypatch.setattr(backend.subprocess, "Popen", lambda *a, **k: pytest.fail("launched after deadline"))
    cfg = make_config(tools_run_tests_enabled=True, analysis_task_format="pytest", sandbox_bash=True,
                      bwrap_bin=sys.executable)
    with budgets.run_time_budget(1):
        facts = {}
        dispatch("run_tests", {}, cwd=str(tmp_path), cfg=cfg, execution_metadata=facts)
    assert facts["executed"] is False
    assert facts["verification_status"] == "budget_exhausted"


@pytest.mark.parametrize("backend_kind", ["subprocess", "bwrap", "persistent"])
def test_real_execution_uses_remaining_time_without_extra_grace(tmp_path, backend_kind):
    from scripts.llm_solver.harness.sandbox import (
        bwrap_preflight, PersistentBashSession, set_persistent_runner,
    )
    from scripts.llm_solver.harness.task_environment import task_environment_scope
    if backend_kind != "subprocess":
        ok, reason = bwrap_preflight("/usr/bin/bwrap")
        if not ok:
            pytest.skip(reason)
    env = {"PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin", "HOME": str(tmp_path)}
    cfg = make_config(tools_run_tests_enabled=True, analysis_task_format="pytest",
                      sandbox_bash=backend_kind != "subprocess", sandbox_required=True,
                      sandbox_env_set=env)
    slow = shlex.join([sys.executable, "-c", "import time; time.sleep(2)"])
    quick = shlex.join([sys.executable, "-c", "print('completed')"])
    runner = None

    @task_environment_scope
    def session():
        nonlocal runner
        if backend_kind == "persistent":
            runner = PersistentBashSession(cwd=str(tmp_path), bwrap_bin="/usr/bin/bwrap",
                                            effective_env=env, sandbox_required=True)
            set_persistent_runner(runner)
        try:
            facts = {}
            started = time.monotonic()
            with budgets.run_time_budget(0.2):
                dispatch("run_tests", {"_base_cmd_override": slow}, cwd=str(tmp_path),
                         cfg=cfg, execution_metadata=facts)
            assert time.monotonic() - started < 0.9
            assert facts["timed_out"] is True and facts["exit_status"] is None
            assert facts["verification_status"] == "timed_out"
            assert 0 < facts["execution_budget"]["effective_seconds"] <= 0.2
            facts = {}
            result = dispatch("run_tests", {"_base_cmd_override": quick}, cwd=str(tmp_path),
                              cfg=cfg, execution_metadata=facts)
            assert "completed" in result and facts["exit_status"] == 0
            assert facts["execution_budget"]["status"] == "unbounded"
        finally:
            set_persistent_runner(None)
            if runner is not None:
                runner.close()
    session()


def test_timeout_does_not_wait_for_inherited_child_output_pipes(tmp_path):
    import os
    code = (
        "import subprocess,sys,time; from pathlib import Path; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(2)']); "
        "Path('child.pid').write_text(str(p.pid)); time.sleep(2)"
    )
    command = shlex.join([sys.executable, "-c", code])
    started = time.monotonic()
    facts = {}
    with budgets.run_time_budget(0.2):
        dispatch("run_tests", {"_base_cmd_override": command}, cwd=str(tmp_path),
                 cfg=config(), execution_metadata=facts)
    assert time.monotonic() - started < 0.9
    assert facts["timed_out"] and facts["exit_status"] is None
    pid = int((tmp_path / "child.pid").read_text())
    stat = Path(f"/proc/{pid}/stat")
    # Signal delivery to the child can finish after the direct parent is reaped.
    # This fixture wait is outside the timed tool call checked above.
    observed_until = time.monotonic() + 0.5
    while time.monotonic() < observed_until:
        try:
            state = stat.read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            return
        if state == "Z":
            return
        time.sleep(0.005)
    os.kill(pid, 9)  # Cleanup only the child created by this fixture.
    pytest.fail("runner child survived the timeout")


def test_interrupted_command_cleans_up_its_process_group(monkeypatch):
    from io import StringIO
    from scripts.llm_solver.harness._tools import _run_in_sandbox as backend
    actions = []
    def interrupted(**kwargs):
        raise KeyboardInterrupt
    process = SimpleNamespace(pid=123, stdout=StringIO(), stderr=StringIO(),
                              communicate=interrupted, wait=lambda: actions.append("reaped"))
    monkeypatch.setattr(backend.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(backend.os, "killpg", lambda pid, sig: actions.append((pid, sig)))
    @budgets.budgeted_test_execution
    def execute(*, cfg):
        backend._execute(["fixture"], timeout=None)
    with budgets.run_time_budget(10), pytest.raises(KeyboardInterrupt):
        execute(cfg=config())
    assert actions == [(123, backend.signal.SIGKILL), "reaped"]
    assert process.stdout.closed and process.stderr.closed
    assert not budgets.command_is_scoped()
