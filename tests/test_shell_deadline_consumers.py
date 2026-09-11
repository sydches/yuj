"""062 consumer regressions: local cell protocol and stubbed container transport."""
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness import time_budget as budgets, tools, task_environment
from scripts.llm_solver.harness._tools import _run_in_sandbox as backend, exec_cell


@pytest.mark.parametrize("ceiling", [0, 120])
def test_cell_deadline_bounds_inner_shell_without_an_extra_cap(tmp_path, monkeypatch, ceiling):
    # Run only the fixed protocol interpreter directly; nested tool dispatch,
    # process termination and the cell's own deadline remain production code.
    monkeypatch.setattr(exec_cell, "_build_cell_process", lambda **kwargs:
                        ([sys.executable, "-u", "-c", exec_cell._CELL_RUNNER], str(tmp_path), None))
    cfg = make_config(tools_exec_cell_enabled=True, tools_exec_cell_timeout=1,
                      bash_timeout=ceiling, sandbox_bash=False, sandbox_env_inherit="none")
    command = shlex.join([sys.executable, "-c", "import time; time.sleep(3)"])
    facts = {}
    started = time.monotonic()
    result = tools.dispatch("exec_cell", {"source": "print(bash(" + repr(command) + "))"},
        cwd=str(tmp_path), cfg=cfg, effective_env={}, allow_login_shell=False, execution_metadata=facts)
    assert time.monotonic() - started < 2
    assert "timed out" in result.lower()
    inner = facts["exec_cell"]["inner_calls"][0]["execution_metadata"]
    assert inner["timed_out"]
    assert 0 < inner["execution_budget"]["effective_seconds"] <= 1
    assert inner["execution_budget"]["status"] == "allocated"


@pytest.mark.parametrize("finished_at,probe_error", [(0.5, False), (1.0, False), (0.5, True)])
def test_completed_shell_result_survives_unavailable_container_diagnostic(
    tmp_path, monkeypatch, caplog, finished_at, probe_error,
):
    clock = [0.0]
    monkeypatch.setattr(budgets.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(backend, "container_mode", lambda: "fixture-container")
    monkeypatch.setattr(task_environment, "discover_task_environment",
                        lambda *a, **k: SimpleNamespace(container_id="fixture-container"))
    monkeypatch.setattr(backend, "_build_bwrap_argv", lambda *a, **k: ["fixture-only"])
    executed, probes = [], []

    def execute(argv, **kwargs):
        executed.append(argv)
        clock[0] = finished_at
        return subprocess.CompletedProcess(argv, 1, "ordinary diagnostic", "")

    def probe(argv, **kwargs):
        assert argv[:2] == ["docker", "inspect"]
        probes.append(argv)
        if probe_error:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 0, "true", "")

    monkeypatch.setattr(backend, "_execute", execute)
    monkeypatch.setattr(backend.subprocess, "run", probe)
    monkeypatch.setattr(backend.subprocess, "Popen", lambda *a, **k: pytest.fail("unexpected process"))
    with budgets.run_time_budget(1):
        result = tools.bash("printf diagnostic; exit 1", cwd=str(tmp_path), timeout=0,
                            sandbox=True, effective_env={})
    assert len(executed) == 1
    assert len(probes) == (1 if finished_at < 1 else 0)
    assert result.executed is not False
    assert result.exit_status == 1
    assert "ordinary diagnostic" in result
    assert ("container status diagnostic unavailable" in caplog.text) is (finished_at == 1 or probe_error)


def test_parent_only_shell_budget_keeps_allocated_label(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(budgets.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(tools, "_run_in_sandbox", lambda *a, **k: ("done", 0, False))
    with budgets.command_time_budget(3):
        clock[0] = 1
        result = tools.bash("printf done", cwd=str(tmp_path), timeout=0, sandbox=False)
    assert result.execution_budget["effective_seconds"] == 2
    assert result.execution_budget["limiting_source"] == "parent_command"
    assert result.execution_budget["status"] == "allocated"
