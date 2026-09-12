"""Actual discovery and execution in contrasting permitted environments."""
import json
import shlex
from pathlib import Path
import sys
import venv
from unittest.mock import MagicMock

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness._loop._driver_setup import resolve_task_format
from scripts.llm_solver.harness.runtime_discovery import bind_command_environment, discover_runtime
from scripts.llm_solver.harness.tools import dispatch


def _environment(path, *, runner):
    venv.EnvBuilder(with_pip=False).create(path)
    if runner:
        packages = path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        # Test fixture only: reuse the already installed test dependency. No
        # package installation or network is needed to create this environment.
        (packages / "fixture.pth").write_text(str(Path(pytest.__file__).parent.parent) + "\n")
    return path


def _task(path):
    path.mkdir()
    (path / "pyproject.toml").write_text('[project]\nrequires-python=">=3.8"\n[tool.pytest.ini_options]\n')
    (path / "test_math.py").write_text("def test_math():\n    assert 2 + 2 == 4\n")
    return path


@pytest.mark.parametrize("startup_selection", [True, False])
def test_project_environment_outside_initial_path_is_discovered_and_runs_tests(tmp_path, monkeypatch, startup_selection):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    task = _task(tmp_path / "task")
    base = _environment(tmp_path / "base", runner=False)
    selected = _environment(task / "an arbitrary environment", runner=True)
    cfg = make_config(sandbox_bash=False, analysis_task_format="auto", tools_run_tests_enabled=True)
    env = {"PATH": str(base / "bin") + ":/usr/bin:/bin"}
    report = discover_runtime(task, cfg, effective_env=env)
    selection = report["runner_selection"]
    assert selection["status"] == "selected"
    assert selection["selected"]["source"] == "project_environment_marker"
    assert selection["selected"]["runtime"]["prefix"] == str(selected)
    assert selection["selected"]["runtime"]["available"] is True
    if startup_selection:
        cfg = resolve_task_format(cfg, task, runtime_observations=report)
    facts = {}
    result = dispatch("run_tests", {"path": "test_math.py"}, cwd=str(task), cfg=cfg,
                      effective_env=env, execution_metadata=facts)
    assert facts["verification_status"] == "passed", result
    assert facts["exit_status"] == 0
    assert facts["runner_request"]["family"] == "pytest"
    assert facts["runner_request"]["basis"] == "runtime_selection"
    assert facts["runner_request"]["targets"] == ["test_math.py"]
    assert report["elapsed_seconds"] < 10


@pytest.mark.parametrize("explicit_path", [False, True])
def test_ordinary_python_uses_observed_environment_unless_path_policy_overrides(tmp_path, monkeypatch, explicit_path):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    task = _task(tmp_path / "unfamiliar task")
    base = _environment(tmp_path / "base", runner=False)
    selected = _environment(task / "unfamiliar tools", runner=True)
    original = {"PATH": str(base / "bin") + ":/usr/bin:/bin", "LANG": "C"}
    cfg = make_config(sandbox_bash=False, analysis_task_format="auto",
                      sandbox_env_set={"PATH": original["PATH"]} if explicit_path else {})
    report = discover_runtime(task, cfg, effective_env=original)
    bound = bind_command_environment(cfg, report, original)
    facts = {}
    expected = str(base if explicit_path else selected)
    command = "python -I -c " + shlex.quote(f"import sys; print(sys.prefix == {expected!r})")
    result = dispatch("bash", {"cmd": command}, cwd=str(task), cfg=cfg,
                      effective_env=bound, execution_metadata=facts)
    assert "True" in result
    assert bound["LANG"] == original["LANG"]
    assert original["PATH"].startswith(str(base))
    assert report["facts"][0] is report["command_binding"]
    assert report["command_binding"]["status"] == ("not_applied" if explicit_path else "bound")
    result = dispatch("bash", {"cmd": "python -I -m pytest --version"}, cwd=str(task),
                      cfg=cfg, effective_env=bound)
    assert ("No module named pytest" in result) == explicit_path


@pytest.mark.parametrize("selection", [
    {"status": "ambiguous"}, {"status": "unresolved"},
    {"status": "selected", "selected": {"runner": "pytest", "status": "unavailable"}},
    {"status": "selected", "selected": {"runner": "go", "status": "available"}},
])
def test_unavailable_or_unbound_runners_do_not_change_command_environment(selection):
    original = {"PATH": "/declared/tools"}
    report = {"runner_selection": selection}
    assert bind_command_environment(make_config(), report, original) == original


@pytest.mark.parametrize("executable,path", [
    ("/observed/bin/python", None), ("relative/python", "/bin"),
    ("/path:with/colon/python", "/bin"), (None, "/bin"),
])
def test_missing_path_or_invalid_runtime_does_not_grant_a_command_path(executable, path):
    original = {} if path is None else {"PATH": path}
    report = {"runner_selection": {"status": "selected", "selected": {
        "status": "available", "runner": "pytest", "runtime": {"executable": executable}}}}
    assert bind_command_environment(make_config(), report, original) == original
    assert report["command_binding"]["status"] == "not_applied"


def test_driver_briefs_and_executes_the_same_discovered_python(tmp_path, monkeypatch):
    from scripts.llm_solver.harness.loop import solve_task
    from scripts.llm_solver.server.types import ToolCall, TurnResult, Usage
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    task = _task(tmp_path / "task")
    base = _environment(tmp_path / "base", runner=False)
    selected = _environment(task / "observed runtime", runner=True)
    monkeypatch.setenv("PATH", str(base / "bin") + ":/usr/bin:/bin")
    (task / "prompt.txt").write_text("Check the runtime")
    client = MagicMock()
    calls = []
    startup_system = []
    def chat(*args, **kwargs):
        visible = str(args) + str(kwargs)
        calls.append(visible)
        system = next(message["content"] for message in args[0] if message["role"] == "system")
        if len(calls) == 1:
            startup_system.append(system)
            assert "Runtime executable: " + str(selected / "bin" / "python") in system
            assert "Run tests with:" in system
            assert "command_runtime_binding" not in system
            assert str(selected / "bin") in visible
            command = "python -I -c " + shlex.quote(
                f"import sys, pytest; print('RUNTIME_MATCH=' + str(sys.prefix == {str(selected)!r}))")
            return TurnResult(None, [ToolCall("runtime", "bash", {"cmd": command})],
                              "tool_calls", Usage(100, 10))
        assert system == startup_system[0]
        assert "RUNTIME_MATCH=True" in visible
        return TurnResult("done", [], "stop", Usage(100, 10))
    client.chat.side_effect = chat
    client.build_assistant_message.return_value = {"role": "assistant", "content": "check runtime"}
    solve_task(task, make_config(sandbox_bash=False, analysis_task_format="auto",
        max_sessions=1, max_turns=2, state_writer_enabled=False), client)
    assert len(calls) == 2


def test_multiple_project_environments_remain_visible_and_ambiguous(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    task = _task(tmp_path / "task")
    for name in ("environment one", "environment two"):
        _environment(task / name, runner=True)
    cfg = make_config(sandbox_bash=False, analysis_task_format="auto")
    report = discover_runtime(task, cfg, effective_env={"PATH": "/usr/bin:/bin"})
    assert report["runner_selection"]["status"] == "ambiguous"
    assert len([c for c in report["runner_selection"]["candidates"]
                if c["source"] == "project_environment_marker" and c["status"] == "available"]) == 2
    assert "environment one" in json.dumps(report["facts"])
    assert "environment two" in json.dumps(report["facts"])


def test_blocked_runner_configuration_cannot_select_an_installed_runner(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    task = _task(tmp_path / "task")
    cfg = make_config(sandbox_bash=False, analysis_task_format="auto")
    report = discover_runtime(task, cfg, effective_env={"PATH": str(Path(sys.executable).parent)},
                              unreadable_paths=(str(task / "pyproject.toml"),))
    assert report["runner_declarations"]["candidates"] == []
    resolved = resolve_task_format(cfg, task, runtime_observations=report)
    assert resolved.analysis_task_format == "generic"
    assert resolved.runtime_test_selection["status"] == "no_declared_check"


def test_incompatible_interpreter_is_reported_without_becoming_an_automatic_choice(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    task = _task(tmp_path / "task")
    (task / "pyproject.toml").write_text('[project]\nrequires-python=">=99"\n[tool.pytest.ini_options]\n')
    _environment(task / "environment", runner=True)
    cfg = make_config(sandbox_bash=False, analysis_task_format="auto")
    report = discover_runtime(task, cfg, effective_env={"PATH": "/usr/bin:/bin"})
    assert report["runner_selection"]["status"] == "unresolved"
    candidates = report["runner_selection"]["candidates"]
    assert any(c.get("runtime", {}).get("available") is True
               and c["runtime"]["compatible"] is False for c in candidates)


def test_project_script_uses_its_declared_manager_and_preserves_arguments(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    task = tmp_path / "task"
    task.mkdir()
    (task / "package.json").write_text(json.dumps({"packageManager": "pnpm@9.0.0", "scripts": {"test": "node --test"}}))
    tools = tmp_path / "tools"
    tools.mkdir()
    runner = tools / "pnpm"
    runner.write_text('#!/bin/sh\nif [ "$1" = "--version" ]; then echo fixture; exit; fi\n[ "$1" = run ] && [ "$2" = test ] && [ "$#" = 2 ]\n')
    runner.chmod(0o755)
    cfg = make_config(sandbox_bash=False, analysis_task_format="auto", tools_run_tests_enabled=True)
    env = {"PATH": str(tools) + ":/usr/bin:/bin"}
    report = discover_runtime(task, cfg, effective_env=env)
    assert report["runner_selection"]["status"] == "selected", json.dumps(report, indent=2)
    assert report["runner_selection"]["selected"]["executable"] == str(runner)
    cfg = resolve_task_format(cfg, task, runtime_observations=report)
    facts = {}
    dispatch("run_tests", {}, cwd=str(task), cfg=cfg, effective_env=env, execution_metadata=facts)
    assert facts["verification_status"] == "passed"


def test_actual_node_test_script_runs_without_selecting_jest(tmp_path, monkeypatch):
    import shutil
    import os
    if not shutil.which("node") or not shutil.which("npm"):
        pytest.skip("Node and npm are not installed")
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "node --test check.cjs"}}))
    (tmp_path / "check.cjs").write_text("require('node:test')('arithmetic', () => require('node:assert/strict').equal(2+2,4));\n")
    cfg = make_config(sandbox_bash=False, analysis_task_format="auto", tools_run_tests_enabled=True)
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    report = discover_runtime(tmp_path, cfg, effective_env=env)
    assert report["runner_selection"]["status"] == "selected", json.dumps(report, indent=2)
    cfg = resolve_task_format(cfg, tmp_path, runtime_observations=report)
    facts = {}
    result = dispatch("run_tests", {}, cwd=str(tmp_path), cfg=cfg,
                      effective_env=env, execution_metadata=facts)
    assert cfg.analysis_task_format == "npm"
    assert facts["verification_status"] == "passed", result
    assert "arithmetic" in result


def test_failed_peer_probe_cannot_turn_one_observed_environment_into_a_unique_choice():
    from scripts.llm_solver.harness.runner_runtime import observe_runner_runtime
    declarations = {"candidates": [{"runner": "pytest"}]}
    facts = [{"source": "executable_locations", "available": {}},
             {"source": "manager_inventory", "environment_candidates": ["/env/one", "/env/two"]}]
    def inspect(identifier, command, overrides=None):
        if "candidate_0" in identifier:
            return json.dumps({"prefix": "/env/one", "available": True, "compatible": True})
        return None  # A timeout says nothing about the second environment.
    report = observe_runner_runtime(declarations, facts, requested="auto", inspect=inspect, folders=[])
    assert report["status"] == "unresolved"
    assert "selected" not in report


def test_unknown_environment_does_not_grant_an_unavailable_runner_exception(tmp_path):
    from scripts.llm_solver.harness._guardrails.verification import verification_runner_unavailable
    cfg = make_config(tools_run_tests_enabled=True, analysis_task_format="pytest",
                      runtime_test_selection={"status": "unresolved", "task_root": str(tmp_path)})
    facts = {}
    result = dispatch("run_tests", {}, cwd=str(tmp_path), cfg=cfg, execution_metadata=facts)
    assert facts["verification_status"] == "selection_unresolved"
    assert not verification_runner_unavailable(result, tc_name="run_tests", execution_metadata=facts)


def test_observed_go_command_preserves_requested_package_scope(tmp_path, monkeypatch):
    from scripts.llm_solver.harness.runner_runtime import observe_runner_runtime
    import scripts.llm_solver.harness.tools as tools
    selection = observe_runner_runtime(
        {"candidates": [{"runner": "go"}]},
        [{"source": "executable_locations", "available": {"go": "/tools/go"}}],
        requested="auto", inspect=lambda *args: "go version go1.23 linux/amd64", folders=[],
    )
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        return "ok", 0, False
    monkeypatch.setattr(tools, "_run_in_sandbox", execute)
    cfg = make_config(tools_run_tests_enabled=True, analysis_task_format="go",
                      runtime_test_selection={**selection, "task_root": str(tmp_path)})
    dispatch("run_tests", {"path": "one_package"}, cwd=str(tmp_path), cfg=cfg)
    assert calls == ["/tools/go test ./one_package"]


def test_actual_go_check_targets_only_the_requested_package(tmp_path, monkeypatch):
    import shutil
    if not shutil.which("go"):
        pytest.skip("Go is not installed")
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    (tmp_path / "go.mod").write_text("module example.org/check\n\ngo 1.18\n")
    for name, body in (("selected", ""), ("unrelated", 't.Fatal("must not run")')):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "check_test.go").write_text(
            'package check\nimport "testing"\nfunc TestCheck(t *testing.T) { ' + body + ' }\n'
        )
    cfg = make_config(sandbox_bash=False, analysis_task_format="auto", tools_run_tests_enabled=True)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "GOTOOLCHAIN": "local"}
    report = discover_runtime(tmp_path, cfg, effective_env=env)
    cfg = resolve_task_format(cfg, tmp_path, runtime_observations=report)
    facts = {}
    result = dispatch("run_tests", {"path": "selected"}, cwd=str(tmp_path), cfg=cfg,
                      effective_env=env, execution_metadata=facts)
    assert facts["verification_status"] == "passed", result
    assert "example.org/check/selected" in result
