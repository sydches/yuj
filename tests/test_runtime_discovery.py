"""Startup assistance must come from permitted observations, including failures."""
import hashlib
import json
import sys
import venv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness import runtime_discovery as discovery
from scripts.llm_solver.harness._loop._driver_setup import load_system_prompt_and_provenance
from scripts.llm_solver.harness._tools._run_in_sandbox import SandboxUnavailableError


def _commands():
    return discovery.tomllib.loads(discovery.package_data_path(
        "scripts.llm_solver.language_quirks", "runtime.toml",
    ).read_text())["discovery"]["commands"]

def _inventory(locations):
    return "".join(f"{name}\0{locations.get(name, '')}\0" for name in _commands())


def test_declarations_and_multiple_installed_environments_keep_their_sources(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.11"\n[tool.uv]\n')
    (tmp_path / "uv.lock").write_text("declared lock")
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "node --test"}}))
    locations = {"conda": "/tools/conda", "uv": "/tools/uv"}
    calls = []

    def inspect(command, **kwargs):
        calls.append((command, kwargs))
        if command.startswith("for name in"):
            return _inventory(locations), 0, False
        if command.startswith("tool_path=/tools/conda"):
            return json.dumps({"envs": ["/envs/one", "/envs/two"]}), 0, False
        if command == "/tools/conda --version":
            return "conda 1.fixture\n", 0, False
        assert command == "/tools/uv --version"
        return "uv 1.fixture\n", 0, False

    monkeypatch.setattr(discovery, "_run_in_sandbox", inspect)
    effective = {"PATH": "/tools"}
    report = discovery.discover_runtime(tmp_path, make_config(), effective_env=effective)
    files = {f["path"]: f for f in report["facts"] if f["source"] == "project_file"}
    assert files["pyproject.toml"]["values"] == {"project.requires-python": ">=3.11", "tool.uv": True}
    assert files["pyproject.toml"]["sha256"] == hashlib.sha256((tmp_path / "pyproject.toml").read_bytes()).hexdigest()
    assert files["package.json"]["values"] == {"scripts.test": "node --test"}
    environments = next(f for f in report["facts"] if f["source"] == "conda_environments")
    assert environments["values"] == {"envs": ["/envs/one", "/envs/two"]}
    assert "suitability are not established" in environments["meaning"]
    assert all(call[1]["effective_env"] is effective for call in calls)
    assert all(call[1]["normalize_addresses"] is False for call in calls)
    assert not any("install" in call[0] or "activate" in call[0] for call in calls)
    assert report["facts_chars"] == len(json.dumps(report["facts"], ensure_ascii=True))


def test_actual_local_probe_observes_tools_in_the_supplied_path(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    tools = tmp_path / "tools with spaces"
    tools.mkdir()
    uv = tools / "uv"
    uv.write_text('#!/bin/sh\n[ "$1" = "--version" ] || exit 9\nprintf "uv fixture-0xdeadbeef123456\\n"\n')
    uv.chmod(0o755)
    report = discovery.discover_runtime(tmp_path, make_config(), effective_env={"PATH": str(tools) + ":/bin"})
    inventory = next(f for f in report["facts"] if f["source"] == "executable_locations")
    assert inventory["available"]["uv"] == str(uv)
    version = next(f for f in report["facts"] if f["source"] == "uv_version")
    assert version["values"] == "uv fixture-0xdeadbeef123456"
    assert version["executable"] == str(uv)


def test_conda_inventory_reads_installed_metadata_without_cleaning_the_registry(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    runtime = tmp_path / "runtime"
    venv.EnvBuilder(with_pip=False).create(runtime)
    package = runtime / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "conda"
    (package / "base").mkdir(parents=True)
    (package / "__init__.py").write_text('__version__="fixture"\n')
    (package / "base" / "__init__.py").write_text("")
    envdir = tmp_path / "environments"
    for name in ("first", "second"):
        (envdir / name / "conda-meta").mkdir(parents=True)
        (envdir / name / "conda-meta" / "history").write_text("fixture")
    (package / "base" / "context.py").write_text(
        f"from types import SimpleNamespace\ncontext=SimpleNamespace(root_prefix={str(runtime)!r}, envs_dirs={[str(envdir)]!r})\n"
    )
    conda = runtime / "bin" / "conda"
    conda.write_text(f'#!{runtime / "bin" / "python"}\nimport sys,conda\nassert sys.argv[1:]==["--version"]\nprint("conda " + conda.__version__)\n')
    conda.chmod(0o755)
    home = tmp_path / "home"
    (home / ".conda").mkdir(parents=True)
    registry = home / ".conda" / "environments.txt"
    registry.write_text("/missing/stale-entry\n" + str(envdir / "first") + "\n")
    before = registry.read_bytes()
    task = tmp_path / "task"
    task.mkdir()
    report = discovery.discover_runtime(task, make_config(), effective_env={"PATH": str(runtime / "bin") + ":/bin", "HOME": str(home)})
    candidates = next(f for f in report["facts"] if f["source"] == "conda_environments")
    assert candidates["status"] == "observed"
    assert candidates["values"]["envs"] == sorted([str(runtime), str(envdir / "first"), str(envdir / "second")])
    assert candidates["values"]["registry"] == str(registry)
    assert registry.read_bytes() == before


def test_blocked_and_escaping_declarations_never_reach_the_briefing(tmp_path, monkeypatch):
    task = tmp_path / "task"
    task.mkdir()
    (tmp_path / "secret").write_text('[project]\nrequires-python="PRIVATE"\n')
    (task / "pyproject.toml").symlink_to(tmp_path / "secret")
    (task / "package.json").write_text('{"packageManager":"PRIVATE"}')
    monkeypatch.setattr(discovery, "_run_in_sandbox", lambda *a, **k: (_inventory({}), 0, False))
    report = discovery.discover_runtime(task, make_config(), effective_env={}, unreadable_paths=(str(task / "package.json"),))
    assert "PRIVATE" not in json.dumps(report)
    files = {f["path"]: f["status"] for f in report["facts"] if f["source"] == "project_file"}
    assert files["package.json"] == "blocked"
    assert files["pyproject.toml"] == "unavailable_or_invalid"


@pytest.mark.parametrize("output,code,timed_out,status", [
    ("not json", 0, False, "invalid_output"),
    ("private error detail", 1, False, "failed"),
    ("", None, True, "failed"),
    ("x" * (discovery.MAX_SOURCE_BYTES + 1), 0, False, "too_large"),
])
def test_probe_failure_and_invalid_output_remain_unknown(tmp_path, monkeypatch, output, code, timed_out, status):
    def inspect(command, **kwargs):
        if command.startswith("for name in"):
            return _inventory({"conda": "/bin/conda"}), 0, False
        return output, code, timed_out
    monkeypatch.setattr(discovery, "_run_in_sandbox", inspect)
    report = discovery.discover_runtime(tmp_path, make_config(), effective_env={})
    fact = next(f for f in report["facts"] if f["source"] == "conda_environments")
    assert fact["status"] == status and "values" not in fact
    assert "private error detail" not in json.dumps(report)


def test_declared_but_uninstalled_tools_are_not_claimed_available(tmp_path, monkeypatch):
    (tmp_path / "uv.lock").write_text("fixture")
    monkeypatch.setattr(discovery, "_run_in_sandbox", lambda *a, **k: (_inventory({}), 0, False))
    report = discovery.discover_runtime(tmp_path, make_config(), effective_env={})
    inventory = next(f for f in report["facts"] if f["source"] == "executable_locations")
    assert "uv" in inventory["not_found"] and inventory["available"] == {}
    assert any(f.get("path") == "uv.lock" and f["status"] == "present" for f in report["facts"])
    assert len(report["probes"]) == 1


def test_source_cap_preserves_status_and_briefing_admission_keeps_full_observations(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("x" * (discovery.MAX_SOURCE_BYTES + 1))
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "x" * 9000}}))
    monkeypatch.setattr(discovery, "_run_in_sandbox", lambda *a, **k: (_inventory({}), 0, False))
    report = discovery.discover_runtime(tmp_path, make_config(), effective_env={})
    assert next(f for f in report["facts"] if f.get("path") == "pyproject.toml")["status"] == "too_large"
    assert report["omitted_facts"] == 0  # Admission waits for the actual request.
    assert report["facts_chars"] > 9000


def test_exhausted_time_does_not_start_a_command(tmp_path, monkeypatch):
    from scripts.llm_solver.harness.time_budget import run_time_budget
    times = iter([0.0])
    monkeypatch.setattr(discovery.time, "monotonic", lambda: next(times, 11.0))
    monkeypatch.setattr(discovery, "_run_in_sandbox", lambda *a, **k: pytest.fail("budget exhausted"))
    with run_time_budget(10):
        report = discovery.discover_runtime(tmp_path, make_config(), effective_env={})
    assert report["facts"][-1] == {"source": "executable_locations", "status": "budget_exhausted"}


def test_large_model_fact_cannot_erase_the_declared_package_manager(tmp_path, monkeypatch):
    (tmp_path / "package.json").write_text(json.dumps({
        "packageManager": "pnpm@9", "scripts": {"test": "x" * 9000},
    }))
    def inspect(command, **kwargs):
        if command.startswith("for name in"):
            return _inventory({"npm": "/bin/npm", "pnpm": "/bin/pnpm"}), 0, False
        return "9.0.0", 0, False
    monkeypatch.setattr(discovery, "_run_in_sandbox", inspect)
    report = discovery.discover_runtime(tmp_path, make_config(analysis_task_format="auto"), effective_env={})
    assert report["omitted_facts"] == 0
    assert report["runner_selection"]["selected"]["executable"] == "/bin/pnpm"
    assert any(f.get("path") == "package.json" for f in report["observations"])


def test_layout_reports_languages_and_folders_without_dependency_or_blocked_contents(tmp_path, monkeypatch):
    for directory in ("src", "tests", "node_modules", "blocked"):
        (tmp_path / directory).mkdir()
    (tmp_path / "src" / "main.go").write_text("package main")
    (tmp_path / "tests" / "check.py").write_text("assert True")
    (tmp_path / "node_modules" / "dependency.js").write_text("hidden by scan scope")
    (tmp_path / "blocked" / "answer.rs").write_text("must not contribute")
    monkeypatch.setattr(discovery, "_run_in_sandbox", lambda *a, **k: (_inventory({}), 0, False))
    report = discovery.discover_runtime(tmp_path, make_config(), effective_env={}, unreadable_paths=(str(tmp_path / "blocked"),))
    layout = report["facts"][0]
    assert layout["folders"] == ["node_modules", "src", "tests"]
    assert layout["source_languages"] == {
        "Go": {"files": 1, "examples": ["src/main.go"]},
        "Python": {"files": 1, "examples": ["tests/check.py"]},
    }


def test_language_version_priority_and_no_download_override_follow_observed_project(tmp_path, monkeypatch):
    (tmp_path / "go.mod").write_text("module fixture")
    calls = []
    def inspect(command, **kwargs):
        calls.append(command)
        if command.startswith("for name in"):
            return _inventory({"go": "/tools/go", "uv": "/tools/uv"}), 0, False
        if command == "/tools/go version":
            assert kwargs["effective_env"]["GOTOOLCHAIN"] == "local"
            return "go version go1.fixture linux/amd64", 0, False
        return "uv fixture", 0, False
    monkeypatch.setattr(discovery, "_run_in_sandbox", inspect)
    report = discovery.discover_runtime(tmp_path, make_config(), effective_env={"GOTOOLCHAIN": "auto"})
    assert calls[1] == "/tools/go version"
    version = next(f for f in report["facts"] if f["source"] == "go_version")
    assert version["environment_overrides"] == {"GOTOOLCHAIN": "local"}


def test_sandbox_failure_is_not_downgraded_to_a_missing_tool(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise SandboxUnavailableError("boundary unavailable")
    monkeypatch.setattr(discovery, "_run_in_sandbox", fail)
    with pytest.raises(SandboxUnavailableError):
        discovery.discover_runtime(tmp_path, make_config(), effective_env={})


def test_prompt_and_provenance_share_exact_observations_and_count_the_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery, "_run_in_sandbox", lambda *a, **k: (_inventory({}), 0, False))
    report = discovery.discover_runtime(tmp_path, make_config(), effective_env={})
    client = SimpleNamespace(profile=SimpleNamespace(preamble=""))
    prompt, provenance, _, metadata = load_system_prompt_and_provenance(
        make_config(analysis_task_format="pytest"), client, tmp_path, None, None, None, None,
        runtime_observations=report,
    )
    prefix, observations = prompt.split("\n\nTask environment (observed at startup):\n")
    facts = json.loads(observations.splitlines()[0])
    assert facts["runtime_observations"] == report["facts"]
    assert "verification_runner" not in facts
    assert provenance["configured_analysis_runner"] == "pytest"
    assert provenance["runtime_discovery"] is report
    assert metadata.task_environment_chars == len(prompt) - len(prefix)


def test_driver_discovers_before_first_model_call_once_for_the_solve(tmp_path, monkeypatch):
    from scripts.llm_solver.harness.loop import solve_task
    from scripts.llm_solver.server.types import TurnResult, Usage
    events = []
    report = {"facts": [{"source": "fixture", "value": "observed-runtime"}], "omitted_facts": 0}
    def discover(*args, **kwargs):
        events.append("discover")
        assert kwargs["effective_env"]["PATH"] == "/usr/bin:/bin"
        return report
    monkeypatch.setattr(discovery, "discover_runtime", discover)
    class Counter:
        last = {"count_basis": "backend_input_tokens", "count_precision": "backend_reported", "prompt_tokens": 10}
        def count(self, messages, tools=None):
            return 10
    monkeypatch.setattr("scripts.llm_solver.harness.request_counting.resolve_counter", lambda *a, **kw: Counter())
    client = MagicMock()
    def chat(*args, **kwargs):
        events.append("model")
        assert "observed-runtime" in str(args) + str(kwargs)
        return TurnResult(content="continue", tool_calls=[], finish_reason="stop", usage=Usage(prompt_tokens=10, completion_tokens=2))
    client.chat.side_effect = chat
    client.build_assistant_message.return_value = {"role": "assistant", "content": "continue"}
    (tmp_path / "prompt.txt").write_text("Work on the task")
    solve_task(tmp_path, make_config(max_sessions=2, max_turns=1, allow_implicit_done=False,
        state_writer_enabled=False, sandbox_env_inherit="none", sandbox_env_set={"PATH": "/usr/bin:/bin"}), client)
    assert events == ["discover", "model", "model"]
