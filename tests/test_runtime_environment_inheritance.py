"""Observe one permitted SDK environment and carry it to facts and commands."""
import json
from types import SimpleNamespace

import pytest

from _config_helpers import make_config
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness import runtime_discovery
from scripts.llm_solver.harness.sandbox import env_policy
from scripts.llm_solver.harness.tools import _effective_command_environment, _run_in_sandbox


def payload(values):
    return b"".join(name.encode() + b"=" + value.encode() + b"\0" for name, value in values.items())


@pytest.mark.parametrize("selected", ["prepared-one", "prepared-two"])
def test_selected_container_values_reach_the_actual_command_environment(tmp_path, monkeypatch, selected, bound_container_environment):
    cache = tmp_path / selected / "cache"
    cache.mkdir(parents=True)
    source = {"PATH": "/usr/bin:/bin", "HOME": str(cache.parent), "GOCACHE": str(cache),
              "GOPATH": "/installed/" + selected, "GOTOOLCHAIN": "go1.fixture+path",
              "SETUP": '{"credential":"do-not-forward"}', "PYTEST_ADDOPTS": "--reruns=3"}
    monkeypatch.setenv("YUJ_CONTAINER", selected)
    monkeypatch.setenv("GOPATH", "/wrong-host-sdk")
    cfg = make_config(sandbox_bash=True, sandbox_env_inherit="runtime")
    seen = []

    def inspect(argv, **kwargs):
        seen.append(argv)
        assert argv[:5] == ['docker', 'exec', '--workdir', '/selected/task', 'a' * 64]
        assert selected not in argv
        return SimpleNamespace(returncode=0, stdout=payload(source))

    with monkeypatch.context() as boundary:
        boundary.setattr(bound_container_environment, "_execute", inspect)
        effective, login = _effective_command_environment(cfg, cwd=tmp_path)
    # Use a real, small local child to check delivery and cache writes. The
    # selected-container observation above is mocked; no Docker operation runs.
    output, code, timed = _run_in_sandbox(
        '/usr/bin/touch "$GOCACHE/probe" && /usr/bin/env', cwd=str(tmp_path),
        sandbox=False, bwrap_bin=cfg.bwrap_bin, timeout=5, effective_env=effective, allow_login_shell=login,
        normalize_output=False,
    )
    assert code == 0 and not timed
    actual = dict(line.split("=", 1) for line in output.splitlines())
    assert actual["GOPATH"] == source["GOPATH"]
    assert actual["GOTOOLCHAIN"] == source["GOTOOLCHAIN"]
    assert actual["HOME"] == source["HOME"]
    assert (cache / "probe").exists()
    assert "SETUP" not in actual and "PYTEST_ADDOPTS" not in actual
    assert len(seen) == 1


def test_failed_selected_environment_never_falls_back_to_host(monkeypatch):
    monkeypatch.setenv("YUJ_CONTAINER", "missing-container")
    monkeypatch.setenv("GOPATH", "/host-must-not-substitute")
    monkeypatch.setattr(env_policy.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("fixture")))
    with pytest.raises(env_policy.EnvironmentPolicyError, match="cannot inspect"):
        _effective_command_environment(make_config(sandbox_bash=True, sandbox_env_inherit="runtime"))


def test_real_config_loader_accepts_runtime_policy_without_supplying_sdk_values(tmp_path):
    path = tmp_path / "runtime.toml"
    path.write_text('[sandbox.env]\ninherit="runtime"\n')
    cfg = load_config(user_config=[path])
    assert cfg.sandbox_env_inherit == "runtime"
    assert "GOPATH" not in cfg.sandbox_env_set
    assert "GOTOOLCHAIN" not in cfg.sandbox_env_set


def test_runtime_facts_distinguish_inherited_values_policy_and_read_only_probe_override(tmp_path, monkeypatch):
    (tmp_path / "go.mod").write_text("module example.invalid/task\n")
    (tmp_path / "main.go").write_text("package main\n")
    effective = env_policy.EnvironmentPolicy(inherit="runtime", set={"GOCACHE": "/owned/cache"}).resolve(
        {"PATH": "/observed/bin", "GOPATH": "/observed/modules", "GOTOOLCHAIN": "go1.fixture+auto",
         "GOCACHE": "/ignored/cache", "SETUP": "private", "PYTEST_ADDOPTS": "--reruns=3"})
    cfg = make_config(sandbox_env_inherit="runtime", sandbox_env_set={"GOCACHE": "/owned/cache"})
    calls = []

    def inspect(command, **kwargs):
        calls.append((command, dict(kwargs["effective_env"])))
        if command.startswith("for name in"):
            spec = runtime_discovery.tomllib.loads(runtime_discovery.package_data_path(
                "scripts.llm_solver.language_quirks", "runtime.toml").read_text())
            return "".join(name + "\0" + ("/observed/bin/go" if name == "go" else "") + "\0"
                           for name in spec["discovery"]["commands"]), 0, False
        assert command == "/observed/bin/go version"
        assert kwargs["effective_env"]["GOTOOLCHAIN"] == "local"
        assert kwargs["effective_env"]["GOPATH"] == "/observed/modules"
        return "go version go1.local linux/amd64", 0, False

    monkeypatch.setattr(runtime_discovery, "_run_in_sandbox", inspect)
    report = runtime_discovery.discover_runtime(tmp_path, cfg, effective_env=effective)
    facts = {fact["name"]: fact for fact in report["facts"] if fact["source"] == "command_runtime_setting"}
    assert facts["GOPATH"]["value"] == effective["GOPATH"]
    assert facts["GOPATH"]["origin"] == "inherited_environment"
    assert facts["GOCACHE"]["origin"] == "explicit_policy"
    assert facts["GOCACHE"]["value"] == "/owned/cache"
    assert facts["GOTOOLCHAIN"]["value"] == "go1.fixture+auto"
    assert "GOMODCACHE" not in facts
    assert "private" not in json.dumps(report)
    assert effective["GOTOOLCHAIN"] == "go1.fixture+auto", "probe overrides must not change tool commands"
    probe = next(fact for fact in report["facts"] if fact["source"] == "go_version")
    assert probe["environment_overrides"] == {"GOTOOLCHAIN": "local"}
    assert calls[0][1] == effective
