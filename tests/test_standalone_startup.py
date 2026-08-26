import json
from pathlib import Path

import pytest

from scripts.llm_assist import __main__ as cli
from scripts.llm_assist import store
from scripts.llm_assist.startup import preflight_assistant_startup
from scripts.llm_solver import runtime_resources as runtime_resource_module
from scripts.llm_solver._shared import paths
from scripts.llm_solver._shared.paths import package_data_path, resource_origin
from scripts.llm_solver.runtime_resources import validate_runtime_resources


def _write_no_sandbox_overlay(path: Path) -> None:
    path.write_text(
        "[tools]\n"
        "sandbox_bash = false\n"
        "sandbox_required = false\n"
    )


def test_source_resource_contract_and_package_owned_data_load():
    report = validate_runtime_resources()
    assert report.origin == "source-checkout"
    assert report.root_resource_count == 57
    assert report.package_resource_count == 10
    assert resource_origin() == "source-checkout"
    assert package_data_path(
        "scripts.llm_solver.language_quirks", "pytest.toml"
    ).is_file()


def test_installed_local_config_uses_xdg_without_creating_it(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONFIG", raising=False)
    monkeypatch.delenv("YUJ_CONFIG_LOCAL", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(paths, "_find_source_root", lambda start=None: None)
    expected = tmp_path / "xdg" / "yuj" / "config.local.toml"
    assert paths.local_config_path() == expected.resolve()
    assert not expected.exists()


def test_local_config_environment_override_is_exact(tmp_path, monkeypatch):
    configured = tmp_path / "machine.toml"
    monkeypatch.setenv("YUJ_CONFIG_LOCAL", str(configured))
    assert paths.local_config_path() == configured.resolve()


def test_installed_session_state_uses_xdg_state_home(tmp_path, monkeypatch):
    monkeypatch.delenv("HARNESS_ASSIST_HOME", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(store, "resource_origin", lambda: "installed-package")
    expected = tmp_path / "state" / "yuj"
    assert store.assist_home() == expected.resolve()
    assert not expected.exists()


def test_installed_resource_validation_ignores_only_interpreter_caches(
    tmp_path, monkeypatch
):
    root = tmp_path / "resources"
    root.mkdir()
    (root / "config.toml").write_text("")
    cache = root / "profiles" / "_base" / "__pycache__" / "behavioral.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"interpreter cache")
    monkeypatch.setattr(
        runtime_resource_module, "ROOT_RUNTIME_FILES", ("config.toml",)
    )
    monkeypatch.setattr(runtime_resource_module, "PACKAGE_RUNTIME_FILES", ())
    monkeypatch.setattr(runtime_resource_module, "project_root", lambda: root)
    monkeypatch.setattr(
        runtime_resource_module, "resource_origin", lambda: "installed-package"
    )

    report = runtime_resource_module.validate_runtime_resources()
    assert report.root_resource_count == 1

    (root / "undeclared.txt").write_text("not cache data")
    with pytest.raises(RuntimeError, match="undeclared.txt"):
        runtime_resource_module.validate_runtime_resources()


def test_config_json_reports_resource_contract_without_absolute_root(capsys):
    assert cli.main(["config", "--json", "--agent", "research"]) == 0
    payload = json.loads(capsys.readouterr().out)
    resources = payload["references"]["runtime_resources"]
    assert resources == {
        "origin": "source-checkout",
        "package_resource_count": 10,
        "root": "<yuj-root>",
        "root_resource_count": 57,
    }


def test_code_dry_run_stops_before_network_and_creates_no_session(
    tmp_path, monkeypatch, capsys
):
    task = tmp_path / "task"
    task.mkdir()
    (task / "pyproject.toml").write_text("[project]\nname = 'smoke'\nversion = '0'\n")
    overlay = tmp_path / "no-sandbox.toml"
    _write_no_sandbox_overlay(overlay)
    assist_home = tmp_path / "assist-home"
    monkeypatch.setenv("HARNESS_ASSIST_HOME", str(assist_home))

    def network_boundary(*args, **kwargs):
        raise AssertionError("model discovery must not run during --dry-run")

    monkeypatch.setattr(cli, "_resolve_model_or_exit", network_boundary)
    result = cli.main([
        "code",
        "--dry-run",
        "--cwd",
        str(task),
        "--config",
        str(overlay),
        "verify standalone startup",
    ])
    output = capsys.readouterr().out
    assert result == 0
    assert "Yuj startup preflight: ready" in output
    assert "Runtime resources: source-checkout (57 root, 10 package)" in output
    assert "Runner: pytest" in output
    assert "Model network: not contacted" in output
    assert not assist_home.exists()


def test_preflight_accepts_disabled_optional_bash_layers(tmp_path):
    task = tmp_path / "task"
    task.mkdir()

    report = preflight_assistant_startup(
        config_paths=(),
        cwd=task,
        context_mode="full",
        config_overrides={
            "sandbox_bash": False,
            "sandbox_required": False,
            "bash_transforms_universal_enabled": False,
            "bash_quirks_forbidden_enabled": False,
        },
    )

    assert report.network_contacted is False
    assert report.detected_runner == "pytest"


def test_normal_code_runs_local_preflight_before_model_discovery(
    tmp_path, monkeypatch
):
    task = tmp_path / "task"
    task.mkdir()
    events = []

    def local_preflight(**kwargs):
        events.append("local-preflight")
        return object()

    def model_discovery(*args, **kwargs):
        events.append("model-discovery")
        raise SystemExit("stop at model boundary")

    monkeypatch.setattr(cli, "preflight_assistant_startup", local_preflight)
    monkeypatch.setattr(cli, "_resolve_model_or_exit", model_discovery)
    monkeypatch.setattr(cli, "_maybe_offer_first_run_setup", lambda args: None)
    with pytest.raises(SystemExit, match="stop at model boundary"):
        cli.main(["code", "--cwd", str(task), "verify ordering"])
    assert events == ["local-preflight", "model-discovery"]
