"""Installed rustup selects fixture toolchains; no download or benchmark jobs."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import tomllib

import pytest

from scripts.llm_solver.harness.sandbox import _build_bwrap_argv, bwrap_preflight
from scripts.llm_solver.harness.sandbox._filesystem import freeze_filesystem_view
from scripts.llm_solver.harness.task_environment import task_environment_scope


@pytest.fixture
def managed_task(tmp_path, monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    ok, reason = bwrap_preflight("/usr/bin/bwrap")
    if not ok:
        pytest.skip(reason)
    manager = shutil.which("rustup")
    if not manager:
        pytest.skip("native rustup is not installed")
    observed_env = {k: os.environ[k] for k in ("HOME", "PATH", "RUSTUP_HOME", "RUSTUP_TOOLCHAIN") if k in os.environ}
    observed_env["RUSTUP_AUTO_INSTALL"] = "0"
    native = subprocess.run([manager, "which", "rustc"], cwd=tmp_path,
                            env=observed_env, capture_output=True, text=True, timeout=10)
    if native.returncode:
        pytest.skip("native rustup has no installed selected compiler")
    toolchain = Path(native.stdout.strip()).parent.parent
    native_home = Path(observed_env.get("RUSTUP_HOME", str(Path(observed_env["HOME"]) / ".rustup")))
    version = tomllib.loads((native_home / "settings.toml").read_text())["version"]
    task = tmp_path / "task"
    task.mkdir()
    home = tmp_path / "home"
    binaries = home / ".cargo" / "bin"
    binaries.mkdir(parents=True)
    for name in ("rustup", "rustc", "cargo"):
        (binaries / name).symlink_to(manager)
    manager_home = home / ".rustup"
    inventory = manager_home / "toolchains"
    inventory.mkdir(parents=True)
    for name in ("fixture-first", "fixture-second"):
        (inventory / name).symlink_to(toolchain, target_is_directory=True)
    settings = manager_home / "settings.toml"
    settings.write_text(f'version={json.dumps(version)}\ndefault_toolchain="fixture-first"\n')
    (home / "private.txt").write_text("PRIVATE HOME FIXTURE")
    (tmp_path / "sibling.txt").write_text("PRIVATE SIBLING FIXTURE")
    (task / "escape").symlink_to(tmp_path / "sibling.txt")
    env = {"HOME": str(home), "PATH": str(binaries) + ":/usr/bin:/bin"}
    return task, home, manager_home, settings, env


def freeze(task, env, **kwargs):
    return freeze_filesystem_view(task, env, deadline=time.monotonic() + 10,
                                  bwrap_bin="/usr/bin/bwrap", **kwargs)


def command(task, env, script):
    argv = _build_bwrap_argv(script, str(task), effective_env=env,
                             bwrap_bin="/usr/bin/bwrap", sandbox_required=True)
    return subprocess.run(argv, pass_fds=argv.pass_fds, capture_output=True, text=True, timeout=20)


def test_startup_query_uses_bound_task_and_refuses_changed_publication(managed_task, monkeypatch):
    from scripts.llm_solver.harness.task_environment import TaskEnvironmentUnavailable

    task, home, manager_home, settings, env = managed_task
    (task / 'rust-toolchain.toml').write_text('[toolchain]\nchannel="fixture-first"\n')
    run = subprocess.run
    observed = []

    def launch(argv, **kwargs):
        # The fixture's installed manager is queried only after the harness
        # has inspected task metadata and prepared the read-only sandbox.
        task.rename(task.parent / 'selected')
        task.mkdir()
        (task / 'rust-toolchain.toml').write_text('[toolchain]\nchannel="fixture-second"\n')
        result = run(argv, **kwargs)
        output = kwargs['stdout']
        output.seek(0)
        observed.append((result.returncode, output.read()))
        return result

    monkeypatch.setattr(subprocess, 'run', launch)
    with pytest.raises(TaskEnvironmentUnavailable, match='task root changed'):
        freeze(task, env)
    assert len(observed) == 1 and observed[0][0] == 0
    assert b'/fixture-first/bin/rustc' in observed[0][1]
    assert b'/fixture-second/' not in observed[0][1]


@pytest.mark.parametrize("source", ["default", "project", "environment", "task_override"])
def test_native_selection_and_compilation_preserve_containment(managed_task, source):
    task, home, manager_home, settings, env = managed_task
    selected = "fixture-first" if source == "default" else "fixture-second"
    if source == "project":
        (task / "rust-toolchain.toml").write_text('[toolchain]\nchannel="fixture-second"\n')
    elif source == "environment":
        env["RUSTUP_TOOLCHAIN"] = "fixture-second"
    elif source == "task_override":
        with settings.open("a") as stream:
            stream.write(f'[overrides]\n{json.dumps(str(task))}="fixture-second"\n')
    (task / "main.rs").write_text('fn main() { println!("42"); }\n')

    @task_environment_scope
    def session():
        view = freeze(task, env)
        record, = view.native_selections
        assert record["status"] == "selected", record
        prefix = manager_home / "toolchains" / selected
        assert dict(view.runtime_bindings) == {"RUSTUP_TOOLCHAIN": str(prefix)}
        result = command(task, env, "rustc main.rs -o check && ./check && cargo --version")
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("42\ncargo ")
        unselected = "fixture-second" if selected == "fixture-first" else "fixture-first"
        private_paths = [home / "private.txt", task.parent / "sibling.txt", task / "escape",
                         manager_home / "toolchains" / unselected, settings]
        result = command(task, env, " && ".join("test ! -e " + shlex.quote(str(p)) for p in private_paths))
        assert result.returncode == 0, result.stderr
        # A later project change cannot grant another installed toolchain.
        (task / "rust-toolchain.toml").write_text('[toolchain]\nchannel="not-installed"\n')
        result = command(task, env, "rustc --version")
        assert result.returncode == 0, result.stderr
    session()


def test_unrelated_ancestor_override_does_not_supply_selection(managed_task):
    task, home, manager_home, settings, env = managed_task
    with settings.open("a") as stream:
        stream.write(f'[overrides]\n{json.dumps(str(task.parent))}="fixture-second"\n')
    view = freeze(task, env)
    assert view.native_selections[0]["status"] == "selected"
    assert dict(view.runtime_bindings)["RUSTUP_TOOLCHAIN"].endswith("/fixture-first")


def test_alternate_installation_and_absolute_selection(managed_task):
    task, home, manager_home, settings, env = managed_task
    relocated = task.parent / "arbitrary sdk location"
    manager_home.rename(relocated)
    env.update(RUSTUP_HOME=str(relocated), RUSTUP_TOOLCHAIN=str(relocated / "toolchains" / "fixture-second"))

    @task_environment_scope
    def session():
        view = freeze(task, env)
        assert dict(view.runtime_bindings)["RUSTUP_TOOLCHAIN"] == env["RUSTUP_TOOLCHAIN"]
        result = command(task, env, "rustc --version")
        assert result.returncode == 0, result.stderr
    session()


def test_startup_observations_and_command_environment_share_native_selection(managed_task):
    from _config_helpers import make_config
    from scripts.llm_solver.harness.runtime_discovery import discover_runtime, bind_command_environment, MAX_PROBES

    task, home, manager_home, settings, env = managed_task
    cfg = make_config(sandbox_bash=True, sandbox_required=True)

    @task_environment_scope
    def session():
        report = discover_runtime(task, cfg, effective_env=env)
        bound = bind_command_environment(cfg, report, env)
        expected = str(manager_home / "toolchains" / "fixture-first")
        assert bound["RUSTUP_TOOLCHAIN"] == expected
        assert report["probes"][0]["id"] == "native_manager:rustup"
        assert len([p for p in report["probes"] if p["status"] not in {"probe_limit", "budget_exhausted"}]) <= MAX_PROBES + 1
        assert any(f.get("source") == "native_toolchain_selection" and f.get("toolchain") == expected
                   for f in report["facts"])
        result = command(task, bound, "rustc --version")
        assert result.returncode == 0, result.stderr
    session()


def test_native_query_has_read_only_task_and_cannot_read_private_paths(managed_task):
    task, home, manager_home, settings, env = managed_task
    source = task / "source.txt"
    source.write_text("UNCHANGED")
    helper = task.parent / "query-helper"
    private = (home / "private.txt", task.parent / "sibling.txt")
    selected = manager_home / "toolchains" / "fixture-first" / "bin" / "rustc"
    helper.write_text("#!/bin/sh\n" +
                      "\n".join("test ! -e " + shlex.quote(str(p)) + " || exit 11" for p in private) +
                      "\nif echo changed > " + shlex.quote(str(source)) + "; then exit 12; fi\n" +
                      "printf '%s\\n' " + shlex.quote(str(selected)) + "\n")
    helper.chmod(0o755)
    for name in ("rustup", "rustc", "cargo"):
        proxy = home / ".cargo" / "bin" / name
        proxy.unlink()
        proxy.symlink_to(helper)
    view = freeze(task, env)
    assert view.native_selections[0]["status"] == "selected", view.native_selections
    assert source.read_text() == "UNCHANGED"


def test_settings_link_outside_manager_is_not_read(managed_task):
    task, home, manager_home, settings, env = managed_task
    external = task.parent / "unrelated-settings.toml"
    settings.rename(external)
    settings.symlink_to(external)
    view = freeze(task, env)
    record, = view.native_selections
    assert record["reason"] == "selection_metadata_outside_manager"
    assert not record["inputs"] and not record.get("executed")
    assert not view.runtime_bindings


def test_unavailable_rust_selection_does_not_block_observed_python_runner(managed_task):
    from _config_helpers import make_config
    from scripts.llm_solver.harness.runtime_discovery import discover_runtime

    task, home, manager_home, settings, env = managed_task
    # Isolate toolchain selection from incomplete directory discovery.
    (task / "escape").rename(task.parent / "unused-escape")
    env["PATH"] = str(Path(sys.executable).parent) + ":" + env["PATH"]
    (task / "pyproject.toml").write_text('[tool.pytest.ini_options]\n')
    (task / "rust-toolchain.toml").write_text('[toolchain]\nchannel="not-installed"\n')

    @task_environment_scope
    def session():
        cfg = make_config(sandbox_bash=True, sandbox_required=True, analysis_task_format="auto")
        report = discover_runtime(task, cfg, effective_env=env)
        assert report["filesystem_view"]["native_selections"][0]["status"] == "unresolved"
        assert report["runner_selection"]["status"] == "selected", json.dumps(report["runner_selection"], indent=2)
        assert report["runner_selection"]["selected"]["runner"] == "pytest"
    session()


@pytest.mark.parametrize("defect", ["missing_toolchain", "conflicting_project", "masked_settings", "expired_budget"])
def test_incomplete_selection_stays_unknown_without_extra_grants(managed_task, defect):
    task, home, manager_home, settings, env = managed_task
    if defect == "missing_toolchain":
        (task / "rust-toolchain.toml").write_text('[toolchain]\nchannel="not-installed"\n')
    elif defect == "conflicting_project":
        (task / "rust-toolchain.toml").write_text('[toolchain]\nchannel="fixture-first"\npath="/missing"\n')
    kwargs = {"unreadable_paths": (str(settings),)} if defect == "masked_settings" else {}
    if defect == "expired_budget":
        view = freeze_filesystem_view(task, env, deadline=time.monotonic() - 1,
                                      bwrap_bin="/usr/bin/bwrap")
    else:
        view = freeze(task, env, **kwargs)
    assert not view.runtime_bindings
    assert view.native_selections[0]["status"] == "unresolved"
    assert not any("native_manager:" in mount.evidence for mount in view.mounts)
