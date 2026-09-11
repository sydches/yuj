"""Real bwrap checks: useful runtime access and task-external containment."""
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tempfile
import venv

import pytest

from scripts.llm_solver.harness.sandbox import _build_bwrap_argv, bwrap_preflight
from scripts.llm_solver.harness.sandbox import _filesystem as filesystem
from scripts.llm_solver.harness.task_environment import task_environment_scope


@pytest.fixture
def task_view(monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    ok, reason = bwrap_preflight("/usr/bin/bwrap")
    if not ok:
        pytest.skip(reason)
    # Outside /tmp: the old /tmp overlay already hid neighbors there.
    with tempfile.TemporaryDirectory(prefix=".audit-filesystem-", dir=Path.home()) as raw:
        root = Path(raw)
        task = root / "task"
        task.mkdir()
        home = root / "private-home"
        home.mkdir()
        yield task, home, {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(home)}


def run(task, env, command, **kwargs):
    argv = _build_bwrap_argv(command, str(task), effective_env=env,
                             sandbox_required=True, **kwargs)
    return subprocess.run(argv, pass_fds=argv.pass_fds,
                          capture_output=True, text=True, timeout=20)


def test_host_files_socket_and_escape_link_absent_but_task_writable(task_view):
    task, home, env = task_view
    private = home / "credentials.txt"
    private.write_text("SYNTHETIC PRIVATE FIXTURE")
    sibling = task.parent / "other-attempt.txt"
    sibling.write_text("SYNTHETIC OTHER ATTEMPT")
    (task / "escape").symlink_to(sibling)
    address = task.parent / "control.sock"
    with socket.socket(socket.AF_UNIX) as control:
        control.bind(str(address))
        # Positive control reproduces the old whole-host read exposure.
        old = subprocess.run(["/usr/bin/bwrap", "--ro-bind", "/", "/",
                              "--tmpfs", "/tmp", "cat", str(private), str(sibling)],
                             capture_output=True, text=True, timeout=10)
        assert old.returncode == 0 and "SYNTHETIC OTHER ATTEMPT" in old.stdout
        script = (
            "from pathlib import Path; import socket; "
            f"paths={list(map(str, (private, sibling, task / 'escape', address)))!r}; "
            "assert not any(Path(p).exists() for p in paths); "
            "assert not Path('/run/docker.sock').exists(); "
            "Path('result.txt').write_text('task write works'); "
            "print('contained')"
        )
        result = run(task, env, "python3 -c " + shlex.quote(script))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "contained"
    assert (task / "result.txt").read_text() == "task write works"
    assert private.read_text() == "SYNTHETIC PRIVATE FIXTURE"


def test_external_environment_runs_tests_without_exposing_parent(task_view):
    task, home, env = task_view
    runtime = task.parent / "arbitrary-runtime-name"
    venv.EnvBuilder(with_pip=False).create(runtime)
    launcher = task.parent / "intermediate-launcher"
    launcher.symlink_to(Path(sys.executable).resolve())
    interpreter = runtime / "bin/python"
    interpreter.unlink()
    interpreter.symlink_to(launcher)
    library = runtime / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    (library / "fixture_dependency.py").write_text("VALUE = 42\n")
    (runtime / "unrelated-private.txt").write_text("PRIVATE")
    (task / "test_task.py").write_text(
        "import unittest, fixture_dependency\n"
        "class Check(unittest.TestCase):\n"
        "    def test_dependency(self):\n"
        "        self.assertEqual(fixture_dependency.VALUE, 42)\n"
    )
    env["PATH"] = str(runtime / "bin") + ":" + env["PATH"]
    result = run(task, env, "python -m unittest -v")
    assert result.returncode == 0 and "OK" in result.stderr, result.stderr
    result = run(task, env, "test ! -e " + shlex.quote(str(runtime / "unrelated-private.txt")))
    assert result.returncode == 0
    view = filesystem.discover_filesystem_view(task, env)
    assert any("startup_PATH:python" in m.evidence and "layout=python-venv" in m.evidence
               for m in view.mounts)


def test_declared_resources_are_read_only_and_child_mask_wins(task_view):
    task, home, env = task_view
    resources = task.parent / "resource"
    resources.mkdir()
    (resources / "public.txt").write_text("ALLOWED")
    secret = resources / "withheld.txt"
    secret.write_text("WITHHELD")
    result = run(task, env, "cat " + shlex.quote(str(resources / "public.txt")) +
                 "; test ! -s " + shlex.quote(str(secret)),
                 readable_paths=(str(resources),), unreadable_paths=(str(secret),))
    assert result.returncode == 0 and result.stdout == "ALLOWED", result.stderr
    result = run(task, env, "echo bad > " + shlex.quote(str(resources / "public.txt")),
                 readable_paths=(str(resources),))
    assert result.returncode != 0 and (resources / "public.txt").read_text() == "ALLOWED"


def test_frozen_view_does_not_follow_later_path_or_task_changes(task_view):
    task, home, env = task_view
    extra = task.parent / "later-runtime"
    venv.EnvBuilder(with_pip=False).create(extra)

    @task_environment_scope
    def session():
        original = filesystem.freeze_filesystem_view(task, env)
        later = {**env, "PATH": str(extra / "bin") + ":" + env["PATH"]}
        (task / "later-env").symlink_to(extra, target_is_directory=True)
        assert filesystem.filesystem_view(task, later) is original
        result = run(task, later, shlex.quote(str(extra / "bin/python")) + " -V")
        assert result.returncode != 0
    session()
    assert filesystem._ACTIVE.get() is None


def test_runtime_component_symlink_cannot_admit_its_outside_target(task_view):
    task, home, env = task_view
    runtime = task.parent / "pretend-runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (runtime / "bin/python").symlink_to("/usr/bin/python3")
    (runtime / "lib").symlink_to(home, target_is_directory=True)
    (home / "private.txt").write_text("PRIVATE")
    env["PATH"] = str(runtime / "bin") + ":" + env["PATH"]
    result = run(task, env, "test ! -e " + shlex.quote(str(runtime / "lib/private.txt")))
    assert result.returncode == 0, result.stderr


def test_changed_resource_link_refuses_execution(task_view):
    task, home, env = task_view
    resource = task.parent / "resource"
    resource.mkdir()
    view = filesystem.discover_filesystem_view(task, env, (str(resource),))
    resource.rmdir()
    resource.symlink_to(home, target_is_directory=True)
    with pytest.raises(RuntimeError, match="changed after startup"):
        filesystem.build_filesystem_argv(view)


def test_native_compiler_and_private_home_work(task_view):
    task, home, env = task_view
    if not Path("/usr/bin/cc").exists():
        pytest.skip("system C compiler unavailable")
    (task / "main.c").write_text('#include <stdio.h>\nint main(void) { puts("42"); return 0; }\n')
    result = run(task, env, 'mkdir -p "$HOME/cache" && cc main.c -o program && ./program')
    assert result.returncode == 0 and result.stdout == "42\n", result.stderr
    assert not (home / "cache").exists()


def test_task_local_copied_interpreter_discovers_its_base(task_view):
    task, home, env = task_view
    runtime = task / "chosen-folder"
    venv.EnvBuilder(with_pip=False, symlinks=False).create(runtime)
    result = run(task, env, shlex.quote(str(runtime / "bin/python")) +
                 " -c 'import sys, json; print(json.dumps(sys.version_info[:2]))'")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"[{sys.version_info.major}, {sys.version_info.minor}]"


def test_masked_metadata_cannot_grant_runtime_components(task_view):
    task, home, env = task_view
    runtime = task / "chosen-folder"
    venv.EnvBuilder(with_pip=False).create(runtime)
    metadata = runtime / "pyvenv.cfg"
    view = filesystem.discover_filesystem_view(task, env, unreadable_paths=(str(metadata),))
    assert not any(str(metadata) in m.evidence for m in view.mounts)
    metadata.unlink()
    external = task.parent / "private-metadata"
    external.write_text(f"home = {Path(sys.base_prefix) / 'bin'}\n")
    metadata.symlink_to(external)
    view = filesystem.discover_filesystem_view(task, env)
    assert not any(str(metadata) in m.evidence for m in view.mounts)


@pytest.mark.parametrize("inventory_limited", [False, True])
def test_startup_discovery_and_run_tests_share_the_live_bwrap_view(task_view, monkeypatch, inventory_limited):
    from _config_helpers import make_config
    from scripts.llm_solver.harness._loop._driver_setup import resolve_task_format
    from scripts.llm_solver.harness.runtime_discovery import discover_runtime
    from scripts.llm_solver.harness.tools import dispatch

    task, home, env = task_view
    (task / "pyproject.toml").write_text('[tool.pytest.ini_options]\n')
    (task / "test_task.py").write_text("def test_check():\n    assert 6 * 7 == 42\n")
    env["PATH"] = str(Path(sys.executable).parent) + ":" + env["PATH"]
    cfg = make_config(sandbox_bash=True, sandbox_required=True,
                      analysis_task_format="auto", tools_run_tests_enabled=True)
    if inventory_limited:
        from dataclasses import replace
        original = filesystem.discover_filesystem_view

        def limited(*args, **kwargs):
            return replace(original(*args, **kwargs), unresolved=("runtime_inventory_limit",))
        monkeypatch.setattr(filesystem, "discover_filesystem_view", limited)

    @task_environment_scope
    def session():
        report = discover_runtime(task, cfg, effective_env=env)
        view = filesystem._ACTIVE.get()
        assert report["filesystem_view"] == view.record()
        if inventory_limited:
            selection = report["runner_selection"]
            assert any(c["status"] == "available" for c in selection["candidates"])
            assert selection["limited"] and selection["status"] == "unresolved"
            assert "selected" not in selection
            return
        assert report["runner_selection"]["status"] == "selected", report
        assert any(f["source"] == "sandbox_filesystem" for f in report["facts"])
        resolved = resolve_task_format(cfg, task, runtime_observations=report)
        facts = {}
        result = dispatch("run_tests", {"path": "test_task.py"}, cwd=str(task),
                          cfg=resolved, effective_env=env, execution_metadata=facts)
        assert facts["verification_status"] == "passed", result
        assert filesystem._ACTIVE.get() is view
    session()


def test_task_runtime_alias_uses_namespace_links_across_task_replacement(task_view):
    task, home, env = task_view
    runtime = task / "runtime"
    venv.EnvBuilder(with_pip=False).create(runtime)
    alias = task.parent / "tool-alias"
    alias.symlink_to(runtime, target_is_directory=True)
    env["PATH"] = str(alias / "bin") + ":" + env["PATH"]

    @task_environment_scope
    def session():
        view = filesystem.freeze_filesystem_view(task, env)
        task_sources = [m for m in view.mounts if Path(m.source).is_relative_to(task)]
        assert task_sources and all(m.kind == "task_alias" for m in task_sources)
        result = run(task, env, "python -c 'import json; print(json.dumps(42))'")
        assert result.returncode == 0 and result.stdout == "42\n", result.stderr
        # Replace the exact source of the frozen alias. Binding this host path
        # again would follow its new target and expose the outside directory.
        private = task.parent / "private-directory"
        private.mkdir()
        (private / "secret").write_text("SYNTHETIC PRIVATE")
        command = "test ! -e " + shlex.quote(str(alias / "bin/secret"))
        argv = _build_bwrap_argv(command, str(task), effective_env=env)
        # Build the prior bind shape before replacement as a positive control.
        # Both commands start after the change, reproducing the check/mount gap.
        from dataclasses import replace
        prior = replace(view, mounts=tuple(replace(m, kind="read_only") for m in view.mounts))
        token = filesystem._ACTIVE.set(prior)
        try:
            prior_argv = _build_bwrap_argv(command, str(task), effective_env=env)
        finally:
            filesystem._ACTIVE.reset(token)
        # Recreate the historical path-based binds explicitly. The current
        # read-only builder now pins sources and would also prevent this leak.
        for index in prior_argv.descriptor_arguments:
            if prior_argv[index - 1] == '--ro-bind-fd':
                source = next(m.source for m in prior.mounts if m.target == prior_argv[index + 1])
                prior_argv[index - 1:index + 1] = ['--ro-bind', source]
        (runtime / "bin").rename(runtime / "saved-bin")
        (runtime / "bin").symlink_to(private, target_is_directory=True)
        prior_result = subprocess.run(prior_argv, pass_fds=prior_argv.pass_fds,
                                      capture_output=True, text=True, timeout=20)
        assert prior_result.returncode == 1 and not prior_result.stderr, prior_result.stderr
        result = subprocess.run(argv, pass_fds=argv.pass_fds,
                                capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        assert filesystem._ACTIVE.get() is view
    session()
