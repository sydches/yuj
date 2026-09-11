"""All ambient execution paths must honor the same network decision."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from scripts.llm_solver.harness._tools import _run_in_sandbox as runner
from scripts.llm_solver.harness._tools.exec_cell import _build_cell_process
from scripts.llm_solver.harness.lsp_support import build_lsp_sandbox_argv
from scripts.llm_solver.harness.process_manager import build_background_sandbox_argv
from scripts.llm_solver.harness.time_budget import (
    BudgetExhausted, command_time_budget, run_time_budget,
)


@pytest.fixture(autouse=True)
def ambient(monkeypatch):
    monkeypatch.setenv("YUJ_CONTAINER", "ambient")
    monkeypatch.delenv("YUJ_AMBIENT_UNSHARE_NET", raising=False)
    monkeypatch.setattr(runner, "_AMBIENT_UNSHARE_PROBED", False)
    monkeypatch.setattr(runner, "_AMBIENT_UNSHARE_AVAILABLE", False)
    monkeypatch.setattr(runner, "_AMBIENT_UNSHARE_PREFIX", ())


def _request(kind, cwd, environment):
    common = dict(cwd=str(cwd), bwrap_bin="unused", effective_env=environment)
    if kind == "shell":
        return runner._run_in_sandbox(
            "echo permitted", sandbox=True, timeout=5, **common,
        )
    if kind == "background":
        return build_background_sandbox_argv("echo permitted", **common)
    if kind == "lsp":
        return build_lsp_sandbox_argv(["echo", "permitted"], **common)
    return _build_cell_process(
        cwd=str(cwd), cfg=SimpleNamespace(sandbox_backend="bwrap"),
        unreadable_paths=(), readable_paths=(), effective_env=environment,
        allow_login_shell=False,
    )[0]


@pytest.mark.parametrize("kind", ["shell", "background", "lsp", "cell"])
@pytest.mark.parametrize("failure", ["nonzero", "missing", "permission"])
def test_failed_boundary_cannot_produce_an_unwrapped_command(
    monkeypatch, tmp_path, kind, failure,
):
    calls = []

    def fail_probe(argv, **kwargs):
        calls.append(argv)
        assert Path(argv[0]).name == "unshare"
        assert argv[1] == "-n"
        assert Path(argv[2]).name == "setpriv"
        assert argv[-1] == "/bin/true"
        if failure == "missing":
            raise FileNotFoundError("unshare")
        if failure == "permission":
            raise PermissionError("unshare")
        return SimpleNamespace(returncode=1, stderr="Operation not permitted")

    monkeypatch.setattr(runner.subprocess, "run", fail_probe)
    with pytest.raises(runner.SandboxUnavailableError, match="network isolation"):
        _request(kind, tmp_path, {"PATH": os.defpath})
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["shell", "background", "lsp", "cell"])
def test_execution_uses_the_absolute_executable_that_was_probed(
    monkeypatch, tmp_path, kind,
):
    calls = []

    def successful(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="permitted", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", successful)
    argv = _request(kind, tmp_path, {"PATH": str(tmp_path / "different-task-bin")})
    probed = calls[0][0]
    assert Path(probed).is_absolute()
    executed = calls[1] if kind == "shell" else argv
    assert Path(calls[0][2]).is_absolute()
    assert executed[:len(calls[0]) - 1] == calls[0][:-1]


@pytest.mark.parametrize("kind", ["shell", "background", "lsp", "cell"])
def test_explicit_outer_boundary_policy_preserves_execution(
    monkeypatch, tmp_path, kind,
):
    monkeypatch.setenv("YUJ_AMBIENT_UNSHARE_NET", "0")
    calls = []

    def run_command(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="permitted", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", run_command)
    argv = _request(kind, tmp_path, {"PATH": os.defpath})
    assert runner.ambient_unshare_net_status() == (False, False)
    if kind == "shell":
        assert len(calls) == 1
        assert calls[0][0] == "bash"
    else:
        assert not calls
        assert "unshare" not in argv[0]


def test_outer_boundary_policy_does_not_cache_a_failed_probe(monkeypatch):
    monkeypatch.setenv("YUJ_AMBIENT_UNSHARE_NET", "0")
    assert runner._probe_ambient_unshare_net() is False
    monkeypatch.delenv("YUJ_AMBIENT_UNSHARE_NET")
    monkeypatch.setattr(
        runner.subprocess, "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stderr=""),
    )
    assert runner._probe_ambient_unshare_net() is True


@pytest.mark.parametrize("missing", ["unshare", "setpriv"])
def test_missing_executable_discovery_refuses_without_spawning(monkeypatch, tmp_path, missing):
    which = shutil.which
    monkeypatch.setattr(runner.shutil, "which", lambda name: None if name == missing else which(name))

    def unexpected_spawn(*args, **kwargs):
        pytest.fail("a missing isolation executable must not launch a process")

    monkeypatch.setattr(runner.subprocess, "run", unexpected_spawn)
    with pytest.raises(runner.SandboxUnavailableError, match="network isolation"):
        _request("shell", tmp_path, {"PATH": os.defpath})


@pytest.mark.parametrize("kind", ["shell", "background", "lsp", "cell"])
@pytest.mark.parametrize("run_limit,call_limit", [(0.05, 1), (1, 0.05)])
def test_slow_probe_obeys_deadline_without_caching_failure(
    monkeypatch, tmp_path, kind, run_limit, call_limit,
):
    slow = tmp_path / "slow_probe"
    slow.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(1)\n")
    slow.chmod(0o755)
    which = shutil.which
    monkeypatch.setattr(runner.shutil, "which", lambda name: str(slow) if name == "unshare" else which(name))
    started = time.monotonic()
    with run_time_budget(run_limit), command_time_budget(call_limit):
        with pytest.raises(BudgetExhausted):
            _request(kind, tmp_path, {"PATH": os.defpath})
    assert time.monotonic() - started < 0.75
    assert runner.ambient_unshare_net_status() == (False, False)
    assert runner._AMBIENT_UNSHARE_PREFIX == ()
    # A later invocation gets a fresh capability observation.
    monkeypatch.setattr(runner.shutil, "which", which)
    monkeypatch.setattr(runner.subprocess, "run",
                        lambda *a, **kw: SimpleNamespace(returncode=0, stderr=""))
    assert runner._probe_ambient_unshare_net() is True


def test_live_namespace_preserves_task_work_and_blocks_parent_network(tmp_path):
    unshare = shutil.which("unshare")
    if not unshare:
        pytest.skip("unshare is unavailable")
    support = subprocess.run(
        [unshare, "-Urn", "/bin/true"], capture_output=True, text=True,
    )
    if support.returncode:
        pytest.skip("unprivileged user/network namespaces are unavailable")
    # Give the fixture parent its own network namespace, owned by the same
    # user namespace as its privileged children: re-entry used to succeed.
    script = tmp_path / "check_boundary.py"
    script.write_text('''
import json, os, shlex, socket, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
from scripts.llm_solver.harness._tools import _run_in_sandbox as runner
from scripts.llm_solver.harness._tools.exec_cell import _build_cell_process
from scripts.llm_solver.harness.lsp_support import build_lsp_sandbox_argv
from scripts.llm_solver.harness.process_manager import build_background_sandbox_argv

task = Path(sys.argv[1])
subprocess.run(['ip', 'link', 'set', 'lo', 'up'], check=True)
parent_namespace = os.readlink('/proc/self/ns/net')
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen()
port = listener.getsockname()[1]
socket.create_connection(("127.0.0.1", port), timeout=1).close()
fake_bin = task / "bin"
fake_bin.mkdir()
for name in ('unshare', 'setpriv'):
    fake = fake_bin / name
    fake.write_text("#!/bin/sh\\nexit 97\\n")
    fake.chmod(0o755)
environment = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ["PATH"])
reentry = ['nsenter', '--net=' + f'/proc/{os.getpid()}/ns/net',
           sys.executable, '-c', f"import socket; socket.create_connection(('127.0.0.1', {port})).close()"]
# Prove the fixture permits re-entry before the harness restricts the child.
subprocess.run(['unshare', '-n', *reentry], check=True)
for kind in ("shell", "background", "lsp", "cell"):
    source = f"""import os, socket, subprocess
from pathlib import Path
assert os.readlink('/proc/self/ns/net') != {parent_namespace!r}
Path({kind!r} + '.txt').write_text('task work')
try:
    socket.create_connection(('127.0.0.1', {port}), timeout=0.5).close()
except OSError:
    print('NETWORK_BLOCKED')
else:
    raise RuntimeError('reached parent network')
reentry = subprocess.run({reentry!r}, capture_output=True, text=True)
assert reentry.returncode != 0, reentry.stdout + reentry.stderr
assert 'Operation not permitted' in reentry.stderr, reentry.stderr
print('REENTRY_BLOCKED')
"""
    command = shlex.join([sys.executable, "-c", source])
    common = dict(cwd=str(task), bwrap_bin="unused", effective_env=environment)
    if kind == "shell":
        out, rc, timed = runner._run_in_sandbox(command, timeout=10, sandbox=True, **common)
        assert not timed
    else:
        data = None
        child_env = None
        if kind == "background":
            argv = build_background_sandbox_argv(command, **common)
        elif kind == "lsp":
            argv = build_lsp_sandbox_argv([sys.executable, "-c", source], **common)
        else:
            argv, _, child_env = _build_cell_process(
                cwd=str(task), cfg=SimpleNamespace(sandbox_backend="bwrap"),
                unreadable_paths=(), readable_paths=(), effective_env=environment,
                allow_login_shell=False)
            data = json.dumps(dict(source=source, timeout=5)) + "\\n"
        result = subprocess.run(argv, cwd=task, env=child_env, input=data,
                                text=True, capture_output=True, timeout=10)
        out, rc = result.stdout + result.stderr, result.returncode
    assert rc == 0, (kind, rc, out)
    assert 'NETWORK_BLOCKED' in out, (kind, out)
    assert 'REENTRY_BLOCKED' in out, (kind, out)
    assert (task / (kind + '.txt')).read_text() == 'task work'
    socket.create_connection(("127.0.0.1", port), timeout=1).close()
    print(kind + ': isolated, task writable, parent network reachable')
listener.close()
''')
    result = subprocess.run(
        [unshare, "-Urn", sys.executable, str(script), str(tmp_path)],
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1])),
        capture_output=True, text=True, timeout=50,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("isolated, task writable, parent network reachable") == 4
