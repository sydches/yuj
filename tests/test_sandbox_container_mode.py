"""Tests for the YUJ_CONTAINER dispatch in `_run_in_sandbox`.

The precise failure mode we're regression-testing:

  - The harness runs inside a container that does NOT have bwrap installed.
  - sandbox_required=True (production setting in config.toml).
  - Before the fix: bash hard-errored with "sandbox_required=true but
    bwrap binary missing" because the dispatch checked bwrap-existence
    BEFORE the YUJ_CONTAINER container-mode branch.
  - After the fix: YUJ_CONTAINER=ambient bypasses the bwrap-existence
    check entirely; bash runs as plain subprocess.run.

These tests pin the dispatch order so a future refactor can't silently
re-introduce the bug.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.llm_solver.harness.sandbox import (
    AMBIENT_CONTAINER,
    _build_bwrap_argv,
    container_mode,
)
from scripts.llm_solver.harness._tools._run_in_sandbox import (
    SandboxUnavailableError,
    _run_in_sandbox,
)


# ----- container_mode() resolver -----

def test_container_mode_unset(monkeypatch):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    assert container_mode() is None


def test_container_mode_empty(monkeypatch):
    monkeypatch.setenv("YUJ_CONTAINER", "")
    assert container_mode() is None


def test_container_mode_ambient(monkeypatch):
    monkeypatch.setenv("YUJ_CONTAINER", "ambient")
    assert container_mode() == AMBIENT_CONTAINER == "ambient"


def test_container_mode_container_id(monkeypatch):
    monkeypatch.setenv("YUJ_CONTAINER", "yuj-task-abc-123")
    assert container_mode() == "yuj-task-abc-123"


# ----- _build_bwrap_argv mode dispatch -----

def test_build_bwrap_argv_rejects_ambient(monkeypatch, tmp_path):
    """ambient mode should be routed away from _build_bwrap_argv;
    if it isn't, fail loudly rather than silently fall through."""
    monkeypatch.setenv("YUJ_CONTAINER", "ambient")
    with pytest.raises(RuntimeError, match="ambient container mode"):
        _build_bwrap_argv("echo hi", str(tmp_path), "/usr/bin/bwrap")


def test_build_bwrap_argv_docker_exec(monkeypatch, tmp_path):
    monkeypatch.setenv("YUJ_CONTAINER", "yuj-task-xyz")
    argv = _build_bwrap_argv("echo hi", str(tmp_path), "/usr/bin/bwrap")
    assert argv[:3] == ["docker", "exec", "--workdir"]
    assert "yuj-task-xyz" in argv
    assert "/usr/bin/env" in argv
    assert "-i" in argv
    assert argv[-7:] == [
        "bash", "--noprofile", "--norc", "-o", "pipefail", "-c", "echo hi",
    ]


def test_bash_trivial_read_uses_container_path(monkeypatch, tmp_path):
    """cat/head must not be served by host-side Python I/O in container mode."""
    from scripts.llm_solver.harness import tools as tools_mod
    from scripts.llm_solver.harness._tools.bash import bash

    (tmp_path / "hello.txt").write_text("host-side file\n")
    calls = []

    def fake_run_in_sandbox(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return "container-side file\n", 0, False

    monkeypatch.setenv("YUJ_CONTAINER", "yuj-task-xyz")
    monkeypatch.setattr(tools_mod, "_run_in_sandbox", fake_run_in_sandbox)

    out = bash("cat hello.txt", cwd=str(tmp_path), timeout=10, sandbox=True)

    assert calls and calls[0][0] == "cat hello.txt"
    assert out == "container-side file\n"


def test_build_bwrap_argv_legacy_bwrap(monkeypatch, tmp_path):
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    argv = _build_bwrap_argv("echo hi", str(tmp_path), "/usr/bin/bwrap")
    assert argv[0] == "/usr/bin/bwrap"
    assert "--ro-bind" in argv
    assert str(tmp_path) in argv  # cwd binding


# ----- _run_in_sandbox dispatch — the actual fix -----

# The exact failure condition: bwrap missing, sandbox_required=True,
# but YUJ_CONTAINER=ambient set. Before the fix this raised
# RuntimeError. After the fix this runs successfully via plain
# subprocess.run.

NONEXISTENT_BWRAP = "/this/path/does/not/exist/bwrap"


def test_missing_legacy_container_raises_typed_unavailable(monkeypatch, tmp_path):
    """A vanished docker-exec target is infrastructure, not tool output."""
    runner_module = importlib.import_module(
        "scripts.llm_solver.harness._tools._run_in_sandbox"
    )
    calls = []
    responses = iter([
        SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Error response from daemon: No such container: gone\n",
        ),
        SimpleNamespace(returncode=1, stdout="", stderr="not found\n"),
    ])

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return next(responses)

    monkeypatch.setenv("YUJ_CONTAINER", "gone")
    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)

    with pytest.raises(SandboxUnavailableError, match="sandbox is unavailable"):
        _run_in_sandbox(
            "echo test", cwd=str(tmp_path), timeout=10,
            sandbox=True, bwrap_bin=NONEXISTENT_BWRAP,
            sandbox_required=True,
        )

    assert calls[0][0][:2] == ["docker", "exec"]
    assert calls[1][0] == [
        "docker", "inspect", "--format", "{{.State.Running}}", "gone",
    ]


def test_live_legacy_container_preserves_normal_nonzero(monkeypatch, tmp_path):
    """A command failure inside a live sandbox still belongs to the model."""
    runner_module = importlib.import_module(
        "scripts.llm_solver.harness._tools._run_in_sandbox"
    )
    responses = iter([
        SimpleNamespace(returncode=7, stdout="command failed\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="true\n", stderr=""),
    ])

    monkeypatch.setenv("YUJ_CONTAINER", "still-running")
    monkeypatch.setattr(
        runner_module.subprocess, "run", lambda *_args, **_kwargs: next(responses)
    )

    assert _run_in_sandbox(
        "false", cwd=str(tmp_path), timeout=10,
        sandbox=True, bwrap_bin=NONEXISTENT_BWRAP,
        sandbox_required=True,
    ) == ("command failed\n", 7, False)


def test_ambient_mode_bypasses_missing_bwrap(monkeypatch, tmp_path):
    """The bug we just fixed: in ambient mode, missing bwrap binary
    should NOT raise even with sandbox_required=True."""
    monkeypatch.setenv("YUJ_CONTAINER", "ambient")
    out, rc, timed_out = _run_in_sandbox(
        "echo hello-from-ambient", cwd=str(tmp_path), timeout=10,
        sandbox=True, bwrap_bin=NONEXISTENT_BWRAP, sandbox_required=True,
    )
    assert rc == 0, f"expected exit 0, got rc={rc}, out={out!r}"
    assert "hello-from-ambient" in out
    assert not timed_out


def test_ambient_mode_can_write_to_cwd(monkeypatch, tmp_path):
    """Ambient mode runs in cwd directly — verify shell semantics work."""
    monkeypatch.setenv("YUJ_CONTAINER", "ambient")
    out, rc, _ = _run_in_sandbox(
        "echo content > test.txt && cat test.txt",
        cwd=str(tmp_path), timeout=10,
        sandbox=True, bwrap_bin=NONEXISTENT_BWRAP, sandbox_required=True,
    )
    assert rc == 0
    assert "content" in out
    assert (tmp_path / "test.txt").exists()


def test_legacy_bwrap_mode_still_hard_fails_when_missing(monkeypatch, tmp_path):
    """Regression check: when YUJ_CONTAINER is unset, the
    sandbox_required hard-fail behavior MUST be preserved."""
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    out, rc, timed_out = _run_in_sandbox(
        "echo whatever", cwd=str(tmp_path), timeout=10,
        sandbox=True, bwrap_bin=NONEXISTENT_BWRAP, sandbox_required=True,
    )
    # The implementation catches RuntimeError and returns it as
    # "ERROR: ..." text rather than re-raising — matching pre-fix behavior.
    assert "sandbox_required" in out or "missing" in out.lower()
    assert rc is None  # exception path returns None for exit code


def test_selected_bwrap_never_degrades_without_required(monkeypatch, tmp_path):
    """A selected backend cannot turn into implicit host execution."""
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    out, rc, _ = _run_in_sandbox(
        "echo must-not-run", cwd=str(tmp_path), timeout=10,
        sandbox=True, bwrap_bin=NONEXISTENT_BWRAP, sandbox_required=False,
    )
    assert rc is None
    assert "Refusing to substitute another backend or run unsandboxed" in out


# ----- File ops bypass the sandbox dispatch entirely -----
# This is documented behavior: write/edit/read tools use path-resolution,
# not bwrap. Pinning it so a future refactor doesn't accidentally route
# them through sandbox dispatch.

def test_file_ops_unaffected_by_container_mode(monkeypatch, tmp_path):
    """Verify the critical property: file tools are not dependent on
    the sandbox dispatch and work regardless of container mode."""
    from scripts.llm_solver.harness._tools.write import write
    monkeypatch.setenv("YUJ_CONTAINER", "ambient")
    # write() takes (path, content, cwd, ...) — operates on path, not via bwrap
    result = write("hello.py", "print('hi')\n", cwd=str(tmp_path))
    assert (tmp_path / "hello.py").exists()
    assert "print('hi')" in (tmp_path / "hello.py").read_text()
