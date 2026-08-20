"""Sandbox escape-attempt regression tests.

docs/sandbox.md declares the bwrap mount namespace load-bearing for
the experimental setup: the model cannot write outside the task's
cwd. That guarantee is enforced by argv construction in
``harness/sandbox.py::_build_bwrap_argv``. A regression here silently
breaks the security boundary.

docs/sandbox.md has a manual verification checklist — three ``touch``
commands that must return "Read-only file system". This test
automates the same checklist.

Skipped when bwrap is not installed (CI without bwrap support).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from scripts.llm_solver.harness.tools import bash


BWRAP = "/usr/bin/bwrap"


pytestmark = pytest.mark.skipif(
    not Path(BWRAP).is_file(),
    reason="bwrap not installed — sandbox escape tests require /usr/bin/bwrap",
)


def _assert_ro_error(result: str, attempt_description: str) -> None:
    """A successful sandbox blocks the write with a Read-only filesystem error."""
    assert "Read-only file system" in result or "read-only" in result.lower(), (
        f"{attempt_description}: expected Read-only rejection, got:\n{result}"
    )


def test_sandbox_blocks_home_write(tmp_path: Path):
    """Writing to the user's home directory is blocked by bwrap."""
    home = os.path.expanduser("~")
    target = f"{home}/.yuj_escape_test_marker"
    # Sanity: if a prior failed test dropped a marker, clean it here via host.
    if os.path.exists(target):
        os.remove(target)
    result = bash(
        f"touch {target} 2>&1",
        cwd=str(tmp_path),
        timeout=10,
    )
    _assert_ro_error(result, f"touch {target}")
    assert not os.path.exists(target), (
        f"ESCAPE: sandbox allowed write to {target} — bwrap argv construction regressed"
    )


def test_sandbox_blocks_parent_dir_write(tmp_path: Path):
    """Writing via relative path traversal (../../..) is blocked."""
    result = bash(
        "touch ../../../../../escape_test_relative 2>&1",
        cwd=str(tmp_path),
        timeout=10,
    )
    _assert_ro_error(result, "touch ../../../../../escape_test_relative")


def test_sandbox_blocks_absolute_prepared_write(tmp_path: Path):
    """Writing to an absolute path outside cwd is blocked."""
    # Use a guaranteed-outside-cwd absolute path; /tmp/yuj_escape is on
    # the tmpfs mount inside the sandbox, which is a FRESH tmpfs per
    # call — so we pick a path on the host filesystem instead to
    # guarantee the write is to the real (read-only-bound) filesystem.
    result = bash(
        "touch /usr/local/lib/yuj_escape_test 2>&1",
        cwd=str(tmp_path),
        timeout=10,
    )
    _assert_ro_error(result, "touch /usr/local/lib/yuj_escape_test")


def test_sandbox_allows_cwd_write(tmp_path: Path):
    """A legitimate write inside cwd succeeds — sanity check the sandbox didn't block everything."""
    result = bash(
        "echo hello > inside.txt && cat inside.txt",
        cwd=str(tmp_path),
        timeout=10,
    )
    assert "hello" in result, f"cwd write unexpectedly failed: {result}"
    assert (tmp_path / "inside.txt").is_file(), "inside.txt not created in cwd"
