"""Sandbox escape-attempt regression tests.

The model may write the task and private temporary/home storage. Writes must
not reach unrelated host files. System runtime mounts remain read-only.

Skipped when bwrap is not operational on the test host.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.llm_solver.harness.sandbox._preflight import bwrap_preflight
from scripts.llm_solver.harness.tools import bash


BWRAP = "/usr/bin/bwrap"
BWRAP_AVAILABLE, BWRAP_FAILURE = bwrap_preflight(BWRAP)


pytestmark = pytest.mark.skipif(
    not BWRAP_AVAILABLE,
    reason=f"operational bwrap is required: {BWRAP_FAILURE or 'unavailable'}",
)


def _assert_ro_error(result: str, attempt_description: str) -> None:
    """A successful sandbox blocks the write with a Read-only filesystem error."""
    assert "Read-only file system" in result or "read-only" in result.lower(), (
        f"{attempt_description}: expected Read-only rejection, got:\n{result}"
    )


def test_sandbox_home_write_stays_private(tmp_path: Path):
    """A useful home cache write has no effect on the host home."""
    home = tmp_path / "home"
    home.mkdir()
    task = tmp_path / "task"
    task.mkdir()
    result = bash(
        'echo private > "$HOME/marker" && cat "$HOME/marker"',
        cwd=str(task),
        timeout=10,
        effective_env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
    )
    assert "private" in result
    assert not (home / "marker").exists()


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
