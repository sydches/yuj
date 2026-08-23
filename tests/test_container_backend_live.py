"""Live first-class container escape checks against a local fixture image.

Set ``YUJ_TEST_CONTAINER_IMAGE`` to an already-present image. The backend uses
``--pull=never``; these tests never acquire image bits or use host networking.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.llm_solver.harness._tools.bash import bash
from scripts.llm_solver.harness._loop._driver_setup import (
    compute_runtime_envelope_fields,
)

from _config_helpers import make_config


IMAGE = os.environ.get("YUJ_TEST_CONTAINER_IMAGE", "")

pytestmark = pytest.mark.skipif(
    not IMAGE,
    reason="set YUJ_TEST_CONTAINER_IMAGE to a pre-provisioned local image",
)


def _run(cwd: Path, command: str, *, unreadable_paths=()) -> str:
    return bash(
        command,
        cwd=str(cwd),
        timeout=60,
        sandbox=True,
        sandbox_required=True,
        sandbox_backend="container",
        container_runtime="docker",
        container_image=IMAGE,
        unreadable_paths=tuple(unreadable_paths),
    )


def test_live_container_preserves_cwd_path_and_allows_task_write(
    tmp_path: Path,
) -> None:
    result = _run(
        tmp_path,
        "pwd; printf container-write > created.txt; cat created.txt",
    )

    # The common output filter renders the identical absolute task cwd as '.'
    # for both bwrap and container results.
    assert result.splitlines()[0] == "."
    assert "container-write" in result
    assert (tmp_path / "created.txt").read_text() == "container-write"


def test_live_container_and_bwrap_return_the_same_task_path(tmp_path: Path) -> None:
    if not Path("/usr/bin/bwrap").is_file():
        pytest.skip("bwrap is not installed")

    bwrap_result = bash(
        "pwd", cwd=str(tmp_path), timeout=20, sandbox=True,
        sandbox_required=True, sandbox_backend="bwrap",
    )
    container_result = _run(tmp_path, "pwd")

    assert bwrap_result == container_result == ".\n"


def test_live_container_startup_preflight_records_local_image_digest(
    tmp_path: Path,
) -> None:
    fields = compute_runtime_envelope_fields(
        make_config(
            sandbox_bash=True,
            sandbox_required=True,
            sandbox_backend="container",
            sandbox_container_runtime="docker",
            sandbox_container_image=IMAGE,
        ),
        tmp_path,
    )

    assert fields["sandbox_engaged"] is True
    assert fields["container_runtime"] == "docker"
    assert fields["container_image_digest"].startswith("sha256:")
    assert len(fields["container_image_digest"]) == 71


def test_live_container_cannot_write_home_parent_or_host_root(
    tmp_path: Path,
) -> None:
    parent_target = tmp_path.parent / f"{tmp_path.name}-parent-escape"
    home_target = Path.home() / f".{tmp_path.name}-home-escape"
    assert not parent_target.exists()
    assert not home_target.exists()

    result = _run(
        tmp_path,
        "set +e; "
        f"touch {parent_target}; parent_rc=$?; "
        f"touch {home_target}; home_rc=$?; "
        "touch /usr/local/lib/yuj-container-escape; root_rc=$?; "
        "printf 'outside=%s,%s,%s\\n' \"$parent_rc\" \"$home_rc\" "
        "\"$root_rc\"; "
        "test \"$parent_rc\" -ne 0; test \"$home_rc\" -ne 0; "
        "test \"$root_rc\" -ne 0",
    )

    assert "outside=" in result
    assert "[exit code:" not in result
    assert not parent_target.exists()
    assert not home_target.exists()


def test_live_container_has_no_network_socket_or_host_home(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "test ! -e /var/run/docker.sock; "
        "test ! -e /home/syd; "
        "if exec 3<>/dev/tcp/1.1.1.1/80; then exit 91; "
        "else echo network-blocked; fi",
    )

    assert "network-blocked" in result
    assert "[exit code:" not in result


def test_live_container_masks_unreadable_file_and_directory(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("DO_NOT_EXPOSE")
    private = tmp_path / "private"
    private.mkdir()
    (private / "answer.txt").write_text("ANSWER_KEY")

    result = _run(
        tmp_path,
        "test ! -s secret.txt; "
        "test -z \"$(ls -A private 2>/dev/null)\"; "
        "echo masks-active",
        unreadable_paths=(str(secret), str(private)),
    )

    assert "masks-active" in result
    assert "DO_NOT_EXPOSE" not in result
    assert "ANSWER_KEY" not in result
    assert "[exit code:" not in result
