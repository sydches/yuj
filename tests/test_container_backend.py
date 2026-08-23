from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.llm_solver.harness.sandbox import _build_bwrap_argv
from scripts.llm_solver.harness.sandbox.container_backend import (
    ContainerBackend,
    ContainerBackendError,
    ContainerConfigurationError,
    ContainerRuntimeUnavailable,
    _build_container_argv,
    inspect_container_image_digest,
    normalize_container_flags,
    resolve_container_runtime,
)
from scripts.llm_solver.harness.sandbox.env_policy import EnvironmentPolicy


IMAGE = "example.invalid/yuj-task@sha256:" + ("a" * 64)


def _value_after(argv: list[str], option: str) -> str:
    index = argv.index(option)
    return argv[index + 1]


def test_container_and_bwrap_use_the_identical_absolute_task_path(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv("YUJ_CONTAINER", raising=False)
    cwd = str(tmp_path)

    bwrap = _build_bwrap_argv("pwd", cwd, "/usr/bin/bwrap")
    container = _build_container_argv(
        "pwd", cwd, image=IMAGE, runtime_bin="/usr/bin/docker", uid=1000, gid=1000,
    )

    assert _value_after(bwrap, "--chdir") == cwd
    assert _value_after(container, "--workdir") == cwd
    assert (
        f"type=bind,source={cwd},target={cwd},bind-propagation=rprivate"
        in container
    )


def test_container_argv_has_fail_closed_isolation_defaults(tmp_path: Path) -> None:
    argv = _build_container_argv(
        "printf ok", tmp_path, image=IMAGE,
        runtime="podman", runtime_bin="/opt/bin/podman", uid=12, gid=34,
    )

    assert argv[:2] == ["/opt/bin/podman", "run"]
    assert "--pull=never" in argv
    assert _value_after(argv, "--network") == "none"
    assert "--read-only" in argv
    assert _value_after(argv, "--cap-drop") == "ALL"
    assert _value_after(argv, "--security-opt") == "no-new-privileges"
    assert "--pid" not in argv  # the runtime default is a private namespace
    assert _value_after(argv, "--ipc") == "private"
    assert _value_after(argv, "--user") == "12:34"
    assert _value_after(argv, "--entrypoint") == "/usr/bin/env"
    assert "/run/docker.sock" not in "\n".join(argv)


def test_command_environment_is_cleared_and_name_sorted(tmp_path: Path) -> None:
    effective = EnvironmentPolicy(
        inherit="none", set={"ZED": "last", "ALPHA": "first"},
    ).resolve({"HOST_TOKEN": "must-not-leak"})

    argv = _build_container_argv(
        "env", tmp_path, image=IMAGE, effective_env=effective,
        runtime_bin="docker", uid=1, gid=2,
    )

    image_index = argv.index(IMAGE)
    assert argv[image_index + 1:image_index + 5] == [
        "-i", "--", "ALPHA=first", "ZED=last",
    ]
    assert argv[image_index + 5:] == [
        "/bin/bash", "--noprofile", "--norc",
        "-o", "pipefail", "-c", "env",
    ]
    assert all("HOST_TOKEN" not in token for token in argv)


def test_login_shell_is_explicitly_opt_in(tmp_path: Path) -> None:
    argv = _build_container_argv(
        "true", tmp_path, image=IMAGE, allow_login_shell=True,
        runtime_bin="docker", uid=1, gid=1,
    )
    assert argv[-6:] == [
        "/bin/bash", "--login", "-o", "pipefail", "-c", "true",
    ]


def test_safe_resource_flags_are_preserved_as_tokens(tmp_path: Path) -> None:
    flags = normalize_container_flags(
        "--memory 1g --cpus=2 --pids-limit 128 --init",
    )
    argv = _build_container_argv(
        "true", tmp_path, image=IMAGE, container_flags=flags,
        runtime_bin="docker", uid=1, gid=1,
    )

    assert flags == (
        "--memory", "1g", "--cpus=2", "--pids-limit", "128", "--init",
    )
    start = argv.index("--memory")
    assert tuple(argv[start:start + len(flags)]) == flags


@pytest.mark.parametrize(
    "flags",
    [
        ("--network", "host"),
        ("--privileged",),
        ("--volume", "/:/host"),
        ("-v", "/:/host"),
        ("--env", "TOKEN=secret"),
        ("--entrypoint", "/bin/sh"),
        ("--security-opt", "seccomp=unconfined"),
        ("--device", "/dev/kvm"),
        ("--unknown-future-flag",),
    ],
)
def test_flags_cannot_weaken_or_bypass_the_boundary(flags) -> None:
    with pytest.raises(ContainerConfigurationError, match="not permitted"):
        normalize_container_flags(flags)


def test_safe_flag_cannot_consume_an_option_as_its_value() -> None:
    with pytest.raises(ContainerConfigurationError, match="invalid value"):
        normalize_container_flags(("--memory", "--privileged"))


def test_unreadable_files_and_directories_become_container_masks(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("secret")
    private = tmp_path / "private"
    private.mkdir()
    (private / "answer.txt").write_text("answer")

    argv = _build_container_argv(
        "true",
        tmp_path,
        image=IMAGE,
        runtime_bin="docker",
        uid=1,
        gid=1,
        unreadable_paths=(str(secret), str(private)),
    )

    assert f"type=bind,source=/dev/null,target={secret},readonly" in argv
    assert f"type=tmpfs,target={private},readonly,tmpfs-mode=0000" in argv
    assert all(str(private / "answer.txt") not in token for token in argv)


def test_masks_map_back_to_a_lexical_symlink_workdir(tmp_path: Path) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    secret = physical / "secret.txt"
    secret.write_text("secret")
    lexical = tmp_path / "task-link"
    lexical.symlink_to(physical, target_is_directory=True)

    argv = _build_container_argv(
        "true",
        lexical,
        image=IMAGE,
        runtime_bin="docker",
        uid=1,
        gid=1,
        unreadable_paths=(str(lexical / "secret.txt"),),
    )

    assert _value_after(argv, "--workdir") == str(lexical)
    assert (
        f"type=bind,source=/dev/null,target={lexical / 'secret.txt'},readonly"
        in argv
    )


def test_git_hooks_are_covered_by_an_ephemeral_mount(tmp_path: Path) -> None:
    hooks = tmp_path / ".git" / "hooks"
    hooks.mkdir(parents=True)

    argv = _build_container_argv(
        "true", tmp_path, image=IMAGE, runtime_bin="docker", uid=1, gid=1,
    )

    assert f"type=tmpfs,target={hooks},tmpfs-mode=0755" in argv


def test_missing_runtime_fails_closed_only_when_required(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.llm_solver.harness.sandbox.container_backend.shutil.which",
        lambda _name: None,
    )

    with pytest.raises(ContainerRuntimeUnavailable, match="refusing"):
        resolve_container_runtime("docker", sandbox_required=True)
    assert resolve_container_runtime("docker", sandbox_required=False) is None


def test_runtime_resolution_accepts_only_docker_or_podman(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.llm_solver.harness.sandbox.container_backend.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    assert resolve_container_runtime("podman") == "/usr/bin/podman"
    with pytest.raises(ContainerConfigurationError, match="container_runtime"):
        resolve_container_runtime("nerdctl")


def test_image_digest_is_inspected_locally_and_normalized(monkeypatch) -> None:
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(
            returncode=0, stdout=("B" * 64) + "\n", stderr="",
        )

    monkeypatch.setattr(
        "scripts.llm_solver.harness.sandbox.container_backend.subprocess.run",
        fake_run,
    )

    digest = inspect_container_image_digest("/usr/bin/docker", "local/image:tag")

    assert digest == "sha256:" + ("b" * 64)
    assert calls[0][0] == [
        "/usr/bin/docker", "image", "inspect",
        "--format={{.Id}}", "local/image:tag",
    ]
    assert calls[0][1]["timeout"] == 15


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(returncode=1, stdout="", stderr="image missing\n"),
        SimpleNamespace(returncode=0, stdout="not-a-digest\n", stderr=""),
    ],
)
def test_image_inspection_failure_is_loud(monkeypatch, result) -> None:
    monkeypatch.setattr(
        "scripts.llm_solver.harness.sandbox.container_backend.subprocess.run",
        lambda *_args, **_kwargs: result,
    )

    with pytest.raises(ContainerBackendError):
        inspect_container_image_digest("docker", "local/image:tag")


def test_backend_supplies_secret_free_session_start_trace_fields() -> None:
    backend = ContainerBackend(
        runtime="docker", image="private.registry.invalid/task:latest",
        flags=("--memory", "2g"),
    )

    assert backend.trace_fields("sha256:" + ("C" * 64)) == {
        "sandbox_backend": "container",
        "container_runtime": "docker",
        "container_image_digest": "sha256:" + ("c" * 64),
    }


def test_invalid_image_and_cwd_fail_before_runtime_execution(tmp_path: Path) -> None:
    with pytest.raises(ContainerConfigurationError, match="container_image"):
        _build_container_argv("true", tmp_path, image="--privileged")
    with pytest.raises(ContainerConfigurationError, match="not a directory"):
        _build_container_argv("true", tmp_path / "missing", image=IMAGE)
