"""Docker/Podman sandbox backend argv construction and provenance.

The backend starts one short-lived container per command.  The task cwd is
bind-mounted read/write at the *same absolute path* and selected unreadable
paths are over-mounted.  No other host path is mounted.  The container root is
read-only, networking is disabled, and the command environment is cleared by
an explicit ``env -i`` entrypoint before bash starts.

This leaf does not select the backend from ``Config`` or emit trace rows.  It
provides validated values for those central seams without importing them.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

from ._unreadable import _expand_unreadable_paths
from .env_policy import build_bash_argv, build_subprocess_env


CONTAINER_RUNTIMES = frozenset({"docker", "podman"})
_IMAGE_DIGEST_RE = re.compile(r"(?:sha256:)?([0-9a-fA-F]{64})")

# Extra flags are intentionally allowlisted.  A denylist would become unsafe
# whenever a runtime added a new mount/network/privilege option.  These flags
# constrain resources or add inert metadata; none changes namespaces, mounts,
# identity, environment, entrypoint, devices, or host connectivity.
_SAFE_VALUE_FLAGS = frozenset({
    "--annotation",
    "--cpu-period",
    "--cpu-quota",
    "--cpu-rt-period",
    "--cpu-rt-runtime",
    "--cpu-shares",
    "--cpus",
    "--cpuset-cpus",
    "--cpuset-mems",
    "--label",
    "--memory",
    "--memory-reservation",
    "--memory-swap",
    "--memory-swappiness",
    "--pids-limit",
    "--platform",
    "--shm-size",
    "--ulimit",
})
_SAFE_BOOLEAN_FLAGS = frozenset({"--init"})


class ContainerBackendError(RuntimeError):
    """Base error for unavailable or unusable container isolation."""


class ContainerConfigurationError(ContainerBackendError, ValueError):
    """A container setting would be invalid or weaken the boundary."""


class ContainerRuntimeUnavailable(ContainerBackendError):
    """The configured Docker/Podman executable is unavailable."""


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContainerConfigurationError(f"{name} must be a boolean")
    return value


def _validate_runtime(runtime: object) -> str:
    if not isinstance(runtime, str) or runtime not in CONTAINER_RUNTIMES:
        allowed = ", ".join(sorted(CONTAINER_RUNTIMES))
        raise ContainerConfigurationError(
            f"sandbox.container_runtime must be one of {allowed}; got {runtime!r}"
        )
    return runtime


def _validate_image(image: object) -> str:
    if (
        not isinstance(image, str)
        or not image
        or image.startswith("-")
        or any(char.isspace() or ord(char) < 32 for char in image)
    ):
        raise ContainerConfigurationError(
            "sandbox.container_image must be a non-empty image reference "
            "without whitespace, control characters, or a leading '-'"
        )
    return image


def normalize_container_flags(
    flags: str | Sequence[str] | None,
) -> tuple[str, ...]:
    """Parse and validate extra runtime flags against the safe allowlist."""
    if flags is None:
        tokens: tuple[str, ...] = ()
    elif isinstance(flags, str):
        try:
            tokens = tuple(shlex.split(flags))
        except ValueError as exc:
            raise ContainerConfigurationError(
                f"sandbox.container_flags could not be parsed: {exc}"
            ) from exc
    elif isinstance(flags, Sequence):
        copied: list[str] = []
        for token in flags:
            if not isinstance(token, str):
                raise ContainerConfigurationError(
                    "sandbox.container_flags entries must be strings"
                )
            copied.append(token)
        tokens = tuple(copied)
    else:
        raise ContainerConfigurationError(
            "sandbox.container_flags must be a string or list of argv tokens"
        )

    validated: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token or "\x00" in token:
            raise ContainerConfigurationError(
                "sandbox.container_flags contains an empty or NUL-bearing token"
            )
        if token in _SAFE_BOOLEAN_FLAGS:
            validated.append(token)
            index += 1
            continue
        if token in _SAFE_VALUE_FLAGS:
            if index + 1 >= len(tokens):
                raise ContainerConfigurationError(
                    f"sandbox.container_flags option {token!r} requires a value"
                )
            value = tokens[index + 1]
            if not value or "\x00" in value or value.startswith("--"):
                raise ContainerConfigurationError(
                    f"sandbox.container_flags option {token!r} has an invalid value"
                )
            validated.extend((token, value))
            index += 2
            continue
        if token.startswith("--") and "=" in token:
            option, value = token.split("=", 1)
            if option in _SAFE_VALUE_FLAGS and value and "\x00" not in value:
                validated.append(token)
                index += 1
                continue
        raise ContainerConfigurationError(
            f"sandbox.container_flags option {token!r} is not permitted by "
            "the fail-closed sandbox flag policy"
        )
    return tuple(validated)


def resolve_container_runtime(
    runtime: str, *, sandbox_required: bool = True,
) -> str | None:
    """Resolve Docker/Podman on PATH, raising when strict isolation requires it."""
    runtime = _validate_runtime(runtime)
    required = _require_bool(sandbox_required, name="sandbox_required")
    executable = shutil.which(runtime)
    if executable is None and required:
        raise ContainerRuntimeUnavailable(
            f"sandbox_required=true but container runtime {runtime!r} is missing "
            "from PATH; refusing to run the command unsandboxed"
        )
    return executable


def _absolute_cwd(cwd: str | os.PathLike[str]) -> Path:
    raw = os.fspath(cwd)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ContainerConfigurationError("sandbox cwd must be a non-empty path")
    path = Path(os.path.abspath(raw))
    if not path.is_dir():
        raise ContainerConfigurationError(f"sandbox cwd is not a directory: {path}")
    # Docker/Podman --mount values use comma-separated key/value syntax.
    if "," in str(path):
        raise ContainerConfigurationError(
            f"sandbox cwd contains ',' and cannot be encoded safely: {path}"
        )
    return path


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _container_mask_argv(
    cwd: Path,
    unreadable_paths: tuple[str, ...],
    *,
    sandbox_required: bool,
) -> list[str]:
    """Translate existing bwrap unreadable-path expansion into OCI mounts."""
    if not unreadable_paths:
        return []
    bwrap_args, _, _ = _expand_unreadable_paths(
        unreadable_paths, sandbox_required=sandbox_required,
    )
    physical_cwd = cwd.resolve()
    files: list[Path] = []
    directories: list[Path] = []

    def _container_target(host_target: Path) -> Path | None:
        """Map a resolved host match back under the lexical cwd mount."""
        try:
            relative = host_target.relative_to(physical_cwd)
        except ValueError:
            return None
        return cwd / relative

    index = 0
    while index < len(bwrap_args):
        operation = bwrap_args[index]
        if operation == "--ro-bind":
            source = bwrap_args[index + 1]
            target = _container_target(Path(bwrap_args[index + 2]))
            if source == "/dev/null" and target is not None:
                files.append(target)
            index += 3
        elif operation == "--tmpfs":
            target = _container_target(Path(bwrap_args[index + 1]))
            if target is not None:
                directories.append(target)
            index += 2
        else:  # Defensive: the imported expansion has a fixed private shape.
            raise ContainerBackendError(
                f"unsupported unreadable-path mask operation {operation!r}"
            )

    kept_directories: list[Path] = []
    for directory in sorted(
        set(directories), key=lambda path: (len(path.parts), str(path)),
    ):
        if directory == cwd:
            raise ContainerConfigurationError(
                "unreadable_paths cannot mask the container task cwd itself"
            )
        if not any(_inside(directory, parent) for parent in kept_directories):
            kept_directories.append(directory)
    kept_files = [
        path for path in sorted(set(files))
        if not any(_inside(path, parent) for parent in kept_directories)
    ]

    argv: list[str] = []
    for path in kept_files:
        if "," in str(path):
            raise ContainerConfigurationError(
                f"unreadable file path contains ',' and cannot be masked: {path}"
            )
        argv.extend((
            "--mount",
            f"type=bind,source=/dev/null,target={path},readonly",
        ))
    for path in kept_directories:
        if "," in str(path):
            raise ContainerConfigurationError(
                f"unreadable directory path contains ',' and cannot be masked: {path}"
            )
        argv.extend((
            "--mount",
            f"type=tmpfs,target={path},readonly,tmpfs-mode=0000",
        ))
    return argv


def _validate_identity(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContainerConfigurationError(f"{name} must be a non-negative integer")
    return value


def _build_container_argv(
    cmd: str,
    cwd: str | os.PathLike[str],
    *,
    runtime: str = "docker",
    image: str,
    container_flags: str | Sequence[str] | None = (),
    effective_env: Mapping[str, str] | None = None,
    unreadable_paths: tuple[str, ...] = (),
    sandbox_required: bool = True,
    allow_login_shell: bool = False,
    runtime_bin: str | None = None,
    uid: int | None = None,
    gid: int | None = None,
) -> list[str]:
    """Build the isolated ``docker/podman run`` argv for one command.

    ``runtime_bin`` is an already-resolved executable supplied by the central
    dispatch.  When omitted, the validated runtime name is used in the argv;
    call :func:`resolve_container_runtime` before execution to enforce
    ``sandbox_required`` without making pure argv tests host-dependent.
    """
    if not isinstance(cmd, str) or "\x00" in cmd:
        raise ContainerConfigurationError("sandbox command must be a NUL-free string")
    runtime_name = _validate_runtime(runtime)
    image_name = _validate_image(image)
    flags = normalize_container_flags(container_flags)
    required = _require_bool(sandbox_required, name="sandbox_required")
    login_shell = _require_bool(
        allow_login_shell, name="sandbox.env.allow_login_shell",
    )
    workdir = _absolute_cwd(cwd)
    user_id = _validate_identity(os.getuid() if uid is None else uid, name="uid")
    group_id = _validate_identity(os.getgid() if gid is None else gid, name="gid")
    executable = runtime_name if runtime_bin is None else runtime_bin
    if not isinstance(executable, str) or not executable or "\x00" in executable:
        raise ContainerConfigurationError("container runtime path is invalid")

    explicit_env = build_subprocess_env(effective_env or {})
    workdir_mount = (
        f"type=bind,source={workdir},target={workdir},"
        "bind-propagation=rprivate"
    )
    argv = [
        executable,
        "run",
        "--rm",
        # Never let a missing local image turn command execution into a host-
        # network pull.  Image acquisition is an explicit operator action.
        "--pull=never",
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        # Docker and Podman create a private PID namespace by default.
        # Docker rejects the tempting explicit spelling `--pid private`.
        "--ipc", "private",
        "--hostname", "yuj-sandbox",
        "--user", f"{user_id}:{group_id}",
        *flags,
        "--workdir", str(workdir),
        "--mount", workdir_mount,
        "--tmpfs", "/tmp:rw,nosuid,nodev",
        # Override image ENTRYPOINT as well as CMD.  The entrypoint clears all
        # image ENV values before it execs bash with the resolved policy.
        "--entrypoint", "/usr/bin/env",
    ]

    git_hooks = workdir / ".git" / "hooks"
    if git_hooks.is_dir():
        argv.extend((
            "--mount",
            f"type=tmpfs,target={git_hooks},tmpfs-mode=0755",
        ))
    argv.extend(_container_mask_argv(
        workdir,
        tuple(unreadable_paths),
        sandbox_required=required,
    ))
    argv.append(image_name)
    argv.extend(("-i", "--"))
    argv.extend(f"{name}={value}" for name, value in explicit_env.items())
    argv.extend(build_bash_argv(
        cmd, allow_login_shell=login_shell, executable="/bin/bash",
    ))
    return argv


def inspect_container_image_digest(
    runtime_bin: str,
    image: str,
    *,
    timeout: int = 15,
) -> str:
    """Return the local immutable image ID as normalized ``sha256:<hex>``."""
    if not isinstance(runtime_bin, str) or not runtime_bin or "\x00" in runtime_bin:
        raise ContainerConfigurationError("container runtime path is invalid")
    image_name = _validate_image(image)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ContainerConfigurationError("image inspect timeout must be positive")
    try:
        result = subprocess.run(
            [runtime_bin, "image", "inspect", "--format={{.Id}}", image_name],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContainerBackendError(
            f"could not inspect container image {image_name!r}: {exc}"
        ) from exc
    if result.returncode != 0:
        first = (result.stderr or "").strip().splitlines()
        reason = first[0][:300] if first else f"exit code {result.returncode}"
        raise ContainerBackendError(
            f"container image {image_name!r} is unavailable: {reason}"
        )
    value = result.stdout.strip().strip('"')
    match = _IMAGE_DIGEST_RE.fullmatch(value)
    if match is None:
        raise ContainerBackendError(
            f"container image inspect returned an invalid digest: {value!r}"
        )
    return "sha256:" + match.group(1).lower()


@dataclass(frozen=True, slots=True)
class ContainerBackend:
    """Validated container backend settings reusable across command calls."""

    runtime: str = "docker"
    image: str = ""
    flags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime", _validate_runtime(self.runtime))
        object.__setattr__(self, "image", _validate_image(self.image))
        object.__setattr__(self, "flags", normalize_container_flags(self.flags))

    def resolve_runtime(self, *, sandbox_required: bool = True) -> str | None:
        return resolve_container_runtime(
            self.runtime, sandbox_required=sandbox_required,
        )

    def build_argv(
        self,
        cmd: str,
        cwd: str | os.PathLike[str],
        **kwargs,
    ) -> list[str]:
        return _build_container_argv(
            cmd,
            cwd,
            runtime=self.runtime,
            image=self.image,
            container_flags=self.flags,
            **kwargs,
        )

    def image_digest(self, runtime_bin: str, *, timeout: int = 15) -> str:
        return inspect_container_image_digest(
            runtime_bin, self.image, timeout=timeout,
        )

    def trace_fields(self, image_digest: str) -> dict[str, str]:
        match = _IMAGE_DIGEST_RE.fullmatch(image_digest)
        if match is None:
            raise ContainerConfigurationError(
                f"container image digest is invalid: {image_digest!r}"
            )
        return {
            "sandbox_backend": "container",
            "container_runtime": self.runtime,
            "container_image_digest": "sha256:" + match.group(1).lower(),
        }
