"""Discover the task's host/container path mapping without benchmark literals."""
from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import wraps
import json
import hashlib
import os
from pathlib import Path, PurePosixPath
import subprocess
from .docker_identity import observe_docker_identity
from .process_identity import ProcessIdentity, ProcessIdentityError, observe_container_process


class TaskEnvironmentUnavailable(RuntimeError):
    """The selected container cannot be mapped to the task filesystem."""


@dataclass(frozen=True)
class TaskEnvironment:
    host_root: str
    working_directory: str
    aliases: tuple[str, ...]
    container: str = ""
    docker_host: str = ""
    container_id: str = ""
    configured_user: str = ""
    docker_client_fingerprint: str = ""
    docker_engine_id: str = ""
    docker_context_fingerprint: str = ""
    process_identity: ProcessIdentity | None = None

    def relative_path(self, path: str) -> str:
        for root in sorted(set((self.host_root, *self.aliases)), key=len, reverse=True):
            try:
                return str(PurePosixPath(path).relative_to(root))
            except ValueError:
                continue
        return path


_ACTIVE: ContextVar[TaskEnvironment | None] = ContextVar("task_environment", default=None)
_LIVE_SCOPE = ContextVar('live_task_environment_scope', default=False)


def docker_client_fingerprint():
    """Detect changes to declared Docker connection selectors without logging them."""
    selectors = ('DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_CONFIG',
                 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH')
    record = {name: os.environ.get(name) for name in selectors}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


@contextmanager
def use_task_environment(environment):
    """Re-enter an already observed binding without resolving its name again."""
    token = _ACTIVE.set(environment)
    scope = _LIVE_SCOPE.set(True)
    try:
        yield environment
    finally:
        _LIVE_SCOPE.reset(scope)
        _ACTIVE.reset(token)


@contextmanager
def recorded_task_environment(record):
    """Bind only recorded path facts for replay, without filesystem discovery.

    Missing or malformed records mean unknown. Never borrow the live solve's
    mapping, and restore it even when projection fails.
    """
    environment = None
    if isinstance(record, dict):
        host = record.get("host_root")
        workdir = record.get("working_directory")
        aliases = record.get("aliases")
        if isinstance(aliases, (list, tuple)) and all(
            isinstance(path, str) and path.startswith("/")
            and ".." not in PurePosixPath(path).parts
            for path in (host, workdir, *aliases)
        ):
            environment = TaskEnvironment(host, workdir, tuple(aliases))
    token = _ACTIVE.set(environment)
    try:
        yield environment
    finally:
        _ACTIVE.reset(token)


def task_environment_scope(function):
    """Keep task facts local to one solve, including exceptional exits."""
    @wraps(function)
    def scoped(*args, **kwargs):
        from .sandbox._filesystem import _ACTIVE as filesystem_active
        from .sandbox._filesystem import _SCOPED as filesystem_scoped
        from .sandbox._filesystem import _ADMISSION, FilesystemAdmission
        from .container_binding import container_image_scope
        token = _ACTIVE.set(None)
        live_token = _LIVE_SCOPE.set(True)
        filesystem_token = filesystem_active.set(None)
        scope_token = filesystem_scoped.set(True)
        admission_token = _ADMISSION.set(FilesystemAdmission())
        try:
            with container_image_scope(fresh=True):
                return function(*args, **kwargs)
        finally:
            _ADMISSION.reset(admission_token)
            filesystem_scoped.reset(scope_token)
            filesystem_active.reset(filesystem_token)
            _ACTIVE.reset(token)
            _LIVE_SCOPE.reset(live_token)
    return scoped


def from_container_metadata(cwd: str | Path, metadata: dict, *, container: str,
                            docker_host: str = "", client_fingerprint: str = "",
                            docker_identity=None) -> TaskEnvironment:
    host = Path(cwd).resolve()
    aliases = set()
    for mount in metadata.get("mounts", ()):
        if mount.get("Type") != "bind":
            continue
        source = Path(mount.get("Source", ""))
        destination = PurePosixPath(mount.get("Destination", ""))
        if not source.is_absolute() or not destination.is_absolute():
            continue
        try:
            suffix = host.relative_to(source.resolve())
        except ValueError:
            continue
        aliases.add(str(destination / suffix.as_posix()))
    # A more specific mount can hide the task inherited from a parent bind.
    # Only the mount(s) effective at the candidate root can establish its source.
    mounts = [(mount, PurePosixPath(mount.get("Destination", "")))
              for mount in metadata.get("mounts", ())]
    for alias in tuple(aliases):
        path = PurePosixPath(alias)
        covering = [(mount, destination) for mount, destination in mounts
                    if destination.is_absolute() and path.is_relative_to(destination)]
        depth = max(len(destination.parts) for _, destination in covering)
        for mount, destination in covering:
            if len(destination.parts) != depth:
                continue
            source = Path(mount.get("Source", ""))
            if (mount.get("Type") != "bind" or not source.is_absolute()
                    or (source / path.relative_to(destination).as_posix()).resolve() != host):
                aliases.discard(alias)
                break
    if not aliases:
        raise TaskEnvironmentUnavailable("selected container has no bind mount for the task directory")
    declared = PurePosixPath(metadata.get("workdir") or "/")
    preferred = [alias for alias in aliases if declared.is_relative_to(alias)]
    if preferred:
        workdir = max(preferred, key=len)
    elif len(aliases) == 1:
        workdir = next(iter(aliases))
    else:
        raise TaskEnvironmentUnavailable("task has multiple container mounts but no matching working directory")
    identity = metadata.get('id', '')
    user = metadata.get('user', '')
    if not isinstance(identity, str) or not isinstance(user, str):
        raise TaskEnvironmentUnavailable('invalid container identity metadata')
    return TaskEnvironment(str(host), workdir, tuple(sorted(aliases)), container,
                           docker_host, identity, user, client_fingerprint,
                           docker_identity.engine_id if docker_identity else '',
                           docker_identity.context_fingerprint if docker_identity else '')


def discover_task_environment(cwd: str | Path, *, refresh: bool = False,
                              timeout: float | None = None) -> TaskEnvironment:
    from .sandbox import AMBIENT_CONTAINER, container_mode

    host = str(Path(cwd).resolve())
    mode = container_mode() or ""
    docker_host = os.environ.get("DOCKER_HOST", "")
    fingerprint = docker_client_fingerprint() if mode and mode != AMBIENT_CONTAINER else ''
    existing = _ACTIVE.get()
    if not refresh and existing is not None and (
        existing.host_root, existing.container, existing.docker_host
    ) == (host, mode, docker_host) and (
        not existing.docker_client_fingerprint or existing.docker_client_fingerprint == fingerprint
    ):
        # The solve retains the observed container ID and process identity.
        # Reusing a task path must not launch another Docker discovery probe.
        return existing
    if not refresh and existing is not None and _LIVE_SCOPE.get():
        raise TaskEnvironmentUnavailable('task execution selection changed within the solve')
    if mode and mode != AMBIENT_CONTAINER:
        from .time_budget import command_time_budget, execution_deadline, remaining_before
        try:
            with command_time_budget(0 if timeout is None else timeout):
                identity = observe_docker_identity()
                result = subprocess.run(
                    ["docker", "inspect", "--format",
                     '{"mounts":{{json .Mounts}},"workdir":{{json .Config.WorkingDir}},'
                     '"id":{{json .Id}},"user":{{json .Config.User}}}', mode],
                    capture_output=True, text=True, check=True,
                    timeout=remaining_before(execution_deadline()),
                )
                environment = from_container_metadata(host, json.loads(result.stdout),
                                                       container=mode, docker_host=docker_host,
                                                       client_fingerprint=fingerprint,
                                                       docker_identity=identity)
                if not environment.container_id or environment.container_id.startswith('-'):
                    raise TaskEnvironmentUnavailable('selected container has no inspected identity')
                environment = replace(environment, process_identity=observe_container_process(
                    environment.container_id, environment.working_directory))
        except (OSError, subprocess.SubprocessError, ValueError, TypeError, ProcessIdentityError) as exc:
            raise TaskEnvironmentUnavailable("cannot discover the selected container's task mount") from exc
    else:
        environment = TaskEnvironment(host, host, (), mode, docker_host)
    _ACTIVE.set(environment)
    return environment


def relative_task_path(path: str) -> str:
    """Normalize a trace path only against facts discovered for this task."""
    environment = _ACTIVE.get()
    return environment.relative_path(path) if environment is not None else path
