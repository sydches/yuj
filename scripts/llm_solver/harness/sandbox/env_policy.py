"""Deterministic environment policy for sandboxed command processes.

This module is deliberately independent of the harness ``Config`` object.  It
accepts the shape of ``[sandbox.env]`` and produces two small, reusable values:

* an explicit environment mapping for ``subprocess.run(env=...)``; and
* ``bwrap`` arguments beginning with ``--clearenv`` followed by sorted
  ``--setenv`` entries.

Keeping resolution here gives every command surface the same policy.  The
caller is responsible for resolving the policy once at session start and
passing the resulting mapping to each sandbox backend.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import fnmatch
import os
import subprocess
from types import MappingProxyType


CORE_ENVIRONMENT_NAMES: tuple[str, ...] = ("PATH", "HOME", "LANG", "TERM")
"""Names inherited by ``inherit = "core"`` from the execution environment."""

RUNTIME_ENVIRONMENT_NAMES: tuple[str, ...] = (
    "XDG_CACHE_HOME",
    "GOROOT", "GOPATH", "GOMODCACHE", "GOCACHE", "GOTOOLCHAIN",
    "VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONHOME", "PYTHONPATH",
    "UV_PYTHON", "UV_PROJECT_ENVIRONMENT", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR",
    "CARGO_HOME", "RUSTUP_HOME", "RUSTUP_TOOLCHAIN", "JAVA_HOME", "NODE_PATH",
)
"""Opt-in SDK locations, caches and toolchain selectors; see docs/sandbox.md.

These names define permission, not discovered values or proof of installation.
Do not inherit option strings, credential/setup payloads or arbitrary SDK-name
prefixes. Unknown names stay out until their semantics have been reviewed.
"""

DEFAULT_EXCLUDED_NAME_PARTS: tuple[str, ...] = ("KEY", "SECRET", "TOKEN")
"""Case-insensitive substrings excluded from inherited values by default."""

DEFAULT_FIXED_ENVIRONMENT: Mapping[str, str] = MappingProxyType({
    "FORCE_COLOR": "0",
    "NO_COLOR": "1",
    "PAGER": "cat",
    "PYTHONIOENCODING": "utf-8",
    "TERM": "dumb",
})
"""Non-interactive terminal defaults owned by ``sandbox.env``."""

_INHERIT_MODES = frozenset({"all", "core", "runtime", "none"})
_FILTER_ACTIONS = frozenset({"include", "exclude"})
_POLICY_KEYS = frozenset({
    "inherit",
    "set",
    "filters",
    "ignore_default_excludes",
    "allow_login_shell",
})

_active_environment: ContextVar[
    tuple[Mapping[str, str], bool] | None
] = ContextVar("yuj_active_environment", default=None)


class EnvironmentPolicyError(ValueError):
    """Raised when an environment policy cannot be applied safely."""


_READ_INITIAL_ENVIRONMENT = r'''while IFS= read -r -d '' entry; do
    printf '%s\0' "$entry"
done < /proc/self/environ'''


def discover_execution_environment(*, sandbox: bool = True, cwd=None,
                                   timeout=None, image_backend=None) -> dict[str, str]:
    """Inspect the environment a command inherits at its execution boundary.

    Read the entry process's original environment in the selected namespace
    and identity, using the named-container guard or bound image builder.
    Do not read shell profiles or substitute the harness host's environment
    when the selected container is unavailable. Callers freeze the result for
    the solve and apply EnvironmentPolicy before giving it to any command.
    """
    from . import AMBIENT_CONTAINER, container_mode

    mode = container_mode() if sandbox else None
    if not sandbox or image_backend is None and mode in (None, AMBIENT_CONTAINER):
        return dict(os.environ)
    from ..task_environment import discover_task_environment, TaskEnvironmentUnavailable
    from ..process_identity import guarded_script_argv, ProcessIdentityError
    from .._tools._run_in_sandbox import _execute
    from ..time_budget import command_time_budget, execution_deadline, remaining_before
    from .container_backend import ContainerBackendError
    from ..container_binding import bind_container_image
    try:
        with command_time_budget(0 if timeout is None else timeout):
            if image_backend is not None:
                if mode is not None:
                    raise EnvironmentPolicyError('image backend conflicts with YUJ_CONTAINER')
                runtime = image_backend.resolve_runtime(sandbox_required=True)
                bound = bind_container_image(image_backend, runtime)
                argv = bound.build_argv(
                    _READ_INITIAL_ENVIRONMENT, os.getcwd() if cwd is None else cwd,
                    runtime_bin=runtime, _initial_environment=True,
                )
            else:
                task = discover_task_environment(os.getcwd() if cwd is None else cwd)
                argv = guarded_script_argv(
                    ['docker', 'exec', '--workdir', task.working_directory, task.container_id],
                    # Read procfs with builtins, avoiding shell-added values
                    # and assumptions about an env utility's location.
                    _READ_INITIAL_ENVIRONMENT,
                    task.process_identity,
                )
            result = _execute(argv, timeout=remaining_before(execution_deadline()), binary=True)
            if result.returncode:
                raise EnvironmentPolicyError('container environment probe failed')
    except (OSError, subprocess.SubprocessError, TaskEnvironmentUnavailable,
            ProcessIdentityError, EnvironmentPolicyError, ContainerBackendError):
        # Errors can contain command output, including environment secrets.
        raise EnvironmentPolicyError(
            "cannot inspect the selected container's execution environment"
        ) from None
    payload = result.stdout
    if not isinstance(payload, bytes) or (payload and not payload.endswith(b"\0")):
        raise EnvironmentPolicyError("invalid container environment response")
    source: dict[str, str] = {}
    for entry in payload.split(b"\0")[:-1]:
        name, separator, value = entry.partition(b"=")
        if not name or not separator or os.fsdecode(name) in source:
            raise EnvironmentPolicyError("invalid container environment response")
        source[os.fsdecode(name)] = os.fsdecode(value)
    return source


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise EnvironmentPolicyError(f"{field_name} must be a boolean")
    return value


def _validate_name(name: object, *, field_name: str) -> str:
    if not isinstance(name, str) or not name:
        raise EnvironmentPolicyError(
            f"{field_name} environment variable name must be a non-empty string"
        )
    if "=" in name or "\x00" in name:
        raise EnvironmentPolicyError(
            f"{field_name} environment variable name {name!r} contains '=' or NUL"
        )
    return name


def _validate_value(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise EnvironmentPolicyError(
            f"sandbox.env.set[{name!r}] must be a string"
        )
    if "\x00" in value:
        raise EnvironmentPolicyError(
            f"sandbox.env.set[{name!r}] contains NUL"
        )
    return value


def _copy_environment(
    environment: Mapping[str, str], *, field_name: str,
) -> dict[str, str]:
    copied: dict[str, str] = {}
    for raw_name, raw_value in environment.items():
        name = _validate_name(raw_name, field_name=field_name)
        value = _validate_value(raw_value, name=name)
        copied[name] = value
    return copied


def _matches(pattern: str, name: str) -> bool:
    """Return a case-insensitive shell-wildcard match for one env name."""
    return fnmatch.fnmatchcase(name.casefold(), pattern.casefold())


@dataclass(frozen=True, slots=True)
class EnvironmentPolicy:
    """Validated ``[sandbox.env]`` policy.

    Resolution order is intentionally visible in :meth:`resolve`:

    1. choose the inherited host-name set;
    2. apply the default secret-name exclusions;
    3. apply custom ``exclude`` patterns;
    4. overlay fixed ``set`` values; and
    5. when any ``include`` pattern exists, retain only matching names.

    ``set`` therefore provides an explicit way to restore a name removed by a
    default or custom exclusion.  An include allowlist remains the final
    boundary and applies to inherited and fixed values alike.
    """

    inherit: str = "core"
    set: Mapping[str, str] = field(default_factory=dict)
    filters: Mapping[str, str] = field(default_factory=dict)
    ignore_default_excludes: bool = False
    allow_login_shell: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.inherit, str) or self.inherit not in _INHERIT_MODES:
            allowed = ", ".join(sorted(_INHERIT_MODES))
            raise EnvironmentPolicyError(
                f"sandbox.env.inherit must be one of {allowed}; got {self.inherit!r}"
            )
        if not isinstance(self.set, Mapping):
            raise EnvironmentPolicyError("sandbox.env.set must be a table")
        if not isinstance(self.filters, Mapping):
            raise EnvironmentPolicyError("sandbox.env.filters must be a table")
        _require_bool(
            self.ignore_default_excludes,
            field_name="sandbox.env.ignore_default_excludes",
        )
        _require_bool(
            self.allow_login_shell,
            field_name="sandbox.env.allow_login_shell",
        )

        fixed = _copy_environment(self.set, field_name="sandbox.env.set")
        normalized_filters: dict[str, str] = {}
        for raw_pattern, raw_action in self.filters.items():
            if not isinstance(raw_pattern, str) or not raw_pattern:
                raise EnvironmentPolicyError(
                    "sandbox.env.filters pattern must be a non-empty string"
                )
            if "\x00" in raw_pattern:
                raise EnvironmentPolicyError(
                    f"sandbox.env.filters pattern {raw_pattern!r} contains NUL"
                )
            if not isinstance(raw_action, str) or raw_action not in _FILTER_ACTIONS:
                raise EnvironmentPolicyError(
                    f"sandbox.env.filters[{raw_pattern!r}] must be "
                    "'include' or 'exclude'"
                )
            normalized_filters[raw_pattern] = raw_action

        # Copy and freeze caller-owned mappings.  Without this, mutating the
        # original dict after session start would silently change isolation.
        object.__setattr__(self, "set", MappingProxyType(fixed))
        object.__setattr__(
            self, "filters", MappingProxyType(normalized_filters),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "EnvironmentPolicy":
        """Build a policy from the decoded ``[sandbox.env]`` TOML table.

        Unknown fields are rejected so a misspelled isolation setting cannot
        silently degrade to a default.
        """
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise EnvironmentPolicyError("sandbox.env must be a table")
        unknown = sorted(set(value) - _POLICY_KEYS)
        if unknown:
            raise EnvironmentPolicyError(
                "sandbox.env contains unknown field(s): " + ", ".join(unknown)
            )
        return cls(
            inherit=value.get("inherit", "core"),
            set=value.get("set", {}),
            filters=value.get("filters", {}),
            ignore_default_excludes=_require_bool(
                value.get("ignore_default_excludes", False),
                field_name="sandbox.env.ignore_default_excludes",
            ),
            allow_login_shell=_require_bool(
                value.get("allow_login_shell", False),
                field_name="sandbox.env.allow_login_shell",
            ),
        )

    def resolve(
        self, host_environment: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Return a validated, name-sorted environment mapping.

        The historical ``host_environment`` argument accepts the inspected
        execution environment. ``None`` snapshots ``os.environ`` for local
        callers. Retain the result for the session rather than resolving again
        before every command, so later mutations cannot change a run's
        command environment.
        """
        source = _copy_environment(
            os.environ if host_environment is None else host_environment,
            field_name="execution environment",
        )
        if self.inherit == "all":
            effective = dict(source)
        elif self.inherit in {"core", "runtime"}:
            names = CORE_ENVIRONMENT_NAMES
            if self.inherit == "runtime":
                names += RUNTIME_ENVIRONMENT_NAMES
            effective = {
                name: source[name]
                for name in names
                if name in source
            }
        else:
            effective = {}

        if not self.ignore_default_excludes:
            effective = {
                name: value
                for name, value in effective.items()
                if not any(
                    part in name.upper() for part in DEFAULT_EXCLUDED_NAME_PARTS
                )
            }

        exclude_patterns = tuple(
            pattern
            for pattern, action in self.filters.items()
            if action == "exclude"
        )
        if exclude_patterns:
            effective = {
                name: value
                for name, value in effective.items()
                if not any(_matches(pattern, name) for pattern in exclude_patterns)
            }

        effective.update(self.set)

        include_patterns = tuple(
            pattern
            for pattern, action in self.filters.items()
            if action == "include"
        )
        if include_patterns:
            effective = {
                name: value
                for name, value in effective.items()
                if any(_matches(pattern, name) for pattern in include_patterns)
            }

        return {name: effective[name] for name in sorted(effective)}

    def effective_names(
        self, host_environment: Mapping[str, str] | None = None,
    ) -> tuple[str, ...]:
        """Return only the names suitable for secret-free trace provenance."""
        return tuple(self.resolve(host_environment))


def resolve_environment(
    policy: EnvironmentPolicy,
    host_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Functional wrapper for callers that do not retain a policy method."""
    if not isinstance(policy, EnvironmentPolicy):
        raise TypeError("policy must be an EnvironmentPolicy")
    return policy.resolve(host_environment)


def build_bwrap_env_argv(environment: Mapping[str, str]) -> list[str]:
    """Return deterministic ``bwrap`` arguments for an explicit env block."""
    explicit = _copy_environment(environment, field_name="effective environment")
    argv = ["--clearenv"]
    for name in sorted(explicit):
        argv.extend(("--setenv", name, explicit[name]))
    return argv


def build_subprocess_env(environment: Mapping[str, str]) -> dict[str, str]:
    """Return an independent validated mapping for ``subprocess.run(env=...)``."""
    explicit = _copy_environment(environment, field_name="effective environment")
    return {name: explicit[name] for name in sorted(explicit)}


def build_clean_exec_argv(
    argv: list[str] | tuple[str, ...], environment: Mapping[str, str],
) -> list[str]:
    """Prefix an argv with ``env -i`` and one deterministic environment.

    This is used by ambient and unsandboxed long-lived children, where there
    is no bwrap/container boundary at which to apply ``--clearenv``.
    """
    explicit = build_subprocess_env(environment)
    return [
        "/usr/bin/env", "-i",
        *(f"{name}={explicit[name]}" for name in explicit),
        *argv,
    ]


@contextmanager
def activate_environment(
    environment: Mapping[str, str], *, allow_login_shell: bool = False,
):
    """Install one immutable command environment for nested tool handlers."""
    explicit = MappingProxyType(build_subprocess_env(environment))
    token = _active_environment.set((explicit, bool(allow_login_shell)))
    try:
        yield explicit
    finally:
        _active_environment.reset(token)


def active_environment() -> tuple[Mapping[str, str] | None, bool]:
    """Return the dispatch-scoped command environment and login-shell flag."""
    active = _active_environment.get()
    if active is None:
        return None, False
    return active


def build_bash_argv(
    command: str | None,
    *,
    allow_login_shell: bool = False,
    executable: str = "bash",
) -> list[str]:
    """Build non-interactive bash argv with explicit profile semantics.

    ``command=None`` selects stdin mode for the persistent runner.  Login
    shells are opt-in; the default explicitly disables profile and rc files so
    they cannot mutate the resolved environment behind the policy's back.
    """
    if not isinstance(allow_login_shell, bool):
        raise EnvironmentPolicyError("allow_login_shell must be a boolean")
    if not isinstance(executable, str) or not executable or "\x00" in executable:
        raise EnvironmentPolicyError("bash executable must be a non-empty string")
    argv = [executable]
    if allow_login_shell:
        argv.append("--login")
    else:
        argv.extend(("--noprofile", "--norc"))
    argv.append("-s" if command is None else "-o")
    if command is not None:
        if not isinstance(command, str) or "\x00" in command:
            raise EnvironmentPolicyError("bash command must be a NUL-free string")
        argv.extend(("pipefail", "-c", command))
    return argv
