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
from types import MappingProxyType


CORE_ENVIRONMENT_NAMES: tuple[str, ...] = ("PATH", "HOME", "LANG", "TERM")
"""Names inherited by ``inherit = "core"`` when present in the host env."""

DEFAULT_EXCLUDED_NAME_PARTS: tuple[str, ...] = ("KEY", "SECRET", "TOKEN")
"""Case-insensitive substrings excluded from inherited values by default."""

DEFAULT_FIXED_ENVIRONMENT: Mapping[str, str] = MappingProxyType({
    "FORCE_COLOR": "0",
    "MPLCONFIGDIR": "/tmp/mpl",
    "NO_COLOR": "1",
    "PAGER": "cat",
    "PYTHONIOENCODING": "utf-8",
    "TERM": "dumb",
})
"""Historical deterministic command defaults now owned by ``sandbox.env``."""

_INHERIT_MODES = frozenset({"all", "core", "none"})
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

        ``host_environment=None`` snapshots ``os.environ`` at call time.  A
        caller should retain this result for the session rather than resolving
        again before every command, so later host-process mutations cannot
        change a run's command environment.
        """
        source = _copy_environment(
            os.environ if host_environment is None else host_environment,
            field_name="host environment",
        )
        if self.inherit == "all":
            effective = dict(source)
        elif self.inherit == "core":
            effective = {
                name: source[name]
                for name in CORE_ENVIRONMENT_NAMES
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
