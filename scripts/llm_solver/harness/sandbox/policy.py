"""Central sandbox capability, selection, and execution policy.

``sandbox.backend`` is the only current user choice.  This module resolves
that choice once, before model or tool work, and supplies the compatibility
arguments used by the existing command runners.  Compatibility fields never
choose or weaken the resolved backend.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import os
from pathlib import Path
import platform
import shutil


SANDBOX_CHOICES = frozenset({"none", "auto", "bwrap", "docker", "podman"})
SANDBOXED_BACKENDS = frozenset({"bwrap", "docker", "podman"})
_SANDBOX_BACKEND_ORDER = ("bwrap", "docker", "podman")
_PLATFORM_BACKENDS = {
    "linux": ("bwrap", "docker", "podman"),
    "macos": ("docker", "podman"),
    # The command contract requires one identical absolute cwd inside the
    # sandbox. Native Windows paths cannot satisfy that Linux-container
    # identity. WSL reports Linux and uses the Linux backend matrix.
    "windows": (),
}


class SandboxResolutionError(RuntimeError):
    """The selected sandbox cannot resolve without changing its meaning."""


def _platform_key(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized.startswith("linux"):
        return "linux"
    if normalized in {"darwin", "mac", "macos"}:
        return "macos"
    if normalized.startswith("win"):
        return "windows"
    return normalized or "unknown"


@dataclass(frozen=True, slots=True)
class SandboxCapabilities:
    """One platform probe with stable supported and installed backend lists."""

    platform: str
    supported: tuple[str, ...]
    installed: tuple[str, ...]
    executables: Mapping[str, str]

    @property
    def available(self) -> tuple[str, ...]:
        """Return supported backends whose executable was found."""
        return self.installed

    @property
    def unavailable(self) -> tuple[str, ...]:
        """Return named backends that cannot be selected on this host."""
        return tuple(
            name
            for name in _SANDBOX_BACKEND_ORDER
            if name not in self.installed
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "platform": self.platform,
            "supported": list(self.supported),
            "installed": list(self.installed),
            "available": list(self.available),
            "unavailable": list(self.unavailable),
        }


@dataclass(frozen=True, slots=True)
class SandboxResolution:
    """The configured choice and the one backend it resolves to."""

    selected: str
    resolved: str
    capabilities: SandboxCapabilities
    executable: str | None
    explicit_unsandboxed: bool
    engaged: bool | None = None
    container_image_digest: str | None = None
    legacy_container: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            **self.capabilities.as_dict(),
            "selected": self.selected,
            "resolved": self.resolved,
            "explicit_unsandboxed": self.explicit_unsandboxed,
            "engaged": self.engaged,
            "container_image_digest": self.container_image_digest,
        }


def probe_sandbox_capabilities(
    *,
    bwrap_bin: str,
    platform_name: str | None = None,
    which: Callable[[str], str | None] | None = None,
    is_file: Callable[[str], bool] | None = None,
) -> SandboxCapabilities:
    """Probe supported executables once without starting a sandbox or image."""
    platform_key = _platform_key(platform_name or platform.system())
    supported = _PLATFORM_BACKENDS.get(platform_key, ())
    which = which or shutil.which
    file_check = is_file or (
        lambda value: Path(value).is_file() and os.access(value, os.X_OK)
    )
    executables: dict[str, str] = {}
    for backend in supported:
        if backend == "bwrap":
            candidate = os.path.expandvars(os.path.expanduser(str(bwrap_bin)))
            executable = (
                candidate
                if os.path.isabs(candidate) and file_check(candidate)
                else which(candidate)
            )
        else:
            executable = which(backend)
        if executable:
            executables[backend] = str(executable)
    installed = tuple(name for name in supported if name in executables)
    return SandboxCapabilities(
        platform=platform_key,
        supported=tuple(supported),
        installed=installed,
        executables=executables,
    )


def resolve_sandbox_selection(
    selected: object,
    capabilities: SandboxCapabilities,
) -> SandboxResolution:
    """Resolve one canonical choice without fallback to unsandboxed execution."""
    if not isinstance(selected, str) or selected not in SANDBOX_CHOICES:
        allowed = ", ".join(sorted(SANDBOX_CHOICES))
        raise SandboxResolutionError(
            f"sandbox.backend must be one of {allowed}; got {selected!r}"
        )
    if selected == "none":
        return SandboxResolution(
            selected="none",
            resolved="none",
            capabilities=capabilities,
            executable=None,
            explicit_unsandboxed=True,
            engaged=False,
        )
    if selected == "auto":
        if not capabilities.installed:
            supported = ", ".join(capabilities.supported) or "none"
            raise SandboxResolutionError(
                "sandbox.backend='auto' found no installed supported sandbox "
                f"backend on {capabilities.platform}; supported={supported}. "
                "Auto never selects unsandboxed execution."
            )
        resolved = capabilities.installed[0]
        return SandboxResolution(
            selected="auto",
            resolved=resolved,
            capabilities=capabilities,
            executable=capabilities.executables[resolved],
            explicit_unsandboxed=False,
        )
    if selected not in capabilities.supported:
        supported = ", ".join(capabilities.supported) or "none"
        raise SandboxResolutionError(
            f"selected sandbox backend {selected!r} is not supported on "
            f"{capabilities.platform}; supported={supported}. Refusing to "
            "substitute another backend or run unsandboxed."
        )
    if selected not in capabilities.installed:
        raise SandboxResolutionError(
            f"selected sandbox backend {selected!r} is not installed on "
            f"{capabilities.platform}. Refusing to substitute another backend "
            "or run unsandboxed."
        )
    return SandboxResolution(
        selected=selected,
        resolved=selected,
        capabilities=capabilities,
        executable=capabilities.executables[selected],
        explicit_unsandboxed=False,
    )


def _configured_selection(cfg) -> str:
    """Return the canonical choice, with direct-fixture legacy compatibility."""
    selected = str(getattr(cfg, "sandbox_backend", "bwrap"))
    if selected == "container":
        return str(getattr(cfg, "sandbox_container_runtime", "docker"))
    # Direct Config fixtures predate sandbox.backend='none'.  Loaded configs
    # are normalized by the config loader, but preserve those unit fixtures.
    if (
        selected == "bwrap"
        and not bool(getattr(cfg, "sandbox_bash", True))
        and not bool(getattr(cfg, "sandbox_required", True))
    ):
        return "none"
    return selected


def _resolution_is_compatible(selected: str, resolved: str) -> bool:
    return (
        resolved == selected
        or selected == "auto" and resolved in SANDBOXED_BACKENDS
        or selected == "bwrap" and resolved in {"ambient", "docker-exec"}
    )


def _validate_pinned_legacy_environment(
    resolved: str,
    legacy_container: str,
    expected_container: str,
) -> None:
    """Reject an outer-container change after startup pinned the policy."""
    if resolved == "ambient":
        compatible = legacy_container == "ambient"
    elif resolved == "docker-exec":
        compatible = (
            bool(legacy_container)
            and legacy_container != "ambient"
            and (
                not expected_container
                or legacy_container == expected_container
            )
        )
    else:
        compatible = not legacy_container
    if compatible:
        return
    observed = legacy_container or "unset"
    raise SandboxResolutionError(
        "legacy YUJ_CONTAINER changed after sandbox preflight: "
        f"pinned={resolved!r}, observed={observed!r}. Refusing to change "
        "the resolved execution policy."
    )


def inspect_sandbox_selection(
    cfg,
    *,
    capabilities: SandboxCapabilities | None = None,
) -> SandboxResolution:
    """Resolve installed capability for configuration and diagnostic views."""
    capabilities = capabilities or probe_sandbox_capabilities(
        bwrap_bin=str(getattr(cfg, "bwrap_bin", "/usr/bin/bwrap")),
    )
    return resolve_sandbox_selection(_configured_selection(cfg), capabilities)


def preflight_sandbox(
    cfg,
    *,
    capabilities: SandboxCapabilities | None = None,
    environment: Mapping[str, str] | None = None,
) -> SandboxResolution:
    """Prove the exact selected backend before model or command execution."""
    environment = os.environ if environment is None else environment
    selected = _configured_selection(cfg)
    pinned = str(getattr(cfg, "sandbox_resolved_backend", "") or "")
    if pinned:
        legacy_container = str(environment.get("YUJ_CONTAINER") or "")
        _validate_pinned_legacy_environment(
            pinned,
            legacy_container,
            str(getattr(cfg, "sandbox_legacy_container", "") or ""),
        )
        supported = tuple(
            getattr(cfg, "sandbox_supported_backends", ()) or ()
        )
        installed = tuple(
            getattr(cfg, "sandbox_installed_backends", ()) or ()
        )
        capabilities = capabilities or SandboxCapabilities(
            platform=str(getattr(cfg, "sandbox_platform", "") or "unknown"),
            supported=supported,
            installed=installed,
            executables=(
                {pinned: str(cfg.sandbox_backend_executable)}
                if getattr(cfg, "sandbox_backend_executable", "")
                else {}
            ),
        )
        if not _resolution_is_compatible(selected, pinned):
            raise SandboxResolutionError(
                f"pinned sandbox resolution {pinned!r} contradicts selected "
                f"sandbox.backend={selected!r}"
            )
        return SandboxResolution(
            selected=selected,
            resolved=pinned,
            capabilities=capabilities,
            executable=(
                str(getattr(cfg, "sandbox_backend_executable", "")) or None
            ),
            explicit_unsandboxed=pinned == "none",
            engaged=pinned != "none",
            container_image_digest=(
                str(getattr(cfg, "sandbox_container_image_digest", "")) or None
            ),
            legacy_container=(
                legacy_container or None
            ),
        )

    capabilities = capabilities or probe_sandbox_capabilities(
        bwrap_bin=str(getattr(cfg, "bwrap_bin", "/usr/bin/bwrap")),
    )
    legacy_container = str(environment.get("YUJ_CONTAINER") or "")
    if legacy_container:
        if selected != "bwrap":
            raise SandboxResolutionError(
                f"sandbox.backend={selected!r} cannot be combined with legacy "
                "YUJ_CONTAINER. Refusing to change the selected backend."
            )
        return SandboxResolution(
            selected=selected,
            resolved=(
                "ambient" if legacy_container == "ambient" else "docker-exec"
            ),
            capabilities=capabilities,
            executable=None,
            explicit_unsandboxed=False,
            engaged=True,
            legacy_container=legacy_container,
        )

    if selected == "auto":
        # Resolve installed capability first so the empty set gets the same
        # stable error as configuration inspection. Then test candidates in
        # platform order until one proves operational. Host execution is not
        # a candidate.
        resolve_sandbox_selection(selected, capabilities)
        failures: list[str] = []
        for backend_name in capabilities.installed:
            candidate = resolve_sandbox_selection(
                backend_name, capabilities
            )
            try:
                operational = _preflight_resolved_backend(cfg, candidate)
            except SandboxResolutionError as exc:
                failures.append(f"{backend_name}: {exc}")
                continue
            return replace(operational, selected="auto")
        detail = "; ".join(failures) or "no installed candidates"
        raise SandboxResolutionError(
            "sandbox.backend='auto' found no operational sandbox backend; "
            f"attempts={detail}. Auto never selects unsandboxed execution."
        )

    resolution = resolve_sandbox_selection(selected, capabilities)
    return _preflight_resolved_backend(cfg, resolution)


def _preflight_resolved_backend(
    cfg,
    resolution: SandboxResolution,
) -> SandboxResolution:
    """Operationally prove one already resolved backend."""
    if resolution.resolved == "none":
        return resolution
    if resolution.resolved == "bwrap":
        from ._preflight import bwrap_preflight

        passed, detail = bwrap_preflight(str(resolution.executable))
        if not passed:
            raise SandboxResolutionError(
                "selected sandbox backend 'bwrap' failed startup preflight: "
                f"{detail or 'unknown bwrap failure'}. Refusing to substitute "
                "another backend or run unsandboxed."
            )
        return replace(resolution, engaged=True)

    from .container_backend import ContainerBackend, ContainerBackendError

    try:
        backend = ContainerBackend(
            runtime=resolution.resolved,
            image=str(getattr(cfg, "sandbox_container_image", "")),
            flags=tuple(getattr(cfg, "sandbox_container_flags", ()) or ()),
        )
        runtime_bin = resolution.executable
        if not runtime_bin:
            raise ContainerBackendError(
                f"container runtime {resolution.resolved!r} has no resolved "
                "executable"
            )
        digest = backend.image_digest(runtime_bin)
    except ContainerBackendError as exc:
        raise SandboxResolutionError(
            f"selected sandbox backend {resolution.resolved!r} failed startup "
            f"preflight: {exc}. Refusing to substitute another backend or run "
            "unsandboxed."
        ) from exc
    return replace(
        resolution,
        engaged=True,
        container_image_digest=digest,
    )


def bind_sandbox_resolution(cfg, resolution: SandboxResolution):
    """Return a Config whose runtime-only fields pin one resolved policy."""
    image = getattr(cfg, "sandbox_container_image", "")
    if resolution.container_image_digest:
        image = resolution.container_image_digest
    runtime = getattr(cfg, "sandbox_container_runtime", "docker")
    if resolution.resolved in {"docker", "podman"}:
        runtime = resolution.resolved
    bwrap_bin = str(getattr(cfg, "bwrap_bin", "/usr/bin/bwrap"))
    if resolution.resolved == "bwrap" and resolution.executable:
        bwrap_bin = resolution.executable
    return replace(
        cfg,
        sandbox_resolved_backend=resolution.resolved,
        sandbox_platform=resolution.capabilities.platform,
        sandbox_supported_backends=resolution.capabilities.supported,
        sandbox_installed_backends=resolution.capabilities.installed,
        sandbox_backend_executable=resolution.executable or "",
        sandbox_container_runtime=runtime,
        sandbox_container_image=image,
        sandbox_container_image_digest=(
            resolution.container_image_digest or ""
        ),
        sandbox_legacy_container=resolution.legacy_container or "",
        bwrap_bin=bwrap_bin,
        sandbox_bash=resolution.resolved != "none",
        sandbox_required=resolution.resolved != "none",
    )


def bind_sandbox_envelope(cfg, fields: Mapping[str, object]):
    """Pin a Config from already-computed runtime-envelope fields."""
    resolved = str(
        fields.get("sandbox_resolved")
        or fields.get("sandbox_backend")
        or _configured_selection(cfg)
    )
    runtime = str(
        fields.get("container_runtime")
        or getattr(cfg, "sandbox_container_runtime", "docker")
    )
    if resolved == "container":
        resolved = runtime
    image_digest = str(fields.get("container_image_digest") or "")
    image = image_digest or str(getattr(cfg, "sandbox_container_image", ""))
    return replace(
        cfg,
        sandbox_resolved_backend=resolved,
        sandbox_platform=str(fields.get("sandbox_platform") or ""),
        sandbox_supported_backends=tuple(fields.get("sandbox_supported") or ()),
        sandbox_installed_backends=tuple(fields.get("sandbox_installed") or ()),
        sandbox_backend_executable=str(
            fields.get("sandbox_backend_executable") or ""
        ),
        sandbox_container_runtime=runtime,
        sandbox_container_image=image,
        sandbox_container_image_digest=image_digest,
        sandbox_legacy_container=str(fields.get("yuj_container") or ""),
        bwrap_bin=(
            str(fields.get("sandbox_backend_executable"))
            if resolved == "bwrap" and fields.get("sandbox_backend_executable")
            else str(getattr(cfg, "bwrap_bin", "/usr/bin/bwrap"))
        ),
        sandbox_bash=resolved != "none",
        sandbox_required=resolved != "none",
    )


def sandbox_execution_kwargs(cfg) -> dict[str, object]:
    """Translate one pinned value for legacy-compatible command interfaces."""
    selected = _configured_selection(cfg)
    resolved = str(getattr(cfg, "sandbox_resolved_backend", "") or "")
    if not resolved:
        resolved = selected
    if resolved == "auto":
        raise SandboxResolutionError(
            "sandbox.backend='auto' reached command execution before startup "
            "resolution"
        )
    if not _resolution_is_compatible(selected, resolved):
        raise SandboxResolutionError(
            f"resolved sandbox backend {resolved!r} contradicts selected "
            f"sandbox.backend={selected!r}"
        )
    runtime = str(getattr(cfg, "sandbox_container_runtime", "docker"))
    execution_backend = resolved
    if resolved in {"docker", "podman"}:
        runtime = resolved
        execution_backend = "container"
    elif resolved in {"ambient", "docker-exec"}:
        execution_backend = "bwrap"
    sandboxed = resolved != "none"
    return {
        "sandbox": sandboxed,
        "sandbox_required": sandboxed,
        "sandbox_backend": execution_backend,
        "container_runtime": runtime,
        "container_runtime_bin": (
            str(getattr(cfg, "sandbox_backend_executable", ""))
            if resolved in {"docker", "podman"}
            else ""
        ),
        "container_image": str(
            getattr(cfg, "sandbox_container_image", "")
        ),
        "container_flags": tuple(
            getattr(cfg, "sandbox_container_flags", ()) or ()
        ),
    }


__all__ = [
    "SANDBOX_CHOICES",
    "SANDBOXED_BACKENDS",
    "SandboxCapabilities",
    "SandboxResolution",
    "SandboxResolutionError",
    "bind_sandbox_envelope",
    "bind_sandbox_resolution",
    "inspect_sandbox_selection",
    "preflight_sandbox",
    "probe_sandbox_capabilities",
    "resolve_sandbox_selection",
    "sandbox_execution_kwargs",
]
