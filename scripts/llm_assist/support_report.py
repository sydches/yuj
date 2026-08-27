"""Build one shareable environment-level Yuj support report."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..llm_solver._config_layers import (
    ConfigLayerSpec,
    format_setting_path,
    iter_setting_leaves,
)
from ..llm_solver._config_redaction import redact_config_value
from ..llm_solver.config import Config, ResolvedConfig, resolve_config
from ..llm_solver.config_inspection import validate_configuration_references
from ..llm_solver.harness.sandbox.policy import inspect_sandbox_selection
from ..llm_solver.runtime_resources import validate_runtime_resources


SUPPORT_SCHEMA = "yuj.support-report"
SUPPORT_SCHEMA_VERSION = 1
_GIT_VERSION = re.compile(r"\Agit version ([0-9][0-9A-Za-z.+-]*)")


class SupportReportError(ValueError):
    """A support report destination or document is unsafe."""


def _ok(**values: object) -> dict[str, object]:
    return {"status": "ok", **values}


def _unavailable(exc: BaseException, *, summary: str) -> dict[str, object]:
    return {
        "status": "unavailable",
        "error_type": type(exc).__name__,
        "summary": summary,
    }


def _collect_installation(version: str) -> dict[str, object]:
    return _ok(
        yuj_version=str(version),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
    )


def _collect_platform() -> dict[str, object]:
    return _ok(
        operating_system=platform.system().lower() or "unknown",
        architecture=platform.machine().lower() or "unknown",
        byte_order=sys.byteorder,
    )


def _collect_git() -> dict[str, object]:
    executable = shutil.which("git")
    if executable is None:
        return {"status": "unavailable", "available": False}
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env={
                "HOME": os.devnull,
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": os.environ.get("PATH", ""),
            },
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _unavailable(exc, summary="Git version check failed")
    match = _GIT_VERSION.match(result.stdout.strip())
    if result.returncode != 0 or match is None:
        return {
            "status": "unavailable",
            "available": True,
            "exit_code": result.returncode,
            "summary": "Git returned an unrecognized version result",
        }
    return _ok(available=True, version=match.group(1))


def _collect_resources() -> dict[str, object]:
    report = validate_runtime_resources().to_dict()
    return _ok(
        origin=report.get("origin"),
        root_resource_count=report.get("root_resource_count"),
        package_resource_count=report.get("package_resource_count"),
    )


def _value_type(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "table"
    return type(value).__name__


def _configuration_section(resolved: ResolvedConfig) -> dict[str, object]:
    settings = []
    sensitive_count = 0
    for path, value in sorted(iter_setting_leaves(resolved.data)):
        _safe, redacted, reasons = redact_config_value(
            value,
            path=path,
            environment_references=resolved.environment_references,
        )
        sensitive_count += int(redacted)
        source = resolved.provenance.get(path)
        settings.append({
            "path": format_setting_path(path),
            "source_layer": source.layer_id if source is not None else "unknown",
            "value_type": _value_type(value),
            "value_redacted": bool(redacted),
            "redaction_reasons": list(reasons),
        })
    return _ok(
        layers=[layer.as_dict() for layer in resolved.layers],
        settings=settings,
        setting_count=len(settings),
        sensitive_setting_count=sensitive_count,
        values_included=False,
    )


def _collect_configuration(
    *,
    specs: Sequence[ConfigLayerSpec],
    overrides: Mapping[str, object],
    named_agents: Sequence[str],
) -> tuple[dict[str, object], ResolvedConfig | None]:
    try:
        resolved = resolve_config(
            user_config=[spec.path for spec in specs],
            overrides=dict(overrides),
            layer_specs=specs,
        )
        validate_configuration_references(
            resolved.config,
            named_agents=named_agents,
        )
        return _configuration_section(resolved), resolved
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        return (
            _unavailable(
                exc,
                summary="Resolved configuration validation failed",
            ),
            None,
        )


def _collect_sandbox(resolved: ResolvedConfig | None) -> dict[str, object]:
    if resolved is None:
        return {
            "status": "unavailable",
            "summary": "Sandbox resolution needs a valid configuration",
        }
    try:
        raw = inspect_sandbox_selection(resolved.config).as_dict()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        return _unavailable(exc, summary="Sandbox resolution failed")
    return _ok(
        platform=raw.get("platform"),
        supported=list(raw.get("supported") or ()),
        installed=list(raw.get("installed") or ()),
        available=list(raw.get("available") or ()),
        unavailable=list(raw.get("unavailable") or ()),
        selected=raw.get("selected"),
        resolved=raw.get("resolved"),
        engaged=raw.get("engaged"),
        explicit_unsandboxed=raw.get("explicit_unsandboxed"),
    )


def _collect_network(
    *,
    requested: bool,
    resolved: ResolvedConfig | None,
    check: Callable[[Config], Mapping[str, object]] | None,
) -> dict[str, object]:
    if not requested:
        return {
            "status": "omitted",
            "requested": False,
            "summary": "Network diagnostics require --network",
        }
    if resolved is None:
        return {
            "status": "unavailable",
            "requested": True,
            "summary": "Network diagnostics need a valid configuration",
        }
    if check is None:
        return {
            "status": "unavailable",
            "requested": True,
            "summary": "No network diagnostic implementation is available",
        }
    try:
        result = dict(check(resolved.config))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        return {
            **_unavailable(exc, summary="Model-service diagnostic failed"),
            "requested": True,
        }
    return _ok(
        requested=True,
        model_count=int(result.get("model_count") or 0),
        selected_model_listed=bool(result.get("selected_model_listed")),
    )


def _run_section(
    name: str,
    collector: Callable[[], dict[str, object]],
) -> dict[str, object]:
    try:
        return collector()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        return _unavailable(exc, summary=f"{name} collection failed")


def build_support_report(
    *,
    version: str,
    specs: Sequence[ConfigLayerSpec],
    overrides: Mapping[str, object],
    named_agents: Sequence[str] = (),
    network_requested: bool = False,
    network_check: Callable[[Config], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Collect deterministic sections without discarding independent results."""
    sections: dict[str, dict[str, object]] = {}
    sections["installation"] = _run_section(
        "installation", lambda: _collect_installation(version)
    )
    sections["platform"] = _run_section("platform", _collect_platform)
    sections["git"] = _run_section("git", _collect_git)
    sections["runtime_resources"] = _run_section(
        "runtime_resources", _collect_resources
    )
    try:
        configuration, resolved = _collect_configuration(
            specs=specs,
            overrides=overrides,
            named_agents=named_agents,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        configuration = _unavailable(
            exc,
            summary="configuration collection failed",
        )
        resolved = None
    sections["configuration"] = configuration
    sections["sandbox"] = _collect_sandbox(resolved)
    sections["network"] = _collect_network(
        requested=network_requested,
        resolved=resolved,
        check=network_check,
    )

    checks = [
        {"name": name, "status": section.get("status", "unavailable")}
        for name, section in sorted(sections.items())
    ]
    collected = sorted(
        name for name, section in sections.items() if section.get("status") == "ok"
    )
    unavailable = sorted(
        name
        for name, section in sections.items()
        if section.get("status") == "unavailable"
    )
    omitted = [
        "absolute configuration paths",
        "configuration values",
        "credential identifiers and values",
        "environment names and values",
        "home directory, user name, and host name",
        "repository paths and content",
        "session identifiers and artifacts",
        "task text and model messages",
    ]
    if not network_requested:
        omitted.append("network diagnostics")
    return {
        "schema": SUPPORT_SCHEMA,
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "sections": sections,
        "checks": checks,
        "inventory": {
            "collected": collected,
            "omitted": sorted(omitted),
            "redacted": [
                "configuration sensitive-value categories",
                "resolved configuration values",
            ],
            "unavailable": unavailable,
        },
        "privacy": {
            "environment_level_only": True,
            "target_repository_read": False,
            "session_store_read": False,
            "uploaded": False,
            "external_issue_opened": False,
            "network_requested": bool(network_requested),
        },
    }


def render_support_report(document: Mapping[str, object]) -> bytes:
    """Return stable pretty JSON with one trailing newline."""
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def write_support_report(
    path: Path,
    document: Mapping[str, object],
    *,
    force: bool,
) -> tuple[int, str]:
    """Atomically create one report without following a symbolic link."""
    destination = Path(path).expanduser().absolute()
    if destination.is_symlink():
        raise SupportReportError("support report path cannot be a symbolic link")
    if destination.exists() and not force:
        raise SupportReportError(
            "support report already exists; pass --force to replace it"
        )
    if destination.exists() and not destination.is_file():
        raise SupportReportError("support report path is not a regular file")
    if not destination.parent.is_dir():
        raise SupportReportError("support report parent directory does not exist")
    payload = render_support_report(document)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise SupportReportError(
            f"cannot write support report ({exc.strerror or type(exc).__name__})"
        ) from exc
    return len(payload), hashlib.sha256(payload).hexdigest()


__all__ = [
    "SUPPORT_SCHEMA",
    "SUPPORT_SCHEMA_VERSION",
    "SupportReportError",
    "build_support_report",
    "render_support_report",
    "write_support_report",
]
