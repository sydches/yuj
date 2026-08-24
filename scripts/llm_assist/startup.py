"""Side-effect-free local startup preflight for assistant commands."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

from ..llm_solver._shared.paths import project_root
from ..llm_solver.config import load_config, require_runtime_mode
from ..llm_solver.config_inspection import validate_configuration_references
from ..llm_solver.harness._loop._driver_setup import (
    compute_runtime_envelope_fields,
    load_session_injections,
    load_session_stream_rules,
    load_system_prompt_and_provenance,
    resolve_task_format,
    scan_session_injections,
    scan_session_stream_rules,
)
from ..llm_solver.harness._loop.profile_resolution import build_tool_surface
from ..llm_solver.harness._loop.session_io import _load_bash_transforms
from ..llm_solver.harness.context_strategies import resolve_context_class
from ..llm_solver.harness.sandbox.ignore_policy import load_ignore_policy
from ..llm_solver.harness.skills import discover_skills
from ..llm_solver.harness.tools import _effective_command_environment
from ..llm_solver.models import resolve_model
from ..llm_solver.runtime_resources import validate_runtime_resources
from ..llm_solver.server.profile_loader import load_profile
from ._auth import AuthBinding, CredentialStore
from .runner import _protect_auth_environment


@dataclass(frozen=True)
class StartupPreflightReport:
    """Value-safe proof that startup reached the model-network boundary."""

    resource_origin: str
    root_resource_count: int
    package_resource_count: int
    profile: str
    context_mode: str
    detected_runner: str
    registered_tool_count: int
    active_tool_count: int
    agent_count: int
    project_instruction_count: int
    skill_count: int
    injection_count: int
    stream_rule_count: int
    sandbox_mode: str
    sandbox_engaged: bool
    network_contacted: bool = False


def preflight_assistant_startup(
    *,
    config_paths: Sequence[Path],
    cwd: Path,
    context_mode: str,
    requested_model: str | None = None,
    config_overrides: Mapping[str, object] | None = None,
    system_prompt_file: Path | None = None,
    auth_binding: AuthBinding | None = None,
    auth_store: CredentialStore | None = None,
) -> StartupPreflightReport:
    """Run ordinary local discovery and validation without model I/O or writes."""
    target = Path(cwd).expanduser().resolve()
    if not target.is_dir():
        raise NotADirectoryError(f"Yuj task cwd is not a directory: {target}")
    if system_prompt_file is not None:
        system_prompt_file = Path(system_prompt_file).expanduser().resolve()

    overrides: dict[str, object] = {
        "runtime_mode": "assistant",
        "max_sessions": 1,
        **dict(config_overrides or {}),
    }
    if requested_model:
        overrides["model"] = resolve_model(requested_model)
    cfg = load_config(user_config=list(config_paths), overrides=overrides)
    if auth_binding is not None:
        auth_store = auth_store or CredentialStore()
        auth_store.require_outside_target(target)
    cfg = _protect_auth_environment(
        cfg, auth_binding, store=auth_store
    )
    require_runtime_mode(cfg, expected="assistant", caller="yuj startup preflight")

    resources = validate_runtime_resources()
    references = validate_configuration_references(cfg)
    profiles_dir = project_root() / "profiles"
    profile_key = cfg.profile_name or cfg.model
    profile = load_profile(
        profile_key,
        profiles_dir,
        allow_base_fallback=not bool(cfg.profile_name),
    )
    context_class = resolve_context_class(context_mode)
    local_client = SimpleNamespace(profile=profile)
    tool_surface = build_tool_surface(cfg, local_client)
    _effective_command_environment(cfg)

    ignore_policy = load_ignore_policy(
        target,
        enabled=getattr(cfg, "state_ignore_file_enabled", True),
        file_names=getattr(cfg, "state_ignore_file_names", (".yujignore",)),
    )
    unreadable_paths = tuple(
        dict.fromkeys((
            *tuple(cfg.unreadable_paths),
            *ignore_policy.sandbox_unreadable_paths(),
        ))
    )
    skill_catalog = discover_skills(
        target,
        enabled=getattr(cfg, "skills_enabled", False),
        skills_dirs=getattr(cfg, "skills_dirs", ()),
        skill_paths=getattr(cfg, "skill_paths", ()),
        root_markers=cfg.project_root_markers,
        unreadable_paths=unreadable_paths,
    )
    _, _, _, prompt_metadata = load_system_prompt_and_provenance(
        cfg,
        local_client,
        target,
        system_prompt_file,
        profile.profile_dir / "profile.toml" if profile.profile_dir else None,
        None,
        context_class,
        unreadable_paths=unreadable_paths,
        skill_catalog=skill_catalog,
    )
    injections, _ = load_session_injections(
        cfg,
        target,
        unreadable_paths=unreadable_paths,
    )
    injections, _, injection_blocked = scan_session_injections(cfg, injections)
    stream_rules, _ = load_session_stream_rules(cfg, target)
    stream_rules, _, stream_blocked = scan_session_stream_rules(cfg, stream_rules)
    if prompt_metadata.security_blocked or injection_blocked or stream_blocked:
        raise RuntimeError(
            "security scan blocked local instruction content before model startup"
        )

    cfg = resolve_task_format(cfg, target)
    transforms = _load_bash_transforms(cfg)
    missing_transforms = [
        name
        for name, required, transform in (
            (
                "universal rewrites",
                cfg.bash_transforms_universal_enabled,
                transforms[1],
            ),
            (
                "forbidden rules",
                cfg.bash_quirks_forbidden_enabled,
                transforms[2],
            ),
            ("redirect rules", True, transforms[3]),
            ("redactions", True, transforms[4]),
        )
        if required and transform is None
    ]
    if missing_transforms:
        raise RuntimeError(
            "required bash rule resources did not load: "
            + ", ".join(missing_transforms)
        )
    environment = compute_runtime_envelope_fields(cfg, target)
    if (
        getattr(cfg, "sandbox_required", False)
        and cfg.sandbox_bash
        and not environment["sandbox_engaged"]
    ):
        raise RuntimeError(
            "sandbox_required=true but the configured sandbox did not pass "
            "local startup preflight"
        )

    return StartupPreflightReport(
        resource_origin=resources.origin,
        root_resource_count=resources.root_resource_count,
        package_resource_count=resources.package_resource_count,
        profile=profile.name,
        context_mode=context_mode,
        detected_runner=str(environment["detected_runner"]),
        registered_tool_count=len(tool_surface.registered_names),
        active_tool_count=len(tool_surface.default_active_names),
        agent_count=len(references["agents"]),
        project_instruction_count=len(prompt_metadata.project_instruction_files),
        skill_count=len(prompt_metadata.loaded_skills),
        injection_count=len(injections),
        stream_rule_count=len(stream_rules),
        sandbox_mode=str(environment["sandbox_mode"]),
        sandbox_engaged=bool(environment["sandbox_engaged"]),
    )


def render_startup_preflight(report: StartupPreflightReport) -> str:
    """Render a stable, secret-free human summary."""
    return (
        "Yuj startup preflight: ready\n"
        f"Runtime resources: {report.resource_origin} "
        f"({report.root_resource_count} root, "
        f"{report.package_resource_count} package)\n"
        f"Profile: {report.profile}\n"
        f"Context: {report.context_mode}\n"
        f"Runner: {report.detected_runner}\n"
        f"Tools: {report.active_tool_count} active / "
        f"{report.registered_tool_count} registered\n"
        f"Agents: {report.agent_count}\n"
        f"Local inputs: {report.project_instruction_count} project docs, "
        f"{report.skill_count} skills, {report.injection_count} injections, "
        f"{report.stream_rule_count} stream rules\n"
        f"Sandbox: {report.sandbox_mode} "
        f"(engaged={str(report.sandbox_engaged).lower()})\n"
        "Model network: not contacted\n"
    )


__all__ = [
    "StartupPreflightReport",
    "preflight_assistant_startup",
    "render_startup_preflight",
]
