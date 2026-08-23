"""Runtime adapter for named model roles.

``model_roles`` owns policy and validation.  This module binds that policy to
the configured transport clients without making the loop or side-request
consumers know how endpoint-specific ``Config`` objects are constructed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from ...config import Config
from ...server.profile_loader import load_profile
from .model_roles import (
    MAIN_MODEL_ROLE,
    ModelRoleResolver,
    ModelRoleRouter,
    ModelTarget,
    ResolvedModelRole,
    ResolvedRoleClient,
    RoleTokenLedger,
)


@dataclass(frozen=True)
class ModelRoleRuntime:
    """One run's shared resolver, router, and post-run token ledger."""

    resolver: ModelRoleResolver
    router: ModelRoleRouter
    token_ledger: RoleTokenLedger


@dataclass(frozen=True)
class ConsumerRoleClient:
    """A routed consumer client, including a legacy-main fallback."""

    client: Any
    requested_role: str
    resolution: ResolvedModelRole | None = None

    @property
    def effective_role(self) -> str:
        if self.resolution is None:
            return MAIN_MODEL_ROLE
        return self.resolution.effective_role

    def trace_fields(self) -> dict[str, object]:
        if self.resolution is None:
            return {"role": MAIN_MODEL_ROLE}
        return self.resolution.trace_fields()


def _stored_attr(owner: Any, name: str, default: Any = None) -> Any:
    """Read an explicitly stored attribute without triggering mock/proxy magic."""
    namespace = getattr(owner, "__dict__", None)
    if isinstance(namespace, dict):
        return namespace.get(name, default)
    return default


def _target_config(cfg: Config, resolution: ResolvedModelRole) -> Config:
    """Build a complete endpoint config for one resolved role target."""
    target = resolution.target
    context_size = int(target.context_size or cfg.context_size)
    token_budget = int(context_size * cfg.context_fill_ratio)
    return replace(
        cfg,
        model=str(target.model or cfg.model),
        profile_name=target.profile_name,
        base_url=str(target.base_url or cfg.base_url),
        api_key=str(target.api_key or cfg.api_key),
        context_size=context_size,
        max_tokens=int(context_size * cfg.max_tokens_fraction),
        recent_tool_results_chars=int(token_budget * 0.45 * 4),
        max_output_chars=int(token_budget * 0.40 * 4),
    )


def build_model_role_runtime(
    *,
    cfg: Config,
    main_client: Any,
    profiles_dir: Path,
    client_factory: Callable[[Config, Any], Any],
) -> ModelRoleRuntime:
    """Validate all role profiles and attach lazy role routing to ``main_client``."""
    main_profile_name = cfg.profile_name or cfg.model
    main_profile = getattr(main_client, "profile", None)

    def profile_loader(name: str) -> Any:
        if name == main_profile_name and (
            main_profile is not None or not (profiles_dir / name / "profile.toml").is_file()
        ):
            # Legacy main-model operation permits no profile.  Configured role
            # profiles still go through the production loader below.
            return main_profile
        return load_profile(name, profiles_dir)

    resolver = ModelRoleResolver(
        main_target=ModelTarget(
            profile_name=main_profile_name,
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            context_size=cfg.context_size,
        ),
        role_specs=cfg.model_roles,
        profile_loader=profile_loader,
    )
    token_ledger = RoleTokenLedger()

    router: ModelRoleRouter

    def make_client(resolution: ResolvedModelRole) -> Any:
        if resolution.target == resolver.main.target:
            return main_client
        role_client = client_factory(_target_config(cfg, resolution), resolution.profile)
        session_id = str(getattr(main_client, "_session_id", "") or "")
        if session_id and hasattr(role_client, "set_session_id"):
            role_client.set_session_id(session_id)
        _attach_runtime(role_client, router, token_ledger, resolution)
        return role_client

    router = ModelRoleRouter(resolver, make_client)
    _attach_runtime(main_client, router, token_ledger, resolver.main)
    return ModelRoleRuntime(resolver, router, token_ledger)


def validate_model_role_profiles(
    *,
    cfg: Config,
    main_profile: Any,
    profiles_dir: Path,
) -> None:
    """Eagerly validate configured role profiles for client-free startup paths."""
    main_profile_name = cfg.profile_name or cfg.model

    def profile_loader(name: str) -> Any:
        if name == main_profile_name and main_profile is not None:
            return main_profile
        if name == main_profile_name and not (profiles_dir / name / "profile.toml").is_file():
            return None
        return load_profile(name, profiles_dir)

    ModelRoleResolver(
        main_target=ModelTarget(
            profile_name=main_profile_name,
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            context_size=cfg.context_size,
        ),
        role_specs=cfg.model_roles,
        profile_loader=profile_loader,
    )


def _attach_runtime(
    client: Any,
    router: ModelRoleRouter,
    ledger: RoleTokenLedger,
    resolution: ResolvedModelRole,
) -> None:
    client._model_role_router = router
    client._role_token_ledger = ledger
    client._model_role_resolution = resolution


def consumer_role_client(owner: Any, role: str = "weak") -> ConsumerRoleClient:
    """Route a side consumer, preserving injected legacy clients in tests/replay."""
    direct_client = _stored_attr(owner, "client", owner)
    router = _stored_attr(owner, "_model_role_router") or _stored_attr(
        direct_client, "_model_role_router"
    )
    if router is None:
        return ConsumerRoleClient(direct_client, requested_role=role)
    routed: ResolvedRoleClient = router.client_for(role)
    return ConsumerRoleClient(
        routed.client,
        requested_role=role,
        resolution=routed.resolution,
    )


def record_role_usage(owner: Any, routed: ConsumerRoleClient, usage: Any) -> None:
    """Record one response exactly once in the run's effective-role ledger."""
    direct_client = _stored_attr(owner, "client", owner)
    ledger = _stored_attr(owner, "_role_token_ledger") or _stored_attr(
        direct_client, "_role_token_ledger"
    )
    if ledger is None or usage is None:
        return
    cached = getattr(usage, "cached_tokens", 0)
    ledger.record_usage(
        routed.resolution or routed.effective_role,
        usage,
        cached_tokens=int(cached or 0),
    )


def role_token_ledger(client: Any) -> RoleTokenLedger:
    """Return the configured run ledger, or create one for injected clients."""
    return _stored_attr(client, "_role_token_ledger") or RoleTokenLedger()


def bind_session_model_roles(
    session: Any,
    client: Any,
    ledger: RoleTokenLedger,
) -> None:
    """Expose the run router and main resolution to one loop session."""
    router = _stored_attr(client, "_model_role_router")
    session._model_role_router = router
    session._role_token_ledger = ledger
    resolution = _stored_attr(client, "_model_role_resolution")
    session._active_model_resolution = resolution
    session._active_model_role = (
        resolution.effective_role if resolution is not None else MAIN_MODEL_ROLE
    )


__all__ = [
    "ConsumerRoleClient",
    "ModelRoleRuntime",
    "build_model_role_runtime",
    "bind_session_model_roles",
    "consumer_role_client",
    "record_role_usage",
    "role_token_ledger",
    "validate_model_role_profiles",
]
