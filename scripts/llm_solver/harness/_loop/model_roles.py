"""Model-role resolution, client routing, and per-role token accounting.

This module is deliberately independent of the central ``Config`` and HTTP
client classes.  Integration code supplies the already-resolved main target,
the profile loader, and a client factory.  That keeps named side-model policy
reusable by compaction, handoff, advisor, and classifier consumers.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit


MAIN_MODEL_ROLE = "main"
FALLBACK_REVERT_POLICIES: tuple[str, ...] = ("never", "next_session")
_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")
_REASON_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_TARGET_KEYS = frozenset(
    {
        "api_key",
        "base_url",
        "context_size",
        "endpoint",
        "model",
        "profile",
        "profile_name",
    }
)


class ModelRoleError(ValueError):
    """A model-role target or role-routing setting is invalid."""


@dataclass(frozen=True)
class ModelTarget:
    """Configured profile plus optional request-endpoint overrides."""

    profile_name: str
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    context_size: int | None = None

    def label(self) -> str:
        """Return a secret-free target label for trace/provenance fields."""
        endpoint = self.base_url or "<main-endpoint>"
        return f"{self.profile_name}@{endpoint}"


@dataclass(frozen=True)
class ResolvedModelRole:
    """A fully resolved model target and its validated profile object."""

    requested_role: str
    effective_role: str
    target: ModelTarget
    profile: Any
    uses_main_fallback: bool = False

    def trace_fields(self) -> dict[str, object]:
        """Return additive fields for a side-request trace event."""
        fields: dict[str, object] = {"role": self.effective_role}
        if self.uses_main_fallback:
            fields.update(
                {
                    "requested_role": self.requested_role,
                    "role_fallback": "main",
                }
            )
        return fields

    def provenance_fields(self) -> dict[str, object]:
        """Return non-secret resolved target metadata."""
        return {
            "role": self.effective_role,
            "profile_name": self.target.profile_name,
            "model": self.target.model,
            "base_url": self.target.base_url,
            "context_size": self.target.context_size,
            "uses_main_fallback": self.uses_main_fallback,
        }


@dataclass(frozen=True)
class ResolvedRoleClient:
    """A client together with the role decision that selected it."""

    client: Any
    resolution: ResolvedModelRole

    def trace_fields(self) -> dict[str, object]:
        return self.resolution.trace_fields()


@dataclass(frozen=True)
class ContextWindowCheck:
    """Result of checking a prompt against one resolved target window."""

    prompt_tokens: int
    context_size: int
    context_fill_ratio: float
    prompt_token_limit: int
    fits: bool


@dataclass(frozen=True)
class ModelFallbackTransition:
    """One role-local switch after the active client's retries are exhausted."""

    role: str
    from_resolution: ResolvedModelRole
    to_resolution: ResolvedModelRole
    reason: str
    from_index: int
    to_index: int

    def trace_fields(self) -> dict[str, object]:
        """Return the payload for a ``model_fallback`` trace event."""
        return {
            "role": self.role,
            "from": self.from_resolution.target.label(),
            "to": self.to_resolution.target.label(),
            "reason": self.reason,
            "from_profile": self.from_resolution.target.profile_name,
            "to_profile": self.to_resolution.target.profile_name,
            "from_model": self.from_resolution.target.model,
            "to_model": self.to_resolution.target.model,
            "from_context_size": self.from_resolution.target.context_size,
            "to_context_size": self.to_resolution.target.context_size,
        }


@dataclass(frozen=True)
class RoutedModelFallback:
    """A fallback transition plus the ready client for its new target."""

    transition: ModelFallbackTransition
    routed_client: ResolvedRoleClient


def _normalize_name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelRoleError(f"{field} must be a non-empty string")
    normalized = value.strip()
    if not _NAME_RE.fullmatch(normalized):
        raise ModelRoleError(
            f"{field} must contain only letters, digits, '.', '_' or '-'"
        )
    return normalized


def normalize_model_role(role: str) -> str:
    """Validate and normalize a consumer-selected role name."""
    normalized = _normalize_name(role, field="model role").lower()
    return normalized


def _validate_optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ModelRoleError(f"{field} must be a non-empty string when set")
    return value.strip()


def _validate_base_url(value: object, *, field: str) -> str | None:
    endpoint = _validate_optional_string(value, field=field)
    if endpoint is None:
        return None
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ModelRoleError(f"{field} must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ModelRoleError(
            f"{field} must not embed credentials; configure api_key separately"
        )
    if parsed.query or parsed.fragment:
        raise ModelRoleError(f"{field} must not contain a query or fragment")
    return endpoint.rstrip("/")


def _validate_context_size(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelRoleError(f"{field} must be a positive integer")
    return value


def parse_model_target(value: object, *, field: str) -> ModelTarget:
    """Parse a role target from ``"profile"`` or an inline table.

    Inline tables accept ``profile``, plus optional ``endpoint``/``base_url``,
    ``model``, ``api_key``, and ``context_size`` overrides.
    """
    if isinstance(value, str):
        return ModelTarget(profile_name=_normalize_name(value, field=field))
    if not isinstance(value, Mapping):
        raise ModelRoleError(f"{field} must be a profile string or table/mapping")

    unknown = set(value) - _TARGET_KEYS
    if unknown:
        raise ModelRoleError(f"{field} has unknown fields: {sorted(unknown)}")
    if "profile" in value and "profile_name" in value:
        raise ModelRoleError(f"{field} cannot set both profile and profile_name")
    if "endpoint" in value and "base_url" in value:
        raise ModelRoleError(f"{field} cannot set both endpoint and base_url")

    profile_value = value.get("profile", value.get("profile_name"))
    profile_name = _normalize_name(profile_value, field=f"{field}.profile")
    endpoint_value = value.get("endpoint", value.get("base_url"))
    return ModelTarget(
        profile_name=profile_name,
        model=_validate_optional_string(value.get("model"), field=f"{field}.model"),
        base_url=_validate_base_url(endpoint_value, field=f"{field}.endpoint"),
        api_key=_validate_optional_string(
            value.get("api_key"), field=f"{field}.api_key"
        ),
        context_size=_validate_context_size(
            value.get("context_size"), field=f"{field}.context_size"
        ),
    )


def validate_role_specs(role_specs: Mapping[str, object] | None) -> dict[str, ModelTarget]:
    """Validate configured named roles, ignoring explicit empty defaults."""
    if role_specs is None:
        return {}
    if not isinstance(role_specs, Mapping):
        raise ModelRoleError("models.roles must be a table/mapping")

    validated: dict[str, ModelTarget] = {}
    for raw_role, raw_target in role_specs.items():
        role = normalize_model_role(raw_role)
        if role == MAIN_MODEL_ROLE:
            raise ModelRoleError(
                "models.roles.main is reserved; configure the main model under [model]"
            )
        if raw_target is None or (
            isinstance(raw_target, str) and not raw_target.strip()
        ):
            continue
        if role in validated:
            raise ModelRoleError(f"duplicate model role after normalization: {raw_role!r}")
        validated[role] = parse_model_target(
            raw_target,
            field=f"models.roles.{role}",
        )
    return validated


def parse_fallback_target(value: object, *, field: str) -> ModelTarget:
    """Parse one ``<profile>@<endpoint>`` fallback entry or inline table."""
    if isinstance(value, str):
        raw = value.strip()
        profile_name, separator, endpoint = raw.partition("@")
        if not separator or not profile_name or not endpoint:
            raise ModelRoleError(
                f"{field} must use '<profile>@<endpoint>' syntax"
            )
        return parse_model_target(
            {"profile": profile_name, "endpoint": endpoint},
            field=field,
        )
    target = parse_model_target(value, field=field)
    if target.base_url is None:
        raise ModelRoleError(f"{field}.endpoint is required for a fallback target")
    return target


def validate_fallback_chains(
    fallback_chain: Mapping[str, object] | Sequence[object] | None,
) -> dict[str, tuple[ModelTarget, ...]]:
    """Validate role-keyed fallback chains; a bare list applies to ``main``."""
    if fallback_chain is None:
        return {}
    if isinstance(fallback_chain, Sequence) and not isinstance(
        fallback_chain, (str, bytes, bytearray)
    ):
        raw_chains: Mapping[str, object] = {MAIN_MODEL_ROLE: fallback_chain}
    elif isinstance(fallback_chain, Mapping):
        raw_chains = fallback_chain
    else:
        raise ModelRoleError(
            "models.fallback_chain must be a list or a role-to-list table/mapping"
        )

    validated: dict[str, tuple[ModelTarget, ...]] = {}
    for raw_role, raw_entries in raw_chains.items():
        role = normalize_model_role(raw_role)
        if not isinstance(raw_entries, Sequence) or isinstance(
            raw_entries, (str, bytes, bytearray)
        ):
            raise ModelRoleError(f"models.fallback_chain.{role} must be a list")
        targets = tuple(
            parse_fallback_target(
                entry,
                field=f"models.fallback_chain.{role}[{index}]",
            )
            for index, entry in enumerate(raw_entries)
        )
        if targets:
            validated[role] = targets
    return validated


def normalize_fallback_revert(value: str) -> str:
    """Validate when a fallback chain returns to its primary target."""
    if not isinstance(value, str) or not value.strip():
        raise ModelRoleError("models.fallback_revert must be a non-empty string")
    normalized = value.strip().lower()
    if normalized not in FALLBACK_REVERT_POLICIES:
        raise ModelRoleError(
            "models.fallback_revert must be one of: "
            + ", ".join(FALLBACK_REVERT_POLICIES)
        )
    return normalized


def check_context_window(
    prompt_tokens: int,
    resolution: ResolvedModelRole,
    context_fill_ratio: object,
) -> ContextWindowCheck:
    """Recheck an outgoing prompt against a newly resolved model window."""
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens < 0:
        raise ModelRoleError("prompt_tokens must be a non-negative integer")
    if isinstance(context_fill_ratio, bool) or not isinstance(
        context_fill_ratio, (int, float)
    ):
        raise ModelRoleError("context_fill_ratio must be a number in (0, 1]")
    ratio = float(context_fill_ratio)
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ModelRoleError("context_fill_ratio must be a number in (0, 1]")
    context_size = resolution.target.context_size
    if context_size is None or context_size <= 0:
        raise ModelRoleError("resolved fallback target has no positive context size")
    limit = math.floor(context_size * ratio)
    return ContextWindowCheck(
        prompt_tokens=prompt_tokens,
        context_size=context_size,
        context_fill_ratio=ratio,
        prompt_token_limit=limit,
        fits=prompt_tokens <= limit,
    )


class ModelRoleResolver:
    """Eagerly validate configured profiles and resolve requested roles."""

    def __init__(
        self,
        *,
        main_target: ModelTarget,
        role_specs: Mapping[str, object] | None,
        profile_loader: Callable[[str], Any],
    ) -> None:
        if main_target.model is None:
            raise ModelRoleError("main target must define its served model ID")
        if main_target.base_url is None:
            raise ModelRoleError("main target must define its base URL")
        if main_target.context_size is None:
            raise ModelRoleError("main target must define its context size")
        # Reuse the same validation rules as role targets.
        main_target = ModelTarget(
            profile_name=_normalize_name(
                main_target.profile_name, field="model.profile_name"
            ),
            model=_validate_optional_string(main_target.model, field="model.name"),
            base_url=_validate_base_url(main_target.base_url, field="server.base_url"),
            api_key=_validate_optional_string(
                main_target.api_key, field="server.api_key"
            ),
            context_size=_validate_context_size(
                main_target.context_size, field="model.context_size"
            ),
        )
        self._profile_loader = profile_loader
        self._main = self._resolve_target(
            requested_role=MAIN_MODEL_ROLE,
            effective_role=MAIN_MODEL_ROLE,
            target=main_target,
            defaults=main_target,
        )
        self._roles: dict[str, ResolvedModelRole] = {}
        for role, target in validate_role_specs(role_specs).items():
            self._roles[role] = self._resolve_target(
                requested_role=role,
                effective_role=role,
                target=target,
                defaults=main_target,
            )

    def _load_profile(self, role: str, profile_name: str) -> Any:
        try:
            return self._profile_loader(profile_name)
        except Exception as exc:
            raise ModelRoleError(
                f"model role {role!r} profile {profile_name!r} failed validation: {exc}"
            ) from exc

    def _resolve_target(
        self,
        *,
        requested_role: str,
        effective_role: str,
        target: ModelTarget,
        defaults: ModelTarget,
    ) -> ResolvedModelRole:
        profile = self._load_profile(requested_role, target.profile_name)
        profile_context = getattr(profile, "context_size", None)
        context_size = target.context_size
        if context_size is None and isinstance(profile_context, int) and profile_context > 0:
            context_size = profile_context
        if context_size is None:
            context_size = defaults.context_size
        resolved_target = ModelTarget(
            profile_name=target.profile_name,
            model=target.model or target.profile_name,
            base_url=target.base_url or defaults.base_url,
            api_key=target.api_key if target.api_key is not None else defaults.api_key,
            context_size=context_size,
        )
        return ResolvedModelRole(
            requested_role=requested_role,
            effective_role=effective_role,
            target=resolved_target,
            profile=profile,
        )

    @property
    def configured_roles(self) -> tuple[str, ...]:
        return tuple(self._roles)

    @property
    def main(self) -> ResolvedModelRole:
        return self._main

    def resolve(self, role: str = MAIN_MODEL_ROLE) -> ResolvedModelRole:
        """Resolve a consumer role, falling back to main when it is unset."""
        requested = normalize_model_role(role)
        if requested == MAIN_MODEL_ROLE:
            return self._main
        configured = self._roles.get(requested)
        if configured is not None:
            return configured
        return ResolvedModelRole(
            requested_role=requested,
            effective_role=MAIN_MODEL_ROLE,
            target=self._main.target,
            profile=self._main.profile,
            uses_main_fallback=True,
        )

    def resolve_explicit_target(
        self,
        role: str,
        target: ModelTarget,
    ) -> ResolvedModelRole:
        """Resolve and validate one explicit target in a role's fallback chain."""
        requested = normalize_model_role(role)
        defaults = self.resolve(requested).target
        return self._resolve_target(
            requested_role=requested,
            effective_role=requested,
            target=target,
            defaults=defaults,
        )


def _normalize_fallback_reason(reason: str) -> str:
    """Require a stable reason code rather than raw exception or prompt text."""
    if not isinstance(reason, str) or not reason.strip():
        raise ModelRoleError("fallback reason must be a non-empty code")
    normalized = reason.strip()
    if not _REASON_RE.fullmatch(normalized):
        raise ModelRoleError(
            "fallback reason must contain only letters, digits, '.', '_', ':' or '-'"
        )
    return normalized


class ModelFallbackController:
    """Track role-local fallback position and explicit session reversion."""

    def __init__(
        self,
        resolver: ModelRoleResolver,
        *,
        fallback_chain: Mapping[str, object] | Sequence[object] | None,
        fallback_revert: str = "never",
    ) -> None:
        self.resolver = resolver
        self.fallback_revert = normalize_fallback_revert(fallback_revert)
        self._chains: dict[str, tuple[ResolvedModelRole, ...]] = {}
        self._indices: dict[str, int] = {}
        self._fallback_count = 0
        self._fallback_roles: set[str] = set()

        for role, targets in validate_fallback_chains(fallback_chain).items():
            primary = resolver.resolve(role)
            resolved = [primary]
            seen_targets = {primary.target}
            for target in targets:
                candidate = resolver.resolve_explicit_target(role, target)
                if candidate.target in seen_targets:
                    raise ModelRoleError(
                        f"models.fallback_chain.{role} repeats target "
                        f"{candidate.target.label()}"
                    )
                seen_targets.add(candidate.target)
                resolved.append(candidate)
            self._chains[role] = tuple(resolved)
            self._indices[role] = 0

    @property
    def configured_roles(self) -> tuple[str, ...]:
        return tuple(self._chains)

    def current(self, role: str = MAIN_MODEL_ROLE) -> ResolvedModelRole:
        requested = normalize_model_role(role)
        chain = self._chains.get(requested)
        if chain is None:
            return self.resolver.resolve(requested)
        return chain[self._indices[requested]]

    def advance(
        self,
        role: str,
        *,
        reason: str,
    ) -> ModelFallbackTransition | None:
        """Advance once after retry exhaustion, or return ``None`` at chain end."""
        requested = normalize_model_role(role)
        reason_code = _normalize_fallback_reason(reason)
        chain = self._chains.get(requested)
        if chain is None:
            return None
        from_index = self._indices[requested]
        to_index = from_index + 1
        if to_index >= len(chain):
            return None
        self._indices[requested] = to_index
        self._fallback_count += 1
        self._fallback_roles.add(requested)
        return ModelFallbackTransition(
            role=requested,
            from_resolution=chain[from_index],
            to_resolution=chain[to_index],
            reason=reason_code,
            from_index=from_index,
            to_index=to_index,
        )

    def has_next(self, role: str = MAIN_MODEL_ROLE) -> bool:
        """Return whether advancing this role would select another target."""
        requested = normalize_model_role(role)
        chain = self._chains.get(requested)
        return bool(
            chain is not None
            and self._indices[requested] + 1 < len(chain)
        )

    def begin_session(self) -> bool:
        """Apply ``next_session`` reversion; return whether any role reverted."""
        if self.fallback_revert != "next_session":
            return False
        reverted = any(index > 0 for index in self._indices.values())
        for role in self._indices:
            self._indices[role] = 0
        return reverted

    def metrics_fields(self) -> dict[str, object]:
        """Return post-run fields that let studies exclude changed treatments."""
        active = {
            role: self.current(role).target.label()
            for role, index in sorted(self._indices.items())
            if index > 0
        }
        return {
            "model_fallback_used": self._fallback_count > 0,
            "model_fallback_count": self._fallback_count,
            "model_fallback_roles": sorted(self._fallback_roles),
            "model_fallback_active_targets": active,
        }

    def provenance_fields(self) -> dict[str, object]:
        """Return the configured secret-free chain and initial target."""
        return {
            "model_fallback_revert": self.fallback_revert,
            "model_fallback_chains": {
                role: [resolution.target.label() for resolution in chain[1:]]
                for role, chain in sorted(self._chains.items())
            },
            "initial_model_target": self.resolver.main.target.label(),
        }


class ModelRoleRouter:
    """Return lazily constructed clients for effective model targets."""

    def __init__(
        self,
        resolver: ModelRoleResolver,
        client_factory: Callable[[ResolvedModelRole], Any],
        fallback_controller: ModelFallbackController | None = None,
    ) -> None:
        if (
            fallback_controller is not None
            and fallback_controller.resolver is not resolver
        ):
            raise ModelRoleError(
                "fallback controller and client router must share one role resolver"
            )
        self.resolver = resolver
        self.fallback_controller = fallback_controller
        self._client_factory = client_factory
        self._clients: dict[ModelTarget, Any] = {}

    def _client_for_resolution(
        self,
        resolution: ResolvedModelRole,
    ) -> ResolvedRoleClient:
        client = self._clients.get(resolution.target)
        if client is None:
            client = self._client_factory(resolution)
            self._clients[resolution.target] = client
        return ResolvedRoleClient(client=client, resolution=resolution)

    def client_for(self, role: str = MAIN_MODEL_ROLE) -> ResolvedRoleClient:
        resolution = (
            self.fallback_controller.current(role)
            if self.fallback_controller is not None
            else self.resolver.resolve(role)
        )
        return self._client_for_resolution(resolution)

    def switch_after_retry_exhaustion(
        self,
        role: str,
        *,
        reason: str,
    ) -> RoutedModelFallback | None:
        """Advance a role chain and return its lazily built replacement client."""
        if self.fallback_controller is None:
            return None
        transition = self.fallback_controller.advance(role, reason=reason)
        if transition is None:
            return None
        return RoutedModelFallback(
            transition=transition,
            routed_client=self._client_for_resolution(transition.to_resolution),
        )

    def begin_session(self) -> bool:
        if self.fallback_controller is None:
            return False
        return self.fallback_controller.begin_session()


@dataclass
class _RoleTokenTotals:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


class RoleTokenLedger:
    """Accumulate post-run token metrics by the effective model role."""

    def __init__(self) -> None:
        self._totals: dict[str, _RoleTokenTotals] = {}

    def record(
        self,
        role: str | ResolvedModelRole,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> None:
        effective_role = (
            role.effective_role
            if isinstance(role, ResolvedModelRole)
            else normalize_model_role(role)
        )
        values = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelRoleError(f"{name} must be a non-negative integer")
        if cached_tokens > prompt_tokens:
            raise ModelRoleError("cached_tokens cannot exceed prompt_tokens")

        totals = self._totals.setdefault(effective_role, _RoleTokenTotals())
        totals.requests += 1
        totals.prompt_tokens += prompt_tokens
        totals.completion_tokens += completion_tokens
        totals.cached_tokens += cached_tokens

    def record_usage(
        self,
        role: str | ResolvedModelRole,
        usage: object,
        *,
        cached_tokens: int = 0,
    ) -> None:
        """Record an object exposing ``prompt_tokens``/``completion_tokens``."""
        self.record(
            role,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cached_tokens=cached_tokens,
        )

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Return deterministic JSON-ready per-role totals."""
        return {
            role: {
                "requests": totals.requests,
                "prompt_tokens": totals.prompt_tokens,
                "completion_tokens": totals.completion_tokens,
                "cached_tokens": totals.cached_tokens,
                "total_tokens": totals.prompt_tokens + totals.completion_tokens,
            }
            for role, totals in sorted(self._totals.items())
        }

    def metrics_fields(self) -> dict[str, object]:
        return {"tokens_by_role": self.snapshot()}
