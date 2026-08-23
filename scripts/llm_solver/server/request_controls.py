"""Pure policy helpers for per-request model controls.

The transport client owns HTTP and SDK behavior.  This module owns the
validated, provider-neutral decisions that the client applies to one request.
Keeping the policy here makes profile/config integration testable without a
live model server.
"""
from __future__ import annotations

import copy
import hashlib
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


THINKING_LEVELS: tuple[str, ...] = (
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
"""Canonical user-facing thinking-level ladder, lowest to highest."""

# ``on`` is a profile capability, not a user-facing effort.  It lets a model
# that only supports a boolean switch participate in the effort policy: every
# requested non-off effort maps to the profile's generic ``on`` body.
PROFILE_THINKING_LEVELS: frozenset[str] = frozenset((*THINKING_LEVELS, "on"))

DEFAULT_REASONING_LEVELS: dict[str, dict[str, object]] = {
    "off": {"chat_template_kwargs": {"enable_thinking": False}},
    "on": {"chat_template_kwargs": {"enable_thinking": True}},
}
"""Boolean fallback for the supported legacy/no-profile request path."""

CACHE_RETENTION_LEVELS: tuple[str, ...] = ("off", "session")
"""llama-server cache-retention modes supported by the request layer."""

# Request extras are merged into the SDK's ``extra_body``.  They must not be
# able to replace the transport-owned request envelope.
_RESERVED_REQUEST_FIELDS: frozenset[str] = frozenset(
    {
        "extra_body",
        "max_tokens",
        "messages",
        "model",
        "stream",
        "stream_options",
        "tool_choice",
        "tools",
    }
)


class RequestControlError(ValueError):
    """A request-control setting or profile mapping is invalid."""


@dataclass(frozen=True)
class ThinkingLevelResolution:
    """One validated thinking-level decision for a run or request."""

    requested_level: str
    effective_level: str
    request_extra: Mapping[str, Any]
    supported_levels: tuple[str, ...]
    clamped: bool

    def trace_fields(self) -> dict[str, object]:
        """Return additive fields suitable for ``session_start``."""
        fields: dict[str, object] = {"thinking_level": self.effective_level}
        if self.clamped:
            fields["thinking_level_requested"] = self.requested_level
        return fields

    def provenance_fields(self) -> dict[str, object]:
        """Return explicit requested/effective run-provenance fields."""
        return {
            "thinking_level_requested": self.requested_level,
            "thinking_level_effective": self.effective_level,
            "thinking_level_clamped": self.clamped,
        }


@dataclass(frozen=True)
class CacheRequestResolution:
    """Explicit llama-server prompt-cache fields for one request."""

    cache_retention: str
    slot_id: int | None
    cache_prompt: bool
    side_request: bool
    request_extra: Mapping[str, Any]

    def trace_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {
            "cache_retention": self.cache_retention,
            "cache_prompt": self.cache_prompt,
        }
        if self.slot_id is not None:
            fields["cache_slot"] = self.slot_id
        if self.side_request:
            fields["cache_side_request"] = True
        return fields


@dataclass(frozen=True)
class CacheObservation:
    """Cache telemetry extracted from one model-server response."""

    prompt_tokens: int
    cached_tokens: int | None
    hit_ratio: float | None
    source: str

    def trace_fields(self) -> dict[str, object]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_hit_ratio": (
                round(self.hit_ratio, 6) if self.hit_ratio is not None else None
            ),
        }


def _validate_json_value(value: object, path: str) -> None:
    """Reject values that cannot be represented in a JSON request body."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RequestControlError(f"{path} must be a finite JSON number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise RequestControlError(
                    f"{path} keys must be non-empty strings, got {key!r}"
                )
            _validate_json_value(child, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]")
        return
    raise RequestControlError(
        f"{path} must contain only JSON-compatible values, got {type(value).__name__}"
    )


def validate_request_extra(
    request_extra: Mapping[str, object],
    *,
    path: str = "request_extra",
    allow_empty: bool = True,
) -> dict[str, Any]:
    """Validate and defensively copy an OpenAI-compatible extra request body.

    The returned mapping is safe to place under the OpenAI SDK's
    ``extra_body`` keyword, whose members are merged into the JSON body sent to
    the compatible endpoint.
    """
    if not isinstance(request_extra, Mapping):
        raise RequestControlError(f"{path} must be a table/mapping")
    if not request_extra and not allow_empty:
        raise RequestControlError(f"{path} must define at least one request field")

    copied: dict[str, Any] = {}
    for key, value in request_extra.items():
        if not isinstance(key, str) or not key:
            raise RequestControlError(f"{path} keys must be non-empty strings")
        if key in _RESERVED_REQUEST_FIELDS:
            raise RequestControlError(
                f"{path}.{key} is transport-owned and cannot be set by a profile"
            )
        _validate_json_value(value, f"{path}.{key}")
        copied[key] = copy.deepcopy(value)
    return copied


def validate_reasoning_levels(
    reasoning_levels: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, Any]]:
    """Validate a profile's ``reasoning_levels`` request-body mapping.

    Profile keys may use the canonical effort ladder and the special generic
    capability ``on``.  Every entry is a complete per-request extra body.
    """
    if not isinstance(reasoning_levels, Mapping) or not reasoning_levels:
        raise RequestControlError(
            "profile.reasoning_levels must be a non-empty table/mapping"
        )

    validated: dict[str, dict[str, Any]] = {}
    for raw_level, request_extra in reasoning_levels.items():
        if not isinstance(raw_level, str):
            raise RequestControlError("profile.reasoning_levels keys must be strings")
        level = raw_level.strip().lower()
        if level not in PROFILE_THINKING_LEVELS:
            allowed = ", ".join((*THINKING_LEVELS, "on"))
            raise RequestControlError(
                f"unknown profile reasoning level {raw_level!r}; expected one of: {allowed}"
            )
        if level in validated:
            raise RequestControlError(
                f"duplicate profile reasoning level after normalization: {raw_level!r}"
            )
        validated[level] = validate_request_extra(
            request_extra,
            path=f"profile.reasoning_levels.{level}",
            allow_empty=False,
        )
    return validated


def normalize_thinking_level(level: str) -> str:
    """Return a canonical user-facing thinking level or raise."""
    if not isinstance(level, str) or not level.strip():
        raise RequestControlError("model.thinking_level must be a non-empty string")
    normalized = level.strip().lower()
    if normalized not in THINKING_LEVELS:
        raise RequestControlError(
            "model.thinking_level must be one of: " + ", ".join(THINKING_LEVELS)
        )
    return normalized


def _clamp_thinking_level(
    requested: str,
    supported: Mapping[str, Mapping[str, Any]],
) -> str:
    """Choose the closest non-exceeding concrete effort when possible."""
    if requested in supported:
        return requested

    concrete_positive = [
        level for level in THINKING_LEVELS[1:] if level in supported
    ]
    if requested != "off" and concrete_positive:
        requested_index = THINKING_LEVELS.index(requested)
        at_or_below = [
            level
            for level in concrete_positive
            if THINKING_LEVELS.index(level) <= requested_index
        ]
        # Do not silently spend more reasoning than requested when a lower
        # supported level exists.  If all supported levels are higher, select
        # the lowest positive level so a positive request stays positive.
        return at_or_below[-1] if at_or_below else concrete_positive[0]

    if requested != "off" and "on" in supported:
        return "on"
    if "off" in supported:
        return "off"
    if concrete_positive:
        return concrete_positive[0]
    # Validation guarantees at least one entry.  The only remaining key is
    # the generic boolean capability ``on``.
    return "on"


def resolve_thinking_level(
    requested_level: str,
    reasoning_levels: Mapping[str, Mapping[str, object]],
    *,
    logger: logging.Logger | None = None,
) -> ThinkingLevelResolution:
    """Resolve one requested effort against a model profile's capabilities.

    Unsupported levels clamp deterministically and emit a warning.  The
    returned request mapping is a defensive copy so callers cannot mutate the
    loaded profile.
    """
    requested = normalize_thinking_level(requested_level)
    supported = validate_reasoning_levels(reasoning_levels)
    effective = _clamp_thinking_level(requested, supported)
    clamped = effective != requested
    if clamped:
        (logger or log).warning(
            "thinking level %s is unsupported by this profile; using %s "
            "(supported: %s)",
            requested,
            effective,
            ", ".join(supported),
        )
    return ThinkingLevelResolution(
        requested_level=requested,
        effective_level=effective,
        request_extra=copy.deepcopy(supported[effective]),
        supported_levels=tuple(supported),
        clamped=clamped,
    )


def merge_request_extra(*parts: Mapping[str, object] | None) -> dict[str, Any]:
    """Merge validated request-extra mappings from left to right.

    Later policy layers intentionally win.  This lets a side-request policy
    force ``cache_prompt=false`` even when the normal session policy enables
    cache retention.
    """
    merged: dict[str, Any] = {}
    for index, part in enumerate(parts):
        if part is None:
            continue
        validated = validate_request_extra(part, path=f"request_extra[{index}]")
        merged.update(validated)
    return merged


def attach_request_extra(
    payload: Mapping[str, object],
    request_extra: Mapping[str, object] | None,
) -> dict[str, Any]:
    """Return SDK call kwargs with custom JSON fields under ``extra_body``.

    ``openai`` merges ``extra_body`` into the actual HTTP JSON body.  Existing
    caller extras are preserved, with the explicit per-request policy taking
    precedence.  Neither input mapping is mutated.
    """
    if not isinstance(payload, Mapping):
        raise RequestControlError("request payload must be a mapping")
    result = copy.deepcopy(dict(payload))
    if not request_extra:
        return result

    existing = result.get("extra_body")
    if existing is not None and not isinstance(existing, Mapping):
        raise RequestControlError("request payload extra_body must be a mapping")
    result["extra_body"] = merge_request_extra(existing, request_extra)
    return result


def normalize_cache_affinity(value: object) -> int:
    """Return the configured number of affinity slots (zero disables it).

    ``true`` is accepted as the common single-slot shorthand.  A positive
    integer enables deterministic session-to-slot hashing across that many
    server slots.
    """
    if value is None or value is False:
        return 0
    if value is True:
        return 1
    if not isinstance(value, int) or value < 0:
        raise RequestControlError(
            "server.cache_affinity must be false, true, zero, or a positive slot count"
        )
    return value


def normalize_cache_retention(value: str) -> str:
    """Validate llama-server prompt-cache retention policy."""
    if not isinstance(value, str) or not value.strip():
        raise RequestControlError("server.cache_retention must be a non-empty string")
    normalized = value.strip().lower()
    if normalized not in CACHE_RETENTION_LEVELS:
        raise RequestControlError(
            "server.cache_retention must be one of: "
            + ", ".join(CACHE_RETENTION_LEVELS)
        )
    return normalized


def derive_cache_slot(session_id: str, cache_affinity: object) -> int | None:
    """Map a stable session identifier to a valid llama-server slot ID."""
    slot_count = normalize_cache_affinity(cache_affinity)
    if slot_count == 0:
        return None
    if not isinstance(session_id, str) or not session_id:
        raise RequestControlError(
            "session_id must be a non-empty string when cache affinity is enabled"
        )
    digest = hashlib.sha256(session_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % slot_count


def resolve_cache_request(
    *,
    session_id: str,
    cache_affinity: object,
    cache_retention: str,
    side_request: bool = False,
) -> CacheRequestResolution:
    """Build explicit llama-server cache fields for one model request.

    Side requests never select the main session's slot and force
    ``cache_prompt=false`` so compaction, handoff, and classifier prompts do
    not replace or extend the task conversation's reusable prefix.
    """
    retention = normalize_cache_retention(cache_retention)
    affinity_slots = normalize_cache_affinity(cache_affinity)
    slot_id = None
    if not side_request:
        slot_id = derive_cache_slot(session_id, affinity_slots)

    cache_prompt = retention == "session" and not side_request
    request_extra: dict[str, object] = {"cache_prompt": cache_prompt}
    if slot_id is not None:
        request_extra["id_slot"] = slot_id
    return CacheRequestResolution(
        cache_retention=retention,
        slot_id=slot_id,
        cache_prompt=cache_prompt,
        side_request=side_request,
        request_extra=validate_request_extra(request_extra),
    )


def apply_request_controls(
    payload: Mapping[str, object],
    *,
    session_id: str,
    server_request_extra: Mapping[str, object],
    cache_affinity: object,
    cache_retention: str,
    side_request: bool,
    policy_extra: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Merge configured/per-request extras with cache policy last."""
    cache = resolve_cache_request(
        session_id=session_id,
        cache_affinity=cache_affinity,
        cache_retention=cache_retention,
        side_request=side_request,
    )
    request = copy.deepcopy(dict(payload))
    existing = dict(request.get("extra_body") or {})
    configured = dict(server_request_extra or {})
    # Cache fields are policy-owned. Removing earlier copies also guarantees
    # that a side request cannot retain an id_slot merely because the final
    # side policy intentionally omits it.
    for field in ("cache_prompt", "id_slot"):
        existing.pop(field, None)
        configured.pop(field, None)
    if existing or "extra_body" in request:
        request["extra_body"] = existing
    merged = merge_request_extra(
        configured,
        policy_extra,
        cache.request_extra,
    )
    return attach_request_extra(request, merged)


def validate_cache_miss_warn_ratio(value: object) -> float:
    """Validate a cache-hit ratio threshold in the closed interval [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestControlError(
            "server.cache_miss_warn_ratio must be a number between 0 and 1"
        )
    ratio = float(value)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise RequestControlError(
            "server.cache_miss_warn_ratio must be a number between 0 and 1"
        )
    return ratio


def _field(value: object, name: str) -> object:
    """Read a normal, mapping, or Pydantic-extra response field."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(name)
    direct = getattr(value, name, None)
    if direct is not None:
        return direct
    model_extra = getattr(value, "model_extra", None)
    if isinstance(model_extra, Mapping):
        return model_extra.get(name)
    return None


def _token_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def extract_cache_observation(response: object) -> CacheObservation | None:
    """Extract total/cached prompt tokens from OpenAI usage or llama timings.

    Current llama-server chat responses expose cached tokens through OpenAI
    usage details and may also expose ``timings.cache_n`` with
    ``timings.prompt_n`` (the processed suffix).  Missing cache telemetry stays
    explicit as ``None`` rather than being misreported as a cache miss.
    """
    usage = _field(response, "usage")
    usage_prompt = _token_count(_field(usage, "prompt_tokens"))
    if usage_prompt is None:
        usage_prompt = _token_count(_field(usage, "input_tokens"))

    details = _field(usage, "prompt_tokens_details")
    if details is None:
        details = _field(usage, "input_tokens_details")
    usage_cached = _token_count(_field(details, "cached_tokens"))

    timings = _field(response, "timings")
    timing_cached = _token_count(_field(timings, "cache_n"))
    timing_processed = _token_count(_field(timings, "prompt_n"))
    timing_total = None
    if timing_processed is not None and timing_cached is not None:
        timing_total = timing_processed + timing_cached

    cached = usage_cached if usage_cached is not None else timing_cached
    prompt = usage_prompt if usage_prompt is not None else timing_total
    source_parts: list[str] = []
    if usage_prompt is not None or usage_cached is not None:
        source_parts.append("usage")
    if timing_processed is not None or timing_cached is not None:
        source_parts.append("timings")

    if prompt is None:
        return None
    if cached is not None and cached > prompt:
        # Some non-OpenAI-compatible surfaces report only the processed suffix
        # as prompt_n.  Prefer the unambiguous timings total when available;
        # otherwise keep token telemetry unknown rather than fabricate a ratio.
        if timing_total is not None and timing_total >= cached:
            prompt = timing_total
        else:
            log.warning(
                "ignoring inconsistent prompt-cache telemetry: cached=%d prompt=%d",
                cached,
                prompt,
            )
            cached = None

    hit_ratio = None
    if cached is not None and prompt > 0:
        hit_ratio = cached / prompt
    return CacheObservation(
        prompt_tokens=prompt,
        cached_tokens=cached,
        hit_ratio=hit_ratio,
        source="+".join(source_parts) or "unknown",
    )


def usage_from_response(response: object):
    """Return canonical Usage with additive cache telemetry."""
    from .types import Usage

    usage = _field(response, "usage")
    completion_tokens = int(_field(usage, "completion_tokens") or 0)
    observation = extract_cache_observation(response)
    if observation is None:
        return Usage(
            prompt_tokens=int(_field(usage, "prompt_tokens") or 0),
            completion_tokens=completion_tokens,
        )
    return Usage(
        prompt_tokens=observation.prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=observation.cached_tokens,
        cache_hit_ratio=observation.hit_ratio,
    )


def warn_on_cache_miss(
    observation: CacheObservation | None,
    *,
    warn_ratio: object,
    prior_turns: int,
    logger: logging.Logger | None = None,
) -> bool:
    """Log a low cache-hit warning after the first turn and return whether sent."""
    threshold = validate_cache_miss_warn_ratio(warn_ratio)
    if isinstance(prior_turns, bool) or not isinstance(prior_turns, int) or prior_turns < 0:
        raise RequestControlError("prior_turns must be a non-negative integer")
    if (
        threshold <= 0
        or prior_turns == 0
        or observation is None
        or observation.hit_ratio is None
        or observation.hit_ratio >= threshold
    ):
        return False
    (logger or log).warning(
        "prompt cache hit ratio %.3f is below %.3f "
        "(cached_tokens=%d prompt_tokens=%d)",
        observation.hit_ratio,
        threshold,
        observation.cached_tokens,
        observation.prompt_tokens,
    )
    return True


class CacheUsageAccumulator:
    """Aggregate an observed session cache-hit ratio without averaging ratios."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.requests_observed = 0
        self.requests_unobserved = 0

    def record(self, observation: CacheObservation | None) -> None:
        if observation is None or observation.cached_tokens is None:
            self.requests_unobserved += 1
            return
        self.requests_observed += 1
        self.prompt_tokens += observation.prompt_tokens
        self.cached_tokens += observation.cached_tokens

    @property
    def hit_ratio(self) -> float | None:
        if self.prompt_tokens <= 0:
            return None
        return self.cached_tokens / self.prompt_tokens

    def snapshot(self) -> dict[str, object]:
        ratio = self.hit_ratio
        return {
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_hit_ratio": round(ratio, 6) if ratio is not None else None,
            "requests_observed": self.requests_observed,
            "requests_unobserved": self.requests_unobserved,
        }

    def metrics_fields(self) -> dict[str, object]:
        return {"prompt_cache": self.snapshot()}
