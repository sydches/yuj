"""Pure policy helpers for per-request model controls.

The transport client owns HTTP and SDK behavior.  This module owns the
validated, provider-neutral decisions that the client applies to one request.
Keeping the policy here makes profile/config integration testable without a
live model server.
"""
from __future__ import annotations

import copy
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
