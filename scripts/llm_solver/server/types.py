"""Canonical output types for the server layer."""
from dataclasses import dataclass
from typing import Any, NamedTuple


@dataclass(frozen=True)
class ImageInput:
    """One validated raster image bound to an assistant request."""

    media_type: str
    data: bytes


class ToolCall(NamedTuple):
    id: str
    name: str
    arguments: dict
    # Provider-owned fields that must accompany this call when it is replayed.
    # The harness stores and returns the mapping without interpreting it.
    extra_content: dict[str, Any] | None = None


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None = None
    cache_hit_ratio: float | None = None
    prompt_tokens_known: bool = True
    completion_tokens_known: bool = True


@dataclass(frozen=True)
class TurnResult:
    """Canonical turn result. Access via attributes; never iterate."""

    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str  # "stop" | "tool_calls" | "length"
    usage: Usage


@dataclass(frozen=True)
class SideRequestResult:
    """No-tool model response used by harness-owned side requests."""

    content: str
    usage: Usage
