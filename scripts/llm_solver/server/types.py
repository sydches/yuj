"""Canonical output types for the server layer."""
from dataclasses import dataclass
from typing import NamedTuple


class ToolCall(NamedTuple):
    id: str
    name: str
    arguments: dict


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
