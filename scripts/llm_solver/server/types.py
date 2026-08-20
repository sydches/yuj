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


@dataclass(frozen=True)
class TurnResult:
    """Canonical turn result. Access via attributes; never iterate."""

    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str  # "stop" | "tool_calls" | "length"
    usage: Usage
