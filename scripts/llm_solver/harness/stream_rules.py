"""Validated mid-stream rule loader and session runtime.

Rule files live under ``.harness/stream_rules/*.md`` and use TOML
frontmatter between ``+++`` fences. The runtime is session-scoped: stream
chunks and same-turn retries never advance repeat control.
"""
from ._stream_rule_loader import (
    LoadedStreamRules,
    StreamRule,
    StreamRuleError,
    StreamRuleScope,
    load_stream_rules,
    parse_stream_rule,
)
from ._stream_rule_runtime import (
    StreamRuleRuntime,
    format_interrupt_fragment,
    format_tool_reminder,
)


class NarrationBudget:
    """Cheap, per-attempt text bound for the autonomous reply contract."""

    def __init__(self, *, context_size: int, fraction: float, message: str):
        self.limit_chars = max(1, int(context_size * fraction * 4))
        self.message = message
        self.chars = 0

    def observe(self, delta) -> None:
        if delta.source != "text":
            return
        self.chars += len(delta.delta or "")
        if self.chars <= self.limit_chars:
            return
        from ..server._streaming import StreamRuleInterrupt
        raise StreamRuleInterrupt(({
            "rule": "autonomous_narration",
            "kind": "narration_limit",
            "scope": "text",
            "offset": self.limit_chars,
            "observed_chars": self.chars,
            "interrupt": True,
            "body": self.message,
        },))


__all__ = [
    "NarrationBudget",
    "LoadedStreamRules",
    "StreamRule",
    "StreamRuleError",
    "StreamRuleRuntime",
    "StreamRuleScope",
    "format_interrupt_fragment",
    "format_tool_reminder",
    "load_stream_rules",
    "parse_stream_rule",
]
