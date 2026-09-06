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


def inject_interrupt_fragments(session, records, *, turn, mark_runtime=False):
    """Deliver recovery advice and record its existing transformation surface."""
    inserted = "\n\n".join(format_interrupt_fragment(record) for record in records)
    session.context.add_injected_fragment(inserted)
    from .savings import get_ledger
    get_ledger().record_transform(
        bucket="stream_rule_intervention", layer="harness",
        mechanism="retry_interrupt_fragment", before="", after=inserted,
        surface="injected_message", change_count=len(records),
        ctx={"rules": [str(record.get("rule") or "") for record in records],
             "delivery": "retry"},
    )
    runtime = getattr(session, "_stream_rule_runtime", None)
    if runtime is not None and mark_runtime:
        runtime.mark_injected(records, turn=turn)
    session._record_stream_rule_injection(records, turn=turn, delivery="retry")


__all__ = [
    "NarrationBudget",
    "LoadedStreamRules",
    "StreamRule",
    "StreamRuleError",
    "StreamRuleRuntime",
    "StreamRuleScope",
    "format_interrupt_fragment",
    "format_tool_reminder",
    "inject_interrupt_fragments",
    "load_stream_rules",
    "parse_stream_rule",
]
