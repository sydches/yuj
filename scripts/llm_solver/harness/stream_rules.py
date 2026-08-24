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


__all__ = [
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
