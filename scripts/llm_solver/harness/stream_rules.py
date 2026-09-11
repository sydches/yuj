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
    """Bound visible prose using backend text tokens or an explicit estimate."""

    def __init__(self, *, context_size: int, fraction: float, message: str, text_counter=None):
        self.policy_limit_tokens = max(1, int(context_size * fraction))
        self.limit_tokens = self.policy_limit_tokens
        self.text_counter = text_counter
        self.text = ""
        self.tokens = 0
        self.next_count_chars = self.limit_tokens
        self.measurement = {}
        self.policy_limit_chars = max(1, int(context_size * fraction * 4))
        self.limit_chars = self.policy_limit_chars
        self.message = message
        self.chars = 0
        self.completion_tokens = None
        self.request_model = None

    def prepare_request(self, *, completion_tokens, model=None) -> None:
        """Bind after transport budget resolution; never reset attempt usage.

        A length continuation has its own request cap but shares the attempt's
        narration policy. Retokenize its retained prose before rebinding.
        Missing limits stay unknown; four characters per token is fallback only.
        """
        self.completion_tokens = (
            completion_tokens if type(completion_tokens) is int
            and completion_tokens > 0 else None
        )
        self.request_model = model if isinstance(model, str) and model else None
        if self.text_counter is not None and self.text:
            self._count_text()
        self.limit_tokens = self.policy_limit_tokens
        if self.completion_tokens is not None:
            self.limit_tokens = min(self.limit_tokens, self.tokens + self.completion_tokens)
        self.next_count_chars = min(self.next_count_chars,
                                   self.chars + max(1, self.limit_tokens - self.tokens))
        self.limit_chars = self.policy_limit_chars
        if self.completion_tokens is not None:
            self.limit_chars = min(
                self.policy_limit_chars, self.chars + self.completion_tokens * 4,
            )

    def _count_text(self):
        from .time_budget import remaining_run_seconds
        value = self.text_counter.count(self.text, model=self.request_model,
                                        remaining_seconds=remaining_run_seconds())
        self.measurement = dict(self.text_counter.last)
        if value is None:
            self.text_counter = None
        else:
            self.tokens = value
        return value

    def observe(self, delta) -> None:
        if delta.source != "text":
            return
        self.chars += len(delta.delta or "")
        self.text += delta.delta or ""
        if self.text_counter is not None:
            if self.chars < self.next_count_chars:
                return
            count = self._count_text()
            if count is not None and count <= self.limit_tokens:
                # Batch counting by remaining token allowance, not stream chunks.
                self.next_count_chars = self.chars + max(1, self.limit_tokens - count)
                return
        if self.text_counter is None and self.chars <= self.limit_chars:
            return
        from ..server._streaming import StreamRuleInterrupt
        raise StreamRuleInterrupt(({
            "rule": "autonomous_narration",
            "kind": "narration_limit",
            "scope": "text",
            "offset": self.chars if self.text_counter is not None else self.limit_chars,
            "observed_chars": self.chars,
            "measurement_basis": "backend_text_tokens" if self.text_counter is not None else "character_estimate",
            "observed_text_tokens": self.tokens if self.text_counter is not None else None,
            "limit_text_tokens": self.limit_tokens,
            "text_count": self.measurement,
            "characters_per_estimated_token": None if self.text_counter is not None else 4,
            "policy_limit_chars": self.policy_limit_chars,
            "request_completion_tokens": self.completion_tokens,
            "request_budget_basis": (
                "prepared_request_max_tokens" if self.completion_tokens is not None
                else "unknown"
            ),
            "request_model": self.request_model,
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
