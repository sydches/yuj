"""HalfLifeContext - append-only transcript with decayed old tool output.

The mode keeps full transcript behavior while the prompt is cheap. Once the
estimated full prompt crosses the activation threshold, it preserves causal
message order but replaces older tool-result payloads with bounded head/tail
views. Full raw output remains in the trace/transcript artifacts.
"""
from __future__ import annotations

from collections.abc import Callable

from ..context import ContextManager, chars_div_4
from ._metadata import (
    HALFLIFE_BUDGET_CONFIG_ATTRS,
    HALFLIFE_CONSTRUCTOR_CONFIG_ATTRS,
    ContextModeMetadata,
)


HALFLIFE_SECTION_ORDER = (
    "append_only_chronological_messages",
    "decayed_tool_results_when_active",
)

HALFLIFE_SECTION_LABELS = {
    "append_only_chronological_messages": "<chronological transcript>",
    "decayed_tool_results_when_active": "<halflife tool-result stubs>",
}


def _message_chars(messages: list[dict]) -> int:
    return sum(len(str(message)) for message in messages)


class HalfLifeContext(ContextManager):
    """Full transcript until pressure, then age-band old tool outputs."""

    def __init__(
        self,
        original_prompt: str | None = None,
        *,
        context_size: int = 0,
        context_limit_tokens: int = 0,
        activation_ratio: float = 0.50,
        verbatim_tool_results: int = 4,
        cap_7_chars: int = 4096,
        cap_15_chars: int = 2048,
        cap_31_chars: int = 1024,
        cap_63_chars: int = 512,
        cap_older_chars: int = 256,
        token_estimator: Callable[[list[dict]], int] = chars_div_4,
    ):
        super().__init__(token_estimator)
        self._messages: list[dict] = []
        self._context_limit_tokens = int(context_limit_tokens or context_size or 0)
        self._activation_ratio = max(0.0, float(activation_ratio))
        self._verbatim_tool_results = max(0, int(verbatim_tool_results))
        self._cap_7_chars = max(0, int(cap_7_chars))
        self._cap_15_chars = max(0, int(cap_15_chars))
        self._cap_31_chars = max(0, int(cap_31_chars))
        self._cap_63_chars = max(0, int(cap_63_chars))
        self._cap_older_chars = max(0, int(cap_older_chars))
        self._turn_count = 0
        self._first_decay_turn: int | None = None
        self._decay_render_count = 0
        self._msg_cache: list[dict] | None = None
        self._tok_cache: int | None = None

    def add_system(self, content: str) -> None:
        self._messages.append({"role": "system", "content": content})
        self._invalidate()

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})
        self._invalidate()

    def add_assistant(self, message: dict) -> None:
        self._messages.append(message)
        self._turn_count += 1
        self._invalidate()

    def add_tool_result(
        self,
        tool_call_id: str,
        content: str,
        *,
        tool_name: str = "",
        cmd_signature: str = "",
        gate_blocked: bool = False,
    ) -> None:
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })
        self._invalidate()

    def get_messages(self) -> list[dict]:
        if self._msg_cache is None:
            self._msg_cache = self._build_messages()
        return self._msg_cache

    def estimate_tokens(self) -> int:
        if self._tok_cache is None:
            self._tok_cache = self._token_estimator(self.get_messages())
        return self._tok_cache

    def message_count(self) -> int:
        return len(self._messages)

    def replace_all_messages(self, new_messages: list[dict]) -> bool:
        self._messages = list(new_messages)
        self._invalidate()
        return True

    def _invalidate(self) -> None:
        self._msg_cache = None
        self._tok_cache = None

    def _decay_active(self, messages: list[dict]) -> tuple[bool, int, int]:
        if self._context_limit_tokens <= 0:
            return False, 0, 0
        threshold = int(self._context_limit_tokens * self._activation_ratio)
        if threshold <= 0:
            return True, self._token_estimator(messages), threshold
        full_tokens = self._token_estimator(messages)
        return full_tokens >= threshold, full_tokens, threshold

    def _cap_for_age(self, age: int) -> tuple[str, int | None]:
        if age < self._verbatim_tool_results:
            return "verbatim", None
        if age <= 7:
            return "cap_7", self._cap_7_chars
        if age <= 15:
            return "cap_15", self._cap_15_chars
        if age <= 31:
            return "cap_31", self._cap_31_chars
        if age <= 63:
            return "cap_63", self._cap_63_chars
        return "cap_older", self._cap_older_chars

    @staticmethod
    def _fit_head_tail(content: str, cap: int, marker: str) -> str:
        if cap <= 0:
            return marker[: max(0, cap)]
        if len(content) <= cap:
            return content
        if cap <= len(marker) + 8:
            return marker[:cap]
        body_budget = cap - len(marker) - 2
        head_budget = max(1, body_budget // 2)
        tail_budget = max(1, body_budget - head_budget)
        head = content[:head_budget]
        tail = content[-tail_budget:]
        return f"{head}\n{marker}\n{tail}"

    def _decay_tool_content(
        self,
        content: str,
        *,
        age: int,
        tier: str,
        cap: int | None,
    ) -> tuple[str, int]:
        if cap is None or len(content) <= cap:
            return content, 0
        omitted = len(content) - cap
        marker = (
            f"[halflife: omitted {omitted} chars from older tool result; "
            f"age={age}; tier={tier}; full output remains in trace/transcript artifacts]"
        )
        decayed = self._fit_head_tail(content, cap, marker)
        return decayed, max(0, len(content) - len(decayed))

    def _build_messages(self) -> list[dict]:
        visible_messages = self._filter_expired_thought_messages(self._messages)
        active, full_tokens, threshold = self._decay_active(visible_messages)
        if not active:
            return visible_messages

        if self._first_decay_turn is None:
            self._first_decay_turn = self._turn_count
        self._decay_render_count += 1

        tool_indices = [
            index for index, message in enumerate(visible_messages)
            if message.get("role") == "tool"
        ]
        tool_age_by_index = {
            index: (len(tool_indices) - 1 - position)
            for position, index in enumerate(tool_indices)
        }
        tier_counts: dict[str, int] = {}
        tier_chars: dict[str, int] = {}
        saved_chars = 0
        decayed_messages: list[dict] = []

        for index, message in enumerate(visible_messages):
            if message.get("role") != "tool":
                decayed_messages.append(message)
                continue
            content = message.get("content")
            if not isinstance(content, str):
                decayed_messages.append(message)
                continue
            age = tool_age_by_index.get(index, 0)
            tier, cap = self._cap_for_age(age)
            new_content, saved = self._decay_tool_content(
                content, age=age, tier=tier, cap=cap,
            )
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            tier_chars[tier] = tier_chars.get(tier, 0) + len(new_content)
            saved_chars += saved
            if new_content == content:
                decayed_messages.append(message)
            else:
                replacement = dict(message)
                replacement["content"] = new_content
                decayed_messages.append(replacement)

        if saved_chars > 0:
            from ..savings import get_ledger
            get_ledger().record(
                bucket="context_projection",
                layer="context_strategy",
                mechanism="halflife_decay",
                input_chars=_message_chars(visible_messages),
                output_chars=_message_chars(decayed_messages),
                measure_type="exact",
                ctx={
                    "turn_count": self._turn_count,
                    "full_tokens_est": full_tokens,
                    "activation_threshold_tokens": threshold,
                    "context_limit_tokens": self._context_limit_tokens,
                    "first_decay_turn": self._first_decay_turn,
                    "decay_render_count": self._decay_render_count,
                    "tool_result_count": len(tool_indices),
                    "tier_counts": tier_counts,
                    "tier_chars": tier_chars,
                },
            )
        return decayed_messages


CONTEXT_MODE = "halflife"
CONTEXT_CLASS = HalfLifeContext
CONTEXT_METADATA = ContextModeMetadata(
    cli_order=12,
    message_shape="append-only transcript with age-decayed old tool results",
    state_source="append_only_messages",
    source_type="append_log",
    normal_prompt_sources=("in_memory_append_log",),
    section_order=HALFLIFE_SECTION_ORDER,
    section_labels=HALFLIFE_SECTION_LABELS,
    file_freshness="append_only+decay",
    injection_support="verbatim",
    budget_config_attrs=HALFLIFE_BUDGET_CONFIG_ATTRS,
    constructor_config_attrs=HALFLIFE_CONSTRUCTOR_CONFIG_ATTRS,
)
