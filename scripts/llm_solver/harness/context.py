"""Context management protocol for the agentic loop.

Defines ContextManager (the interface) and FullTranscript (the default
append-only implementation that preserves current behavior exactly).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable


def chars_div_4(msgs: list[dict]) -> int:
    """Default token estimator: total chars across all messages / 4."""
    return sum(len(str(m)) for m in msgs) // 4


class ContextManager(ABC):
    """Interface for managing conversation context sent to the model.

    A ContextManager controls which messages are stored and which are
    shipped to the model on each turn.  Implementations may prune,
    summarize, or reorder messages — the only contract is on
    ``get_messages()``.

    **Contract for ``get_messages()``:**
    - Returns a ``list[dict]`` suitable for ``client.chat(messages, ...)``.
    - The first element MUST be ``{"role": "system", ...}`` if a system
      message was added via ``add_system()``.
    - Message order must preserve causal dependencies: every tool-result
      message must follow the assistant message that issued the tool call.

    **Accuracy of ``estimate_tokens()``:**
    - Under-estimation risks context overflow (API rejects the request or
      the model sees a truncated window).
    - Over-estimation causes premature session rotation (wasted context
      budget, unnecessary resume overhead).
    - The default heuristic (chars / 4) is deliberately cheap and tends to
      over-estimate; callers that need precision should inject a proper
      tokenizer.

    **Implementing a new strategy:**
    Subclass ``ContextManager`` and implement all abstract methods.  The
    ``token_estimator`` callable is available as ``self._token_estimator``
    for use in ``estimate_tokens()``.
    """

    def __init__(self, token_estimator: Callable[[list[dict]], int] = chars_div_4):
        self._token_estimator = token_estimator

    @abstractmethod
    def add_system(self, content: str) -> None:
        """Append a system message."""

    @abstractmethod
    def add_user(self, content: str) -> None:
        """Append a user message."""

    @abstractmethod
    def add_assistant(self, message: dict) -> None:
        """Append an assistant message (may contain tool_calls)."""

    @abstractmethod
    def add_tool_result(self, tool_call_id: str, content: str, *, tool_name: str = "", cmd_signature: str = "", gate_blocked: bool = False) -> None:
        """Append a tool result message."""

    @abstractmethod
    def get_messages(self) -> list[dict]:
        """Return the message list to send to the model."""

    @abstractmethod
    def estimate_tokens(self) -> int:
        """Estimate total token count of current messages."""

    @abstractmethod
    def message_count(self) -> int:
        """Return the number of messages currently stored."""

    def replace_all_messages(self, new_messages: list[dict]) -> bool:
        """Replace the internal append-log with the supplied list.

        Used by the harness to persist mid-session compaction (the loop
        renders a digest + recent verbatim pair, then needs the strategy
        to forget the old append log). Default returns False — strategies
        that cannot compact opt out and the caller no-ops.

        Strategies that override MUST also clear any per-message cache
        they maintain (msg cache, tok cache) and return True.

        Replaces the previous attribute-poke pattern in
        ``Session._maybe_compact_messages`` (`ctx._all_messages = list(new)`
        wrapped in ``try/except AttributeError``), which reached past the
        ABC and silently no-op'd when private names changed.
        """
        return False


class FullTranscript(ContextManager):
    """Append-only context — ships every message, no pruning.

    This is a transparent wrapper that preserves the exact behavior of
    the previous ``Session.messages: list[dict]`` implementation.
    """

    def __init__(
        self,
        original_prompt: str | None = None,
        token_estimator: Callable[[list[dict]], int] = chars_div_4,
    ):
        super().__init__(token_estimator)
        self._messages: list[dict] = []
        # Token-count cache. Invalidated on every mutation. get_messages is
        # already O(1) (returns the stored list by reference) so it does not
        # need its own cache; estimate_tokens scans all message bytes and is
        # called once per turn — caching eliminates the redundant scan when
        # estimate_tokens is invoked more than once between mutations.
        self._tok_cache: int | None = None

    def add_system(self, content: str) -> None:
        self._messages.append({"role": "system", "content": content})
        self._tok_cache = None

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})
        self._tok_cache = None

    def add_assistant(self, message: dict) -> None:
        self._messages.append(message)
        self._tok_cache = None

    def add_tool_result(self, tool_call_id: str, content: str, *, tool_name: str = "", cmd_signature: str = "", gate_blocked: bool = False) -> None:
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })
        self._tok_cache = None

    def get_messages(self) -> list[dict]:
        return self._messages

    def replace_all_messages(self, new_messages: list[dict]) -> bool:
        self._messages = list(new_messages)
        self._tok_cache = None
        return True

    def estimate_tokens(self) -> int:
        if self._tok_cache is None:
            self._tok_cache = self._token_estimator(self._messages)
        return self._tok_cache

    def message_count(self) -> int:
        return len(self._messages)
