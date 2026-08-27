"""Context management protocol for the agentic loop.

Defines ContextManager (the interface) and FullTranscript (the default
append-only implementation that preserves current behavior exactly).
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable

from .thoughts import (
    filter_expired_thought_messages,
    redact_expired_thought_state,
    thought_is_expired,
)


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
        # Configured by the session factory after construction so every
        # context family follows one scratchpad-retention contract without
        # growing a duplicate constructor argument.
        self._think_keep_turns: int | None = None
        self._context_session_number: int | None = None

    def configure_thought_retention(
        self, keep_turns: int, *, session_number: int,
    ) -> None:
        """Apply the model-facing thought window without touching raw logs."""
        if isinstance(keep_turns, bool) or not isinstance(keep_turns, int):
            raise TypeError("think_keep_turns must be an integer")
        if keep_turns < 0:
            raise ValueError("think_keep_turns must be non-negative")
        self._think_keep_turns = keep_turns
        self._context_session_number = session_number
        if hasattr(self, "_msg_cache"):
            self._msg_cache = None
        if hasattr(self, "_tok_cache"):
            self._tok_cache = None

    def _filter_expired_thought_messages(
        self, messages: list[dict],
    ) -> list[dict]:
        filtered = filter_expired_thought_messages(
            messages, self._think_keep_turns,
        )
        if filtered is not messages:
            from .savings import get_ledger, serialize_messages
            get_ledger().record_transform(
                bucket="context_projection",
                layer="context_strategy",
                mechanism="think_retention_window",
                before=serialize_messages(messages),
                after=serialize_messages(filtered),
                surface="context_render",
                ctx={
                    "keep_turns": self._think_keep_turns,
                    "encoding": "message_list_json_utf8_v1",
                },
            )
        return filtered

    def _thought_turn_expired(self, turn: object) -> bool:
        return thought_is_expired(
            turn,
            current_turn=int(getattr(self, "_turn_count", 0)),
            keep_turns=self._think_keep_turns,
        )

    def _redact_expired_thought_state(self, data: dict) -> dict:
        # state.json turn numbers are zero-based; the in-memory context
        # counter is the number of assistant turns already ingested.
        current_turn = max(0, int(getattr(self, "_turn_count", 0)) - 1)
        return redact_expired_thought_state(
            data,
            current_turn=current_turn,
            keep_turns=self._think_keep_turns,
            current_session=self._context_session_number,
        )

    @abstractmethod
    def add_system(self, content: str) -> None:
        """Append a system message."""

    @abstractmethod
    def add_user(self, content: str) -> None:
        """Append a user message."""

    def add_injected_fragment(self, content: str) -> None:
        """Add a harness fragment that must reach the next model request.

        Append-log strategies naturally preserve it as a user message.
        Projection strategies override this method to retain a transient copy
        inside their synthesized user payload.
        """
        self.add_user(content)

    def consume_injected_fragments(self) -> None:
        """Mark transient fragments delivered after a successful request."""
        return

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

    def snapshot_messages(self) -> list[dict]:
        """Return a detached copy of the strategy's canonical append log.

        Projecting strategies override this method because ``get_messages``
        may return a synthesized model request rather than their append log.
        """
        return copy.deepcopy(list(self.get_messages()))

    def rewind_messages(self, new_messages: list[dict]) -> bool:
        """Replace history at a protocol-safe checkpoint boundary.

        Simple transcript strategies can use their ordinary replacement path.
        Strategies with derived working sets or recent-result windows override
        this method to rebuild those dependent views from the retained prefix.
        """
        return self.replace_all_messages(new_messages)

    def get_history_messages(self) -> list[dict]:
        """Return the canonical append-log behind the model projection.

        Rewind snapshots need both this lossless history and
        :meth:`get_messages`, which may be a compact projection.  Current
        strategies deliberately share one of the two storage names below;
        keeping the lookup here gives future strategies one public contract
        to override instead of making rewind depend on their implementation.
        """
        for attribute in ("_all_messages", "_messages"):
            messages = getattr(self, attribute, None)
            if isinstance(messages, list):
                return messages
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_history_messages()"
        )

    def pin_model_messages(self, new_messages: list[dict]) -> bool:
        """Pin the exact projection restored from a rewind snapshot.

        Projection strategies invalidate this cache on their next mutation,
        at which point their normally derived view resumes from the restored
        canonical history.  FullTranscript has no projection cache, so exact
        equality with its restored append-log is the required condition.
        """
        if hasattr(self, "_msg_cache"):
            self._msg_cache = copy.deepcopy(new_messages)
            if hasattr(self, "_tok_cache"):
                self._tok_cache = None
            return True
        return self.get_messages() == new_messages

    def set_token_estimator(
        self,
        token_estimator: Callable[[list[dict]], int],
    ) -> None:
        """Replace model-specific token counting and invalidate projections.

        Model fallback can change both tokenization and the point at which a
        strategy prunes messages.  The loop calls this public owner method
        instead of reaching into each strategy's private cache fields.
        """
        if not callable(token_estimator):
            raise TypeError("token_estimator must be callable")
        self._token_estimator = token_estimator
        # Every current strategy uses these names for projections affected by
        # token pressure.  Keeping invalidation in the ABC makes that cache
        # contract explicit for future strategies too.
        if hasattr(self, "_msg_cache"):
            self._msg_cache = None
        if hasattr(self, "_tok_cache"):
            self._tok_cache = None


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
        # Caches are invalidated on every mutation. The message cache prevents
        # token estimation and request rendering from logging the same thought
        # expiry twice for one unchanged transcript.
        self._tok_cache: int | None = None
        self._msg_cache: list[dict] | None = None

    def add_system(self, content: str) -> None:
        self._messages.append({"role": "system", "content": content})
        self._tok_cache = None
        self._msg_cache = None

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})
        self._tok_cache = None
        self._msg_cache = None

    def add_assistant(self, message: dict) -> None:
        self._messages.append(message)
        self._tok_cache = None
        self._msg_cache = None

    def add_tool_result(self, tool_call_id: str, content: str, *, tool_name: str = "", cmd_signature: str = "", gate_blocked: bool = False) -> None:
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })
        self._tok_cache = None
        self._msg_cache = None

    def get_messages(self) -> list[dict]:
        if self._msg_cache is None:
            self._msg_cache = self._filter_expired_thought_messages(
                self._messages
            )
        return self._msg_cache

    def replace_all_messages(self, new_messages: list[dict]) -> bool:
        self._messages = list(new_messages)
        self._tok_cache = None
        self._msg_cache = None
        return True

    def estimate_tokens(self) -> int:
        if self._tok_cache is None:
            self._tok_cache = self._token_estimator(self.get_messages())
        return self._tok_cache

    def message_count(self) -> int:
        return len(self._messages)

    def pin_model_messages(self, new_messages: list[dict]) -> bool:
        """Pin only the thought-filtered view of the restored append log."""
        expected = filter_expired_thought_messages(
            self._messages, self._think_keep_turns,
        )
        if expected != new_messages:
            return False
        self._msg_cache = copy.deepcopy(new_messages)
        self._tok_cache = None
        return True
