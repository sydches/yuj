"""Assistant-only accumulator for immutable per-segment usage facts."""
from __future__ import annotations

from typing import Any


class SessionUsageAccumulator:
    """Count each complete model response once without changing run metrics."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self._input_known = True
        self._output_known = True
        self._cache_known = True

    def record(self, usage: Any) -> None:
        input_tokens = _known_token_count(usage, "prompt_tokens")
        output_tokens = _known_token_count(usage, "completion_tokens")
        cached = getattr(usage, "cached_tokens", None)
        if input_tokens is None:
            self._input_known = False
        else:
            self.input_tokens += input_tokens
        if output_tokens is None:
            self._output_known = False
        else:
            self.output_tokens += output_tokens
        if (
            isinstance(cached, bool)
            or not isinstance(cached, int)
            or cached < 0
            or (
                input_tokens is not None
                and cached > input_tokens
            )
        ):
            self._cache_known = False
        else:
            self.cached_tokens += cached

    def trace_fields(self) -> dict[str, object]:
        """Return the complete typed fact persisted for one assistant segment."""
        return {
            "scope": "all_model_responses",
            "input_tokens": self.input_tokens if self._input_known else None,
            "output_tokens": self.output_tokens if self._output_known else None,
            "cached_tokens": self.cached_tokens if self._cache_known else None,
            "cost": None,
            "quota": None,
        }


def _known_token_count(usage: Any, field: str) -> int | None:
    if getattr(usage, f"{field}_known", True) is not True:
        return None
    value = getattr(usage, field, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


__all__ = ["SessionUsageAccumulator"]
