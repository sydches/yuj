"""Native Messages counting, using the adapter's converted request and route."""
from __future__ import annotations

from ..llm_solver.server.token_counting import BackendTokenCounter, UncountableRequest

# Count endpoint inputs, as documented by the provider. Generation-only
# controls are excluded; unknown fields fail explicitly instead of disappearing.
_COUNT_FIELDS = frozenset({
    "model", "messages", "system", "tools", "tool_choice", "thinking",
    "output_config", "cache_control",
})
_GENERATION_FIELDS = frozenset({
    "max_tokens", "stream", "temperature", "top_p", "top_k", "stop_sequences",
    "metadata", "service_tier",
})


class AnthropicTokenCounter(BackendTokenCounter):
    # The provider explicitly describes its count as an estimate. Precision
    # differs from provenance: it is still the active provider's current count.
    count_precision = "estimate"

    @property
    def endpoint(self) -> str:
        return self.client.cfg.base_url

    @property
    def supported(self) -> bool:
        return True

    def count(self, messages: list[dict], tools: list[dict] | None = None) -> int:
        canonical = self.client.prepare_chat_request(messages, tools or [])
        return self.count_payload(self.client.prepare_anthropic_request(canonical))

    def _request_count(self, payload: dict, timeout: float) -> dict:
        unknown = payload.keys() - _COUNT_FIELDS - _GENERATION_FIELDS
        if unknown:
            raise UncountableRequest("unsupported native counting inputs")
        body = {key: value for key, value in payload.items() if key in _COUNT_FIELDS}
        response = self.client._post_anthropic(
            "/messages/count_tokens", body, read_timeout=timeout, connect_timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
