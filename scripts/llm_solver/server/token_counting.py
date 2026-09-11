"""Count prepared requests with the serving backend, without generation."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from urllib.parse import urlsplit, urlunsplit

log = logging.getLogger(__name__)
_SDK_OPTIONS = {"extra_body", "extra_headers", "extra_query", "timeout"}


class UncountableRequest(ValueError):
    """An adapter cannot express this input in its count API; no HTTP was sent."""


class TextTokenCounter:
    """Retokenize visible text with the serving model, without a chat template.

    One narration attempt shares the existing connection-time allowance across
    counts. A failure disables further counts for that attempt.
    """

    def __init__(self, counter):
        self.counter = counter
        self.remaining_seconds = float(counter.client.cfg.timeout_connect)
        self.last = {}

    def count(self, text, *, model, remaining_seconds=None):
        started = time.monotonic()
        timeout = self.remaining_seconds
        if remaining_seconds is not None:
            timeout = min(timeout, remaining_seconds)
        value, reason, calls = None, "", 0
        try:
            if timeout <= 0:
                raise ValueError("counting_allowance_exhausted")
            if model != self.counter.client.cfg.model:
                raise ValueError("model_binding_changed")
            url = urlsplit(self.counter.endpoint)
            route = url.path.rstrip("/").removesuffix("/v1") + "/tokenize"
            endpoint = urlunsplit((url.scheme, url.netloc, route, "", ""))
            calls = 1
            response = self.counter.client.client.post(
                endpoint, body={"model": model, "content": text,
                                "add_special": False, "parse_special": False},
                cast_to=dict[str, object], options={"timeout": timeout, "max_retries": 0},
            )
            tokens = response.get("tokens") if isinstance(response, dict) else None
            if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError("invalid_tokenize_response")
            value = len(tokens)
            if text and not value:
                raise ValueError("empty_tokenize_response")
        except Exception as exc:
            value, reason = None, type(exc).__name__
            self.remaining_seconds = 0
        elapsed = time.monotonic() - started
        self.remaining_seconds = max(0, self.remaining_seconds - elapsed)
        self.last = {"count_basis": "backend_text_tokens" if value is not None else "unavailable",
                     "count_reason": reason, "model": model, "text_tokens": value,
                     "text_chars": len(text), "counting_calls": calls,
                     "counting_seconds": elapsed, "counting_timeout_seconds": max(0, timeout)}
        self.counter._emit("narration_token_count", dict(self.last))
        return value


def estimate_payload(payload: dict) -> int:
    """A declared character estimate, not proof that a request fits."""
    body = {k: v for k, v in payload.items() if k not in _SDK_OPTIONS}
    body.update(payload.get("extra_body") or {})
    return max(1, len(json.dumps(body, ensure_ascii=False, default=str)) // 4)


class BackendTokenCounter:
    """Bind counting to a client's route and exact request preparation.

    The llama chat input_tokens extension parses and tokenizes in one request.
    Other dialects remain explicit estimates until their adapters provide a
    counting contract. No vocabulary is inferred from a model name. Counts are
    not cached: a serving process can change behind an unchanged URL or alias.
    """

    count_precision = "backend_reported"

    def __init__(self, client):
        self.client = client
        self.event_sink = None
        self.last: dict = {}
        self.calls = 0
        self.elapsed_seconds = 0.0
        self._binding = None
        self._retry_at = 0.0
        self._unavailable_reason = ""
        self._usage_mismatch = False
        self.count_timeout = None  # Optional caller-owned remaining-time callback.

    @property
    def id(self) -> str:
        return f"backend:{self.client.cfg.model}"

    def count(self, messages: list[dict], tools: list[dict] | None = None) -> int:
        return self.count_payload(self.client.prepare_chat_request(messages, tools or []))

    @property
    def endpoint(self) -> str:
        return str(self.client.client.base_url)

    @property
    def supported(self) -> bool:
        return self.client.cfg.request_dialect == "llama"

    def _request_count(self, payload: dict, timeout: float) -> dict:
        body = {k: v for k, v in payload.items() if k not in _SDK_OPTIONS}
        options = {"timeout": timeout, "max_retries": 0}
        for source, target in (("extra_body", "extra_json"),
                               ("extra_headers", "headers"),
                               ("extra_query", "params")):
            if source in payload:
                options[target] = payload[source]
        return self.client.client.post(
            "/chat/completions/input_tokens", body=body,
            cast_to=dict[str, object], options=options,
        )

    def _emit(self, event: str, record: dict) -> None:
        if self.event_sink is not None:
            self.event_sink(event, **record)
        log.debug("%s %s", event, json.dumps(record, sort_keys=True))

    def count_payload(self, payload: dict) -> int:
        """Count an already prepared SDK payload without denormalizing again."""
        started = time.monotonic()
        cfg = self.client.cfg
        # Use the SDK route, not a separately constructed server URL or client.
        endpoint = self.endpoint
        binding = (endpoint, payload.get("model"), cfg.request_dialect, id(self.client.profile))
        if binding != self._binding:
            self._binding = binding
            self._retry_at = 0.0
            self._unavailable_reason = ""
            self._usage_mismatch = False
        body = {k: v for k, v in payload.items() if k not in _SDK_OPTIONS}
        extra_body = payload.get("extra_body") or {}
        fingerprint = hashlib.sha256(json.dumps(
            {"body": body, "extra_body": extra_body}, sort_keys=True,
            ensure_ascii=False, default=str,
        ).encode()).hexdigest()
        count = None
        basis = "character_estimate"
        reason = "unsupported_request_dialect"
        calls = 0
        timeout = float(cfg.timeout_connect)
        budget_error = None
        if self.count_timeout is not None:
            try:
                timeout = min(timeout, self.count_timeout(timeout))
            except Exception as exc:
                budget_error, timeout = exc, 0.0
        if budget_error is not None:
            reason = type(budget_error).__name__
        elif timeout <= 0:
            reason = "counting_allowance_exhausted"
        elif self.supported:
            reason = self._unavailable_reason
            if started >= self._retry_at:
                # These bound transport phases, not total wall-clock duration.
                # The caller can cap them by its remaining invocation allowance.
                calls = 1
                try:
                    response = self._request_count(payload, timeout)
                    value = response.get("input_tokens") if isinstance(response, dict) else None
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError("invalid input_tokens response")
                    count, basis, reason = value, "backend_input_tokens", ""
                    if self.count_precision == "estimate":
                        reason = "provider_reports_estimate"
                    if self._usage_mismatch:
                        basis, reason = "backend_input_tokens_unverified", "response_usage_disagreement"
                    self._retry_at = 0.0
                    self._unavailable_reason = ""
                except Exception as exc:
                    local_failure = isinstance(exc, UncountableRequest)
                    if local_failure:
                        calls = 0
                    # Do not log response bodies, headers or credentials.
                    status = getattr(exc, "status_code", None)
                    if status is None:
                        status = getattr(getattr(exc, "response", None), "status_code", None)
                    reason = f"{type(exc).__name__}" + (f":{status}" if status else "")
                    self._unavailable_reason = reason
                    # An incomplete history (for example, an isolated tool
                    # result measured for clipping) can be rejected while the
                    # next complete request remains countable.
                    self._retry_at = (0.0 if local_failure or status in {400, 422}
                                      else time.monotonic() + timeout)
                    log.warning("backend token counting unavailable (%s); using character estimate", reason)
        if count is None:
            count = estimate_payload(payload)
        elapsed = time.monotonic() - started
        self.calls += calls
        self.elapsed_seconds += elapsed
        url = urlsplit(endpoint)
        host = url.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        if url.port:
            host += f":{url.port}"
        self.last = {
            "count_basis": basis, "count_reason": reason,
            "count_precision": (self.count_precision if basis == "backend_input_tokens"
                                else "unverified" if basis.endswith("_unverified") else "estimate"),
            "prompt_tokens": count, "request_sha256": fingerprint,
            "model": payload.get("model", ""),
            "endpoint": urlunsplit((url.scheme, host, url.path, "", "")),
            "counting_calls": calls, "counting_seconds": elapsed,
            "counting_timeout_seconds": max(0.0, timeout),
        }
        self._emit("request_token_count", dict(self.last))
        if budget_error is not None:
            raise budget_error
        return count

    def observe_usage(self, response, record: dict) -> None:
        """Compare the count with usage from this completed request only."""
        from .request_controls import extract_cache_observation

        usage = extract_cache_observation(response)
        if usage is None:
            return
        fields = {
            **record, "reported_prompt_tokens": usage.prompt_tokens,
            "count_delta": usage.prompt_tokens - record["prompt_tokens"],
        }
        # This event reports agreement, not a second counting operation.
        fields.pop("counting_calls", None)
        fields.pop("counting_seconds", None)
        self._emit("request_token_count_usage", fields)
        if (record["count_basis"] == "backend_input_tokens"
                and record.get("count_precision") != "estimate" and fields["count_delta"]):
            self._usage_mismatch = True
            log.warning("backend count differs from response usage: count=%d reported=%d",
                        record["prompt_tokens"], usage.prompt_tokens)
