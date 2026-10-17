"""Legacy local-tokenizer estimates for explicitly selected artifacts.

An artifact's vocabulary and renderer are not verified against the backend.
Plain-string agreement or copying a template does not establish full-request
agreement. Automatic request counting is owned by the active transport.

Configuration: cfg.tokenizer_id is either a HuggingFace model id
(downloaded once, cached locally) or a path to a directory containing
tokenizer.json + tokenizer_config.json.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ..server.token_counting import _request_count_reuse, clear_request_count_reuse

log = logging.getLogger(__name__)


def _deterministic_encoding_binding(tokenizer):
    """Recognize the standard native path; other tokenizers keep legacy counting."""
    try:
        from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
        from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            return None
        for name, expected in (
            ("apply_chat_template", PreTrainedTokenizerBase.apply_chat_template),
            ("__call__", PreTrainedTokenizerBase.__call__),
            ("_encode_plus", PreTrainedTokenizerFast._encode_plus),
            ("_batch_encode_plus", getattr(PreTrainedTokenizerFast, "_batch_encode_plus", None)),
        ):
            if getattr(getattr(tokenizer, name, None), "__func__", None) is not expected:
                return None
        backend = tokenizer.backend_tokenizer
        model = backend.model
        if type(model) not in (BPE, Unigram, WordLevel, WordPiece):
            return None
        if isinstance(model, BPE) and model.dropout not in (None, 0):
            return None
        # Keep the actual objects alive for this scope; replacement cannot
        # accidentally match a recycled object id. Never copy the base vocab.
        return (tokenizer, backend,
                tuple((key, repr(value)) for key, value in tokenizer.added_tokens_decoder.items()),
                tokenizer.split_special_tokens)
    except (ImportError, AttributeError, TypeError, ValueError):
        return None


def _parse_tool_call_args(messages: list[dict]) -> list[dict]:
    """Return messages with tool_call arguments parsed to objects.

    The wire format carries ``function.arguments`` as a JSON STRING;
    llama.cpp parses it into an object before rendering its chat
    template, and the GGUF-embedded Qwen template requires a mapping
    ("Can only get item pairs from a mapping"). HF's renderer passes
    the string through untouched, so we parse here to match the
    server. Copy-on-write: only affected messages are copied.
    """
    out = []
    for m in messages:
        tcs = m.get("tool_calls")
        if not tcs:
            out.append(m)
            continue
        new_tcs = []
        for tc in tcs:
            fn = (tc.get("function") or {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    parsed = json.loads(args) if args.strip() else {}
                except Exception:
                    parsed = {}
                tc = {**tc, "function": {**fn, "arguments": parsed}}
            new_tcs.append(tc)
        out.append({**m, "tool_calls": new_tcs})
    return out


class LocalTokenizer:
    """Estimate with an explicit HF artifact; serving identity is unverified."""

    def __init__(self, tokenizer_id: str):
        if not tokenizer_id:
            raise ValueError("LocalTokenizer requires a non-empty tokenizer_id")
        from transformers import AutoTokenizer
        self._id = tokenizer_id
        self.last_basis = "local_tokenizer_estimate"
        self._tok = AutoTokenizer.from_pretrained(tokenizer_id)
        # A GGUF may embed a different chat template from its hub tokenizer.
        # When set, use the server's /props template. None uses the bundled
        # hub template.
        self._chat_template: str | None = None

    def sync_chat_template(self, base_url: str) -> bool:
        """Fetch the server's chat template from /props; True on success.

        Call once at session start. Failure leaves the bundled template
        in place — counts stay approximate rather than exact.
        """
        try:
            import httpx
            base = base_url.rstrip("/").removesuffix("/v1")
            props = httpx.get(f"{base}/props", timeout=10).json()
            tmpl = props.get("chat_template") or ""
            if tmpl:
                self._chat_template = tmpl
                return True
        except Exception:
            pass
        return False

    @property
    def id(self) -> str:
        return self._id

    def count(self, messages: list[dict], tools: list[dict] | None = None) -> int:
        """Estimate tokens under the selected artifact's chat template.

        ``tools`` is the OAI-style tool-schema list the request will carry.
        Pass the same schemas that the client sends so the count includes
        the rendered tool catalog.

        For mixed message shapes (assistant tool_calls, tool role
        results), backend rendering may differ. If the template raises (unsupported
        message shape), falls back to per-message encode summed
        with a small framing budget.
        """
        try:
            self.last_basis = "local_tokenizer_estimate"
            scoped = _request_count_reuse.get() is not None
            binding = _deterministic_encoding_binding(self._tok) if scoped else None
            kwargs = {}
            if self._chat_template:
                # Server template render: match the real request, which
                # always ends with the generation prompt.
                kwargs = {"chat_template": self._chat_template,
                          "add_generation_prompt": True}
            reuse_key = None
            if binding is None:
                if scoped:
                    clear_request_count_reuse()
                result = self._tok.apply_chat_template(
                    _parse_tool_call_args(messages), tools=tools or None,
                    tokenize=True, **kwargs)
            else:
                # Render every time: templates may depend on the current time.
                rendered = self._tok.apply_chat_template(
                    _parse_tool_call_args(messages), tools=tools or None,
                    tokenize=False, **kwargs)
                reuse_key = (self, "local_rendered_encoding", binding, rendered)
                reused = _request_count_reuse.get()
                if reused and reused[0] == reuse_key:
                    return reused[1]
                result = self._tok(
                    rendered, padding=False, truncation=False, max_length=None,
                    add_special_tokens=False, return_tensors=None,
                )
            # Newer transformers (≥4.42 or so) return a BatchEncoding /
            # dict {input_ids: [...], attention_mask: [...]} instead of
            # a flat list when tokenize=True. Older versions returned a
            # list directly. Handle both shapes; the previous version
            # of this code did `len(result)` which on the dict returned
            # 2 (the number of keys) — silently giving the same wrong
            # token count for every input.
            if isinstance(result, dict) or hasattr(result, "input_ids"):
                ids = result["input_ids"]
            else:
                ids = result
            count = len(ids)
            if reuse_key is not None:
                _request_count_reuse.set((reuse_key, count, None))
            return count
        except Exception as exc:
            clear_request_count_reuse()
            self.last_basis = "local_message_estimate"
            log.warning("local chat template failed (%s); using per-message estimate", type(exc).__name__)
            total = 0
            for m in messages:
                content = m.get("content") or ""
                if not isinstance(content, str):
                    content = str(content)
                total += len(self._tok.encode(content, add_special_tokens=False))
                tcs = m.get("tool_calls") or []
                for tc in tcs:
                    fn = tc.get("function") or {}
                    s = str(fn.get("name", "")) + str(fn.get("arguments", ""))
                    total += len(self._tok.encode(s, add_special_tokens=False))
                total += 8
            if tools:
                import json
                total += len(self._tok.encode(
                    json.dumps(tools), add_special_tokens=False))
            return total


def load(tokenizer_id: str) -> LocalTokenizer | None:
    """Load a LocalTokenizer or return None when not configured.

    Empty string or None disables — caller falls back to the
    chars_div_4 estimator. Any other failure (network, missing
    files, malformed config) raises so misconfiguration is loud.
    """
    if not tokenizer_id:
        return None
    return LocalTokenizer(tokenizer_id)
