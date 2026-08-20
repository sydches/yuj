"""Regression: the assistant runner must derive max_tokens before any request.

The config loader ships max_tokens=0 as a placeholder for the measurement
entrypoint (scripts.llm_solver.__main__) to derive after server-context
resolution. The assistant path does the same in _apply_effective_context;
a missing derivation would send max_tokens=0 and stop generation at turn 0.
"""
from unittest.mock import MagicMock

from scripts.llm_assist.runner import _apply_effective_context
from scripts.llm_solver.config import load_config


def _assistant_cfg():
    return load_config(overrides={"runtime_mode": "assistant"})


def test_max_tokens_derived_without_server_ctx():
    cfg = _assistant_cfg()
    client = MagicMock()
    client.query_server_context.return_value = None

    out = _apply_effective_context(cfg, client)

    assert out.max_tokens > 0, "max_tokens placeholder must not reach the wire"
    assert out.max_tokens == int(out.context_size * out.max_tokens_fraction)


def test_max_tokens_follows_shrunk_server_ctx():
    cfg = _assistant_cfg()
    client = MagicMock()
    client.query_server_context.return_value = 8192

    out = _apply_effective_context(cfg, client)

    assert out.context_size == 8192
    assert out.max_tokens == int(8192 * out.max_tokens_fraction)
    assert out.max_tokens > 0
