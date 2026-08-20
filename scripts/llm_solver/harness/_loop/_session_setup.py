"""Per-session setup helpers for ``solve_task``.

Extracted from ``driver.py``. Each helper takes everything it needs as
an explicit parameter — no shared mutable state with the driver's
per-iteration locals. The driver retains the for-session loop and all
aggregation counters; these helpers handle the parts that are pure
construction.
"""
from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from ...config import Config
from ..context import ContextManager
from ..context_strategies import resolve_context_mode_for_class
from ..context_strategies._metadata import LEGACY_CONTEXT_CONSTRUCTOR_CONFIG_ATTRS

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


def _constructor_config_values(
    config_attrs: Mapping[str, str],
    cfg: Config,
    repo_dir: Path,
) -> dict[str, object]:
    """Resolve constructor kwarg values declared by context-mode metadata."""
    values: dict[str, object] = {}
    for param, config_attr in config_attrs.items():
        if config_attr == "cwd":
            values[param] = str(repo_dir)
        else:
            values[param] = getattr(cfg, config_attr)
    return values


def build_context_manager(
    context_class: type[ContextManager] | None,
    cfg: Config,
    repo_dir: Path,
    initial: str,
    session_num: int,
    token_estimator,
) -> ContextManager | None:
    """Instantiate the selected context class with introspection-driven kwargs.

    Different context classes take different constructor arguments —
    FullTranscript takes none, CompactTranscript/YujTranscript take
    original_prompt, SolverStateContext and its subclasses
    (CompoundContext) also require cwd and accept trace_lines /
    evidence_lines. Use introspection so new context classes can be
    added without editing this dispatch.

    On session 2+, pre-populate the rolling tool-result window with
    files modified in prior sessions so the model doesn't edit from
    stale memory.  Only SolverStateContext subclasses (stateful,
    compound) have ``prepopulate_from_trace``; others skip silently.

    Returns None when ``context_class`` is None (legacy path).
    """
    if context_class is None:
        return None
    sig = inspect.signature(context_class.__init__)
    kwargs: dict = {"original_prompt": initial}
    context_mode = resolve_context_mode_for_class(context_class)
    config_attrs = (
        context_mode.metadata.constructor_config_attrs
        if context_mode is not None
        else LEGACY_CONTEXT_CONSTRUCTOR_CONFIG_ATTRS
    )
    _cfg_map = _constructor_config_values(config_attrs, cfg, repo_dir)
    if token_estimator is not None:
        _cfg_map["token_estimator"] = token_estimator
    for param, value in _cfg_map.items():
        if param in sig.parameters:
            kwargs[param] = value
    ctx = context_class(**kwargs)
    # Session 2+: pre-populate the rolling tool-result window
    # with files modified in prior sessions so the model doesn't
    # edit from stale memory.  Only SolverStateContext subclasses
    # (stateful, compound) have this method; others skip silently.
    if session_num > 1 and hasattr(ctx, "prepopulate_from_trace"):
        n_files = ctx.prepopulate_from_trace()
        if n_files:
            log.info("Pre-populated rolling window with %d file(s) from prior sessions", n_files)
    return ctx


def inject_resume_messages(session, resume_path: Path, initial: str) -> None:
    """Replace session.context's initial messages with a parsed prior transcript.

    Caller invokes only on session 1 when ``resume_path`` is set. The
    function parses the prior verbatim transcript and replaces the
    freshly-built [system, initial_user] context with the full prior
    conversation plus a synthesized tool-result for any unanswered
    tool_calls plus the new user message (the caller passed the
    resume message as ``initial``).

    Raises RuntimeError if the context manager does not support
    ``replace_all_messages``.
    """
    from .resume import parse_resume_transcript, build_resumed_messages
    prior_msgs, last_assistant = parse_resume_transcript(resume_path)
    resumed_msgs = build_resumed_messages(prior_msgs, last_assistant, initial)
    if not session.context.replace_all_messages(resumed_msgs):
        raise RuntimeError(
            f"Resume injection failed: context manager "
            f"{type(session.context).__name__} does not support "
            f"replace_all_messages(). Use a different --context."
        )
    log.info(
        "resume: loaded %d prior messages from %s, appended %s tool-result(s) + new user message",
        len(prior_msgs), resume_path,
        len((last_assistant or {}).get("tool_calls") or []),
    )
