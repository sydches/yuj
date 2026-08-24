"""Public types and startup resolver for in-process compaction hooks.

Hook modules are trusted harness extensions.  They run in the Yuj process,
not in the model-command sandbox, and are imported while configuration is
validated so a bad reference fails before model work starts.
"""
from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, TypeAlias


Message = dict[str, Any]


@dataclass(frozen=True)
class CompactionFileOps:
    """Content-blind file and test facts projected from the trace prefix."""

    read_files: tuple[str, ...]
    modified_files: tuple[str, ...]
    last_test_runner_digest: str
    mutation_count: int


@dataclass(frozen=True)
class CompactionPreparation:
    """Immutable envelope passed to a configured compaction hook.

    Message dictionaries are detached copies of the harness conversation.
    The hook may inspect or mutate those copies without changing live context.
    """

    messages_to_summarize: tuple[Message, ...]
    kept_tail: tuple[Message, ...]
    previous_summary: str
    file_ops: CompactionFileOps
    tokens_before: int
    first_kept_turn: int
    knobs: Mapping[str, object]


@dataclass(frozen=True)
class Cancel:
    """Tell Yuj to leave the current conversation un-compacted."""


@dataclass(frozen=True)
class Compaction:
    """Propose a checkpoint-format summary and assistant-turn boundary."""

    summary: str
    first_kept_turn: int


CompactionHook: TypeAlias = Callable[
    [CompactionPreparation], None | Cancel | Compaction
]


def resolve_compaction_hook(reference: str) -> CompactionHook | None:
    """Resolve ``module:function`` once, with config-facing errors."""
    if not isinstance(reference, str):
        raise ValueError(
            "config error: context.compaction_hook must be a string in "
            "'module:function' form."
        )
    normalized = reference.strip()
    if not normalized:
        return None
    return _resolve_compaction_hook(normalized)


@lru_cache(maxsize=None)
def _resolve_compaction_hook(reference: str) -> CompactionHook:
    if reference.count(":") != 1:
        raise ValueError(
            "config error: context.compaction_hook must use exactly "
            f"'module:function', got {reference!r}."
        )
    module_name, function_name = reference.split(":", 1)
    if not module_name or not function_name or not function_name.isidentifier():
        raise ValueError(
            "config error: context.compaction_hook must use exactly "
            f"'module:function', got {reference!r}."
        )
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ValueError(
            "config error: could not import context.compaction_hook "
            f"module {module_name!r}: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        hook = getattr(module, function_name)
    except AttributeError as exc:
        raise ValueError(
            "config error: context.compaction_hook function "
            f"{function_name!r} was not found in module {module_name!r}."
        ) from exc
    if not callable(hook):
        raise ValueError(
            "config error: context.compaction_hook target "
            f"{reference!r} is not callable."
        )
    if inspect.iscoroutinefunction(hook):
        raise ValueError(
            "config error: context.compaction_hook target "
            f"{reference!r} must be synchronous."
        )
    return hook


__all__ = [
    "Cancel",
    "Compaction",
    "CompactionFileOps",
    "CompactionHook",
    "CompactionPreparation",
    "resolve_compaction_hook",
]
