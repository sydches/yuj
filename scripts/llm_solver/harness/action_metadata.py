"""Content-blind metadata for model tool actions.

These helpers inspect the model's own tool arguments, not task output.  They
exist because the human-readable ``args_summary`` is intentionally short, while
the state-backed salience projector needs to know whether a tool call was a
source mutation rather than another read of the same file.
"""
from __future__ import annotations

from typing import Any

from .bash_write_classification import (
    classify_bash_write,
    extract_source_write_paths,
)
from .tool_specs import ACTION_WRITE_LIKE_TOOL_NAMES


def action_metadata(tool_name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Return trace-safe metadata for a tool call.

    The result is intentionally mechanical:
    - ``write_like`` says the action shape is a write/edit/apply command.
    - ``source_write_like`` additionally requires at least one source-looking
      path in the action arguments.
    - ``source_write_paths`` is a small ordered path list for state projection.
    """
    arguments = arguments or {}
    tool_name = tool_name or ""
    text = ""
    write_like = False

    if tool_name in ACTION_WRITE_LIKE_TOOL_NAMES:
        write_like = True
        text = " ".join(str(v) for v in arguments.values())
    elif tool_name == "bash":
        text = str(arguments.get("cmd") or "")
        classification = classify_bash_write(text)
        write_like = classification.action_write_like
        paths = list(classification.source_write_paths)
        return {
            "write_like": write_like,
            "source_write_like": classification.source_write_like,
            "source_write_paths": paths[:8],
        }
    else:
        text = " ".join(str(v) for v in arguments.values())

    paths = list(extract_source_write_paths(text)) if write_like else []
    return {
        "write_like": write_like,
        "source_write_like": bool(write_like and paths),
        "source_write_paths": paths[:8],
    }


__all__ = ["action_metadata"]
