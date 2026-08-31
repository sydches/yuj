"""Universal rewrite rules — quiet flags for noisy commands.

Sourced from ``bash_quirks/rewrites.toml``: pip -q, npm --loglevel=error,
make -s, etc. Apply to every run regardless of task format.
"""
from __future__ import annotations

import logging
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from .._shared.paths import package_data_path
from ._output import (
    _find_command_span,
    _shell_command_spans,
    _strip_leading_assignments,
)

log = logging.getLogger(__name__)


_DISPLAY_ONLY_PIPE_COMMANDS = frozenset({
    "awk", "cut", "egrep", "fgrep", "grep", "head", "rg", "sed",
    "sort", "tail", "uniq", "wc",
})


def strip_display_only_test_pipeline(
    command: str,
    patterns: tuple[re.Pattern, ...],
) -> str:
    """Remove a trailing display-only pipeline from a test invocation."""
    runner_span = _find_command_span(command, patterns)
    if runner_span is None:
        return command
    spans = _shell_command_spans(command)
    try:
        runner_index = spans.index(runner_span)
    except ValueError:
        return command
    if runner_index >= len(spans) - 1:
        return command
    for index in range(runner_index, len(spans) - 1):
        separator = command[spans[index][1] : spans[index + 1][0]].strip()
        if separator not in {"|", "|&"}:
            return command
    for start, end in spans[runner_index + 1 :]:
        try:
            words = shlex.split(_strip_leading_assignments(command[start:end]))
        except ValueError:
            return command
        if not words or Path(words[0]).name not in _DISPLAY_ONLY_PIPE_COMMANDS:
            return command
        if any("<" in word or ">" in word for word in words[1:]):
            return command
    return command[: runner_span[1]].rstrip()


@dataclass(frozen=True)
class RewriteRule:
    """One command rewrite rule from rewrites.toml."""
    name: str
    pattern: re.Pattern
    flag: str
    skip_if: tuple[str, ...]


def load_universal_rewrites(path: Path | None = None) -> list[RewriteRule]:
    """Load [[rewrite]] entries from rewrites.toml.

    Defaults to the rewrites.toml next to this module.
    Returns empty list if file is missing.
    """
    from .._shared.toml_compat import tomllib

    if path is None:
        path = package_data_path(__package__, "rewrites.toml")
    if not path.is_file():
        return []

    with open(path, "rb") as f:
        data = tomllib.load(f)

    rules = []
    for entry in data.get("rewrite", []):
        # Handle each entry separately so one malformed pattern cannot break the
        # whole list (mirrors load_forbidden_rules / load_redactions).
        try:
            rules.append(RewriteRule(
                name=entry["name"],
                pattern=re.compile(entry["pattern"], re.IGNORECASE),
                flag=entry["flag"],
                skip_if=tuple(entry.get("skip_if", [])),
            ))
        except (KeyError, re.error) as e:
            log.warning(
                "skipping malformed universal_rewrite entry %r: %s",
                entry.get("name", "<no name>"), e,
            )
    return rules
