"""Forbidden command patterns — hard refusal layer.

When a pattern matches a bash command, the harness substitutes the
command with ``false  # [HARNESS: <reason>]`` so the model sees the
explanation in the tool result.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForbiddenRule:
    """One forbidden-pattern entry from forbidden.toml.

    When `pattern` matches a bash cmd, the harness substitutes the command
    with `false  # [HARNESS: <reason>]` so the model sees the explanation
    in the result. Same shape as RewriteRule but with a hard refusal
    semantics rather than a flag-append.
    """
    name: str
    pattern: re.Pattern
    reason: str


def load_forbidden_rules(path: Path | None = None) -> list[ForbiddenRule]:
    """Load [[forbidden]] entries from forbidden.toml.

    Defaults to the forbidden.toml next to this module. Returns empty
    list if the file is missing.
    """
    from .._shared.toml_compat import tomllib

    if path is None:
        path = Path(__file__).parent / "forbidden.toml"
    if not path.is_file():
        return []
    with path.open("rb") as f:
        data = tomllib.load(f)
    out: list[ForbiddenRule] = []
    for entry in data.get("forbidden", []):
        try:
            out.append(ForbiddenRule(
                name=entry["name"],
                # DOTALL lets multi-line patterns (python heredocs, here-docs
                # spanning newlines) match across the body of the bash cmd.
                pattern=re.compile(entry["pattern"], re.IGNORECASE | re.DOTALL),
                reason=entry["reason"],
            ))
        except (KeyError, re.error) as e:
            log.warning("forbidden.toml: skipping entry %r — %s", entry, e)
    return out


def apply_forbidden(cmd: str, rules: list[ForbiddenRule] | None) -> str:
    """If any forbidden rule matches, replace cmd with a refusal `false` no-op.

    Returns the original cmd when no rule matches.
    """
    if not rules:
        return cmd
    for rule in rules:
        if rule.pattern.search(cmd):
            # Show the reason in the tool result so the model can choose a
            # permitted command instead of repeating the refused command.
            safe = rule.reason.replace("\\", "").replace('"', "'")
            safe = safe.replace("$", "").replace("`", "")
            return (f'echo "[HARNESS refused this command: {safe}]" >&2; '
                    f'false  # [HARNESS: {rule.reason}]')
    return cmd
