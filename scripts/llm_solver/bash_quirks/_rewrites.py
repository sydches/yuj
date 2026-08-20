"""Universal rewrite rules — quiet flags for noisy commands.

Sourced from ``bash_quirks/rewrites.toml``: pip -q, npm --loglevel=error,
make -s, etc. Apply to every run regardless of task format.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


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
        path = Path(__file__).parent / "rewrites.toml"
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
