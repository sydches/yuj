"""Secret redaction rules — applied to every tool result before truncation."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .._shared.paths import package_data_path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RedactionRule:
    """One [[redaction]] entry from redactions.toml.

    Patterns are applied to every tool result text BEFORE truncation, so
    a secret near the tail of a long output gets masked even when the
    head/tail truncator would have kept the bytes. `replace` may
    contain back-references (\\1, \\2) to capture groups in `pattern`.
    """
    name: str
    pattern: re.Pattern
    replace: str


def load_redactions(path: Path | None = None) -> list[RedactionRule]:
    """Load [[redaction]] entries from redactions.toml.

    Defaults to the redactions.toml next to this module. Returns empty
    list if the file is missing — redaction is opt-in by file presence.
    """
    from .._shared.toml_compat import tomllib

    if path is None:
        path = package_data_path(__package__, "redactions.toml")
    if not path.is_file():
        return []
    with path.open("rb") as f:
        data = tomllib.load(f)
    out: list[RedactionRule] = []
    for entry in data.get("redaction", []):
        try:
            out.append(RedactionRule(
                name=entry["name"],
                pattern=re.compile(entry["pattern"]),
                replace=entry.get("replace", "[REDACTED]"),
            ))
        except (KeyError, re.error) as e:
            log.warning("redactions.toml: skipping entry %r — %s", entry, e)
    return out


def apply_redactions(text: str, rules: list[RedactionRule] | None) -> str:
    """Apply each redaction pattern in order.

    Order matters only when one pattern is a superset of another; the
    redactions.toml ordering puts most-specific patterns first
    (AWS access key id before generic env_token_assignment) so the
    specific tag survives. Empty/missing rule list is a no-op.
    """
    if not rules:
        return text
    for rule in rules:
        before = text
        text, count = rule.pattern.subn(rule.replace, text)
        if text != before:
            from ..harness.savings import get_ledger
            get_ledger().record_transform(
                bucket="tool_output_redaction",
                layer="L2_bash_quirks",
                mechanism=rule.name,
                before=before,
                after=text,
                surface="tool_output",
                change_count=count,
                retain_debug_content=False,
            )
    return text
