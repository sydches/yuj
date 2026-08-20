"""tool_quirks/transforms.py — transforms applied to non-bash tool results.

Currently exposes glob caps; future tools land here.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from .._shared.toml_compat import tomllib


# Cache the TOML parse. The file is shipped with the package and never
# changes during a run, so re-
# parsing on every glob call is pure overhead.
@functools.lru_cache(maxsize=1)
def _load_glob_data() -> dict:
    p = Path(__file__).parent / "glob.toml"
    if not p.is_file():
        return {}
    with p.open("rb") as f:
        return tomllib.load(f)


def _xml_attr(s: str) -> str:
    """Escape a string for inclusion in an XML attribute value.

    Pattern, scope, and hint once landed verbatim in the envelope
    attributes, so a model-supplied
    pattern containing `"`/`<`/`>`/`&`/`'` would emit malformed XML
    and the search_result envelope would no longer parse.
    """
    return (
        s.replace("&", "&amp;")
         .replace('"', "&quot;")
         .replace("'", "&apos;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


def _hint_envelope(tool: str, pattern: str, scope: str, total: int, hint: str) -> str:
    """Return an empty `<search_result>` envelope with a hint and zero paths."""
    return (
        f'<search_result tool="{_xml_attr(tool)}" total="{total}" '
        f'shown="0" page="1" next_page="0" '
        f'pattern="{_xml_attr(pattern)}" scope="{_xml_attr(scope)}" '
        f'hint="{_xml_attr(hint)}">\n</search_result>'
    )


def apply_glob_caps(
    pattern: str, scope: str, total: int, cfg: Any,
    *, data: dict | None = None,
) -> str | None:
    """Decide whether to refuse the glob listing.

    Returns a refusal envelope when a cap fires, or None when the listing
    is allowed through. Caller is responsible for emitting the normal
    paginated envelope when None is returned.
    """
    if data is None:
        data = _load_glob_data()
    hints = (data.get("refusal_hints") or {})
    refuse_unscoped = bool(getattr(cfg, "tools_glob_refuse_unscoped_recursive", True))
    max_listed = int(getattr(cfg, "tools_glob_max_listed_paths", 50) or 0)

    if refuse_unscoped and pattern.startswith("**/") and scope in (".", "./", ""):
        envelope = _hint_envelope(
            tool="glob", pattern=pattern, scope=scope, total=total,
            hint=hints.get("unscoped_recursive", "unscoped recursive glob"),
        )
        _record_refusal(
            mechanism="unscoped",
            pattern=pattern, scope=scope, total=total, envelope=envelope,
        )
        return envelope
    if max_listed > 0 and total > max_listed:
        envelope = _hint_envelope(
            tool="glob", pattern=pattern, scope=scope, total=total,
            hint=hints.get("too_broad", "pattern too broad"),
        )
        _record_refusal(
            mechanism="cap",
            pattern=pattern, scope=scope, total=total, envelope=envelope,
        )
        return envelope
    return None


def _record_refusal(
    *, mechanism: str, pattern: str, scope: str, total: int, envelope: str,
) -> None:
    """Record glob-refusal savings.

    This mirrors bash_quirks/condense_output's ledger pattern:
    input_chars is a rough estimate
    of the listing the model would have received (40 chars/path), and
    output_chars is the actual envelope size.
    """
    try:
        from ..harness.savings import get_ledger
        # Conservative estimate: ~40 chars per path (typical filename
        # + newline) for the listing the model would have seen.
        approx_listing_chars = max(0, total) * 40
        get_ledger().record(
            bucket="tool_quirks_glob_refusal",
            layer="L2_tool_quirks",
            mechanism=mechanism,
            input_chars=approx_listing_chars,
            output_chars=len(envelope),
            measure_type="estimate",
            ctx={"pattern": pattern, "scope": scope, "total": total},
        )
    except Exception:
        # Savings ledger is best-effort. A failure here must never
        # block the refusal itself.
        pass
