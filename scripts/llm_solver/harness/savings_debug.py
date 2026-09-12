"""Complete transformation sidecars and their bounded readable previews."""
from pathlib import Path
from typing import Any
import logging

log = logging.getLogger(__name__)
_DEBUG_CONTEXT_CHARS = 48
_DEBUG_SNIPPET_MAX_CHARS = 800


def write_debug_values(
    path: Path, event_id: str, before: str, after: str,
) -> dict[str, Any]:
    debug_dir = path.parent / f"{path.stem}.transform_debug"
    before_path = debug_dir / f"{event_id}.before.txt"
    after_path = debug_dir / f"{event_id}.after.txt"
    debug_fields: dict[str, Any] = {
        "changes": _changed_snippets(before, after),
    }
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        before_path.write_text(before, encoding="utf-8")
        after_path.write_text(after, encoding="utf-8")
        debug_fields["input_full_path"] = str(before_path.relative_to(path.parent))
        debug_fields["output_full_path"] = str(after_path.relative_to(path.parent))
    except (OSError, UnicodeError) as exc:
        log.warning("Transformation debug write failed: %s", exc)
        debug_fields["debug_write_error"] = str(exc)
    return debug_fields


def _changed_snippets(before: str, after: str) -> list[dict[str, Any]]:
    """Return readable, bounded before/after regions for a debug record."""
    # Full before/after bytes are saved alongside this preview.
    return [] if before == after else [_one_changed_region(before, after)]


def _one_changed_region(before: str, after: str) -> dict[str, Any]:
    limit = min(len(before), len(after))
    prefix = _matching_edge(before, after, limit, suffix=False)
    suffix = _matching_edge(before, after, limit - prefix, suffix=True)
    before_tail = len(before) - suffix
    after_tail = len(after) - suffix
    return _change_region(
        before, after, prefix, before_tail, prefix, after_tail,
    )


def _matching_edge(before: str, after: str, limit: int, *, suffix: bool) -> int:
    """Find equal edge length with native string comparisons, not char loops."""
    low, high = 0, limit
    while low < high:
        middle = (low + high + 1) // 2
        left = before[-middle:] if suffix else before[:middle]
        right = after[-middle:] if suffix else after[:middle]
        if left == right:
            low = middle
        else:
            high = middle - 1
    return low


def _change_region(
    before: str,
    after: str,
    i1: int,
    i2: int,
    j1: int,
    j2: int,
) -> dict[str, Any]:
    before_start = max(0, i1 - _DEBUG_CONTEXT_CHARS)
    before_end = min(len(before), i2 + _DEBUG_CONTEXT_CHARS)
    after_start = max(0, j1 - _DEBUG_CONTEXT_CHARS)
    after_end = min(len(after), j2 + _DEBUG_CONTEXT_CHARS)
    return {
        "input_byte_range": [
            len(before[:i1].encode("utf-8")),
            len(before[:i2].encode("utf-8")),
        ],
        "output_byte_range": [
            len(after[:j1].encode("utf-8")),
            len(after[:j2].encode("utf-8")),
        ],
        "before": _bounded_snippet(before[before_start:before_end]),
        "after": _bounded_snippet(after[after_start:after_end]),
    }


def _bounded_snippet(value: str) -> str:
    """Keep a debug JSON region readable; full values live in sidecars."""
    if len(value) <= _DEBUG_SNIPPET_MAX_CHARS:
        return value
    head = _DEBUG_SNIPPET_MAX_CHARS // 2
    tail = _DEBUG_SNIPPET_MAX_CHARS - head
    omitted = len(value) - head - tail
    return (
        value[:head]
        + f"\n[... {omitted} chars omitted from debug snippet ...]\n"
        + value[-tail:]
    )
