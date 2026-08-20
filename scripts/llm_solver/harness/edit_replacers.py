"""Fuzzy edit replacer cascade — strategies for locating ``old_str``
inside a file when the exact bytes in the model's ``old_str`` do not
occur verbatim.

Borrowed in spirit from opencode's nine-strategy cascade
(``sst/opencode`` ``packages/opencode/src/tool/edit.ts``). Trimmed to
the strategies that match our failure distribution — we see
whitespace drift, indentation drift, and serialization escaping
artifacts; we do not see multi-match replace-all requests (the
``edit`` tool contract is first-occurrence).

Each strategy is a pure function ``(text, old_str) -> (start, end) |
None``. Non-whitespace characters MUST still match exactly — the
cascade is a relaxation of whitespace / boundary / escape concerns,
not of identifier content. A one-character typo in a variable name
still fails (and should — we never silently rewrite the wrong
function).

Strategies are ordered cheapest-and-strictest first. ``edit()``
iterates the chain, stops at the first span produced, and records a
``fuzzy_edit_recovery`` event on the savings ledger tagged with the
mechanism that fired.
"""
from __future__ import annotations

import re


# ── Strategy 1: exact ────────────────────────────────────────────────────
def exact(text: str, old_str: str) -> tuple[int, int] | None:
    """Pure substring match; replicates ``str.find`` / ``in``."""
    idx = text.find(old_str)
    if idx < 0:
        return None
    return (idx, idx + len(old_str))


# ── Strategy 2: whitespace-normalized (token regex) ──────────────────────
def whitespace_normalized(text: str, old_str: str) -> tuple[int, int] | None:
    """Split ``old_str`` on whitespace, rejoin with ``\\s+`` regex.

    Handles whitespace runs of different shape (tabs vs spaces,
    extra blank lines) while keeping non-whitespace tokens exact.
    This is the existing ``_whitespace_normalized_match`` logic
    lifted out of ``tools.py`` verbatim.
    """
    tokens = old_str.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(t) for t in tokens)
    m = re.search(pattern, text)
    if m is None:
        return None
    return (m.start(), m.end())


# ── Strategy 3: line-trimmed ─────────────────────────────────────────────
def line_trimmed(text: str, old_str: str) -> tuple[int, int] | None:
    """Match line-by-line after trimming leading/trailing whitespace
    on each line pair.

    Catches the "model has trailing whitespace on one line, file
    doesn't" case without relaxing line ordering or inter-line
    content.
    """
    old_lines = old_str.split("\n")
    if not old_lines:
        return None
    old_trimmed = [ln.strip() for ln in old_lines]

    text_lines = text.split("\n")
    text_trimmed = [ln.strip() for ln in text_lines]

    n = len(old_trimmed)
    if n == 0 or n > len(text_trimmed):
        return None

    # Cumulative byte offset for each line start in ``text``.
    offsets = [0]
    for ln in text_lines:
        offsets.append(offsets[-1] + len(ln) + 1)  # +1 for '\n'

    for i in range(len(text_trimmed) - n + 1):
        if text_trimmed[i:i + n] == old_trimmed:
            start = offsets[i]
            # End offset is the start of line i+n minus the trailing '\n'
            # that offsets[i+n] includes — clamp if it runs past text.
            end = offsets[i + n] - 1
            if end > len(text):
                end = len(text)
            return (start, end)
    return None


# ── Strategy 4: indentation-flexible ─────────────────────────────────────
_LEADING_WS = re.compile(r"^[ \t]*")


def indentation_flexible(text: str, old_str: str) -> tuple[int, int] | None:
    """Match after stripping all leading indentation on each line.

    Distinct from ``line_trimmed``: this keeps trailing whitespace
    (which may be meaningful) but removes any level of leading
    indentation, so the model can produce the block at column 0 when
    the file has it indented under a class / function.
    """
    def lstrip_lines(s: str) -> list[str]:
        return [_LEADING_WS.sub("", ln) for ln in s.split("\n")]

    old_lines = lstrip_lines(old_str)
    if not old_lines:
        return None

    text_lines = text.split("\n")
    text_lstripped = [_LEADING_WS.sub("", ln) for ln in text_lines]

    n = len(old_lines)
    if n == 0 or n > len(text_lstripped):
        return None

    offsets = [0]
    for ln in text_lines:
        offsets.append(offsets[-1] + len(ln) + 1)

    for i in range(len(text_lstripped) - n + 1):
        if text_lstripped[i:i + n] == old_lines:
            start = offsets[i]
            end = offsets[i + n] - 1
            if end > len(text):
                end = len(text)
            return (start, end)
    return None


# ── Strategy 5: escape-normalized ────────────────────────────────────────
_ESCAPE_MAP = [
    ("\\\\", "\\"),
    ("\\n", "\n"),
    ("\\t", "\t"),
    ("\\r", "\r"),
    ('\\"', '"'),
    ("\\'", "'"),
]


def _unescape(s: str) -> str:
    for literal, real in _ESCAPE_MAP:
        s = s.replace(literal, real)
    return s


def escape_normalized(text: str, old_str: str) -> tuple[int, int] | None:
    """If ``old_str`` differs from ``text`` only by serialization
    escapes (``\\n`` instead of newline, ``\\t`` instead of tab, etc.),
    retry the match after unescaping.

    Only activates when unescaping actually changes the string —
    otherwise it duplicates ``exact`` and is skipped.
    """
    unescaped = _unescape(old_str)
    if unescaped == old_str:
        return None
    idx = text.find(unescaped)
    if idx < 0:
        return None
    return (idx, idx + len(unescaped))


# ── Strategy 6: trimmed-boundary ─────────────────────────────────────────
def trimmed_boundary(text: str, old_str: str) -> tuple[int, int] | None:
    """Try ``old_str.strip()`` (leading/trailing whitespace removed)
    as the literal search key.

    Catches the "model included a trailing newline the file does not
    have" case without modifying internal content.
    """
    stripped = old_str.strip()
    if stripped == old_str or not stripped:
        return None
    idx = text.find(stripped)
    if idx < 0:
        return None
    return (idx, idx + len(stripped))


# ── Strategy 7: block-anchor ─────────────────────────────────────────────
def _line_similarity(a: str, b: str) -> float:
    """Cheap similarity: fraction of shared split tokens over max(len)."""
    at, bt = a.split(), b.split()
    if not at and not bt:
        return 1.0
    if not at or not bt:
        return 0.0
    shared = len(set(at) & set(bt))
    return shared / max(len(at), len(bt))


def block_anchor(text: str, old_str: str) -> tuple[int, int] | None:
    """Anchor on first+last line of ``old_str`` (trimmed); require
    >=50% interior token similarity over the block between anchors.

    The block in ``text`` is identified by scanning for matching
    first and last anchor lines — its line count need not equal
    ``len(old_str.split("\\n"))``. This matches opencode's behavior
    where the anchors define the block and the old_str interior is
    compared against whatever text lines fall between them.

    Only fires when ``old_str`` has at least 3 lines. Returns the
    earliest (smallest-j) block meeting both checks.
    """
    old_lines = old_str.split("\n")
    if len(old_lines) < 3:
        return None

    first = old_lines[0].strip()
    last = old_lines[-1].strip()
    interior = [ln.strip() for ln in old_lines[1:-1]]
    if not first or not last:
        return None

    text_lines = text.split("\n")
    text_trimmed = [ln.strip() for ln in text_lines]

    offsets = [0]
    for ln in text_lines:
        offsets.append(offsets[-1] + len(ln) + 1)

    for i in range(len(text_trimmed)):
        if text_trimmed[i] != first:
            continue
        # Find the earliest j > i whose trimmed content equals the
        # last anchor.
        for j in range(i + 2, len(text_trimmed)):
            if text_trimmed[j] != last:
                continue
            text_interior = text_trimmed[i + 1:j]
            if interior:
                # Pairwise similarity over min(len), averaged.
                pair_count = min(len(interior), len(text_interior))
                if pair_count == 0:
                    # Old has interior but block between anchors is
                    # empty — reject.
                    break
                sims = [
                    _line_similarity(interior[k], text_interior[k])
                    for k in range(pair_count)
                ]
                mean_sim = sum(sims) / pair_count
                if mean_sim < 0.5:
                    continue
            start = offsets[i]
            end = offsets[j + 1] - 1
            if end > len(text):
                end = len(text)
            return (start, end)
    return None


# ── Cascade entry point ──────────────────────────────────────────────────
# Ordered cheapest-and-strictest first. Exact is handled by the caller
# before this module is reached, so it is NOT repeated in the cascade —
# the recorded mechanism name on exact-match is not useful ledger data.
CASCADE: list[tuple[str, object]] = [
    ("whitespace_normalized", whitespace_normalized),
    ("line_trimmed", line_trimmed),
    ("indentation_flexible", indentation_flexible),
    ("escape_normalized", escape_normalized),
    ("trimmed_boundary", trimmed_boundary),
    ("block_anchor", block_anchor),
]


from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """A near-miss span produced by one of the cascade strategies.

    Used in strict-match mode (``edit_fuzzy_cascade_enabled=false``)
    to surface ranked alternatives back to the agent rather than
    silently applying the edit to the best-match span.
    """
    strategy: str
    start: int
    end: int
    similarity: float
    line_number: int


def _line_number(text: str, offset: int) -> int:
    """1-based line index for a byte offset in ``text``."""
    return text.count("\n", 0, offset) + 1


def _similarity(old_str: str, candidate_text: str) -> float:
    """Ratcliff-Obershelp style token-based similarity in [0, 1].

    Cheap and symmetric. Used only to rank candidates; the actual
    match test (identifier exactness) already happened inside the
    strategy that produced the span.
    """
    if old_str == candidate_text:
        return 1.0
    old_tokens = old_str.split()
    cand_tokens = candidate_text.split()
    if not old_tokens and not cand_tokens:
        return 1.0
    if not old_tokens or not cand_tokens:
        return 0.0
    shared = sum(1 for t in old_tokens if t in cand_tokens)
    return shared / max(len(old_tokens), len(cand_tokens))


def rank_candidates(
    text: str, old_str: str, k: int = 3,
) -> list[Candidate]:
    """Run every cascade strategy as a scorer; return top-k spans
    sorted by similarity to ``old_str``.

    Duplicate spans across strategies (same start+end) collapse to
    the highest-similarity entry; strategy-name ordering preserved
    from the cascade when scores tie.
    """
    seen: dict[tuple[int, int], Candidate] = {}
    for name, fn in CASCADE:
        span = fn(text, old_str)
        if span is None:
            continue
        start, end = span
        cand_text = text[start:end]
        sim = _similarity(old_str, cand_text)
        existing = seen.get((start, end))
        if existing is None or sim > existing.similarity:
            seen[(start, end)] = Candidate(
                strategy=name, start=start, end=end,
                similarity=sim, line_number=_line_number(text, start),
            )
    ranked = sorted(seen.values(), key=lambda c: (-c.similarity, c.line_number))
    return ranked[:max(1, k)] if ranked else []


def find_span(text: str, old_str: str) -> tuple[str, int, int] | None:
    """Run the cascade; return ``(mechanism, start, end)`` for the
    first strategy that yields a span, or None if every strategy
    returns None.

    ``exact`` is not part of the cascade — callers handle exact-match
    before invoking ``find_span`` so the common case stays fast and
    avoids a recovery-ledger entry for it.
    """
    for name, fn in CASCADE:
        span = fn(text, old_str)
        if span is not None:
            return (name, span[0], span[1])
    return None
