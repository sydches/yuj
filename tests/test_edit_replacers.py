"""Tests for the fuzzy edit replacer cascade (L5 Harness).

Each strategy is a pure function; all tests exercise it through the
module surface (``find_span`` and the individual replacers) so the
ordering and ledger side-effect can be tested at ``edit`` level
separately.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from llm_solver.harness import edit_replacers as er


# ── line_trimmed ─────────────────────────────────────────────────────────
class TestLineTrimmed:

    def test_trailing_whitespace_on_one_line(self):
        text = "def f():\n    return 1\n"
        old = "def f():\n    return 1   "   # trailing spaces on second line
        span = er.line_trimmed(text, old)
        assert span is not None
        assert text[span[0]:span[1]].rstrip() == "def f():\n    return 1"

    def test_no_match_on_identifier_change(self):
        text = "def foo():\n    return 1\n"
        old = "def bar():\n    return 1"
        assert er.line_trimmed(text, old) is None

    def test_empty_old_str(self):
        assert er.line_trimmed("hello", "") is None


# ── indentation_flexible ────────────────────────────────────────────────
class TestIndentationFlexible:

    def test_dedented_old_against_indented_text(self):
        text = "class C:\n    def m(self):\n        return 1\n"
        # old_str has the method at column 0
        old = "def m(self):\n    return 1"
        span = er.indentation_flexible(text, old)
        assert span is not None
        # Span covers the indented block
        matched = text[span[0]:span[1]]
        assert "def m(self):" in matched
        assert "return 1" in matched

    def test_non_whitespace_exactness_preserved(self):
        text = "class C:\n    def m(self):\n        return 1\n"
        old = "def m(self):\n    return 2"   # wrong return value
        assert er.indentation_flexible(text, old) is None


# ── escape_normalized ───────────────────────────────────────────────────
class TestEscapeNormalized:

    def test_literal_newline_escape(self):
        text = "a\nb\n"
        old = "a\\nb"   # model sent the two-char sequence \n instead of newline
        span = er.escape_normalized(text, old)
        assert span is not None
        assert text[span[0]:span[1]] == "a\nb"

    def test_no_change_when_no_escapes_present(self):
        assert er.escape_normalized("hello", "hello") is None

    def test_literal_tab_escape(self):
        text = "a\tb\n"
        old = "a\\tb"
        span = er.escape_normalized(text, old)
        assert span is not None
        assert text[span[0]:span[1]] == "a\tb"


# ── trimmed_boundary ────────────────────────────────────────────────────
class TestTrimmedBoundary:

    def test_leading_and_trailing_whitespace_stripped(self):
        text = "pass\n"
        old = "   pass   "
        span = er.trimmed_boundary(text, old)
        assert span is not None
        assert text[span[0]:span[1]] == "pass"

    def test_noop_when_already_trimmed(self):
        assert er.trimmed_boundary("pass\n", "pass") is None


# ── block_anchor ────────────────────────────────────────────────────────
class TestBlockAnchor:

    def test_requires_three_lines(self):
        text = "a\nb\n"
        old = "a\nb"
        assert er.block_anchor(text, old) is None

    def test_anchors_plus_similar_interior(self):
        text = "def f():\n    x = compute()\n    return x\n"
        # interior drifted from 'x = compute()' to 'x = result()'
        old = "def f():\n    x = result()\n    return x"
        span = er.block_anchor(text, old)
        # token-similarity between 'x = compute()' and 'x = result()' is
        # 2/3 which is above the 0.5 threshold
        assert span is not None

    def test_rejects_when_interior_too_different(self):
        text = "def f():\n    x = 1\n    y = 2\n    z = 3\n    return x\n"
        old = "def f():\n    nope nope nope nope\n    return x"
        assert er.block_anchor(text, old) is None


# ── find_span cascade ordering ──────────────────────────────────────────
class TestFindSpan:

    def test_prefers_whitespace_normalized_before_indentation_flexible(self):
        """On a text where both strategies could match, the
        whitespace-normalized strategy should fire first and be named."""
        text = "def f():\n    return 1\n"
        old = "def f():\n\treturn 1"   # tab instead of spaces
        hit = er.find_span(text, old)
        assert hit is not None
        mechanism, start, end = hit
        assert mechanism == "whitespace_normalized"

    def test_returns_none_when_all_strategies_fail(self):
        text = "hello world\n"
        old = "goodbye world"
        assert er.find_span(text, old) is None

    def test_falls_through_to_block_anchor(self):
        """No whitespace, line-trim, escape, or trimmed-boundary match;
        cascade should fall through to block_anchor."""
        text = "def f():\n    a = 1\n    b = 2\n    return a\n"
        old = "def f():\n    a = 2\n    return a"  # interior changed
        hit = er.find_span(text, old)
        # block_anchor fires (anchors match; interior similarity above
        # threshold because 'a = 1' vs 'a = 2' shares 'a' and '=' tokens).
        # The earlier strategies should all return None for this input.
        assert hit is not None
        assert hit[0] == "block_anchor"
