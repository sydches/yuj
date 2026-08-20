"""Tests for the apply_patch DSL's @@ scope-header disambiguation.

The @@ headers were parsed but ignored in v1 (the parser used them only
as hunk separators; the verifier's _find_unique scanned the whole file
for uniqueness). This file pins the post-fix behavior:

  - Empty headers list → search whole file, same as before.
  - Single header → first matching line narrows the search to lines
    after that match.
  - Multiple headers → nested narrowing, each subsequent header is
    found within the prior scope.
  - Header that doesn't match → PatchVerifyError, no fs mutation.
  - Hunk old_lines that are ambiguous in the whole file but unique
    within a @@ scope → succeeds (the headline behavior the fix
    enables).
  - Old behavior preserved: no header + ambiguous → still rejects.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from llm_solver.harness.apply_patch import (
    Hunk,
    PatchVerifyError,
    _find_unique,
    _scope_start,
    parse_patch,
    verify_and_apply,
)


# ── _scope_start unit tests ──────────────────────────────────────────────

class TestScopeStart:

    def test_empty_headers_returns_zero(self):
        assert _scope_start(["a", "b", "c"], []) == 0

    def test_single_header_finds_first_match(self):
        # Returns index AFTER the matched line.
        assert _scope_start(["a", "class Foo:", "    pass", "b"],
                            ["class Foo:"]) == 2

    def test_single_header_substring_match(self):
        # Header is matched as substring (allows leading whitespace etc.).
        assert _scope_start(["    def bar(self):", "        return 1"],
                            ["def bar"]) == 1

    def test_nested_headers_narrow_progressively(self):
        # @@ class Foo: then @@ def bar(): — second header searched after first.
        lines = [
            "class Foo:",
            "    def bar(self):",
            "        return 1",
            "class Baz:",
            "    def bar(self):",
            "        return 2",
        ]
        # Single header `def bar` would hit line 1 first; nested headers
        # land us inside Baz instead.
        assert _scope_start(lines, ["class Baz:", "def bar"]) == 5

    def test_missing_header_raises(self):
        with pytest.raises(PatchVerifyError, match="@@ scope header not found"):
            _scope_start(["a", "b"], ["nonexistent"])

    def test_nested_missing_within_scope_raises(self):
        # First header matches; second doesn't appear after it → error
        # mentions "within prior @@ scope".
        with pytest.raises(PatchVerifyError, match="within prior @@ scope"):
            _scope_start(["class Foo:", "    pass"],
                         ["class Foo:", "def bar"])


# ── _find_unique with start_idx ──────────────────────────────────────────

class TestFindUniqueWithStartIdx:

    def test_zero_start_unchanged(self):
        # Pre-fix call site (no start_idx) still works via default.
        assert _find_unique(["a", "b", "c"], ["b"]) == 1

    def test_start_idx_skips_earlier_matches(self):
        # `b` appears twice; uniqueness only looks after start_idx.
        lines = ["b", "x", "b", "y"]
        # Whole file: ambiguous → error.
        with pytest.raises(PatchVerifyError, match="2 positions in file"):
            _find_unique(lines, ["b"])
        # Skip past the first b → unique within the remaining window.
        assert _find_unique(lines, ["b"], start_idx=1) == 2

    def test_start_idx_unique_failure_message_mentions_scope(self):
        lines = ["x", "b", "y", "b", "z"]
        with pytest.raises(PatchVerifyError, match="within @@ scope"):
            _find_unique(lines, ["b"], start_idx=1)


# ── End-to-end: parse + verify + apply ───────────────────────────────────

@pytest.fixture
def cwd(tmp_path):
    return tmp_path


def _write_file(cwd: Path, name: str, body: str) -> Path:
    p = cwd / name
    p.write_text(body)
    return p


class TestParseHeaders:
    """Parser captures @@ header text and attaches it to the right hunk."""

    def test_no_header_yields_empty_list(self):
        text = (
            "*** Begin Patch\n"
            "*** Update File: foo.py\n"
            "-old\n"
            "+new\n"
            "*** End Patch\n"
        )
        ops = parse_patch(text)
        assert ops[0].hunks[0].headers == []

    def test_single_header_captured(self):
        text = (
            "*** Begin Patch\n"
            "*** Update File: foo.py\n"
            "@@ class Foo:\n"
            "-old\n"
            "+new\n"
            "*** End Patch\n"
        )
        ops = parse_patch(text)
        assert ops[0].hunks[0].headers == ["class Foo:"]

    def test_nested_headers_on_same_hunk(self):
        text = (
            "*** Begin Patch\n"
            "*** Update File: foo.py\n"
            "@@ class Foo:\n"
            "@@ def bar(self):\n"
            "-old\n"
            "+new\n"
            "*** End Patch\n"
        )
        ops = parse_patch(text)
        assert ops[0].hunks[0].headers == ["class Foo:", "def bar(self):"]

    def test_bare_at_at_no_text_yields_no_header(self):
        # `@@` with no text is a hunk separator only; doesn't add a header.
        text = (
            "*** Begin Patch\n"
            "*** Update File: foo.py\n"
            "@@\n"
            "-old\n"
            "+new\n"
            "*** End Patch\n"
        )
        ops = parse_patch(text)
        assert ops[0].hunks[0].headers == []


class TestEndToEndDisambiguation:
    """The headline behavior: ambiguous-globally-but-unique-in-scope succeeds."""

    def test_ambiguous_without_header_rejected(self, cwd):
        # `    def foo(self):` appears twice in this file (A and B).
        # A patch that matches that single line — without an @@ scope
        # header — is ambiguous and the verifier rejects it.
        body = (
            "class A:\n"
            "    def foo(self):\n"
            "        return 1\n"
            "class B:\n"
            "    def foo(self):\n"
            "        return 2\n"
        )
        _write_file(cwd, "m.py", body)
        text = (
            "*** Begin Patch\n"
            "*** Update File: m.py\n"
            "-    def foo(self):\n"
            "+    def renamed(self):\n"
            "*** End Patch\n"
        )
        ops = parse_patch(text)
        with pytest.raises(PatchVerifyError, match="2 positions in file"):
            verify_and_apply(ops, str(cwd))
        # File untouched.
        assert (cwd / "m.py").read_text() == body

    def test_ambiguous_with_scope_header_resolves(self, cwd):
        # Same ambiguous old_line, now with @@ class B: → unique within scope.
        body = (
            "class A:\n"
            "    def foo(self):\n"
            "        return 1\n"
            "class B:\n"
            "    def foo(self):\n"
            "        return 2\n"
        )
        _write_file(cwd, "m.py", body)
        text = (
            "*** Begin Patch\n"
            "*** Update File: m.py\n"
            "@@ class B:\n"
            "-    def foo(self):\n"
            "+    def renamed(self):\n"
            "*** End Patch\n"
        )
        ops = parse_patch(text)
        verify_and_apply(ops, str(cwd))
        new = (cwd / "m.py").read_text()
        # A's `def foo` untouched, B's renamed.
        assert new.count("    def foo(self):\n") == 1
        assert "    def renamed(self):\n" in new

    def test_ambiguous_with_header_succeeds(self, cwd):
        # Same file. Add `@@ class B:` and the same hunk applies.
        body = (
            "class A:\n"
            "    def foo(self):\n"
            "        return 1\n"
            "class B:\n"
            "    def foo(self):\n"
            "        return 2\n"
        )
        _write_file(cwd, "m.py", body)
        text = (
            "*** Begin Patch\n"
            "*** Update File: m.py\n"
            "@@ class B:\n"
            "     def foo(self):\n"
            "-        return 2\n"
            "+        return 999\n"
            "*** End Patch\n"
        )
        ops = parse_patch(text)
        out = verify_and_apply(ops, str(cwd))
        assert "ok=\"true\"" in out
        new = (cwd / "m.py").read_text()
        # Class A unchanged, class B's foo updated.
        assert "        return 1\n" in new   # A unchanged
        assert "        return 999\n" in new # B updated
        assert "        return 2\n" not in new

    def test_missing_header_clearly_rejected(self, cwd):
        body = "class A:\n    pass\n"
        _write_file(cwd, "m.py", body)
        text = (
            "*** Begin Patch\n"
            "*** Update File: m.py\n"
            "@@ class Nonexistent:\n"
            "-    pass\n"
            "+    return 1\n"
            "*** End Patch\n"
        )
        ops = parse_patch(text)
        with pytest.raises(PatchVerifyError,
                           match="@@ scope header not found"):
            verify_and_apply(ops, str(cwd))
        # File untouched.
        assert (cwd / "m.py").read_text() == body

    def test_no_header_still_works_for_unique_hunk(self, cwd):
        # Regression: pre-fix patches without @@ headers must still apply.
        body = "x = 1\ny = 2\nz = 3\n"
        _write_file(cwd, "m.py", body)
        text = (
            "*** Begin Patch\n"
            "*** Update File: m.py\n"
            "-y = 2\n"
            "+y = 99\n"
            "*** End Patch\n"
        )
        ops = parse_patch(text)
        verify_and_apply(ops, str(cwd))
        assert (cwd / "m.py").read_text() == "x = 1\ny = 99\nz = 3\n"

    def test_nested_header_disambiguation(self, cwd):
        # Two classes, both have __init__. @@ class B: then @@ def __init__
        # picks the right one.
        body = (
            "class A:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "class B:\n"
            "    def __init__(self):\n"
            "        self.x = 2\n"
        )
        _write_file(cwd, "m.py", body)
        text = (
            "*** Begin Patch\n"
            "*** Update File: m.py\n"
            "@@ class B:\n"
            "@@ def __init__(self):\n"
            "-        self.x = 2\n"
            "+        self.x = 999\n"
            "*** End Patch\n"
        )
        ops = parse_patch(text)
        verify_and_apply(ops, str(cwd))
        new = (cwd / "m.py").read_text()
        assert "        self.x = 1\n" in new   # A unchanged
        assert "        self.x = 999\n" in new # B updated
        assert "        self.x = 2\n" not in new
