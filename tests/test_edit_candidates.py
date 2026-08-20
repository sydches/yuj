"""Tests for strict-mode edit() with ranked-candidate surfacing on miss.

Exercises both the replacer-level ``rank_candidates`` and the end-to-end
``edit()`` strict-mode behavior (no mutation, XML <candidates/> block,
cause-hint attribute).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from _config_helpers import make_config
from llm_solver.harness import edit_replacers as er
from llm_solver.harness.tools import edit


class TestRankCandidates:

    def test_empty_on_no_match(self):
        cands = er.rank_candidates("hello world\n", "goodbye")
        assert cands == []

    def test_returns_top_k(self):
        src = "def f():\n\treturn 1\n\ndef g():\n    return 1\n"
        cands = er.rank_candidates(src, "def f():\n    return 1", k=3)
        assert len(cands) >= 1
        assert all(isinstance(c, er.Candidate) for c in cands)
        # Sorted descending by similarity.
        sims = [c.similarity for c in cands]
        assert sims == sorted(sims, reverse=True)

    def test_line_number_is_one_based(self):
        src = "line1\nline2\nline3\n"
        cands = er.rank_candidates(src, "line3   ")  # trailing ws
        assert cands and cands[0].line_number == 3

    def test_k_cap_applied(self):
        src = "\n".join(f"pass" for _ in range(10))
        cands = er.rank_candidates(src, "pass   ", k=2)
        assert len(cands) <= 2


class TestStrictEditMode:

    def test_default_mode_is_strict(self):
        cfg = make_config()
        assert cfg.edit_strict_match is True
        assert cfg.edit_fuzzy_cascade_enabled is False

    def test_exact_hit_applies(self, tmp_path):
        cfg = make_config()
        (tmp_path / "f.py").write_text("old\n")
        result = edit("f.py", "old", "new", cwd=str(tmp_path), cfg=cfg)
        assert result == "OK"
        assert (tmp_path / "f.py").read_text() == "new\n"

    def test_strict_miss_no_mutation(self, tmp_path):
        cfg = make_config()
        src = "def foo():\n\treturn 1\n"
        (tmp_path / "f.py").write_text(src)
        result = edit(
            "f.py",
            "def foo():\n    return 1",  # spaces vs tab
            "new",
            cwd=str(tmp_path), cfg=cfg,
        )
        assert result.startswith("ERROR:")
        assert (tmp_path / "f.py").read_text() == src

    def test_strict_miss_emits_candidates_block(self, tmp_path):
        cfg = make_config()
        src = "def foo():\n\treturn 1\n"
        (tmp_path / "f.py").write_text(src)
        result = edit(
            "f.py",
            "def foo():\n    return 1",
            "new",
            cwd=str(tmp_path), cfg=cfg,
        )
        assert "<candidates" in result
        assert "</candidates>" in result
        # cause_hint attribute present
        assert re.search(r'cause_hint="\w+"', result)
        # at least one <candidate> element
        assert "<candidate " in result

    def test_cascade_arm_auto_applies(self, tmp_path):
        cfg = make_config(edit_strict_match=False,
                          edit_fuzzy_cascade_enabled=True)
        src = "def foo():\n\treturn 1\n"
        (tmp_path / "f.py").write_text(src)
        result = edit(
            "f.py",
            "def foo():\n    return 1",
            "def foo():\n    return 2",
            cwd=str(tmp_path), cfg=cfg,
        )
        assert "OK" in result
        assert "whitespace-normalized" in result
        assert "return 2" in (tmp_path / "f.py").read_text()

    def test_strict_miss_with_no_candidates(self, tmp_path):
        cfg = make_config()
        (tmp_path / "f.py").write_text("alpha\nbeta\n")
        result = edit(
            "f.py", "gamma", "new",
            cwd=str(tmp_path), cfg=cfg,
        )
        assert result.startswith("ERROR:")
        # With zero candidates, no empty block is emitted.
        assert "<candidates" not in result
