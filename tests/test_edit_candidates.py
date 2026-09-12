"""Tests for strict-mode edit() with ranked-candidate surfacing on miss.

Exercises both the replacer-level ``rank_candidates`` and the end-to-end
``edit()`` strict-mode behavior (no mutation, XML <candidates/> block,
cause-hint attribute).
"""
from __future__ import annotations

import re
import random
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from _config_helpers import make_config
from llm_solver.harness import edit_replacers as er
from llm_solver.harness.tools import edit


def _original_block_anchor(text, old):
    """Reference ordering and scoring before the anchor-search optimization."""
    requested = old.split("\n")
    if len(requested) < 3 or not requested[0].strip() or not requested[-1].strip():
        return None
    lines = text.split("\n")
    for first, line in enumerate(lines):
        if line.strip() != requested[0].strip():
            continue
        for last in range(first + 2, len(lines)):
            if lines[last].strip() != requested[-1].strip():
                continue
            pairs = list(zip(requested[1:-1], lines[first + 1:last]))
            score = sum(er._line_similarity(a.strip(), b.strip()) for a, b in pairs)
            if score / len(pairs) >= 0.5:
                start = sum(len(line) + 1 for line in lines[:first])
                return start, min(len(text), sum(len(line) + 1 for line in lines[:last + 1]) - 1)
    return None


def test_candidate_order_matches_original_anchor_search(monkeypatch):
    rng = random.Random(432)
    vocabulary = ["anchor", "end", "x = y", "x = z", "other", "", " anchor "]
    cascade = er.CASCADE
    reference = [(name, _original_block_anchor if name == "block_anchor" else fn)
                 for name, fn in cascade]
    for _ in range(250):
        text = "\n".join(rng.choices(vocabulary, k=30))
        old = "\n".join([rng.choice(["anchor", "end"]),
                         *rng.choices(vocabulary, k=rng.randrange(1, 7)),
                         rng.choice(["anchor", "end"])])
        monkeypatch.setattr(er, "CASCADE", reference)
        expected = er.rank_candidates(text, old)
        monkeypatch.setattr(er, "CASCADE", cascade)
        assert er.rank_candidates(text, old) == expected


def test_repeated_anchors_keep_the_late_candidate():
    prefix = "anchor\nwrong_body\nend\n" * 6000
    tail = "anchor\nx = actual\nend"
    candidates = er.rank_candidates(prefix + tail, "anchor\nx = expected\nend")
    assert len(candidates) == 1
    assert candidates[0].strategy == "block_anchor"
    assert candidates[0].start == len(prefix)
    assert candidates[0].end == len(prefix + tail)
    assert candidates[0].line_number == 18001


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

    @pytest.mark.parametrize("cascade", [False, True])
    @pytest.mark.parametrize("source,old", [(b"aaa", "aa"), (b"a\r\nb\r\na\r\n", "a")])
    def test_ambiguous_exact_edit_preserves_bytes(self, tmp_path, cascade, source, old):
        path = tmp_path / "f.py"
        path.write_bytes(source)
        result = edit("f.py", old, "new", cwd=str(tmp_path),
                      cfg=make_config(edit_fuzzy_cascade_enabled=cascade))
        assert result.startswith("ERROR: old_str matches more than once")
        assert "unique surrounding text" in result
        assert path.read_bytes() == source

    def test_ambiguous_edit_retry_targets_the_named_function(self, tmp_path):
        source = "def first():\n    return True\n\ndef second():\n    return True\n"
        path = tmp_path / "f.py"
        path.write_text(source)
        assert edit("f.py", "return True", "return False", cwd=str(tmp_path)).startswith("ERROR:")
        assert path.read_text() == source
        old = "def second():\n    return True"
        assert edit("f.py", old, old.replace("True", "False"), cwd=str(tmp_path)) == "OK"
        assert path.read_text() == source.replace(old, old.replace("True", "False"))

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
