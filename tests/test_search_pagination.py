"""Tests for paginated <search_result/> envelope on grep and glob."""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from _config_helpers import make_config
from llm_solver.harness.tools import glob_files, grep_files


def _parse_envelope(text: str) -> dict:
    """Extract attributes from a <search_result .../> opening tag.

    Use re.search because a unified <tool_result> may wrap the inner tag.
    """
    m = re.search(r'<search_result ([^>]*)>', text)
    assert m is not None, f"no envelope: {text!r}"
    attrs = {}
    for match in re.finditer(r'(\w+)="([^"]*)"', m.group(1)):
        attrs[match.group(1)] = match.group(2)
    return attrs


class TestGlobPagination:

    def test_disabled_returns_raw_lines(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        cfg = make_config(search_pagination_enabled=False)
        result = glob_files("*.py", cwd=str(tmp_path), cfg=cfg)
        assert "<search_result" not in result
        assert "a.py" in result

    def test_envelope_on_single_match(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        cfg = make_config(search_pagination_enabled=True,
                          glob_max_matches_per_page=25)
        result = glob_files("*.py", cwd=str(tmp_path), cfg=cfg)
        attrs = _parse_envelope(result)
        assert attrs["tool"] == "glob"
        assert attrs["total"] == "1"
        assert attrs["shown"] == "1"
        assert attrs["page"] == "1"
        assert attrs["next_page"] == "0"
        assert "a.py" in result

    def test_multi_page_next_page_pointer(self, tmp_path):
        for i in range(7):
            (tmp_path / f"f{i}.py").write_text("")
        cfg = make_config(search_pagination_enabled=True,
                          glob_max_matches_per_page=3)
        page1 = glob_files("*.py", cwd=str(tmp_path), cfg=cfg, page=1)
        attrs1 = _parse_envelope(page1)
        assert attrs1["total"] == "7"
        assert attrs1["shown"] == "3"
        assert attrs1["next_page"] == "2"
        page3 = glob_files("*.py", cwd=str(tmp_path), cfg=cfg, page=3)
        attrs3 = _parse_envelope(page3)
        assert attrs3["shown"] == "1"
        assert attrs3["next_page"] == "0"

    def test_empty_match_envelope(self, tmp_path):
        cfg = make_config(search_pagination_enabled=True)
        result = glob_files("*.py", cwd=str(tmp_path), cfg=cfg)
        attrs = _parse_envelope(result)
        assert attrs["total"] == "0"
        assert attrs["shown"] == "0"
        assert attrs["next_page"] == "0"

    def test_pattern_attr_xml_escaped(self, tmp_path):
        cfg = make_config(search_pagination_enabled=True)
        result = glob_files('foo&bar"baz', cwd=str(tmp_path), cfg=cfg)
        assert "&amp;" in result
        assert "&quot;" in result


class TestGrepPagination:

    def test_disabled_returns_raw(self, tmp_path):
        (tmp_path / "a.py").write_text("needle\nother\n")
        cfg = make_config(search_pagination_enabled=False)
        result = grep_files("needle", cwd=str(tmp_path), cfg=cfg)
        assert "<search_result" not in result

    def test_envelope_on_match(self, tmp_path):
        (tmp_path / "a.py").write_text("needle here\nother\nneedle too\n")
        cfg = make_config(search_pagination_enabled=True,
                          grep_max_matches_per_page=25)
        result = grep_files("needle", cwd=str(tmp_path), cfg=cfg)
        attrs = _parse_envelope(result)
        assert attrs["tool"] == "grep"
        assert int(attrs["total"]) == 2
        assert "needle here" in result
        assert "needle too" in result

    def test_multi_page(self, tmp_path):
        lines = "\n".join(f"match {i}" for i in range(10))
        (tmp_path / "a.py").write_text(lines)
        cfg = make_config(search_pagination_enabled=True,
                          grep_max_matches_per_page=4)
        page1 = grep_files("match", cwd=str(tmp_path), cfg=cfg, page=1)
        attrs1 = _parse_envelope(page1)
        assert attrs1["total"] == "10"
        assert attrs1["shown"] == "4"
        assert attrs1["next_page"] == "2"
        page3 = grep_files("match", cwd=str(tmp_path), cfg=cfg, page=3)
        attrs3 = _parse_envelope(page3)
        assert attrs3["shown"] == "2"
        assert attrs3["next_page"] == "0"


class TestDispatchSurface:

    def test_dispatch_passes_cfg_and_page(self, tmp_path):
        from llm_solver.harness.tools import dispatch
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text("")
        cfg = make_config(search_pagination_enabled=True,
                          glob_max_matches_per_page=2)
        result = dispatch(
            "glob", {"pattern": "*.py", "page": 2},
            cwd=str(tmp_path), cfg=cfg,
        )
        attrs = _parse_envelope(result)
        assert attrs["page"] == "2"
        assert int(attrs["total"]) == 5
