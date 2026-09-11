"""Tests for paginated <search_result/> envelope on grep and glob."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from _config_helpers import make_config
from llm_solver.harness.tools import glob_files, grep_files


def test_glob_does_not_enumerate_outside_symlink_targets(tmp_path):
    root = tmp_path / "task"
    root.mkdir()
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "answer.py").write_text("private")
    (root / "source.py").write_text("local")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    (root / "alias.py").symlink_to(outside / "answer.py")
    (root / "local.py").symlink_to(root / "source.py")
    assert glob_files("escape/*.py", cwd=str(root)) == "No files found."
    result = glob_files("*.py", cwd=str(root))
    assert "alias.py" not in result
    assert "local.py" in result
    assert "source.py" in result


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


@pytest.mark.parametrize("pagination", [False, True])
@pytest.mark.parametrize("workspace_alias", [False, True])
def test_glob_containment_preserves_aliases_and_page_counts(
    tmp_path, pagination, workspace_alias,
):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "unrelated"
    outside.mkdir()
    (outside / "private-result.txt").write_text("outside")
    (root / "source.txt").write_text("inside")
    (root / "local.txt").symlink_to("source.txt")
    (root / "chain.txt").symlink_to("local.txt")
    (root / "escape-file.txt").symlink_to(outside / "private-result.txt")
    (root / "escape-chain.txt").symlink_to("escape-file.txt")
    (root / "src").mkdir()
    (root / "src" / "keep.txt").write_text("inside directory")
    (outside / "private-alias.txt").symlink_to(root / "source.txt")
    (outside / "private-dir").symlink_to(root / "src", target_is_directory=True)
    (root / "link").symlink_to("src", target_is_directory=True)
    (root / "escape").symlink_to(outside, target_is_directory=True)
    workspace = tmp_path / "workspace" if workspace_alias else root
    if workspace_alias:
        workspace.symlink_to(root, target_is_directory=True)
    cfg = make_config(
        search_pagination_enabled=pagination,
        glob_max_matches_per_page=1,
    )

    def check_pages(pattern, expected, path="."):
        for page in range(1, max(1, len(expected)) + 1):
            result = glob_files(
                pattern, path, cwd=str(workspace), cfg=cfg, page=page,
            )
            assert "private-result" not in result
            assert "private-alias" not in result
            assert "private-dir" not in result
            assert "escape-file" not in result
            assert "escape-chain" not in result
            if pagination:
                attrs = _parse_envelope(result)
                assert int(attrs["total"]) == len(expected)
                assert int(attrs["shown"]) == int(bool(expected))
                assert int(attrs["next_page"]) == (
                    page + 1 if page < len(expected) else 0
                )
                assert "\n".join(result.splitlines()[1:-1]) == "\n".join(
                    expected[page - 1:page]
                )
            else:
                assert result == ("\n".join(expected) or "No files found.")
                break

    check_pages("*.txt", ["chain.txt", "local.txt", "source.txt"])
    check_pages("*/*.txt", ["link/keep.txt", "src/keep.txt"])
    check_pages("*.txt", ["src/keep.txt"], path="link")
    check_pages("escape/*.txt", [])
    check_pages("escape/*/*.txt", [])
    refused = glob_files("*.txt", "escape", cwd=str(workspace), cfg=cfg)
    assert refused.startswith("ERROR: path escapes cwd:")
    assert "private-result" not in refused


class TestGlobPagination:

    def test_disabled_returns_raw_lines(self, tmp_path):
        (tmp_path / "z.py").write_text("")
        (tmp_path / "a.py").write_text("")
        cfg = make_config(
            transformations_explicit=True,
            output_cleanup_and_normalization=False,
            search_pagination_enabled=False,
        )
        result = glob_files("*.py", cwd=str(tmp_path), cfg=cfg)
        assert "<search_result" not in result
        assert result.splitlines() == ["a.py", "z.py"]

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

    def test_broad_glob_returns_a_sorted_bounded_page(self, tmp_path):
        for name in ("z.py", "a.py", "m.py"):
            (tmp_path / name).write_text("")
        cfg = make_config(
            search_pagination_enabled=True,
            glob_max_matches_per_page=2,
            tools_glob_refuse_unscoped_recursive=True,
        )

        page1 = glob_files("**/*.py", cwd=str(tmp_path), cfg=cfg)
        attrs1 = _parse_envelope(page1)
        assert attrs1["total"] == "3"
        assert attrs1["shown"] == "2"
        assert attrs1["next_page"] == "2"
        assert page1.splitlines()[1:3] == ["a.py", "m.py"]
        assert 'hint="' in page1

        page2 = glob_files("**/*.py", cwd=str(tmp_path), cfg=cfg, page=2)
        attrs2 = _parse_envelope(page2)
        assert attrs2["shown"] == "1"
        assert attrs2["next_page"] == "0"
        assert page2.splitlines()[1] == "z.py"

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

    def test_symlinked_workspace_returns_relative_paths(self, tmp_path):
        real_root = tmp_path / "real"
        real_root.mkdir()
        (real_root / "a.py").write_text("")
        linked_root = tmp_path / "linked_workspace"
        linked_root.symlink_to(real_root, target_is_directory=True)

        result = glob_files("*.py", cwd=str(linked_root), cfg=make_config())

        assert "ERROR" not in result
        assert "a.py" in result
        assert str(real_root) not in result
        assert str(linked_root) not in result


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

    def test_order_and_paths_stay_stable_when_cleanup_factor_is_off(
        self, tmp_path,
    ):
        (tmp_path / "z.py").write_text("needle\n")
        (tmp_path / "a.py").write_text("needle\n")
        cfg = make_config(
            transformations_explicit=True,
            output_cleanup_and_normalization=False,
            search_pagination_enabled=False,
        )

        result = grep_files("needle", cwd=str(tmp_path), cfg=cfg)

        assert str(tmp_path) not in result
        assert result.splitlines() == [
            "./a.py:1:needle",
            "./z.py:1:needle",
        ]

    def test_symlinked_workspace_hides_resolved_host_path(self, tmp_path):
        real_root = tmp_path / "real"
        real_root.mkdir()
        (real_root / "a.py").write_text("needle\n")
        linked_root = tmp_path / "linked_workspace"
        linked_root.symlink_to(real_root, target_is_directory=True)

        result = grep_files("needle", cwd=str(linked_root), cfg=make_config())

        assert "./a.py:1:needle" in result
        assert str(real_root) not in result
        assert str(linked_root) not in result


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
