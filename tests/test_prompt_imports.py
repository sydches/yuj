"""Focused tests for deterministic, policy-bounded prompt imports."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from llm_solver.harness.prompt_imports import process_imports


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_nested_relative_and_allowed_absolute_imports_return_ordered_tree(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = _write(root / "AGENTS.md", "ROOT\n@./rules/one.md\nTAIL\n")
    _write(root / "rules" / "one.md", "ONE\n@../shared.md\n")
    shared = _write(root / "shared.md", "SHARED")

    result = process_imports(
        source.read_text() + f"@{shared}\n",
        source.parent,
        (root,),
        source_path=source,
    )

    assert result.content == "ROOT\nONE\nSHARED\nTAIL\nSHARED\n"
    assert result.imported_files == (
        "rules/one.md",
        "shared.md",
        "shared.md",
    )
    assert result.imported_bytes == sum(
        len(path.read_bytes())
        for path in (root / "rules" / "one.md", shared, shared)
    )
    tree = result.trace_tree()
    assert [node["path"] for node in tree] == ["rules/one.md", "shared.md"]
    assert tree[0]["children"][0]["path"] == "shared.md"
    assert tree[0]["children"][0]["status"] == "loaded"
    assert str(tmp_path) not in str(tree)


def test_cycle_is_stopped_before_the_repeated_file_is_read(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    source = _write(root / "a.md", "A\n@b.md\n")
    imported = _write(root / "b.md", "B\n@a.md\n")
    original_read_bytes = Path.read_bytes
    reads: list[Path] = []

    def counted_read(path: Path) -> bytes:
        reads.append(path.resolve())
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read)
    result = process_imports(
        source.read_text(),
        root,
        (root,),
        source_path=source,
    )

    assert reads == [imported.resolve()]
    assert result.imports[0].children[0].status == "cycle"
    assert 'status="cycle" path="a.md"' in result.content


def test_depth_cap_is_zero_based_and_prevents_excess_file_read(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    source = _write(root / "root.md", "@one.md\n")
    first = _write(root / "one.md", "ONE\n@two.md\n")
    second = _write(root / "two.md", "MUST_NOT_READ")
    original_read_bytes = Path.read_bytes

    def guarded_read(path: Path) -> bytes:
        if path.resolve() == second.resolve():
            raise AssertionError("depth-exceeded file was read")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    result = process_imports(
        source.read_text(),
        root,
        (root,),
        max_depth=1,
        source_path=source,
    )

    assert result.imported_files == ("one.md",)
    assert result.imported_bytes == len(first.read_bytes())
    assert result.imports[0].children[0].status == "depth_exceeded"
    assert 'status="depth_exceeded" path="two.md"' in result.content


def test_outside_and_symlink_escape_are_rejected_before_read(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    outside = _write(tmp_path / "secret.md", "SECRET")
    (root / "link.md").symlink_to(outside)
    original_read_bytes = Path.read_bytes

    def guarded_read(path: Path) -> bytes:
        if path.resolve() == outside.resolve():
            raise AssertionError("outside file was read")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    result = process_imports(
        "@../secret.md\n@link.md\n",
        root,
        (root,),
    )

    assert result.imported_files == ()
    assert [node.status for node in result.imports] == [
        "outside_allowed_dirs",
        "outside_allowed_dirs",
    ]
    assert "SECRET" not in result.content
    assert str(tmp_path) not in result.content
    assert str(tmp_path) not in str(result.trace_tree())

    absolute = process_imports(f"@{outside}\n", root, (root,))
    assert absolute.imports[0].request == "secret.md"
    assert str(tmp_path) not in str(absolute.trace_tree())


def test_unreadable_path_and_glob_are_rejected_before_read(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    blocked = _write(root / "private.md", "PRIVATE")
    wildcard = _write(root / "credentials-secret.md", "CREDENTIALS")
    visible = _write(root / "visible.md", "VISIBLE")
    original_read_bytes = Path.read_bytes

    def guarded_read(path: Path) -> bytes:
        if path.resolve() in {blocked.resolve(), wildcard.resolve()}:
            raise AssertionError("unreadable file was read")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    result = process_imports(
        "@private.md\n@credentials-secret.md\n@visible.md\n",
        root,
        (root,),
        unreadable_paths=(str(blocked), str(root / "credentials-*.md")),
    )

    assert [node.status for node in result.imports] == [
        "unreadable",
        "unreadable",
        "loaded",
    ]
    assert result.imported_files == ("visible.md",)
    assert result.imported_bytes == len(visible.read_bytes())
    assert "VISIBLE" in result.content
    assert "PRIVATE" not in result.content


def test_fenced_indented_and_inline_code_remain_literal(tmp_path):
    root = tmp_path / "repo"
    _write(root / "real.md", "EXPANDED")
    text = (
        "```md\n"
        "@real.md\n"
        "```\n"
        "~~~~\n"
        "@real.md\n"
        "~~~~\n"
        "    @real.md\n"
        "`@real.md`\n"
        "Use `@real.md` literally.\n"
        "A multiline `code span starts here\n"
        "@real.md\n"
        "and ends here`.\n"
        "@real.md\n"
    )

    result = process_imports(text, root, (root,))

    assert result.content.count("@real.md") == 6
    assert result.content.endswith("EXPANDED\n")
    assert result.imported_files == ("real.md",)


def test_missing_and_non_markdown_imports_leave_error_comments(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _write(root / "rules.txt", "DO NOT LOAD")

    result = process_imports("@missing.md\n@rules.txt\n", root, (root,))

    assert [node.status for node in result.imports] == ["missing", "not_markdown"]
    assert '<!-- yuj-import-error status="missing" path="missing.md" -->' in result.content
    assert '<!-- yuj-import-error status="not_markdown" path="rules.txt" -->' in result.content
    assert "DO NOT LOAD" not in result.content


def test_utf8_bom_byte_accounting_and_newline_preservation(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    imported = root / "unicode.md"
    imported.write_bytes(b"\xef\xbb\xbfRULE \xc3\xa9")

    result = process_imports("before\r\n@unicode.md\r\nafter", root, (root,))

    assert result.content == "before\r\nRULE é\r\nafter"
    assert result.imported_bytes == len(imported.read_bytes())
    assert result.imports[0].byte_count == len(imported.read_bytes())


@pytest.mark.parametrize("max_depth", [-1, True, 1.5])
def test_rejects_invalid_depth(tmp_path, max_depth):
    with pytest.raises(ValueError, match="non-negative integer"):
        process_imports("", tmp_path, (tmp_path,), max_depth=max_depth)


def test_requires_an_explicit_allowed_directory(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        process_imports("", tmp_path, ())

    file_root = _write(tmp_path / "not-a-directory", "content")
    with pytest.raises(ValueError, match="not a directory"):
        process_imports("", tmp_path, (file_root,))

    with pytest.raises(ValueError, match="base is not a directory"):
        process_imports("", file_root, (tmp_path,))
