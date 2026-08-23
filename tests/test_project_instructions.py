"""Focused tests for deterministic project-instruction discovery."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from llm_solver.harness.project_instructions import (
    discover_project_instructions,
    find_project_root,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_global_and_project_walk_order_override_precedence_and_fallback(tmp_path):
    global_dir = tmp_path / "global"
    root = tmp_path / "repo"
    cwd = root / "packages" / "worker"
    (root / ".git").mkdir(parents=True)
    cwd.mkdir(parents=True)

    _write(global_dir / "AGENTS.md", "GLOBAL_DEFAULT")
    _write(global_dir / "AGENTS.override.md", "GLOBAL_OVERRIDE")
    _write(root / "AGENTS.md", "ROOT_AGENTS")
    _write(root / "CLAUDE.md", "ROOT_FALLBACK_NOT_USED")
    _write(root / "packages" / "AGENTS.md", "PACKAGE_DEFAULT_NOT_USED")
    _write(root / "packages" / "AGENTS.override.md", "PACKAGE_OVERRIDE")
    _write(cwd / "AGENTS.md", "   \n")
    _write(cwd / "CLAUDE.md", "WORKER_FALLBACK")

    result = discover_project_instructions(cwd, global_dir=global_dir)

    assert result.files == (
        "global/AGENTS.override.md",
        "AGENTS.md",
        "packages/AGENTS.override.md",
        "packages/worker/CLAUDE.md",
    )
    positions = [
        result.content.index(marker)
        for marker in (
            "GLOBAL_OVERRIDE",
            "ROOT_AGENTS",
            "PACKAGE_OVERRIDE",
            "WORKER_FALLBACK",
        )
    ]
    assert positions == sorted(positions)
    assert "GLOBAL_DEFAULT" not in result.content
    assert "ROOT_FALLBACK_NOT_USED" not in result.content
    assert "PACKAGE_DEFAULT_NOT_USED" not in result.content
    assert result.content.count("<project-instructions path=") == 4


def test_find_project_root_uses_nearest_marker_or_cwd(tmp_path):
    outer = tmp_path / "outer"
    inner = outer / "inner"
    cwd = inner / "src"
    (outer / ".git").mkdir(parents=True)
    (inner / ".hg").mkdir(parents=True)
    cwd.mkdir(parents=True)

    assert find_project_root(cwd) == inner.resolve()

    unmarked = tmp_path / "plain" / "child"
    unmarked.mkdir(parents=True)
    assert find_project_root(unmarked, markers=("never.marker",)) == unmarked.resolve()


def test_empty_file_skip_and_utf8_byte_cap(tmp_path):
    root = tmp_path / "repo"
    child = root / "child"
    (root / ".git").mkdir(parents=True)
    child.mkdir()
    _write(root / "AGENTS.md", "\n\t")
    _write(root / "CLAUDE.md", "ROOT")
    _write(child / "AGENTS.md", "ééé")

    result = discover_project_instructions(child, max_bytes=9)

    assert result.files == ("CLAUDE.md", "child/AGENTS.md")
    assert result.documents[0].content == "ROOT"
    assert result.documents[1].content == "éé"
    assert result.documents[1].byte_count == 4
    assert result.document_bytes == 8
    assert result.truncated is True
    assert len("".join(doc.content for doc in result.documents).encode()) <= 9


def test_unreadable_candidates_are_skipped_before_read(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    blocked_dir = root / "private"
    cwd = blocked_dir / "nested"
    (root / ".git").mkdir(parents=True)
    cwd.mkdir(parents=True)
    blocked_override = root / "AGENTS.override.md"
    _write(blocked_override, "MUST_NOT_LOAD")
    _write(root / "AGENTS.md", "VISIBLE_FALLBACK")
    _write(blocked_dir / "AGENTS.md", "BLOCKED_DIR")
    _write(cwd / "AGENTS.md", "BLOCKED_NESTED")

    original_read_bytes = Path.read_bytes

    def guarded_read(path: Path) -> bytes:
        resolved = path.resolve()
        if resolved == blocked_override.resolve() or blocked_dir.resolve() in resolved.parents:
            raise AssertionError(f"blocked path was read: {path.name}")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    result = discover_project_instructions(
        cwd,
        unreadable_paths=(str(blocked_override), str(blocked_dir)),
    )

    assert result.files == ("AGENTS.md",)
    assert "VISIBLE_FALLBACK" in result.content
    assert "MUST_NOT_LOAD" not in result.content


def test_unreadable_glob_masks_matching_files(tmp_path):
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    _write(root / "AGENTS.override.md", "BLOCKED")
    _write(root / "AGENTS.md", "FALLBACK")

    result = discover_project_instructions(
        root,
        unreadable_paths=(str(root / "*.override.md"),),
    )

    assert result.files == ("AGENTS.md",)
    assert "FALLBACK" in result.content


def test_read_diagnostic_does_not_retain_absolute_host_path(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    source = root / "AGENTS.md"
    source.write_text("RULE")
    original_read_bytes = Path.read_bytes

    def failed_read(path: Path) -> bytes:
        if path.resolve() == source.resolve():
            raise PermissionError(13, "permission denied", str(path))
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", failed_read)
    result = discover_project_instructions(root)

    assert result.documents == ()
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].path == "AGENTS.md"
    assert result.diagnostics[0].message == (
        "could not read instruction file (PermissionError errno=13)"
    )
    assert str(tmp_path) not in str(result.diagnostics)


def test_symlink_outside_scope_is_not_loaded(tmp_path):
    root = tmp_path / "repo"
    outside = tmp_path / "outside.md"
    (root / ".git").mkdir(parents=True)
    outside.write_text("OUTSIDE")
    (root / "AGENTS.override.md").symlink_to(outside)
    _write(root / "AGENTS.md", "INSIDE")

    result = discover_project_instructions(root)

    assert result.files == ("AGENTS.md",)
    assert "INSIDE" in result.content
    assert "OUTSIDE" not in result.content


def test_developer_text_precedes_files_and_trace_paths_are_safe(tmp_path):
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    _write(root / "AGENTS.md", "PROJECT")

    result = discover_project_instructions(
        root,
        developer_instructions="DEVELOPER",
    )

    assert result.content.index("DEVELOPER") < result.content.index("PROJECT")
    assert result.trace_records() == [
        {
            "path": "AGENTS.md",
            "bytes": 7,
            "scope": "project",
            "truncated": False,
        }
    ]
    assert str(tmp_path) not in result.content
    assert str(tmp_path) not in str(result.trace_records())


@pytest.mark.parametrize(
    "bad_name",
    ["", "../AGENTS.md", "nested/AGENTS.md", "."],
)
def test_rejects_non_filename_configuration_entries(tmp_path, bad_name):
    with pytest.raises(ValueError, match="filenames"):
        discover_project_instructions(tmp_path, doc_names=(bad_name,))
