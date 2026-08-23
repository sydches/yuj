"""Focused tests for the reusable structural-index leaf module."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from llm_solver.harness.structural_index import (
    StructuralIndex,
    StructuralRow,
    TreeSitterTagExtractor,
    format_rows,
)


class _RecordingExtractor:
    """Small deterministic backend for index policy/cache tests."""

    def __init__(self) -> None:
        self.extracted: list[str] = []

    def detect_language(self, path: Path) -> str | None:
        return "fixture" if path.suffix == ".fixture" else None

    def extract(
        self,
        source: bytes,
        *,
        language: str,
        display_path: str,
    ) -> tuple[StructuralRow, ...]:
        self.extracted.append(display_path)
        rows = []
        for number, raw in enumerate(source.decode().splitlines(), start=1):
            action, _, name = raw.partition(" ")
            if action not in {"DEF", "REF"} or not name:
                continue
            kind = "def" if action == "DEF" else "ref"
            rows.append(
                StructuralRow(
                    path=display_path,
                    line=number,
                    column=1,
                    kind=kind,
                    name=name,
                    signature=f"{action.lower()} {name}",
                    language=language,
                    capture=f"{'definition' if kind == 'def' else 'reference'}.fixture",
                )
            )
        return tuple(rows)


def test_repository_search_is_sorted_filtered_and_paginated(tmp_path):
    (tmp_path / "z.fixture").write_text("REF target\nDEF target\n")
    (tmp_path / "a.fixture").write_text("DEF target\nREF other\n")
    index = StructuralIndex(tmp_path, extractor=_RecordingExtractor())

    first = index.search(symbol="target", page=1, per_page=2, max_rows=10)
    second = index.search(symbol="target", page=2, per_page=2, max_rows=10)

    assert [(row.path, row.line, row.kind) for row in first.rows] == [
        ("a.fixture", 1, "def"),
        ("z.fixture", 1, "ref"),
    ]
    assert [(row.path, row.line, row.kind) for row in second.rows] == [
        ("z.fixture", 2, "def"),
    ]
    assert first.total == 3
    assert first.available == 3
    assert first.next_page == 2
    assert second.next_page == 0
    assert second.cache_hits == 2


def test_content_hash_cache_reparses_only_changed_file(tmp_path):
    source = tmp_path / "code.fixture"
    source.write_text("DEF before\n")
    extractor = _RecordingExtractor()
    index = StructuralIndex(tmp_path, extractor=extractor)

    assert index.search().cache_hits == 0
    assert index.search().cache_hits == 1
    source.write_text("DEF after\n")
    changed = index.search()

    assert changed.cache_hits == 0
    assert [row.name for row in changed.rows] == ["after"]
    assert extractor.extracted == ["code.fixture", "code.fixture"]


def test_unreadable_file_and_directory_are_never_loaded(tmp_path):
    (tmp_path / "visible.fixture").write_text("DEF visible\n")
    secret = tmp_path / "secret.fixture"
    secret.write_text("DEF hidden_file\n")
    blocked_dir = tmp_path / "blocked"
    blocked_dir.mkdir()
    (blocked_dir / "nested.fixture").write_text("DEF hidden_dir\n")
    extractor = _RecordingExtractor()

    result = StructuralIndex(
        tmp_path,
        extractor=extractor,
        unreadable_paths=(str(secret), str(blocked_dir)),
    ).search()

    assert [row.name for row in result.rows] == ["visible"]
    assert extractor.extracted == ["visible.fixture"]


def test_read_diagnostic_does_not_retain_absolute_host_path(tmp_path, monkeypatch):
    source = tmp_path / "broken.fixture"
    source.write_text("DEF hidden\n")
    original_read_bytes = Path.read_bytes

    def failed_read(path: Path) -> bytes:
        if path.resolve() == source.resolve():
            raise PermissionError(13, "permission denied", str(path))
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", failed_read)
    snapshot = StructuralIndex(tmp_path, extractor=_RecordingExtractor()).scan()

    assert snapshot.rows == ()
    assert len(snapshot.diagnostics) == 1
    assert snapshot.diagnostics[0].path == "broken.fixture"
    assert snapshot.diagnostics[0].message == (
        "could not read source file (PermissionError errno=13)"
    )
    assert str(tmp_path) not in str(snapshot.diagnostics)


def test_max_rows_and_character_cap_are_explicit(tmp_path):
    for number in range(5):
        (tmp_path / f"{number}.fixture").write_text(f"DEF item{number}\n")
    page = StructuralIndex(tmp_path, extractor=_RecordingExtractor()).search(
        max_rows=3,
        per_page=10,
    )

    assert page.total == 5
    assert page.available == 3
    assert page.capped is True
    rendered = format_rows(page.rows, max_output_chars=len(page.rows[0].render()))
    assert rendered.shown == 1
    assert rendered.char_limited is True
    assert len(rendered.text) <= len(page.rows[0].render())


def _real_tag_extractor() -> TreeSitterTagExtractor:
    tree_sitter = pytest.importorskip("tree_sitter")
    language_pack = pytest.importorskip("tree_sitter_language_pack")
    grammar_modules = {
        "python": pytest.importorskip("tree_sitter_python"),
        "javascript": pytest.importorskip("tree_sitter_javascript"),
        "typescript": pytest.importorskip("tree_sitter_typescript"),
        "go": pytest.importorskip("tree_sitter_go"),
        "rust": pytest.importorskip("tree_sitter_rust"),
        "java": pytest.importorskip("tree_sitter_java"),
    }

    def load_language(name: str):
        module = grammar_modules[name]
        if name == "typescript":
            capsule = module.language_typescript()
        else:
            capsule = module.language()
        return tree_sitter.Language(capsule)

    return TreeSitterTagExtractor(
        language_loader=load_language,
        tags_query_loader=language_pack.get_tags_query,
    )


@pytest.mark.parametrize(
    ("filename", "definition", "reference", "source"),
    [
        (
            "sample.py",
            "greet",
            "greet",
            "def greet(name: str) -> str:\n"
            "    return name\n\n"
            "result = greet('Ada')\n",
        ),
        (
            "sample.js",
            "greet",
            "greet",
            "function greet(name) { return name; }\n"
            "const result = greet('Ada');\n",
        ),
        (
            "sample.ts",
            "greet",
            "greet",
            "function greet(name: string): string { return name; }\n"
            "const result = greet('Ada');\n",
        ),
        (
            "sample.go",
            "greet",
            "greet",
            "package sample\n"
            "func greet(name string) string { return name }\n"
            "func run() { greet(\"Ada\") }\n",
        ),
        (
            "sample.rs",
            "greet",
            "greet",
            "fn greet(name: &str) -> &str { name }\n"
            "fn run() { greet(\"Ada\"); }\n",
        ),
        (
            "Sample.java",
            "greet",
            "greet",
            "class Sample {\n"
            "  static String greet(String name) { return name; }\n"
            "  void run() { greet(\"Ada\"); }\n"
            "}\n",
        ),
    ],
)
def test_real_language_pack_tag_queries_find_definitions_and_references(
    tmp_path,
    filename,
    definition,
    reference,
    source,
):
    (tmp_path / filename).write_text(source)
    index = StructuralIndex(tmp_path, extractor=_real_tag_extractor())

    definitions = index.search(symbol=definition, kind="def")
    references = index.search(symbol=reference, kind="ref")

    assert any(row.name == definition for row in definitions.rows)
    assert any(row.name == reference for row in references.rows)
    assert all(row.path == filename for row in definitions.rows + references.rows)
    assert all(row.signature for row in definitions.rows)
