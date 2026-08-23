"""Focused tests for the reusable structural-index leaf module."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from llm_solver.harness.structural_index import (
    StructuralBackendUnavailable,
    StructuralIndex,
    StructuralRow,
    TreeSitterTagExtractor,
    format_rows,
)
from _config_helpers import make_config
from llm_solver.config import load_config
from llm_solver.harness._tools import list_definitions as list_definitions_module
from llm_solver.harness._tools.list_definitions import list_definitions
from llm_solver.harness.schemas import get_tool_schemas
from llm_solver.harness.tools import dispatch


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


def test_default_backend_uses_preinstalled_grammars_for_acceptance_languages(tmp_path):
    fixtures = {
        "sample.py": (
            "python_target",
            "def python_target():\n    return 1\npython_target()\n",
        ),
        "sample.js": (
            "javascript_target",
            "function javascript_target() { return 1; }\njavascript_target();\n",
        ),
        "sample.ts": (
            "typescript_target",
            "function typescript_target(): number { return 1; }\ntypescript_target();\n",
        ),
        "sample.go": (
            "go_target",
            "package sample\nfunc go_target() int { return 1 }\nfunc run() { go_target() }\n",
        ),
        "sample.rs": (
            "rust_target",
            "fn rust_target() -> i32 { 1 }\nfn run() { rust_target(); }\n",
        ),
        "Sample.java": (
            "java_target",
            "class Sample { static int java_target() { return 1; } "
            "void run() { java_target(); } }\n",
        ),
    }
    for filename, (_symbol, source) in fixtures.items():
        (tmp_path / filename).write_text(source)

    index = StructuralIndex(tmp_path)
    for symbol, _source in fixtures.values():
        assert index.search(symbol=symbol, kind="def").rows
        assert index.search(symbol=symbol, kind="ref").rows


def test_list_definitions_legacy_single_file_shape_is_unchanged(tmp_path):
    (tmp_path / "one.py").write_text("def alpha(value: int) -> int:\n    return value\n")
    cfg = make_config(tools_list_definitions_enabled=True)

    result = list_definitions("one.py", cwd=str(tmp_path), cfg=cfg)

    assert result == (
        '<list_definitions status="ok" path="one.py" count="1" '
        'surface="0" v="1">\n'
        '# definitions\n'
        '[L   1] def alpha(value: int) -&gt; int\n'
        '</list_definitions>'
    )


def test_repository_mode_requires_both_feature_gates(tmp_path):
    (tmp_path / "one.py").write_text("def alpha():\n    pass\n")
    cfg = make_config(
        tools_list_definitions_enabled=True,
        tools_ast_search_enabled=False,
    )

    result = list_definitions(
        ".", cwd=str(tmp_path), cfg=cfg, repo_wide=True, symbol="alpha",
    )

    assert 'status="error" error_kind="ast_search_disabled"' in result


def test_dispatch_threads_repository_search_pages_cache_and_unreadable_policy(tmp_path):
    (tmp_path / "a.py").write_text("def target():\n    return 1\ntarget()\n")
    (tmp_path / "b.py").write_text("def target():\n    return 2\ntarget()\n")
    secret = tmp_path / "secret.py"
    secret.write_text("def target():\n    return 3\ntarget()\n")
    cfg = make_config(
        tools_list_definitions_enabled=True,
        tools_ast_search_enabled=True,
        tools_ast_search_max_rows=10,
        search_pagination_enabled=True,
        grep_max_matches_per_page=1,
        unreadable_paths=(str(secret),),
        max_output_chars=2000,
    )
    list_definitions_module._clear_structural_index_registry()
    arguments = {
        "path": ".", "repo_wide": True,
        "symbol": "target", "kind": "def", "page": 1,
    }

    first = dispatch("list_definitions", arguments, cwd=str(tmp_path), cfg=cfg)
    second = dispatch(
        "list_definitions", {**arguments, "page": 2},
        cwd=str(tmp_path), cfg=cfg,
    )

    assert first.startswith(
        '<list_definitions status="ok" mode="repository" path="." '
        'total="2" available="2" shown="1" page="1" next_page="2"'
    )
    assert "a.py:1 def target" in first
    assert "secret.py" not in first + second
    assert 'page="2" next_page="0"' in second
    assert 'cache_hits="2"' in second
    assert second.endswith("</list_definitions>")


def test_repository_mode_honors_exact_output_budget_without_cutting_rows(tmp_path):
    for number in range(12):
        (tmp_path / f"file_{number}.py").write_text(
            f"def symbol_{number}_with_a_long_descriptive_name(value: int) -> int:\n"
            "    return value\n"
        )
    cfg = make_config(
        tools_list_definitions_enabled=True,
        tools_ast_search_enabled=True,
        tools_ast_search_max_rows=100,
        search_pagination_enabled=False,
        max_output_chars=420,
    )
    list_definitions_module._clear_structural_index_registry()

    result = list_definitions(
        ".", cwd=str(tmp_path), cfg=cfg, repo_wide=True,
    )

    assert len(result) <= cfg.max_output_chars
    assert result.endswith("</list_definitions>")
    assert 'char_limited="true"' in result
    body = result.split("\n")[1:-1]
    assert all(":1 def symbol_" in row for row in body)


def test_repository_mode_returns_typed_missing_backend_error(tmp_path, monkeypatch):
    class MissingIndex:
        def search(self, **_kwargs):
            raise StructuralBackendUnavailable("missing")

    monkeypatch.setattr(list_definitions_module, "_structural_index", lambda *_a: MissingIndex())
    cfg = make_config(
        tools_list_definitions_enabled=True,
        tools_ast_search_enabled=True,
    )

    result = list_definitions(".", cwd=str(tmp_path), cfg=cfg, repo_wide=True)

    assert 'status="error" error_kind="backend_unavailable"' in result
    assert str(tmp_path) not in result


def test_structural_tool_schema_declares_repository_arguments():
    schema = next(
        item["function"]
        for item in get_tool_schemas("minimal")
        if item["function"]["name"] == "list_definitions"
    )
    properties = schema["parameters"]["properties"]

    assert schema["parameters"]["required"] == ["path"]
    assert properties["symbol"]["type"] == "string"
    assert properties["kind"]["enum"] == ["def", "ref"]
    assert properties["repo_wide"]["type"] == "boolean"
    assert properties["page"]["type"] == "integer"


def test_ast_search_config_defaults_overlay_and_cap_validation(tmp_path):
    defaults = load_config()
    assert defaults.tools_ast_search_enabled is False
    assert defaults.tools_ast_search_max_rows == 1000

    overlay = tmp_path / "ast.toml"
    overlay.write_text(
        "[tools]\nast_search_enabled = true\nast_search_max_rows = 17\n"
    )
    configured = load_config(user_config=overlay)
    assert configured.tools_ast_search_enabled is True
    assert configured.tools_ast_search_max_rows == 17

    for invalid in ("0", "true", "1.5"):
        overlay.write_text(f"[tools]\nast_search_max_rows = {invalid}\n")
        with pytest.raises(ValueError, match="tools.ast_search_max_rows"):
            load_config(user_config=overlay)
