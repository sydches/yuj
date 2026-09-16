"""Focused tests for the reusable structural-index leaf module."""
from __future__ import annotations

from pathlib import Path
import os
import subprocess
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


@pytest.fixture
def native_index(tmp_path):
    from llm_solver.harness.task_files import NamespaceFiles
    from llm_solver.harness.task_path import activate_task_files, bound_task_path

    def run(script, args, data):
        return subprocess.run(
            ['bash', '--noprofile', '--norc', '-c', script, 'index-test', *args],
            input=data, capture_output=True, cwd=tmp_path, check=False,
        )

    files = NamespaceFiles(str(tmp_path), run, binding={'test': 'native-index'})
    with activate_task_files(files, host_root=tmp_path):
        yield files, StructuralIndex(bound_task_path(str(tmp_path), '.'),
                                     extractor=_RecordingExtractor())


def test_native_pages_check_bytes_without_retransferring_unchanged_sources(
    tmp_path, native_index, monkeypatch,
):
    files, index = native_index
    source = tmp_path / 'a.fixture'
    source.write_text('DEF before\nREF target\n' + '# padding\n' * 100_000)
    (tmp_path / 'b.fixture').write_text('DEF target\nREF target\n')
    reads = []
    original_read = files.read_bytes

    def record_read(path):
        reads.append(str(path))
        return original_read(path)

    monkeypatch.setattr(files, 'read_bytes', record_read)
    first = index.search(page=1, per_page=2)
    assert [row.name for row in first.rows] == ['before', 'target']
    assert len(reads) == 2
    reads.clear()
    second = index.search(page=2, per_page=2)
    assert [row.name for row in second.rows] == ['target', 'target']
    assert second.cache_hits == 2
    assert reads == []

    # A writer outside the tool protocol can preserve size and timestamps.
    previous = source.stat()
    source.write_bytes(source.read_bytes().replace(b'before', b'newest'))
    os.utime(source, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    changed = index.search(page=1, per_page=2)
    assert [row.name for row in changed.rows] == ['newest', 'target']
    assert changed.cache_hits == 1
    assert reads == [str(source)]

    source.unlink()
    (tmp_path / 'c.fixture').write_text('DEF added\n')
    refreshed = index.search(page=2, per_page=2)
    assert [row.name for row in refreshed.rows] == ['added']
    assert refreshed.total == 3


def test_structural_scan_checks_supported_sources_before_other_file_metadata(
    tmp_path, native_index, monkeypatch,
):
    files, index = native_index
    (tmp_path / 'code.fixture').write_text('DEF target\n')
    (tmp_path / 'ignored').mkdir()
    (tmp_path / 'ignored' / 'secret.fixture').write_text('DEF secret\n')
    index._ignored_dir_names = frozenset({'ignored'})
    for number in range(23):
        (tmp_path / f'cache{number}.pyc').write_bytes(b'not source')
    checked = []
    original = files.entry_modes

    def track(paths):
        paths = list(paths)
        checked.extend(Path(path).name for path in paths)
        return original(paths)

    monkeypatch.setattr(files, 'entry_modes', track)
    first = index.search()
    assert [row.name for row in first.rows] == ['target']
    assert checked == ['code.fixture']
    assert index.search().cache_hits == 1
    assert not any(name.endswith('.pyc') for name in checked)


def test_native_index_admission_batches_and_rechecks_changed_entries(tmp_path, native_index, monkeypatch):
    files, index = native_index
    for number in range(130):
        (tmp_path / f'f{number}.fixture').write_text(f'DEF item{number}\n')
    assert index.scan().files_scanned == 130
    calls = []
    original = files.run
    def record(script, args, data):
        calls.append(args[3] if len(args) > 3 else 'discovery')
        return original(script, args, data)
    monkeypatch.setattr(files, 'run', record)
    warm = index.scan()
    assert warm.cache_hits == 130
    assert calls.count('entry_modes') == 3
    assert calls.count('resolve_files') == 1
    assert calls.count('digest_batch') == 1
    assert 'symlink' not in calls and 'resolve' not in calls and 'read' not in calls
    (tmp_path / 'f0.fixture').unlink()
    (tmp_path / 'f0.fixture').symlink_to('f1.fixture')
    (tmp_path / 'f2.fixture').write_text('DEF changed\n')
    changed = index.scan()
    assert changed.files_scanned == 129 and changed.cache_hits == 128
    assert 'item0' not in {row.name for row in changed.rows}
    assert 'changed' in {row.name for row in changed.rows}


def test_native_index_batch_keeps_missing_file_diagnostics(tmp_path, native_index, monkeypatch):
    files, index = native_index
    source = tmp_path / 'gone.fixture'
    source.write_text('DEF missing\n')
    original = files.resolve_paths
    def remove_then_resolve(paths):
        source.unlink()
        return original(paths)
    monkeypatch.setattr(files, 'resolve_paths', remove_then_resolve)
    result = index.scan()
    assert result.rows == () and len(result.diagnostics) == 1
    assert result.diagnostics[0].path == 'gone.fixture'
    assert 'FileNotFoundError' in result.diagnostics[0].message


def test_native_index_batch_rejects_late_outside_link(tmp_path, native_index, monkeypatch):
    files, index = native_index
    source = tmp_path / 'swap.fixture'
    source.write_text('DEF initial\n')
    outside = tmp_path.parent / (tmp_path.name + '-outside.fixture')
    outside.write_text('DEF secret\n')
    original = files.resolve_paths
    def swap_then_resolve(paths):
        source.unlink()
        source.symlink_to(outside)
        return original(paths)
    monkeypatch.setattr(files, 'resolve_paths', swap_then_resolve)
    result = index.scan()
    assert result.rows == () and index.extractor.extracted == []


def test_native_index_batch_matches_scalar_mask_and_ignore_policy(tmp_path, native_index, monkeypatch):
    from llm_solver.harness.sandbox.ignore_policy import load_ignore_policy, activate_ignore_policy
    from llm_solver.harness.structural_index import _UnreadableMatcher
    files, index = native_index
    for name in ('visible.fixture', 'secret.fixture', 'ignored.fixture', 'other.fixture'):
        (tmp_path / name).write_text(f'DEF {name}\n')
    (tmp_path / 'folder').mkdir()
    (tmp_path / 'folder/nested.fixture').write_text('DEF nested\n')
    (tmp_path / '.yujignore').write_text('ignored.fixture\nfolder/\n')
    index._unreadable = _UnreadableMatcher(index.root, ['secret.fixture'])
    index._path_globs = ('visible*', 'secret*', 'ignored*', 'folder/*')
    policy = load_ignore_policy(tmp_path)
    with activate_ignore_policy(policy):
        batched = index.scan()
        monkeypatch.setattr(index, '_readable_paths',
                            lambda paths: {path for path in paths if index._is_readable_path(path)})
        scalar = index.scan()
    assert batched.rows == scalar.rows and batched.diagnostics == scalar.diagnostics
    assert [row.name for row in batched.rows] == ['visible.fixture']


@pytest.mark.parametrize('operation', ['entry_modes', 'resolve_paths'])
def test_native_index_batch_failure_preserves_scalar_fallback(tmp_path, native_index, monkeypatch, operation):
    from llm_solver.harness.task_files import TaskUtilityUnavailable
    files, index = native_index
    (tmp_path / 'visible.fixture').write_text('DEF target\n')
    def unavailable(*args):
        raise TaskUtilityUnavailable('batch utility unavailable')
    monkeypatch.setattr(files, operation, unavailable)
    result = index.scan()
    assert [row.name for row in result.rows] == ['target']
    assert result.diagnostics == ()


def test_native_index_read_rechecks_parent_after_batched_admission(tmp_path, native_index, monkeypatch):
    files, index = native_index
    parent = tmp_path / 'folder'
    parent.mkdir()
    (parent / 'source.fixture').write_text('DEF initial\n')
    outside = tmp_path.parent / (tmp_path.name + '-outside')
    outside.mkdir()
    (outside / 'source.fixture').write_text('DEF secret\n')
    original = files.read_bytes
    def swap_then_read(path):
        parent.rename(tmp_path / 'saved')
        parent.symlink_to(outside, target_is_directory=True)
        return original(path)
    monkeypatch.setattr(files, 'read_bytes', swap_then_read)
    result = index.scan()
    assert result.rows == () and index.extractor.extracted == []
    assert result.diagnostics[0].path == 'folder/source.fixture'
    assert result.diagnostics[0].error_kind == 'read_error'


def test_native_index_batch_keeps_subdirectory_scope(tmp_path, native_index, monkeypatch):
    files, index = native_index
    scope = tmp_path / 'scope'
    scope.mkdir()
    source = scope / 'swap.fixture'
    source.write_text('DEF initial\n')
    (tmp_path / 'sibling.fixture').write_text('DEF sibling\n')
    index.root = index.root / 'scope'
    original = files.resolve_paths
    def swap_then_resolve(paths):
        source.unlink()
        source.symlink_to('../sibling.fixture')
        return original(paths)
    monkeypatch.setattr(files, 'resolve_paths', swap_then_resolve)
    result = index.scan()
    assert result.rows == () and index.extractor.extracted == []


@pytest.mark.parametrize('readable', [False, True])
def test_native_digest_failure_preserves_ordinary_reads_and_diagnostics(
    tmp_path, native_index, monkeypatch, readable,
):
    files, index = native_index
    source = tmp_path / 'a.fixture'
    source.write_text('DEF before\n')
    assert index.scan().files_scanned == 1
    source.write_text('DEF after\n')

    def unavailable(*args):
        raise PermissionError(13, 'unreadable')

    monkeypatch.setattr(files, 'sha256_many', unavailable)
    if not readable:
        monkeypatch.setattr(files, 'read_bytes', unavailable)
    result = index.scan()
    if readable:
        assert [row.name for row in result.rows] == ['after']
        assert result.diagnostics == ()
    else:
        assert result.rows == ()
        assert result.diagnostics[0].path == 'a.fixture'
        assert result.diagnostics[0].error_kind == 'read_error'


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
    from llm_solver.harness import local_file_access
    original_read_bytes = local_file_access.read_bytes

    def failed_read(cwd, path: Path) -> bytes:
        if path.resolve() == source.resolve():
            raise PermissionError(13, "permission denied", str(path))
        return original_read_bytes(cwd, path)

    monkeypatch.setattr(local_file_access, "read_bytes", failed_read)
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
