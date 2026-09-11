"""Single-file navigation uses installed grammars without widening its scope."""
from dataclasses import replace
import xml.etree.ElementTree as ET

import pytest

from _config_helpers import make_config
from scripts.llm_solver.harness._tools.list_definitions import list_definitions
from scripts.llm_solver.harness.structural_index import (
    StructuralBackendUnavailable, TreeSitterTagExtractor,
)


@pytest.mark.parametrize("suffix,source", [
    ("js", "function sample() { return 1; }\nsample();\n"),
    ("ts", "function sample(): number { return 1; }\n"),
    ("tsx", "function sample() { return <div />; }\n"),
    ("go", "package fixture\nfunc sample() int { return 1 }\n"),
    ("rs", "fn sample() -> i32 { 1 }\n"),
    ("java", "class Fixture { int sample() { return 1; } }\n"),
])
def test_single_file_uses_installed_grammar_without_repository_gate(tmp_path, suffix, source, monkeypatch):
    target = tmp_path / f"one.{suffix}"
    target.write_text(source)
    (tmp_path / f"other.{suffix}").write_text(source.replace("sample", "unrequested"))
    original = TreeSitterTagExtractor.extract
    inspected = []
    def extract(self, data, **kwargs):
        inspected.append(kwargs["display_path"])
        return original(self, data, **kwargs)
    monkeypatch.setattr(TreeSitterTagExtractor, "extract", extract)
    cfg = make_config(tools_list_definitions_enabled=True, tools_ast_search_enabled=False)
    result = list_definitions(target.name, cwd=str(tmp_path), cfg=cfg)
    envelope = ET.fromstring(result)
    assert envelope.attrib["status"] == "ok"
    assert envelope.attrib["mode"] == "file"
    assert " def sample " in result
    assert " ref " not in result
    assert "unrequested" not in result
    assert inspected == [target.name]
    assert envelope.attrib["files_scanned"] == "1"


def test_file_rows_obey_existing_row_and_character_limits(tmp_path):
    (tmp_path / "one.js").write_text("\n".join(f"function name_{i}() {{}}" for i in range(20)))
    cfg = make_config(tools_list_definitions_enabled=True, tools_ast_search_max_rows=2)
    result = list_definitions("one.js", cwd=str(tmp_path), cfg=cfg)
    attrs = ET.fromstring(result).attrib
    assert attrs["total"] == "20"
    assert attrs["shown"] == "2"
    assert attrs["capped"] == "true"
    bounded = list_definitions("one.js", cwd=str(tmp_path), cfg=replace(cfg, max_output_chars=240))
    assert len(bounded) <= 240
    assert ET.fromstring(bounded).attrib["char_limited"] == "true"


@pytest.mark.parametrize("suffix", ["js", "py"])
def test_explicitly_unreadable_file_is_not_parsed(tmp_path, suffix, monkeypatch):
    path = tmp_path / f"one.{suffix}"
    path.write_text("secret")
    cfg = make_config(tools_list_definitions_enabled=True, unreadable_paths=(path.name,))
    def forbidden(*args, **kwargs):
        raise AssertionError("blocked content must not be parsed")
    monkeypatch.setattr(TreeSitterTagExtractor, "extract", forbidden)
    result = list_definitions(path.name, cwd=str(tmp_path), cfg=cfg)
    assert ET.fromstring(result).attrib["error_kind"] == "not_found"
    assert "secret" not in result


def test_missing_backend_has_typed_error_without_download(tmp_path, monkeypatch):
    (tmp_path / "one.js").write_text("function sample() {}")
    def unavailable(*args, **kwargs):
        raise StructuralBackendUnavailable("fixture missing grammar")
    monkeypatch.setattr(TreeSitterTagExtractor, "extract", unavailable)
    cfg = make_config(tools_list_definitions_enabled=True)
    result = list_definitions("one.js", cwd=str(tmp_path), cfg=cfg)
    assert ET.fromstring(result).attrib["error_kind"] == "backend_unavailable"


def test_disabled_tool_never_loads_parser(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled tool must not load a grammar")
    monkeypatch.setattr(TreeSitterTagExtractor, "detect_language", forbidden)
    result = list_definitions("one.js", cwd=str(tmp_path), cfg=make_config(tools_list_definitions_enabled=False))
    assert ET.fromstring(result).attrib["error_kind"] == "disabled"


def test_signature_is_escaped_and_missing_queries_are_explicit(tmp_path, monkeypatch):
    from scripts.llm_solver.harness.structural_index import StructuralLanguageUnsupported
    (tmp_path / "one.ts").write_text("function sample<T>(x: T): T { return x; }")
    cfg = make_config(tools_list_definitions_enabled=True)
    result = list_definitions("one.ts", cwd=str(tmp_path), cfg=cfg)
    assert "sample<T>" in ET.fromstring(result).text
    assert "sample&lt;T&gt;" in result
    def unsupported(*args, **kwargs):
        raise StructuralLanguageUnsupported("fixture no query")
    monkeypatch.setattr(TreeSitterTagExtractor, "extract", unsupported)
    result = list_definitions("one.ts", cwd=str(tmp_path), cfg=cfg)
    assert ET.fromstring(result).attrib["error_kind"] == "unsupported_language"
