"""Loaded content identity must not imply permission to inject guidance."""
import hashlib
from pathlib import Path

import pytest

from scripts.llm_solver.harness.injections import load_injections_with_metadata


def _load(root: Path, text: str):
    directory = root / "rules"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "hint.md").write_bytes(text.encode("utf-8"))
    return load_injections_with_metadata(
        directory, imports_enabled=True, allowed_dirs=(root,),
    ).prompt_import_tree[0]


BASE = '+++\nname = "hint"\nkeywords = ["alpha"]\npaths = ["src/*.py"]\n+++\nRead the API.'


def test_exact_bytes_and_effective_rule_have_distinct_identity(tmp_path):
    first = _load(tmp_path, BASE)
    assert first["source_sha256"] == hashlib.sha256(BASE.encode()).hexdigest()
    assert first["rule_name"] == "hint"
    assert first["resolved_rule_schema"] == "injection-rule-v1"
    assert first["admission_status"] == "unverified"
    formatted = _load(tmp_path, BASE.replace('name =', '# comment\nname ='))
    assert formatted["source_sha256"] != first["source_sha256"]
    assert formatted["resolved_rule_sha256"] == first["resolved_rule_sha256"]
    relocated = _load(tmp_path / "elsewhere", BASE)
    assert relocated["resolved_rule_sha256"] == first["resolved_rule_sha256"]
    assert relocated["admission_status"] == "unverified"


@pytest.mark.parametrize("old,new", [
    ('"hint"', '"help"'),
    ('"alpha"', '"omega"'),
    ('"src/*.py"', '"lib/*.py"'),
    ('Read the API.', 'Read the FAQ.'),
    ('paths =', 'repeat = true\npaths ='),
    ('paths =', 'fire_once = false\npaths ='),
    ('paths =', 'trigger = "path"\npaths ='),
])
def test_rule_fields_cannot_change_without_changing_identity(tmp_path, old, new):
    first = _load(tmp_path, BASE)
    second = _load(tmp_path, BASE.replace(old, new))
    assert first["source_sha256"] != second["source_sha256"]
    assert first["resolved_rule_sha256"] != second["resolved_rule_sha256"]
    assert second["admission_status"] == "unverified"


def test_import_change_is_bound_even_when_root_and_byte_count_match(tmp_path):
    shared = tmp_path / "shared.md"
    shared.write_text("Read the API.")
    root = BASE.replace("Read the API.", "@../shared.md")
    first = _load(tmp_path, root)
    shared.write_text("Read the FAQ.")
    second = _load(tmp_path, root)
    assert first["source_sha256"] == second["source_sha256"]
    assert first["imported_bytes"] == second["imported_bytes"]
    assert first["resolved_rule_sha256"] != second["resolved_rule_sha256"]
    assert second["admission_status"] == "unverified"


@pytest.mark.parametrize("body", ["Use button__label.", "For CASE-0042 return 7."])
def test_identifier_spelling_is_not_admission(tmp_path, body):
    loaded = _load(tmp_path, BASE.replace("Read the API.", body))
    assert loaded["admission_status"] == "unverified"
    assert loaded["source_sha256"]
    assert loaded["resolved_rule_sha256"]
