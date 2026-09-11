"""Project declarations must support the runner selected automatically."""
import json
from pathlib import Path

import pytest

from scripts.llm_solver.language_quirks import detect_runner
from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness._loop._driver_setup import resolve_task_format


@pytest.mark.parametrize("name,contents,unsupported", [
    ("pyproject.toml", '[project]\nname="example"\n', "pytest"),
    ("setup.py", "from setuptools import setup\nsetup()\n", "pytest"),
    ("package.json", json.dumps({"scripts": {"test": "node --test"}}), "jest"),
    ("CMakeLists.txt", "cmake_minimum_required(VERSION 3.20)\nproject(example)\n", "ctest"),
])
def test_packaging_and_other_runner_declarations_do_not_invent_a_runner(tmp_path, name, contents, unsupported):
    (tmp_path / name).write_text(contents)
    assert detect_runner(tmp_path) != unsupported


def test_multiple_project_runners_are_not_resolved_by_descriptor_priority(tmp_path):
    (tmp_path / "Cargo.toml").write_text('[package]\nname="example"\nversion="0.1.0"\n')
    (tmp_path / "go.mod").write_text("module example.org/task\n")
    assert detect_runner(tmp_path) == "generic"


def test_explicit_pytest_configuration_remains_a_supported_candidate(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[tool.pytest.ini_options]\ntestpaths=["checks"]\n')
    assert detect_runner(tmp_path) == "pytest"


@pytest.mark.parametrize("files,blocked,expected", [
    ({}, (), "generic"),
    ({"pyproject.toml": '[project]\nname="example"\n'}, (), "generic"),
    ({"pytest.ini": "[pytest]\n"}, (), "pytest"),
    ({"go.mod": "module example.org/task\n"}, (), "go"),
    ({"Cargo.toml": '[package]\nname="example"\n'}, (), "cargo"),
    ({"package.json": '{"scripts":{"test":"node --test"}}'}, (), "npm"),
    ({"pytest.ini": "[pytest]\n", "go.mod": "module example.org/task\n"}, (), "generic"),
    ({"pytest.ini": "invalid configuration"}, (), "generic"),
    ({"pytest.ini": "[pytest]\n"}, ("pytest.ini",), "generic"),
])
def test_shipped_default_selects_analysis_from_permitted_task_declarations(
    tmp_path, files, blocked, expected,
):
    for name, contents in files.items():
        (tmp_path / name).write_text(contents)
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.toml")
    assert cfg.analysis_task_format == "auto"
    resolved = resolve_task_format(cfg, tmp_path, unreadable_paths=blocked)
    assert resolved.analysis_task_format == expected
    # Declaration-only analysis must not invent a verified executable choice.
    assert resolved.runtime_test_selection is None
