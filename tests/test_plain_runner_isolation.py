"""Observed runner metadata must not activate treatment in the plain arm."""
from pathlib import Path
import shlex

import pytest

from scripts.llm_solver.config import load_config
from scripts.llm_solver.harness import tools
from scripts.llm_solver.harness._loop._driver_setup import resolve_task_format
from scripts.llm_solver.harness._loop.session_io import _load_bash_transforms


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "configs/regimes/baselines/plain_long_solve.toml"
CONTROL_RECIPES = [
    f"configs/paper/practitioner_grid/{quant}-43008-a00.toml"
    for quant in ("q2_k_xl", "iq4_xs", "q4_k_xl")
]


def _dispatch_with_loaded_transforms(cfg, task, monkeypatch):
    loaded = _load_bash_transforms(cfg, force_load_all=cfg.adaptive_policy_enabled)
    names = ("output_control", "universal_rewrites", "forbidden_rules",
             "redirect_rules", "redactions")
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        return "checks/test_unit.py::test_first PASSED\nchecks/test_unit.py::test_second PASSED\n", 0, False

    monkeypatch.setattr(tools, "_run_in_sandbox", execute)
    # The primitive shell tool is available to both arms. The process stub
    # exposes the exact command handed to execution without running pytest.
    result = tools.dispatch(
        "bash", {"cmd": "pytest -vv -s"}, cwd=str(task), cfg=cfg,
        **dict(zip(names, loaded)),
    )
    return loaded, commands, result


@pytest.mark.parametrize("recipe", [BASELINE, *CONTROL_RECIPES])
@pytest.mark.parametrize("declaration,contents,expected", [
    ("pytest.ini", "[pytest]\n", "pytest"),
    ("go.mod", "module example.org/task\n", "go"),
    (None, "", "generic"),
])
def test_control_preserves_discovery_without_activating_transforms(
    tmp_path, monkeypatch, recipe, declaration, contents, expected,
):
    if declaration:
        (tmp_path / declaration).write_text(contents)
    cfg = load_config(ROOT / recipe)
    assert cfg.analysis_task_format == "auto"
    cfg = resolve_task_format(cfg, tmp_path)
    assert cfg.analysis_task_format == expected
    assert not cfg.adaptive_policy_enabled
    assert not cfg.bash_transforms_task_format_enabled
    assert not cfg.tools_run_tests_enabled
    loaded, commands, result = _dispatch_with_loaded_transforms(cfg, tmp_path, monkeypatch)
    # Secret redaction has its own policy and is not runner-specific treatment.
    assert all(value is None for index, value in enumerate(loaded) if index != 4)
    assert commands == ["pytest -vv -s"]
    assert "checks/test_unit.py::test_first PASSED" in result
    assert "checks/test_unit.py::test_second PASSED" in result


def test_declared_treatment_still_applies_discovered_pytest_rewrites(tmp_path, monkeypatch):
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    cfg = resolve_task_format(load_config(ROOT / "configs/regimes/treatment.toml"), tmp_path)
    assert cfg.analysis_task_format == "pytest"
    assert cfg.bash_transforms_task_format_enabled
    loaded, commands, _ = _dispatch_with_loaded_transforms(cfg, tmp_path, monkeypatch)
    assert loaded[0] is not None
    assert len(commands) == 1
    arguments = shlex.split(commands[0])
    assert "--tb=short" in arguments
    assert "-vv" not in arguments
    assert "-s" not in arguments
