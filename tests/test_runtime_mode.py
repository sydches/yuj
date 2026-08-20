import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.llm_solver.config import require_runtime_mode

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_require_runtime_mode_rejects_wrong_entrypoint():
    cfg = SimpleNamespace(runtime_mode="measurement")
    with pytest.raises(ValueError):
        require_runtime_mode(cfg, expected="assistant", caller="scripts.llm_assist")


def test_require_runtime_mode_accepts_matching_entrypoint():
    cfg = SimpleNamespace(runtime_mode="measurement")
    require_runtime_mode(cfg, expected="measurement", caller="scripts.llm_solver")


def test_harness_core_does_not_import_cli_wrapper():
    forbidden_roots = ("scripts.llm_assist", "scripts.yuj")
    offenders: list[str] = []

    for path in sorted((PROJECT_ROOT / "scripts" / "llm_solver").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported = [node.module]
                    if node.level:
                        imported.append(f"{'.' * node.level}{node.module}")
                elif node.level:
                    imported = [f"{'.' * node.level}{alias.name}" for alias in node.names]

            for name in imported:
                absolute_forbidden = name == "scripts.yuj" or name.startswith(forbidden_roots)
                relative_forbidden = name.startswith("..llm_assist") or name.startswith("..yuj")
                if absolute_forbidden or relative_forbidden:
                    rel = path.relative_to(PROJECT_ROOT)
                    offenders.append(f"{rel}: imports {name}")

    assert offenders == []
