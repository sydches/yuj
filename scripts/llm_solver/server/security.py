"""Security validator — AST-walk .py files in profiles for banned patterns.

Allowed stdlib: re, json, typing, collections, dataclasses.
Banned: Import/ImportFrom (non-stdlib), open, subprocess, eval, exec, os.*, sys.*.
"""
import ast
import logging
from pathlib import Path

log = logging.getLogger(__name__)

ALLOWED_MODULES = frozenset({"re", "json", "typing", "collections", "dataclasses"})

BANNED_NAMES = frozenset({"open", "eval", "exec", "__import__"})

BANNED_MODULE_PREFIXES = ("os", "sys", "subprocess", "shutil", "socket", "http",
                          "urllib", "pathlib", "importlib", "ctypes")


class _SecurityVisitor(ast.NodeVisitor):
    """AST visitor that collects security violations."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name not in ALLOWED_MODULES and not alias.name.startswith("__"):
                self.violations.append(
                    f"{self.filepath}:{node.lineno}: banned import '{alias.name}'"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        root = mod.split(".")[0]
        if root not in ALLOWED_MODULES and root != "":
            self.violations.append(
                f"{self.filepath}:{node.lineno}: banned import from '{mod}'"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check direct calls: open(), eval(), exec()
        if isinstance(node.func, ast.Name) and node.func.id in BANNED_NAMES:
            self.violations.append(
                f"{self.filepath}:{node.lineno}: banned call '{node.func.id}()'"
            )
        # Check attribute calls: os.system(), subprocess.run()
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                prefix = node.func.value.id
                if prefix in BANNED_MODULE_PREFIXES:
                    self.violations.append(
                        f"{self.filepath}:{node.lineno}: banned call '{prefix}.{node.func.attr}()'"
                    )
        self.generic_visit(node)


def validate_file(path: Path) -> list[str]:
    """Validate a single .py file. Returns list of violations (empty = clean)."""
    source = path.read_text()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return [f"{path}:{e.lineno}: syntax error: {e.msg}"]

    visitor = _SecurityVisitor(str(path))
    visitor.visit(tree)
    return visitor.violations


def validate_profile(profile_dir: Path) -> list[str]:
    """Validate all .py files in a profile directory. Returns list of violations."""
    violations: list[str] = []
    for py_file in sorted(profile_dir.rglob("*.py")):
        # Skip __pycache__ and fixtures
        if "__pycache__" in py_file.parts:
            continue
        violations.extend(validate_file(py_file))
    return violations
