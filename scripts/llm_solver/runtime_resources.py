"""Validation and reporting for Yuj's public runtime-resource closure."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ._resource_contract import PACKAGE_RUNTIME_FILES, ROOT_RUNTIME_FILES
from ._shared.paths import package_data_path, project_root, resource_origin
from ._shared.toml_compat import tomllib


@dataclass(frozen=True)
class RuntimeResourceReport:
    """Inspectable result of validating the local runtime resource closure."""

    origin: str
    root: str
    root_resource_count: int
    package_resource_count: int

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def _validate_file(path: Path, logical_path: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required Yuj runtime resource is missing: {path}")
    if path.suffix == ".toml":
        with path.open("rb") as stream:
            tomllib.load(stream)
    elif path.suffix == ".py":
        compile(path.read_text(encoding="utf-8"), logical_path, "exec")
    else:
        path.read_text(encoding="utf-8")


def _installed_resource_files(root: Path) -> set[str]:
    """List declared-data candidates while tolerating Python bytecode caches."""
    installed_files: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or relative.suffix in {".pyc", ".pyo"}:
            continue
        installed_files.add(relative.as_posix())
    return installed_files


def validate_runtime_resources() -> RuntimeResourceReport:
    """Validate every file in the exact public runtime-resource contract."""
    root = project_root()
    for logical_path in ROOT_RUNTIME_FILES:
        _validate_file(root / logical_path, logical_path)

    origin = resource_origin()
    if origin == "installed-package":
        installed_files = _installed_resource_files(root)
        expected_files = set(ROOT_RUNTIME_FILES)
        if installed_files != expected_files:
            raise RuntimeError(
                "installed Yuj runtime-resource bundle differs from its manifest: "
                f"unexpected={sorted(installed_files - expected_files)}, "
                f"missing={sorted(expected_files - installed_files)}"
            )

    solver_package = __package__ or "scripts.llm_solver"
    for logical_path in PACKAGE_RUNTIME_FILES:
        _validate_file(
            package_data_path(solver_package, *logical_path.split("/")),
            logical_path,
        )

    return RuntimeResourceReport(
        origin=origin,
        root=str(root.resolve()),
        root_resource_count=len(ROOT_RUNTIME_FILES),
        package_resource_count=len(PACKAGE_RUNTIME_FILES),
    )


__all__ = [
    "PACKAGE_RUNTIME_FILES",
    "ROOT_RUNTIME_FILES",
    "RuntimeResourceReport",
    "validate_runtime_resources",
]
