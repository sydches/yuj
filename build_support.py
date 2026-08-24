"""Small setuptools hook that installs Yuj's root-owned runtime resources."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

from setuptools.command.build_py import build_py


_REPOSITORY = Path(__file__).resolve().parent
_CONTRACT_PATH = _REPOSITORY / "scripts" / "llm_solver" / "_resource_contract.py"
_SPEC = importlib.util.spec_from_file_location("_yuj_resource_contract", _CONTRACT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load runtime-resource contract: {_CONTRACT_PATH}")
_CONTRACT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONTRACT)
ROOT_RUNTIME_FILES = tuple(_CONTRACT.ROOT_RUNTIME_FILES)


class BuildPy(build_py):
    """Copy the exact root-resource manifest into the installed package."""

    def run(self) -> None:
        super().run()
        destination_root = Path(self.build_lib) / "scripts" / "llm_solver" / "_resources"
        if destination_root.exists():
            shutil.rmtree(destination_root)
        if tuple(sorted(set(ROOT_RUNTIME_FILES))) != ROOT_RUNTIME_FILES:
            raise RuntimeError("Yuj root runtime-resource manifest must be unique and sorted")
        for logical_path in ROOT_RUNTIME_FILES:
            source = _REPOSITORY / logical_path
            if not source.is_file():
                raise FileNotFoundError(
                    f"declared Yuj runtime resource is missing: {logical_path}"
                )
            destination = destination_root / logical_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        outputs = super().get_outputs(include_bytecode=include_bytecode)
        destination_root = Path(self.build_lib) / "scripts" / "llm_solver" / "_resources"
        outputs.extend(str(destination_root / path) for path in ROOT_RUNTIME_FILES)
        return outputs
