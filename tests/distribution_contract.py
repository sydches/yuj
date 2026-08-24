"""Strict archive-member contract used by tests, CI, and release checks."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


REPOSITORY = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = REPOSITORY / "scripts" / "llm_solver" / "_resource_contract.py"
_SPEC = importlib.util.spec_from_file_location("_yuj_resource_contract_test", _CONTRACT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load resource contract: {_CONTRACT_PATH}")
_CONTRACT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONTRACT)
ROOT_RUNTIME_FILES = tuple(_CONTRACT.ROOT_RUNTIME_FILES)
PACKAGE_RUNTIME_FILES = tuple(_CONTRACT.PACKAGE_RUNTIME_FILES)

FORBIDDEN_PARTS = frozenset({
    ".agents",
    ".codex",
    ".git",
    ".github",
    ".internal",
    ".llm_assist",
    ".pytest_cache",
    "__pycache__",
    "benches",
    "paper",
    "sessions",
    "studies",
    "tests",
    "traces",
})
WHEEL_METADATA_FILES = frozenset({
    "METADATA",
    "RECORD",
    "WHEEL",
    "entry_points.txt",
    "top_level.txt",
})
SDIST_GENERATED_FILES = frozenset({
    "PKG-INFO",
    "setup.cfg",
    "yuj.egg-info/PKG-INFO",
    "yuj.egg-info/SOURCES.txt",
    "yuj.egg-info/dependency_links.txt",
    "yuj.egg-info/entry_points.txt",
    "yuj.egg-info/requires.txt",
    "yuj.egg-info/top_level.txt",
})


def _repository_files(root: Path, pattern: str) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.glob(pattern)
        if path.is_file()
    }


def expected_wheel_payload(root: Path = REPOSITORY) -> set[str]:
    python_files = _repository_files(root, "scripts/**/*.py")
    python_files = {
        path for path in python_files if not path.startswith("scripts/serve/")
    }
    package_data = {
        f"scripts/llm_solver/{path}" for path in PACKAGE_RUNTIME_FILES
    }
    root_data = {
        f"scripts/llm_solver/_resources/{path}" for path in ROOT_RUNTIME_FILES
    }
    return python_files | package_data | root_data


def expected_sdist_payload(root: Path = REPOSITORY) -> set[str]:
    files = {
        "LICENSE",
        "LICENSES/Apache-2.0.txt",
        "MANIFEST.in",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "build_support.py",
        "pyproject.toml",
        "yuj",
        "agents/README.md",
        "configs/regimes/README.md",
        "configs/treatment/README.md",
    }
    files.update(_repository_files(root, "scripts/**/*.py"))
    files.update(_repository_files(root, "scripts/**/*.toml"))
    files.update(_repository_files(root, "scripts/**/*.sh"))
    files.update(_repository_files(root, "docs/**/*.md"))
    files.update(_repository_files(root, "docs/**/*.yml"))
    files.update(_repository_files(root, "docs/**/*.yaml"))
    files.update(_repository_files(root, "configs/runtime/*.toml"))
    files.update(ROOT_RUNTIME_FILES)
    return files


def _forbidden_members(members: set[str]) -> list[str]:
    forbidden = []
    for member in sorted(members):
        parts = PurePosixPath(member).parts
        if any(part in FORBIDDEN_PARTS for part in parts):
            forbidden.append(member)
        elif any(part.endswith((".pyc", ".pyo")) for part in parts):
            forbidden.append(member)
        elif any(part in {"config.local.toml", ".env"} for part in parts):
            forbidden.append(member)
    return forbidden


def validate_wheel_members(
    members: set[str], root: Path = REPOSITORY
) -> dict[str, int]:
    expected = expected_wheel_payload(root)
    dist_info_dirs = {
        member.split("/", 1)[0]
        for member in members
        if ".dist-info/" in member
    }
    if len(dist_info_dirs) != 1:
        raise AssertionError(f"expected one wheel dist-info directory: {dist_info_dirs}")
    dist_info = next(iter(dist_info_dirs))
    payload = {member for member in members if not member.startswith(f"{dist_info}/")}
    metadata = {
        member.removeprefix(f"{dist_info}/")
        for member in members
        if member.startswith(f"{dist_info}/")
    }
    license_files = {item for item in metadata if item.startswith("licenses/")}
    if metadata - license_files != WHEEL_METADATA_FILES:
        raise AssertionError(
            "wheel metadata contract mismatch: "
            f"unexpected={sorted((metadata - license_files) - WHEEL_METADATA_FILES)} "
            f"missing={sorted(WHEEL_METADATA_FILES - metadata)}"
        )
    required_license_suffixes = {
        "LICENSE",
        "LICENSES/Apache-2.0.txt",
        "THIRD_PARTY_NOTICES.md",
    }
    actual_license_suffixes = {
        item.removeprefix("licenses/") for item in license_files
    }
    if actual_license_suffixes != required_license_suffixes:
        raise AssertionError(
            "wheel license contract mismatch: "
            f"actual={sorted(actual_license_suffixes)}"
        )
    if payload != expected:
        raise AssertionError(
            "wheel payload contract mismatch: "
            f"unexpected={sorted(payload - expected)} "
            f"missing={sorted(expected - payload)}"
        )
    forbidden = _forbidden_members(members)
    if forbidden:
        raise AssertionError(f"forbidden wheel members: {forbidden}")
    return {
        "members": len(members),
        "payload": len(payload),
        "root_resources": len(ROOT_RUNTIME_FILES),
        "package_resources": len(PACKAGE_RUNTIME_FILES),
    }


def validate_sdist_members(
    members: set[str], root: Path = REPOSITORY
) -> dict[str, int]:
    top_levels = {member.split("/", 1)[0] for member in members}
    if len(top_levels) != 1:
        raise AssertionError(f"expected one sdist top directory: {top_levels}")
    top = next(iter(top_levels))
    payload = {
        member.removeprefix(f"{top}/")
        for member in members
        if member != top
    }
    expected = expected_sdist_payload(root) | SDIST_GENERATED_FILES
    if payload != expected:
        raise AssertionError(
            "sdist payload contract mismatch: "
            f"unexpected={sorted(payload - expected)} "
            f"missing={sorted(expected - payload)}"
        )
    forbidden = _forbidden_members(payload)
    if forbidden:
        raise AssertionError(f"forbidden sdist members: {forbidden}")
    return {
        "members": len(members),
        "payload": len(payload),
        "root_resources": len(ROOT_RUNTIME_FILES),
        "package_resources": len(PACKAGE_RUNTIME_FILES),
    }


def archive_members(path: Path) -> set[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return {
                name.rstrip("/")
                for name in archive.namelist()
                if name.rstrip("/") and not name.endswith("/")
            }
    with tarfile.open(path, "r:gz") as archive:
        return {
            member.name.rstrip("/")
            for member in archive.getmembers()
            if member.isfile() and member.name.rstrip("/")
        }


def validate_archive(path: Path, root: Path = REPOSITORY) -> dict[str, int]:
    members = archive_members(path)
    if path.suffix == ".whl":
        return validate_wheel_members(members, root)
    if path.name.endswith(".tar.gz"):
        return validate_sdist_members(members, root)
    raise ValueError(f"unsupported distribution archive: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    for archive in args.archives:
        counts = validate_archive(archive)
        print(
            f"archive-ok: {archive.name} members={counts['members']} "
            f"payload={counts['payload']} "
            f"root_resources={counts['root_resources']} "
            f"package_resources={counts['package_resources']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
