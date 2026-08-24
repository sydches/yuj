"""Path helpers for source checkouts and installed package resources.

The active runtime root has one deterministic precedence contract:

1. ``YUJ_CONFIG`` names the explicit main configuration file and its parent
   owns relative runtime resources.
2. A source/editable checkout owns the checked-in ``config.toml`` and public
   resource tree.
3. A wheel installation uses the build-generated ``_resources`` tree shipped
   inside :mod:`scripts.llm_solver`.

Most wheel installs expose package resources as ordinary files.  For an
importer that does not, :func:`package_data_path` materializes only the
requested file or directory in a process-lifetime temporary directory.  It
never copies defaults into a user or project directory.
"""
from __future__ import annotations

import atexit
from contextlib import ExitStack
from functools import lru_cache
from importlib import resources
import os
from pathlib import Path
import tempfile


_RESOURCE_STACK = ExitStack()
atexit.register(_RESOURCE_STACK.close)


def _solver_package() -> str:
    """Return ``scripts.llm_solver`` or the legacy ``llm_solver`` spelling."""
    package = __package__ or "scripts.llm_solver._shared"
    return package.rsplit("._shared", 1)[0]


def _find_source_root(start: Path | None = None) -> Path | None:
    """Find a checked-in runtime root above *start*, if one exists."""
    directory = Path(start or __file__).resolve()
    if directory.is_file():
        directory = directory.parent
    for _ in range(10):
        if (
            (directory / "config.toml").is_file()
            and (directory / "profiles").is_dir()
            and (directory / "scripts" / "llm_solver").is_dir()
        ):
            return directory
        parent = directory.parent
        if parent == directory:
            break
        directory = parent
    return None


def _copy_traversable(source, destination: Path) -> None:
    """Copy one importlib Traversable without assuming filesystem backing."""
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            _copy_traversable(child, destination / child.name)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


@lru_cache(maxsize=None)
def _package_data_path(package: str, parts: tuple[str, ...]) -> Path:
    traversable = resources.files(package)
    for part in parts:
        traversable = traversable.joinpath(part)

    try:
        candidate = Path(os.fspath(traversable)).resolve()
    except TypeError:
        candidate = None
    if candidate is not None and candidate.exists():
        return candidate

    try:
        materialized = _RESOURCE_STACK.enter_context(resources.as_file(traversable))
    except (IsADirectoryError, NotADirectoryError, TypeError):
        materialized = None
    if materialized is not None and materialized.exists():
        return materialized.resolve()

    temp_root = Path(
        _RESOURCE_STACK.enter_context(
            tempfile.TemporaryDirectory(prefix="yuj-package-resources-")
        )
    )
    destination = temp_root / (parts[-1] if parts else package.rsplit(".", 1)[-1])
    _copy_traversable(traversable, destination)
    return destination.resolve()


def package_data_path(package: str, *parts: str) -> Path:
    """Return a stable filesystem path for one installed package resource."""
    return _package_data_path(str(package), tuple(str(part) for part in parts))


def default_config_path() -> Path:
    """Return the exact active main-config path under the precedence contract."""
    configured = os.environ.get("YUJ_CONFIG")
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError("YUJ_CONFIG does not name an existing file")
    return project_root() / "config.toml"


def local_config_path() -> Path:
    """Return the mutable machine-local config path without creating it."""
    configured_local = os.environ.get("YUJ_CONFIG_LOCAL")
    if configured_local:
        return Path(configured_local).expanduser().resolve()

    configured_main = os.environ.get("YUJ_CONFIG")
    if configured_main:
        return default_config_path().with_name("config.local.toml")

    source_root = _find_source_root()
    if source_root is not None:
        return source_root / "config.local.toml"

    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = (
        Path(config_home).expanduser()
        if config_home
        else Path.home() / ".config"
    )
    return (base / "yuj" / "config.local.toml").resolve()


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the root that owns config-relative Yuj runtime resources."""
    configured = os.environ.get("YUJ_CONFIG")
    if configured:
        return default_config_path().parent

    source_root = _find_source_root()
    if source_root is not None:
        return source_root

    bundle = package_data_path(_solver_package(), "_resources")
    if (bundle / "config.toml").is_file():
        return bundle
    raise FileNotFoundError(
        "Yuj runtime resources are unavailable: no source checkout or "
        "installed _resources/config.toml was found"
    )


def resource_origin() -> str:
    """Describe which branch of the runtime-root precedence contract won."""
    if os.environ.get("YUJ_CONFIG"):
        return "explicit-config"
    if _find_source_root() is not None:
        return "source-checkout"
    return "installed-package"


def expand_user_path(raw: str | Path) -> Path:
    """Expand ``~`` and environment variables, return an absolute Path."""
    return Path(os.path.expandvars(str(raw))).expanduser()


__all__ = [
    "default_config_path",
    "expand_user_path",
    "local_config_path",
    "package_data_path",
    "project_root",
    "resource_origin",
]
