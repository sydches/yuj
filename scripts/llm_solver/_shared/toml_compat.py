"""TOML loader — single import point for the project.

Python 3.11+ ships ``tomllib`` in the stdlib. On older interpreters we fall back
to the third-party ``tomli`` package which exposes the same API. Every other
module must import ``tomllib`` from here (or call :func:`load_toml`) instead of
re-declaring the version guard. When the minimum interpreter moves to 3.11 this
file is the only edit.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib  # noqa: F401  (re-exported)
else:
    import tomli as tomllib  # type: ignore[no-redef]  # noqa: F401


def load_toml(path: Path | str) -> dict[str, Any]:
    """Read and parse a TOML file. Raises FileNotFoundError if absent."""
    p = Path(path)
    with open(p, "rb") as f:
        return tomllib.load(f)


__all__ = ["tomllib", "load_toml"]
