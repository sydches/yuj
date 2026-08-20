"""Path helpers — project root and external tool binaries.

Project root is located once, authoritative for the whole package. External
tool locations (e.g. ``llama-server``) are read from config, not hardcoded.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Walk up from this file until ``config.toml`` is found."""
    d = Path(__file__).resolve().parent
    for _ in range(10):
        if (d / "config.toml").is_file():
            return d
        parent = d.parent
        if parent == d:
            break
        d = parent
    raise FileNotFoundError(
        "config.toml not found. Set YUJ_CONFIG or run from project root."
    )


def expand_user_path(raw: str | Path) -> Path:
    """Expand ``~`` and environment variables, return an absolute Path."""
    import os
    return Path(os.path.expandvars(str(raw))).expanduser()


__all__ = ["project_root", "expand_user_path"]
