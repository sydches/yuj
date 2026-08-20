"""Where harness-owned telemetry lives.

The task workspace is the model's world. Telemetry — ``.trace.jsonl``, the
adaptive-control ledger, detector verdicts — is the harness's record *about*
the model, and must never appear inside that workspace.

Telemetry lives in a sibling of the workspace. Under container mode
only the workspace is bind-mounted, so the sibling is invisible to the model.
Under bwrap the whole host is readable, so the sibling is added to the sandbox
mask list (see ``sandbox._expand_unreadable_paths``).

Readers also accept the older in-workspace layout.
"""
from __future__ import annotations

from pathlib import Path

TRACE_NAME = ".trace.jsonl"
_DIR_PREFIX = ".yuj_"


def telemetry_dir(repo_dir: Path) -> Path:
    """The harness-owned directory paired with a task workspace."""
    repo_dir = Path(repo_dir)
    return repo_dir.parent / f"{_DIR_PREFIX}{repo_dir.name}"


def ensure_telemetry_dir(repo_dir: Path) -> Path:
    d = telemetry_dir(repo_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def trace_path(repo_dir: Path) -> Path:
    """Write target for this run's trace."""
    return telemetry_dir(repo_dir) / TRACE_NAME


def legacy_trace_path(repo_dir: Path) -> Path:
    """Pre-split location: inside the workspace."""
    return Path(repo_dir) / TRACE_NAME


def resolve_trace_path(repo_dir: Path) -> Path:
    """Read target: current layout, else the legacy in-workspace file.

    Returns the current-layout path when neither exists, so callers report a
    missing file against the location writes actually use.
    """
    current = trace_path(repo_dir)
    if current.exists():
        return current
    legacy = legacy_trace_path(repo_dir)
    if legacy.exists():
        return legacy
    return current


def telemetry_file(repo_dir: Path, name: str) -> Path:
    """Path for a named telemetry file (ledger, detector verdicts, ...).

    An absolute ``name`` is honoured as-is: operators pin ledgers to explicit
    locations, and those are already outside the workspace.
    """
    p = Path(name)
    if p.is_absolute():
        return p
    return telemetry_dir(repo_dir) / p
