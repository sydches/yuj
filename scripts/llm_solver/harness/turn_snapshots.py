"""Invisible per-turn git snapshots of the task workspace.

After any executed turn that wrote a source file, record the full workspace
state as a git object — WITHOUT a branch, log entry, or index change the
model could observe. Any turn then becomes a rewind/branch point, forever,
and each snapshot usually needs little storage.

Mechanism: git plumbing through a private index file. ``git add -A`` into
that index (never the repo's own), ``write-tree``, then ``commit-tree`` with
HEAD as parent — producing a dangling commit that ``git log``/``status``
cannot see. The turn→sha map lives in the telemetry dir beside the trace,
outside the model's world. Invisibility is a leak-class requirement, not a
nicety because the model may read its own ``git log``.

Under ``YUJ_CONTAINER``, run Git inside the container and keep the private
index in ``/tmp``. Otherwise, run Git on the host and keep the private index
in the telemetry directory.

Failure policy: snapshots are telemetry, never load-bearing for the solve.
Any failure logs once per session and returns None; the run continues.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from .._shared.telemetry_paths import ensure_telemetry_dir, telemetry_dir

log = logging.getLogger(__name__)

MAP_NAME = "turn_snapshots.tsv"
_CONTAINER_INDEX = "/tmp/.yuj_snapshot_index"
# Container repositories may have a different owner. ``safe.directory`` lets
# Git use them. ``commit-tree`` also needs the temporary identity below.
_SNAPSHOT_SH = (
    "git config --global --add safe.directory {workdir} 2>/dev/null; "
    "export GIT_INDEX_FILE={index} "
    "GIT_AUTHOR_NAME=yuj GIT_AUTHOR_EMAIL=yuj@local "
    "GIT_COMMITTER_NAME=yuj GIT_COMMITTER_EMAIL=yuj@local; "
    "git add -A -- ':!.tool_output' ':!.solver' ':!prompt.txt' "
    "':!checkpoint.json' ':!metrics.json' >/dev/null 2>&1; "
    "tree=$(git write-tree 2>/dev/null) && "
    "echo 'yuj turn snapshot' | git commit-tree $tree -p HEAD 2>/dev/null"
)


def _container_id() -> str:
    return os.environ.get("YUJ_CONTAINER", "") or ""


def _run(repo_dir: Path, script: str) -> str:
    """Run a git shell snippet in the right place; return stdout."""
    cid = _container_id()
    if cid:
        argv = ["docker", "exec", "--workdir", "/testbed", cid, "bash", "-c", script]
        cwd = None
    else:
        argv = ["bash", "-c", script]
        cwd = str(repo_dir)
    out = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=60)
    return (out.stdout or "").strip()


def ensure_snapshot_setup(repo_dir: Path) -> None:
    """One-time per session: keep dangling snapshot objects alive.

    ``gc.auto 0`` stops background gc from pruning ref-less commits. Safe to
    call repeatedly; failure is non-fatal (snapshot() will then also fail
    and log its single warning).
    """
    try:
        _run(Path(repo_dir), "git config gc.auto 0")
    except Exception:
        pass


def snapshot(repo_dir: Path, turn: int, session=None) -> str | None:
    """Record an invisible workspace snapshot; return its sha (or None).

    Appends ``turn<TAB>sha`` to the telemetry map on success. The private
    index persists across calls (container /tmp or telemetry dir), so after
    the first snapshot each subsequent one stages only the delta.
    """
    repo_dir = Path(repo_dir)
    try:
        if _container_id():
            index = _CONTAINER_INDEX
            workdir = "/testbed"
        else:
            index = str(ensure_telemetry_dir(repo_dir) / ".snapshot_index")
            workdir = str(repo_dir)
        sha = _run(repo_dir, _SNAPSHOT_SH.format(index=index, workdir=workdir))
        if not sha or len(sha) < 7:
            raise RuntimeError(f"no sha (got {sha!r})")
        ensure_telemetry_dir(repo_dir)
        with open(telemetry_dir(repo_dir) / MAP_NAME, "a") as f:
            f.write(f"{int(turn)}\t{sha}\n")
        return sha
    except Exception as e:  # noqa: BLE001 — telemetry must never kill the run
        if session is not None and not getattr(session, "_snapshot_warned", False):
            setattr(session, "_snapshot_warned", True)
            log.warning("turn snapshot failed (disabled for session): %s", e)
        return None


def read_map(repo_dir: Path) -> list[tuple[int, str]]:
    """Return [(turn, sha), ...] recorded for this workspace's run."""
    p = telemetry_dir(Path(repo_dir)) / MAP_NAME
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[0].strip().isdigit():
            out.append((int(parts[0]), parts[1].strip()))
    return out


def sha_at_or_before(repo_dir: Path, turn: int) -> str | None:
    """The latest snapshot at or before ``turn`` — the rewind target."""
    best = None
    for t, sha in read_map(repo_dir):
        if t <= turn:
            best = sha
    return best


__all__ = [
    "ensure_snapshot_setup",
    "snapshot",
    "read_map",
    "sha_at_or_before",
    "MAP_NAME",
]
