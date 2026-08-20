"""Checkpoint scanning — enumerate pending task repos inside a run directory.

A "run dir" contains ``repos/<task_id>/`` directories, each with a
``prompt.txt`` and (after first attempt) a ``checkpoint.json``. A repo is
"pending" when it has a prompt but no completed checkpoint. This primitive is
used by both the local harness and the Agent-SDK ``solve_bare`` driver.
"""
from __future__ import annotations

import json
from pathlib import Path


def collect_pending(run_dir: Path | str) -> list[Path]:
    """Return task repo directories that still need work, in sorted order.

    A repo is skipped when its ``checkpoint.json`` has ``status == "completed"``.
    A repo is skipped when it has no ``prompt.txt``. A malformed checkpoint is
    treated as pending (better to re-run than silently drop).

    Raises :class:`FileNotFoundError` if the run directory has no ``repos/``.
    """
    run_dir = Path(run_dir)
    repos_dir = run_dir / "repos"
    if not repos_dir.is_dir():
        raise FileNotFoundError(f"No repos/ directory in {run_dir}")

    pending: list[Path] = []
    for repo_dir in sorted(repos_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
        if not (repo_dir / "prompt.txt").exists():
            continue
        checkpoint = repo_dir / "checkpoint.json"
        if checkpoint.exists():
            try:
                data = json.loads(checkpoint.read_text())
            except json.JSONDecodeError:
                pending.append(repo_dir)
                continue
            if data.get("status") == "completed":
                continue
        pending.append(repo_dir)
    return pending


__all__ = ["collect_pending"]
