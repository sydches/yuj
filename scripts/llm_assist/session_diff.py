"""Read-only unified diffs for retained assistant worktrees."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class SessionDiffError(RuntimeError):
    """A saved session diff cannot be produced safely."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SessionDiff:
    patch: bytes
    tracked_changes: bool
    untracked_files: int


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
    )
    return env


def _git(
    cwd: Path,
    args: list[str],
    *,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[bytes]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=_git_env(),
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SessionDiffError(
            "git_failed", f"git {' '.join(args)} failed: {exc}"
        ) from exc
    if proc.returncode not in allowed_returncodes:
        detail = (proc.stderr or proc.stdout).decode(
            "utf-8", errors="replace"
        ).strip()
        raise SessionDiffError(
            "git_failed",
            f"git {' '.join(args)} exited {proc.returncode}: {detail}",
        )
    return proc


def _safe_relative_path(raw: bytes) -> str:
    value = os.fsdecode(raw)
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise SessionDiffError(
            "unsafe_path", f"Git returned an unsafe untracked path: {value!r}"
        )
    return value


def build_session_worktree_diff(
    worktree_path: Path,
    base_commit: str,
) -> SessionDiff:
    """Compare one retained worktree with its saved starting commit."""
    worktree = Path(worktree_path).resolve()
    if not worktree.is_dir():
        raise SessionDiffError(
            "worktree_missing", f"retained worktree is missing: {worktree}"
        )

    top_level = _git(worktree, ["rev-parse", "--show-toplevel"]).stdout
    resolved_top_level = Path(os.fsdecode(top_level).strip()).resolve()
    if resolved_top_level != worktree:
        raise SessionDiffError(
            "worktree_mismatch",
            "saved worktree path is not the root of the inspected Git worktree",
        )

    baseline = _git(
        worktree,
        ["rev-parse", "--verify", f"{base_commit}^{{commit}}"],
        allowed_returncodes=(0, 128),
    )
    if baseline.returncode != 0:
        raise SessionDiffError(
            "baseline_missing",
            f"saved worktree baseline is missing: {base_commit}",
        )
    resolved_baseline = baseline.stdout.decode("ascii", errors="replace").strip()
    if resolved_baseline != base_commit:
        raise SessionDiffError(
            "baseline_mismatch",
            "saved worktree baseline does not resolve to its recorded commit",
        )

    tracked = _git(
        worktree,
        [
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            "--binary",
            base_commit,
            "--",
        ],
    ).stdout
    untracked_output = _git(
        worktree,
        ["ls-files", "--others", "--exclude-standard", "-z", "--"],
    ).stdout
    untracked_paths = [
        _safe_relative_path(raw)
        for raw in untracked_output.split(b"\0")
        if raw
    ]

    parts = [tracked] if tracked else []
    for relative_path in untracked_paths:
        addition = _git(
            worktree,
            [
                "diff",
                "--no-index",
                "--no-ext-diff",
                "--no-textconv",
                "--no-color",
                "--binary",
                "--src-prefix=a/",
                "--dst-prefix=b/",
                "--",
                "/dev/null",
                relative_path,
            ],
            allowed_returncodes=(0, 1),
        ).stdout
        if addition:
            parts.append(addition)

    return SessionDiff(
        patch=b"".join(parts),
        tracked_changes=bool(tracked),
        untracked_files=len(untracked_paths),
    )
