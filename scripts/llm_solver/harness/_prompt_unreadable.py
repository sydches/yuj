"""Match prompt visibility masks through their selected file view."""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Sequence

from .task_path import TaskPath, NativeUnreadableMatcher, native_requested_path

_GLOB_META = frozenset("*?[")


class _UnreadableMatcher:
    """Resolve sandbox unreadable patterns once, before any prompt read."""

    def __init__(self, base_dir: Path, patterns: Sequence[str]) -> None:
        self._base_dir, self._patterns = base_dir, patterns
        self._native_views = {}
        self._native = NativeUnreadableMatcher(base_dir, patterns) if isinstance(base_dir, TaskPath) else None
        self._host_source = (Path(str(base_dir)), patterns) if self._native is not None else None
        if self._native is not None:
            return
        blocked: set[Path] = set()
        for original in patterns:
            pattern = str(original)
            if pattern.startswith("optional:"):
                pattern = pattern[len("optional:"):]
            expanded = os.path.expandvars(os.path.expanduser(pattern))
            from .task_path import captured_host_path
            candidate = Path(str(captured_host_path(base_dir, expanded)))
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            candidate_text = str(candidate)
            if any(character in candidate_text for character in _GLOB_META):
                blocked.update(
                    Path(match).resolve(strict=False)
                    for match in glob.glob(
                        candidate_text,
                        recursive=True,
                        include_hidden=True,
                    )
                )
            else:
                blocked.add(candidate.resolve(strict=False))
        self._blocked = tuple(sorted(blocked, key=str))

    def blocks(self, path: Path) -> bool:
        if isinstance(path, TaskPath) and self._native is None:
            if path.files not in self._native_views:
                base = native_requested_path(TaskPath(path.files, path.files.root), str(self._base_dir))
                self._native_views[path.files] = NativeUnreadableMatcher(base, self._patterns)
            return self._native_views[path.files].blocks(path)
        if self._native is not None:
            if isinstance(path, TaskPath):
                return self._native.blocks(path)
            return _UnreadableMatcher(*self._host_source).blocks(path)
        resolved = path.resolve(strict=False)
        return any(
            resolved == blocked or blocked in resolved.parents
            for blocked in self._blocked
        )
