"""Session-local read-before-edit ledger with trace reconstruction.

The ledger is mechanical harness state.  A successful typed read, successful
mutation, or safely classified single-file shell read records an exact file
fingerprint.  Before edit, the current bytes are compared with that record.
Observation events make the ledger reconstructible from the append-only trace;
the model never owns or updates it.
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Literal, Mapping

from .task_path import resolve_task_path, TaskPath


StaleGuardMode = Literal["off", "warn", "block"]
EventSink = Callable[[dict[str, object]], None]


class StaleGuardError(RuntimeError):
    """The ledger could not obtain a stable, contained file observation."""


@dataclass(frozen=True)
class FileFingerprint:
    mtime_ns: int
    size: int
    sha256: str

    def as_trace(self) -> dict[str, object]:
        return {
            "mtime_ns": self.mtime_ns,
            "size": self.size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_trace(cls, value: Mapping[str, object]) -> "FileFingerprint":
        return cls(
            mtime_ns=int(value["mtime_ns"]),
            size=int(value["size"]),
            sha256=str(value["sha256"]),
        )


@dataclass(frozen=True)
class GuardDecision:
    path: str
    reason: str
    mode: StaleGuardMode
    allowed: bool
    message: str = ""

    @property
    def blocked(self) -> bool:
        return not self.allowed


from ._single_file_read import ShellRead, classify_single_file_read

class StaleFileGuard:
    """Read ledger and policy decision point for one solver session."""

    def __init__(
        self,
        *,
        cwd: str | Path,
        mode: StaleGuardMode = "warn",
        event_sink: EventSink | None = None,
    ) -> None:
        if mode not in {"off", "warn", "block"}:
            raise ValueError("stale guard mode must be off, warn, or block")
        self.cwd = Path(cwd).resolve()
        self.mode = mode
        self.event_sink = event_sink
        self._ledger: dict[str, FileFingerprint] = {}
        self._views: dict[str, dict] = {}
        self._lock = threading.RLock()

    def _emit(self, event: str, **fields: object) -> None:
        if self.event_sink is not None:
            self.event_sink({"event": event, **fields})

    def _target(self, path: str):
        try:
            root = resolve_task_path(self.cwd, '.')
            target = resolve_task_path(self.cwd, path)
            relative = target.relative_to(root).as_posix()
        except ValueError as exc:
            raise StaleGuardError(str(exc)) from exc
        if relative in {"", "."}:
            raise StaleGuardError("stale guard path must name a file")
        return target, relative

    @staticmethod
    def _view(target):
        return dict(target.files.binding) if isinstance(target, TaskPath) else None

    @staticmethod
    def _fingerprint(target: Path) -> FileFingerprint:
        for _attempt in range(3):
            before = target.stat()
            data = target.read_bytes()
            after = target.stat()
            before_key = (before.st_ino, before.st_mtime_ns, before.st_size)
            after_key = (after.st_ino, after.st_mtime_ns, after.st_size)
            if before_key == after_key and len(data) == after.st_size:
                return FileFingerprint(
                    mtime_ns=after.st_mtime_ns,
                    size=after.st_size,
                    sha256=hashlib.sha256(data).hexdigest(),
                )
        raise StaleGuardError(f"file changed while being fingerprinted: {target.name}")

    def observe(self, path: str, *, source: str) -> FileFingerprint:
        """Record a successful read or mutation and emit reconstruction data."""
        target, relative = self._target(path)
        try:
            fingerprint = self._fingerprint(target)
        except FileNotFoundError as exc:
            raise StaleGuardError(f"file not found: {relative}") from exc
        with self._lock:
            self._ledger[relative] = fingerprint
            view = self._view(target)
            if view is None:
                self._views.pop(relative, None)
            else:
                self._views[relative] = view
        self._emit(
            "stale_guard_observe",
            path=relative,
            source=source,
            fingerprint=fingerprint.as_trace(),
            **({'task_view': view} if view is not None else {}),
        )
        return fingerprint

    def observe_read(self, path: str) -> FileFingerprint:
        return self.observe(path, source="read")

    def observe_mutation(self, path: str, *, source: str) -> FileFingerprint:
        if source not in {
            "write", "edit", "notebook_edit", "structural_edit", "apply_patch", "udiff",
        }:
            raise ValueError(
                "mutation source must be write, edit, notebook_edit, structural_edit, "
                "apply_patch, or udiff"
            )
        return self.observe(path, source=source)

    def observe_shell_read(self, command: str) -> ShellRead | None:
        """Credit one safely classified, successful bash read."""
        classified = classify_single_file_read(command)
        if classified is None:
            return None
        target, _relative = self._target(classified.path)
        if not target.is_file():
            return None
        self.observe(classified.path, source=f"bash:{classified.verb}")
        return classified

    def forget(self, path: str, *, source: str = "apply_patch") -> None:
        """Remove a deleted path from the ledger and record that transition."""
        _target, relative = self._target(path)
        with self._lock:
            self._ledger.pop(relative, None)
            self._views.pop(relative, None)
        self._emit(
            "stale_guard_observe", path=relative, source=source, fingerprint=None
        )

    def check_edit(self, path: str) -> GuardDecision:
        """Decide whether an edit may run under the configured policy."""
        if self.mode == "off":
            return GuardDecision(path=path, reason="off", mode=self.mode, allowed=True)
        target, relative = self._target(path)
        with self._lock:
            expected = self._ledger.get(relative)
            current: FileFingerprint | None = None
            if expected is None:
                reason = "unread"
            elif self._views.get(relative) != self._view(target):
                reason = 'view_changed'
            else:
                try:
                    current = self._fingerprint(target)
                except FileNotFoundError:
                    reason = "missing"
                else:
                    reason = "modified" if current.sha256 != expected.sha256 else ""
                    if not reason and current != expected:
                        # Metadata-only change does not make content stale.
                        self._ledger[relative] = current
                        self._emit(
                            "stale_guard_observe",
                            path=relative,
                            source="metadata_refresh",
                            fingerprint=current.as_trace(),
                            **({'task_view': self._views[relative]}
                               if relative in self._views else {}),
                        )
            if not reason:
                return GuardDecision(
                    path=relative, reason="fresh", mode=self.mode, allowed=True
                )

        prefix = "ERROR" if self.mode == "block" else "WARNING"
        message = f"{prefix}: stale_file: read {relative} first"
        blocked = self.mode == "block"
        self._emit(
            "stale_guard",
            path=relative,
            reason=reason,
            mode=self.mode,
            blocked=blocked,
            expected=(expected.as_trace() if expected is not None else None),
            current=(current.as_trace() if current is not None else None),
        )
        return GuardDecision(
            path=relative,
            reason=reason,
            mode=self.mode,
            allowed=not blocked,
            message=message,
        )

    @classmethod
    def from_trace(
        cls,
        *,
        cwd: str | Path,
        mode: StaleGuardMode,
        events: Iterable[Mapping[str, object]],
        event_sink: EventSink | None = None,
    ) -> "StaleFileGuard":
        """Rebuild the last observation for each path from trace events."""
        guard = cls(cwd=cwd, mode=mode, event_sink=event_sink)
        for event in events:
            if event.get("event") != "stale_guard_observe":
                continue
            path = str(event.get("path", ""))
            pure = PurePosixPath(path)
            if not path or pure.is_absolute() or ".." in pure.parts:
                raise StaleGuardError(f"invalid path in stale guard trace: {path!r}")
            fingerprint = event.get("fingerprint")
            if fingerprint is None:
                guard._ledger.pop(path, None)
                guard._views.pop(path, None)
            elif isinstance(fingerprint, Mapping):
                guard._ledger[path] = FileFingerprint.from_trace(fingerprint)
                view = event.get('task_view')
                if view is None:
                    guard._views.pop(path, None)
                elif isinstance(view, Mapping):
                    guard._views[path] = dict(view)
                else:
                    raise StaleGuardError(f'invalid task view in stale guard trace for {path}')
            else:
                raise StaleGuardError(
                    f"invalid fingerprint in stale guard trace for {path}"
                )
        return guard

    def ledger_snapshot(self) -> dict[str, FileFingerprint]:
        with self._lock:
            return dict(self._ledger)


__all__ = [
    "FileFingerprint", "GuardDecision", "ShellRead", "StaleFileGuard",
    "StaleGuardError", "StaleGuardMode", "classify_single_file_read",
]
