"""Independent, byte-preserving Git checkpoints for a task workspace."""
from __future__ import annotations

import fnmatch
import json
import os
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .._shared.telemetry_paths import telemetry_dir
from .tool_specs import ACTION_WRITE_LIKE_TOOL_NAMES

_HEAD_REF = "refs/heads/checkpoints"
_TURN_REF_PREFIX = "refs/yuj/checkpoints/turn-"
_METRICS_LOG = "checkpoint_metrics.jsonl"
_FORMAT_VERSION = "1"


class WorkspaceCheckpointError(RuntimeError):
    pass


class CheckpointNotFoundError(WorkspaceCheckpointError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_timestamp(clock: Callable[[], datetime]) -> str:
    value = clock()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def default_shadow_dir(workspace: Path) -> Path:
    return telemetry_dir(Path(workspace).resolve()) / ".shadow_git"


def tool_call_needs_checkpoint(tool_name: str, *, executed: bool = True) -> bool:
    """True for executed calls that may mutate; every bash call qualifies."""
    return bool(
        executed
        and (tool_name == "bash" or tool_name in ACTION_WRITE_LIKE_TOOL_NAMES)
    )


@dataclass(frozen=True)
class WorkspaceCheckpoint:
    turn: int
    commit: str
    duration_ms: float
    file_count: int
    byte_count: int
    captured_at: str

    def trace_fields(self) -> dict[str, object]:
        return {
            "turn": self.turn,
            "commit": self.commit,
            "duration_ms": self.duration_ms,
            "file_count": self.file_count,
            "byte_count": self.byte_count,
        }


@dataclass(frozen=True)
class RestoredCheckpoint:
    turn: int
    commit: str
    files_restored: int
    files_removed: int
    bytes_restored: int


@dataclass(frozen=True)
class _TreeEntry:
    mode: int
    object_id: str
    path: str


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\x00" in value:
        raise WorkspaceCheckpointError(f"unsafe workspace path: {value!r}")
    return path.as_posix()


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("zero-byte write")
        view = view[written:]


def _normalize_exclude(pattern: str) -> str:
    value = str(pattern or "").strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    if not value or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts:
        raise ValueError(f"checkpoint exclude must be a safe relative pattern: {pattern!r}")
    return value


class WorkspaceCheckpointStore:
    def __init__(
        self,
        workspace: Path,
        *,
        shadow_dir: Path | None = None,
        excludes: Iterable[str] = (),
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.shadow_dir = Path(shadow_dir or default_shadow_dir(self.workspace)).resolve()
        self.excludes = tuple(_normalize_exclude(value) for value in excludes)
        self._clock = clock
        self._lock = threading.RLock()
        if not self.workspace.is_dir():
            raise ValueError(f"workspace is not a directory: {self.workspace}")
        try:
            self.shadow_dir.relative_to(self.workspace)
        except ValueError:
            pass
        else:
            raise ValueError("shadow Git directory must live outside the task workspace")

    @property
    def sandbox_unreadable_paths(self) -> tuple[str, ...]:
        return (f"optional:{self.shadow_dir}",)

    def _env(self, *, index: Path | None = None) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "GIT_DIR": str(self.shadow_dir),
                "GIT_WORK_TREE": str(self.workspace),
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_ATTR_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_AUTHOR_NAME": "yuj-harness",
                "GIT_AUTHOR_EMAIL": "yuj@localhost",
                "GIT_COMMITTER_NAME": "yuj-harness",
                "GIT_COMMITTER_EMAIL": "yuj@localhost",
                "LC_ALL": "C",
            }
        )
        env["GIT_INDEX_FILE"] = str(index or self.shadow_dir / "index")
        return env

    def _git(
        self,
        args: list[str],
        *,
        input_bytes: bytes | None = None,
        index: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=self.workspace,
                env=self._env(index=index),
                input=input_bytes,
                capture_output=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise WorkspaceCheckpointError(f"git {' '.join(args)} failed: {exc}") from exc
        if check and proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", errors="replace").strip()
            raise WorkspaceCheckpointError(
                f"git {' '.join(args)} exited {proc.returncode}: {detail}"
            )
        return proc

    def _ensure_initialized(self) -> None:
        marker = self.shadow_dir / "HEAD"
        if not marker.exists():
            if self.shadow_dir.exists() and any(self.shadow_dir.iterdir()):
                raise WorkspaceCheckpointError(
                    f"shadow directory exists but is not a Git repository: {self.shadow_dir}"
                )
            self.shadow_dir.parent.mkdir(parents=True, exist_ok=True)
            init_env = self._env()
            init_env.pop("GIT_DIR", None)
            init_env.pop("GIT_WORK_TREE", None)
            init_env.pop("GIT_INDEX_FILE", None)
            proc = subprocess.run(
                ["git", "init", "--bare", "-q", str(self.shadow_dir)],
                cwd=self.workspace,
                env=init_env,
                capture_output=True,
                timeout=30,
                check=False,
            )
            if proc.returncode != 0:
                detail = proc.stderr.decode("utf-8", errors="replace").strip()
                raise WorkspaceCheckpointError(f"cannot initialize shadow Git: {detail}")
            self._git(["symbolic-ref", "HEAD", _HEAD_REF])
            self._git(["config", "yuj.checkpointVersion", _FORMAT_VERSION])
            self._git(["config", "yuj.workspace", str(self.workspace)])
            self._git(["read-tree", "--empty"])
            return
        version = self._git(["config", "--get", "yuj.checkpointVersion"]).stdout.strip()
        owner = self._git(["config", "--get", "yuj.workspace"]).stdout.decode().strip()
        if version.decode() != _FORMAT_VERSION or Path(owner).resolve() != self.workspace:
            raise WorkspaceCheckpointError(
                "shadow Git metadata does not belong to this workspace"
            )

    def _is_excluded(self, rel_path: str) -> bool:
        if rel_path == ".git" or rel_path.startswith(".git/"):
            return True
        for pattern in self.excludes:
            prefix = pattern[:-3].rstrip("/") if pattern.endswith("/**") else ""
            if prefix and (rel_path == prefix or rel_path.startswith(prefix + "/")):
                return True
            if pattern.endswith("/") and (
                rel_path == pattern.rstrip("/") or rel_path.startswith(pattern)
            ):
                return True
            if fnmatch.fnmatchcase(rel_path, pattern) or PurePosixPath(rel_path).match(pattern):
                return True
        return False

    def _candidate_paths(self) -> list[str]:
        output = self._git(
            ["ls-files", "-z", "--cached", "--others", "--exclude-standard", "--"]
        ).stdout
        paths: list[str] = []
        for raw in output.split(b"\x00"):
            if not raw:
                continue
            rel_path = _safe_relative_path(os.fsdecode(raw))
            target = self.workspace / rel_path
            if self._is_excluded(rel_path):
                continue
            if target.is_symlink() or target.is_file():
                paths.append(rel_path)
        return sorted(set(paths))

    def _hash_file(self, rel_path: str) -> tuple[int, str, int]:
        path = self.workspace / rel_path
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            data = os.fsencode(os.readlink(path))
            mode = 0o120000
        elif stat.S_ISREG(info.st_mode):
            data = path.read_bytes()
            mode = 0o100755 if info.st_mode & 0o111 else 0o100644
        else:
            raise WorkspaceCheckpointError(f"unsupported file type at {rel_path!r}")
        object_id = self._git(
            ["hash-object", "-w", "--stdin"], input_bytes=data
        ).stdout.decode().strip()
        if len(object_id) < 40:
            raise WorkspaceCheckpointError(f"invalid blob id for {rel_path!r}: {object_id!r}")
        return mode, object_id, len(data)

    def _build_tree(self, paths: list[str], index: Path) -> tuple[str, int]:
        self._git(["read-tree", "--empty"], index=index)
        records = bytearray()
        byte_count = 0
        for rel_path in paths:
            mode, object_id, size = self._hash_file(rel_path)
            records.extend(f"{mode:o} blob {object_id}\t".encode())
            records.extend(os.fsencode(rel_path))
            records.append(0)
            byte_count += size
        if records:
            self._git(
                ["update-index", "-z", "--index-info"],
                input_bytes=bytes(records),
                index=index,
            )
        tree = self._git(["write-tree"], index=index).stdout.decode().strip()
        if len(tree) < 40:
            raise WorkspaceCheckpointError(f"invalid tree id: {tree!r}")
        return tree, byte_count

    def _current_commit(self) -> str | None:
        proc = self._git(["rev-parse", "--verify", _HEAD_REF], check=False)
        return proc.stdout.decode().strip() if proc.returncode == 0 else None

    @staticmethod
    def _turn_ref(turn: int) -> str:
        return f"{_TURN_REF_PREFIX}{turn:012d}"

    def _append_metric(self, checkpoint: WorkspaceCheckpoint) -> None:
        path = self.shadow_dir / _METRICS_LOG
        payload = json.dumps(
            {
                "turn": checkpoint.turn,
                "commit": checkpoint.commit,
                "duration_ms": checkpoint.duration_ms,
                "file_count": checkpoint.file_count,
                "byte_count": checkpoint.byte_count,
                "captured_at": checkpoint.captured_at,
            },
            separators=(",", ":"),
        ).encode() + b"\n"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            _write_all(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)

    def capture(self, turn: int) -> WorkspaceCheckpoint:
        if int(turn) < 0:
            raise ValueError("checkpoint turn must be non-negative")
        turn = int(turn)
        started = time.perf_counter()
        with self._lock:
            self._ensure_initialized()
            paths = self._candidate_paths()
            temporary_index = self.shadow_dir / f".index-{uuid.uuid4().hex}"
            try:
                tree, byte_count = self._build_tree(paths, temporary_index)
                args = ["commit-tree", tree]
                parent = self._current_commit()
                if parent:
                    args.extend(["-p", parent])
                message = f"yuj workspace checkpoint turn {turn}\n".encode()
                commit = self._git(args, input_bytes=message).stdout.decode().strip()
                transaction = (
                    f"start\nupdate {_HEAD_REF} {commit}\n"
                    f"update {self._turn_ref(turn)} {commit}\nprepare\ncommit\n"
                ).encode()
                self._git(["update-ref", "--stdin"], input_bytes=transaction)
                os.replace(temporary_index, self.shadow_dir / "index")
            finally:
                try:
                    temporary_index.unlink()
                except FileNotFoundError:
                    pass
            checkpoint = WorkspaceCheckpoint(
                turn=turn,
                commit=commit,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
                file_count=len(paths),
                byte_count=byte_count,
                captured_at=_iso_timestamp(self._clock),
            )
            self._append_metric(checkpoint)
            return checkpoint

    def _resolve_turn(self, turn: int) -> str:
        self._ensure_initialized()
        proc = self._git(["rev-parse", "--verify", self._turn_ref(int(turn))], check=False)
        commit = proc.stdout.decode().strip()
        if proc.returncode != 0 or len(commit) < 40:
            raise CheckpointNotFoundError(f"no workspace checkpoint for turn {turn}")
        return commit

    def checkpoint_for_turn(self, turn: int) -> str:
        """Return the exact checkpoint commit bound to ``turn``."""
        with self._lock:
            return self._resolve_turn(turn)

    def _tree_entries(self, commit: str) -> dict[str, _TreeEntry]:
        output = self._git(["ls-tree", "-rz", commit]).stdout
        entries: dict[str, _TreeEntry] = {}
        for raw in output.split(b"\x00"):
            if not raw:
                continue
            header, raw_path = raw.split(b"\t", 1)
            mode, object_type, object_id = header.decode().split(" ", 2)
            if object_type != "blob":
                raise WorkspaceCheckpointError(f"unsupported tree object type: {object_type}")
            rel_path = _safe_relative_path(os.fsdecode(raw_path))
            entries[rel_path] = _TreeEntry(int(mode, 8), object_id, rel_path)
        return entries

    def _remove_path(self, rel_path: str) -> bool:
        target = self.workspace / rel_path
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.exists():
            return False
        else:
            return False
        parent = target.parent
        while parent != self.workspace:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
        return True

    def _write_regular_file(self, target: Path, data: bytes, mode: int) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=".yuj-restore-", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            _write_all(fd, data)
            os.fchmod(fd, 0o755 if mode & 0o111 else 0o644)
            os.fsync(fd)
            os.close(fd)
            fd = -1
            if target.is_dir() and not target.is_symlink():
                try:
                    target.rmdir()
                except OSError as exc:
                    raise WorkspaceCheckpointError(
                        "cannot replace non-empty directory at "
                        f"{target.relative_to(self.workspace)}"
                    ) from exc
            os.replace(temporary, target)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def restore_checkpoint(self, turn: int) -> RestoredCheckpoint:
        with self._lock:
            commit = self._resolve_turn(turn)
            target_entries = self._tree_entries(commit)
            current_paths = set(self._candidate_paths())
            removed = sum(
                self._remove_path(path)
                for path in sorted(
                    current_paths - target_entries.keys(), reverse=True
                )
            )
            bytes_restored = 0
            for entry in target_entries.values():
                target = self.workspace / entry.path
                data = self._git(["cat-file", "blob", entry.object_id]).stdout
                bytes_restored += len(data)
                if entry.mode == 0o120000:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.is_symlink() or target.is_file():
                        target.unlink()
                    elif target.exists():
                        try:
                            target.rmdir()
                        except OSError as exc:
                            raise WorkspaceCheckpointError(
                                f"cannot replace non-empty directory at {entry.path}"
                            ) from exc
                    os.symlink(os.fsdecode(data), target)
                else:
                    self._write_regular_file(target, data, entry.mode)
            self._git(["read-tree", commit])
            return RestoredCheckpoint(
                turn=int(turn),
                commit=commit,
                files_restored=len(target_entries),
                files_removed=removed,
                bytes_restored=bytes_restored,
            )

    def metrics_payload(self) -> dict[str, object]:
        self._ensure_initialized()
        rows: list[dict] = []
        path = self.shadow_dir / _METRICS_LOG
        if path.is_file():
            for line in path.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        return {
            "enabled": True,
            "count": len(rows),
            "total_duration_ms": round(
                sum(float(row.get("duration_ms", 0.0) or 0.0) for row in rows), 3
            ),
            "per_call": rows,
        }


def restore_checkpoint(
    workspace: Path,
    turn: int,
    *,
    shadow_dir: Path | None = None,
    excludes: Iterable[str] = (),
) -> RestoredCheckpoint:
    return WorkspaceCheckpointStore(
        workspace, shadow_dir=shadow_dir, excludes=excludes
    ).restore_checkpoint(turn)
