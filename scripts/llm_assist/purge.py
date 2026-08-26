"""Fail-closed preview and deletion for one archived assistant session."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..llm_solver.harness.worktree_runtime import (
    WorktreeRuntimeError,
    inspect_session_worktree_presence,
)
from .store import (
    SessionLockedError,
    SessionPurgeJournal,
    SessionPurgeStateError,
    SessionRecord,
    SessionStore,
    is_full_session_id,
)


MAX_ARTIFACT_ENTRIES = 100_000
MAX_ARTIFACT_PATH_BYTES = 4 * 1024 * 1024
MAX_RELATIVE_PATH_BYTES = 4096
MAX_METADATA_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_ESTIMATED_BYTES = (1 << 63) - 1
_STAGING_DIRECTORY = "purge-staging"
_JOURNAL_PHASES = frozenset(
    {"prepared", "staged", "artifacts_removed", "completed"}
)


class PurgeSessionError(RuntimeError):
    """The selected session cannot be purged or remains incomplete."""

    def __init__(
        self,
        message: str,
        *,
        incomplete: bool = False,
        preview: PurgePreview | None = None,
    ):
        self.incomplete = incomplete
        self.preview = preview
        super().__init__(message)


@dataclass(frozen=True)
class ArtifactEntry:
    relative: str
    kind: str
    mode: int
    size: int
    dev: int
    ino: int
    nlink: int
    mtime_ns: int
    ctime_ns: int

    def as_json(self) -> dict[str, object]:
        return {
            "relative": self.relative,
            "kind": self.kind,
            "mode": self.mode,
            "size": self.size,
            "dev": self.dev,
            "ino": self.ino,
            "nlink": self.nlink,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


@dataclass(frozen=True)
class ArtifactSnapshot:
    root_dev: int
    root_ino: int
    entries: tuple[ArtifactEntry, ...]

    @property
    def estimated_bytes(self) -> int:
        return sum(entry.size for entry in self.entries if entry.kind == "file")

    @property
    def manifest_json(self) -> str:
        return json.dumps(
            [entry.as_json() for entry in self.entries],
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def manifest_digest(self) -> str:
        return hashlib.sha256(self.manifest_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PurgePreview:
    session_id: str
    state: str
    entries: tuple[ArtifactEntry, ...]
    entry_count: int
    estimated_bytes: int
    remaining_entries: int
    remaining_bytes: int
    failure_detail: str | None = None


def preview_session_purge(store: SessionStore, session_id: str) -> PurgePreview:
    """Inspect one exact session and its journal without changing either."""
    _require_full_session_id(session_id)
    record = store.get_session(session_id)
    journal = store.get_session_purge(session_id)
    if record is None:
        if journal is not None and journal.phase == "completed":
            return _completed_preview(store, journal)
        raise PurgeSessionError(f"unknown session: {session_id}")
    if journal is not None:
        if journal.phase == "completed":
            raise PurgeSessionError(
                "completed purge journal still has a session row"
            )
        return _journal_preview(store, record, journal)
    snapshot = _inspect_new_candidate(store, record)
    return PurgePreview(
        session_id=session_id,
        state="ready",
        entries=snapshot.entries,
        entry_count=len(snapshot.entries),
        estimated_bytes=snapshot.estimated_bytes,
        remaining_entries=len(snapshot.entries),
        remaining_bytes=snapshot.estimated_bytes,
    )


def purge_archived_session(store: SessionStore, session_id: str) -> PurgePreview:
    """Delete one confirmed archived identity and its exact owned tree."""
    _require_full_session_id(session_id)
    record = store.get_session(session_id)
    journal = store.get_session_purge(session_id)
    if record is None:
        if journal is not None and journal.phase == "completed":
            return _completed_preview(store, journal)
        raise PurgeSessionError(f"unknown session: {session_id}")
    if journal is not None and journal.phase == "completed":
        raise PurgeSessionError(
            "completed purge journal still has a session row"
        )

    initial_snapshot: ArtifactSnapshot | None = None
    if journal is None:
        initial_snapshot = _inspect_new_candidate(store, record)
    else:
        _validate_record_boundary(store, record)
        _validate_retry_state(store, record)

    lock = store.get_session_lock(session_id)
    if lock is not None:
        raise PurgeSessionError(
            "cannot purge a locked session; "
            f"pid {lock.owner_pid} on {lock.owner_host} holds the lock"
        )
    try:
        store.acquire_session_lock(session_id)
    except SessionLockedError as exc:
        raise PurgeSessionError(f"cannot purge a locked session: {exc}") from exc

    acquired = True
    try:
        current = store.get_session(session_id)
        if current is None:
            raise PurgeSessionError("session disappeared before purge started")
        record = current
        _validate_record_boundary(store, record)
        _validate_retry_state(store, record, allow_owned_lock=True)
        if journal is None:
            assert initial_snapshot is not None
            source = _source_path(store, session_id)
            current_snapshot = _snapshot_artifacts(source)
            _validate_artifact_metadata(record, current_snapshot, source)
            if current_snapshot != initial_snapshot:
                raise PurgeSessionError(
                    "session artifacts changed before purge preparation"
                )
            try:
                journal = store.prepare_session_purge(
                    record,
                    manifest_json=current_snapshot.manifest_json,
                    manifest_digest=current_snapshot.manifest_digest,
                    entry_count=len(current_snapshot.entries),
                    estimated_bytes=current_snapshot.estimated_bytes,
                    root_dev=current_snapshot.root_dev,
                    root_ino=current_snapshot.root_ino,
                )
            except SessionPurgeStateError as exc:
                raise PurgeSessionError(str(exc)) from exc
            _purge_boundary("after_journal_prepare")
        assert journal is not None
        completed = _resume_journaled_purge(store, record, journal)
        acquired = False
        return completed
    except BaseException as exc:
        current_journal = store.get_session_purge(session_id)
        if current_journal is None:
            if isinstance(exc, PurgeSessionError):
                raise
            detail = _bounded_failure_detail(exc)
            raise PurgeSessionError(detail) from exc
        if current_journal.phase == "completed":
            return _completed_preview(store, current_journal)
        detail = _bounded_failure_detail(exc)
        try:
            store.record_session_purge_failure(session_id, detail)
        except Exception as journal_exc:
            detail = (
                detail
                + "; failure journal update also failed: "
                + _bounded_failure_detail(journal_exc)
            )[:512]
        try:
            refreshed = store.get_session(session_id)
            refreshed_journal = store.get_session_purge(session_id)
            report = (
                _journal_preview(store, refreshed, refreshed_journal)
                if refreshed is not None and refreshed_journal is not None
                else None
            )
        except Exception:
            report = None
        raise PurgeSessionError(
            f"purge incomplete: {detail}",
            incomplete=True,
            preview=report,
        ) from exc
    finally:
        if acquired:
            store.release_session_lock(session_id)


def _inspect_new_candidate(
    store: SessionStore,
    record: SessionRecord,
) -> ArtifactSnapshot:
    _validate_record_boundary(store, record)
    _validate_retry_state(store, record)
    source = _source_path(store, record.session_id)
    snapshot = _snapshot_artifacts(source)
    _validate_artifact_metadata(record, snapshot, source)
    return snapshot


def _validate_record_boundary(store: SessionStore, record: SessionRecord) -> None:
    if not is_full_session_id(record.session_id):
        raise PurgeSessionError("saved session has a malformed immutable ID")
    expected = _source_path(store, record.session_id)
    if (
        not isinstance(record.artifact_dir, str)
        or record.artifact_dir != str(expected)
    ):
        raise PurgeSessionError(
            "saved artifact path does not match its assistant-root artifact boundary"
        )
    try:
        config_paths = json.loads(record.config_paths_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise PurgeSessionError("saved session metadata is malformed") from exc
    if not isinstance(config_paths, list) or not all(
        isinstance(value, str) for value in config_paths
    ):
        raise PurgeSessionError("saved session metadata is malformed")
    if record.parent_session_id is not None and (
        not isinstance(record.parent_session_id, str)
        or not is_full_session_id(record.parent_session_id)
    ):
        raise PurgeSessionError("saved session lineage metadata is malformed")


def _validate_retry_state(
    store: SessionStore,
    record: SessionRecord,
    *,
    allow_owned_lock: bool = False,
) -> None:
    if record.archived_at is None:
        raise PurgeSessionError("session must be archived before purge")
    if not isinstance(record.archived_at, str) or not record.archived_at:
        raise PurgeSessionError("saved archive metadata is malformed")
    if not isinstance(record.status, str) or not record.status:
        raise PurgeSessionError("saved session status is malformed")
    identity = (
        record.worktree_path,
        record.worktree_branch,
        record.worktree_base_commit,
    )
    if identity != (None, None, None):
        if not all(isinstance(value, str) and value for value in identity):
            raise PurgeSessionError(
                "saved managed worktree metadata is malformed"
            )
        worktree_steps = (
            f"run yuj unarchive {record.session_id}, then yuj worktree rm "
            f"{record.session_id}, then archive it again before purge"
        )
        try:
            expected_path, inspected = inspect_session_worktree_presence(
                Path(record.cwd),
                record.session_id,
            )
        except (ValueError, WorktreeRuntimeError) as exc:
            raise PurgeSessionError(
                "cannot prove that the managed worktree was removed; "
                f"{worktree_steps}: {exc}"
            ) from exc
        if Path(str(record.worktree_path)) != expected_path:
            raise PurgeSessionError(
                "saved managed worktree path does not match its derived boundary"
            )
        if inspected is not None:
            saved_identity = (
                expected_path,
                str(record.worktree_branch),
                str(record.worktree_base_commit),
            )
            live_identity = (
                inspected.worktree_path,
                inspected.branch,
                inspected.base_commit,
            )
            if live_identity != saved_identity:
                raise PurgeSessionError(
                    "saved managed worktree metadata does not match the live "
                    "owned worktree"
                )
            raise PurgeSessionError(
                f"session retains a live managed worktree; {worktree_steps}"
            )
    if record.session_id in store.list_active_session_ids():
        raise PurgeSessionError(
            "cannot purge a session with an active-session pointer"
        )
    if record.status == "running":
        raise PurgeSessionError("cannot purge a running session")
    lock = store.get_session_lock(record.session_id)
    if lock is not None and not (
        allow_owned_lock
        and lock.owner_host == socket.gethostname()
        and lock.owner_pid == os.getpid()
    ):
        raise PurgeSessionError(
            "cannot purge a locked session; "
            f"pid {lock.owner_pid} on {lock.owner_host} holds the lock"
        )


def _validate_artifact_metadata(
    record: SessionRecord,
    snapshot: ArtifactSnapshot,
    root: Path,
) -> None:
    metadata = _read_json_entry(
        root,
        snapshot,
        "session.json",
    )
    if metadata is None:
        raise PurgeSessionError("saved session metadata is missing")
    if metadata.get("session_id") != record.session_id:
        raise PurgeSessionError(
            "saved session metadata identity does not match the SQLite row"
        )
    if metadata.get("parent_session_id") != record.parent_session_id:
        raise PurgeSessionError(
            "saved session metadata lineage does not match the SQLite row"
        )

    approval = _read_json_entry(
        root,
        snapshot,
        "approval_request.json",
    )
    if approval is not None:
        status_value = approval.get("status")
        if status_value == "pending":
            raise PurgeSessionError(
                "cannot purge a session with a pending approval"
            )
        if status_value not in {"approved", "rejected", "rewound"}:
            raise PurgeSessionError("saved approval metadata is malformed")

    request = _read_json_entry(
        root,
        snapshot,
        "clarification_request.json",
    )
    answer = _read_json_entry(
        root,
        snapshot,
        "clarification_answer.json",
    )
    consumption = _read_json_entry(
        root,
        snapshot,
        "clarification_consumption.json",
    )
    if request is None and (answer is not None or consumption is not None):
        raise PurgeSessionError("saved clarification metadata is malformed")
    if request is not None:
        request_id = request.get("request_id")
        request_status = request.get("status")
        if (
            not isinstance(request_id, str)
            or not request_id
            or request.get("session_id") != record.session_id
            or request_status not in {"pending", "rewound"}
        ):
            raise PurgeSessionError("saved clarification metadata is malformed")
        if answer is not None and (
            answer.get("request_id") != request_id
            or answer.get("session_id") != record.session_id
        ):
            raise PurgeSessionError("saved clarification metadata is malformed")
        if consumption is not None and (
            answer is None or consumption.get("request_id") != request_id
        ):
            raise PurgeSessionError("saved clarification metadata is malformed")
        if request_status == "pending" and consumption is None:
            raise PurgeSessionError(
                "cannot purge a session with a pending clarification"
            )
        if request_status == "rewound" and consumption is not None:
            raise PurgeSessionError("saved clarification metadata is malformed")

    correction = _read_json_entry(
        root,
        snapshot,
        "correction.json",
    )
    correction_consumption = _read_json_entry(
        root,
        snapshot,
        "correction_consumption.json",
    )
    if correction is None and correction_consumption is not None:
        raise PurgeSessionError("saved correction metadata is malformed")
    if correction is not None:
        correction_id = correction.get("correction_id")
        text_digest = correction.get("text_sha256")
        if (
            not isinstance(correction_id, str)
            or not correction_id
            or not isinstance(text_digest, str)
            or not text_digest
            or correction.get("status") != "pending"
        ):
            raise PurgeSessionError("saved correction metadata is malformed")
        if correction_consumption is None:
            raise PurgeSessionError(
                "cannot purge a session with a pending correction"
            )
        if (
            correction_consumption.get("correction_id") != correction_id
            or correction_consumption.get("text_sha256") != text_digest
        ):
            raise PurgeSessionError("saved correction metadata is malformed")


def _snapshot_artifacts(root: Path) -> ArtifactSnapshot:
    root = Path(root)
    root_metadata = _lstat_directory(root, label="saved artifact boundary")
    parent = root.parent
    assistant_root = parent.parent
    assistant_root_metadata = _lstat_directory(
        assistant_root,
        label="configured assistant root",
        reject_mount=False,
    )
    parent_metadata = _lstat_directory(
        parent,
        label="assistant artifact parent boundary",
    )
    if parent_metadata.st_dev != assistant_root_metadata.st_dev:
        raise PurgeSessionError(
            "assistant artifact parent boundary is a mount point"
        )
    if root_metadata.st_dev != parent_metadata.st_dev:
        raise PurgeSessionError("saved artifact boundary is a mount point")
    entries: list[ArtifactEntry] = []
    total_path_bytes = 0
    total_file_bytes = 0

    def visit(directory_fd: int, prefix: str) -> None:
        nonlocal total_file_bytes, total_path_bytes
        try:
            with os.scandir(directory_fd) as iterator:
                children = sorted(iterator, key=lambda item: item.name)
        except OSError as exc:
            raise PurgeSessionError(
                "cannot inspect saved artifact directory"
            ) from exc
        for child in children:
            name = child.name
            relative = f"{prefix}/{name}" if prefix else name
            try:
                relative.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise PurgeSessionError(
                    "saved artifact path is not valid UTF-8"
                ) from exc
            encoded_length = len(os.fsencode(relative))
            if encoded_length > MAX_RELATIVE_PATH_BYTES:
                raise PurgeSessionError("saved artifact path is too long")
            total_path_bytes += encoded_length
            if total_path_bytes > MAX_ARTIFACT_PATH_BYTES:
                raise PurgeSessionError("saved artifact paths exceed the preview bound")
            try:
                metadata = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise PurgeSessionError(
                    f"cannot inspect saved artifact path: {relative}"
                ) from exc
            if metadata.st_uid != os.geteuid():
                raise PurgeSessionError(
                    f"saved artifact has an unexpected owner: {relative}"
                )
            path = root.joinpath(*PurePosixPath(relative).parts)
            if stat.S_ISLNK(metadata.st_mode):
                raise PurgeSessionError(
                    f"saved artifact cannot be a symbolic link: {relative}"
                )
            if os.path.ismount(path):
                raise PurgeSessionError(
                    f"saved artifact entry is a mount point: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                flags |= getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                try:
                    child_fd = os.open(name, flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise PurgeSessionError(
                        f"cannot open saved artifact directory: {relative}"
                    ) from exc
                opened = os.fstat(child_fd)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or opened.st_dev != metadata.st_dev
                    or opened.st_ino != metadata.st_ino
                ):
                    os.close(child_fd)
                    raise PurgeSessionError(
                        f"saved artifact directory changed while opening: {relative}"
                    )
                if metadata.st_dev != root_metadata.st_dev:
                    os.close(child_fd)
                    raise PurgeSessionError(
                        f"saved artifact directory is a mount point: {relative}"
                    )
                kind = "dir"
                size = 0
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_dev != root_metadata.st_dev:
                    raise PurgeSessionError(
                        f"saved artifact file crosses a mount boundary: {relative}"
                    )
                if metadata.st_nlink != 1:
                    raise PurgeSessionError(
                        f"saved artifact has a hard link risk: {relative}"
                    )
                kind = "file"
                size = metadata.st_size
                total_file_bytes += size
                if total_file_bytes > MAX_ESTIMATED_BYTES:
                    raise PurgeSessionError(
                        "saved artifact bytes exceed the preview bound"
                    )
            else:
                raise PurgeSessionError(
                    f"saved artifact has an unsupported entry type: {relative}"
                )
            entries.append(
                ArtifactEntry(
                    relative=relative,
                    kind=kind,
                    mode=stat.S_IMODE(metadata.st_mode),
                    size=size,
                    dev=metadata.st_dev,
                    ino=metadata.st_ino,
                    nlink=metadata.st_nlink,
                    mtime_ns=metadata.st_mtime_ns,
                    ctime_ns=metadata.st_ctime_ns,
                )
            )
            if len(entries) > MAX_ARTIFACT_ENTRIES:
                raise PurgeSessionError("too many artifact entries to preview safely")
            if kind == "dir":
                try:
                    visit(child_fd, relative)
                finally:
                    os.close(child_fd)

    root_fd = _open_directory(root)
    try:
        opened_root = os.fstat(root_fd)
        if (
            opened_root.st_dev != root_metadata.st_dev
            or opened_root.st_ino != root_metadata.st_ino
        ):
            raise PurgeSessionError(
                "saved artifact boundary changed while opening"
            )
        visit(root_fd, "")
    finally:
        os.close(root_fd)
    entries.sort(key=lambda entry: entry.relative)
    return ArtifactSnapshot(
        root_dev=root_metadata.st_dev,
        root_ino=root_metadata.st_ino,
        entries=tuple(entries),
    )


def _read_json_entry(
    root: Path,
    snapshot: ArtifactSnapshot,
    relative: str,
) -> dict[str, object] | None:
    entries = {entry.relative: entry for entry in snapshot.entries}
    entry = entries.get(relative)
    if entry is None:
        return None
    if entry.kind != "file" or entry.size > MAX_METADATA_BYTES:
        raise PurgeSessionError(f"saved {relative} metadata is malformed")
    if len(PurePosixPath(relative).parts) != 1:
        raise PurgeSessionError(f"saved {relative} metadata is malformed")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    root_descriptor = _open_directory(root)
    try:
        opened_root = os.fstat(root_descriptor)
        if (
            opened_root.st_dev != snapshot.root_dev
            or opened_root.st_ino != snapshot.root_ino
        ):
            raise PurgeSessionError(
                "saved artifact boundary changed while reading metadata"
            )
        descriptor = os.open(relative, flags, dir_fd=root_descriptor)
    except OSError as exc:
        os.close(root_descriptor)
        raise PurgeSessionError(f"cannot read saved metadata: {relative}") from exc
    except BaseException:
        os.close(root_descriptor)
        raise
    try:
        opened = os.fstat(descriptor)
        _require_entry_identity(entry, opened)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_METADATA_BYTES:
                raise PurgeSessionError(f"saved {relative} metadata is malformed")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        _require_entry_identity(entry, after)
    finally:
        os.close(descriptor)
        os.close(root_descriptor)
    try:
        payload = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PurgeSessionError(
            f"saved session metadata is malformed: {relative}"
        ) from exc
    if not isinstance(payload, dict):
        raise PurgeSessionError(f"saved {relative} metadata is malformed")
    return payload


def _require_entry_identity(entry: ArtifactEntry, metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_dev != entry.dev
        or metadata.st_ino != entry.ino
        or metadata.st_size != entry.size
        or metadata.st_nlink != 1
        or metadata.st_mtime_ns != entry.mtime_ns
        or metadata.st_ctime_ns != entry.ctime_ns
    ):
        raise PurgeSessionError(
            f"saved artifact changed while reading metadata: {entry.relative}"
        )


def _resume_journaled_purge(
    store: SessionStore,
    record: SessionRecord,
    journal: SessionPurgeJournal,
) -> PurgePreview:
    expected = _snapshot_from_journal(journal)
    source = _source_path(store, record.session_id)
    staged = _staged_path(store, record.session_id)
    journal = store.get_session_purge(record.session_id) or journal

    if journal.phase == "prepared":
        source_exists = _lexists(source)
        staged_exists = _lexists(staged)
        if source_exists and staged_exists:
            raise PurgeSessionError(
                "both live and staged artifact boundaries exist"
            )
        if source_exists:
            actual = _snapshot_artifacts(source)
            _require_snapshot_matches(expected, actual, require_full=True)
            _ensure_staging_parent(store)
            _rename_to_staging(store, journal)
            _purge_boundary("after_artifact_stage")
        elif staged_exists:
            actual = _snapshot_artifacts(staged)
            _require_snapshot_matches(expected, actual, require_full=True)
        else:
            raise PurgeSessionError(
                "prepared purge has neither live nor staged artifacts"
            )
        journal = store.transition_session_purge(
            record.session_id,
            expected={"prepared"},
            phase="staged",
        )

    if journal.phase == "staged":
        if _lexists(source):
            raise PurgeSessionError(
                "live artifact boundary reappeared after purge staging"
            )
        if _lexists(staged):
            actual = _snapshot_artifacts(staged)
            _require_snapshot_matches(expected, actual, require_full=False)
            ordered = sorted(
                actual.entries,
                key=lambda entry: (
                    len(PurePosixPath(entry.relative).parts),
                    entry.relative,
                ),
                reverse=True,
            )
            expected_by_path = {
                entry.relative: entry for entry in expected.entries
            }
            for entry in ordered:
                _remove_staged_entry(
                    staged,
                    expected_by_path[entry.relative],
                    expected_by_path=expected_by_path,
                    root_dev=expected.root_dev,
                    root_ino=expected.root_ino,
                )
                _purge_boundary("after_artifact_entry_remove")
            _require_empty_staged_root(staged, expected)
            _remove_staged_root(store, journal)
            _purge_boundary("after_artifact_root_remove")
        journal = store.transition_session_purge(
            record.session_id,
            expected={"staged"},
            phase="artifacts_removed",
        )

    if journal.phase == "artifacts_removed":
        if _lexists(source) or _lexists(staged):
            raise PurgeSessionError(
                "artifact boundary remains after removal was journaled"
            )
        _purge_boundary("before_session_row_remove")
        try:
            journal = store.finalize_session_purge(record)
        except SessionPurgeStateError as exc:
            raise PurgeSessionError(str(exc)) from exc
        _purge_boundary("after_session_row_remove")

    if journal.phase != "completed":
        raise PurgeSessionError(f"unsupported purge phase: {journal.phase}")
    return _completed_preview(store, journal)


def _journal_preview(
    store: SessionStore,
    record: SessionRecord,
    journal: SessionPurgeJournal,
) -> PurgePreview:
    expected = _snapshot_from_journal(journal)
    source = _source_path(store, record.session_id)
    staged = _staged_path(store, record.session_id)
    source_exists = _lexists(source)
    staged_exists = _lexists(staged)
    if source_exists and staged_exists:
        raise PurgeSessionError(
            "both live and staged artifact boundaries exist"
        )
    if source_exists:
        actual = _snapshot_artifacts(source)
        _require_snapshot_matches(
            expected,
            actual,
            require_full=journal.phase == "prepared",
        )
    elif staged_exists:
        actual = _snapshot_artifacts(staged)
        _require_snapshot_matches(expected, actual, require_full=False)
    else:
        actual = ArtifactSnapshot(
            root_dev=journal.root_dev,
            root_ino=journal.root_ino,
            entries=(),
        )
        if journal.phase == "prepared":
            raise PurgeSessionError(
                "prepared purge has neither live nor staged artifacts"
            )
    return PurgePreview(
        session_id=record.session_id,
        state=journal.phase,
        entries=actual.entries,
        entry_count=journal.entry_count,
        estimated_bytes=journal.estimated_bytes,
        remaining_entries=len(actual.entries),
        remaining_bytes=actual.estimated_bytes,
        failure_detail=journal.failure_detail,
    )


def _snapshot_from_journal(journal: SessionPurgeJournal) -> ArtifactSnapshot:
    _validate_journal_summary(journal)
    try:
        manifest_bytes = journal.manifest_json.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise PurgeSessionError("purge journal manifest is malformed") from exc
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise PurgeSessionError("purge journal manifest exceeds its bound")
    try:
        payload = json.loads(journal.manifest_json)
    except json.JSONDecodeError as exc:
        raise PurgeSessionError("purge journal manifest is malformed") from exc
    if not isinstance(payload, list):
        raise PurgeSessionError("purge journal manifest is malformed")
    entries = tuple(_entry_from_json(value) for value in payload)
    if len(entries) > MAX_ARTIFACT_ENTRIES:
        raise PurgeSessionError("purge journal has too many artifact entries")
    if tuple(sorted(entries, key=lambda entry: entry.relative)) != entries:
        raise PurgeSessionError("purge journal manifest order is malformed")
    relative_paths = [entry.relative for entry in entries]
    if len(set(relative_paths)) != len(relative_paths):
        raise PurgeSessionError("purge journal has duplicate artifact paths")
    total_path_bytes = 0
    for entry in entries:
        try:
            encoded_length = len(entry.relative.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise PurgeSessionError("purge journal path is malformed") from exc
        if encoded_length > MAX_RELATIVE_PATH_BYTES:
            raise PurgeSessionError("purge journal path exceeds its bound")
        total_path_bytes += encoded_length
        if total_path_bytes > MAX_ARTIFACT_PATH_BYTES:
            raise PurgeSessionError("purge journal paths exceed their bound")
        if entry.dev != journal.root_dev:
            raise PurgeSessionError(
                "purge journal entry crosses its artifact boundary"
            )
        if entry.mode > 0o7777:
            raise PurgeSessionError("purge journal entry mode is malformed")
        if entry.kind == "dir" and entry.size != 0:
            raise PurgeSessionError("purge journal directory size is malformed")
        if entry.kind == "file" and entry.nlink != 1:
            raise PurgeSessionError("purge journal has a hard link risk")
    if len(entries) != journal.entry_count and journal.phase != "completed":
        raise PurgeSessionError("purge journal entry count is malformed")
    snapshot = ArtifactSnapshot(
        root_dev=journal.root_dev,
        root_ino=journal.root_ino,
        entries=entries,
    )
    if journal.phase != "completed" and (
        snapshot.manifest_digest != journal.manifest_digest
        or snapshot.estimated_bytes != journal.estimated_bytes
    ):
        raise PurgeSessionError("purge journal manifest does not match its summary")
    return snapshot


def _validate_journal_summary(journal: SessionPurgeJournal) -> None:
    if not is_full_session_id(journal.session_id):
        raise PurgeSessionError("purge journal session ID is malformed")
    if journal.phase not in _JOURNAL_PHASES:
        raise PurgeSessionError("purge journal has an unsupported phase")
    if not 0 <= journal.entry_count <= MAX_ARTIFACT_ENTRIES:
        raise PurgeSessionError("purge journal entry count is malformed")
    if not 0 <= journal.estimated_bytes <= MAX_ESTIMATED_BYTES:
        raise PurgeSessionError("purge journal byte estimate is malformed")
    if journal.root_dev < 0 or journal.root_ino <= 0:
        raise PurgeSessionError("purge journal root identity is malformed")
    if len(journal.manifest_digest) != 64 or any(
        character not in "0123456789abcdef"
        for character in journal.manifest_digest
    ):
        raise PurgeSessionError("purge journal digest is malformed")
    if journal.failure_detail is not None and len(journal.failure_detail) > 512:
        raise PurgeSessionError("purge journal failure detail is malformed")
    if journal.phase == "completed":
        if (
            journal.manifest_json != "[]"
            or journal.completed_at is None
            or journal.failure_detail is not None
        ):
            raise PurgeSessionError("completed purge journal is malformed")
    elif journal.completed_at is not None:
        raise PurgeSessionError("incomplete purge journal is malformed")


def _entry_from_json(value: object) -> ArtifactEntry:
    keys = {
        "relative",
        "kind",
        "mode",
        "size",
        "dev",
        "ino",
        "nlink",
        "mtime_ns",
        "ctime_ns",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise PurgeSessionError("purge journal entry is malformed")
    relative = value["relative"]
    kind = value["kind"]
    numbers = [
        value[key]
        for key in (
            "mode",
            "size",
            "dev",
            "ino",
            "nlink",
            "mtime_ns",
            "ctime_ns",
        )
    ]
    if (
        not isinstance(relative, str)
        or not relative
        or kind not in {"file", "dir"}
        or not all(type(number) is int and number >= 0 for number in numbers)
    ):
        raise PurgeSessionError("purge journal entry is malformed")
    path = PurePosixPath(relative)
    if (
        "\x00" in relative
        or path.is_absolute()
        or path.as_posix() != relative
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PurgeSessionError("purge journal path is malformed")
    return ArtifactEntry(
        relative=relative,
        kind=str(kind),
        mode=int(value["mode"]),
        size=int(value["size"]),
        dev=int(value["dev"]),
        ino=int(value["ino"]),
        nlink=int(value["nlink"]),
        mtime_ns=int(value["mtime_ns"]),
        ctime_ns=int(value["ctime_ns"]),
    )


def _require_snapshot_matches(
    expected: ArtifactSnapshot,
    actual: ArtifactSnapshot,
    *,
    require_full: bool,
) -> None:
    if (
        actual.root_dev != expected.root_dev
        or actual.root_ino != expected.root_ino
    ):
        raise PurgeSessionError("artifact boundary identity changed")
    expected_by_path = {entry.relative: entry for entry in expected.entries}
    actual_by_path = {entry.relative: entry for entry in actual.entries}
    unexpected = sorted(actual_by_path.keys() - expected_by_path.keys())
    if unexpected:
        raise PurgeSessionError(
            "artifact boundary gained an unexpected entry: " + unexpected[0]
        )
    if require_full and actual_by_path.keys() != expected_by_path.keys():
        raise PurgeSessionError("artifact boundary lost an entry before staging")
    for relative, observed in actual_by_path.items():
        saved = expected_by_path[relative]
        if not _same_entry(saved, observed):
            raise PurgeSessionError(
                f"artifact entry identity changed: {relative}"
            )


def _same_entry(expected: ArtifactEntry, actual: ArtifactEntry) -> bool:
    common = (
        expected.relative == actual.relative
        and expected.kind == actual.kind
        and expected.mode == actual.mode
        and expected.dev == actual.dev
        and expected.ino == actual.ino
    )
    if not common:
        return False
    if expected.kind == "dir":
        return True
    return (
        expected.size == actual.size
        and actual.nlink == 1
        and expected.mtime_ns == actual.mtime_ns
        and expected.ctime_ns == actual.ctime_ns
    )


def _ensure_staging_parent(store: SessionStore) -> Path:
    root = Path(store.root)
    staging = root / _STAGING_DIRECTORY
    root_metadata = _lstat_directory(
        root,
        label="configured assistant root",
        reject_mount=False,
    )
    sessions_metadata = _lstat_directory(
        root / "sessions",
        label="assistant sessions boundary",
    )
    if sessions_metadata.st_dev != root_metadata.st_dev:
        raise PurgeSessionError("assistant sessions boundary crosses a mount point")
    if not _lexists(staging):
        try:
            os.mkdir(staging, mode=0o700)
        except OSError as exc:
            raise PurgeSessionError("cannot create purge staging boundary") from exc
        _fsync_directory(root)
    metadata = _lstat_directory(staging, label="purge staging boundary")
    if metadata.st_dev != root_metadata.st_dev:
        raise PurgeSessionError("purge staging boundary crosses a mount point")
    return staging


def _rename_to_staging(
    store: SessionStore,
    journal: SessionPurgeJournal,
) -> None:
    sessions = Path(store.root) / "sessions"
    staging = Path(store.root) / _STAGING_DIRECTORY
    sessions_metadata = _lstat_directory(
        sessions,
        label="assistant sessions boundary",
    )
    staging_metadata = _lstat_directory(
        staging,
        label="purge staging boundary",
    )
    if sessions_metadata.st_dev != staging_metadata.st_dev:
        raise PurgeSessionError("purge staging boundary crosses a mount point")
    sessions_fd = _open_directory(sessions)
    staging_fd = _open_directory(staging)
    try:
        opened_sessions = os.fstat(sessions_fd)
        opened_staging = os.fstat(staging_fd)
        if (
            opened_sessions.st_dev != sessions_metadata.st_dev
            or opened_sessions.st_ino != sessions_metadata.st_ino
            or opened_staging.st_dev != staging_metadata.st_dev
            or opened_staging.st_ino != staging_metadata.st_ino
        ):
            raise PurgeSessionError(
                "assistant artifact parent changed before staging"
            )
        current = os.stat(
            journal.session_id,
            dir_fd=sessions_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(current.st_mode)
            or current.st_dev != journal.root_dev
            or current.st_ino != journal.root_ino
        ):
            raise PurgeSessionError("artifact boundary identity changed before staging")
        try:
            os.stat(
                journal.session_id,
                dir_fd=staging_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise PurgeSessionError("purge staging path already exists")
        os.rename(
            journal.session_id,
            journal.session_id,
            src_dir_fd=sessions_fd,
            dst_dir_fd=staging_fd,
        )
        os.fsync(sessions_fd)
        os.fsync(staging_fd)
    except OSError as exc:
        raise PurgeSessionError("cannot stage the owned artifact boundary") from exc
    finally:
        os.close(staging_fd)
        os.close(sessions_fd)


def _remove_staged_entry(
    root: Path,
    expected: ArtifactEntry,
    *,
    expected_by_path: dict[str, ArtifactEntry],
    root_dev: int,
    root_ino: int,
) -> None:
    parts = PurePosixPath(expected.relative).parts
    descriptor = _open_directory(root)
    try:
        opened_root = os.fstat(descriptor)
        if opened_root.st_dev != root_dev or opened_root.st_ino != root_ino:
            raise PurgeSessionError("staged artifact boundary identity changed")
        parent_parts: list[str] = []
        for part in parts[:-1]:
            next_descriptor = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                parent_parts.append(part)
                parent_relative = PurePosixPath(*parent_parts).as_posix()
                parent_expected = expected_by_path.get(parent_relative)
                opened_parent = os.fstat(next_descriptor)
                parent_observed = ArtifactEntry(
                    relative=parent_relative,
                    kind=(
                        "dir"
                        if stat.S_ISDIR(opened_parent.st_mode)
                        else "other"
                    ),
                    mode=stat.S_IMODE(opened_parent.st_mode),
                    size=0,
                    dev=opened_parent.st_dev,
                    ino=opened_parent.st_ino,
                    nlink=opened_parent.st_nlink,
                    mtime_ns=opened_parent.st_mtime_ns,
                    ctime_ns=opened_parent.st_ctime_ns,
                )
                if parent_expected is None or not _same_entry(
                    parent_expected,
                    parent_observed,
                ):
                    raise PurgeSessionError(
                        f"artifact parent identity changed before removal: "
                        f"{parent_relative}"
                    )
            except BaseException:
                os.close(next_descriptor)
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        name = parts[-1]
        current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        observed = ArtifactEntry(
            relative=expected.relative,
            kind=(
                "dir"
                if stat.S_ISDIR(current.st_mode)
                else "file"
                if stat.S_ISREG(current.st_mode)
                else "other"
            ),
            mode=stat.S_IMODE(current.st_mode),
            size=current.st_size if stat.S_ISREG(current.st_mode) else 0,
            dev=current.st_dev,
            ino=current.st_ino,
            nlink=current.st_nlink,
            mtime_ns=current.st_mtime_ns,
            ctime_ns=current.st_ctime_ns,
        )
        if not _same_entry(expected, observed):
            raise PurgeSessionError(
                f"artifact entry identity changed before removal: {expected.relative}"
            )
        if expected.kind == "dir":
            os.rmdir(name, dir_fd=descriptor)
        else:
            os.unlink(name, dir_fd=descriptor)
        os.fsync(descriptor)
    except OSError as exc:
        raise PurgeSessionError(
            f"cannot remove owned artifact entry: {expected.relative}"
        ) from exc
    finally:
        os.close(descriptor)


def _require_empty_staged_root(
    root: Path,
    expected: ArtifactSnapshot,
) -> None:
    descriptor = _open_directory(root)
    try:
        opened = os.fstat(descriptor)
        if opened.st_dev != expected.root_dev or opened.st_ino != expected.root_ino:
            raise PurgeSessionError("staged artifact boundary identity changed")
        with os.scandir(descriptor) as iterator:
            if next(iterator, None) is not None:
                raise PurgeSessionError(
                    "staged artifact boundary gained unexpected entries"
                )
    finally:
        os.close(descriptor)


def _remove_staged_root(
    store: SessionStore,
    journal: SessionPurgeJournal,
) -> None:
    staging = Path(store.root) / _STAGING_DIRECTORY
    staging_metadata = _lstat_directory(
        staging,
        label="purge staging boundary",
    )
    descriptor = _open_directory(staging)
    try:
        opened_staging = os.fstat(descriptor)
        if (
            opened_staging.st_dev != staging_metadata.st_dev
            or opened_staging.st_ino != staging_metadata.st_ino
        ):
            raise PurgeSessionError(
                "purge staging boundary changed before root removal"
            )
        current = os.stat(
            journal.session_id,
            dir_fd=descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(current.st_mode)
            or current.st_dev != journal.root_dev
            or current.st_ino != journal.root_ino
        ):
            raise PurgeSessionError("staged artifact boundary identity changed")
        os.rmdir(journal.session_id, dir_fd=descriptor)
        os.fsync(descriptor)
    except OSError as exc:
        raise PurgeSessionError("cannot remove staged artifact boundary") from exc
    finally:
        os.close(descriptor)


def _lstat_directory(
    path: Path,
    *,
    label: str,
    reject_mount: bool = True,
) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise PurgeSessionError(f"{label} is missing") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise PurgeSessionError(f"{label} cannot be a symbolic link")
    if not stat.S_ISDIR(metadata.st_mode):
        raise PurgeSessionError(f"{label} is not a directory")
    if metadata.st_uid != os.geteuid():
        raise PurgeSessionError(f"{label} has an unexpected owner")
    if reject_mount and os.path.ismount(path):
        raise PurgeSessionError(f"{label} is a mount point")
    return metadata


def _open_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags)
    except OSError as exc:
        raise PurgeSessionError("cannot open owned artifact directory") from exc


def _source_path(store: SessionStore, session_id: str) -> Path:
    return Path(store.root) / "sessions" / session_id


def _staged_path(store: SessionStore, session_id: str) -> Path:
    return Path(store.root) / _STAGING_DIRECTORY / session_id


def _lexists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise PurgeSessionError("cannot inspect purge artifact boundary") from exc
    return True


def _fsync_directory(path: Path) -> None:
    descriptor = _open_directory(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _completed_preview(
    store: SessionStore,
    journal: SessionPurgeJournal,
) -> PurgePreview:
    _validate_journal_summary(journal)
    if _lexists(_source_path(store, journal.session_id)) or _lexists(
        _staged_path(store, journal.session_id)
    ):
        raise PurgeSessionError(
            "completed purge journal still has an artifact boundary"
        )
    return PurgePreview(
        session_id=journal.session_id,
        state="completed",
        entries=(),
        entry_count=journal.entry_count,
        estimated_bytes=journal.estimated_bytes,
        remaining_entries=0,
        remaining_bytes=0,
        failure_detail=None,
    )


def _bounded_failure_detail(exc: BaseException) -> str:
    detail = str(exc).replace("\n", " ").strip() or type(exc).__name__
    rendered = f"{type(exc).__name__}: {detail}"
    return rendered[:512]


def _require_full_session_id(value: str) -> None:
    if not is_full_session_id(value):
        raise PurgeSessionError(
            "purge requires one full immutable session ID"
        )


def _purge_boundary(_name: str) -> None:
    """Named no-op boundary used by interruption and recovery tests."""


__all__ = [
    "ArtifactEntry",
    "MAX_ARTIFACT_ENTRIES",
    "PurgePreview",
    "PurgeSessionError",
    "preview_session_purge",
    "purge_archived_session",
]
