"""Explicit, fail-closed saved-session fork creation."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import stat
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from ..llm_solver.harness._loop.interrupted_turn import append_trace_event_fsync
from ..llm_solver.harness._loop.trace_schema import TRACE_SCHEMA_VERSION
from ..llm_solver.harness.clarifications import (
    ClarificationStateError,
    clarification_state,
)
from ..llm_solver.harness.corrections import (
    CorrectionState,
    CorrectionStateError,
    validate_correction_trace,
)
from ..llm_solver.harness.worktree_runtime import (
    WorktreeRuntimeError,
    WorktreeRuntimeInfo,
    fork_session_worktree,
    inspect_session_worktree,
    remove_session_worktree,
)
from ..llm_solver.harness.workspace_checkpoints import (
    rebind_checkpoint_workspace,
)
from ._images import ImageInputError, load_session_images
from .github_context import GitHubContextError, load_github_context
from ._path_attachments import PathAttachmentError, load_path_attachments
from ._reviews import ReviewTargetError, load_review_target
from .store import SessionLockedError, SessionRecord, SessionStore


class ForkSessionError(RuntimeError):
    """A saved session cannot be forked without weakening ownership."""


@dataclass(frozen=True)
class ForkResult:
    parent: SessionRecord
    child: SessionRecord
    source_artifact_sha256: str


@dataclass(frozen=True)
class _ArtifactEntry:
    relative: str
    kind: str
    mode: int
    digest: str
    size: int


@dataclass(frozen=True)
class _ArtifactSnapshot:
    entries: tuple[_ArtifactEntry, ...]
    digest: str

    @property
    def by_path(self) -> dict[str, _ArtifactEntry]:
        return {entry.relative: entry for entry in self.entries}


def fork_saved_session(
    store: SessionStore,
    source: SessionRecord,
) -> ForkResult:
    """Create one stopped child whose mutable state is independently owned."""
    current = store.get_session(source.session_id)
    if current is None:
        raise ForkSessionError(f"unknown session: {source.session_id}")
    source = current
    _refuse_unavailable_source(store, source)
    try:
        store.acquire_session_lock(source.session_id)
    except SessionLockedError as exc:
        raise ForkSessionError(f"cannot fork a locked session: {exc}") from exc

    stage: Path | None = None
    final: Path | None = None
    stage_owned = False
    final_owned = False
    child_worktree_created = False
    child: SessionRecord | None = None
    try:
        source = store.get_session(source.session_id) or source
        _refuse_unavailable_source(store, source, allow_owned_lock=True)
        source_root = _source_artifact_root(store, source)
        before = _snapshot_artifact_tree(source_root)
        source_worktree = _validate_source_worktree(source)
        endpoint_session = _validate_source_evidence(source, before)

        child = store.prepare_forked_session(
            source,
            config_paths=[],
            system_prompt_path=None,
        )
        final = child.artifact_path
        rebased_config = _rebase_paths(
            source.config_paths,
            source_root=source_root,
            child_root=final,
        )
        rebased_system = _rebase_optional_path(
            source.system_prompt_path,
            source_root=source_root,
            child_root=final,
        )
        child = replace(
            child,
            config_paths_json=json.dumps([str(path) for path in rebased_config]),
            system_prompt_path=(
                str(rebased_system) if rebased_system is not None else None
            ),
        )
        if final.exists() or final.is_symlink():
            raise ForkSessionError(
                f"child artifact path already exists: {final}"
            )

        stage = final.parent / f".{child.session_id}.fork-{uuid.uuid4().hex}.tmp"
        stage.mkdir(mode=0o700, parents=False, exist_ok=False)
        stage_owned = True
        _copy_artifact_snapshot(source_root, stage, before)
        copied = _snapshot_artifact_tree(stage)
        if copied != before:
            raise ForkSessionError(
                "child artifact history does not match the selected source"
            )

        source_workspace = (
            source_worktree.session_cwd
            if source_worktree is not None
            else Path(source.cwd)
        ).resolve()
        child_workspace = source_workspace
        if source.worktree_path is not None:
            info = fork_session_worktree(
                Path(source.cwd),
                source_run_id=source.session_id,
                child_run_id=child.session_id,
            )
            child_worktree_created = True
            child_workspace = info.session_cwd.resolve()
            child = replace(
                child,
                worktree_path=str(info.worktree_path.resolve()),
                worktree_branch=info.branch,
                worktree_base_commit=info.base_commit,
            )

        checkpoint_dir = stage / ".shadow_git"
        if checkpoint_dir.exists():
            rebind_checkpoint_workspace(
                checkpoint_dir,
                source_workspace=source_workspace,
                child_workspace=child_workspace,
            )

        from .runner import _write_session_metadata

        _write_session_metadata(child, artifact_dir=stage)
        append_trace_event_fsync(
            stage / ".trace.jsonl",
            {
                "event": "session_fork",
                "trace_schema_version": TRACE_SCHEMA_VERSION,
                "session_number": endpoint_session,
                "session_id": child.session_id,
                "parent_session_id": source.session_id,
                "forked_at": _utc_now(),
                "source_artifact_sha256": before.digest,
            },
        )

        after = _snapshot_artifact_tree(source_root)
        if after != before:
            raise ForkSessionError(
                "source artifacts changed while the child was staged"
            )
        if store.get_session(source.session_id) != source:
            raise ForkSessionError(
                "source metadata changed while the child was staged"
            )
        if source.session_id in store.list_active_session_ids():
            raise ForkSessionError(
                "source session became active while the child was staged"
            )

        final.mkdir(mode=0o700, parents=False, exist_ok=False)
        final_owned = True
        for entry in sorted(os.scandir(stage), key=lambda item: item.name):
            os.rename(entry.path, final / entry.name)
        stage.rmdir()
        stage_owned = False
        stage = None
        _fsync_directory(final)
        _fsync_directory(final.parent)
        store.insert_forked_session(child, expected_parent=source)
        return ForkResult(
            parent=source,
            child=child,
            source_artifact_sha256=before.digest,
        )
    except BaseException as exc:
        cleanup_errors = _cleanup_partial_child(
            store,
            source,
            child,
            stage=stage,
            final=final,
            stage_owned=stage_owned,
            final_owned=final_owned,
            worktree_created=child_worktree_created,
        )
        detail = str(exc) or type(exc).__name__
        if cleanup_errors:
            detail += "; cleanup also failed: " + "; ".join(cleanup_errors)
        if isinstance(exc, ForkSessionError) and not cleanup_errors:
            raise
        raise ForkSessionError(f"fork failed before child publication: {detail}") from exc
    finally:
        store.release_session_lock(source.session_id)


def validate_correction_owner(
    record: SessionRecord,
    state: CorrectionState,
) -> None:
    """Keep a live correction local while permitting consumed ancestor history."""
    if state.correction is None:
        return
    owner = str(state.correction["session_id"])
    if owner == record.session_id:
        return
    ancestors = _lineage_ancestors(record)
    if state.phase == "consumed" and owner in ancestors:
        return
    raise CorrectionStateError("correction belongs to another session")


def _lineage_ancestors(record: SessionRecord) -> set[str]:
    if record.parent_session_id is None:
        return set()
    trace_path = record.artifact_path / ".trace.jsonl"
    try:
        events = _load_trace_objects(trace_path)
    except ForkSessionError as exc:
        raise CorrectionStateError(str(exc)) from exc
    parents: dict[str, str] = {}
    for event in events:
        if event.get("event") != "session_fork":
            continue
        child = event.get("session_id")
        parent = event.get("parent_session_id")
        if not isinstance(child, str) or not child:
            raise CorrectionStateError("fork trace has an invalid child identity")
        if not isinstance(parent, str) or not parent:
            raise CorrectionStateError("fork trace has an invalid parent identity")
        if child in parents and parents[child] != parent:
            raise CorrectionStateError("fork trace has conflicting parent identities")
        parents[child] = parent
    if parents.get(record.session_id) != record.parent_session_id:
        raise CorrectionStateError("fork trace does not match session parent identity")
    ancestors: set[str] = set()
    selected = record.parent_session_id
    while selected is not None:
        if selected in ancestors or selected == record.session_id:
            raise CorrectionStateError("fork trace contains a lineage cycle")
        ancestors.add(selected)
        selected = parents.get(selected)
    return ancestors


def _refuse_unavailable_source(
    store: SessionStore,
    source: SessionRecord,
    *,
    allow_owned_lock: bool = False,
) -> None:
    if source.archived_at is not None:
        raise ForkSessionError(
            "session is archived; run "
            f"yuj unarchive {source.short_id} before forking it"
        )
    if source.session_id in store.list_active_session_ids():
        raise ForkSessionError("cannot fork an active session")
    lock = store.get_session_lock(source.session_id)
    if lock is not None and not (
        allow_owned_lock
        and lock.owner_host == socket.gethostname()
        and lock.owner_pid == os.getpid()
    ):
        raise ForkSessionError(
            "cannot fork a locked session; "
            f"pid {lock.owner_pid} on {lock.owner_host} holds the lock"
        )


def _source_artifact_root(store: SessionStore, source: SessionRecord) -> Path:
    expected = store.root / "sessions" / source.session_id
    candidate = Path(source.artifact_dir)
    if candidate != expected:
        raise ForkSessionError(
            "saved artifact path is outside its session artifact boundary"
        )
    try:
        metadata = os.lstat(candidate)
    except OSError as exc:
        raise ForkSessionError("saved artifact boundary is missing") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ForkSessionError("saved artifact boundary cannot be a symbolic link")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ForkSessionError("saved artifact boundary is not a directory")
    return candidate


def _validate_source_evidence(
    source: SessionRecord,
    snapshot: _ArtifactSnapshot,
) -> int:
    entries = snapshot.by_path
    for required in (
        "prompt.txt",
        ".solver/state.json",
        "session.json",
        ".trace.jsonl",
    ):
        entry = entries.get(required)
        if entry is None or entry.kind != "file":
            raise ForkSessionError(f"saved session is missing {required}")

    if (source.artifact_path / "prompt.txt").read_text() != source.prompt_text:
        raise ForkSessionError("saved prompt does not match session metadata")
    state = _load_json_object(
        source.artifact_path / ".solver" / "state.json",
        "state evidence",
    )
    if not isinstance(state.get("state"), dict):
        raise ForkSessionError("saved state evidence is malformed")
    metadata = _load_json_object(
        source.artifact_path / "session.json",
        "session metadata",
    )
    expected = {
        "session_id": source.session_id,
        "cwd": source.cwd,
        "model": source.model,
        "provider": source.provider,
        "authentication": source.auth_method,
        "prompt_source": source.prompt_source,
        "context_mode": source.context_mode,
        "system_prompt_path": source.system_prompt_path,
        "config_paths": source.config_paths,
        "worktree_path": source.worktree_path,
        "worktree_branch": source.worktree_branch,
        "worktree_base_commit": source.worktree_base_commit,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            label = "session metadata identity" if key == "session_id" else "session metadata"
            raise ForkSessionError(f"{label} does not match the SQLite record")
    if metadata.get("parent_session_id") != source.parent_session_id:
        raise ForkSessionError(
            "session metadata parent identity does not match the SQLite record"
        )
    _validate_recorded_config_hashes(source, metadata)
    _validate_system_prompt_path(source, entries)
    events = _load_trace_objects(source.artifact_path / ".trace.jsonl")
    if not events:
        raise ForkSessionError("saved trace has no session endpoint")
    trace_bytes = (source.artifact_path / ".trace.jsonl").read_bytes()
    if trace_bytes and not trace_bytes.endswith(b"\n"):
        raise ForkSessionError("saved trace ends with a partial JSON line")

    _validate_json_evidence(source.artifact_path)
    try:
        load_session_images(source.artifact_path)
        load_path_attachments(
            source.artifact_path, prompt_text=source.prompt_text
        )
        load_github_context(
            source.artifact_path, prompt_text=source.prompt_text
        )
        load_review_target(
            source.artifact_path, prompt_text=source.prompt_text
        )
        clarification = clarification_state(source.artifact_path)
        correction = validate_correction_trace(source.artifact_path)
        validate_correction_owner(source, correction)
    except (
        ImageInputError,
        GitHubContextError,
        PathAttachmentError,
        ReviewTargetError,
        ClarificationStateError,
        CorrectionStateError,
    ) as exc:
        raise ForkSessionError(str(exc)) from exc
    if clarification.phase in {"input_required", "input_ready"}:
        raise ForkSessionError("cannot fork a session with a pending clarification")
    if correction.phase == "pending":
        raise ForkSessionError("cannot fork a session with a pending correction")

    from .runner import derive_live_state

    live = derive_live_state(source.artifact_path)
    if source.status == "running" or live.status == "running":
        raise ForkSessionError("cannot fork a running session")
    if live.status in {"input_required", "input_ready", "approval_pending"}:
        raise ForkSessionError(f"cannot fork a session with unresolved {live.status}")
    if live.session_number < 1:
        raise ForkSessionError("saved trace has no stopped run boundary")
    return live.session_number


def _validate_json_evidence(artifact_dir: Path) -> None:
    request_path = artifact_dir / "approval_request.json"
    if request_path.exists():
        request = _load_json_object(request_path, "approval request")
        status_value = request.get("status")
        if status_value == "pending":
            raise ForkSessionError("cannot fork a session with a pending approval")
        if status_value not in {"approved", "rejected", "rewound"}:
            raise ForkSessionError("saved approval request has an invalid status")
    decisions_path = artifact_dir / "approval_decisions.json"
    if decisions_path.exists():
        decisions = _load_json_object(decisions_path, "approval decisions")
        if any(value not in {"approved", "rejected"} for value in decisions.values()):
            raise ForkSessionError("saved approval decisions are malformed")
    interrupt_path = artifact_dir / "shell_interrupt.json"
    if interrupt_path.exists():
        _load_json_object(interrupt_path, "interrupt evidence")


def _validate_recorded_config_hashes(
    source: SessionRecord,
    metadata: dict,
) -> None:
    hashes = metadata.get("config_path_hashes")
    if not isinstance(hashes, dict):
        raise ForkSessionError("session metadata config hashes are malformed")
    for raw_path in source.config_paths:
        path = Path(raw_path)
        if not path.is_file() or path.is_symlink():
            raise ForkSessionError(f"saved config path is not a regular file: {path}")
        expected = hashes.get(raw_path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected != actual:
            raise ForkSessionError(f"saved config hash does not match: {path}")


def _validate_system_prompt_path(
    source: SessionRecord,
    entries: dict[str, _ArtifactEntry],
) -> None:
    if source.system_prompt_path is None:
        return
    path = Path(source.system_prompt_path)
    if path.is_symlink() or not path.is_file():
        raise ForkSessionError(
            f"saved system prompt path is not a regular file: {path}"
        )
    try:
        relative = path.relative_to(source.artifact_path).as_posix()
    except ValueError:
        return
    entry = entries.get(relative)
    if entry is None or entry.kind != "file":
        raise ForkSessionError(
            "saved system prompt is outside the copied artifact evidence"
        )


def _validate_source_worktree(
    source: SessionRecord,
) -> WorktreeRuntimeInfo | None:
    identity = (
        source.worktree_path,
        source.worktree_branch,
        source.worktree_base_commit,
    )
    if any(identity) and not all(identity):
        raise ForkSessionError("saved session has incomplete worktree identity")
    if not all(identity):
        return None
    try:
        inspected = inspect_session_worktree(Path(source.cwd), source.session_id)
    except WorktreeRuntimeError as exc:
        raise ForkSessionError(str(exc)) from exc
    expected = (
        Path(str(source.worktree_path)).resolve(),
        str(source.worktree_branch),
        str(source.worktree_base_commit),
    )
    actual = (
        inspected.worktree_path.resolve(),
        inspected.branch,
        inspected.base_commit,
    )
    if actual != expected:
        raise ForkSessionError(
            "saved worktree identity does not match the owned Git worktree"
        )
    return inspected


def _rebase_paths(
    paths: list[str],
    *,
    source_root: Path,
    child_root: Path,
) -> list[Path]:
    rebased: list[Path] = []
    for raw in paths:
        path = Path(raw)
        try:
            relative = path.relative_to(source_root)
        except ValueError:
            rebased.append(path)
        else:
            rebased.append(child_root / relative)
    return rebased


def _rebase_optional_path(
    raw: str | None,
    *,
    source_root: Path,
    child_root: Path,
) -> Path | None:
    if raw is None:
        return None
    return _rebase_paths(
        [raw], source_root=source_root, child_root=child_root
    )[0]


def _load_json_object(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForkSessionError(f"saved {label} is malformed") from exc
    if not isinstance(payload, dict):
        raise ForkSessionError(f"saved {label} is malformed")
    return payload


def _load_trace_objects(path: Path) -> list[dict]:
    events: list[dict] = []
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ForkSessionError("saved trace is unreadable") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ForkSessionError(
                f"saved trace line {line_number} is malformed"
            ) from exc
        if not isinstance(event, dict):
            raise ForkSessionError(
                f"saved trace line {line_number} is not an object"
            )
        events.append(event)
    return events


def _snapshot_artifact_tree(root: Path) -> _ArtifactSnapshot:
    entries: list[_ArtifactEntry] = []

    def visit(directory: Path) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise ForkSessionError(
                f"cannot inspect saved artifact directory: {directory}"
            ) from exc
        for child in children:
            path = Path(child.path)
            relative = path.relative_to(root).as_posix()
            try:
                metadata = os.lstat(path)
            except OSError as exc:
                raise ForkSessionError(
                    f"cannot inspect saved artifact path: {relative}"
                ) from exc
            mode = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISLNK(metadata.st_mode):
                raise ForkSessionError(
                    f"saved artifact cannot be a symbolic link: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                entries.append(_ArtifactEntry(
                    relative=relative,
                    kind="dir",
                    mode=mode,
                    digest=hashlib.sha256(b"dir\0").hexdigest(),
                    size=0,
                ))
                visit(path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ForkSessionError(
                    f"saved artifact is not a regular file: {relative}"
                )
            digest, size = _hash_regular_file(path, metadata)
            entries.append(_ArtifactEntry(
                relative=relative,
                kind="file",
                mode=mode,
                digest=digest,
                size=size,
            ))

    visit(root)
    entries.sort(key=lambda entry: entry.relative)
    mapping = {
        entry.relative: (entry.digest, entry.mode)
        for entry in entries
    }
    digest = hashlib.sha256(
        json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return _ArtifactSnapshot(tuple(entries), digest)


def _hash_regular_file(path: Path, expected: os.stat_result) -> tuple[str, int]:
    descriptor = _open_source_file(path, expected)
    digest = hashlib.sha256(b"file\0")
    size = 0
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        _verify_open_file_unchanged(path, descriptor, expected)
    finally:
        os.close(descriptor)
    return digest.hexdigest(), size


def _open_source_file(path: Path, expected: os.stat_result) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ForkSessionError(f"cannot read saved artifact: {path}") from exc
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_dev != expected.st_dev
        or opened.st_ino != expected.st_ino
        or opened.st_size != expected.st_size
    ):
        os.close(descriptor)
        raise ForkSessionError(f"saved artifact changed while opening: {path}")
    return descriptor


def _verify_open_file_unchanged(
    path: Path,
    descriptor: int,
    expected: os.stat_result,
) -> None:
    after = os.fstat(descriptor)
    if (
        after.st_dev != expected.st_dev
        or after.st_ino != expected.st_ino
        or after.st_size != expected.st_size
        or after.st_mtime_ns != expected.st_mtime_ns
    ):
        raise ForkSessionError(f"saved artifact changed while reading: {path}")


def _copy_artifact_snapshot(
    source: Path,
    destination: Path,
    snapshot: _ArtifactSnapshot,
) -> None:
    directories = [entry for entry in snapshot.entries if entry.kind == "dir"]
    directories.sort(key=lambda entry: (len(Path(entry.relative).parts), entry.relative))
    for entry in directories:
        (destination / entry.relative).mkdir(mode=0o700)
    for entry in snapshot.entries:
        if entry.kind != "file":
            continue
        source_path = source / entry.relative
        metadata = os.lstat(source_path)
        source_fd = _open_source_file(source_path, metadata)
        target_path = destination / entry.relative
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        target_fd = os.open(target_path, flags, entry.mode or 0o600)
        digest = hashlib.sha256(b"file\0")
        size = 0
        try:
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                _write_all(target_fd, chunk)
            _verify_open_file_unchanged(source_path, source_fd, metadata)
            if digest.hexdigest() != entry.digest or size != entry.size:
                raise ForkSessionError(
                    f"saved artifact changed while copying: {entry.relative}"
                )
            os.fchmod(target_fd, entry.mode)
            os.fsync(target_fd)
            copied = os.fstat(target_fd)
            opened = os.fstat(source_fd)
            if copied.st_dev == opened.st_dev and copied.st_ino == opened.st_ino:
                raise ForkSessionError(
                    f"child artifact shares a source inode: {entry.relative}"
                )
        finally:
            os.close(target_fd)
            os.close(source_fd)
    for entry in reversed(directories):
        os.chmod(destination / entry.relative, entry.mode)


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise ForkSessionError("zero-byte child artifact write")
        view = view[written:]


def _cleanup_partial_child(
    store: SessionStore,
    source: SessionRecord,
    child: SessionRecord | None,
    *,
    stage: Path | None,
    final: Path | None,
    stage_owned: bool,
    final_owned: bool,
    worktree_created: bool,
) -> list[str]:
    errors: list[str] = []
    if child is not None:
        try:
            store.discard_unreturned_fork(child)
        except Exception as exc:
            errors.append(
                f"child session {child.session_id} remains published: {exc}"
            )
            return errors
    owned_paths = (
        stage if stage_owned else None,
        final if final_owned else None,
    )
    for path in owned_paths:
        if path is None:
            continue
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        except OSError as exc:
            errors.append(f"could not remove {path}: {exc}")
    if worktree_created and child is not None:
        try:
            remove_session_worktree(
                Path(source.cwd), child.session_id, force=True
            )
        except Exception as exc:
            errors.append(f"could not remove child worktree: {exc}")
    return errors


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "ForkResult",
    "ForkSessionError",
    "fork_saved_session",
    "validate_correction_owner",
]
