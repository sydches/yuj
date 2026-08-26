"""SQLite-backed session store for assistant-mode runs."""
from __future__ import annotations

import json
import os
import re
import socket
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..llm_solver._shared.paths import project_root, resource_origin


_SCHEMA = """
create table if not exists sessions (
    session_id text primary key,
    created_at text not null,
    updated_at text not null,
    cwd text not null,
    artifact_dir text not null,
    model text not null,
    status text not null,
    last_finish_reason text,
    prompt_text text not null,
    prompt_source text not null,
    context_mode text not null,
    system_prompt_path text,
    config_paths_json text not null,
    worktree_path text,
    worktree_branch text,
    worktree_base_commit text,
    provider text,
    auth_method text,
    credential_id text,
    parent_session_id text,
    label text,
    archived_at text
);

create table if not exists active_sessions (
    cwd text primary key,
    session_id text not null,
    updated_at text not null
);

create table if not exists session_locks (
    session_id text primary key,
    owner_host text not null,
    owner_pid integer not null,
    acquired_at text not null
);

create table if not exists session_purges (
    session_id text primary key,
    created_at text not null,
    updated_at text not null,
    phase text not null,
    manifest_json text not null,
    manifest_digest text not null,
    entry_count integer not null,
    estimated_bytes integer not null,
    root_dev integer not null,
    root_ino integer not null,
    failure_detail text,
    completed_at text
)
"""

SESSION_LABEL_MAX_LENGTH = 64
_SESSION_LABEL_PATTERN = re.compile(
    rf"[A-Za-z][A-Za-z0-9._-]{{0,{SESSION_LABEL_MAX_LENGTH - 1}}}\Z",
    re.ASCII,
)
_FULL_SESSION_ID_PATTERN = re.compile(
    r"[0-9]{8}_[0-9]{6}_[0-9a-f]{8}\Z",
    re.IGNORECASE | re.ASCII,
)
_SHORT_SESSION_ID_PATTERN = re.compile(
    r"[0-9a-f]{8}\Z",
    re.IGNORECASE | re.ASCII,
)
_RESERVED_SESSION_LABELS = frozenset({"last", "latest"})
_SESSION_LABEL_INDEX = "sessions_label_unique"
_PARENT_ID_TRIGGER = "sessions_parent_session_id_immutable"


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    created_at: str
    updated_at: str
    cwd: str
    artifact_dir: str
    model: str
    status: str
    last_finish_reason: str | None
    prompt_text: str
    prompt_source: str
    context_mode: str
    system_prompt_path: str | None
    config_paths_json: str
    worktree_path: str | None = None
    worktree_branch: str | None = None
    worktree_base_commit: str | None = None
    provider: str | None = None
    auth_method: str | None = None
    credential_id: str | None = None
    parent_session_id: str | None = None
    label: str | None = None
    archived_at: str | None = None

    @property
    def artifact_path(self) -> Path:
        return Path(self.artifact_dir)

    @property
    def config_paths(self) -> list[str]:
        return list(json.loads(self.config_paths_json))

    @property
    def short_id(self) -> str:
        return self.session_id.rsplit("_", 1)[-1]


@dataclass(frozen=True)
class SessionLock:
    session_id: str
    owner_host: str
    owner_pid: int
    acquired_at: str


@dataclass(frozen=True)
class SessionPurgeJournal:
    session_id: str
    created_at: str
    updated_at: str
    phase: str
    manifest_json: str
    manifest_digest: str
    entry_count: int
    estimated_bytes: int
    root_dev: int
    root_ino: int
    failure_detail: str | None
    completed_at: str | None


class SessionLockedError(RuntimeError):
    def __init__(self, lock: SessionLock):
        self.lock = lock
        super().__init__(
            "session is locked by "
            f"pid {lock.owner_pid} on {lock.owner_host} since {lock.acquired_at}"
        )


class AmbiguousSessionRefError(RuntimeError):
    pass


class SessionLabelError(ValueError):
    pass


class SessionArchiveError(RuntimeError):
    pass


class SessionPurgeInProgressError(RuntimeError):
    pass


class SessionPurgeStateError(RuntimeError):
    pass


def validate_session_label(label: str) -> str:
    """Validate one exact, case-sensitive operator label without rewriting it."""
    if not label:
        raise SessionLabelError("session label must not be empty")
    if len(label) > SESSION_LABEL_MAX_LENGTH:
        raise SessionLabelError(
            f"session label must be at most {SESSION_LABEL_MAX_LENGTH} characters"
        )
    if (
        _FULL_SESSION_ID_PATTERN.fullmatch(label)
        or _SHORT_SESSION_ID_PATTERN.fullmatch(label)
    ):
        raise SessionLabelError("session label must not look like a session ID")
    if label.lower() in _RESERVED_SESSION_LABELS:
        raise SessionLabelError(f"session label {label!r} is reserved")
    if _SESSION_LABEL_PATTERN.fullmatch(label) is None:
        raise SessionLabelError(
            "session label must start with an ASCII letter and contain only "
            "ASCII letters, digits, '.', '_', or '-'"
        )
    return label


def assist_home() -> Path:
    """Return the assistant-state root."""
    raw = os.environ.get("HARNESS_ASSIST_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    if resource_origin() == "installed-package":
        state_home = os.environ.get("XDG_STATE_HOME")
        base = (
            Path(state_home).expanduser()
            if state_home
            else Path.home() / ".local" / "state"
        )
        return (base / "yuj").resolve()
    return project_root() / ".llm_assist"


class SessionStore:
    """Persistent assistant session metadata."""

    def __init__(self, root: Path | None = None, *, read_only: bool = False):
        self.root = Path(root) if root is not None else assist_home()
        self.read_only = read_only
        self.db_path = self.root / "sessions.sqlite3"
        if read_only:
            if not self.db_path.is_file():
                raise FileNotFoundError(f"session store does not exist: {self.db_path}")
            with self._connect() as conn:
                self._has_archive_column = any(
                    str(row["name"]) == "archived_at"
                    for row in conn.execute("pragma table_info(sessions)").fetchall()
                )
                self._has_purge_table = conn.execute(
                    """
                    select 1 from sqlite_master
                    where type = 'table' and name = 'session_purges'
                    """
                ).fetchone() is not None
            return
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "sessions").mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("begin immediate")
            for statement in _SCHEMA.split(";"):
                if statement.strip():
                    conn.execute(statement)
            columns = {
                str(row["name"])
                for row in conn.execute("pragma table_info(sessions)").fetchall()
            }
            for name in (
                "worktree_path",
                "worktree_branch",
                "worktree_base_commit",
                "provider",
                "auth_method",
                "credential_id",
                "parent_session_id",
                "label",
                "archived_at",
            ):
                if name not in columns:
                    conn.execute(f"alter table sessions add column {name} text")
            conn.execute(
                f"""
                create unique index if not exists {_SESSION_LABEL_INDEX}
                on sessions(label) where label is not null
                """
            )
            conn.execute(
                f"""
                create trigger if not exists {_PARENT_ID_TRIGGER}
                before update of parent_session_id on sessions
                when new.parent_session_id is not old.parent_session_id
                begin
                    select raise(abort, 'parent_session_id is immutable');
                end
                """
            )
        self._has_archive_column = True
        self._has_purge_table = True

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            uri = self.db_path.resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            conn.execute("pragma query_only = on")
        else:
            conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_session(
        self,
        *,
        cwd: Path,
        model: str,
        prompt_text: str,
        prompt_source: str,
        context_mode: str,
        system_prompt_path: Path | None,
        config_paths: list[Path],
        provider: str | None = None,
        auth_method: str | None = None,
        credential_id: str | None = None,
    ) -> SessionRecord:
        now = _utc_now()
        session_id = _new_session_id()
        artifact_dir = self.root / "sessions" / session_id
        record = SessionRecord(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            cwd=str(Path(cwd).resolve()),
            artifact_dir=str(artifact_dir),
            model=model,
            status="created",
            last_finish_reason=None,
            prompt_text=prompt_text,
            prompt_source=prompt_source,
            context_mode=context_mode,
            system_prompt_path=str(system_prompt_path.resolve()) if system_prompt_path else None,
            config_paths_json=json.dumps([str(Path(p).resolve()) for p in config_paths]),
            provider=provider,
            auth_method=auth_method,
            credential_id=credential_id,
        )
        with self._connect() as conn:
            conn.execute(
                """
                insert into sessions (
                    session_id, created_at, updated_at, cwd, artifact_dir, model, status,
                    last_finish_reason, prompt_text, prompt_source, context_mode,
                    system_prompt_path, config_paths_json, provider, auth_method,
                    credential_id, parent_session_id
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    record.created_at,
                    record.updated_at,
                    record.cwd,
                    record.artifact_dir,
                    record.model,
                    record.status,
                    record.last_finish_reason,
                    record.prompt_text,
                    record.prompt_source,
                    record.context_mode,
                    record.system_prompt_path,
                    record.config_paths_json,
                    record.provider,
                    record.auth_method,
                    record.credential_id,
                    record.parent_session_id,
                ),
            )
        return record

    def prepare_forked_session(
        self,
        source: SessionRecord,
        *,
        config_paths: list[Path],
        system_prompt_path: Path | None,
        worktree_path: Path | None = None,
        worktree_branch: str | None = None,
        worktree_base_commit: str | None = None,
    ) -> SessionRecord:
        """Build, but do not publish, one child identity for a saved source."""
        now = _utc_now()
        session_id = _new_session_id()
        artifact_dir = self.root / "sessions" / session_id
        return SessionRecord(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            cwd=source.cwd,
            artifact_dir=str(artifact_dir),
            model=source.model,
            status="paused",
            last_finish_reason="forked",
            prompt_text=source.prompt_text,
            prompt_source=source.prompt_source,
            context_mode=source.context_mode,
            system_prompt_path=(
                str(Path(system_prompt_path).resolve())
                if system_prompt_path is not None
                else None
            ),
            config_paths_json=json.dumps(
                [str(Path(path).resolve()) for path in config_paths]
            ),
            worktree_path=(
                str(Path(worktree_path).resolve())
                if worktree_path is not None
                else None
            ),
            worktree_branch=worktree_branch,
            worktree_base_commit=worktree_base_commit,
            provider=source.provider,
            auth_method=source.auth_method,
            credential_id=source.credential_id,
            parent_session_id=source.session_id,
        )

    def insert_forked_session(
        self,
        record: SessionRecord,
        *,
        expected_parent: SessionRecord,
    ) -> None:
        """Atomically publish one fully staged child after rechecking its parent."""
        if record.parent_session_id != expected_parent.session_id:
            raise RuntimeError("fork child does not name its selected parent")
        expected_artifact = self.root / "sessions" / record.session_id
        if Path(record.artifact_dir) != expected_artifact:
            raise RuntimeError("fork child artifact path is outside the session store")
        with self._connect() as conn:
            conn.execute("begin immediate")
            parent_row = conn.execute(
                "select * from sessions where session_id = ?",
                (expected_parent.session_id,),
            ).fetchone()
            if parent_row is None:
                raise RuntimeError("fork source disappeared before publication")
            if _row_to_record(parent_row) != expected_parent:
                raise RuntimeError("fork source metadata changed during the copy")
            if expected_parent.archived_at is not None:
                raise RuntimeError("fork source became archived during the copy")
            active = conn.execute(
                "select 1 from active_sessions where session_id = ? limit 1",
                (expected_parent.session_id,),
            ).fetchone()
            if active is not None:
                raise RuntimeError("fork source became active during the copy")
            lock_row = conn.execute(
                "select * from session_locks where session_id = ?",
                (expected_parent.session_id,),
            ).fetchone()
            if lock_row is None or not _is_same_owner(
                _row_to_lock(lock_row), socket.gethostname(), os.getpid()
            ):
                raise RuntimeError("fork source lock changed during the copy")
            conn.execute(
                """
                insert into sessions (
                    session_id, created_at, updated_at, cwd, artifact_dir,
                    model, status, last_finish_reason, prompt_text,
                    prompt_source, context_mode, system_prompt_path,
                    config_paths_json, worktree_path, worktree_branch,
                    worktree_base_commit, provider, auth_method, credential_id,
                    parent_session_id, label, archived_at
                ) values (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?
                )
                """,
                (
                    record.session_id,
                    record.created_at,
                    record.updated_at,
                    record.cwd,
                    record.artifact_dir,
                    record.model,
                    record.status,
                    record.last_finish_reason,
                    record.prompt_text,
                    record.prompt_source,
                    record.context_mode,
                    record.system_prompt_path,
                    record.config_paths_json,
                    record.worktree_path,
                    record.worktree_branch,
                    record.worktree_base_commit,
                    record.provider,
                    record.auth_method,
                    record.credential_id,
                    record.parent_session_id,
                    None,
                    None,
                ),
            )

    def discard_unreturned_fork(self, record: SessionRecord) -> None:
        """Remove a child row when its creating command has not returned."""
        with self._connect() as conn:
            conn.execute("begin immediate")
            row = conn.execute(
                "select * from sessions where session_id = ?",
                (record.session_id,),
            ).fetchone()
            if row is None:
                return
            if _row_to_record(row) != record:
                raise RuntimeError(
                    "cannot clean up a fork child whose metadata changed"
                )
            active = conn.execute(
                "select 1 from active_sessions where session_id = ? limit 1",
                (record.session_id,),
            ).fetchone()
            lock = conn.execute(
                "select 1 from session_locks where session_id = ? limit 1",
                (record.session_id,),
            ).fetchone()
            if active is not None or lock is not None:
                raise RuntimeError(
                    "cannot clean up a fork child that another command claimed"
                )
            conn.execute(
                "delete from sessions where session_id = ?",
                (record.session_id,),
            )

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from sessions where session_id = ?",
                (session_id,),
            ).fetchone()
        return _row_to_record(row) if row is not None else None

    def list_sessions(
        self,
        *,
        limit: int = 50,
        archived: bool = False,
    ) -> list[SessionRecord]:
        if not self._has_archive_column and archived:
            return []
        if not self._has_archive_column:
            archive_filter = ""
        elif archived:
            archive_filter = "where archived_at is not null"
        else:
            archive_filter = "where archived_at is null"
        purge_filter = (
            "and session_id not in ("
            "select session_id from session_purges"
            ")"
            if self._has_purge_table and archive_filter
            else (
                "where session_id not in ("
                "select session_id from session_purges"
                ")"
                if self._has_purge_table
                else ""
            )
        )
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                select * from sessions
                {archive_filter}
                {purge_filter}
                order by updated_at desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def resolve_session_ref(self, session_ref: str) -> SessionRecord | None:
        with self._connect() as conn:
            rows = conn.execute("select * from sessions").fetchall()
        records = [_row_to_record(row) for row in rows]
        label_matches = [
            record for record in records if record.label == session_ref
        ]
        id_matches = [
            record
            for record in records
            if _session_id_ref_matches(record.session_id, session_ref)
        ]
        matches = {
            record.session_id: record for record in (*label_matches, *id_matches)
        }
        if len(matches) == 1:
            record = next(iter(matches.values()))
            journal = self.get_session_purge(record.session_id)
            if journal is not None:
                if journal.phase == "completed":
                    raise SessionPurgeInProgressError(
                        "session purge state is inconsistent; a completed "
                        "journal still has a session row"
                    )
                raise SessionPurgeInProgressError(
                    "session purge is incomplete; inspect it with "
                    f"yuj purge {record.session_id} --preview"
                )
            return record
        if len(matches) > 1:
            if label_matches:
                raise AmbiguousSessionRefError(
                    f"session ref '{session_ref}' is ambiguous between an exact "
                    "label and a session ID prefix; use the full session ID"
                )
            raise AmbiguousSessionRefError(
                f"session ref '{session_ref}' matches multiple sessions; use a longer prefix or the full id"
            )
        return None

    def get_session_purge(
        self,
        session_id: str,
    ) -> SessionPurgeJournal | None:
        if not self._has_purge_table:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "select * from session_purges where session_id = ?",
                (session_id,),
            ).fetchone()
        return _row_to_purge_journal(row) if row is not None else None

    def prepare_session_purge(
        self,
        record: SessionRecord,
        *,
        manifest_json: str,
        manifest_digest: str,
        entry_count: int,
        estimated_bytes: int,
        root_dev: int,
        root_ino: int,
    ) -> SessionPurgeJournal:
        """Publish the durable pre-deletion journal after atomic rechecks."""
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("begin immediate")
            existing = conn.execute(
                "select * from session_purges where session_id = ?",
                (record.session_id,),
            ).fetchone()
            if existing is not None:
                journal = _row_to_purge_journal(existing)
                if journal.phase == "completed":
                    raise SessionPurgeStateError(
                        "completed purge journal still has a session row"
                    )
                return journal
            row = conn.execute(
                "select * from sessions where session_id = ?",
                (record.session_id,),
            ).fetchone()
            if row is None or _row_to_record(row) != record:
                raise SessionPurgeStateError(
                    "session metadata changed before purge preparation"
                )
            if record.archived_at is None:
                raise SessionPurgeStateError("session must be archived before purge")
            if record.status == "running":
                raise SessionPurgeStateError("cannot purge a running session")
            active = conn.execute(
                "select 1 from active_sessions where session_id = ? limit 1",
                (record.session_id,),
            ).fetchone()
            if active is not None:
                raise SessionPurgeStateError(
                    "cannot purge a session with an active-session pointer"
                )
            lock_row = conn.execute(
                "select * from session_locks where session_id = ?",
                (record.session_id,),
            ).fetchone()
            if lock_row is None or not _is_same_owner(
                _row_to_lock(lock_row), socket.gethostname(), os.getpid()
            ):
                raise SessionPurgeStateError(
                    "purge no longer owns the selected session lock"
                )
            conn.execute(
                """
                insert into session_purges (
                    session_id, created_at, updated_at, phase, manifest_json,
                    manifest_digest, entry_count, estimated_bytes, root_dev,
                    root_ino, failure_detail, completed_at
                ) values (?, ?, ?, 'prepared', ?, ?, ?, ?, ?, ?, null, null)
                """,
                (
                    record.session_id,
                    now,
                    now,
                    manifest_json,
                    manifest_digest,
                    entry_count,
                    estimated_bytes,
                    root_dev,
                    root_ino,
                ),
            )
            created = conn.execute(
                "select * from session_purges where session_id = ?",
                (record.session_id,),
            ).fetchone()
        assert created is not None
        return _row_to_purge_journal(created)

    def transition_session_purge(
        self,
        session_id: str,
        *,
        expected: set[str],
        phase: str,
    ) -> SessionPurgeJournal:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("begin immediate")
            row = conn.execute(
                "select * from session_purges where session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise SessionPurgeStateError("purge journal is missing")
            current = _row_to_purge_journal(row)
            if current.phase not in expected:
                raise SessionPurgeStateError(
                    f"purge phase is {current.phase}, expected "
                    + ", ".join(sorted(expected))
                )
            conn.execute(
                """
                update session_purges
                set phase = ?, updated_at = ?, failure_detail = null
                where session_id = ?
                """,
                (phase, now, session_id),
            )
            updated = conn.execute(
                "select * from session_purges where session_id = ?",
                (session_id,),
            ).fetchone()
        assert updated is not None
        return _row_to_purge_journal(updated)

    def record_session_purge_failure(
        self,
        session_id: str,
        detail: str,
    ) -> None:
        if not self._has_purge_table:
            return
        with self._connect() as conn:
            conn.execute(
                """
                update session_purges
                set updated_at = ?, failure_detail = ?
                where session_id = ? and phase != 'completed'
                """,
                (_utc_now(), detail, session_id),
            )

    def finalize_session_purge(
        self,
        expected_record: SessionRecord,
    ) -> SessionPurgeJournal:
        """Remove the indexed identity and complete its journal atomically."""
        session_id = expected_record.session_id
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("begin immediate")
            journal_row = conn.execute(
                "select * from session_purges where session_id = ?",
                (session_id,),
            ).fetchone()
            if journal_row is None:
                raise SessionPurgeStateError("purge journal is missing")
            journal = _row_to_purge_journal(journal_row)
            if journal.phase == "completed":
                row = conn.execute(
                    "select 1 from sessions where session_id = ?",
                    (session_id,),
                ).fetchone()
                if row is not None:
                    raise SessionPurgeStateError(
                        "completed purge journal still has a session row"
                    )
                return journal
            if journal.phase != "artifacts_removed":
                raise SessionPurgeStateError(
                    "cannot remove a session row before its artifacts"
                )
            row = conn.execute(
                "select * from sessions where session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise SessionPurgeStateError(
                    "session row disappeared before purge finalization"
                )
            record = _row_to_record(row)
            if record != expected_record:
                raise SessionPurgeStateError(
                    "session metadata changed before purge finalization"
                )
            if record.archived_at is None:
                raise SessionPurgeStateError(
                    "session became unarchived during purge"
                )
            if record.status == "running":
                raise SessionPurgeStateError(
                    "session became running during purge"
                )
            active = conn.execute(
                "select 1 from active_sessions where session_id = ? limit 1",
                (session_id,),
            ).fetchone()
            if active is not None:
                raise SessionPurgeStateError(
                    "session gained an active-session pointer during purge"
                )
            lock_row = conn.execute(
                "select * from session_locks where session_id = ?",
                (session_id,),
            ).fetchone()
            if lock_row is None or not _is_same_owner(
                _row_to_lock(lock_row), socket.gethostname(), os.getpid()
            ):
                raise SessionPurgeStateError(
                    "purge no longer owns the selected session lock"
                )
            conn.execute(
                "delete from sessions where session_id = ?",
                (session_id,),
            )
            conn.execute(
                "delete from session_locks where session_id = ?",
                (session_id,),
            )
            conn.execute(
                """
                update session_purges
                set phase = 'completed', updated_at = ?, manifest_json = '[]',
                    failure_detail = null, completed_at = ?
                where session_id = ?
                """,
                (now, now, session_id),
            )
            completed = conn.execute(
                "select * from session_purges where session_id = ?",
                (session_id,),
            ).fetchone()
        assert completed is not None
        return _row_to_purge_journal(completed)

    def set_session_label(self, session_id: str, label: str) -> None:
        exact_label = validate_session_label(label)
        try:
            with self._connect() as conn:
                conn.execute("begin immediate")
                rows = conn.execute(
                    "select session_id from sessions"
                ).fetchall()
                session_ids = [str(row["session_id"]) for row in rows]
                if session_id not in session_ids:
                    raise SessionLabelError(f"unknown session: {session_id}")
                if any(
                    _session_id_ref_matches(candidate, exact_label)
                    for candidate in session_ids
                ):
                    raise SessionLabelError(
                        f"session label {exact_label!r} conflicts with an "
                        "existing session ID selector"
                    )
                conn.execute(
                    "update sessions set label = ? where session_id = ?",
                    (exact_label, session_id),
                )
        except sqlite3.IntegrityError as exc:
            raise SessionLabelError(
                f"session label {exact_label!r} is already assigned to another session"
            ) from exc

    def clear_session_label(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("begin immediate")
            cursor = conn.execute(
                "update sessions set label = null where session_id = ?",
                (session_id,),
            )
            if cursor.rowcount != 1:
                raise SessionLabelError(f"unknown session: {session_id}")

    def archive_session(self, session_id: str) -> tuple[SessionRecord, bool]:
        """Archive one inactive, unlocked session by changing only its timestamp."""
        with self._connect() as conn:
            conn.execute("begin immediate")
            row = conn.execute(
                "select * from sessions where session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise SessionArchiveError(f"unknown session: {session_id}")
            record = _row_to_record(row)
            if record.archived_at is not None:
                return record, False
            active = conn.execute(
                "select 1 from active_sessions where session_id = ? limit 1",
                (session_id,),
            ).fetchone()
            if active is not None:
                raise SessionArchiveError(
                    "cannot archive the active session; start another session first"
                )
            lock_row = conn.execute(
                "select * from session_locks where session_id = ?",
                (session_id,),
            ).fetchone()
            if lock_row is not None:
                lock = _row_to_lock(lock_row)
                if not _is_stale_lock(lock):
                    raise SessionArchiveError(
                        "cannot archive a locked session; "
                        f"pid {lock.owner_pid} on {lock.owner_host} holds the lock"
                    )
            archived_at = _utc_now()
            conn.execute(
                "update sessions set archived_at = ? where session_id = ?",
                (archived_at, session_id),
            )
            updated = conn.execute(
                "select * from sessions where session_id = ?",
                (session_id,),
            ).fetchone()
        assert updated is not None
        return _row_to_record(updated), True

    def unarchive_session(self, session_id: str) -> tuple[SessionRecord, bool]:
        """Restore one archived session without changing any other metadata."""
        with self._connect() as conn:
            conn.execute("begin immediate")
            row = conn.execute(
                "select * from sessions where session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise SessionArchiveError(f"unknown session: {session_id}")
            record = _row_to_record(row)
            if record.archived_at is None:
                return record, False
            conn.execute(
                "update sessions set archived_at = null where session_id = ?",
                (session_id,),
            )
            updated = conn.execute(
                "select * from sessions where session_id = ?",
                (session_id,),
            ).fetchone()
        assert updated is not None
        return _row_to_record(updated), True

    def set_active_session(self, cwd: Path | str, session_id: str) -> None:
        now = _utc_now()
        resolved_cwd = str(Path(cwd).resolve())
        with self._connect() as conn:
            conn.execute(
                """
                insert into active_sessions (cwd, session_id, updated_at)
                values (?, ?, ?)
                on conflict(cwd) do update set
                    session_id = excluded.session_id,
                    updated_at = excluded.updated_at
                """,
                (resolved_cwd, session_id, now),
            )

    def clear_active_session(self, cwd: Path | str, *, session_id: str | None = None) -> None:
        resolved_cwd = str(Path(cwd).resolve())
        with self._connect() as conn:
            if session_id is None:
                conn.execute(
                    "delete from active_sessions where cwd = ?",
                    (resolved_cwd,),
                )
                return
            conn.execute(
                "delete from active_sessions where cwd = ? and session_id = ?",
                (resolved_cwd, session_id),
            )

    def get_active_session_id(self, cwd: Path | str) -> str | None:
        resolved_cwd = str(Path(cwd).resolve())
        with self._connect() as conn:
            row = conn.execute(
                "select session_id from active_sessions where cwd = ?",
                (resolved_cwd,),
            ).fetchone()
        return str(row["session_id"]) if row is not None else None

    def get_active_session(self, cwd: Path | str) -> SessionRecord | None:
        session_id = self.get_active_session_id(cwd)
        if session_id is None:
            return None
        record = self.get_session(session_id)
        if record is not None and self.get_session_purge(record.session_id) is not None:
            return None
        if record is not None and record.archived_at is None:
            return record
        if record is not None:
            return None
        if self.read_only:
            return None
        self.clear_active_session(cwd, session_id=session_id)
        return None

    def list_active_session_ids(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("select session_id from active_sessions").fetchall()
        return {str(row["session_id"]) for row in rows}

    def acquire_session_lock(self, session_id: str) -> SessionLock:
        owner_host = socket.gethostname()
        owner_pid = os.getpid()
        acquired_at = _utc_now()
        with self._connect() as conn:
            conn.execute("begin immediate")
            row = conn.execute(
                "select * from session_locks where session_id = ?",
                (session_id,),
            ).fetchone()
            if row is not None:
                existing = _row_to_lock(row)
                if _is_same_owner(existing, owner_host, owner_pid) or _is_stale_lock(existing):
                    conn.execute(
                        "delete from session_locks where session_id = ?",
                        (session_id,),
                    )
                else:
                    raise SessionLockedError(existing)
            conn.execute(
                """
                insert into session_locks (
                    session_id, owner_host, owner_pid, acquired_at
                ) values (?, ?, ?, ?)
                """,
                (session_id, owner_host, owner_pid, acquired_at),
            )
        return SessionLock(
            session_id=session_id,
            owner_host=owner_host,
            owner_pid=owner_pid,
            acquired_at=acquired_at,
        )

    def release_session_lock(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                delete from session_locks
                where session_id = ? and owner_host = ? and owner_pid = ?
                """,
                (session_id, socket.gethostname(), os.getpid()),
            )

    def get_session_lock(self, session_id: str) -> SessionLock | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from session_locks where session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        lock = _row_to_lock(row)
        if _is_stale_lock(lock):
            if self.read_only:
                return None
            with self._connect() as conn:
                conn.execute(
                    "delete from session_locks where session_id = ?",
                    (session_id,),
                )
            return None
        return lock

    def list_locked_session_ids(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("select session_id from session_locks").fetchall()
        locks: set[str] = set()
        for row in rows:
            session_id = str(row["session_id"])
            if self.get_session_lock(session_id) is not None:
                locks.add(session_id)
        return locks

    def update_session(
        self,
        session_id: str,
        *,
        status: str,
        last_finish_reason: str | None,
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                update sessions
                set updated_at = ?, status = ?, last_finish_reason = ?
                where session_id = ?
                """,
                (now, status, last_finish_reason, session_id),
            )

    def update_session_config_paths(self, session_id: str, config_paths: list[Path]) -> None:
        now = _utc_now()
        config_paths_json = json.dumps([str(Path(p).resolve()) for p in config_paths])
        with self._connect() as conn:
            conn.execute(
                """
                update sessions
                set updated_at = ?, config_paths_json = ?
                where session_id = ?
                """,
                (now, config_paths_json, session_id),
            )

    def update_session_worktree(
        self,
        session_id: str,
        *,
        path: Path,
        branch: str,
        base_commit: str,
    ) -> None:
        """Persist the stable owned-worktree identity used by resume/cleanup."""
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                update sessions
                set updated_at = ?, worktree_path = ?, worktree_branch = ?,
                    worktree_base_commit = ?
                where session_id = ?
                """,
                (
                    now,
                    str(Path(path).resolve()),
                    branch,
                    base_commit,
                    session_id,
                ),
            )


def _row_to_record(row: sqlite3.Row) -> SessionRecord:
    optional = set(row.keys())
    return SessionRecord(
        session_id=row["session_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        cwd=row["cwd"],
        artifact_dir=row["artifact_dir"],
        model=row["model"],
        status=row["status"],
        last_finish_reason=row["last_finish_reason"],
        prompt_text=row["prompt_text"],
        prompt_source=row["prompt_source"],
        context_mode=row["context_mode"],
        system_prompt_path=row["system_prompt_path"],
        config_paths_json=row["config_paths_json"],
        worktree_path=row["worktree_path"] if "worktree_path" in optional else None,
        worktree_branch=(
            row["worktree_branch"] if "worktree_branch" in optional else None
        ),
        worktree_base_commit=(
            row["worktree_base_commit"]
            if "worktree_base_commit" in optional
            else None
        ),
        provider=row["provider"] if "provider" in optional else None,
        auth_method=row["auth_method"] if "auth_method" in optional else None,
        credential_id=(
            row["credential_id"] if "credential_id" in optional else None
        ),
        parent_session_id=(
            row["parent_session_id"]
            if "parent_session_id" in optional
            else None
        ),
        label=row["label"] if "label" in optional else None,
        archived_at=(
            row["archived_at"] if "archived_at" in optional else None
        ),
    )


def _row_to_lock(row: sqlite3.Row) -> SessionLock:
    return SessionLock(
        session_id=row["session_id"],
        owner_host=row["owner_host"],
        owner_pid=int(row["owner_pid"]),
        acquired_at=row["acquired_at"],
    )


def _row_to_purge_journal(row: sqlite3.Row) -> SessionPurgeJournal:
    return SessionPurgeJournal(
        session_id=str(row["session_id"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        phase=str(row["phase"]),
        manifest_json=str(row["manifest_json"]),
        manifest_digest=str(row["manifest_digest"]),
        entry_count=int(row["entry_count"]),
        estimated_bytes=int(row["estimated_bytes"]),
        root_dev=int(row["root_dev"]),
        root_ino=int(row["root_ino"]),
        failure_detail=(
            str(row["failure_detail"])
            if row["failure_detail"] is not None
            else None
        ),
        completed_at=(
            str(row["completed_at"])
            if row["completed_at"] is not None
            else None
        ),
    )


def _is_same_owner(lock: SessionLock, owner_host: str, owner_pid: int) -> bool:
    return lock.owner_host == owner_host and lock.owner_pid == owner_pid


def _is_stale_lock(lock: SessionLock) -> bool:
    if lock.owner_host != socket.gethostname():
        return False
    try:
        os.kill(lock.owner_pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _session_id_ref_matches(session_id: str, session_ref: str) -> bool:
    short_id = session_id.rsplit("_", 1)[-1]
    return session_id.startswith(session_ref) or short_id.startswith(session_ref)


def is_full_session_id(value: str) -> bool:
    return _FULL_SESSION_ID_PATTERN.fullmatch(value) is not None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_session_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


__all__ = [
    "SESSION_LABEL_MAX_LENGTH",
    "SessionLock",
    "SessionPurgeJournal",
    "AmbiguousSessionRefError",
    "SessionArchiveError",
    "SessionLabelError",
    "SessionLockedError",
    "SessionPurgeInProgressError",
    "SessionPurgeStateError",
    "SessionRecord",
    "SessionStore",
    "assist_home",
    "is_full_session_id",
    "validate_session_label",
]
