"""SQLite-backed session store for assistant-mode runs."""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..llm_solver._shared.paths import project_root


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
    config_paths_json text not null
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
)
"""


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


class SessionLockedError(RuntimeError):
    def __init__(self, lock: SessionLock):
        self.lock = lock
        super().__init__(
            "session is locked by "
            f"pid {lock.owner_pid} on {lock.owner_host} since {lock.acquired_at}"
        )


class AmbiguousSessionRefError(RuntimeError):
    pass


def assist_home() -> Path:
    """Return the assistant-state root."""
    raw = os.environ.get("HARNESS_ASSIST_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return project_root() / ".llm_assist"


class SessionStore:
    """Persistent assistant session metadata."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else assist_home()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "sessions").mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "sessions.sqlite3"
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
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
    ) -> SessionRecord:
        now = _utc_now()
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
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
        )
        with self._connect() as conn:
            conn.execute(
                """
                insert into sessions (
                    session_id, created_at, updated_at, cwd, artifact_dir, model, status,
                    last_finish_reason, prompt_text, prompt_source, context_mode,
                    system_prompt_path, config_paths_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
        return record

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from sessions where session_id = ?",
                (session_id,),
            ).fetchone()
        return _row_to_record(row) if row is not None else None

    def list_sessions(self, *, limit: int = 50) -> list[SessionRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                select * from sessions
                order by updated_at desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def resolve_session_ref(self, session_ref: str) -> SessionRecord | None:
        exact = self.get_session(session_ref)
        if exact is not None:
            return exact

        records = self.list_sessions(limit=1000)
        short_exact = [record for record in records if record.short_id == session_ref]
        if len(short_exact) == 1:
            return short_exact[0]
        if len(short_exact) > 1:
            raise AmbiguousSessionRefError(
                f"session ref '{session_ref}' matches multiple sessions; use a longer prefix or the full id"
            )

        prefix_matches = [
            record
            for record in records
            if record.session_id.startswith(session_ref) or record.short_id.startswith(session_ref)
        ]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        if len(prefix_matches) > 1:
            raise AmbiguousSessionRefError(
                f"session ref '{session_ref}' matches multiple sessions; use a longer prefix or the full id"
            )
        return None

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
        if record is not None:
            return record
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


def _row_to_record(row: sqlite3.Row) -> SessionRecord:
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
    )


def _row_to_lock(row: sqlite3.Row) -> SessionLock:
    return SessionLock(
        session_id=row["session_id"],
        owner_host=row["owner_host"],
        owner_pid=int(row["owner_pid"]),
        acquired_at=row["acquired_at"],
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "SessionLock",
    "AmbiguousSessionRefError",
    "SessionLockedError",
    "SessionRecord",
    "SessionStore",
    "assist_home",
]
