"""Create, inspect, and safely remove per-session Git worktrees."""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

WORKTREE_DIR_NAME = ".yuj_worktrees"
_LOCAL_EXCLUDE = f"/{WORKTREE_DIR_NAME}/"
_METADATA_NAME = "yuj-runtime.json"
_METADATA_VERSION = 1
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_GIT_LOCK = threading.RLock()


class WorktreeRuntimeError(RuntimeError):
    pass


class WorktreeExistsError(WorktreeRuntimeError):
    pass


class WorktreeDirtyError(WorktreeRuntimeError):
    pass


@dataclass(frozen=True)
class WorktreeRuntimeInfo:
    enabled: bool
    run_id: str
    source_cwd: Path
    repo_root: Path
    worktree_path: Path
    session_cwd: Path
    branch: str
    base_commit: str
    reused: bool = False

    def session_start_fields(self) -> dict[str, object]:
        return {
            "worktree_path": str(self.worktree_path),
            "worktree_branch": self.branch,
            "worktree_base_commit": self.base_commit,
        }


@dataclass(frozen=True)
class RemovedWorktree:
    run_id: str
    worktree_path: Path
    branch: str
    forced: bool


@dataclass(frozen=True)
class _RegisteredWorktree:
    path: Path
    head: str
    branch_ref: str

    @property
    def branch(self) -> str:
        prefix = "refs/heads/"
        return self.branch_ref[len(prefix):] if self.branch_ref.startswith(prefix) else ""


def _run(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
    )
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorktreeRuntimeError(f"{' '.join(args)} failed: {exc}") from exc
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise WorktreeRuntimeError(
            f"{' '.join(args)} exited {proc.returncode}: {detail}"
        )
    return proc


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], cwd=cwd, check=check)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("zero-byte write")
        view = view[written:]


def _repo_root(cwd: Path) -> Path:
    source = Path(cwd).resolve()
    proc = _git(source, "rev-parse", "--show-toplevel")
    return Path(proc.stdout.strip()).resolve()


def _validate_run_id(run_id: str) -> str:
    value = str(run_id or "")
    if not _RUN_ID_RE.fullmatch(value) or value in {".", ".."} or value.endswith(".lock"):
        raise ValueError(
            "worktree run_id must be a safe 1-128 character path component"
        )
    return value


def _validate_branch(repo_root: Path, branch: str) -> str:
    value = str(branch or "")
    proc = _git(repo_root, "check-ref-format", "--branch", value, check=False)
    if proc.returncode != 0:
        raise ValueError(f"invalid worktree branch name: {value!r}")
    return value


def _auto_branch(run_id: str) -> str:
    candidate = f"worktree-{run_id}"
    if len(candidate.encode()) <= 180:
        return candidate
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:20]
    return f"worktree-{digest}"


def _resolve_names(mode: str, run_id: str | None) -> tuple[str, str]:
    value = str(mode or "off").strip()
    if value == "off":
        return "", ""
    resolved_run_id = _validate_run_id(run_id or f"worktree-{secrets.token_hex(4)}")
    branch = _auto_branch(resolved_run_id) if value == "auto" else value
    return resolved_run_id, branch


def _git_path(repo_root: Path, name: str) -> Path:
    value = _git(repo_root, "rev-parse", "--git-path", name).stdout.strip()
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _ensure_local_exclude(repo_root: Path) -> None:
    exclude = _git_path(repo_root, "info/exclude")
    exclude.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude.read_bytes() if exclude.is_file() else b""
    lines = {line.strip() for line in existing.decode(errors="replace").splitlines()}
    if _LOCAL_EXCLUDE in lines:
        return
    fd = os.open(exclude, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if existing and not existing.endswith(b"\n"):
            _write_all(fd, b"\n")
        _write_all(fd, (_LOCAL_EXCLUDE + "\n").encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def _registered_worktrees(repo_root: Path) -> tuple[_RegisteredWorktree, ...]:
    text = _git(repo_root, "worktree", "list", "--porcelain").stdout
    entries: list[_RegisteredWorktree] = []
    fields: dict[str, str] = {}
    for line in [*text.splitlines(), ""]:
        if not line:
            if fields.get("worktree"):
                entries.append(
                    _RegisteredWorktree(
                        path=Path(fields["worktree"]).resolve(),
                        head=fields.get("HEAD", ""),
                        branch_ref=fields.get("branch", ""),
                    )
                )
            fields = {}
            continue
        key, _, value = line.partition(" ")
        fields[key] = value
    return tuple(entries)


def _registered_at(repo_root: Path, path: Path) -> _RegisteredWorktree | None:
    target = path.resolve()
    return next((item for item in _registered_worktrees(repo_root) if item.path == target), None)


def _metadata_path(worktree_path: Path) -> Path:
    raw = _git(worktree_path, "rev-parse", "--git-dir").stdout.strip()
    git_dir = Path(raw)
    if not git_dir.is_absolute():
        git_dir = worktree_path / git_dir
    return git_dir.resolve() / _METADATA_NAME


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".yuj-runtime-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        data = json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
        _write_all(fd, data)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temporary, path)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_metadata(worktree_path: Path) -> dict[str, object]:
    path = _metadata_path(worktree_path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise WorktreeRuntimeError(
            f"worktree is missing valid {_METADATA_NAME} ownership metadata"
        ) from exc
    if payload.get("schema_version") != _METADATA_VERSION:
        raise WorktreeRuntimeError("unsupported worktree runtime metadata version")
    return payload


def _build_info(
    *,
    repo_root: Path,
    path: Path,
    metadata: dict[str, object],
    reused: bool,
) -> WorktreeRuntimeInfo:
    relative = PurePosixPath(str(metadata.get("source_relative") or "."))
    if relative.is_absolute() or ".." in relative.parts:
        raise WorktreeRuntimeError("unsafe source_relative in worktree metadata")
    session_cwd = path.joinpath(*relative.parts).resolve()
    try:
        session_cwd.relative_to(path.resolve())
    except ValueError as exc:
        raise WorktreeRuntimeError("session cwd escapes worktree") from exc
    if not session_cwd.is_dir():
        raise WorktreeRuntimeError(f"session subdirectory is missing: {session_cwd}")
    source_cwd = repo_root.joinpath(*relative.parts).resolve()
    return WorktreeRuntimeInfo(
        enabled=True,
        run_id=str(metadata["run_id"]),
        source_cwd=source_cwd,
        repo_root=repo_root,
        worktree_path=path,
        session_cwd=session_cwd,
        branch=str(metadata["branch"]),
        base_commit=str(metadata["base_commit"]),
        reused=reused,
    )


def create_session_worktree(
    source_cwd: Path,
    *,
    mode: str,
    run_id: str | None = None,
    base_commit: str | None = None,
    reuse: bool = False,
    require_clean: bool = True,
) -> WorktreeRuntimeInfo | None:
    """Create or explicitly reuse the worktree selected by runtime settings."""
    if str(mode or "off").strip() == "off":
        return None
    source_cwd = Path(source_cwd).resolve()
    repo_root = _repo_root(source_cwd)
    resolved_run_id, branch = _resolve_names(mode, run_id)
    branch = _validate_branch(repo_root, branch)
    root = repo_root / WORKTREE_DIR_NAME
    path = root / resolved_run_id
    source_relative = source_cwd.relative_to(repo_root)

    with _GIT_LOCK:
        _ensure_local_exclude(repo_root)
        registered = _registered_at(repo_root, path)
        if registered is not None:
            if not reuse:
                raise WorktreeExistsError(f"worktree already exists for run {resolved_run_id}")
            metadata = _read_metadata(path)
            if metadata.get("run_id") != resolved_run_id or metadata.get("branch") != branch:
                raise WorktreeExistsError("existing worktree identity does not match request")
            return _build_info(
                repo_root=repo_root,
                path=path,
                metadata=metadata,
                reused=True,
            )
        if path.exists():
            raise WorktreeExistsError(f"unregistered worktree path already exists: {path}")
        branch_exists = _git(
            repo_root,
            "show-ref",
            "--verify",
            "--quiet",
            f"refs/heads/{branch}",
            check=False,
        ).returncode == 0
        if branch_exists:
            raise WorktreeExistsError(f"branch already exists: {branch}")
        if require_clean:
            dirty = _git(repo_root, "status", "--porcelain=v1").stdout.strip()
            if dirty:
                raise WorktreeDirtyError(
                    "source checkout has uncommitted changes; commit or stash them before isolation"
                )
        requested_base = base_commit or "HEAD"
        resolved_base = _git(
            repo_root, "rev-parse", "--verify", f"{requested_base}^{{commit}}"
        ).stdout.strip()
        root.mkdir(parents=True, exist_ok=True)
        _git(
            repo_root,
            "worktree",
            "add",
            "--no-track",
            "-b",
            branch,
            str(path),
            resolved_base,
        )
        metadata = {
            "schema_version": _METADATA_VERSION,
            "run_id": resolved_run_id,
            "branch": branch,
            "base_commit": resolved_base,
            "source_relative": source_relative.as_posix(),
        }
        _write_metadata(_metadata_path(path), metadata)
        return _build_info(
            repo_root=repo_root,
            path=path,
            metadata=metadata,
            reused=False,
        )


def inspect_session_worktree(repo_cwd: Path, run_id: str) -> WorktreeRuntimeInfo:
    """Inspect a Yuj-owned worktree without changing it."""
    repo_root = _repo_root(Path(repo_cwd))
    resolved_run_id = _validate_run_id(run_id)
    path = repo_root / WORKTREE_DIR_NAME / resolved_run_id
    registered = _registered_at(repo_root, path)
    if registered is None:
        raise WorktreeRuntimeError(f"no registered worktree for run {resolved_run_id}")
    metadata = _read_metadata(path)
    if metadata.get("run_id") != resolved_run_id or metadata.get("branch") != registered.branch:
        raise WorktreeRuntimeError("registered worktree does not match Yuj metadata")
    return _build_info(
        repo_root=repo_root,
        path=path,
        metadata=metadata,
        reused=True,
    )


def remove_session_worktree(
    repo_cwd: Path,
    run_id: str,
    *,
    force: bool = False,
) -> RemovedWorktree:
    """Remove one owned worktree and branch; refuse data loss by default."""
    repo_root = _repo_root(Path(repo_cwd))
    resolved_run_id = _validate_run_id(run_id)
    path = repo_root / WORKTREE_DIR_NAME / resolved_run_id
    with _GIT_LOCK:
        registered = _registered_at(repo_root, path)
        if registered is None or not registered.branch:
            raise WorktreeRuntimeError(f"no removable branch worktree for run {resolved_run_id}")
        metadata = _read_metadata(path)
        if metadata.get("run_id") != resolved_run_id or metadata.get("branch") != registered.branch:
            raise WorktreeRuntimeError("refusing to remove a worktree not owned by this runtime")
        dirty = _git(path, "status", "--porcelain=v1").stdout.strip()
        if dirty and not force:
            raise WorktreeDirtyError("worktree has uncommitted changes; use force to discard them")
        merged = _git(
            repo_root,
            "merge-base",
            "--is-ancestor",
            registered.branch,
            "HEAD",
            check=False,
        ).returncode == 0
        if not merged and not force:
            raise WorktreeDirtyError(
                "worktree branch has unmerged commits; merge it or use force"
            )
        remove_args = ["worktree", "remove"]
        if force:
            remove_args.append("--force")
        remove_args.append(str(path))
        _git(repo_root, *remove_args)
        _git(repo_root, "branch", "-D" if force else "-d", registered.branch)
        return RemovedWorktree(
            run_id=resolved_run_id,
            worktree_path=path,
            branch=registered.branch,
            forced=bool(force),
        )
