"""Isolated workspace handoff for write-capable named subagents."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .worktree_runtime import (
    WORKTREE_DIR_NAME,
    WorkspaceEntry,
    WorkspaceSnapshot,
    WorktreeRuntimeInfo,
    copy_workspace_to_worktree,
    remove_session_worktree,
    snapshot_workspace,
)


CHANGESET_FILE = "changeset.json"
PATCH_FILE = "changes.patch"
APPLICATION_FILE = "application.json"
CHANGESET_SCHEMA_VERSION = 1
MAX_CHANGESET_FILES = 64
MAX_CHANGESET_INPUT_BYTES = 256 * 1024
MAX_CHANGESET_PATCH_BYTES = 512 * 1024
MAX_REVIEW_LINES = 400
_TASK_ID_PREFIX = "task-"


class SubagentWorkspaceError(RuntimeError):
    """An isolated child cannot produce or apply a safe change set."""


@dataclass(frozen=True, slots=True)
class IsolatedWorkspace:
    task_id: str
    run_id: str
    info: WorktreeRuntimeInfo
    baseline: WorkspaceSnapshot
    baseline_tree: str


@dataclass(frozen=True, slots=True)
class SubagentChangeSet:
    task_id: str
    status: str
    file_count: int
    patch_bytes: int
    patch_sha256: str
    paths: tuple[str, ...]
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def result_suffix(self) -> str:
        head = f"Isolated change set {self.task_id}: {self.status}."
        if self.status == "ready":
            shown = json.dumps(list(self.paths), ensure_ascii=True)
            return (
                f"{head} Files: {shown}. Patch bytes: {self.patch_bytes}. "
                f"SHA-256: {self.patch_sha256}. Review it with "
                f'subagent_changes(task_id="{self.task_id}"). Apply it only '
                f'with apply_subagent(task_id="{self.task_id}").'
            )
        if self.status == "empty":
            return f"{head} The child did not change any files."
        detail = self.detail or "The change set cannot be applied."
        return f"{head} {detail}"


class AppliedSubagentResult(str):
    """String-compatible success result with exact applied operations."""

    def __new__(cls, text: str, operations: tuple[tuple[str, str], ...]):
        value = super().__new__(cls, text)
        value.applied_operations = operations
        return value


def _run_id(run_root: Path, task_id: str) -> str:
    digest = hashlib.sha256(
        f"{Path(run_root).resolve()}\0{task_id}".encode("utf-8")
    ).hexdigest()[:16]
    ordinal = task_id.removeprefix(_TASK_ID_PREFIX)
    return f"subagent-{ordinal}-{digest}"


def _git(
    cwd: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    env.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "LC_ALL": "C",
    })
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SubagentWorkspaceError(
            f"git {' '.join(args)} failed: {exc}"
        ) from exc
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).decode(
            "utf-8", errors="replace"
        ).strip()
        raise SubagentWorkspaceError(
            f"git {' '.join(args)} exited {result.returncode}: {detail}"
        )
    return result


def _repo_root(cwd: Path) -> Path:
    value = _git(cwd, "rev-parse", "--show-toplevel").stdout.decode().strip()
    return Path(value).resolve()


def _safe_task_id(value: object) -> str:
    text = str(value or "")
    if (
        not text.startswith(_TASK_ID_PREFIX)
        or len(text) != len(_TASK_ID_PREFIX) + 6
        or not text[len(_TASK_ID_PREFIX):].isdigit()
    ):
        raise SubagentWorkspaceError("task_id must use the task-NNNNNN form")
    return text


def prepare_isolated_workspace(
    parent_cwd: Path,
    *,
    run_root: Path,
    task_id: str,
) -> IsolatedWorkspace:
    """Create one exact child copy and pin its baseline Git tree."""
    task_id = _safe_task_id(task_id)
    run_id = _run_id(run_root, task_id)
    info: WorktreeRuntimeInfo | None = None
    try:
        info, baseline = copy_workspace_to_worktree(
            Path(parent_cwd), child_run_id=run_id
        )
        _git(info.worktree_path, "add", "-A", "--", ".")
        baseline_tree = _git(info.worktree_path, "write-tree").stdout.decode().strip()
        _git(info.worktree_path, "read-tree", "HEAD")
        if snapshot_workspace(info.worktree_path) != baseline:
            raise SubagentWorkspaceError(
                "isolated workspace changed while its baseline was pinned"
            )
        return IsolatedWorkspace(
            task_id=task_id,
            run_id=run_id,
            info=info,
            baseline=baseline,
            baseline_tree=baseline_tree,
        )
    except BaseException as exc:
        if info is not None:
            try:
                remove_session_worktree(parent_cwd, run_id, force=True)
            except Exception as cleanup_exc:
                raise SubagentWorkspaceError(
                    f"{exc}; isolated workspace cleanup also failed: {cleanup_exc}"
                ) from exc
        if isinstance(exc, SubagentWorkspaceError):
            raise
        if not isinstance(exc, Exception):
            raise
        raise SubagentWorkspaceError(str(exc)) from exc


def remove_isolated_workspace(
    parent_cwd: Path,
    workspace: IsolatedWorkspace,
) -> None:
    remove_session_worktree(parent_cwd, workspace.run_id, force=True)


def _changed_entries(
    before: WorkspaceSnapshot,
    after: WorkspaceSnapshot,
) -> tuple[
    list[tuple[str, WorkspaceEntry | None, WorkspaceEntry | None]],
    list[tuple[str, WorkspaceEntry | None, WorkspaceEntry | None]],
]:
    old = before.by_path
    new = after.by_path
    file_changes: list[tuple[str, WorkspaceEntry | None, WorkspaceEntry | None]] = []
    directory_changes: list[
        tuple[str, WorkspaceEntry | None, WorkspaceEntry | None]
    ] = []
    for path in sorted(set(old) | set(new)):
        left = old.get(path)
        right = new.get(path)
        if left == right:
            continue
        if (left is None or left.kind == "dir") and (
            right is None or right.kind == "dir"
        ):
            directory_changes.append((path, left, right))
        else:
            file_changes.append((path, left, right))
    return file_changes, directory_changes


def _uncovered_directories(
    directories: list[tuple[str, WorkspaceEntry | None, WorkspaceEntry | None]],
    files: list[tuple[str, WorkspaceEntry | None, WorkspaceEntry | None]],
) -> list[str]:
    file_paths = [PurePosixPath(path) for path, _old, _new in files]
    unsupported: list[str] = []
    for value, before, after in directories:
        directory = PurePosixPath(value)
        mode_only = before is not None and after is not None
        if mode_only or not any(directory in path.parents for path in file_paths):
            unsupported.append(value)
    return unsupported


def _in_scope(path: str, source_relative: str) -> bool:
    scope = PurePosixPath(source_relative)
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return False
    return source_relative == "." or scope in candidate.parents


def _operation(
    path: str,
    before: WorkspaceEntry | None,
    after: WorkspaceEntry | None,
) -> dict[str, Any]:
    if before is None:
        kind = "add"
    elif after is None:
        kind = "delete"
    else:
        kind = "update"
    return {
        "kind": kind,
        "path": path,
        "before_sha256": before.sha256 if before is not None else None,
        "after_sha256": after.sha256 if after is not None else None,
        "before_bytes": before.size if before is not None else 0,
        "after_bytes": after.size if after is not None else 0,
        "before_mode": before.mode if before is not None else None,
        "after_mode": after.mode if after is not None else None,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    )


def _save_changeset(
    child_dir: Path,
    *,
    workspace: IsolatedWorkspace,
    status: str,
    final_sha256: str,
    operations: list[dict[str, Any]],
    patch: bytes = b"",
    detail: str = "",
) -> SubagentChangeSet:
    child_dir = Path(child_dir)
    patch_sha256 = hashlib.sha256(patch).hexdigest() if patch else ""
    if patch:
        (child_dir / PATCH_FILE).write_bytes(patch)
    payload = {
        "schema_version": CHANGESET_SCHEMA_VERSION,
        "task_id": workspace.task_id,
        "workspace": "isolated",
        "status": status,
        "source_relative": workspace.info.source_cwd.relative_to(
            workspace.info.repo_root
        ).as_posix(),
        "base_sha256": workspace.baseline.sha256,
        "final_sha256": final_sha256,
        "file_count": len(operations),
        "patch_file": PATCH_FILE if patch else None,
        "patch_bytes": len(patch),
        "patch_sha256": patch_sha256 or None,
        "limits": {
            "max_files": MAX_CHANGESET_FILES,
            "max_input_bytes": MAX_CHANGESET_INPUT_BYTES,
            "max_patch_bytes": MAX_CHANGESET_PATCH_BYTES,
        },
        "operations": operations,
        "detail": detail,
    }
    _write_json(child_dir / CHANGESET_FILE, payload)
    return SubagentChangeSet(
        task_id=workspace.task_id,
        status=status,
        file_count=len(operations),
        patch_bytes=len(patch),
        patch_sha256=patch_sha256,
        paths=tuple(item["path"] for item in operations),
        detail=detail,
    )


def capture_isolated_changes(
    workspace: IsolatedWorkspace,
    *,
    parent_cwd: Path,
    child_dir: Path,
    outcome_ready: bool,
) -> SubagentChangeSet:
    """Save a bounded patch only when source and child state stay valid."""
    try:
        if Path(parent_cwd).resolve() != workspace.info.source_cwd.resolve():
            return _save_changeset(
                child_dir,
                workspace=workspace,
                status="parent_changed",
                final_sha256="",
                operations=[],
                detail="The parent task directory changed while the child ran.",
            )
        parent_now = snapshot_workspace(workspace.info.repo_root)
        if parent_now != workspace.baseline:
            return _save_changeset(
                child_dir,
                workspace=workspace,
                status="parent_changed",
                final_sha256="",
                operations=[],
                detail=(
                    "The parent workspace changed while the child ran. "
                    "No child change can be applied."
                ),
            )
        final = snapshot_workspace(workspace.info.worktree_path)
        files, directories = _changed_entries(workspace.baseline, final)
        unsupported_directories = _uncovered_directories(directories, files)
        operations = [_operation(path, old, new) for path, old, new in files]
        type_changes = [
            path for path, old, new in files
            if old is not None and new is not None and old.kind != new.kind
        ]
        if type_changes:
            return _save_changeset(
                child_dir,
                workspace=workspace,
                status="unsupported",
                final_sha256=final.sha256,
                operations=operations,
                detail=(
                    "Git patch handoff does not support file and directory "
                    f"type changes: {json.dumps(type_changes)}."
                ),
            )
        if not files and not unsupported_directories:
            return _save_changeset(
                child_dir,
                workspace=workspace,
                status="empty",
                final_sha256=final.sha256,
                operations=[],
            )
        if unsupported_directories:
            return _save_changeset(
                child_dir,
                workspace=workspace,
                status="unsupported",
                final_sha256=final.sha256,
                operations=operations,
                detail=(
                    "Git patches cannot carry these empty or mode-only "
                    f"directory changes: {json.dumps(unsupported_directories)}."
                ),
            )
        source_relative = workspace.info.source_cwd.relative_to(
            workspace.info.repo_root
        ).as_posix()
        outside = [item["path"] for item in operations if not _in_scope(
            item["path"], source_relative
        )]
        if outside:
            return _save_changeset(
                child_dir,
                workspace=workspace,
                status="outside_scope",
                final_sha256=final.sha256,
                operations=operations,
                detail=f"Child changes escaped the task directory: {json.dumps(outside)}.",
            )
        input_bytes = sum(
            int(item["before_bytes"]) + int(item["after_bytes"])
            for item in operations
        )
        if len(operations) > MAX_CHANGESET_FILES or input_bytes > MAX_CHANGESET_INPUT_BYTES:
            return _save_changeset(
                child_dir,
                workspace=workspace,
                status="too_large",
                final_sha256=final.sha256,
                operations=operations,
                detail=(
                    f"The child changed {len(operations)} files and "
                    f"{input_bytes} input bytes. The limits are "
                    f"{MAX_CHANGESET_FILES} files and "
                    f"{MAX_CHANGESET_INPUT_BYTES} input bytes."
                ),
            )

        _git(workspace.info.worktree_path, "read-tree", workspace.baseline_tree)
        added = [
            item["path"] for item in operations if item["kind"] == "add"
        ]
        ignored = [
            path for path in added
            if _git(
                workspace.info.worktree_path,
                "check-ignore", "-q", "--", path,
                check=False,
            ).returncode == 0
        ]
        if ignored:
            return _save_changeset(
                child_dir,
                workspace=workspace,
                status="unsupported",
                final_sha256=final.sha256,
                operations=operations,
                detail=f"Git ignores these added files: {json.dumps(ignored)}.",
            )
        if added:
            _git(workspace.info.worktree_path, "add", "-N", "--", *added)
        patch = _git(
            workspace.info.worktree_path,
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            workspace.baseline_tree,
            "--",
        ).stdout
        if len(patch) > MAX_CHANGESET_PATCH_BYTES:
            return _save_changeset(
                child_dir,
                workspace=workspace,
                status="too_large",
                final_sha256=final.sha256,
                operations=operations,
                detail=(
                    f"The patch is {len(patch)} bytes. The limit is "
                    f"{MAX_CHANGESET_PATCH_BYTES} bytes."
                ),
            )
        diff_names = _git(
            workspace.info.worktree_path,
            "diff", "--name-only", "-z", workspace.baseline_tree, "--",
        ).stdout
        patch_paths = {
            raw.decode("utf-8")
            for raw in diff_names.split(b"\0")
            if raw
        }
        operation_paths = {str(item["path"]) for item in operations}
        if not patch or patch_paths != operation_paths:
            return _save_changeset(
                child_dir,
                workspace=workspace,
                status="unsupported",
                final_sha256=final.sha256,
                operations=operations,
                detail=(
                    "The Git patch does not cover every changed workspace path. "
                    "No child change can be applied."
                ),
            )
        return _save_changeset(
            child_dir,
            workspace=workspace,
            status="ready" if outcome_ready else "incomplete",
            final_sha256=final.sha256,
            operations=operations,
            patch=patch,
            detail=(
                "The child did not complete, so this patch is review-only."
                if not outcome_ready else ""
            ),
        )
    except Exception as exc:
        return _save_changeset(
            child_dir,
            workspace=workspace,
            status="error",
            final_sha256="",
            operations=[],
            detail=f"Change capture failed: {type(exc).__name__}: {str(exc)[:300]}",
        )


def _load_changeset(run_root: Path, task_id: object) -> tuple[Path, dict[str, Any]]:
    normalized = _safe_task_id(task_id)
    child_dir = Path(run_root) / "subagents" / normalized
    manifest_path = child_dir / CHANGESET_FILE
    try:
        payload = json.loads(manifest_path.read_text())
    except FileNotFoundError as exc:
        raise SubagentWorkspaceError(
            f"no isolated change set exists for {normalized}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SubagentWorkspaceError(
            f"isolated change set for {normalized} is unreadable"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CHANGESET_SCHEMA_VERSION
        or payload.get("task_id") != normalized
        or payload.get("workspace") != "isolated"
    ):
        raise SubagentWorkspaceError(
            f"isolated change set for {normalized} has invalid ownership metadata"
        )
    return child_dir, payload


def _validated_ready_patch(
    child_dir: Path,
    payload: dict[str, Any],
) -> tuple[bytes, tuple[tuple[str, str], ...]]:
    if payload.get("status") != "ready":
        raise SubagentWorkspaceError(
            f"change set {payload.get('task_id')} is {payload.get('status')}, not ready"
        )
    patch_name = payload.get("patch_file")
    if patch_name != PATCH_FILE:
        raise SubagentWorkspaceError("change set patch path is invalid")
    patch_path = child_dir / PATCH_FILE
    try:
        patch = patch_path.read_bytes()
    except OSError as exc:
        raise SubagentWorkspaceError("change set patch is unreadable") from exc
    if (
        len(patch) != payload.get("patch_bytes")
        or len(patch) > MAX_CHANGESET_PATCH_BYTES
        or hashlib.sha256(patch).hexdigest() != payload.get("patch_sha256")
    ):
        raise SubagentWorkspaceError("change set patch does not match its manifest")
    raw_operations = payload.get("operations")
    if not isinstance(raw_operations, list) or len(raw_operations) != payload.get(
        "file_count"
    ):
        raise SubagentWorkspaceError("change set operations are invalid")
    operations: list[tuple[str, str]] = []
    for item in raw_operations:
        if not isinstance(item, dict):
            raise SubagentWorkspaceError("change set operation is invalid")
        kind = item.get("kind")
        path = item.get("path")
        if kind not in {"add", "update", "delete"} or not isinstance(path, str):
            raise SubagentWorkspaceError("change set operation is invalid")
        candidate = PurePosixPath(path)
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or ".git" in candidate.parts
            or WORKTREE_DIR_NAME in candidate.parts
        ):
            raise SubagentWorkspaceError(f"unsafe change set path: {path!r}")
        operations.append((kind, path))
    if len(operations) > MAX_CHANGESET_FILES:
        raise SubagentWorkspaceError("change set exceeds its file limit")
    return patch, tuple(operations)


def review_subagent_changes(
    run_root: Path,
    task_id: object,
    *,
    offset: object = 0,
    limit: object = 200,
) -> str:
    """Return one bounded page from an immutable child patch."""
    try:
        child_dir, payload = _load_changeset(run_root, task_id)
        patch, _operations = _validated_ready_patch(child_dir, payload)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise SubagentWorkspaceError("offset must be an integer >= 0")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise SubagentWorkspaceError("limit must be an integer >= 1")
        limit = min(limit, MAX_REVIEW_LINES)
        lines = patch.decode("utf-8", errors="replace").splitlines()
        page = lines[offset:offset + limit]
        next_offset = offset + len(page)
        header = (
            f"Change set {payload['task_id']}: {payload['file_count']} file(s), "
            f"{payload['patch_bytes']} bytes, sha256={payload['patch_sha256']}. "
            f"Lines {offset + 1}-{next_offset} of {len(lines)}."
        )
        tail = (
            f"\nNext: subagent_changes(task_id=\"{payload['task_id']}\", "
            f"offset={next_offset}, limit={limit})"
            if next_offset < len(lines) else "\nEnd of change set."
        )
        return header + "\n" + "\n".join(page) + tail
    except SubagentWorkspaceError as exc:
        return f"ERROR: {exc}"


def apply_subagent_changes(
    run_root: Path,
    parent_cwd: Path,
    task_id: object,
) -> str:
    """Apply one reviewed patch after exact parent-state validation."""
    try:
        child_dir, payload = _load_changeset(run_root, task_id)
        patch, operations = _validated_ready_patch(child_dir, payload)
        application = child_dir / APPLICATION_FILE
        if application.exists():
            raise SubagentWorkspaceError(
                f"change set {payload['task_id']} was already applied"
            )
        repo_root = _repo_root(Path(parent_cwd).resolve())
        source_relative = Path(parent_cwd).resolve().relative_to(repo_root).as_posix()
        if source_relative != payload.get("source_relative"):
            raise SubagentWorkspaceError(
                "change set belongs to a different task directory"
            )
        outside = [
            path for _kind, path in operations
            if not _in_scope(path, source_relative)
        ]
        if outside:
            raise SubagentWorkspaceError(
                f"change set paths escape the task directory: {outside}"
            )
        before = snapshot_workspace(repo_root)
        if before.sha256 != payload.get("base_sha256"):
            raise SubagentWorkspaceError(
                "stale_subagent_changes: parent workspace no longer matches "
                "the child's starting state"
            )
        patch_path = child_dir / PATCH_FILE
        numstat = _git(
            repo_root, "apply", "--numstat", "-z", "--", str(patch_path)
        ).stdout
        patch_paths = {
            row.split(b"\t", 2)[2].decode("utf-8")
            for row in numstat.split(b"\0")
            if row and len(row.split(b"\t", 2)) == 3
        }
        if patch_paths != {path for _kind, path in operations}:
            raise SubagentWorkspaceError(
                "change set patch paths do not match its operation manifest"
            )
        _git(repo_root, "apply", "--check", "--binary", "--", str(patch_path))
        _git(repo_root, "apply", "--binary", "--", str(patch_path))
        after = snapshot_workspace(repo_root)
        if after.sha256 != payload.get("final_sha256"):
            rollback_error = ""
            try:
                _git(
                    repo_root,
                    "apply", "--reverse", "--check", "--binary", "--",
                    str(patch_path),
                )
                _git(
                    repo_root,
                    "apply", "--reverse", "--binary", "--", str(patch_path),
                )
            except Exception as exc:
                rollback_error = f" Rollback also failed: {exc}"
            if not rollback_error and snapshot_workspace(repo_root) != before:
                rollback_error = " Rollback did not restore the starting state."
            raise SubagentWorkspaceError(
                "applied patch did not produce the recorded child state."
                + rollback_error
            )
        _write_json(application, {
            "schema_version": CHANGESET_SCHEMA_VERSION,
            "task_id": payload["task_id"],
            "patch_sha256": payload["patch_sha256"],
            "final_sha256": payload["final_sha256"],
        })
        return AppliedSubagentResult(
            f"OK: applied isolated change set {payload['task_id']} "
            f"({len(operations)} file(s), sha256={payload['patch_sha256']})",
            operations,
        )
    except (SubagentWorkspaceError, ValueError) as exc:
        return f"ERROR: {exc}"


def copy_replay_changeset(source_child: Path, replay_child: Path) -> None:
    """Copy immutable handoff artifacts, but never source application state."""
    manifest = source_child / CHANGESET_FILE
    if not manifest.is_file():
        return
    replay_child.mkdir(parents=True, exist_ok=True)
    payload = manifest.read_bytes()
    (replay_child / CHANGESET_FILE).write_bytes(payload)
    parsed = json.loads(payload)
    if parsed.get("patch_file") == PATCH_FILE:
        (replay_child / PATCH_FILE).write_bytes(
            (source_child / PATCH_FILE).read_bytes()
        )


__all__ = [
    "APPLICATION_FILE",
    "CHANGESET_FILE",
    "MAX_CHANGESET_FILES",
    "MAX_CHANGESET_INPUT_BYTES",
    "MAX_CHANGESET_PATCH_BYTES",
    "PATCH_FILE",
    "AppliedSubagentResult",
    "IsolatedWorkspace",
    "SubagentChangeSet",
    "SubagentWorkspaceError",
    "apply_subagent_changes",
    "capture_isolated_changes",
    "copy_replay_changeset",
    "prepare_isolated_workspace",
    "remove_isolated_workspace",
    "review_subagent_changes",
]
