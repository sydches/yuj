"""Invisible per-turn git snapshots of the task workspace.

After any executed turn that wrote a source file, record the full workspace
state as a git object — WITHOUT a branch, log entry, or index change the
model could observe. Any turn then becomes a rewind/branch point, forever,
and each snapshot usually needs little storage.

Mechanism: git plumbing through a private index file. ``git add -A`` into
that index (never the repo's own), ``write-tree``, then ``commit-tree`` with
HEAD as parent — producing a dangling commit that ``git log``/``status``
cannot see. The turn→sha map lives in the telemetry dir beside the trace,
outside the model's world. Invisibility is a leak-class requirement, not a
nicety because the model may read its own ``git log``.

Under ``YUJ_CONTAINER``, run Git inside the container and keep the private
index in ``/tmp``. Otherwise, run Git on the host and keep the private index
in the telemetry directory.

Failure policy: snapshots are telemetry, never load-bearing for the solve.
Any failure logs once per session and returns None; the run continues.
"""
from __future__ import annotations

import copy
import gzip
import json
import logging
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._shared.telemetry_paths import ensure_telemetry_dir, telemetry_dir

log = logging.getLogger(__name__)

MAP_NAME = "turn_snapshots.tsv"
REWIND_SNAPSHOT_DIR = "rewind_snapshots"
REWIND_PENDING_NAME = "rewind_pending.json"
_REWIND_SNAPSHOT_VERSION = 1
_CONTAINER_INDEX = "/tmp/.yuj_snapshot_index"
# Container repositories may have a different owner. ``safe.directory`` lets
# Git use them. ``commit-tree`` also needs the temporary identity below.
_SNAPSHOT_SH = (
    "git config --global --add safe.directory {workdir} 2>/dev/null; "
    "export GIT_INDEX_FILE={index} "
    "GIT_AUTHOR_NAME=yuj GIT_AUTHOR_EMAIL=yuj@local "
    "GIT_COMMITTER_NAME=yuj GIT_COMMITTER_EMAIL=yuj@local; "
    "git add -A -- ':!.tool_output' ':!.solver' ':!prompt.txt' "
    "':!checkpoint.json' ':!metrics.json' >/dev/null 2>&1; "
    "tree=$(git write-tree 2>/dev/null) && "
    "echo 'yuj turn snapshot' | git commit-tree $tree -p HEAD 2>/dev/null"
)


def _container_id() -> str:
    return os.environ.get("YUJ_CONTAINER", "") or ""


def _run(repo_dir: Path, script: str) -> str:
    """Run a git shell snippet in the right place; return stdout."""
    cid = _container_id()
    if cid:
        argv = ["docker", "exec", "--workdir", "/testbed", cid, "bash", "-c", script]
        cwd = None
    else:
        argv = ["bash", "-c", script]
        cwd = str(repo_dir)
    out = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=60)
    return (out.stdout or "").strip()


def ensure_snapshot_setup(repo_dir: Path) -> None:
    """One-time per session: keep dangling snapshot objects alive.

    ``gc.auto 0`` stops background gc from pruning ref-less commits. Safe to
    call repeatedly; failure is non-fatal (snapshot() will then also fail
    and log its single warning).
    """
    try:
        _run(Path(repo_dir), "git config gc.auto 0")
    except Exception:
        pass


def snapshot(repo_dir: Path, turn: int, session=None) -> str | None:
    """Record an invisible workspace snapshot; return its sha (or None).

    Appends ``turn<TAB>sha`` to the telemetry map on success. The private
    index persists across calls (container /tmp or telemetry dir), so after
    the first snapshot each subsequent one stages only the delta.
    """
    repo_dir = Path(repo_dir)
    try:
        if _container_id():
            index = _CONTAINER_INDEX
            workdir = "/testbed"
        else:
            index = str(ensure_telemetry_dir(repo_dir) / ".snapshot_index")
            workdir = str(repo_dir)
        sha = _run(repo_dir, _SNAPSHOT_SH.format(index=index, workdir=workdir))
        if not sha or len(sha) < 7:
            raise RuntimeError(f"no sha (got {sha!r})")
        ensure_telemetry_dir(repo_dir)
        with open(telemetry_dir(repo_dir) / MAP_NAME, "a") as f:
            f.write(f"{int(turn)}\t{sha}\n")
        return sha
    except Exception as e:  # noqa: BLE001 — telemetry must never kill the run
        if session is not None and not getattr(session, "_snapshot_warned", False):
            setattr(session, "_snapshot_warned", True)
            log.warning("turn snapshot failed (disabled for session): %s", e)
        return None


def read_map(repo_dir: Path) -> list[tuple[int, str]]:
    """Return [(turn, sha), ...] recorded for this workspace's run."""
    p = telemetry_dir(Path(repo_dir)) / MAP_NAME
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[0].strip().isdigit():
            out.append((int(parts[0]), parts[1].strip()))
    return out


def sha_at_or_before(repo_dir: Path, turn: int) -> str | None:
    """The latest snapshot at or before ``turn`` — the rewind target."""
    best = None
    for t, sha in read_map(repo_dir):
        if t <= turn:
            best = sha
    return best


class ConversationRewindError(RuntimeError):
    """Conversation and workspace rewind could not be applied atomically."""


@dataclass(frozen=True)
class ConversationSnapshot:
    session_number: int
    turn_number: int
    checkpoint_commit: str
    original_prompt: str
    history_messages: list[dict[str, Any]]
    model_messages: list[dict[str, Any]]


def rewind_snapshot_dir(workspace: Path, artifact_dir: Path | None = None) -> Path:
    """Return the harness-owned directory for exact rewind snapshots."""
    workspace = Path(workspace).resolve()
    owner = Path(artifact_dir).resolve() if artifact_dir is not None else telemetry_dir(workspace)
    candidate = owner / REWIND_SNAPSHOT_DIR
    try:
        candidate.relative_to(workspace)
    except ValueError:
        return candidate
    return telemetry_dir(workspace) / REWIND_SNAPSHOT_DIR


def _snapshot_path(root: Path, session_number: int, turn_number: int) -> Path:
    return Path(root) / (
        f"session-{int(session_number):06d}-turn-{int(turn_number):012d}.json.gz"
    )


def _validate_history(messages: list[dict[str, Any]]) -> None:
    pending: set[str] = set()
    for message in messages:
        role = str(message.get("role") or "")
        if role == "assistant":
            if pending:
                raise ConversationRewindError(
                    "conversation snapshot crosses an unanswered tool call"
                )
            for call in message.get("tool_calls") or []:
                call_id = str(call.get("id") or "")
                if not call_id or call_id in pending:
                    raise ConversationRewindError(
                        "conversation snapshot has an invalid tool-call id"
                    )
                pending.add(call_id)
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if call_id not in pending:
                raise ConversationRewindError(
                    "conversation snapshot has an unmatched tool result"
                )
            pending.remove(call_id)
        elif pending:
            raise ConversationRewindError(
                "conversation snapshot crosses an unanswered tool call"
            )
    if pending:
        raise ConversationRewindError(
            "conversation snapshot ends before every tool result"
        )


def write_conversation_snapshot(
    root: Path,
    *,
    session_number: int,
    turn_number: int,
    checkpoint_commit: str,
    original_prompt: str,
    history_messages: list[dict[str, Any]],
    model_messages: list[dict[str, Any]],
) -> Path:
    """Atomically save the exact conversation and its checkpoint identity."""
    _validate_history(history_messages)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    path = _snapshot_path(root, session_number, turn_number)
    payload = {
        "schema_version": _REWIND_SNAPSHOT_VERSION,
        "session_number": int(session_number),
        "turn_number": int(turn_number),
        "checkpoint_commit": str(checkpoint_commit),
        "original_prompt": str(original_prompt),
        "history_messages": history_messages,
        "model_messages": model_messages,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(gzip.compress(encoded, mtime=0))
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def load_conversation_snapshot(
    root: Path, session_number: int, turn_number: int
) -> ConversationSnapshot:
    path = _snapshot_path(root, session_number, turn_number)
    if not path.is_file():
        raise ConversationRewindError(
            f"no conversation snapshot for session {session_number} turn {turn_number}"
        )
    try:
        payload = json.loads(gzip.decompress(path.read_bytes()))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversationRewindError(f"invalid conversation snapshot: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ConversationRewindError(
            f"invalid conversation snapshot payload in {path.name}"
        )
    if payload.get("schema_version") != _REWIND_SNAPSHOT_VERSION:
        raise ConversationRewindError(
            f"unsupported conversation snapshot version in {path.name}"
        )
    try:
        saved_session = int(payload["session_number"])
        saved_turn = int(payload["turn_number"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConversationRewindError(
            f"invalid conversation snapshot identity in {path.name}"
        ) from exc
    if saved_session != int(session_number) or saved_turn != int(turn_number):
        raise ConversationRewindError(
            f"conversation snapshot identity mismatch in {path.name}"
        )
    raw_history = payload.get("history_messages")
    raw_model = payload.get("model_messages")
    if (
        not isinstance(raw_history, list)
        or not all(isinstance(message, dict) for message in raw_history)
        or not isinstance(raw_model, list)
        or not all(isinstance(message, dict) for message in raw_model)
    ):
        raise ConversationRewindError(
            f"invalid conversation messages in {path.name}"
        )
    history = list(raw_history)
    model = list(raw_model)
    _validate_history(history)
    return ConversationSnapshot(
        session_number=saved_session,
        turn_number=saved_turn,
        checkpoint_commit=str(payload["checkpoint_commit"]),
        original_prompt=str(payload.get("original_prompt") or ""),
        history_messages=history,
        model_messages=model,
    )


def snapshot_turns(root: Path, session_number: int) -> list[int]:
    prefix = f"session-{int(session_number):06d}-turn-"
    turns: list[int] = []
    for path in Path(root).glob(f"{prefix}*.json.gz"):
        value = path.name[len(prefix):].split(".", 1)[0]
        if value.isdigit():
            turns.append(int(value))
    return sorted(set(turns))


def latest_snapshot_session(root: Path, turn_number: int) -> int | None:
    suffix = f"-turn-{int(turn_number):012d}.json.gz"
    sessions: list[int] = []
    for path in Path(root).glob(f"session-*{suffix}"):
        value = path.name[len("session-"):].split("-", 1)[0]
        if value.isdigit():
            sessions.append(int(value))
    return max(sessions) if sessions else None


def capture_conversation_snapshot(session, turn_number: int) -> ConversationSnapshot | None:
    """Capture one balanced turn boundary when rewind is enabled."""
    if not bool(
        getattr(getattr(session, "cfg", None), "rewind_enabled", False)
    ):
        return None
    store = getattr(session, "_checkpoint_store", None)
    if store is None:
        raise ConversationRewindError(
            "rewind requires the workspace checkpoint store"
        )
    from .workspace_checkpoints import CheckpointNotFoundError
    try:
        commit = store.checkpoint_for_turn(turn_number)
    except CheckpointNotFoundError:
        checkpoint = store.capture(turn_number)
        commit = checkpoint.commit
        session._emit(
            "checkpoint",
            session_number=session._session_number,
            checkpoint_reason="rewind_turn_boundary",
            **checkpoint.trace_fields(),
        )
    history = copy.deepcopy(session.context.get_history_messages())
    model_messages = copy.deepcopy(session.context.get_messages())
    root = Path(session._rewind_snapshot_dir)
    original_prompt = str(
        getattr(session.context, "_original_prompt", "")
        or next(
            (
                message.get("content", "")
                for message in history
                if message.get("role") == "user"
            ),
            "",
        )
    )
    write_conversation_snapshot(
        root,
        session_number=session._session_number,
        turn_number=turn_number,
        checkpoint_commit=commit,
        original_prompt=original_prompt,
        history_messages=history,
        model_messages=model_messages,
    )
    snapshot = load_conversation_snapshot(
        root, session._session_number, turn_number
    )
    session._rewind_guard_snapshots[int(turn_number)] = copy.deepcopy(
        session._guards
    )
    return snapshot


def _tool_identity(message: dict[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    identities: dict[str, tuple[str, dict[str, Any]]] = {}
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        raw_args = function.get("arguments") or {}
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                arguments = {}
        else:
            arguments = dict(raw_args)
        identities[str(call.get("id") or "")] = (
            str(function.get("name") or ""), arguments
        )
    return identities


def _build_context_from_snapshot(session, snapshot: ConversationSnapshot):
    from ._loop._session_setup import build_context_manager

    estimator = getattr(session.context, "_token_estimator", None)
    context = build_context_manager(
        type(session.context),
        session.cfg,
        Path(session.cwd),
        snapshot.original_prompt,
        1,
        estimator,
    )
    if context is None:
        raise ConversationRewindError("context mode cannot be rebuilt for rewind")
    tool_identities: dict[str, tuple[str, dict[str, Any]]] = {}
    for raw_message in snapshot.history_messages:
        message = copy.deepcopy(raw_message)
        role = str(message.get("role") or "")
        if role == "system":
            context.add_system(str(message.get("content") or ""))
        elif role == "user":
            context.add_user(str(message.get("content") or ""))
        elif role == "assistant":
            context.add_assistant(message)
            tool_identities.update(_tool_identity(message))
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            tool_name, arguments = tool_identities.get(call_id, ("", {}))
            cmd_signature = (
                json.dumps(arguments, sort_keys=True, separators=(",", ":"))
                if tool_name == "bash"
                else ""
            )
            context.add_tool_result(
                call_id,
                str(message.get("content") or ""),
                tool_name=tool_name,
                cmd_signature=cmd_signature,
            )
        else:
            raise ConversationRewindError(
                f"unsupported canonical message role in rewind snapshot: {role!r}"
            )
    history = context.get_history_messages()
    history[:] = copy.deepcopy(snapshot.history_messages)
    if not context.pin_model_messages(snapshot.model_messages):
        raise ConversationRewindError(
            "context mode cannot restore exact model-facing messages"
        )
    if context.get_messages() != snapshot.model_messages:
        raise ConversationRewindError(
            "restored model-facing messages differ from the rewind snapshot"
        )
    return context


def request_rewind(session, turn_number: int | None, *, reason: str) -> None:
    """Queue a guardrail rewind for the current turn boundary."""
    if getattr(session, "_pending_rewind", None) is not None:
        raise ConversationRewindError("a rewind is already pending")
    if turn_number is None:
        candidates = [
            value
            for value in snapshot_turns(
                session._rewind_snapshot_dir, session._session_number
            )
            if value < int(session._current_turn)
        ]
        if not candidates:
            raise ConversationRewindError("no earlier rewind snapshot is available")
        turn_number = candidates[-1]
    session._pending_rewind = {
        "turn_number": int(turn_number),
        "reason": str(reason or "guardrail"),
    }


def rewind_to(session, turn_number: int, *, reason: str = "operator") -> dict[str, Any]:
    """Restore the exact conversation and tree for an earlier turn."""
    if not bool(getattr(session.cfg, "rewind_enabled", False)):
        raise ConversationRewindError("loop.rewind_enabled is false")
    maximum = int(getattr(session.cfg, "rewind_max_per_session", 1))
    count = int(getattr(session, "_rewind_count", 0))
    if count >= maximum:
        raise ConversationRewindError(
            f"rewind limit reached ({count}/{maximum})"
        )
    from_turn = int(session._current_turn)
    turn_number = int(turn_number)
    if turn_number < 0 or turn_number >= from_turn:
        raise ConversationRewindError(
            f"rewind target must be earlier than turn {from_turn}"
        )
    snapshot = load_conversation_snapshot(
        session._rewind_snapshot_dir, session._session_number, turn_number
    )
    store = getattr(session, "_checkpoint_store", None)
    if store is None:
        raise ConversationRewindError("rewind requires workspace checkpoints")
    commit = store.checkpoint_for_turn(turn_number)
    if commit != snapshot.checkpoint_commit:
        raise ConversationRewindError(
            "conversation snapshot and workspace checkpoint do not match"
        )
    restored_context = _build_context_from_snapshot(session, snapshot)
    restored = store.restore_checkpoint(turn_number)
    session.context = restored_context
    guard_snapshot = session._rewind_guard_snapshots.get(turn_number)
    if guard_snapshot is not None:
        session._guards = copy.deepcopy(guard_snapshot)
    session._output_dedup_cache.clear()
    session._last_actual_prompt_tokens = 0
    session._last_fill = 0.0
    session._preflight_prev_estimate = None
    session._prev_preflight_estimate_pt = 0
    session._rewind_count = count + 1
    rewind_id = uuid.uuid4().hex
    event = {
        "session_number": session._session_number,
        "turn_number": from_turn,
        "from_turn": from_turn,
        "to_turn": turn_number,
        "reason": str(reason or "operator"),
        "commit": restored.commit,
        "rewind_count": session._rewind_count,
        "rewind_id": rewind_id,
        "delivery": "in_session",
    }
    from .state_writer import active_events
    from .stale_guard import StaleFileGuard

    stale_guard = StaleFileGuard.from_trace(
        cwd=session.cwd,
        mode=getattr(session.cfg, "tools_stale_guard_mode", "warn"),
        events=active_events([
            *session._trace_events,
            {"event": "rewind", **event},
        ]),
        event_sink=getattr(session._stale_guard, "event_sink", None),
    )
    session._emit("rewind", **event)
    session._stale_guard = stale_guard
    return event


def _replay_rewinds_at(session, turn_number: int) -> list[dict[str, Any]]:
    client = getattr(session, "client", None)
    if not bool(getattr(client, "is_replay", False)):
        return []
    getter = getattr(client, "rewinds_at", None)
    if getter is None:
        return []
    return list(getter(session._session_number, int(turn_number)))


def process_rewind_turn_boundary(session, turn_number: int) -> bool:
    """Capture a completed turn, then apply queued or replayed rewinds."""
    pending = getattr(session, "_pending_rewind", None)
    if pending is not None:
        session._pending_rewind = None
        rewind_to(
            session,
            int(pending["turn_number"]),
            reason=str(pending.get("reason") or "guardrail"),
        )
        return True
    capture_conversation_snapshot(session, turn_number)
    rewound = False
    for source in _replay_rewinds_at(session, turn_number):
        event = rewind_to(
            session,
            int(source["to_turn"]),
            reason=str(source.get("reason") or "replay"),
        )
        verifier = getattr(session.client, "verify_rewind_event", None)
        if verifier is not None:
            verifier(event)
        rewound = True
    return rewound


def pending_rewind_path(root: Path) -> Path:
    return Path(root) / REWIND_PENDING_NAME


def save_pending_rewind(root: Path, payload: dict[str, Any]) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    path = pending_rewind_path(root)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.chmod(0o600)
    os.replace(temporary, path)
    return path


def load_pending_rewind(root: Path) -> dict[str, Any] | None:
    path = pending_rewind_path(root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConversationRewindError("invalid pending rewind record") from exc
    return payload if isinstance(payload, dict) else None


def apply_pending_rewind_resume(session) -> ConversationSnapshot | None:
    """Restore an operator rewind's exact messages in the next run segment."""
    payload = load_pending_rewind(session._rewind_snapshot_dir)
    if not payload or int(payload.get("applied_session_number", 0) or 0) > 0:
        return None
    snapshot = load_conversation_snapshot(
        session._rewind_snapshot_dir,
        int(payload["target_session_number"]),
        int(payload["to_turn"]),
    )
    if snapshot.checkpoint_commit != str(payload.get("commit") or ""):
        raise ConversationRewindError(
            "pending rewind does not match its conversation snapshot"
        )
    store = getattr(session, "_checkpoint_store", None)
    if store is None:
        raise ConversationRewindError(
            "pending rewind requires workspace checkpoints"
        )
    checkpoint_commit = store.checkpoint_for_turn(snapshot.turn_number)
    if checkpoint_commit != snapshot.checkpoint_commit:
        raise ConversationRewindError(
            "pending rewind does not match its workspace checkpoint"
        )
    store.restore_checkpoint(snapshot.turn_number)
    session.context = _build_context_from_snapshot(session, snapshot)
    payload["applied_session_number"] = session._session_number
    save_pending_rewind(session._rewind_snapshot_dir, payload)
    session._emit(
        "rewind_resume",
        session_number=session._session_number,
        rewind_id=str(payload["rewind_id"]),
        target_session_number=int(payload["target_session_number"]),
        to_turn=int(payload["to_turn"]),
        commit=snapshot.checkpoint_commit,
    )
    return snapshot


__all__ = [
    "ensure_snapshot_setup",
    "snapshot",
    "read_map",
    "sha_at_or_before",
    "MAP_NAME",
    "ConversationRewindError",
    "ConversationSnapshot",
    "REWIND_PENDING_NAME",
    "REWIND_SNAPSHOT_DIR",
    "apply_pending_rewind_resume",
    "capture_conversation_snapshot",
    "latest_snapshot_session",
    "load_conversation_snapshot",
    "load_pending_rewind",
    "pending_rewind_path",
    "process_rewind_turn_boundary",
    "request_rewind",
    "rewind_snapshot_dir",
    "rewind_to",
    "save_pending_rewind",
    "snapshot_turns",
    "write_conversation_snapshot",
]
