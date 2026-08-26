"""Assistant-mode session runner built on the shared harness engine."""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from ..llm_solver.config import (
    PROJECT_ROOT,
    get_server_base_url,
    load_config,
    require_runtime_mode,
)
from ..llm_solver.harness import TaskSpec, solve_task
from ..llm_solver.harness.context_strategies import resolve_context_class
from ..llm_solver.harness.clarifications import (
    clarification_state,
    supersede_clarification_for_rewind,
)
from ..llm_solver.harness.corrections import (
    CorrectionStateError,
    validate_correction_trace,
)
from ..llm_solver.harness._loop.model_role_runtime import build_model_role_runtime
from ..llm_solver.harness.loop import _load_trace_events
from ..llm_solver.harness.sandbox.policy import (
    SandboxResolution,
    bind_sandbox_resolution,
    preflight_sandbox,
)
from ..llm_solver.harness.worktree_runtime import (
    WorktreeRuntimeError,
    create_session_worktree,
    inspect_session_worktree,
)
from ..llm_solver.harness.workspace_checkpoints import (
    WorkspaceCheckpointStore,
    default_shadow_dir,
)
from ..llm_solver.models import model_supports_image_inputs, resolve_model
from ..llm_solver.server import LlamaClient, load_profile
from ..llm_solver.server.types import ImageInput
from ._anthropic import AnthropicClient
from ._auth import (
    AccountIneligibleError,
    AuthBinding,
    CredentialRevokedError,
    CredentialSession,
    CredentialStore,
    ProviderAuthError,
    ProviderRequestError,
    validate_auth_endpoint,
)
from ._codex import CodexSubscriptionClient
from ._images import ImageInputError, load_session_images
from .forking import validate_correction_owner
from .store import SessionRecord, SessionStore


class _ProviderAPIKeyClient(LlamaClient):
    """OpenAI-compatible API-key client with safe provider auth failures."""

    def __init__(self, cfg, profile=None, *, provider: str):
        super().__init__(cfg, profile=profile)
        self._provider_name = provider

    def _call_api(self, payload: dict, *, record_transcript: bool = True):
        n = 0
        if record_transcript:
            self._transcript_call_n += 1
            n = self._transcript_call_n
            self._write_transcript(
                f"turn {n:03d} input", json.dumps(payload, default=str)
            )
        try:
            response = super()._call_api(payload, record_transcript=False)
        except Exception as exc:
            translated = self._translate_provider_error(exc)
            if isinstance(translated, ProviderAuthError):
                self._last_provider_auth_error = translated
            if record_transcript:
                detail = (
                    str(translated)
                    if isinstance(translated, ProviderAuthError)
                    else "provider request failed"
                )
                self._write_transcript(
                    f"turn {n:03d} output",
                    f"{type(translated).__name__}: {detail}",
                )
            if translated is exc:
                raise
            raise translated from exc
        if record_transcript:
            self._write_transcript(
                f"turn {n:03d} output", response.model_dump_json()
            )
        return response

    def health_check(self) -> list[str]:
        try:
            return super().health_check()
        except Exception as exc:
            translated = self._translate_provider_error(exc)
            if translated is exc:
                raise
            raise translated from exc

    def _translate_provider_error(self, exc: Exception) -> Exception:
        status = int(getattr(exc, "status_code", 0) or 0)
        if status == 401:
            return CredentialRevokedError(
                self._provider_name, "API key was rejected or revoked"
            )
        if status == 403:
            return AccountIneligibleError(
                self._provider_name, "account is not eligible for this request"
            )
        if status in {404, 405, 410, 422}:
            return ProviderRequestError(
                self._provider_name,
                f"API request failed with HTTP {status}",
            )
        return exc


def _make_client(
    cfg,
    profile,
    *,
    auth_binding: AuthBinding | None = None,
    auth_store: CredentialStore | None = None,
    http=None,
    now=None,
) -> LlamaClient:
    """Pick one explicit transport and never fall back between credentials."""
    if auth_binding is not None:
        validate_auth_endpoint(cfg, auth_binding)
        auth = CredentialSession(
            auth_binding,
            store=auth_store,
            http=http,
            now=now,
        )
        if auth_binding.provider == "claude":
            return AnthropicClient(
                cfg, profile=profile, auth=auth, http=http
            )
        if auth_binding.auth_method == "subscription":
            return CodexSubscriptionClient(
                cfg, profile=profile, auth=auth, http=http
            )

        credential = auth.access()
        client = _ProviderAPIKeyClient(
            replace(cfg, api_key=credential.token),
            profile=profile,
            provider=auth_binding.provider,
        )
        # The SDK retains the key internally. Keep it out of the config object
        # shared with the harness, its artifacts, and inspection surfaces.
        client.cfg = cfg
        return client
    if getattr(cfg, "provider", "") == "anthropic":
        return AnthropicClient(cfg, profile=profile)
    return LlamaClient(cfg, profile=profile)


def _protect_auth_environment(
    cfg,
    binding: AuthBinding | None,
    *,
    store: CredentialStore | None = None,
):
    """Pin managed authentication and remove its environment from tools."""
    if binding is None:
        return cfg
    # A provider-scoped session must never advance to a configured fallback
    # target after a request failure.
    cfg = replace(cfg, model_fallback_chain={})
    excluded_names = {"YUJ_AUTH_HOME"}
    if binding.auth_method == "api_key":
        session = CredentialSession(binding, store=store)
        environment_name = session.environment_name()
        if environment_name:
            excluded_names.add(environment_name)
    sandbox_set = dict(cfg.sandbox_env_set)
    filters = dict(cfg.sandbox_env_filters)
    for environment_name in excluded_names:
        sandbox_set.pop(environment_name, None)
        filters[environment_name] = "exclude"
    return replace(
        cfg,
        sandbox_env_set=sandbox_set,
        sandbox_env_filters=filters,
    )


_EMPTY_STATE = {
    "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
    "trace": [],
    "gates": [],
    "evidence": [],
    "inference": [],
}
_APPROVAL_REQUEST_FILE = "approval_request.json"
_APPROVAL_DECISIONS_FILE = "approval_decisions.json"
_INTERRUPT_MARKER_FILE = "shell_interrupt.json"


def create_session(
    store: SessionStore,
    *,
    cwd: Path,
    prompt_text: str,
    prompt_source: str,
    model: str,
    config_paths: list[Path],
    system_prompt_path: Path | None,
    context_mode: str,
    auth_binding: AuthBinding | None = None,
) -> SessionRecord:
    record = store.create_session(
        cwd=cwd,
        model=model,
        prompt_text=prompt_text,
        prompt_source=prompt_source,
        context_mode=context_mode,
        system_prompt_path=system_prompt_path,
        config_paths=config_paths,
        provider=auth_binding.provider if auth_binding else None,
        auth_method=auth_binding.auth_method if auth_binding else None,
        credential_id=auth_binding.credential_id if auth_binding else None,
    )
    store.set_active_session(cwd, record.session_id)
    _seed_session_artifacts(record)
    return record


def prepare_smoke_repo(root: Path | None = None) -> Path:
    """Create a minimal throwaway repo for assistant smoke runs."""
    if root is None:
        root = Path(tempfile.mkdtemp(prefix="assist-smoke-"))
    root = Path(root).resolve()
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "calc.py").write_text(
        "def add(a, b):\n"
        "    return a - b\n"
    )
    (root / "tests" / "test_calc.py").write_text(
        "from calc import add\n\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n"
    )
    return root


def run_session(
    store: SessionStore,
    record: SessionRecord,
    *,
    resume: bool,
    resume_prompt_text: str | None = None,
) -> tuple[bool, str | None]:
    """Run exactly one harness outer session for an assistant record."""
    artifact_dir = record.artifact_path
    correction = validate_correction_trace(artifact_dir)
    validate_correction_owner(record, correction)
    clarification = clarification_state(artifact_dir)
    if clarification.phase == "input_required":
        raise RuntimeError(
            "session has a pending clarification; record its answer before resume"
        )
    clear_interrupt_marker(artifact_dir)

    config_paths = [Path(p) for p in record.config_paths]
    overrides = {
        "runtime_mode": "assistant",
        "max_sessions": 1,
        "model": record.model,
    }
    cfg = load_config(user_config=config_paths, overrides=overrides)
    require_runtime_mode(cfg, expected="assistant", caller="scripts.llm_assist")
    worktree_info, record = _resolve_session_worktree(
        store, record, cfg=cfg, resume=resume
    )
    auth_binding = _record_auth_binding(record)
    auth_store = CredentialStore() if auth_binding is not None else None
    if auth_store is not None:
        auth_store.require_outside_target(Path(record.cwd))
    cfg = _protect_auth_environment(cfg, auth_binding, store=auth_store)
    sandbox_resolution = preflight_sandbox(cfg)
    cfg = bind_sandbox_resolution(cfg, sandbox_resolution)
    _record_sandbox_provenance(record, sandbox_resolution)
    profile = _load_profile(cfg)
    stored_images = load_session_images(artifact_dir)
    if stored_images:
        _require_image_capability(cfg, profile, auth_binding=auth_binding)
        # Image evidence belongs to the selected primary transport. Do not
        # let a fallback silently drop it or send it to an unchecked target.
        cfg = replace(cfg, model_fallback_chain={})
    client = _make_client(
        cfg,
        profile,
        auth_binding=auth_binding,
        auth_store=auth_store,
    )
    if hasattr(client, "set_session_id"):
        client.set_session_id(record.session_id)
    if stored_images:
        client.set_image_inputs([
            ImageInput(media_type=image.media_type, data=image.data)
            for image in stored_images
        ])
    cfg = _apply_effective_context(cfg, client)
    client.cfg = cfg
    build_model_role_runtime(
        cfg=cfg,
        main_client=client,
        profiles_dir=PROJECT_ROOT / "profiles",
        client_factory=lambda role_cfg, role_profile: _make_client(
            role_cfg,
            role_profile,
            auth_binding=auth_binding,
            auth_store=auth_store,
        ),
    )
    store.update_session(
        record.session_id, status="running", last_finish_reason=None
    )

    prompt_text = record.prompt_text
    if resume:
        approval = load_approval_request(record.artifact_path)
        if approval is not None and approval.get("status") == "approved":
            reason = approval.get("reason") or "approval granted"
            args_summary = approval.get("args_summary") or approval.get("cmd") or ""
            prompt_text = (
                prompt_text.rstrip()
                + "\n\n"
                + f"Operator note: the previously blocked action is now approved ({reason}). "
                + f"If it is still the right next step, re-issue: {args_summary}"
            )
        elif approval is not None and approval.get("status") == "rejected":
            reason = approval.get("rejection_reason") or approval.get("reason") or "approval rejected"
            args_summary = approval.get("args_summary") or approval.get("cmd") or ""
            prompt_text = (
                prompt_text.rstrip()
                + "\n\n"
                + f"Operator note: the previously blocked action was rejected ({reason}). "
                + f"Do not re-issue it unchanged: {args_summary}"
            )
        if resume_prompt_text is not None:
            prompt_text = (
                prompt_text.rstrip()
                + "\n\nOperator follow-up:\n"
                + resume_prompt_text
            )

    success = solve_task(
        worktree_info.session_cwd if worktree_info is not None else Path(record.cwd),
        cfg,
        client,
        system_prompt_file=Path(record.system_prompt_path) if record.system_prompt_path else None,
        context_class=resolve_context_class(record.context_mode),
        task_spec=TaskSpec(prompt_text=prompt_text),
        artifacts_dir=artifact_dir,
        resume_from_artifacts=resume,
        worktree_info=worktree_info,
    )
    provider_auth_error = getattr(client, "_last_provider_auth_error", None)
    if isinstance(provider_auth_error, ProviderAuthError):
        store.update_session(
            record.session_id,
            status="error",
            last_finish_reason="provider_auth_error",
        )
        raise provider_auth_error
    finish_reason = last_finish_reason(artifact_dir)
    store.update_session(
        record.session_id,
        status=_status_from_result(success, finish_reason),
        last_finish_reason=finish_reason,
    )
    return success, finish_reason


def rewind_session(
    store: SessionStore,
    record: SessionRecord,
    *,
    turn: int,
    reason: str = "operator_cli",
) -> dict[str, object]:
    """Apply an offline assistant rewind and stage its exact next resume."""
    config_paths = [Path(path) for path in record.config_paths]
    cfg = load_config(
        user_config=config_paths,
        overrides={
            "runtime_mode": "assistant",
            "max_sessions": 1,
            "model": record.model,
        },
    )
    if not cfg.rewind_enabled:
        raise RuntimeError("loop.rewind_enabled is false for this session")
    if not cfg.tools_file_checkpoints_enabled:
        raise RuntimeError("rewind requires tools.file_checkpoints_enabled")

    worktree_identity = (
        record.worktree_path,
        record.worktree_branch,
        record.worktree_base_commit,
    )
    if any(worktree_identity) and not all(worktree_identity):
        raise WorktreeRuntimeError(
            "saved session has incomplete worktree identity"
        )
    workspace = (
        Path(record.worktree_path)
        if record.worktree_path
        else Path(record.cwd)
    ).resolve()
    if record.worktree_path:
        inspected = inspect_session_worktree(
            Path(record.cwd), record.session_id
        )
        expected_worktree = (
            workspace,
            str(record.worktree_branch),
            str(record.worktree_base_commit),
        )
        actual_worktree = (
            inspected.worktree_path.resolve(),
            inspected.branch,
            inspected.base_commit,
        )
        if actual_worktree != expected_worktree:
            raise WorktreeRuntimeError(
                "saved worktree identity does not match the owned Git worktree"
            )

    artifact_dir = record.artifact_path.resolve()
    trace_path = artifact_dir / ".trace.jsonl"
    events = _load_trace_events(trace_path)
    session_numbers = [
        int(event["session_number"])
        for event in events
        if isinstance(event.get("session_number"), int)
    ]
    if not session_numbers:
        raise RuntimeError("session trace has no session_number")
    target_session = max(session_numbers)
    from_turns = [
        int(event["turn_number"])
        for event in events
        if event.get("session_number") == target_session
        and isinstance(event.get("turn_number"), int)
    ]
    if not from_turns:
        raise RuntimeError("latest session trace has no completed turn")
    from_turn = max(from_turns)
    turn = int(turn)
    if turn < 0 or turn >= from_turn:
        raise RuntimeError(
            f"rewind target must be earlier than turn {from_turn}"
        )

    from ..llm_solver.harness.turn_snapshots import (
        load_conversation_snapshot,
        load_pending_rewind,
        rewind_snapshot_dir,
        save_pending_rewind,
    )
    snapshot_root = rewind_snapshot_dir(workspace, artifact_dir)
    pending = load_pending_rewind(snapshot_root)
    if pending and not int(pending.get("applied_session_number", 0) or 0):
        raise RuntimeError("a prior rewind is still waiting for resume")
    snapshot = load_conversation_snapshot(
        snapshot_root, target_session, turn
    )
    rewind_count = sum(
        1
        for event in events
        if event.get("event") == "rewind"
        and isinstance(event.get("rewind_id"), str)
        and event.get("session_number") == target_session
    )
    if rewind_count >= cfg.rewind_max_per_session:
        raise RuntimeError(
            "rewind limit reached "
            f"({rewind_count}/{cfg.rewind_max_per_session})"
        )

    checkpoint_candidate = artifact_dir / ".shadow_git"
    try:
        checkpoint_candidate.relative_to(workspace)
    except ValueError:
        checkpoint_shadow_dir = checkpoint_candidate
    else:
        checkpoint_shadow_dir = default_shadow_dir(workspace)
    checkpoint_store = WorkspaceCheckpointStore(
        workspace,
        shadow_dir=checkpoint_shadow_dir,
        excludes=cfg.tools_file_checkpoints_exclude,
    )
    commit = checkpoint_store.checkpoint_for_turn(turn)
    if commit != snapshot.checkpoint_commit:
        raise RuntimeError(
            "conversation snapshot and workspace checkpoint do not match"
        )
    restored = checkpoint_store.restore_checkpoint(turn)
    rewind_id = uuid.uuid4().hex
    event = {
        "session_number": target_session,
        "turn_number": from_turn,
        "from_turn": from_turn,
        "to_turn": turn,
        "reason": str(reason or "operator_cli"),
        "commit": restored.commit,
        "rewind_count": rewind_count + 1,
        "rewind_id": rewind_id,
        "delivery": "next_session",
    }
    from ..llm_solver.harness._loop.trace_schema import emit_trace_event
    with open(trace_path, "a") as trace_file:
        emit_trace_event(trace_file, "rewind", **event)
        superseded = supersede_clarification_for_rewind(
            artifact_dir,
            rewind_id=rewind_id,
            to_turn=turn,
        )
        if superseded is not None:
            emit_trace_event(
                trace_file,
                "clarification_rewound",
                session_number=target_session,
                turn_number=from_turn,
                request_id=superseded["request_id"],
                rewind_id=rewind_id,
                to_turn=turn,
            )
    from ..llm_solver.harness.state_writer import write_state_from_trace
    write_state_from_trace(
        trace_path,
        artifact_dir / ".solver" / "state.json",
        max_result_chars=cfg.max_output_chars,
        imperative_projection=getattr(
            cfg, "state_imperative_projection_enabled", False
        ),
    )
    save_pending_rewind(snapshot_root, {
        "schema_version": 1,
        "rewind_id": rewind_id,
        "target_session_number": target_session,
        "from_turn": from_turn,
        "to_turn": turn,
        "reason": event["reason"],
        "commit": restored.commit,
        "applied_session_number": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    approval = load_approval_request(artifact_dir)
    if approval is not None and approval.get("status") == "pending":
        approval["status"] = "rewound"
        approval["rejection_reason"] = "superseded by conversation rewind"
        save_approval_request(artifact_dir, approval)
    clear_interrupt_marker(artifact_dir)
    store.update_session(
        record.session_id,
        status="paused",
        last_finish_reason="rewind",
    )
    return event


def _resolve_session_worktree(
    store: SessionStore,
    record: SessionRecord,
    *,
    cfg,
    resume: bool,
):
    """Create once or strictly reuse the worktree owned by this record."""
    mode = str(getattr(cfg, "runtime_worktree", "off") or "off").strip()
    stored = (
        record.worktree_path,
        record.worktree_branch,
        record.worktree_base_commit,
    )
    has_any_stored = any(value is not None for value in stored)
    has_all_stored = all(bool(value) for value in stored)

    if resume:
        if has_any_stored and not has_all_stored:
            raise WorktreeRuntimeError(
                "saved session has incomplete worktree identity"
            )
        if has_all_stored and mode == "off":
            raise WorktreeRuntimeError(
                "saved session owns a worktree but runtime.worktree is off"
            )
        if not has_all_stored:
            if mode != "off":
                raise WorktreeRuntimeError(
                    "cannot resume with worktree isolation: the saved session "
                    "has no worktree identity"
                )
            return None, record
        inspected = inspect_session_worktree(Path(record.cwd), record.session_id)
        expected = (
            Path(str(record.worktree_path)).resolve(),
            str(record.worktree_branch),
            str(record.worktree_base_commit),
        )
        inspected_actual = (
            inspected.worktree_path.resolve(),
            inspected.branch,
            inspected.base_commit,
        )
        if inspected_actual != expected:
            raise WorktreeRuntimeError(
                "saved worktree identity does not match the owned Git worktree"
            )
        info = create_session_worktree(
            Path(record.cwd),
            mode=mode,
            run_id=record.session_id,
            reuse=True,
        )
        assert info is not None
        actual = (info.worktree_path.resolve(), info.branch, info.base_commit)
        if actual != expected:
            raise WorktreeRuntimeError(
                "saved worktree identity does not match the owned Git worktree"
            )
        return info, record

    if mode == "off":
        return None, record
    if has_any_stored:
        raise WorktreeRuntimeError("new session already has worktree identity")
    info = create_session_worktree(
        Path(record.cwd),
        mode=mode,
        run_id=record.session_id,
        reuse=False,
    )
    assert info is not None
    store.update_session_worktree(
        record.session_id,
        path=info.worktree_path,
        branch=info.branch,
        base_commit=info.base_commit,
    )
    refreshed = store.get_session(record.session_id)
    if refreshed is None:
        raise WorktreeRuntimeError("session disappeared while saving worktree identity")
    _write_session_metadata(refreshed)
    return info, refreshed


def last_finish_reason(artifact_dir: Path) -> str | None:
    """Return the latest session_end finish_reason from a session bundle."""
    events = _load_trace_events(Path(artifact_dir) / ".trace.jsonl")
    for ev in reversed(events):
        if ev.get("event") == "session_end":
            reason = ev.get("finish_reason")
            return str(reason) if reason is not None else None
    return None


@dataclass(frozen=True)
class LiveState:
    """Shell-owned live status inferred from artifacts.

    ``status`` is empty when artifacts are insufficient; the caller falls
    back to the SQLite row. ``finish_reason`` is non-None only when a
    ``session_end`` has been observed with no later ``session_start``.
    """

    status: str
    finish_reason: str | None
    session_number: int


def derive_live_state(artifact_dir: Path) -> LiveState:
    """Infer live status from trace, approval, clarification, and interrupt state.

    Precedence:
      1. Pending clarification → ``input_required``.
      2. Recorded, unconsumed answer → ``input_ready`` unless approval waits.
      3. Pending approval request → ``approval_pending``.
      4. Last lifecycle event is ``session_end`` → map finish_reason to
         ``completed`` / ``paused`` / ``error``.
      5. Last lifecycle event is ``session_start`` → ``running``.
      6. Nothing observed → empty status; caller uses the SQLite row.
    """
    artifact_dir = Path(artifact_dir)
    events = _load_trace_events(artifact_dir / ".trace.jsonl")
    approval = load_approval_request(artifact_dir)
    interrupt = load_interrupt_marker(artifact_dir)
    clarification = clarification_state(artifact_dir)

    last_lifecycle: dict | None = None
    last_lifecycle_index = -1
    last_rewind: dict | None = None
    last_rewind_index = -1
    last_fork: dict | None = None
    last_fork_index = -1
    for index, ev in enumerate(events):
        if ev.get("event") in {"session_start", "session_end"}:
            last_lifecycle = ev
            last_lifecycle_index = index
        elif ev.get("event") == "rewind":
            last_rewind = ev
            last_rewind_index = index
        elif ev.get("event") == "session_fork":
            last_fork = ev
            last_fork_index = index
    session_number = (
        int(last_lifecycle.get("session_number", 0) or 0)
        if last_lifecycle is not None
        else 0
    )

    if clarification.phase == "input_required":
        assert clarification.request is not None
        return LiveState(
            status="input_required",
            finish_reason="input_required",
            session_number=int(clarification.request["session_number"]),
        )
    if clarification.phase == "input_ready" and not (
        approval is not None and approval.get("status") == "pending"
    ):
        assert clarification.request is not None
        return LiveState(
            status="input_ready",
            finish_reason="input_answered",
            session_number=int(clarification.request["session_number"]),
        )

    if approval is not None and approval.get("status") == "pending":
        return LiveState(
            status="approval_pending",
            finish_reason=None,
            session_number=session_number,
        )

    if interrupt is not None and (
        last_lifecycle is None or last_lifecycle.get("event") == "session_start"
    ):
        finish_reason = str(interrupt.get("finish_reason") or "interrupted")
        return LiveState(
            status="paused",
            finish_reason=finish_reason,
            session_number=session_number,
        )

    if (
        last_rewind is not None
        and last_rewind_index > last_lifecycle_index
        and last_rewind.get("delivery") == "next_session"
    ):
        return LiveState(
            status="paused",
            finish_reason="rewind",
            session_number=int(
                last_rewind.get("session_number", session_number) or 0
            ),
        )

    if last_fork is not None and last_fork_index > last_lifecycle_index:
        return LiveState(
            status="paused",
            finish_reason="forked",
            session_number=int(
                last_fork.get("session_number", session_number) or 0
            ),
        )

    if last_lifecycle is None:
        return LiveState(status="", finish_reason=None, session_number=0)

    if last_lifecycle.get("event") == "session_start":
        return LiveState(status="running", finish_reason=None, session_number=session_number)

    finish_reason_raw = last_lifecycle.get("finish_reason")
    finish_reason = str(finish_reason_raw) if finish_reason_raw is not None else None
    return LiveState(
        status=_status_from_finish_reason(finish_reason or ""),
        finish_reason=finish_reason,
        session_number=session_number,
    )


def _status_from_finish_reason(finish_reason: str) -> str:
    if finish_reason in {"stop", "model_done"}:
        return "completed"
    if finish_reason == "error":
        return "error"
    if finish_reason == "input_required":
        return "input_required"
    return "paused"


def session_turn_count(artifact_dir: Path) -> int:
    """Return the latest session_end turns count, if any."""
    events = _load_trace_events(Path(artifact_dir) / ".trace.jsonl")
    for ev in reversed(events):
        if ev.get("event") == "session_end":
            return int(ev.get("turns", 0) or 0)
    return 0


def session_trace_tail(artifact_dir: Path, *, limit: int = 10) -> list[str]:
    """Return a formatted tail of trace events for CLI inspection."""
    events = _load_trace_events(Path(artifact_dir) / ".trace.jsonl")
    if limit <= 0:
        return []
    tail = events[-limit:]
    return [_format_trace_event(ev) for ev in tail]


def session_turn_tail(artifact_dir: Path, *, limit: int = 5) -> list[str]:
    """Return a formatted tail of tool-call turns for CLI inspection."""
    events = _load_trace_events(Path(artifact_dir) / ".trace.jsonl")
    if limit <= 0:
        return []

    turns: list[dict] = []
    current: dict | None = None
    current_key: tuple[int | None, int | None] | None = None

    for ev in events:
        if ev.get("event") != "tool_call":
            continue
        key = (ev.get("session_number"), ev.get("turn_number"))
        if key != current_key:
            current = {
                "session": ev.get("session_number"),
                "turn": ev.get("turn_number"),
                "reasoning": "",
                "tools": [],
            }
            turns.append(current)
            current_key = key
        assert current is not None
        reasoning = str(ev.get("reasoning") or "").strip()
        if reasoning and not current["reasoning"]:
            current["reasoning"] = reasoning
        current["tools"].append(
            {
                "tool_name": str(ev.get("tool_name") or "?"),
                "args_summary": str(ev.get("args_summary") or ""),
                "result_summary": str(ev.get("result_summary") or ""),
                "gate_blocked": bool(ev.get("gate_blocked", False)),
            }
        )

    rendered: list[str] = []
    for turn in turns[-limit:]:
        rendered.extend(_format_turn_block(turn))
    return rendered


def session_compact_summary(artifact_dir: Path) -> dict[str, object]:
    """Return a compact operator summary derived from trace events."""
    artifact_dir = Path(artifact_dir)
    events = _load_trace_events(artifact_dir / ".trace.jsonl")
    changed_files: list[str] = []
    changed_seen: set[str] = set()
    last_test_cmd: str | None = None
    last_test_result: str | None = None
    finish_reason: str | None = None

    for ev in events:
        event_type = str(ev.get("event") or "")
        if event_type == "session_end":
            reason = ev.get("finish_reason")
            finish_reason = str(reason) if reason is not None else None
            continue
        if event_type != "tool_call":
            continue

        tool_name = str(ev.get("tool_name") or "")
        args_summary = str(ev.get("args_summary") or "")
        result_summary = str(ev.get("result_summary") or "")

        outcome = str(ev.get("outcome") or "").lower()
        pass_fail = str(ev.get("pass_fail") or "").lower()
        succeeded = (
            not bool(ev.get("gate_blocked"))
            and outcome not in {"blocked", "error"}
            and pass_fail != "fail"
        )
        if succeeded:
            structured_paths = [
                str(path)
                for path in (ev.get("source_write_paths") or [])
                if str(path)
            ]
            fallback_paths = (
                _extract_paths_from_args(args_summary)
                if tool_name in {"edit", "write", "multi_edit"}
                else []
            )
            for file_path in structured_paths or fallback_paths:
                if file_path not in changed_seen:
                    changed_seen.add(file_path)
                    changed_files.append(file_path)

        if tool_name == "run_tests":
            last_test_cmd = str(
                ev.get("action_summary") or f"run_tests({args_summary})"
            )
            traced_verdict = str(ev.get("pass_fail") or "").lower()
            last_test_result = (
                traced_verdict
                if traced_verdict in {"pass", "fail"}
                else _classify_test_outcome(result_summary)
            )
        elif tool_name == "bash":
            cmd = _extract_shell_cmd(args_summary)
            if cmd and _looks_like_test_command(cmd):
                last_test_cmd = cmd
                traced_verdict = str(ev.get("pass_fail") or "").lower()
                last_test_result = (
                    traced_verdict
                    if traced_verdict in {"pass", "fail"}
                    else _classify_test_outcome(result_summary)
                )

    cache_metrics: dict = {}
    try:
        metrics_payload = json.loads((artifact_dir / "metrics.json").read_text())
        candidate = metrics_payload.get("metrics", {}).get("prompt_cache", {})
        if isinstance(candidate, dict):
            cache_metrics = candidate
    except (OSError, json.JSONDecodeError, AttributeError):
        pass

    return {
        "changed_files": changed_files,
        "last_test_cmd": last_test_cmd,
        "last_test_result": last_test_result or "unknown",
        "finish_reason": finish_reason,
        "cache_metrics_present": bool(cache_metrics),
        "cache_hit_ratio": cache_metrics.get("cache_hit_ratio"),
        "cache_requests_observed": cache_metrics.get("requests_observed", 0),
        "cache_requests_unobserved": cache_metrics.get("requests_unobserved", 0),
    }


def _seed_session_artifacts(record: SessionRecord) -> None:
    artifact_dir = record.artifact_path
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / ".solver").mkdir(parents=True, exist_ok=True)
    (artifact_dir / "prompt.txt").write_text(record.prompt_text)
    (artifact_dir / ".solver" / "state.json").write_text(
        json.dumps(_EMPTY_STATE, indent=2) + "\n"
    )
    _write_session_metadata(record)


def _write_session_metadata(
    record: SessionRecord,
    *,
    artifact_dir: Path | None = None,
) -> None:
    artifact_dir = Path(artifact_dir or record.artifact_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config_path_hashes = {}
    for raw_path in record.config_paths:
        path = Path(raw_path)
        try:
            relative = path.relative_to(record.artifact_path)
        except ValueError:
            readable_path = path
        else:
            readable_path = artifact_dir / relative
        if readable_path.is_file():
            config_path_hashes[raw_path] = hashlib.sha256(
                readable_path.read_bytes()
            ).hexdigest()
    meta = {
        "session_id": record.session_id,
        "cwd": record.cwd,
        "model": record.model,
        "provider": record.provider,
        "authentication": record.auth_method,
        "parent_session_id": record.parent_session_id,
        "prompt_source": record.prompt_source,
        "context_mode": record.context_mode,
        "system_prompt_path": record.system_prompt_path,
        "config_paths": record.config_paths,
        "config_path_hashes": config_path_hashes,
        "worktree_path": record.worktree_path,
        "worktree_branch": record.worktree_branch,
        "worktree_base_commit": record.worktree_base_commit,
    }
    (artifact_dir / "session.json").write_text(json.dumps(meta, indent=2) + "\n")


def _record_sandbox_provenance(
    record: SessionRecord,
    resolution: SandboxResolution,
) -> None:
    """Merge the pre-model sandbox resolution into assistant session metadata."""
    path = record.artifact_path / "session.json"
    try:
        metadata = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        _write_session_metadata(record)
        metadata = json.loads(path.read_text())
    metadata["sandbox"] = resolution.as_dict()
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def session_sandbox_provenance(artifact_dir: Path) -> dict[str, object] | None:
    """Return the latest trace-backed sandbox identity for status commands."""
    events = _load_trace_events(Path(artifact_dir) / ".trace.jsonl")
    for event in reversed(events):
        if event.get("event") != "session_start":
            continue
        if "sandbox_resolved" not in event:
            continue
        return {
            "selected": event.get("sandbox_selected", event.get("sandbox_backend")),
            "resolved": event.get("sandbox_resolved", event.get("sandbox_backend")),
            "engaged": bool(event.get("sandbox_engaged")),
            "explicit_unsandboxed": bool(
                event.get("sandbox_explicit_unsandboxed", False)
            ),
        }
    for event in reversed(events):
        if event.get("event") != "runtime_envelope":
            continue
        return {
            "selected": event.get("sandbox_selected", event.get("sandbox_backend")),
            "resolved": event.get("sandbox_resolved", event.get("sandbox_backend")),
            "engaged": bool(event.get("sandbox_engaged")),
            "explicit_unsandboxed": bool(
                event.get("sandbox_explicit_unsandboxed", False)
            ),
        }
    try:
        metadata = json.loads(
            (Path(artifact_dir) / "session.json").read_text()
        )
    except (OSError, json.JSONDecodeError):
        return None
    sandbox = metadata.get("sandbox")
    return sandbox if isinstance(sandbox, dict) else None


def _record_auth_binding(record: SessionRecord) -> AuthBinding | None:
    values = (record.provider, record.auth_method, record.credential_id)
    if not any(values):
        return None
    if not all(values):
        raise RuntimeError("saved session has incomplete authentication identity")
    return AuthBinding(
        provider=str(record.provider),
        auth_method=str(record.auth_method),
        credential_id=str(record.credential_id),
    )


def _format_trace_event(event: dict) -> str:
    et = str(event.get("event") or "?")
    if et == "session_start":
        session = event.get("session_number")
        return f"session_start session={session}"
    if et == "tool_call":
        turn = event.get("turn_number")
        tool = event.get("tool_name") or "?"
        args = _truncate_text(str(event.get("args_summary") or ""), 100)
        result = _truncate_text(str(event.get("result_summary") or ""), 120)
        return f"tool_call turn={turn} {tool}({args}) => {result}"
    if et == "session_end":
        session = event.get("session_number")
        finish_reason = event.get("finish_reason")
        turns = event.get("turns")
        return f"session_end session={session} finish_reason={finish_reason} turns={turns}"
    if et == "session_fork":
        child = event.get("session_id") or "?"
        parent = event.get("parent_session_id") or "?"
        return f"session_fork child={child} parent={parent}"
    if et == "adaptive_phase_switch":
        turn = event.get("turn_number")
        phase = event.get("phase")
        return f"adaptive_phase_switch turn={turn} phase={phase}"
    if et == "approval_request":
        turn = event.get("turn_number")
        tool = event.get("tool_name") or "?"
        reason = event.get("reason") or ""
        args = _truncate_text(str(event.get("args_summary") or ""), 100)
        return f"approval_request turn={turn} {tool}({args}) reason={reason}"
    if et == "clarification_request":
        turn = event.get("turn_number")
        request_id = event.get("request_id") or "?"
        question = _truncate_text(str(event.get("question") or ""), 160)
        return (
            f"clarification_request turn={turn} request={request_id} "
            f"question={question}"
        )
    if et in {"clarification_answer", "clarification_consumed"}:
        request_id = event.get("request_id") or "?"
        return f"{et} request={request_id}"
    if et == "correction_created":
        correction_id = event.get("correction_id") or "?"
        chars = event.get("text_chars")
        return f"correction_created correction={correction_id} chars={chars}"
    if et in {"correction_consumed", "correction_replayed"}:
        correction_id = event.get("correction_id") or "?"
        return f"{et} correction={correction_id}"
    if et == "operator_followup":
        session = event.get("session_number")
        source = event.get("prompt_source") or "unknown"
        chars = event.get("text_chars")
        return f"operator_followup session={session} source={source} chars={chars}"
    if et == "regression":
        n_regressed = event.get("n_regressed")
        return f"regression n_regressed={n_regressed}"
    summary = ", ".join(
        f"{key}={value}"
        for key, value in event.items()
        if key != "event"
    )
    return f"{et} {summary}".strip()


def _format_turn_block(turn: dict) -> list[str]:
    session = turn.get("session")
    turn_number = turn.get("turn")
    header = f"turn {turn_number}"
    if session is not None:
        header += f" (session {session})"
    lines = [header]

    reasoning = _truncate_text(str(turn.get("reasoning") or ""), 160)
    if reasoning:
        lines.append(f"  reasoning: {reasoning}")

    for tool in turn.get("tools", []):
        action = f"{tool['tool_name']}({tool['args_summary']})"
        result = _truncate_text(tool["result_summary"], 160)
        if tool["gate_blocked"]:
            lines.append(f"  blocked: {action}")
        else:
            lines.append(f"  tool: {action}")
        if result:
            lines.append(f"    result: {result}")
    return lines


def _truncate_text(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _extract_paths_from_args(args_summary: str) -> list[str]:
    # args_summary is already a compact shell-safe string; simple path heuristics are enough.
    return [match for match in re.findall(r"path='([^']+)'", args_summary) if match]


def _extract_shell_cmd(args_summary: str) -> str:
    match = re.search(r"cmd='([^']*)'", args_summary)
    if match is None:
        return ""
    return match.group(1).strip()


def _looks_like_test_command(cmd: str) -> bool:
    tokens = cmd.lower()
    probes = (
        "pytest",
        "go test",
        "cargo test",
        "npm test",
        "pnpm test",
        "yarn test",
        "ctest",
        "nosetests",
        "unittest",
    )
    return any(probe in tokens for probe in probes)


def _classify_test_outcome(result_summary: str) -> str:
    lowered = result_summary.lower()
    if "[exit code: 0]" in lowered or " passed" in lowered or "1 passed" in lowered:
        return "pass"
    if "[exit code: 1]" in lowered or " failed" in lowered or "error" in lowered:
        return "fail"
    return "unknown"


def approval_request_path(artifact_dir: Path) -> Path:
    return Path(artifact_dir) / _APPROVAL_REQUEST_FILE


def approval_decisions_path(artifact_dir: Path) -> Path:
    return Path(artifact_dir) / _APPROVAL_DECISIONS_FILE


def interrupt_marker_path(artifact_dir: Path) -> Path:
    return Path(artifact_dir) / _INTERRUPT_MARKER_FILE


def load_approval_request(artifact_dir: Path) -> dict | None:
    path = approval_request_path(artifact_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def save_approval_request(artifact_dir: Path, payload: dict) -> None:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    approval_request_path(artifact_dir).write_text(json.dumps(payload, indent=2) + "\n")


def load_approval_decisions(artifact_dir: Path) -> dict:
    path = approval_decisions_path(artifact_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_approval_decisions(artifact_dir: Path, payload: dict) -> None:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    approval_decisions_path(artifact_dir).write_text(json.dumps(payload, indent=2) + "\n")


def load_interrupt_marker(artifact_dir: Path) -> dict | None:
    path = interrupt_marker_path(artifact_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def save_interrupt_marker(artifact_dir: Path, payload: dict) -> None:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    interrupt_marker_path(artifact_dir).write_text(json.dumps(payload, indent=2) + "\n")


def clear_interrupt_marker(artifact_dir: Path) -> None:
    path = interrupt_marker_path(artifact_dir)
    try:
        path.unlink()
    except FileNotFoundError:
        return


def mark_session_interrupted(artifact_dir: Path) -> None:
    save_interrupt_marker(
        artifact_dir,
        {
            "finish_reason": "interrupted",
            "interrupted_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _default_model(config_paths: list[Path], *, config_overrides: dict | None = None) -> str:
    overrides = {"runtime_mode": "assistant", "max_sessions": 1}
    if config_overrides:
        overrides.update(config_overrides)
    cfg = load_config(
        user_config=config_paths,
        overrides=overrides,
    )
    return cfg.model


def resolve_served_model(
    config_paths: list[Path],
    requested_model: str | None = None,
    config_overrides: dict | None = None,
    *,
    auth_binding: AuthBinding | None = None,
    auth_store: CredentialStore | None = None,
) -> tuple[str, list[str]]:
    """Resolve an exact served model id against ``/v1/models``.

    Alias/default is reconciled against the live server list: if the
    alias-resolved id is served verbatim it wins, otherwise the first
    served id is used. Raises RuntimeError if the server returns no
    models. Shared by ``run`` and ``smoke``.
    """
    overrides = {"runtime_mode": "assistant", "max_sessions": 1}
    if config_overrides:
        overrides.update(config_overrides)
    base_model = resolve_model(requested_model) if requested_model else _default_model(
        config_paths,
        config_overrides=overrides,
    )
    overrides["model"] = base_model
    cfg = load_config(
        user_config=config_paths,
        overrides=overrides,
    )
    profile = _load_profile(cfg)
    client = _make_client(
        cfg,
        profile,
        auth_binding=auth_binding,
        auth_store=auth_store,
    )
    served = client.health_check()
    if not served:
        raise RuntimeError("server returned no models from /v1/models")
    if base_model in served:
        return base_model, served
    if auth_binding is not None:
        return base_model, served
    if requested_model and _is_remote_transport(config_overrides):
        return base_model, served
    return served[0], served


def resolve_smoke_model(
    config_paths: list[Path],
    requested_model: str | None = None,
    config_overrides: dict | None = None,
    *,
    auth_binding: AuthBinding | None = None,
    auth_store: CredentialStore | None = None,
) -> tuple[str, list[str]]:
    """Backwards-compatible alias for ``resolve_served_model``."""
    return resolve_served_model(
        config_paths,
        requested_model=requested_model,
        config_overrides=config_overrides,
        auth_binding=auth_binding,
        auth_store=auth_store,
    )


def _is_remote_transport(config_overrides: dict | None) -> bool:
    if not config_overrides:
        return False
    provider = config_overrides.get("provider")
    if provider == "anthropic":
        return True
    return "base_url" in config_overrides


def validate_image_capability(
    config_paths: list[Path],
    *,
    model: str,
    config_overrides: dict | None = None,
    auth_binding: AuthBinding | None = None,
) -> None:
    """Fail closed unless the selected transport and model accept images."""
    overrides = {
        "runtime_mode": "assistant",
        "max_sessions": 1,
        "model": model,
        **(config_overrides or {}),
    }
    cfg = load_config(user_config=config_paths, overrides=overrides)
    require_runtime_mode(cfg, expected="assistant", caller="scripts.llm_assist")
    _require_image_capability(
        cfg,
        _load_profile(cfg),
        auth_binding=auth_binding,
    )


def _require_image_capability(
    cfg,
    profile,
    *,
    auth_binding: AuthBinding | None,
) -> None:
    provider = _image_capability_provider(cfg, auth_binding)
    profile_support = bool(
        getattr(profile, "supports_image_inputs", False)
    )
    if model_supports_image_inputs(
        cfg.model,
        provider=provider,
        profile_supports_image_inputs=profile_support,
    ):
        return
    raise ImageInputError(
        f"model {cfg.model!r} on provider {provider!r} "
        "does not declare image input support"
    )


def _image_capability_provider(
    cfg,
    auth_binding: AuthBinding | None,
) -> str:
    if auth_binding is not None:
        return "anthropic" if auth_binding.provider == "claude" else "openai"
    if getattr(cfg, "provider", "") == "anthropic":
        return "anthropic"
    hostname = (urlparse(str(getattr(cfg, "base_url", ""))).hostname or "").lower()
    if hostname == "api.openai.com":
        return "openai"
    return "openai-compatible"


def _load_profile(cfg):
    profiles_dir = PROJECT_ROOT / "profiles"
    if not profiles_dir.is_dir():
        return None
    profile_key = getattr(cfg, "profile_name", "") or cfg.model
    try:
        return load_profile(profile_key, profiles_dir)
    except FileNotFoundError:
        return None


def _apply_effective_context(cfg, client):
    server_ctx = client.query_server_context()
    if server_ctx:
        effective_ctx = min(cfg.context_size, server_ctx) if cfg.context_size > 0 else server_ctx
        if effective_ctx != cfg.context_size:
            cfg = replace(cfg, context_size=effective_ctx)

    token_budget = int(cfg.context_size * cfg.context_fill_ratio)
    derived_recent = int(token_budget * 0.45 * 4)
    derived_output = int(token_budget * 0.40 * 4)
    if derived_recent != cfg.recent_tool_results_chars or derived_output != cfg.max_output_chars:
        cfg = replace(
            cfg,
            recent_tool_results_chars=derived_recent,
            max_output_chars=derived_output,
        )

    # The config loader leaves max_tokens=0 as a placeholder; the
    # measurement entrypoint (scripts.llm_solver.__main__) derives the real
    # value after server-context resolution. The assistant path must do the
    # same or every request goes out with max_tokens=0 and dies at one
    # generated token (finish_reason=length).
    derived_max_tokens = int(cfg.context_size * cfg.max_tokens_fraction)
    if derived_max_tokens != cfg.max_tokens:
        cfg = replace(cfg, max_tokens=derived_max_tokens)
    return cfg


def _status_from_result(success: bool, finish_reason: str | None) -> str:
    if success:
        return "completed"
    if finish_reason == "error":
        return "error"
    if finish_reason == "input_required":
        return "input_required"
    return "paused"


def override_port(port: int) -> dict[str, str]:
    """Shared port override helper for CLI callers."""
    parsed = urlparse(get_server_base_url())
    netloc = f"{parsed.hostname or 'localhost'}:{port}"
    return {"base_url": urlunparse(parsed._replace(netloc=netloc))}


__all__ = [
    "approval_decisions_path",
    "approval_request_path",
    "clear_interrupt_marker",
    "create_session",
    "derive_live_state",
    "interrupt_marker_path",
    "last_finish_reason",
    "LiveState",
    "load_approval_request",
    "load_approval_decisions",
    "load_interrupt_marker",
    "mark_session_interrupted",
    "prepare_smoke_repo",
    "run_session",
    "resolve_served_model",
    "resolve_smoke_model",
    "save_approval_request",
    "save_approval_decisions",
    "save_interrupt_marker",
    "session_compact_summary",
    "session_trace_tail",
    "session_turn_tail",
    "session_turn_count",
]
