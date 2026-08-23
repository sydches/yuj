"""Assistant-mode session runner built on the shared harness engine."""
from __future__ import annotations

import json
import re
import tempfile
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
from ..llm_solver.harness._loop.model_role_runtime import build_model_role_runtime
from ..llm_solver.harness.loop import _load_trace_events
from ..llm_solver.harness.worktree_runtime import (
    WorktreeRuntimeError,
    create_session_worktree,
    inspect_session_worktree,
)
from ..llm_solver.models import resolve_model
from ..llm_solver.server import LlamaClient, load_profile
from ._anthropic import AnthropicClient
from .store import SessionRecord, SessionStore


def _make_client(cfg, profile) -> LlamaClient:
    """Pick the transport for cfg.provider; OpenAI-compatible is the default."""
    if getattr(cfg, "provider", "") == "anthropic":
        return AnthropicClient(cfg, profile=profile)
    return LlamaClient(cfg, profile=profile)


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
) -> SessionRecord:
    record = store.create_session(
        cwd=cwd,
        model=model,
        prompt_text=prompt_text,
        prompt_source=prompt_source,
        context_mode=context_mode,
        system_prompt_path=system_prompt_path,
        config_paths=config_paths,
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


def run_session(store: SessionStore, record: SessionRecord, *, resume: bool) -> tuple[bool, str | None]:
    """Run exactly one harness outer session for an assistant record."""
    artifact_dir = record.artifact_path
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
    store.update_session(record.session_id, status="running", last_finish_reason=None)

    profile = _load_profile(cfg)
    client = _make_client(cfg, profile)
    if hasattr(client, "set_session_id"):
        client.set_session_id(record.session_id)
    cfg = _apply_effective_context(cfg, client)
    client.cfg = cfg
    build_model_role_runtime(
        cfg=cfg,
        main_client=client,
        profiles_dir=PROJECT_ROOT / "profiles",
        client_factory=_make_client,
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
    finish_reason = last_finish_reason(artifact_dir)
    store.update_session(
        record.session_id,
        status=_status_from_result(success, finish_reason),
        last_finish_reason=finish_reason,
    )
    return success, finish_reason


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
    """Infer live status from ``.trace.jsonl`` + ``approval_request.json``.

    Precedence:
      1. Pending approval request → ``approval_pending``.
      2. Last lifecycle event is ``session_end`` → map finish_reason to
         ``completed`` / ``paused`` / ``error``.
      3. Last lifecycle event is ``session_start`` → ``running``.
      4. Nothing observed → empty status; caller uses the SQLite row.
    """
    artifact_dir = Path(artifact_dir)
    events = _load_trace_events(artifact_dir / ".trace.jsonl")
    approval = load_approval_request(artifact_dir)
    interrupt = load_interrupt_marker(artifact_dir)

    last_lifecycle: dict | None = None
    for ev in events:
        if ev.get("event") in {"session_start", "session_end"}:
            last_lifecycle = ev
    session_number = (
        int(last_lifecycle.get("session_number", 0) or 0)
        if last_lifecycle is not None
        else 0
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

        if tool_name in {"edit", "write", "multi_edit"}:
            for file_path in _extract_paths_from_args(args_summary):
                if file_path not in changed_seen:
                    changed_seen.add(file_path)
                    changed_files.append(file_path)

        if tool_name == "bash":
            cmd = _extract_shell_cmd(args_summary)
            if cmd and _looks_like_test_command(cmd):
                last_test_cmd = cmd
                last_test_result = _classify_test_outcome(result_summary)

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


def _write_session_metadata(record: SessionRecord) -> None:
    artifact_dir = record.artifact_path
    artifact_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "session_id": record.session_id,
        "cwd": record.cwd,
        "model": record.model,
        "prompt_source": record.prompt_source,
        "context_mode": record.context_mode,
        "system_prompt_path": record.system_prompt_path,
        "config_paths": record.config_paths,
        "worktree_path": record.worktree_path,
        "worktree_branch": record.worktree_branch,
        "worktree_base_commit": record.worktree_base_commit,
    }
    (artifact_dir / "session.json").write_text(json.dumps(meta, indent=2) + "\n")


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
    client = _make_client(cfg, profile)
    served = client.health_check()
    if not served:
        raise RuntimeError("server returned no models from /v1/models")
    if base_model in served:
        return base_model, served
    if requested_model and _is_remote_transport(config_overrides):
        return base_model, served
    return served[0], served


def resolve_smoke_model(
    config_paths: list[Path],
    requested_model: str | None = None,
    config_overrides: dict | None = None,
) -> tuple[str, list[str]]:
    """Backwards-compatible alias for ``resolve_served_model``."""
    return resolve_served_model(
        config_paths,
        requested_model=requested_model,
        config_overrides=config_overrides,
    )


def _is_remote_transport(config_overrides: dict | None) -> bool:
    if not config_overrides:
        return False
    provider = config_overrides.get("provider")
    if provider == "anthropic":
        return True
    return "base_url" in config_overrides


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
