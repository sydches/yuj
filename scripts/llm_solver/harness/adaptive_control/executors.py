"""TOML overlay apply surface for adaptive hurdle control.

The harness is configured by TOML. Live adaptive control uses that same
surface: compose the baseline config paths with the
selected candidate knob TOML, replace ``session.cfg``, and refresh copied
runtime surfaces that can be updated in place.
"""
from __future__ import annotations

from collections import deque
import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ...config import dump_config, load_config, resolve_project_path
from .config_transaction import (
    apply_candidate_delta,
    commit_config,
    resolved_baseline_cfg,
)
from .schema import ExecutorResult, InterventionPayload

TOML_OVERLAY_EXECUTOR_ID = "toml_overlay.apply"
TOML_OVERLAY_RESTORE_EXECUTOR_ID = "toml_overlay.restore_baseline"


@dataclass(frozen=True)
class ExecutorSpec:
    executor_id: str
    timing_class: str
    allowed_fields: tuple[str, ...]
    status: str = "implemented"


REGISTRY: dict[str, ExecutorSpec] = {
    TOML_OVERLAY_EXECUTOR_ID: ExecutorSpec(
        TOML_OVERLAY_EXECUTOR_ID,
        "toml_overlay",
        ("baseline_config_paths", "candidate_config_path"),
        status="implemented",
    ),
    TOML_OVERLAY_RESTORE_EXECUTOR_ID: ExecutorSpec(
        TOML_OVERLAY_RESTORE_EXECUTOR_ID,
        "toml_overlay",
        ("baseline_config_paths",),
        status="implemented",
    ),
}

_BLOCKED_FIELD_NAMES = {
    "api_key",
    "base_url",
    "context_size",
    "health_poll_interval",
    "health_timeout",
    "launch_timeout",
    "max_sessions",
    "max_tokens",
    "max_tokens_fraction",
    "max_turns",
    "model",
    "profile_name",
    "timeout_connect",
    "timeout_read",
    "tokenizer_id",
}

_CONTEXT_ATTR_REFRESH = {
    "args_summary_chars": ("_args_summary_chars",),
    "compound_selective_recent_tool_results_chars": ("_selective_recent_tool_results_chars",),
    "compound_selective_resolved_action_repeat_cap": ("_selective_resolved_action_repeat_cap",),
    "compound_selective_resolved_evidence_lines": ("_selective_resolved_evidence_lines",),
    "compound_selective_resolved_evidence_stub_chars": ("_selective_resolved_evidence_stub_chars",),
    "compound_selective_resolved_source_anchor_lines": ("_selective_resolved_source_anchor_lines",),
    "compound_selective_resolved_test_anchor_lines": ("_selective_resolved_test_anchor_lines",),
    "compound_selective_trace_action_repeat_cap": ("_selective_trace_action_repeat_cap",),
    "compound_selective_trace_anchor_lines": ("_selective_trace_anchor_lines",),
    "compound_selective_trace_lines": ("_selective_trace_lines",),
    "compound_selective_trace_source_anchor_lines": ("_selective_trace_source_anchor_lines",),
    "compound_selective_trace_test_anchor_lines": ("_selective_trace_test_anchor_lines",),
    "compound_selective_unresolved_evidence_lines": ("_selective_unresolved_evidence_lines",),
    "context_ignore_state": ("_ignore_state",),
    "focused_compound_evidence_lines": ("_focused_evidence_lines",),
    "focused_compound_include_resolved_evidence": ("_focused_include_resolved_evidence",),
    "focused_compound_recent_tool_results_chars": ("_focused_recent_tool_results_chars",),
    "focused_compound_trace_lines": ("_focused_trace_lines",),
    "halflife_cap_15_chars": ("_cap_15_chars",),
    "halflife_cap_31_chars": ("_cap_31_chars",),
    "halflife_cap_63_chars": ("_cap_63_chars",),
    "halflife_cap_7_chars": ("_cap_7_chars",),
    "halflife_cap_older_chars": ("_cap_older_chars",),
    "halflife_context_limit_tokens": ("_context_limit_tokens",),
    "halflife_no_decay_ratio": ("_activation_ratio",),
    "halflife_verbatim_tool_results": ("_verbatim_tool_results",),
    "min_turns_before_context": ("_min_turns",),
    "recent_tool_results_chars": ("_recent_tool_results_chars", "_recent_results_chars"),
    "solver_evidence_lines": ("_evidence_lines",),
    "solver_inference_lines": ("_inference_lines",),
    "solver_trace_lines": ("_trace_lines",),
    "state_context_suffix": ("_suffix",),
    "trace_reasoning_chars": ("_trace_reasoning_chars",),
    "trace_stub_chars": ("_trace_stub_chars",),
}

_CONTEXT_CACHE_ATTRS = (
    "_msg_cache",
    "_tok_cache",
    "_file_cache",
    "_raw_state_cache",
)

_GUARD_STATE_REFRESH_FIELDS = {
    "duplicate_abort",
    "rumination_enabled",
    "rumination_gate_arm_threshold",
    "rumination_gate_arm_threshold_abs",
    "rumination_min_threshold",
    "rumination_nudge_threshold",
    "rumination_nudge_threshold_abs",
    "rumination_nudge_threshold_abs_post_mutation",
}

_GUARD_STATE_DERIVED_ATTRS = (
    "rumination_nudge_threshold",
    "rumination_nudge_threshold_post_mutation",
    "rumination_arm_threshold",
)


def diagnose_apply(executor_id: str) -> tuple[str, str]:
    spec = REGISTRY.get(executor_id)
    if spec is None:
        return "blocked", "missing_executor"
    if spec.status != "implemented":
        return "blocked", "executor_not_implemented"
    return "applied", ""


def _resolve_path(raw: str) -> Path:
    return resolve_project_path(raw)


def _normalized_paths(paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(_resolve_path(p)) for p in paths if str(p or "").strip())


def _config_digest(cfg) -> str:
    data = dump_config(cfg) if dataclasses.is_dataclass(cfg) else dict(vars(cfg))
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _changed_fields(old_cfg, new_cfg) -> set[str]:
    old = dataclasses.asdict(old_cfg) if dataclasses.is_dataclass(old_cfg) else dict(vars(old_cfg))
    new = dataclasses.asdict(new_cfg) if dataclasses.is_dataclass(new_cfg) else dict(vars(new_cfg))
    return {k for k in new if old.get(k) != new.get(k)}


_PRESERVE_LAUNCH_FIELDS = (
    "api_key",
    "base_url",
    "context_size",
    "health_poll_interval",
    "health_timeout",
    "launch_timeout",
    "max_sessions",
    "max_tokens",
    "max_tokens_fraction",
    "max_turns",
    "model",
    "profile_name",
    "timeout_connect",
    "timeout_read",
    "tokenizer_id",
)


def _preserve_runtime_derived_fields(old_cfg, new_cfg):
    values = {}
    for field in _PRESERVE_LAUNCH_FIELDS:
        if hasattr(old_cfg, field):
            values[field] = getattr(old_cfg, field)
    return dataclasses.replace(new_cfg, **values) if values else new_cfg


def _context_refresh_targets(changed: set[str], context) -> tuple[str, ...]:
    """Return changed config fields copied by the current context instance.

    The config object itself is swapped atomically after validation. Some
    fields are then read dynamically from ``session.cfg`` on later turns;
    other fields were copied into the context object at construction time.
    Only the latter need an attribute refresh here. If the active context
    does not expose any mapped attribute for a config field, that field is not
    a copied surface for this context mode and must not block TOML apply.
    """
    targets: list[str] = []
    for field in sorted(changed):
        attrs = _CONTEXT_ATTR_REFRESH.get(field)
        if attrs is None:
            continue
        if any(hasattr(context, attr) for attr in attrs):
            targets.append(field)
    return tuple(targets)


def _refresh_context(context, new_cfg, targets: tuple[str, ...]) -> tuple[str, ...]:
    refreshed: list[str] = []
    for field in targets:
        value = getattr(new_cfg, field)
        for attr in _CONTEXT_ATTR_REFRESH[field]:
            if hasattr(context, attr):
                setattr(context, attr, value)
                refreshed.append(f"context.{attr}")
    for attr in _CONTEXT_CACHE_ATTRS:
        if hasattr(context, attr):
            setattr(context, attr, None)
    return tuple(refreshed)


def _prepare_tool_schemas(session, new_cfg, changed: set[str]):
    tool_fields = {
        "tool_desc",
        "tools_run_tests_enabled",
        "tools_list_definitions_enabled",
        "tools_apply_patch_enabled",
    }
    if not (changed & tool_fields):
        return None
    if not all(hasattr(session, attr) for attr in ("client", "_tool_registry")):
        return None
    from .._loop.profile_resolution import apply_profile_to_schemas
    from ..schemas import get_tool_schemas
    from ..tool_validation import ToolSchemaSet
    from ..tools import validate_tool_handlers

    schemas = apply_profile_to_schemas(get_tool_schemas(new_cfg.tool_desc), new_cfg, session.client)
    schema_names = [s["function"]["name"] for s in schemas]
    validate_tool_handlers(schema_names, registry=session._tool_registry)
    return schemas, ToolSchemaSet.from_openai_tools(schemas)


def _prepare_guardrail_state(session, new_cfg, changed: set[str]):
    if not (changed & _GUARD_STATE_REFRESH_FIELDS):
        return None
    state = getattr(session, "_guards", None)
    if state is None:
        return None

    from .._guardrails.state import init_guardrail_state

    fresh = init_guardrail_state(new_cfg)
    recent_calls = None
    if "duplicate_abort" in changed and hasattr(state, "recent_calls"):
        maxlen = fresh.recent_calls.maxlen
        recent = list(state.recent_calls)
        if maxlen is not None:
            recent = recent[-maxlen:]
        recent_calls = deque(recent, maxlen=maxlen)
    return state, fresh, recent_calls


def _prepare_permission_policy(new_cfg, changed: set[str]):
    if not changed & {"permissions_rules", "permissions_ask_fallback"}:
        return None
    from ..tool_policy import PermissionPolicy, normalize_ask_fallback

    normalize_ask_fallback(new_cfg.permissions_ask_fallback)
    return PermissionPolicy.from_rule_tables(new_cfg.permissions_rules)


def _commit_guardrail_state(prepared) -> tuple[str, ...]:
    if prepared is None:
        return ()
    state, fresh, recent_calls = prepared
    for attr in _GUARD_STATE_DERIVED_ATTRS:
        setattr(state, attr, getattr(fresh, attr))
    if recent_calls is not None:
        state.recent_calls = recent_calls
    return ("guard_state",)


def _refresh_runtime_surfaces(
    session, old_cfg, new_cfg, changed: set[str],
) -> tuple[bool, str, tuple[str, ...], tuple[str, ...]]:
    blocked = sorted(changed & _BLOCKED_FIELD_NAMES)
    if blocked:
        return False, "config_refresh_not_declared", (), tuple(blocked)

    refreshed: list[str] = []
    context = getattr(session, "context", None)
    targets = _context_refresh_targets(changed, context) if context is not None else ()

    try:
        schemas = _prepare_tool_schemas(session, new_cfg, changed)
        guard_state = _prepare_guardrail_state(session, new_cfg, changed)
        permission_policy = _prepare_permission_policy(new_cfg, changed)
    except Exception:
        return False, "runtime_surface_refresh_failed", (), tuple(sorted(changed))

    if context is not None:
        refreshed.extend(_refresh_context(context, new_cfg, targets))
    if schemas is not None:
        session._tool_schemas, session._tool_schema_set = schemas
        refreshed.append("tool_schemas")
    if permission_policy is not None:
        session._permission_policy = permission_policy
        refreshed.append("permission_policy")
    refreshed.extend(_commit_guardrail_state(guard_state))

    if changed & {f for f in changed if f.startswith("loop_") or f.endswith("_enabled")}:
        refreshed.append("session.cfg")
    elif changed:
        refreshed.append("session.cfg")
    return True, "", tuple(dict.fromkeys(refreshed)), ()


def _baseline_config_paths(session, payload: InterventionPayload) -> tuple[str, ...]:
    if payload.baseline_config_paths:
        return _normalized_paths(payload.baseline_config_paths)
    paths = getattr(session, "adaptive_control_baseline_config_paths", ()) or ()
    if paths:
        return _normalized_paths(paths)
    cfg = getattr(session, "cfg", None)
    return _normalized_paths(getattr(cfg, "adaptive_control_baseline_config_paths", ()) or ())


def apply(session, payload: InterventionPayload) -> ExecutorResult:
    if payload.executor_id != TOML_OVERLAY_EXECUTOR_ID:
        status, reason = diagnose_apply(payload.executor_id)
        return ExecutorResult(payload.executor_id, applied=False, blocked_reason=reason)

    old_cfg = getattr(session, "cfg", None)
    baseline_paths = _baseline_config_paths(session, payload)
    candidate = (
        payload.candidate_config_path
        or payload.fields.get("candidate_config_path", "")
        or getattr(old_cfg, "adaptive_control_candidate_config_path", "")
    )
    candidate_path = str(_resolve_path(candidate)) if candidate else ""
    pre_digest = _config_digest(old_cfg) if old_cfg is not None else ""

    if not baseline_paths:
        return ExecutorResult(
            payload.executor_id, applied=False, pre_digest=pre_digest,
            blocked_reason="missing_baseline_config",
            baseline_config_paths=baseline_paths, candidate_config_path=candidate_path,
            active_config_basis="blocked",
        )
    if not candidate_path:
        return ExecutorResult(
            payload.executor_id, applied=False, pre_digest=pre_digest,
            blocked_reason="missing_config_overlay",
            baseline_config_paths=baseline_paths, candidate_config_path=candidate_path,
            active_config_basis="blocked",
        )
    all_paths = (*baseline_paths, candidate_path)
    if any(not Path(p).is_file() for p in all_paths):
        return ExecutorResult(
            payload.executor_id, applied=False, pre_digest=pre_digest,
            blocked_reason="config_overlay_invalid",
            baseline_config_paths=baseline_paths, candidate_config_path=candidate_path,
            applied_config_paths=all_paths, active_config_basis="blocked",
        )

    try:
        raw_baseline_cfg = load_config(
            user_config=[Path(p) for p in baseline_paths],
        )
        raw_candidate_cfg = load_config(
            user_config=[Path(p) for p in all_paths],
            strict_dial_gates=True,
        )
    except Exception:
        return ExecutorResult(
            payload.executor_id, applied=False, pre_digest=pre_digest,
            blocked_reason="config_overlay_invalid",
            baseline_config_paths=baseline_paths, candidate_config_path=candidate_path,
            applied_config_paths=all_paths, active_config_basis="blocked",
        )

    resolved_baseline = resolved_baseline_cfg(session, old_cfg)
    new_cfg = apply_candidate_delta(
        resolved_baseline, raw_baseline_cfg, raw_candidate_cfg,
    )
    new_cfg = _preserve_runtime_derived_fields(old_cfg, new_cfg)
    changed = _changed_fields(old_cfg, new_cfg)
    ok, reason, refreshed, blocked_fields = _refresh_runtime_surfaces(
        session, old_cfg, new_cfg, changed)
    if not ok:
        return ExecutorResult(
            payload.executor_id, applied=False, pre_digest=pre_digest,
            blocked_reason=reason, baseline_config_paths=baseline_paths,
            candidate_config_path=candidate_path, applied_config_paths=all_paths,
            active_config_basis="blocked",
            changed_config_fields=tuple(sorted(changed)),
            blocked_config_fields=blocked_fields,
        )

    refreshed = (*refreshed, *commit_config(session, new_cfg))
    post_digest = _config_digest(new_cfg)
    return ExecutorResult(
        payload.executor_id,
        applied=True,
        pre_digest=pre_digest,
        post_digest=post_digest,
        baseline_config_paths=baseline_paths,
        candidate_config_path=candidate_path,
        applied_config_paths=all_paths,
        active_config_basis="baseline_plus_candidate",
        refreshed_surfaces=tuple(dict.fromkeys(refreshed)),
        changed_config_fields=tuple(sorted(changed)),
    )


def restore_baseline(session, baseline_config_paths: Iterable[str] | None = None) -> ExecutorResult:
    """Restore the live control basis to the immutable baseline config only."""
    old_cfg = getattr(session, "cfg", None)
    if baseline_config_paths:
        baseline_paths = _normalized_paths(baseline_config_paths)
    else:
        baseline_paths = _baseline_config_paths(
            session,
            InterventionPayload("", TOML_OVERLAY_RESTORE_EXECUTOR_ID, ""),
        )
    pre_digest = _config_digest(old_cfg) if old_cfg is not None else ""

    if not baseline_paths:
        return ExecutorResult(
            TOML_OVERLAY_RESTORE_EXECUTOR_ID,
            applied=False,
            pre_digest=pre_digest,
            blocked_reason="missing_baseline_config",
            baseline_config_paths=baseline_paths,
            active_config_basis="blocked",
        )
    if any(not Path(p).is_file() for p in baseline_paths):
        return ExecutorResult(
            TOML_OVERLAY_RESTORE_EXECUTOR_ID,
            applied=False,
            pre_digest=pre_digest,
            blocked_reason="config_overlay_invalid",
            baseline_config_paths=baseline_paths,
            applied_config_paths=baseline_paths,
            active_config_basis="blocked",
        )

    try:
        load_config(user_config=[Path(p) for p in baseline_paths])
    except Exception:
        return ExecutorResult(
            TOML_OVERLAY_RESTORE_EXECUTOR_ID,
            applied=False,
            pre_digest=pre_digest,
            blocked_reason="config_overlay_invalid",
            baseline_config_paths=baseline_paths,
            applied_config_paths=baseline_paths,
            active_config_basis="blocked",
        )

    new_cfg = resolved_baseline_cfg(session, old_cfg)
    new_cfg = _preserve_runtime_derived_fields(old_cfg, new_cfg)
    changed = _changed_fields(old_cfg, new_cfg)
    ok, reason, refreshed, blocked_fields = _refresh_runtime_surfaces(
        session, old_cfg, new_cfg, changed)
    if not ok:
        return ExecutorResult(
            TOML_OVERLAY_RESTORE_EXECUTOR_ID,
            applied=False,
            pre_digest=pre_digest,
            blocked_reason=reason,
            baseline_config_paths=baseline_paths,
            applied_config_paths=baseline_paths,
            active_config_basis="blocked",
            changed_config_fields=tuple(sorted(changed)),
            blocked_config_fields=blocked_fields,
        )

    refreshed = (*refreshed, *commit_config(session, new_cfg))
    post_digest = _config_digest(new_cfg)
    return ExecutorResult(
        TOML_OVERLAY_RESTORE_EXECUTOR_ID,
        applied=True,
        pre_digest=pre_digest,
        post_digest=post_digest,
        baseline_config_paths=baseline_paths,
        applied_config_paths=baseline_paths,
        active_config_basis="baseline",
        refreshed_surfaces=tuple(dict.fromkeys(refreshed)),
        changed_config_fields=tuple(sorted(changed)),
    )


# ── stop_resume delivery ────────────────────────────────────────────────
#
# Instead of swapping session.cfg mid-run, write a machine-readable
# stop-note to the telemetry dir and ask the loop to end the session
# gracefully. An orchestrator reads the note and resumes the run with the
# noted rung. The controller algorithm is unchanged; only the handoff differs.

STOP_RESUME_EXECUTOR_ID = "stop_resume.apply"
STOP_NOTE_NAME = "adaptive_stop_note.json"

REGISTRY[STOP_RESUME_EXECUTOR_ID] = ExecutorSpec(
    STOP_RESUME_EXECUTOR_ID,
    "stop_resume",
    ("candidate_config_path",),
    status="implemented",
)


# ── user_turn delivery ──────────────────────────────────────────────────
#
# Deliver the controller decision as a synthetic user message at
# the next turn boundary, with the rung overlay applied in place — no
# session end and no KV-cache loss. Compose the message from live verdict
# fields.

USER_TURN_EXECUTOR_ID = "user_turn.apply"

REGISTRY[USER_TURN_EXECUTOR_ID] = ExecutorSpec(
    USER_TURN_EXECUTOR_ID,
    "user_turn",
    ("candidate_config_path",),
    status="implemented",
)

# Fixed one-sentence message parts.
_UT_SUGGESTION = {
    "reread_slump": ("stop re-reading and re-verifying things you have "
                     "already confirmed; make your next action a source "
                     "edit or a test run"),
    "repeat_wall": ("do not repeat the same command again; take a "
                    "different action toward the task"),
}
_UT_GUARD_BY_RUNG = {
    1: "loop_detect — it will warn you if identical calls repeat",
    2: ("duplicate_guard — repeating an identical call now draws a "
        "warning and can end the session"),
    3: ("loop_detect recovery guidance — repeated calls now get explicit "
        "recovery instructions"),
    4: ("unified tool-result envelope — tool results now carry explicit "
        "status fields"),
    5: ("intent gate — tool calls without a stated intent are now "
        "rejected"),
}


def compose_user_turn_message(session, *, evidence: str, rung: int,
                              hurdle_family: str, turn: int,
                              include_guard: bool = True) -> str:
    """Build a user-turn message from live session state.

    ``include_guard=False`` drops the
    "A guard is now active" sentence — no overlay is applied in that
    mode, so the sentence would be false.
    """
    ev = evidence.split(";", 1)[0]
    if ":" in ev and ev.split(":", 1)[0].startswith("T"):
        ev = ev.split(":", 1)[1]
    last_edit = None
    for event in getattr(session, "_trace_events", []) or []:
        t = event.get("turn_number")
        if t is None or int(t) > int(turn):
            continue
        if event.get("source_write_like"):
            last_edit = int(t) if last_edit is None else max(last_edit, int(t))
    edit_line = (f"Your last source edit was at turn {last_edit}."
                 if last_edit is not None
                 else "You have not made any source edits yet.")
    sug = _UT_SUGGESTION.get(hurdle_family,
                             "change your approach and take a different "
                             "action toward the task")
    guard_sentence = ""
    if include_guard:
        guard = _UT_GUARD_BY_RUNG.get(int(rung) or 1, _UT_GUARD_BY_RUNG[1])
        guard_sentence = f"A guard is now active: {guard}. "
    return (f"The session was stopped because a problem was detected: "
            f"{ev}. {edit_line} Suggestion: {sug}. "
            f"{guard_sentence}"
            f"Your previous work is saved and in place. Continue.")


def user_turn_apply(session, payload: InterventionPayload,
                    *, evidence: str = "", rung: int = 0,
                    hurdle_family: str = "", turn: int = -1) -> ExecutorResult:
    """Apply the rung overlay and queue a user-turn message.

    The overlay swap reuses the in_place executor verbatim; on success the
    composed user message is parked on ``_adaptive_user_turn_pending`` and
    the loop appends it as a user-role turn before the next API call.
    """
    result = apply(session, payload)
    if not result.applied:
        return result
    msg = compose_user_turn_message(
        session, evidence=evidence, rung=rung,
        hurdle_family=hurdle_family, turn=turn,
    )
    setattr(session, "_adaptive_user_turn_pending", msg)
    return dataclasses.replace(result, executor_id=USER_TURN_EXECUTOR_ID)


# ── tool_result and message-only user_turn delivery ─────────────────────
#
# tool_result appends the message to the next tool result and applies the
# overlay. user_turn_msg_only sends the message without applying an overlay.

TOOL_RESULT_EXECUTOR_ID = "tool_result.apply"
USER_TURN_MSG_ONLY_EXECUTOR_ID = "user_turn.msg_only"

REGISTRY[TOOL_RESULT_EXECUTOR_ID] = ExecutorSpec(
    TOOL_RESULT_EXECUTOR_ID,
    "tool_result",
    ("candidate_config_path",),
    status="implemented",
)
REGISTRY[USER_TURN_MSG_ONLY_EXECUTOR_ID] = ExecutorSpec(
    USER_TURN_MSG_ONLY_EXECUTOR_ID,
    "user_turn",
    (),
    status="implemented",
)


def tool_result_apply(session, payload: InterventionPayload,
                      *, evidence: str = "", rung: int = 0,
                      hurdle_family: str = "", turn: int = -1) -> ExecutorResult:
    """Apply the overlay and append its note to the next tool result."""
    result = apply(session, payload)
    if not result.applied:
        return result
    msg = compose_user_turn_message(
        session, evidence=evidence, rung=rung,
        hurdle_family=hurdle_family, turn=turn,
    )
    setattr(session, "_adaptive_tool_note_pending", msg)
    return dataclasses.replace(result, executor_id=TOOL_RESULT_EXECUTOR_ID)


def user_turn_msg_only_apply(session, *, evidence: str = "", rung: int = 0,
                             hurdle_family: str = "",
                             turn: int = -1) -> ExecutorResult:
    """Send the user-turn message without applying an overlay."""
    msg = compose_user_turn_message(
        session, evidence=evidence, rung=rung,
        hurdle_family=hurdle_family, turn=turn, include_guard=False,
    )
    setattr(session, "_adaptive_user_turn_pending", msg)
    return ExecutorResult(USER_TURN_MSG_ONLY_EXECUTOR_ID, applied=True,
                          active_config_basis="baseline")


def stop_for_resume(session, payload: InterventionPayload,
                    *, evidence: str = "", rung: int = 0,
                    hurdle_family: str = "", episode_id: str = "",
                    turn: int = -1) -> ExecutorResult:
    """Write the stop-note and request a graceful session end.

    The note carries everything the orchestrator needs to build the
    resume: which rung to ride in (candidate overlay path), why (verbatim
    evidence), and where in the episode the controller stands. The loop
    checks ``session._adaptive_stop_requested`` at the turn boundary and
    ends with ``finish_reason="adaptive_stop"``.
    """
    from ..._shared.telemetry_paths import ensure_telemetry_dir
    note = {
        "turn": int(turn),
        "intervention_id": payload.intervention_id,
        "candidate_config_path": payload.candidate_config_path,
        "rung": int(rung),
        "hurdle_family": hurdle_family,
        "evidence": evidence,
        "episode_id": episode_id,
    }
    try:
        tdir = ensure_telemetry_dir(Path(getattr(session, "cwd", ".")))
        (tdir / STOP_NOTE_NAME).write_text(json.dumps(note, indent=1))
    except Exception:
        return ExecutorResult(STOP_RESUME_EXECUTOR_ID, applied=False,
                              blocked_reason="stop_note_write_failed")
    setattr(session, "_adaptive_stop_requested", note)
    # controller memory must survive the segment boundary (persistence.py)
    from .persistence import save_state
    save_state(session)
    return ExecutorResult(STOP_RESUME_EXECUTOR_ID, applied=True,
                          active_config_basis="stop_resume_pending")
