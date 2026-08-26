"""Trace event schema — version, known event types, required fields, low-level emit."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import orjson

log = logging.getLogger(__name__)


def _dumps(entry: dict) -> str:
    """Serialize a trace entry to a JSON line.

    orjson is 5–10× faster than the stdlib for the dict shapes
    written here (str, int, bool, list, nested dict). Returns a
    string (decoded) so the trace file can stay in text-mode append
    for compatibility with existing read paths and tests that open
    the trace as text.
    """
    return orjson.dumps(entry).decode("utf-8")


# ── Trace event schema ──────────────────────────────────────────────
#
# Bumped on any non-additive change to the trace event envelope (the
# event/trace_schema_version/payload-shape contract emitted by Session._emit
# and emit_trace_event). Readers must allow unknown event types. Version 2
# bounds result_summary and adds output_sha256 and output_full_path.
TRACE_SCHEMA_VERSION = 2

@dataclass(frozen=True)
class TraceEventSpec:
    """First-class schema metadata for one trace event type."""

    event_type: str
    required_fields: frozenset[str]
    optional_fields: frozenset[str] = frozenset()


TRACE_EVENT_SPECS: tuple[TraceEventSpec, ...] = (
    TraceEventSpec(
        "session_start", frozenset({
            "session_number", "thinking_level", "sandbox_backend",
            "container_runtime", "container_image_digest",
            "ignore_file_hash", "sandbox_env_names", "edit_format",
            "repo_map_tokens",
        }), frozenset({
            "worktree_path", "worktree_branch", "worktree_base_commit",
            "project_instruction_files", "project_instruction_bytes",
            "project_instruction_imported_bytes",
            "project_instruction_resolved_bytes",
            "project_instructions_truncated", "prompt_import_tree",
            "ignore_file_names", "stream_rule_files",
            "tool_lazy_loading_enabled", "tool_active_limit", "registered_tools",
            "active_tools",
            "loaded_skills",
            "sandbox_selected", "sandbox_resolved", "sandbox_engaged",
            "sandbox_explicit_unsandboxed",
            "repo_map_refresh", "repo_map_files", "repo_map_symbols",
            "repo_map_cache_hit", "repo_map_sha256",
        })
    ),
    TraceEventSpec("session_end", frozenset({"session_number", "finish_reason"})),
    TraceEventSpec(
        "session_fork",
        frozenset({
            "session_number", "session_id", "parent_session_id",
            "forked_at", "source_artifact_sha256",
        }),
    ),
    TraceEventSpec(
        "plan_mode_enter", frozenset({"session_number", "turn"}),
        frozenset({"turn_number"}),
    ),
    TraceEventSpec(
        "plan_mode_exit",
        frozenset({"session_number", "turn", "plan_chars"}),
        frozenset({"turn_number"}),
    ),
    TraceEventSpec(
        "compaction",
        frozenset({
            "session_number",
            "turn_number",
            "tokens_before",
            "tokens_after",
            "first_kept_turn",
            "method",
            "fallback",
            "role",
            "hook",
            "hook_outcome",
        }),
    ),
    TraceEventSpec(
        "handoff",
        frozenset({
            "session_number", "tokens", "valid", "fallback", "role",
        }),
    ),
    TraceEventSpec(
        "turn",
        frozenset({
            "session_number", "turn_number", "role", "prompt_tokens",
            "cached_tokens", "cache_hit_ratio",
        }),
    ),
    TraceEventSpec(
        "session_usage",
        frozenset({
            "session_number", "scope", "input_tokens", "output_tokens",
            "cached_tokens", "cost", "quota",
        }),
    ),
    TraceEventSpec(
        "subagent",
        frozenset({
            "session_number", "turn_number", "id", "agent", "turns",
            "tokens", "result_chars",
        }),
    ),
    TraceEventSpec(
        "subagent_start",
        frozenset({
            "id", "agent", "parent_session_number", "parent_turn_number",
            "depth", "model_profile", "tools", "read_only", "max_turns",
        }),
    ),
    TraceEventSpec(
        "subagent_result",
        frozenset({
            "id", "agent", "turns", "prompt_tokens",
            "completion_tokens", "own_prompt_tokens",
            "own_completion_tokens", "tokens", "finish_reason", "done",
            "result", "result_chars", "result_sha256",
        }),
    ),
    TraceEventSpec(
        "advisor_note",
        frozenset({"session_number", "turn", "severity", "chars"}),
        frozenset({"turn_number", "ordinal", "note_sha256"}),
    ),
    TraceEventSpec(
        "model_fallback",
        frozenset({
            "session_number",
            "turn_number",
            "role",
            "from",
            "to",
            "reason",
            "from_profile",
            "to_profile",
            "from_model",
            "to_model",
            "from_context_size",
            "to_context_size",
        }),
    ),
    TraceEventSpec(
        "hook",
        frozenset({"hook_event", "command", "exit", "ms", "outcome"}),
        frozenset({
            "session_number", "turn_number", "hook_index", "matcher",
            "tool_call_id", "tool_name", "reason", "updated_input",
            "additional_context", "replayed",
        }),
    ),
    TraceEventSpec(
        "tool_call",
        frozenset({"session_number", "turn_number", "tool_name"}),
        frozenset({
            "parent_tool_call_id", "cell_inner_index", "cell_source",
            "combined_output_chars", "combined_output_bytes",
            "inner_call_count",
        }),
    ),
    TraceEventSpec(
        "tools_activated",
        frozenset({
            "session_number", "turn_number", "requested", "activated",
            "active_tools",
        }),
        frozenset({"already_active"}),
    ),
    TraceEventSpec(
        "todos",
        frozenset({"session_number", "turn_number", "todos"}),
        frozenset({"tool_call_id"}),
    ),
    TraceEventSpec(
        "checkpoint",
        frozenset({
            "session_number", "turn", "commit", "duration_ms",
            "file_count", "byte_count",
        }),
    ),
    TraceEventSpec(
        "rewind",
        frozenset({
            "session_number", "turn_number", "from_turn", "to_turn",
        }),
        frozenset({
            "reason", "commit", "rewind_count", "rewind_id", "delivery",
            "report_chars", "checkpoint_message_count", "goal", "report",
        }),
    ),
    TraceEventSpec(
        "rewind_resume",
        frozenset({
            "session_number", "rewind_id", "target_session_number",
            "to_turn", "commit",
        }),
    ),
    TraceEventSpec(
        "tool_start",
        frozenset({
            "tool_call_id", "tool_name", "session_number", "turn_number",
            "started_at", "args_summary", "intent",
        }),
    ),
    TraceEventSpec(
        "session_exit",
        frozenset({
            "session_number", "reason", "kind", "recorded_at",
            "pending_tool_calls",
        }),
    ),
    TraceEventSpec(
        "turn_aborted",
        frozenset({
            "session_number", "turn_number", "reason", "recovery_mode",
            "recorded_at", "interrupted_tool_call_ids",
            "interrupted_tool_calls",
        }),
    ),
    TraceEventSpec(
        "length_continue",
        frozenset({
            "session_number", "turn_number", "attempt", "tokens",
        }),
    ),
    TraceEventSpec(
        "lsp_diagnostics",
        frozenset({
            "session_number", "file", "errors", "warnings", "ms",
            "server", "status",
        }),
    ),
    TraceEventSpec(
        "stale_guard_observe",
        frozenset({"session_number", "path", "source", "fingerprint"}),
    ),
    TraceEventSpec(
        "stale_guard",
        frozenset({
            "session_number", "path", "reason", "mode", "blocked",
            "expected", "current",
        }),
    ),
    TraceEventSpec(
        "redirect_rule",
        frozenset({
            "session_number", "turn_number", "rule", "tool",
            "fragment_index",
        }),
    ),
    TraceEventSpec(
        "stream_rule_triggered",
        frozenset({
            "session_number", "turn_number", "rule", "scope", "offset",
        }),
    ),
    TraceEventSpec(
        "stream_rule_injection",
        frozenset({
            "session_number", "turn_number", "rules", "delivery",
        }),
    ),
    TraceEventSpec(
        "schema_reject",
        frozenset({"session_number", "turn_number", "tool", "errors"}),
    ),
    TraceEventSpec(
        "security_finding",
        frozenset({
            "session_number", "turn_number", "id", "rule", "stage",
            "action",
        }),
    ),
    TraceEventSpec(
        "injection",
        frozenset({
            "session_number", "turn_number", "rule", "trigger", "path",
        }),
    ),
    TraceEventSpec(
        "proc_start",
        frozenset({
            "session_number", "proc_id", "command_sha256", "log_path",
            "result",
        }),
    ),
    TraceEventSpec(
        "proc_poll",
        frozenset({
            "session_number", "proc_id", "result", "output_sha256",
            "running", "exit_code", "timed_out", "cursor_start",
            "cursor_end",
        }),
    ),
    TraceEventSpec(
        "proc_kill",
        frozenset({
            "session_number", "proc_id", "result", "was_running",
            "exit_code", "reason",
        }),
    ),
    TraceEventSpec("regression", frozenset({"session_number", "n_regressed"})),
    TraceEventSpec(
        "adaptive_phase_switch",
        frozenset({"session_number", "phase"}),
    ),
    TraceEventSpec(
        "harness_observation",
        frozenset({
            "session_number",
            "turn_number",
            "concern_id",
            "concern_type",
            "reason",
        }),
    ),
    TraceEventSpec(
        "runtime_envelope",
        frozenset({
            "session", "sandbox_mode", "sandbox_engaged", "sandbox_backend",
            "container_runtime", "container_image_digest",
        }),
        frozenset({
            "sandbox_selected", "sandbox_resolved",
            "sandbox_explicit_unsandboxed", "sandbox_platform",
            "sandbox_supported", "sandbox_installed",
            "sandbox_available", "sandbox_unavailable",
            "sandbox_backend_executable",
        }),
    ),
    TraceEventSpec("guardrail_init", frozenset({"session_number"})),
    TraceEventSpec("trace_corrupt", frozenset({"session_number", "kind"})),
    TraceEventSpec("pretest_run", frozenset({"session_number"})),
    TraceEventSpec(
        "approval_request",
        frozenset({"session_number", "turn_number", "tool_name", "reason"}),
    ),
    TraceEventSpec(
        "clarification_request",
        frozenset({
            "session_number", "turn_number", "request_id", "tool_call_id",
            "question",
        }),
    ),
    TraceEventSpec(
        "clarification_answer",
        frozenset({
            "session_number", "turn_number", "request_id", "answer_sha256",
            "answer_chars",
        }),
    ),
    TraceEventSpec(
        "clarification_consumed",
        frozenset({
            "session_number", "turn_number", "request_id", "answer_sha256",
            "delivery",
        }),
    ),
    TraceEventSpec(
        "clarification_rewound",
        frozenset({
            "session_number", "turn_number", "request_id", "rewind_id",
            "to_turn",
        }),
    ),
    TraceEventSpec(
        "clarification_rejected",
        frozenset({
            "session_number", "turn_number", "tool_call_id", "reason",
        }),
    ),
    TraceEventSpec(
        "correction_created",
        frozenset({
            "session_number", "correction_id", "text_sha256", "text_chars",
        }),
    ),
    TraceEventSpec(
        "correction_consumed",
        frozenset({
            "session_number", "turn_number", "correction_id", "text_sha256",
            "transcript_segment", "delivery",
        }),
    ),
    TraceEventSpec(
        "correction_replayed",
        frozenset({
            "session_number", "turn_number", "correction_id", "text_sha256",
            "source_session_number", "source_turn_number",
            "source_transcript_segment",
        }),
    ),
    TraceEventSpec(
        "permission",
        frozenset({
            "session_number", "turn_number", "tool", "rule", "decision",
        }),
    ),
    # API errors include the HTTP detail in the trace.
    TraceEventSpec(
        "api_error",
        frozenset({"session_number", "turn_number", "error_type", "detail"}),
    ),
)

# Derived compatibility views. Additional fields are always allowed.
KNOWN_TRACE_EVENT_TYPES = frozenset(
    spec.event_type for spec in TRACE_EVENT_SPECS
)
TRACE_EVENT_REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    spec.event_type: spec.required_fields for spec in TRACE_EVENT_SPECS
}
TRACE_EVENT_OPTIONAL_FIELDS: dict[str, frozenset[str]] = {
    spec.event_type: spec.optional_fields for spec in TRACE_EVENT_SPECS
    if spec.optional_fields
}


def _validate_event(event_type: str, fields_keys) -> None:
    """Warn on unknown event_type or missing required fields. Never raises."""
    if event_type not in KNOWN_TRACE_EVENT_TYPES:
        log.warning("trace: unknown event_type %r — emitted with reduced validation", event_type)
        return
    field_names = set(fields_keys)
    required = TRACE_EVENT_REQUIRED_FIELDS.get(event_type, frozenset())
    if event_type == "rewind":
        workspace_fields = frozenset({
            "reason", "commit", "rewind_count", "rewind_id", "delivery",
        })
        report_fields = frozenset({"report_chars"})
        if field_names & workspace_fields:
            required = required | workspace_fields
        elif field_names & frozenset({
            "report_chars", "checkpoint_message_count", "goal", "report",
        }):
            required = required | report_fields
    missing = required - field_names
    if missing:
        log.warning("trace: %s event missing required fields: %s",
                    event_type, sorted(missing))


def write_trace(session, entry: dict) -> None:
    """Session-level equivalent: write a single pre-built JSON line to the
    session's trace file, append it to the in-memory event mirror, and
    trigger the mechanical state writer.

    Internal API. Most callers should use :func:`emit` which adds the
    canonical envelope (``event``, ``trace_schema_version``) and validates
    the event_type against the known set. ``write_trace`` is retained
    for the rare case where a fully-formed entry is built elsewhere.

    Side-effect: stamps ``session._last_trace_write_ms`` with this
    write+projection cost in milliseconds so the per-turn loop can
    surface trace IO time on the next emit.

    Replay hooks live HERE because this is the single funnel every trace
    event passes through (both Session._write_trace and emit()): the
    trace-level fidelity gate compares each executed tool_call against the
    recording, and the replay-stop capture fires when the stop turn's event
    is written (docs/replay_mode_spec.md).
    """
    if (
        entry.get("event") == "tool_call"
        and not entry.get("parent_tool_call_id")
    ):
        _verify = getattr(getattr(session, "client", None), "verify_executed_turn", None)
        if _verify is not None:
            _verify(entry)
            from .replay_handover import maybe_capture_at_stop
            maybe_capture_at_stop(session, entry)
    _t0 = time.perf_counter()
    if session._trace_file is not None:
        line = _dumps(entry) + "\n"
        # Async writer when available: hot path enqueues + returns,
        # daemon thread does write+flush. Falls back to sync write
        # when no writer is attached (e.g. tests that poke
        # write_trace without going through Session.run).
        writer = getattr(session, "_async_trace_writer", None)
        if writer is not None:
            writer.submit(line)
        else:
            session._trace_file.write(line)
            session._trace_file.flush()
    session._trace_events.append(entry)
    session._refresh_state()
    _elapsed_ms = (time.perf_counter() - _t0) * 1000
    # Per-turn accumulator — run_step zeroes this at the top of each
    # turn and reads it on the bottom emit. Robust to sessions that
    # don't initialize it (tests), in which case we just stamp.
    try:
        session._turn_trace_write_ms = (
            getattr(session, "_turn_trace_write_ms", 0.0) + _elapsed_ms
        )
    except AttributeError:
        pass


def emit(session, event_type: str, **fields) -> None:
    """Centralized trace event emission with schema validation.

    Constructs the canonical envelope:
        {"event": <event_type>,
         "trace_schema_version": TRACE_SCHEMA_VERSION,
         **fields}

    and writes it via :func:`write_trace`. Validation is best-effort
    and warning-only — an unknown event_type or missing required field
    logs but does not raise, so a typo doesn't break a session in flight.
    """
    _validate_event(event_type, fields.keys())
    write_trace(session, {
        "event": event_type,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        **fields,
    })


def emit_trace_event(trace_file, event_type: str, **fields) -> dict:
    """Module-level equivalent of Session._emit for solve_task-level
    writes that occur BEFORE any Session is constructed (e.g. the
    session_start event written by the per-session-loop dispatch and
    the runtime_envelope event written at task start).

    Returns the entry dict it wrote, for callers that want to inspect
    it. Validation behavior is identical to Session._emit (warn on
    unknown / missing-required, never raise). Writes only when
    trace_file is non-None — convenience for paths that may be running
    without a trace target.
    """
    _validate_event(event_type, fields.keys())
    entry = {
        "event": event_type,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        **fields,
    }
    if trace_file is not None:
        trace_file.write(_dumps(entry) + "\n")
        trace_file.flush()
    return entry
