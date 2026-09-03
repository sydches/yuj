"""Agentic loop — Session (inner) + solve_task (outer)."""
from collections import deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import re
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import IO

import openai

from ..config import Config
from ._shell_patterns import TEST_COMMAND_RE as _TEST_COMMAND_RE
from .context import ContextManager
from .context_contract import build_context_contract
from .context_strategies import SolverStateContext
from .tool_specs import PARALLEL_READ_SAFE_TOOL_NAMES
from .tool_loading import (
    ToolLoadingError,
    loader_error,
    loader_success,
    replace_tool_surface,
)
from .schemas import get_tool_schemas

# Tools safe to dispatch concurrently when the flag is set.
_READONLY_TOOLS = PARALLEL_READ_SAFE_TOOL_NAMES
from .injections import (
    InjectionState,
    UserTurnInjection,
    fire_candidates,
    fire_path_candidates,
    load_injections,
    record_fire,
    record_user_turn_delivery,
)
from .stream_rules import (
    StreamRuleRuntime,
    format_interrupt_fragment,
    load_stream_rules,
)
from .guardrails import (
    Action,
    GuardrailState,
    GuardrailRegistry,
    build_guardrail_registry,
    init_guardrail_state,
    validate_guardrail_registry,
)
from ._guardrails.extractors import MUTATION_TOOLS
from .._shared.classification import is_error_result
from .tool_validation import ToolSchemaSet
from .tool_policy import PermissionPolicy
from .sandbox.ignore_policy import (
    PROJECT_INIT_PRIVATE_RULES,
    IgnorePolicy,
    load_ignore_policy,
)
from .solver import build_system_prompt, collect_provenance, write_checkpoint, write_run_metrics
from .state_writer import active_events, write_state_from_events, write_state_from_trace
from .tools import (
    ToolRegistry, _bash_readable_paths, _bash_unreadable_paths,
    admit_tool_output,
    _effective_command_environment, build_tool_registry, dispatch,
    validate_tool_handlers,
)

log = logging.getLogger(__name__)

# Module-level constants — avoid chr(10) calls in hot paths.
_NEWLINE = "\n"

# Trace event schema lives in _loop/trace_schema.py. Names are re-exported
# here under their legacy underscore-prefixed identifiers because
# state_writer.py / analysis tools / tests import them via this module.
from ._loop.trace_schema import (
    KNOWN_TRACE_EVENT_TYPES as _KNOWN_TRACE_EVENT_TYPES,
    TRACE_EVENT_REQUIRED_FIELDS as _TRACE_EVENT_REQUIRED_FIELDS,
    TRACE_SCHEMA_VERSION,
    emit_trace_event as _emit_trace_event,
)


# ── Bash command normalization for duplicate detection ──────────────────
# Strips trailing pipe chains and stderr redirects so trivial variants
# like `cmd | tail -60` and `cmd | tail -80` compare as identical.
# Content-blind: operates on bash syntax structure, not on what the
# command does.  Only used for the duplicate_abort signature — the
# actual command executes unmodified.
_TRAILING_PIPE_RE = re.compile(
    r"""
    \s*                          # optional leading whitespace before pipe
    (?:                          # group: one pipe segment
        \|                       # the pipe character
        \s*                      # optional whitespace after pipe
        (?:head|tail|grep|cat|sort|uniq|wc|tee|less|more)  # common filter commands
        (?:\s+[^\|]*)?)          # their arguments (up to next pipe or end)
    +                            # one or more trailing pipe segments
    $                            # anchored at end
    """,
    re.VERBOSE,
)
_STDERR_REDIRECT_RE = re.compile(r"\s*2>&1\s*")
_BASH_READ_TARGET_RE = re.compile(
    r"^\s*(cat|head|tail|less|more|file)\s+([^\s|;&<>`$()]+)\s*$"
)
_PATH_SUFFIXES = (
    ".py", ".pyi", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh",
    ".rs", ".go", ".java", ".js", ".jsx", ".ts", ".tsx",
)
_SHELL_SEPARATORS = frozenset({"&&", "||", "|", ";"})


# ── Error taxonomy ───────────────────────────────────────────────────────

NORMAL_LIFECYCLE = frozenset({"context_full", "length"})
MODEL_STUCK = frozenset({"duplicate_abort", "max_turns"})
_TRANSIENT_ERRORS = (openai.APIConnectionError, openai.APITimeoutError)



from ._loop import (  # noqa: F401
    _apply_profile_preamble, _apply_profile_schema_simplify,
    _apply_profile_tool_cap, apply_profile_to_schemas,
    bind_effective_edit_format, resolve_effective_edit_format,
    build_tool_surface,
    _auto_commit, _canon_focus_path,
    _dedup_signature, _encode_focus_path, _encode_focus_target,
    _extract_bash_focus_target, _extract_test_target_from_command,
    _filter_disabled_tools, _focus_signature, _load_bash_transforms,
    _looks_like_path_token, _normalize_bash_for_dedup,
    _normalize_repo_timestamps, _path_within_cwd, _pretest_is_green,
    _record_session_start_costs, _resolve_profile,
    _resolve_token_estimator, _sanitize_runner_timing,
    _simplify_tool_schema, _split_bash_segments,
    _truncate_focus_display, _truncate_for_trace,
    _truncate_pretest_output, build_context_manager, build_resume_prompt,
    run_pretest,
)


# Canonical set of finish_reason values emitted by Session.run().
# Adding a new finish_reason: append it here. Pre-fix any typo or
# missing-from-docstring reason silently shipped — analysis tools
# reading the trace had no source of truth for the legal values.
_KNOWN_FINISH_REASONS: frozenset[str] = frozenset({
    "stop",
    "model_done",
    "no_tool_call",
    "max_turns",
    "context_full",
    "duplicate_abort",
    "loop_detected",
    "intent_abort",
    "done_loop",
    "mutation_repeat_abort",
    "contract_recovery_abort",
    "contract_commit_abort",
    "gate_escalation",
    "length",
    "error",
    "sandbox_unavailable",
    "task_wall_clock",
    "approval_required",
    "input_required",
    "hook_block",
    # stop_resume delivery (restart experiment): the adaptive controller
    # requested a graceful stop so an orchestrator can resume with (C) or
    # without (B) the chosen rung. See adaptive_control/executors.py
    # stop_for_resume and the stop-note in the telemetry dir.
    "adaptive_stop",
})


@dataclass(frozen=True)
class SessionResult:
    turns: int
    finish_reason: str  # one of _KNOWN_FINISH_REASONS
    done: bool
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def __post_init__(self):
        # Warn (don't raise) on unknown finish_reason — a typo at the
        # callsite shouldn't abort a run, but it should be visible in
        # logs so the analysis pipeline can flag it.
        if self.finish_reason and self.finish_reason not in _KNOWN_FINISH_REASONS:
            log.warning(
                "SessionResult: unknown finish_reason=%r (not in _KNOWN_FINISH_REASONS)",
                self.finish_reason,
            )


@dataclass(frozen=True)
class TaskSpec:
    """Task substrate input for solve_task (repo layout is only one source)."""

    prompt_text: str
    pretest_script: Path | None = None


def rewind_to(
    session: "Session", turn_number: int, *, reason: str = "operator"
) -> dict[str, object]:
    """Public harness entry point for an atomic conversation/tree rewind."""
    from .turn_snapshots import rewind_to as _rewind_to
    return _rewind_to(session, turn_number, reason=reason)


class Session:
    """One context window — multi-turn tool calling until done or limit."""

    def __init__(
        self,
        cfg: Config,
        client,
        system_prompt: str,
        initial_message: str,
        cwd: str,
        context_manager: ContextManager | None = None,
        trace_file: IO | None = None,
        session_number: int = 0,
        trace_path: Path | None = None,
        state_path: Path | None = None,
        output_control=None,
        universal_rewrites=None,
        forbidden_rules=None,
        redirect_rules=None,
        redactions=None,
        output_parser=None,
        pretest_parsed: dict | None = None,
        guardrail_registry: GuardrailRegistry | None = None,
        tool_registry: ToolRegistry | None = None,
        checkpoint_store=None,
        lsp_manager=None,
        process_manager=None,
        terminal_manager=None,
        injections=None,
        stream_rules=None,
        artifact_dir: Path | None = None,
        adaptive_control_baseline_config_paths: tuple[str, ...] | list[str] | None = None,
        ignore_policy: IgnorePolicy | None = None,
        effective_env: Mapping[str, str] | None = None,
        allow_login_shell: bool | None = None,
        tool_allowlist: frozenset[str] | None = None,
        subagent_level: int = 0,
        subagent_runtime=None,
        subagent_read_only: bool = False,
        local_tokenizer=None,
    ):
        cfg = bind_effective_edit_format(cfg, client)
        self.cfg = cfg
        from .compaction_hooks import resolve_compaction_hook
        self._compaction_hook_reference = str(
            getattr(cfg, "compaction_hook", "") or ""
        ).strip()
        self._compaction_hook = resolve_compaction_hook(
            self._compaction_hook_reference
        )
        self._permission_policy = PermissionPolicy.from_rule_tables(
            getattr(cfg, "permissions_preset_rules", {}),
            getattr(cfg, "permissions_rules", {})
        )
        self.client = client
        self.cwd = cwd
        if effective_env is None:
            resolved_env, resolved_login_shell = (
                _effective_command_environment(cfg)
            )
        else:
            from .sandbox.env_policy import build_subprocess_env
            resolved_env = build_subprocess_env(effective_env)
            resolved_login_shell = bool(
                getattr(cfg, "sandbox_env_allow_login_shell", False)
                if allow_login_shell is None
                else allow_login_shell
            )
        self._effective_env = MappingProxyType(resolved_env)
        self._allow_login_shell = resolved_login_shell
        self._ignore_policy = ignore_policy or load_ignore_policy(
            cwd,
            enabled=getattr(cfg, "state_ignore_file_enabled", True),
            file_names=getattr(
                cfg, "state_ignore_file_names", (".yujignore",)
            ),
            builtin_rules=(
                PROJECT_INIT_PRIVATE_RULES
                if getattr(cfg, "assistant_project_init_destination", "")
                else ()
            ),
        )
        self._session_number = session_number
        self._current_turn = 0
        self._service_event_lock = threading.RLock()
        self._trace_file = trace_file
        self._trace_path = trace_path
        self._state_path = state_path
        self._artifact_dir = Path(
            artifact_dir
            or (trace_path.parent if trace_path is not None else cwd)
        )
        self._tool_allowlist = (
            frozenset(tool_allowlist) if tool_allowlist is not None else None
        )
        self._subagent_level = int(subagent_level)
        self._subagent_read_only = bool(subagent_read_only)
        self._subagent_prompt_tokens = 0
        self._subagent_completion_tokens = 0
        self._subagent_calls = 0
        self._own_prompt_tokens = 0
        self._own_completion_tokens = 0
        self._last_assistant_content = ""
        self._final_text = ""
        self._pending_user_turn_injections: list[UserTurnInjection] = []
        self._pending_user_turn_texts: set[str] = set()
        if subagent_runtime is None and getattr(cfg, "tools_task_enabled", False):
            from .subagents import SubagentRuntime

            run_root = (
                trace_path.parent
                if trace_path is not None
                else Path(artifact_dir or cwd)
            )
            subagent_runtime = SubagentRuntime(run_root)
        self._subagent_runtime = subagent_runtime
        self.output_control = output_control
        self.universal_rewrites = universal_rewrites
        self.forbidden_rules = forbidden_rules
        self.redirect_rules = redirect_rules
        self.redactions = redactions
        self.output_parser = output_parser
        self.pretest_parsed = pretest_parsed
        explicit_instance_id = str(
            getattr(cfg, "adaptive_control_source_instance_id", "") or ""
        )
        task_dir = Path(cwd)
        derived_instance_id = (
            task_dir.parent.name if task_dir.name == "host_task" else task_dir.name
        )
        self.instance_id = explicit_instance_id or (
            derived_instance_id
            if bool(getattr(cfg, "adaptive_control_enabled", False))
            else ""
        )
        self.attempt_id = (
            f"{self.instance_id}:session{int(session_number)}"
            if self.instance_id
            else ""
        )
        self.adaptive_control_baseline_config_paths = tuple(
            adaptive_control_baseline_config_paths
            if adaptive_control_baseline_config_paths is not None
            else getattr(cfg, "adaptive_control_baseline_config_paths", ())
        )
        self.adaptive_control_resolved_baseline_cfg = cfg
        # Monotonic bash counter for sink filenames (.tool_output/<sess>_<N>.log)
        self._sink_counter: int = 0
        registered_tool_schemas = get_tool_schemas(
            cfg.tool_desc,
            code_mode=bool(
                getattr(cfg, "tools_exec_cell_enabled", False)
            ),
        )
        if self._subagent_level:
            registered_tool_schemas = [
                schema
                for schema in registered_tool_schemas
                if schema.get("function", {}).get("name") not in {
                    "ask_user", "subagent_changes", "apply_subagent",
                }
            ]
        if self._subagent_level >= int(
            getattr(cfg, "tools_subagent_depth", 1) or 0
        ):
            registered_tool_schemas = [
                schema for schema in registered_tool_schemas
                if schema.get("function", {}).get("name") != "task"
            ]
        if self._tool_allowlist is not None:
            registered_tool_schemas = [
                schema for schema in registered_tool_schemas
                if schema.get("function", {}).get("name")
                in self._tool_allowlist
            ]
        self._tool_surface = build_tool_surface(
            cfg,
            client,
            registered_tool_schemas,
        )
        self._tool_schemas = self._tool_surface.active_schemas
        from ._loop.profile_resolution import build_plan_mode_schemas
        self._plan_tool_schemas = (
            build_plan_mode_schemas(cfg, client)
            if bool(getattr(cfg, "plan_mode_enabled", False))
            else []
        )

        def _redirect_event_sink(payload: dict[str, object]) -> None:
            fields = dict(payload)
            event_type = str(fields.pop("event", "redirect_rule"))
            with self._service_event_lock:
                self._emit(
                    event_type,
                    session_number=self._session_number,
                    turn_number=self._current_turn,
                    **fields,
                )

        self._redirect_event_sink = _redirect_event_sink
        # Security findings use the same locked, raw-trace-only service sink.
        # Keep a distinct attribute so dispatch wiring names the owner event.
        self._security_event_sink = _redirect_event_sink
        self._lsp_manager = lsp_manager
        if self._lsp_manager is None and (
            getattr(cfg, "lsp_enabled", False)
            or getattr(cfg, "lsp_tool_enabled", False)
        ):
            from .lsp_support import LspManager, parse_server_specs
            from .sandbox.policy import sandbox_execution_kwargs

            def _lsp_event_sink(payload: dict[str, object]) -> None:
                fields = dict(payload)
                event_type = str(fields.pop("event", "lsp_diagnostics"))
                with self._service_event_lock:
                    self._emit(
                        event_type,
                        session_number=self._session_number,
                        turn_number=self._current_turn,
                        **fields,
                    )

            self._lsp_manager = LspManager.sandboxed(
                cwd=cwd,
                servers=parse_server_specs(getattr(cfg, "lsp_servers", {})),
                bwrap_bin=cfg.bwrap_bin,
                unreadable_paths=_bash_unreadable_paths(
                    cwd, cfg, self._ignore_policy,
                ),
                readable_paths=_bash_readable_paths(cfg),
                effective_env=self._effective_env,
                allow_login_shell=self._allow_login_shell,
                **sandbox_execution_kwargs(cfg),
                diagnostics_timeout_s=float(
                    getattr(cfg, "lsp_diagnostics_timeout_s", 2.0)
                ),
                min_severity=getattr(cfg, "lsp_min_severity", "error"),
                enabled=bool(getattr(cfg, "lsp_enabled", False)),
                tool_enabled=bool(getattr(cfg, "lsp_tool_enabled", False)),
                event_sink=_lsp_event_sink,
            )

        base_registry = tool_registry or build_tool_registry()
        handlers = dict(base_registry.handlers)
        self._process_manager = process_manager
        self._terminal_manager = terminal_manager

        base_bash_handler = handlers["bash"]

        def _bash_handler(args, dispatch_cwd, dispatch_cfg):
            if self._subagent_read_only:
                from .subagents import prepare_readonly_bash

                if bool(args.get("background", False)):
                    return "ERROR: read-only subagent blocked bash: background execution"
                prepared, reason = prepare_readonly_bash(
                    str(args.get("cmd", ""))
                )
                if prepared is None:
                    return f"ERROR: read-only subagent blocked bash: {reason}"
                args = dict(args)
                args["cmd"] = prepared
            if not bool(args.get("background", False)):
                return base_bash_handler(args, dispatch_cwd, dispatch_cfg)
            if self._process_manager is None:
                return "ERROR: background processes are not enabled"
            return self._process_manager.start(str(args["cmd"])).result

        def _bash_poll_handler(args, _cwd, _cfg):
            if self._process_manager is None:
                return "ERROR: background processes are not enabled"
            timeout = args.get("timeout_s")
            return self._process_manager.poll(
                str(args["proc_id"]),
                timeout_s=None if timeout is None else float(timeout),
            ).result

        def _bash_kill_handler(args, _cwd, _cfg):
            if self._process_manager is None:
                return "ERROR: background processes are not enabled"
            return self._process_manager.kill(str(args["proc_id"])).result

        def _terminal_start_handler(args, _cwd, _cfg):
            if self._terminal_manager is None:
                return "ERROR: interactive terminals are not enabled"
            return self._terminal_manager.start(str(args["cmd"])).result

        def _terminal_io_handler(args, _cwd, _cfg):
            if self._terminal_manager is None:
                return "ERROR: interactive terminals are not enabled"
            terminal_id = str(args["terminal_id"])
            if bool(args.get("terminate", False)):
                if "input" in args:
                    return (
                        "ERROR: terminal_io cannot send input and terminate "
                        "in the same call"
                    )
                self._terminal_manager.kill(terminal_id)
                return self._terminal_manager.read(
                    terminal_id, timeout_s=0
                ).result
            if "input" in args:
                self._terminal_manager.write(
                    terminal_id,
                    str(args["input"]),
                    append_newline=bool(args.get("append_newline", True)),
                )
            timeout = args.get("timeout_s")
            return self._terminal_manager.read(
                terminal_id,
                timeout_s=None if timeout is None else float(timeout),
            ).result

        def _lsp_handler(args, _cwd, _cfg):
            if self._lsp_manager is None:
                return "ERROR: lsp manager is not configured"
            try:
                target = Path(self.cwd) / str(args["path"])
                self._ignore_policy.require_visible(
                    target, is_dir=target.is_dir()
                )
                query = self._lsp_manager.query(
                    str(args["kind"]),
                    path=str(args["path"]),
                    line=int(args.get("line", 0)),
                    character=int(args.get("character", 0)),
                )
            except Exception as exc:
                return f"ERROR: lsp query failed: {exc}"
            body = query.result or "[]"
            return (
                f"LSP {query.kind} {query.file} status={query.status}\n{body}"
            )

        def _task_handler(args, _cwd, _cfg):
            if self._subagent_runtime is None:
                return "ERROR: task tool has no subagent runtime"
            return self._subagent_runtime.execute(
                self,
                args.get("agent"),
                args.get("prompt"),
            )

        def _subagent_changes_handler(args, _cwd, _cfg):
            if self._subagent_runtime is None:
                return "ERROR: subagent_changes has no subagent runtime"
            return self._subagent_runtime.review_changes(
                self,
                args.get("task_id"),
                offset=args.get("offset", 0),
                limit=args.get("limit", 200),
            )

        def _apply_subagent_handler(args, _cwd, _cfg):
            if self._subagent_runtime is None:
                return "ERROR: apply_subagent has no subagent runtime"
            return self._subagent_runtime.apply_changes(
                self,
                args.get("task_id"),
            )

        def _load_tools_handler(args, _cwd, _cfg):
            try:
                activation = self._tool_surface.activate(args.get("names"))
            except ToolLoadingError as exc:
                return loader_error(exc, self._tool_surface)
            self._sync_tool_surface()
            self._tool_activation_events += 1
            self._activated_tool_names.update(activation.activated)
            self._emit(
                "tools_activated",
                session_number=self._session_number,
                turn_number=self._current_turn,
                requested=list(activation.requested),
                activated=list(activation.activated),
                already_active=list(activation.already_active),
                active_tools=list(activation.active_tools),
            )
            return loader_success(activation)

        def _exit_plan_mode_handler(_args, _cwd, _cfg):
            return self._plan_mode.exit(turn=self._current_turn)

        handlers["lsp"] = _lsp_handler
        handlers["task"] = _task_handler
        handlers["subagent_changes"] = _subagent_changes_handler
        handlers["apply_subagent"] = _apply_subagent_handler
        handlers["load_tools"] = _load_tools_handler
        handlers["exit_plan_mode"] = _exit_plan_mode_handler
        handlers["bash"] = _bash_handler
        handlers["bash_poll"] = _bash_poll_handler
        handlers["bash_kill"] = _bash_kill_handler
        handlers["terminal_start"] = _terminal_start_handler
        handlers["terminal_io"] = _terminal_io_handler
        from ._loop.exec_cell_runtime import build_session_exec_cell_handler

        handlers["exec_cell"] = build_session_exec_cell_handler(
            self, dispatch_getter=lambda: dispatch
        )
        from .checkpoint_rewind import build_session_tool_handlers
        handlers.update(build_session_tool_handlers(self))
        if self._tool_allowlist is not None:
            registered_names = set(self._tool_surface.registered_names)
            handlers = {
                name: handler for name, handler in handlers.items()
                if name in registered_names
            }
        self._tool_registry = ToolRegistry(handlers=handlers)
        self._context_checkpoint = None
        self._pending_context_checkpoint = None
        self._pending_context_rewind = None
        self._checkpoint_store = checkpoint_store
        self._artifact_dir = Path(artifact_dir or cwd)
        from .turn_snapshots import rewind_snapshot_dir
        self._rewind_snapshot_dir = rewind_snapshot_dir(
            Path(cwd), self._artifact_dir
        )
        self._rewind_guard_snapshots: dict[int, GuardrailState] = {}
        self._pending_rewind: dict[str, object] | None = None
        schema_names = list(self._tool_surface.registered_names)
        validate_tool_handlers(
            schema_names,
            allow_extra_handlers=self._tool_allowlist is None,
            registry=self._tool_registry,
        )
        self._tool_schema_set = ToolSchemaSet.from_openai_tools(
            self._tool_schemas
        )
        self._plan_tool_schema_set = ToolSchemaSet.from_openai_tools(
            self._plan_tool_schemas or self._tool_schemas
        )
        if context_manager is not None:
            self.context: ContextManager = context_manager
        else:
            self.context = build_context_manager(
                SolverStateContext,
                cfg,
                Path(cwd),
                initial_message,
                session_number,
                _resolve_token_estimator(client),
            )
            assert self.context is not None
        self.context.configure_thought_retention(
            cfg.tools_think_keep_turns,
            session_number=session_number,
        )
        self.context.add_system(system_prompt)
        self.context.add_user(initial_message)
        self._pending_clarification_delivery: dict | None = None
        self._pending_correction_delivery: dict | None = None
        self._protected_correction_text: str | None = None
        if (
            getattr(cfg, "runtime_mode", "measurement") == "assistant"
            and self._subagent_level == 0
            and not bool(getattr(client, "is_replay", False))
        ):
            from .clarifications import clarification_state

            clarification = clarification_state(self._artifact_dir)
            if clarification.phase == "input_ready":
                assert clarification.request is not None
                assert clarification.answer is not None
                self.context.add_user(
                    "Clarification question: "
                    f"{clarification.request['question']}\n"
                    "The next user message is the operator's exact recorded "
                    "answer. Use it as information only. It does not approve "
                    "any tool or action."
                )
                self.context.add_user(clarification.answer["answer"])
                self._pending_clarification_delivery = {
                    "request_id": clarification.request["request_id"],
                    "answer_sha256": clarification.answer["answer_sha256"],
                }
            from .corrections import (
                CorrectionStateError,
                validate_correction_trace,
            )

            correction = validate_correction_trace(self._artifact_dir)
            if correction.phase == "pending":
                assert correction.correction is not None
                if correction.correction["session_id"] != self._artifact_dir.name:
                    raise CorrectionStateError(
                        "correction belongs to another session"
                    )
                self._pending_correction_delivery = {
                    "correction_id": correction.correction["correction_id"],
                    "text_sha256": correction.correction["text_sha256"],
                    "text": correction.correction["text"],
                    "injected": False,
                }
        # All thrash-control state lives in one place. See harness/guardrails.py.
        # Session is the orchestrator; the guardrails own their own state
        # machines and expose a uniform Decision interface to the turn loop.
        self._guards: GuardrailState = init_guardrail_state(cfg)
        self._guardrail_registry = guardrail_registry or build_guardrail_registry()
        validate_guardrail_registry(self._guardrail_registry)
        # In-memory mirror of the trace file for this task. Seeded at
        # session __init__ from any prior-session events (trace is appended
        # across sessions). Appended to by _write_trace. Consumed by
        # _refresh_state, avoiding a per-tool-call re-read + JSON parse of
        # the full trace file — was O(T^2) across a session.
        self._trace_events: list[dict] = []
        # Re-hydrate from prior sessions' .trace.jsonl. Failure modes are
        # surfaced loudly (not silently zeroed) — a corrupted trace this
        # session means downstream analytics (state.json projection,
        # compaction gate, mutation count, F2P attribution) operate on a
        # truncated history without warning. Prior behavior dropped all
        # events after one mid-file JSONDecodeError, which caused silent
        # data loss.
        #
        # Policy:
        #   - OSError (file unreadable, permissions): keep self._trace_events
        #     empty AND log a warning. Treated as "first session, file does
        #     not exist yet" if trace_path.is_file() lied (rare race).
        #   - JSONDecodeError mid-file: KEEP every event parsed before the
        #     bad line so partial history survives, log the offending line
        #     index and the corruption, do NOT raise (a corrupt trace from
        #     a prior session must not block this session from starting).
        #   - Both cases set self._trace_corrupted=True so a future
        #     centralized emitter (Contract P0 #2) can surface a
        #     `trace_corrupt` event when one exists.
        self._trace_corrupted: bool = False
        # Track structured corruption details so the centralized
        # trace_corrupt emit (after the seed loop) can
        # name the failure cause without re-parsing the log message.
        _trace_corrupt_kind = ""
        _trace_corrupt_detail = ""
        _trace_corrupt_line = 0
        if trace_path is not None and trace_path.is_file():
            try:
                with open(trace_path) as _f:
                    for _idx, _line in enumerate(_f, start=1):
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            self._trace_events.append(json.loads(_line))
                        except json.JSONDecodeError as _je:
                            self._trace_corrupted = True
                            _trace_corrupt_kind = "json_decode_error"
                            _trace_corrupt_detail = str(_je)
                            _trace_corrupt_line = _idx
                            log.warning(
                                "trace_corrupt: %s line %d unparseable (%s); "
                                "keeping %d prior events, dropping rest of file",
                                trace_path, _idx, _je, len(self._trace_events),
                            )
                            break
            except OSError as _oe:
                self._trace_corrupted = True
                _trace_corrupt_kind = "unreadable"
                _trace_corrupt_detail = str(_oe)
                log.warning(
                    "trace_unreadable: %s could not be opened (%s); "
                    "starting session with empty trace mirror",
                    trace_path, _oe,
                )
        if self._trace_corrupted and trace_file is not None:
            _emit_trace_event(
                trace_file, "trace_corrupt",
                session_number=session_number,
                kind=_trace_corrupt_kind,
                detail=_trace_corrupt_detail,
                line=_trace_corrupt_line,
                events_kept=len(self._trace_events),
            )
        from .plan_mode import PlanModeController

        def _plan_mode_event_sink(payload: dict[str, object]) -> None:
            fields = dict(payload)
            event_type = str(fields.pop("event"))
            with self._service_event_lock:
                self._emit(
                    event_type,
                    session_number=self._session_number,
                    **fields,
                )

        self._plan_mode = PlanModeController(
            cwd=self.cwd,
            cfg=cfg,
            events=self._trace_events,
            event_sink=_plan_mode_event_sink,
        )
        self._rewind_count = sum(
            1
            for event in self._trace_events
            if event.get("event") == "rewind"
            and isinstance(event.get("rewind_id"), str)
            and int(event.get("session_number", -1)) == int(session_number)
        )
        if self._trace_events:
            from .checkpoint_rewind import restore_rewind_reports
            restore_rewind_reports(self.context, self._trace_events)
        if self._process_manager is None and getattr(
            cfg, "tools_background_enabled", False
        ):
            from .process_manager import ProcessManager, ReplayProcessManager

            replay_events = getattr(client, "process_events", None)
            if bool(getattr(client, "is_replay", False)):
                self._process_manager = ReplayProcessManager(
                    event for event in (replay_events or ())
                    if int(event.get("session_number", -1)) == session_number
                )
            else:
                def _process_event_sink(payload: dict[str, object]) -> None:
                    fields = dict(payload)
                    event_type = str(fields.pop("event"))
                    with self._service_event_lock:
                        self._emit(
                            event_type,
                            session_number=self._session_number,
                            turn_number=self._current_turn,
                            **fields,
                        )

                def _admit_poll_output(value: str) -> str:
                    from .security_scan import (
                        SecurityScanner,
                        emit_findings,
                        render_security_block,
                    )

                    scanner = SecurityScanner.from_config(self.cfg)
                    outcome = scanner.scan_text(value, stage="result")
                    try:
                        emit_findings(
                            outcome.findings, self._security_event_sink
                        )
                    except Exception as exc:
                        log.warning("security finding emit failed: %s", exc)
                    if outcome.blocked:
                        return render_security_block("bash_poll", outcome)
                    return admit_tool_output(
                        "bash_poll",
                        value,
                        arguments={},
                        cfg=self.cfg,
                        output_control=self.output_control,
                        redactions=self.redactions,
                        security_findings=outcome.findings,
                    )

                manager_run_dir = Path(
                    artifact_dir
                    or (trace_path.parent if trace_path is not None else cwd)
                )
                from .sandbox.policy import sandbox_execution_kwargs

                self._process_manager = ProcessManager.sandboxed(
                    run_dir=manager_run_dir,
                    cwd=cwd,
                    bwrap_bin=cfg.bwrap_bin,
                    unreadable_paths=_bash_unreadable_paths(
                        cwd, cfg, self._ignore_policy,
                    ),
                    readable_paths=_bash_readable_paths(cfg),
                    effective_env=self._effective_env,
                    allow_login_shell=self._allow_login_shell,
                    **sandbox_execution_kwargs(cfg),
                    max_procs=int(cfg.tools_background_max_procs),
                    poll_timeout_s=float(cfg.tools_background_poll_timeout),
                    admit_output=_admit_poll_output,
                    event_sink=_process_event_sink,
                )
        if (
            self._terminal_manager is None
            and bool(getattr(cfg, "tools_terminal_enabled", False))
            and getattr(cfg, "runtime_mode", "measurement") == "assistant"
            and "terminal_start" in self._tool_surface.registered_names
        ):
            from .terminal_process import (
                ReplayTerminalProcessManager,
                TerminalProcessManager,
            )

            replay_events = getattr(client, "process_events", None)
            if bool(getattr(client, "is_replay", False)):
                self._terminal_manager = ReplayTerminalProcessManager(
                    event
                    for event in (replay_events or ())
                    if int(event.get("session_number", -1)) == session_number
                )
            else:
                def _terminal_event_sink(payload: dict[str, object]) -> None:
                    fields = dict(payload)
                    event_type = str(fields.pop("event"))
                    with self._service_event_lock:
                        self._emit(
                            event_type,
                            session_number=self._session_number,
                            turn_number=self._current_turn,
                            **fields,
                        )

                def _admit_terminal_output(value: str) -> str:
                    from .security_scan import (
                        SecurityScanner,
                        emit_findings,
                        render_security_block,
                    )

                    scanner = SecurityScanner.from_config(self.cfg)
                    outcome = scanner.scan_text(value, stage="result")
                    try:
                        emit_findings(
                            outcome.findings, self._security_event_sink
                        )
                    except Exception as exc:
                        log.warning("security finding emit failed: %s", exc)
                    if outcome.blocked:
                        return render_security_block("terminal_io", outcome)
                    return admit_tool_output(
                        "terminal_io",
                        value,
                        arguments={},
                        cfg=self.cfg,
                        output_control=self.output_control,
                        redactions=self.redactions,
                        security_findings=outcome.findings,
                    )

                manager_run_dir = Path(
                    artifact_dir
                    or (trace_path.parent if trace_path is not None else cwd)
                )
                from .sandbox.policy import sandbox_execution_kwargs

                self._terminal_manager = TerminalProcessManager.sandboxed(
                    run_dir=manager_run_dir,
                    cwd=cwd,
                    bwrap_bin=cfg.bwrap_bin,
                    unreadable_paths=_bash_unreadable_paths(
                        cwd, cfg, self._ignore_policy,
                    ),
                    readable_paths=_bash_readable_paths(cfg),
                    effective_env=self._effective_env,
                    allow_login_shell=self._allow_login_shell,
                    **sandbox_execution_kwargs(cfg),
                    read_timeout_s=float(cfg.tools_terminal_read_timeout),
                    max_lifetime_s=float(cfg.tools_terminal_max_lifetime),
                    max_output_bytes=int(
                        cfg.tools_terminal_max_output_bytes
                    ),
                    max_input_chars=int(cfg.tools_terminal_max_input_chars),
                    admit_output=_admit_terminal_output,
                    event_sink=_terminal_event_sink,
                )
        from .stale_guard import StaleFileGuard

        def _stale_guard_event_sink(payload: dict[str, object]) -> None:
            fields = dict(payload)
            event_type = str(fields.pop("event"))
            with self._service_event_lock:
                self._emit(
                    event_type,
                    session_number=self._session_number,
                    turn_number=self._current_turn,
                    **fields,
                )

        self._stale_guard = StaleFileGuard.from_trace(
            cwd=self.cwd,
            mode=getattr(cfg, "tools_stale_guard_mode", "warn"),
            events=active_events(self._trace_events),
            event_sink=_stale_guard_event_sink,
        )

        # Seed pretest parity from session 1's parsed pretest verdict (passed
        # as a dict with 'failing' and 'passing' sets). Later sessions inherit
        # the baseline from session 1 via the same mechanism (caller passes
        # the same dict every time). No-op when pretest was not parseable.
        if pretest_parsed:
            self._guards.pretest_failing_tests = set(pretest_parsed.get("failing") or ())
            self._guards.pretest_passing_tests = set(pretest_parsed.get("passing") or ())
        self._last_fill: float = 0.0
        # Server-reported prompt token count from the prior turn's response.
        # Used as the canonical pt signal for both the context_fill_ratio
        # gate and digest compaction. Updated after each successful API
        # call from chat_result.usage.prompt_tokens. 0 before the first
        # turn returns; callers fall back to chars_div_4 estimate.
        self._last_actual_prompt_tokens: int = 0
        # Local tokenizer for exact request token counts. None when
        # cfg.tokenizer_id is unset — callers fall back to the profile or
        # chars_div_4 estimator.
        from .local_tokenizer import load as _load_tokenizer
        tokenizer_was_preloaded = local_tokenizer is not None
        self._tokenizer = (
            local_tokenizer
            if tokenizer_was_preloaded
            else _load_tokenizer(getattr(cfg, "tokenizer_id", "") or "")
        )
        if self._tokenizer is not None and not tokenizer_was_preloaded:
            synced = self._tokenizer.sync_chat_template(
                getattr(cfg, "base_url", "") or "")
            log.info("local tokenizer loaded: %s (server template %s)",
                     self._tokenizer.id, "synced" if synced else "NOT synced — counts approximate")
        if self._tokenizer is not None:
            def _estimate_request_tokens(messages: list[dict]) -> int:
                return int(self._tokenizer.count(
                    messages, tools=self.model_tool_schemas,
                ))

            # Context strategies make pressure decisions before pre-flight.
            # Give them the same rendered-request count used by the gate,
            # including the active tool catalog.
            self.context.set_token_estimator(_estimate_request_tokens)
        # Server n_ctx fetched from /props on first need. Once known,
        # cfg.context_size is rewritten to match so the fill_ratio math
        # uses the live server window instead of a stale config knob.
        self._server_ctx_synced = False
        self._tool_log: list[tuple[str, str]] = []  # (name, args_summary)
        # Async trace writer — lazy-instantiated by Session.run() when
        # trace_file is set, so tests that poke at internal state
        # without running the loop don't spawn writer threads.
        self._async_trace_writer = None
        # Installed only while ``run`` owns the active trace writer. Tool
        # dispatch uses this recorder to make starts durable before execution
        # and to retain unresolved calls for signal/fatal recovery.
        self._exit_diagnostics = None
        # Number of same-turn follow-up requests, aggregated by the driver
        # into post-run metrics. The initial response is not a continuation.
        self._length_continuation_count = 0
        self._tool_activation_events = 0
        self._activated_tool_names: set[str] = set()
        # Adaptive phase state (config-driven runtime switch).
        self._adaptive_phase = "base"
        self._adaptive_switched = False
        self._observed_test_signal = False
        window = max(0, int(getattr(cfg, "adaptive_low_pressure_window", 0) or 0))
        self._pressure_events = deque(maxlen=window if window > 0 else 1)
        # Byte-identical output dedup maps (tool_name, focus_key) to
        # (sha1[:12], turn_number). Cleared
        # on a successful mutation so post-edit re-reads always reach
        # add_tool_result with fresh bytes.
        self._output_dedup_cache: dict[tuple[str, str], tuple[str, int]] = {}
        # Mechanical state.json writer — harness side, not model side.
        # Gated on state_path being non-None (arm=with_yuj only; wo_yuj runs
        # never seed .solver/state.json and therefore get no state writes).
        # Injection subsystem (harness/injections.py). Off-by-default;
        # when enabled, loads markdown fragments from
        # <cwd>/<cfg.injections_dir> at session start. Fire state is
        # per-session so fire_once fragments inject at most once.
        self._injections = list(injections) if injections is not None else []
        self._injection_state = InjectionState()
        if cfg.injections_enabled and injections is None:
            from .project_instructions import find_project_root

            inj_dir = Path(self.cwd) / cfg.injections_dir
            project_root = find_project_root(
                Path(self.cwd), getattr(cfg, "project_root_markers", ())
            )
            self._injections = load_injections(
                inj_dir,
                imports_enabled=getattr(cfg, "imports_enabled", True),
                imports_max_depth=getattr(cfg, "imports_max_depth", 5),
                allowed_dirs=(project_root,),
                unreadable_paths=_bash_unreadable_paths(
                    cwd, cfg, self._ignore_policy,
                ),
            )
        # Stream-rule files are parsed before the task's first model call by
        # the outer driver. Direct Session construction retains the same loud
        # startup validation path for focused integrations/tests.
        self._stream_rule_runtime: StreamRuleRuntime | None = None
        self._stream_rule_decorated_call_ids: set[str] = set()
        if getattr(cfg, "stream_rules_enabled", False):
            resolved_stream_rules = (
                tuple(stream_rules)
                if stream_rules is not None
                else load_stream_rules(
                    Path(self.cwd) / cfg.stream_rules_dir,
                    display_dir=cfg.stream_rules_dir,
                    allowed_root=Path(self.cwd),
                ).rules
            )
            self._stream_rule_runtime = StreamRuleRuntime(
                resolved_stream_rules,
                repeat_gap=cfg.stream_rules_repeat_gap,
                cwd=Path(self.cwd),
            )
        self._advisor = None
        if (
            bool(getattr(cfg, "advisor_enabled", False))
            and not bool(getattr(client, "is_replay", False))
        ):
            from .advisor import AdvisorRuntime

            self._advisor = AdvisorRuntime(self, self._artifact_dir)

        from .hooks import HookRunner

        hook_run_dir = Path(
            artifact_dir
            or (trace_path.parent if trace_path is not None else self.cwd)
        )

        def _hook_event_sink(fields: dict[str, object]) -> None:
            with self._service_event_lock:
                self._emit(
                    "hook",
                    session_number=self._session_number,
                    turn_number=self._current_turn,
                    **fields,
                )

        self._hook_runner = HookRunner(
            enabled=getattr(cfg, "hooks_enabled", False),
            handlers=getattr(cfg, "hooks", {}),
            task_cwd=self.cwd,
            run_dir=hook_run_dir,
            run_id=hook_run_dir.name,
            session_number=self._session_number,
            # Hook path ownership is a separate host-side safety control. It
            # remains active even when model commands explicitly use none.
            sandbox_required=True,
            event_sink=_hook_event_sink,
            replay=getattr(client, "is_replay", False) is True,
            recorded_events=getattr(client, "hook_events", ()),
        )

    @property
    def active_tool_names(self) -> frozenset[str]:
        """Names in the current profile-filtered model-facing tool surface."""
        return frozenset(
            schema["function"]["name"] for schema in self.model_tool_schemas
        )

    @property
    def model_tool_schemas(self) -> list[dict]:
        """Return the request surface for the current task phase."""
        if self._plan_mode.active:
            return self._plan_tool_schemas
        return self._tool_schemas

    def tool_schema_set_for_phase(self, *, plan_mode_active: bool):
        """Return the validation schema set pinned to one model response."""
        return (
            self._plan_tool_schema_set
            if plan_mode_active
            else self._tool_schema_set
        )

    def is_hidden_tool(
        self, name: str, *, active_names: frozenset[str] | None = None
    ) -> bool:
        """Return whether ``name`` was registered but hidden on a request."""
        return self._tool_surface.is_hidden(name, active_names=active_names)

    def _sync_tool_surface(self) -> None:
        """Atomically rebuild request schemas and their validation view."""
        self._tool_schemas = self._tool_surface.active_schemas
        self._tool_schema_set = ToolSchemaSet.from_openai_tools(
            self._tool_schemas
        )

    def _replace_registered_tool_schemas(
        self, registered_schemas: list[dict], cfg=None
    ) -> None:
        """Refresh config/profile gates without losing prior activations."""
        effective_cfg = cfg or self.cfg
        from ._loop.profile_resolution import _profile_tool_limit

        if self._subagent_level >= int(
            getattr(effective_cfg, "tools_subagent_depth", 1) or 0
        ):
            registered_schemas = [
                schema for schema in registered_schemas
                if schema.get("function", {}).get("name") != "task"
            ]
        if self._subagent_level:
            registered_schemas = [
                schema for schema in registered_schemas
                if schema.get("function", {}).get("name") not in {
                    "ask_user", "subagent_changes", "apply_subagent",
                }
            ]
        if self._tool_allowlist is not None:
            registered_schemas = [
                schema for schema in registered_schemas
                if schema.get("function", {}).get("name")
                in self._tool_allowlist
            ]

        lazy = bool(getattr(
            effective_cfg, "tools_lazy_loading_enabled", False
        ))
        self._tool_surface = replace_tool_surface(
            self._tool_surface,
            registered_schemas,
            lazy_loading_enabled=lazy,
            active_default=getattr(
                effective_cfg, "tools_active_default", ()
            ),
            max_active_tools=(
                _profile_tool_limit(self.client) if lazy else None
            ),
        )
        self._sync_tool_surface()
        from ._loop.profile_resolution import build_plan_mode_schemas
        self._plan_tool_schemas = (
            build_plan_mode_schemas(effective_cfg, self.client)
            if bool(getattr(effective_cfg, "plan_mode_enabled", False))
            else []
        )
        self._plan_tool_schema_set = ToolSchemaSet.from_openai_tools(
            self._plan_tool_schemas or self._tool_schemas
        )

    @property
    def last_tool_calls(self) -> list[tuple[str, str]]:
        """Last N tool calls as (name, args_summary) pairs."""
        return self._tool_log[-self.cfg.duplicate_abort:]

    @property
    def context_fill_ratio(self) -> float:
        """Last known context fill ratio (0.0–1.0)."""
        return self._last_fill

    def _emit_injection_event(
        self, *, rule: str, trigger: str, path: str, turn_number: int | None,
    ) -> None:
        """Write raw conditional-fire metadata without projecting it."""
        emitter = getattr(self, "_emit", None)
        if not callable(emitter):
            return
        emitter(
            "injection",
            session_number=getattr(self, "_session_number", 0),
            turn_number=(
                int(turn_number)
                if turn_number is not None
                else int(getattr(self, "_current_turn", 0))
            ),
            rule=rule,
            trigger=trigger,
            path=path,
        )

    def _apply_injections(self, *, turn_number: int | None = None) -> None:
        """Fire matching injections against the latest user/tool text.

        No-op when the subsystem is disabled or no fragments loaded.
        For each fragment that fires, appends a new user-role message
        containing its ``<injected-fragment source=NAME>`` block so
        the model sees it inline on the next API call, and records a
        per-fire event on the savings ledger (bucket=``injection``,
        mechanism=fragment name).
        """
        if not self._injections:
            return
        messages = self.context.get_messages()
        last_text = ""
        for m in reversed(messages):
            if m.get("role") in ("user", "tool"):
                c = m.get("content", "")
                last_text = c if isinstance(c, str) else str(c)
                break
        fired = fire_candidates(
            self._injections, text=last_text, state=self._injection_state,
        )
        for inj in fired:
            block = inj.format_block()
            add_fragment = getattr(
                self.context, "add_injected_fragment", self.context.add_user
            )
            add_fragment(block)
            record_fire(
                inj.name,
                before="",
                after=block,
                match_mode="keyword" if inj.keywords else "always",
                surface="injected_message",
            )
            if inj.keywords:
                emit_injection = getattr(self, "_emit_injection_event", None)
                if callable(emit_injection):
                    emit_injection(
                        rule=inj.name,
                        trigger="keyword",
                        path="",
                        turn_number=turn_number,
                    )

    def _queue_user_turn_injection(
        self, injection: UserTurnInjection,
    ) -> bool:
        """Queue one next-request fragment, deduplicated within the turn."""
        if injection.text in self._pending_user_turn_texts:
            return False
        self._pending_user_turn_texts.add(injection.text)
        self._pending_user_turn_injections.append(injection)
        return True

    def _queue_execution_user_turn_injections(
        self,
        execution_metadata: Mapping[str, object],
        *,
        tool_call_id: str,
    ) -> None:
        """Collect advice emitted by a tool or nested ``exec_cell`` call."""
        if execution_metadata.get("_user_turn_injections_queued"):
            return
        records = execution_metadata.get("user_turn_injections", ())
        if isinstance(records, (list, tuple)):
            for record in records:
                if isinstance(record, UserTurnInjection):
                    self._queue_user_turn_injection(
                        record.for_tool_call(tool_call_id)
                    )

        cell = execution_metadata.get("exec_cell")
        if isinstance(cell, Mapping):
            inner_calls = cell.get("inner_calls", ())
            if isinstance(inner_calls, (list, tuple)):
                for raw_call in inner_calls:
                    if not isinstance(raw_call, Mapping):
                        continue
                    inner = raw_call.get("execution_metadata")
                    if not isinstance(inner, Mapping):
                        continue
                    index = int(raw_call.get("index") or 0)
                    self._queue_execution_user_turn_injections(
                        inner,
                        tool_call_id=f"{tool_call_id}:cell:{index}",
                    )
        if isinstance(execution_metadata, dict):
            execution_metadata["_user_turn_injections_queued"] = True

    def _deliver_pending_user_turn_injections(self) -> int:
        """Put queued advice into the next outbound synthetic user turn."""
        pending = tuple(self._pending_user_turn_injections)
        if not pending:
            return 0
        add_fragment = getattr(
            self.context, "add_injected_fragment", self.context.add_user
        )
        for injection in pending:
            add_fragment(injection.text)
            record_user_turn_delivery(injection)
            self._emit(
                "user_turn_injection",
                session_number=self._session_number,
                turn_number=int(getattr(self, "_current_turn", 0)),
                bucket=injection.bucket,
                mechanism=injection.mechanism,
                layer=injection.layer,
                delivery="user_turn",
                tool_call_id=injection.tool_call_id,
            )
        self._pending_user_turn_injections.clear()
        self._pending_user_turn_texts.clear()
        return len(pending)

    def _apply_path_injections(
        self,
        result: str,
        *,
        tool_name: str,
        arguments: Mapping[str, object],
        turn_number: int,
        tool_call_id: str,
        executed: bool,
        execution_metadata: Mapping[str, object] | None = None,
        bash_rewritten: bool = False,
    ) -> tuple[str, bool]:
        """Queue matching path fragments after one executed tool call."""
        if (
            not executed
            or not getattr(self.cfg, "injections_path_rules_enabled", False)
            or not self._injections
        ):
            return result, False
        metadata = execution_metadata or {}
        operations = metadata.get("applied_operations", ())
        if not isinstance(operations, (list, tuple)):
            operations = ()
        fired = fire_path_candidates(
            self._injections,
            tool_name=tool_name,
            arguments=arguments,
            cwd=self.cwd,
            state=self._injection_state,
            path_rule_repeat=bool(getattr(
                self.cfg, "injections_path_rule_repeat", False,
            )),
            applied_operations=operations,
            bash_rewritten=bash_rewritten,
        )
        for fire in fired:
            block = fire.injection.format_block(
                trigger="path", path=fire.path,
            )
            self._queue_user_turn_injection(
                UserTurnInjection(
                    text=block,
                    bucket="injection",
                    mechanism=fire.injection.name,
                    tool_call_id=tool_call_id,
                    ctx={
                        "match_mode": "path",
                        "path": fire.path,
                        "tool_name": tool_name,
                    },
                )
            )
            self._emit_injection_event(
                rule=fire.injection.name,
                trigger="path",
                path=fire.path,
                turn_number=turn_number,
            )
        return result, bool(fired)

    def _capture_advisor_turn(
        self, turn: int, content: str | None, tool_calls: list
    ) -> None:
        """Give the advisor only this completed primary response delta."""
        if self._advisor is not None:
            self._advisor.capture_turn(
                turn=turn, content=content, tool_calls=tool_calls
            )

    def _maybe_run_advisor(self, turn: int) -> bool:
        """Run an eligible passive review without failing the primary task."""
        return bool(
            self._advisor is not None
            and self._advisor.review_turn(turn)
        )

    def _inject_pending_advisor(self, turn: int) -> bool:
        """Inject a queued advisory at the next model-request boundary."""
        return bool(
            self._advisor is not None
            and self._advisor.inject_pending(turn)
        )

    def _record_stream_rule_matches(self, records, *, turn: int) -> None:
        """Write one raw trigger row per matched rule, never its body."""
        for record in records:
            self._emit(
                "stream_rule_triggered",
                session_number=self._session_number,
                turn_number=turn,
                rule=str(record.get("rule") or ""),
                scope=str(record.get("scope") or ""),
                offset=int(record.get("offset") or 0),
                path=str(record.get("path") or ""),
                tool_name=str(record.get("tool_name") or ""),
                interrupt=bool(record.get("interrupt", False)),
            )

    def _record_stream_rule_injection(
        self,
        records,
        *,
        turn: int,
        delivery: str,
    ) -> None:
        names = [str(record.get("rule") or "") for record in records]
        if not names:
            return
        self._emit(
            "stream_rule_injection",
            session_number=self._session_number,
            turn_number=turn,
            rules=names,
            delivery=delivery,
            context_mode=getattr(
                self.cfg, "stream_rules_context_mode", "discard"
            ),
        )

    def _apply_pending_stream_rule_injections(self, turn: int) -> None:
        """Deliver deferred prose reminders at the next logical turn."""
        runtime = self._stream_rule_runtime
        if runtime is None:
            return
        records = runtime.take_prose_injections(turn=turn)
        if not records:
            return
        inserted = "\n\n".join(
            format_interrupt_fragment(record) for record in records
        )
        self.context.add_injected_fragment(inserted)
        from .savings import get_ledger
        get_ledger().record_transform(
            bucket="stream_rule_intervention",
            layer="harness",
            mechanism="next_turn_interrupt_fragment",
            before="",
            after=inserted,
            surface="injected_message",
            change_count=len(records),
            ctx={
                "rules": [str(record.get("rule") or "") for record in records],
                "delivery": "next_turn",
            },
        )
        self._record_stream_rule_injection(
            records, turn=turn, delivery="next_turn"
        )

    def _decorate_stream_rule_tool_result(
        self,
        tool_call_id: str,
        result: str,
        *,
        turn: int,
    ) -> str:
        """Prepend queued non-interrupt reminders to their exact tool result."""
        runtime = self._stream_rule_runtime
        if runtime is None:
            return result
        decorated, records = runtime.decorate_tool_result(
            tool_call_id, result, turn=turn
        )
        if records:
            from .savings import get_ledger
            get_ledger().record_transform(
                bucket="stream_rule_intervention",
                layer="harness",
                mechanism="tool_result_reminder",
                before=result,
                after=decorated,
                surface="tool_output",
                change_count=len(records),
                tool_call_id=tool_call_id,
                ctx={
                    "rules": [
                        str(record.get("rule") or "") for record in records
                    ],
                    "delivery": "tool_result",
                },
            )
            self._stream_rule_decorated_call_ids.add(tool_call_id)
            self._record_stream_rule_injection(
                records, turn=turn, delivery="tool_result"
            )
        return decorated

    def _run_hook(self, event: str, **fields: object):
        """Invoke one lifecycle event with the common run/session envelope."""
        return self._hook_runner.run(
            event,
            turn=self._current_turn,
            model=self.cfg.model,
            profile_name=self.cfg.profile_name or self.cfg.model,
            **fields,
        )

    def _add_hook_context(self, effect) -> None:
        """Add a normalized hook annotation to the next model request."""
        block = effect.context_block()
        if block:
            self.context.add_injected_fragment(block)
            from .savings import get_ledger
            get_ledger().record_transform(
                bucket="hook_intervention",
                layer="harness",
                mechanism="lifecycle_hook_context",
                before="",
                after=block,
                surface="injected_message",
                ctx={"delivery": "next_request"},
            )

    def _get_server_ctx(self) -> int:
        from ._loop.compaction import get_server_ctx
        return get_server_ctx(self)

    def _maybe_compact_messages(
        self,
        messages: list[dict],
        *,
        projected_tokens: int | None = None,
    ) -> list[dict]:
        from ._loop.compaction import maybe_compact_messages
        return maybe_compact_messages(
            self,
            messages,
            projected_tokens=projected_tokens,
        )

    def _chat_with_retry(self, turn: int):
        from ._loop.chat_io import chat_with_retry
        return chat_with_retry(self, turn)

    def _write_trace(self, entry: dict) -> None:
        from ._loop.trace_schema import write_trace
        write_trace(self, entry)  # replay hooks live inside write_trace

    def _emit(self, event_type: str, **fields) -> None:
        from ._loop.trace_schema import emit
        emit(self, event_type, **fields)

    def rewind_to(
        self, turn_number: int, *, reason: str = "operator"
    ) -> dict[str, object]:
        """Restore conversation and workspace state at a completed turn."""
        from .turn_snapshots import rewind_to
        return rewind_to(self, turn_number, reason=reason)

    def request_rewind(
        self, turn_number: int | None = None, *, reason: str = "guardrail"
    ) -> None:
        """Queue a rewind for the next complete turn boundary."""
        from .turn_snapshots import request_rewind
        request_rewind(self, turn_number, reason=reason)

    def _refresh_state(self) -> None:
        from ._loop.state_projection import refresh_state
        refresh_state(self)

    def _project_and_sink(self, tc_name: str, cmd: str, result: str, turn: int) -> str:
        from ._loop.state_projection import project_and_sink
        return project_and_sink(self, tc_name, cmd, result, turn)

    def _update_parity_from_parsed(self, parsed: dict) -> None:
        from ._loop.state_projection import update_parity_from_parsed
        update_parity_from_parsed(self, parsed)

    def _sink_to_disk(self, raw: str, turn: int) -> str:
        from ._loop.state_projection import sink_to_disk
        return sink_to_disk(self, raw, turn)

    def run(self) -> SessionResult:
        from ._loop.interrupted_turn import ExitDiagnostics
        from ._loop.run_step import run_session_loop
        from ._loop.persistent_bash import (
            maybe_install_persistent_bash, teardown_persistent_bash,
        )
        from ._loop.trace_async_writer import AsyncTraceWriter
        runner = None
        # Lazy-start the async trace writer only when actually running
        # the loop; tests that construct Session without calling run()
        # never spawn a writer thread.
        if self._trace_file is not None and self._async_trace_writer is None:
            self._async_trace_writer = AsyncTraceWriter(self._trace_file)
        diagnostics = None
        if self._trace_path is not None and self._async_trace_writer is not None:
            diagnostics = ExitDiagnostics(
                self._trace_path,
                session_number=self._session_number,
                sync_before=self._async_trace_writer.barrier,
            )
        try:
            if diagnostics is not None:
                diagnostics.install()
                self._exit_diagnostics = diagnostics
            try:
                runner = maybe_install_persistent_bash(self)
                result = run_session_loop(self)
                self._own_prompt_tokens = result.total_prompt_tokens
                self._own_completion_tokens = result.total_completion_tokens
                if self._subagent_prompt_tokens or self._subagent_completion_tokens:
                    result = replace(
                        result,
                        total_prompt_tokens=(
                            result.total_prompt_tokens
                            + self._subagent_prompt_tokens
                        ),
                        total_completion_tokens=(
                            result.total_completion_tokens
                            + self._subagent_completion_tokens
                        ),
                    )
            except BaseException as exc:
                if diagnostics is not None:
                    diagnostics.record_fatal_exception(exc)
                raise
            if diagnostics is not None:
                if result.finish_reason in {"length", "context_full"}:
                    diagnostics.record_exit(
                        reason=result.finish_reason, kind="truncated"
                    )
                elif result.finish_reason == "sandbox_unavailable":
                    diagnostics.record_exit(
                        reason=result.finish_reason, kind="fatal"
                    )
                else:
                    diagnostics.record_exit(
                        reason="session scope completed", kind="normal"
                    )
            return result
        finally:
            if diagnostics is not None:
                diagnostics.uninstall()
            self._exit_diagnostics = None
            teardown_persistent_bash(runner)
            if self._lsp_manager is not None:
                self._lsp_manager.close()
            if self._process_manager is not None:
                self._process_manager.close()
            if self._terminal_manager is not None:
                self._terminal_manager.close()
            if self._async_trace_writer is not None:
                self._async_trace_writer.stop(timeout=5.0)
                self._async_trace_writer = None


    def _record_pressure_event(self, had_pressure: bool) -> None:
        """Track whether this turn had loop pressure events (errors/blocks/warns)."""
        self._pressure_events.append(bool(had_pressure))

    def _observe_test_signal(self, cmd: str, result: str) -> None:
        from ._loop.adaptive import observe_test_signal
        observe_test_signal(self, cmd, result)

    def _observe_harness_tool_result(
        self,
        *,
        turn: int,
        tool_name: str,
        tool_args: dict | None,
        result: str,
        gate_blocked: bool,
    ) -> None:
        from .harness_observation import observe_tool_result
        observe_tool_result(
            self,
            turn=turn,
            tool_name=tool_name,
            tool_args=tool_args,
            result=result,
            gate_blocked=gate_blocked,
        )

    def _maybe_emit_harness_observation(self, turn: int) -> str | None:
        from .harness_observation import maybe_emit_observation
        return maybe_emit_observation(self, turn=turn)

    def _maybe_run_llm_hurdle_detector(self, turn: int):
        from .adaptive_control.llm_detector import maybe_run_llm_hurdle_detector
        return maybe_run_llm_hurdle_detector(self, turn=turn)

    def _maybe_switch_adaptive_phase(self, turn: int) -> None:
        from ._loop.adaptive import maybe_switch_adaptive_phase
        maybe_switch_adaptive_phase(self, turn)

    def recent_prefix_slots(self, observation_slot: int):
        """Project recorded tool_call events into prefix-only slot facts.

        Read only events whose slot_idx is at most observation_slot. Never read
        future turns, terminal data, or scorer data. Return [] before any tool
        call is recorded.
        """
        from .adaptive_control.slot_recorder import recent_prefix_slots_from_events
        return recent_prefix_slots_from_events(self._trace_events, observation_slot)





# solve_task lives in _loop/driver.py — re-exported here so callers like
# scripts/llm_solver/__main__.py and tests doing
# `from llm_solver.harness.loop import solve_task` keep working unchanged.
from ._loop.driver import (  # noqa: E402, F401
    _load_trace_events,
    _next_session_number,
    build_resume_prompt_from_trace,
    solve_task,
)
