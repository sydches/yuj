"""Agentic loop — Session (inner) + solve_task (outer)."""
from collections import deque
from collections.abc import Callable, Mapping
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

# Tools safe to dispatch concurrently when the flag is set.
_READONLY_TOOLS = PARALLEL_READ_SAFE_TOOL_NAMES
from .injections import (
    InjectionState,
    fire_candidates,
    load_injections,
    record_fire,
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
from .schemas import get_tool_schemas
from .tool_validation import ToolSchemaSet
from .tool_policy import PermissionPolicy
from .sandbox.ignore_policy import IgnorePolicy, load_ignore_policy
from .solver import build_system_prompt, collect_provenance, write_checkpoint, write_run_metrics
from .state_writer import write_state_from_events, write_state_from_trace
from .tools import (
    ToolRegistry, _bash_unreadable_paths, admit_tool_output,
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
    _auto_commit, _canon_focus_path,
    _dedup_signature, _encode_focus_path, _encode_focus_target,
    _extract_bash_focus_target, _extract_test_target_from_command,
    _filter_disabled_tools, _focus_signature, _load_bash_transforms,
    _looks_like_path_token, _normalize_bash_for_dedup,
    _normalize_repo_timestamps, _path_within_cwd, _pretest_is_green,
    _record_session_start_costs, _resolve_profile,
    _resolve_token_estimator, _sanitize_runner_timing,
    _simplify_tool_schema, _split_bash_segments,
    _summarize_args, _truncate_focus_display, _truncate_for_trace,
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
    "task_wall_clock",
    "approval_required",
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
        injections=None,
        artifact_dir: Path | None = None,
        adaptive_control_baseline_config_paths: tuple[str, ...] | list[str] | None = None,
        ignore_policy: IgnorePolicy | None = None,
        effective_env: Mapping[str, str] | None = None,
        allow_login_shell: bool | None = None,
    ):
        self.cfg = cfg
        self._permission_policy = PermissionPolicy.from_rule_tables(
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
        )
        self._session_number = session_number
        self._current_turn = 0
        self._service_event_lock = threading.RLock()
        self._trace_file = trace_file
        self._trace_path = trace_path
        self._state_path = state_path
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
        self._tool_schemas = apply_profile_to_schemas(
            get_tool_schemas(cfg.tool_desc), cfg, client,
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
        self._lsp_manager = lsp_manager
        if self._lsp_manager is None and (
            getattr(cfg, "lsp_enabled", False)
            or getattr(cfg, "lsp_tool_enabled", False)
        ):
            from .lsp_support import LspManager, parse_server_specs

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
                sandbox_required=getattr(cfg, "sandbox_required", False),
                sandbox=bool(getattr(cfg, "sandbox_bash", True)),
                sandbox_backend=getattr(cfg, "sandbox_backend", "bwrap"),
                container_runtime=getattr(
                    cfg, "sandbox_container_runtime", "docker"
                ),
                container_image=getattr(cfg, "sandbox_container_image", ""),
                container_flags=tuple(
                    getattr(cfg, "sandbox_container_flags", ()) or ()
                ),
                effective_env=self._effective_env,
                allow_login_shell=self._allow_login_shell,
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

        base_bash_handler = handlers["bash"]

        def _bash_handler(args, dispatch_cwd, dispatch_cfg):
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

        handlers["lsp"] = _lsp_handler
        handlers["bash"] = _bash_handler
        handlers["bash_poll"] = _bash_poll_handler
        handlers["bash_kill"] = _bash_kill_handler
        from .checkpoint_rewind import build_session_tool_handlers
        handlers.update(build_session_tool_handlers(self))
        self._tool_registry = ToolRegistry(handlers=handlers)
        self._context_checkpoint = None
        self._pending_context_checkpoint = None
        self._pending_context_rewind = None
        self._checkpoint_store = checkpoint_store
        schema_names = [s["function"]["name"] for s in self._tool_schemas]
        validate_tool_handlers(schema_names, registry=self._tool_registry)
        self._tool_schema_set = ToolSchemaSet.from_openai_tools(
            self._tool_schemas
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
        self.context.add_system(system_prompt)
        self.context.add_user(initial_message)
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
                    return admit_tool_output(
                        "bash_poll",
                        value,
                        arguments={},
                        cfg=self.cfg,
                        output_control=self.output_control,
                        redactions=self.redactions,
                    )

                manager_run_dir = Path(
                    artifact_dir
                    or (trace_path.parent if trace_path is not None else cwd)
                )
                self._process_manager = ProcessManager.sandboxed(
                    run_dir=manager_run_dir,
                    cwd=cwd,
                    bwrap_bin=cfg.bwrap_bin,
                    unreadable_paths=_bash_unreadable_paths(
                        cwd, cfg, self._ignore_policy,
                    ),
                    sandbox_required=getattr(cfg, "sandbox_required", False),
                    sandbox=bool(getattr(cfg, "sandbox_bash", True)),
                    sandbox_backend=getattr(cfg, "sandbox_backend", "bwrap"),
                    container_runtime=getattr(
                        cfg, "sandbox_container_runtime", "docker"
                    ),
                    container_image=getattr(
                        cfg, "sandbox_container_image", ""
                    ),
                    container_flags=tuple(
                        getattr(cfg, "sandbox_container_flags", ()) or ()
                    ),
                    effective_env=self._effective_env,
                    allow_login_shell=self._allow_login_shell,
                    max_procs=int(cfg.tools_background_max_procs),
                    poll_timeout_s=float(cfg.tools_background_poll_timeout),
                    admit_output=_admit_poll_output,
                    event_sink=_process_event_sink,
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
            events=self._trace_events,
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
        # Local tokenizer for exact pre-flight token counts in
        # _maybe_compact_messages. None when cfg.tokenizer_id is unset
        # — caller falls back to chars_div_4 estimate.
        from .local_tokenizer import load as _load_tokenizer
        self._tokenizer = _load_tokenizer(getattr(cfg, "tokenizer_id", "") or "")
        if self._tokenizer is not None:
            synced = self._tokenizer.sync_chat_template(
                getattr(cfg, "base_url", "") or "")
            log.info("local tokenizer loaded: %s (server template %s)",
                     self._tokenizer.id, "synced" if synced else "NOT synced — counts approximate")
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

    @property
    def active_tool_names(self) -> frozenset[str]:
        """Names in the current profile-filtered model-facing tool surface."""
        return frozenset(
            schema["function"]["name"] for schema in self._tool_schemas
        )

    @property
    def last_tool_calls(self) -> list[tuple[str, str]]:
        """Last N tool calls as (name, args_summary) pairs."""
        return self._tool_log[-self.cfg.duplicate_abort:]

    @property
    def context_fill_ratio(self) -> float:
        """Last known context fill ratio (0.0–1.0)."""
        return self._last_fill

    def _apply_injections(self) -> None:
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
            self.context.add_user(block)
            record_fire(
                inj.name, body_chars=len(block), match_mode=inj.trigger,
            )

    def _get_server_ctx(self) -> int:
        from ._loop.compaction import get_server_ctx
        return get_server_ctx(self)

    def _maybe_compact_messages(self, messages: list[dict]) -> list[dict]:
        from ._loop.compaction import maybe_compact_messages
        return maybe_compact_messages(self, messages)

    def _chat_with_retry(self, turn: int):
        from ._loop.chat_io import chat_with_retry
        return chat_with_retry(self, turn)

    def _write_trace(self, entry: dict) -> None:
        from ._loop.trace_schema import write_trace
        write_trace(self, entry)  # replay hooks live inside write_trace

    def _emit(self, event_type: str, **fields) -> None:
        from ._loop.trace_schema import emit
        emit(self, event_type, **fields)

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
            except BaseException as exc:
                if diagnostics is not None:
                    diagnostics.record_fatal_exception(exc)
                raise
            if diagnostics is not None:
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
