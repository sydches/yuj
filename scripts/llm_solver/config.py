"""Configuration loader — TOML defaults + user overrides + CLI flags.

Layered config resolution:
  1. config.toml        (project root, checked into git)
  2. config.local.toml  (same directory, gitignored, optional)
  3. CLI overrides       (highest priority)

Project root is located by walking up from this file until config.toml is found,
or via the YUJ_CONFIG env var pointing directly to config.toml.
"""
import logging
import os
import copy
from dataclasses import dataclass, field
from pathlib import Path

from ._shared.post_edit_spec import validate_post_edit_check_dict
from ._shared.toml_compat import tomllib

VALID_RUNTIME_MODES = ("measurement", "assistant")


def _find_project_root() -> Path:
    """Walk up from this file to find config.toml, or use YUJ_CONFIG env var."""
    env = os.environ.get("YUJ_CONFIG")
    if env:
        p = Path(env)
        if p.is_file():
            return p.parent
        raise FileNotFoundError(f"YUJ_CONFIG={env} does not exist")

    d = Path(__file__).resolve().parent
    for _ in range(10):
        if (d / "config.toml").is_file():
            return d
        parent = d.parent
        if parent == d:
            break
        d = parent
    raise FileNotFoundError(
        "config.toml not found. Set YUJ_CONFIG or run from project root."
    )


PROJECT_ROOT = _find_project_root()
_DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"
_LOCAL_CONFIG = PROJECT_ROOT / "config.local.toml"


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a configured runtime path against the harness project root."""
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base (overlay wins)."""
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _load_layered() -> dict:
    """Load config.toml, merge config.local.toml if present."""
    data = _load_toml(_DEFAULT_CONFIG)
    if _LOCAL_CONFIG.is_file():
        _deep_merge(data, _load_toml(_LOCAL_CONFIG))
    return data


# ---------------------------------------------------------------------------
# Model aliases — exported for other scripts
# ---------------------------------------------------------------------------

def _build_model_map(data: dict) -> dict[str, str]:
    return dict(data.get("models", {}).get("aliases", {}))


# Eagerly loaded so importers can do: from scripts.llm_solver.config import MODEL_MAP
_LAYERED = _load_layered()
MODEL_MAP: dict[str, str] = _build_model_map(_LAYERED)


# ---------------------------------------------------------------------------
# SDK / CLI section accessors
# ---------------------------------------------------------------------------

def get_sdk_config() -> dict:
    """Return the [sdk] section with model alias resolved."""
    section = dict(_LAYERED.get("sdk", {}))
    model = section.get("default_model", "sonnet")
    section["default_model_resolved"] = MODEL_MAP.get(model, model)
    return section


def get_cli_config() -> dict:
    """Return the [cli] section with model alias resolved."""
    section = dict(_LAYERED.get("cli", {}))
    model = section.get("default_model", "haiku")
    section["default_model_resolved"] = MODEL_MAP.get(model, model)
    return section


def get_server_base_url() -> str:
    """Return ``[server] base_url`` from the layered config.

    Used by CLI code that needs the scheme+host before building a full Config.
    """
    return _require(_LAYERED, "server", "base_url")  # type: ignore[return-value]


def get_server_config() -> dict:
    """Return the ``[server]`` section for callers outside the Config dataclass.

    Read by tools (e.g. ``run_scenarios``) that only need transport settings.
    """
    return dict(_LAYERED.get("server", {}))


def get_model_default_max_tokens() -> int:
    """Return derived max_tokens for tools that build ad-hoc OpenAI clients.

    Derived from ``context_size * max_tokens_fraction``. The fraction
    knob (default 0.25) replaces a hardcoded ``max_tokens`` integer that
    was wrong-shape whenever ``context_size`` was anything other than
    65,536 (e.g., 8k generation on a 32k-ctx server, 32k generation on
    a 128k-ctx server).
    """
    ctx = int(_require(_LAYERED, "model", "context_size"))  # type: ignore[arg-type]
    frac = float(_require(_LAYERED, "model", "max_tokens_fraction"))  # type: ignore[arg-type]
    return int(ctx * frac)


# ---------------------------------------------------------------------------
# llm_solver Config dataclass (existing interface, preserved)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str
    timeout_connect: int
    timeout_read: int
    health_poll_interval: int
    health_timeout: int
    launch_timeout: int
    stop_settle: int
    model: str
    # Profile dir name under profiles/. Decouples the wire `model:` field
    # (sent on the OpenAI request, must match the runtime's served name)
    # from the harness profile lookup. Empty string falls back to `model`,
    # which preserves the legacy behavior. Set this in a runtime overlay
    # to share one profile (chat template, normalize/denormalize, and
    # behavioral suffix) across several runtimes for the same model.
    profile_name: str
    context_size: int
    context_fill_ratio: float
    max_tokens_fraction: float
    max_tokens: int
    tokenizer_id: str
    max_turns: int
    max_sessions: int
    duplicate_abort: int
    error_nudge_threshold: int
    rumination_nudge_threshold: int
    require_intent: bool
    intent_grace_turns: int
    min_turns_before_context: int
    max_output_chars: int
    truncate_head_ratio: float
    truncate_head_lines: int
    truncate_tail_lines: int
    args_summary_chars: int
    trace_args_summary_chars: int
    trace_reasoning_store_chars: int
    solver_trace_lines: int
    solver_evidence_lines: int
    solver_inference_lines: int
    recent_tool_results_chars: int
    trace_stub_chars: int
    trace_reasoning_chars: int
    pretest_head_chars: int
    pretest_tail_chars: int
    bash_timeout: int
    grep_timeout: int
    pretest_timeout: int
    llama_server_bin: str
    sandbox_bash: bool
    strip_ansi: bool
    collapse_blank_lines: bool
    collapse_duplicate_lines: bool
    collapse_similar_lines: bool
    bwrap_bin: str
    # When True, refuse to run bash unsandboxed if bwrap is requested but
    # missing. Set this for any run that must keep the model from reading
    # the host file system.
    sandbox_required: bool
    # Glob patterns expanded at sandbox-build time and mounted over with
    # /dev/null (files) or empty tmpfs (dirs) so the model can `cat` the
    # path but reads return EOF / empty listing. This blocks protected
    # files without relying on a list of forbidden shell commands. Empty
    # tuple = no behavior change.
    unreadable_paths: tuple[str, ...]
    max_transient_retries: int
    retry_backoff: tuple[int, ...]
    # Prompt fragments (session boundaries, gates, nudges)
    system_header: str
    state_context_suffix: str
    intent_gate_first: str
    intent_gate_repeat: str
    resume_base: str
    error_nudge: str
    rumination_nudge: str
    rumination_gate: str
    rumination_same_target_nudge: str
    rumination_outside_cwd_nudge: str
    test_read_nudge: str
    contract_commit_warn: str
    contract_commit_block: str
    contract_recovery_block: str
    mutation_repeat_warn: str
    mutation_repeat_block: str
    resume_duplicate_abort: str
    resume_context_full: str
    resume_max_turns: str
    resume_length: str
    resume_last_n_actions: int
    tool_desc: str = "minimal"
    # Operator/guardrail rewind of the canonical model-facing conversation
    # together with its shadow-Git workspace checkpoint. Off by default.
    rewind_enabled: bool = False
    rewind_max_per_session: int = 1
    interrupted_turn_mode: str = "mechanical"
    length_continue_max: int = 0
    project_docs_enabled: bool = False
    project_doc_names: tuple[str, ...] = ("AGENTS.md", "CLAUDE.md")
    project_doc_max_bytes: int = 32768
    project_root_markers: tuple[str, ...] = (".git", ".hg", ".sl")
    project_doc_global_dir: str = "~/.config/yuj"
    imports_enabled: bool = True
    imports_max_depth: int = 5
    skills_enabled: bool = False
    skills_dirs: tuple[str, ...] = (
        "~/.pi/agent/skills",
        "~/.agents/skills",
        ".pi/skills",
        ".agents/skills",
    )
    skill_paths: tuple[str, ...] = ()
    # Effective, validated roots fixed by startup discovery. This is not a
    # user knob: it lets read and shell sandboxes expose only loaded skills.
    skills_readable_dirs: tuple[str, ...] = ()
    prompt_addendum: str = ""
    variant_name: str = ""
    runtime_mode: str = "measurement"
    permissions_rules: dict[str, object] = field(default_factory=dict)
    permissions_ask_fallback: str = "deny"
    analysis_task_format: str = "auto"  # resolved per-repo via detect_runner; multilingual default
    provider: str = "openai-compatible"
    rumination_gate_max_blocks: int = 0
    resume_gate_escalation: str = "Session ended: {n} consecutive tool calls were blocked by the rumination gate. Your current code has been preserved."
    # Templates for finish reasons that need tailored recovery advice.
    # Without these, each case would fall through
    # to the generic resume_base — the model never saw a tailored hint.
    resume_done_loop: str = "Previous session was ended by the done-loop failsafe — repeated done() calls were rejected by the gate. Your current code is preserved; address the rejection cause before claiming done again."
    resume_mutation_repeat_abort: str = "Previous session was ended by mutation_repeat — the same edit was issued repeatedly with no change. Take a different approach to the failing case."
    resume_contract_recovery_abort: str = "Previous session was ended by contract_recovery — repeated same-target inspection without progress. Mutate or read a different test target."
    resume_contract_commit_abort: str = "Previous session was ended by contract_commit — broad exploration after a commit was already armed. Commit the pending change before further reads."
    resume_intent_abort: str = "Previous session was ended after {n} consecutive empty/silent rejected turns. Be explicit about what tool you are calling and why on the next turn."
    resume_loop_detect: str = "Previous session was ended by loop_detect — {streak} identical tool calls in a row. The loop detector blocked the call; pick a materially different next action."
    resume_no_tool_call: str = "Previous session ended without a tool call. Either call done() to declare success or call an investigation/edit tool to continue."
    resume_error: str = "Previous session ended with an error. Continue from the current state."
    resume_stop: str = "Previous session ended via natural stop. Continue from the current state."
    # Model-facing strings emitted by guardrails (done_guard, rumination_gate)
    # and by the sink-and-surface mechanism in loop.py. Kept in config per
    # the SoD anti-pattern "prompt text in harness code" — harness code
    # should carry no model-facing text directly.
    done_reject_no_mutation: str = "REJECTED: No code changes since session start. Use the selected file-edit tool to modify the code, then call done."
    done_reject_no_verify: str = "REJECTED: No successful verification since the last code change. Either run_tests must report status=\"passed\", or a bash command must exit 0 with substantial output (>200 chars). Then call done."
    done_loop_abort_after: int = 5
    done_loop_abort_text: str = "Session ended: {n} consecutive done() rejections. Your current code has been preserved as the final patch."
    done_reject_parity_no_run: str = "REJECTED: done_require_pretest_parity is on but no structured test run has been observed yet. Run the test suite first."
    done_reject_parity_still_failing: str = "REJECTED: pretest-failing tests not yet passing: {shown}{extra}"
    done_reject_parity_regression: str = "REJECTED: regression — previously-passing tests now failing: {shown}{extra}"
    done_reject_parity_streak: str = "REJECTED: pretest parity observed {count} time(s); need {required}. Run tests again to confirm."
    rumination_gate_grace_prefix: str = "[HARNESS: Gate armed. Next call must mutate a file — all else blocked.]"
    pre_mutation_gate: str = "[HARNESS: {turn_number} read-only turns elapsed without a file mutation; the next tool call must use the selected file-edit tool, run a bash command that mutates a source file, or call done(). This call was not executed.]"
    sink_pointer: str = '<tool_result_meta truncated="true" original_bytes="{chars}" original_lines="{lines}" full_path="{path}"/>'
    # Head/tail slice sizes around the sink marker — prompt literals
    # previously hardcoded in loop.py::_filter_bash_output live here
    # so the whole sink-surface shape is config-adjustable.
    sink_head_bytes: int = 1000
    sink_tail_bytes: int = 1000
    sink_body_marker: str = "... [body truncated — full output available via full_path attribute] ..."
    # read() tool system-reminder text (placeholders filled at call site).
    read_truncated_reminder: str = (
        "<system-reminder>Read returned the first {returned_lines} lines of "
        "{path}. The file is longer — re-read with a higher limit or a "
        "specific offset to see more.</system-reminder>"
    )
    read_empty_reminder: str = (
        "<system-reminder>File {path} exists but is empty (0 bytes)."
        "</system-reminder>"
    )
    read_offset_past_eof_reminder: str = (
        "<system-reminder>Read offset {offset} is past EOF for {path} "
        "(file has {total} lines). No content returned — call read "
        "with a smaller offset to inspect the file.</system-reminder>"
    )
    # Guardrail escalation ladders (additions, 0 = disabled).
    # Each guardrail follows: WARN (append text) → BLOCK (reject call) → END (end session).
    rumination_gate_arm_threshold: int = 0  # % of max_turns; values above nudge add a warning window. 0 = activate at nudge.
    rumination_gate_arm_threshold_abs: int = 0  # Absolute non-write count; when > 0, overrides percentage. Decouples gate from max_turns.
    rumination_nudge_threshold_abs: int = 0  # Absolute non-write count for nudge; when > 0, overrides percentage. Decouples nudge from max_turns.
    rumination_nudge_threshold_abs_post_mutation: int = 0  # Absolute non-write count for nudge AFTER state.has_mutated flips True. When > 0, overrides the pre-mutation threshold for post-mutation rumination. When 0, post-mutation uses the same threshold as pre-mutation.
    rumination_nudge_only_pre_mutation: bool = False  # When True, suppress nudge entirely after state.has_mutated=True. Equivalent to post_mutation_threshold = infinity.
    rumination_same_target_warn_count: int = 0  # Repeated same-target non-write calls before same-target nudge (0 = disabled).
    rumination_same_target_arm_count: int = 0  # Repeated same-target non-write calls before arming the rumination gate (0 = disabled).
    test_read_warn_after: int = 0  # Verification runs without reading the target test file before nudging (0 = disabled).
    context_inspect_repeat_threshold: int = 0  # Repeated inspect actions before concise/yconcise switch to an exit-inspect obligation (0 = disabled).
    contract_commit_warn_after: int = 0  # After a source-file read, warn on non-commit actions after N violations (0 = disabled).
    contract_commit_block_after: int = 0  # After a source-file read, block non-commit actions after N violations (0 = disabled).
    contract_recovery_same_target_threshold: int = 0  # Activate recovery after N same-target non-write actions (0 = disabled).
    contract_recovery_verify_repeat_threshold: int = 0  # Activate recovery after N verify runs against the same target without mutation (0 = disabled).
    contract_invalid_repeat_abort_after: int = 0  # End session after N repeated blocked contract violations with the same target/signature (0 = disabled).
    contract_abort_min_turns_since_commit_arm: int = 0  # Minimum calls after commit mode starts before abort (0 = disabled).
    contract_abort_min_turns_since_recovery_arm: int = 0  # Minimum calls after recovery starts before abort (0 = disabled).
    contract_abort_requires_zero_mutation: bool = False  # If True, contract abort is allowed only before the first successful mutation.
    contract_equivalent_action_classes_enabled: bool = False  # Collapse semantically-equivalent off-contract moves into one violation class.
    mutation_repeat_warn_after: int = 0  # Warn when repeating the same successful mutation N times in a row (0 = disabled).
    mutation_repeat_block_after: int = 0  # Block when repeating the same successful mutation N times in a row (0 = disabled).
    mutation_repeat_abort_after: int = 0  # End session after N blocked identical mutation retries (0 = disabled).
    duplicate_warn_count: int = 0  # append warning text at N identical consecutive calls (0 = disabled)
    duplicate_warn: str = "[harness: {count} identical tool calls in a row. Change approach — session ends at {abort} identical.]"
    error_abort_threshold: int = 0  # end session after N consecutive errors of any kind (0 = disabled)
    error_same_class_threshold: int = 0  # end session after N errors with the same signature (exit code or first-token error string), regardless of interleaved non-error turns. 0 = disabled. Catches the "model repeats the same wrong fix" pattern that error_abort_threshold misses because intervening mutation calls reset its counter.
    intent_abort_threshold: int = 0  # end session after N consecutive silent intent-gate rejections (0 = disabled)
    # ── Guardrail enabled flags (default True = preserve current behaviour).
    # Set a flag to False in a config overlay to disable that guardrail.
    # ``require_intent`` remains the intent-gate setting.
    duplicate_guard_enabled: bool = True
    # Post-edit validation runs per-extension checks after matching mutations.
    post_edit_check_enabled: bool = False
    post_edit_check_timeout: int = 10
    # List of declared check dicts. Each dict has: name, trigger,
    # when, cmd, on_fail. Empty list = no-op.
    post_edit_checks: list = field(default_factory=list)
    # Model-callable run_tests tool (pytest with deterministic flags
    # inside the sandbox). Filtered from the schema list when disabled
    # so the model does not see a tool it cannot use.
    tools_lazy_loading_enabled: bool = False
    tools_active_default: tuple[str, ...] = (
        "bash", "read", "edit", "glob", "grep", "done",
    )
    tools_run_tests_enabled: bool = False
    # Allow long package setup and collection before a test run times out.
    tools_run_tests_timeout: int = 240
    # Failing-assertion source-context auto-extraction (run_tests.py:158).
    # On `failed` / `collection_error` verdicts, parse pytest --tb=short
    # frames and append a snippet of surrounding source so the model
    # has the failing code in the same turn the verdict arrived. `lines`
    # is the radius (before/after); `max` caps how many frames get a
    # context block (token-cost guardrail when many tests fail at once).
    tools_run_tests_assertion_context_lines: int = 5
    tools_run_tests_assertion_context_max: int = 3
    # Side-effect-free model scratchpad. Disabled by default. Raw trace keeps
    # every call; model-facing contexts retain each argument for this many
    # turns before removing it.
    tools_think_enabled: bool = False
    tools_think_keep_turns: int = 4
    think_streak_nudge_after: int = 3
    # list_definitions tool — Python-AST source outline. Disabled by
    # default. Enable it through config.local.toml or another overlay.
    tools_list_definitions_enabled: bool = False
    # Optional repository-wide tree-sitter symbol definition/reference mode.
    tools_ast_search_enabled: bool = False
    tools_ast_search_max_rows: int = 1000
    # Conversation checkpoint/rewind pair. Both model tools share this one
    # default-off gate and operate on context only, never workspace files.
    tools_checkpoint_enabled: bool = False
    # Independent shadow-Git checkpoints after every potentially mutating
    # model tool call. The store is outside the task cwd and restore remains
    # a harness/operator function, never a model-facing tool.
    tools_file_checkpoints_enabled: bool = False
    tools_file_checkpoints_exclude: tuple[str, ...] = (
        ".solver/**",
        ".tool_output/**",
        "prompt.txt",
        "checkpoint.json",
        "metrics.json",
    )
    # Session-local read-before-edit policy, rebuilt from raw trace events.
    tools_stale_guard_mode: str = "warn"
    # Compound-aware shell redirects. Read-side interception is an opt-in;
    # write-side rules remain gated by the availability of their target tool.
    tools_bash_redirect_read_side: bool = False
    tools_schema_validation: str = "off"
    tools_constrained_decoding: str = "off"
    # Session-scoped model-authored todo list. The handler validates and the
    # trace/state pipeline owns persistence; this is not a source mutation.
    tools_todos_enabled: bool = False
    tools_todos_max_items: int = 20
    tools_background_enabled: bool = False
    tools_background_max_procs: int = 4
    tools_background_poll_timeout: float = 300.0
    # Sequential nested harness sessions. The task tool is absent from the
    # model-facing schema until explicitly enabled. Depth counts child edges
    # from the root session (root=0), and the global turn limit caps each
    # agent descriptor's own max_turns value.
    tools_task_enabled: bool = False
    tools_subagent_depth: int = 1
    tools_subagent_max_turns: int = 20
    # Code mode replaces the native schema catalog with three meta-tools and
    # executes model-written Python inside the selected fail-closed sandbox.
    tools_exec_cell_enabled: bool = False
    tools_exec_cell_timeout: int = 30
    # First-class shell sandbox backend. Container mode creates one ephemeral
    # Docker/Podman container per command and preserves the absolute cwd.
    sandbox_backend: str = "bwrap"
    sandbox_container_runtime: str = "docker"
    sandbox_container_image: str = ""
    sandbox_container_flags: tuple[str, ...] = ()
    # Explicit environment passed only to sandboxed/model-command children.
    # Provider clients and the harness process retain their host environment.
    sandbox_env_inherit: str = "core"
    sandbox_env_set: dict[str, str] = field(default_factory=lambda: {
        "FORCE_COLOR": "0",
        "MPLCONFIGDIR": "/tmp/mpl",
        "NO_COLOR": "1",
        "PAGER": "cat",
        "PYTHONIOENCODING": "utf-8",
        "TERM": "dumb",
    })
    sandbox_env_filters: dict[str, str] = field(default_factory=dict)
    sandbox_env_ignore_default_excludes: bool = False
    sandbox_env_allow_login_shell: bool = False
    runtime_worktree: str = "off"
    # Lazy language-server diagnostics and optional navigation tool.
    lsp_enabled: bool = False
    lsp_servers: dict[str, object] = field(default_factory=dict)
    lsp_diagnostics_timeout_s: float = 2.0
    lsp_min_severity: str = "error"
    lsp_tool_enabled: bool = False
    # Compatibility selector for old overlays. New settings use
    # tools.edit_format = "apply_patch" instead.
    tools_apply_patch_enabled: bool = False
    # Per-run override for the profile's edit dialect. Empty inherits the
    # selected model profile. `effective_edit_format` is filled mechanically
    # after profile resolution and is never a user-authored setting.
    tools_edit_format: str = ""
    effective_edit_format: str = ""
    # Unified <tool_result> envelope. When
    # true, every dispatched tool result is wrapped in
    #   <tool_result tool_name="..." status="..." [error_kind="..."]>
    #   …content (with legacy in-band markers preserved)…
    #   </tool_result>
    # so readers use one status field. Inner markers (ERROR: /
    # [exit code: N] / [harness gate]) stay in the body. Classifiers read
    # the envelope first and fall back to those markers when it is absent.
    tools_unified_envelope_enabled: bool = True
    # Per-task wall-clock budget in seconds. 0 disables it. When >0, solve_task
    # checks elapsed wall after each session and ends with
    # finish_reason="task_wall_clock" if over budget. This bounds the
    # time that one task can use.
    task_wall_clock_limit_s: int = 0
    # Collapse byte-identical tool outputs to a one-line back-reference.
    # When the model issues the same tool call on the same target and
    # gets the same bytes back, the second result is replaced with
    # `[harness: identical to turn N's output for <focus>]` and the
    # bytes are recorded under savings bucket `output_dedup`. Cache is
    # per-Session, scoped by (tool_name, focus_key), and cleared on a
    # successful mutation. Disabled by setting False for comparison.
    tools_output_dedup_enabled: bool = True
    # Implicit-done semantics. When True (default, backward-compat):
    # `finish_reason="stop"` with no tool calls counts as task success
    # (done=True). When False: counts as session end with done=False
    # and finish_reason="no_tool_call", forcing the model to call the
    # explicit `done` tool to claim success. Recommended True for
    # frontier models; recommend False for tool-trained local models
    # that occasionally narrate without a tool call.
    allow_implicit_done: bool = True
    # When true, run_tests wraps output in <test_results status="..."
    # exit_code="N">…</test_results>. Status discriminates pytest exit
    # codes 1/2/5 (which are easy to confuse in raw output) and
    # separates a hard timeout from any other failure path. False
    # reverts to a raw bash-string contract.
    tools_run_tests_structured_output: bool = True
    # State-entanglement factorial knobs. Let the harness decouple
    # (1) .solver/ dir presence (controlled at prepare time) from
    # (2) state writer activity and (3) context strategy reading state.json.
    # Both default to on (preserves current with_yuj behavior).
    state_writer_enabled: bool = True
    context_ignore_state: bool = False
    state_imperative_projection_enabled: bool = False
    state_todos_char_budget: int = 2000
    state_ignore_file_enabled: bool = True
    state_ignore_file_names: tuple[str, ...] = (".yujignore",)
    # Paginated search envelopes for grep/glob. Defaults ship on.
    search_pagination_enabled: bool = True
    grep_max_matches_per_page: int = 25
    glob_max_matches_per_page: int = 25
    # tool_quirks/glob caps — refuse panic globs (whole-repo `**/*` fishing
    # expeditions). 0 disables the listing cap (pagination still applies).
    tools_glob_max_listed_paths: int = 50
    tools_glob_refuse_unscoped_recursive: bool = True
    # bash_quirks/forbidden — knob-controlled toggle for the forbidden-
    # pattern layer (cd /, cd /home/<other>, etc.).
    bash_quirks_forbidden_enabled: bool = True
    # Orientation gate — block non-write tool calls past N orientation
    # turns when no mutation has happened yet.
    pre_mutation_turn_cap: int = 0  # 0 = disabled
    # Digest compaction trigger — fires when the exact pre-flight token
    # count (via the local tokenizer) crosses the derived threshold:
    #     threshold = (1 - max_tokens_fraction) - digest_compaction_safety_margin
    # The derivation guarantees compaction fires before any non-compacted
    # turn would land est_pt + max_tokens past ctx (server-side OOM).
    # Re-fires on every crossing.
    digest_compaction_safety_margin: float = 0.05
    digest_keep_recent_turns: int = 8
    digest_compaction_gate_min_mutations: int = 0
    # Ranked repository symbol map appended to the stable task message.
    # Zero preserves the existing prompt exactly.  The refresh policy owns
    # only the run-private structural cache; one rendered map stays immutable
    # for the lifetime of a solver session so prompt-prefix reuse is stable.
    repo_map_tokens: int = 0
    repo_map_refresh: str = "auto"
    compaction_method: str = "digest"
    compaction_hook: str = ""
    checkpoint_keep_recent_tokens: int = 0
    checkpoint_max_summary_tokens: int = 4000
    # Optional model-written fresh-session handoff. The existing mechanical
    # resume prompt remains the exact fallback whenever this is disabled or
    # the side request fails validation.
    handoff_summary_enabled: bool = False
    handoff_max_tokens: int = 2000
    # OpenAI-compatible llama-server request controls. Custom fields are
    # transported under SDK extra_body; cache policy is merged last.
    server_request_extra: dict[str, object] = field(default_factory=dict)
    cache_affinity: bool | int = False
    cache_retention: str = "off"
    cache_miss_warn_ratio: float = 0.0
    thinking_level: str = "off"
    model_roles: dict[str, object] = field(default_factory=dict)
    model_fallback_chain: dict[str, object] | list[object] = field(
        default_factory=dict
    )
    model_fallback_revert: str = "never"
    # Passive second-opinion model. The empty target fields reuse the main
    # profile/model endpoint while retaining an isolated advisor conversation.
    advisor_enabled: bool = False
    advisor_model: str = ""
    advisor_endpoint: str = ""
    advisor_every_n_turns: int = 5
    advisor_immune_turns: int = 3
    advisor_max_note_chars: int = 1200
    # edit() match policy. Strict is the default (database-of-
    # primitives principle: no silent relaxation). Cascade restores
    # the optional auto-apply behavior.
    edit_strict_match: bool = True
    edit_fuzzy_cascade_enabled: bool = False
    edit_candidate_count: int = 3
    # loop_detect guardrail (N consecutive identical tool-call signatures).
    # WARN on first reach-threshold (inject recovery text). END if the
    # pattern repeats once more after the warning. Enabled by default because
    # identical bash signatures can repeat without a guard, and the
    # threshold=5 ceiling makes this cheap (one recovery-inject before
    # END limits collateral). Set to False to disable this guardrail.
    loop_detect_enabled: bool = True
    loop_detect_threshold: int = 5
    # Invisible per-turn git snapshots of the workspace (rewind/branch
    # points). After each executed source-write turn, a dangling git
    # commit records the full workspace state; the turn->sha map lives
    # in the telemetry dir. Invisible to the model by construction
    # (plumbing objects, no ref, private index). Telemetry-grade: any
    # failure logs once and never affects the solve. See
    # harness/turn_snapshots.py. This stays on by default so a run can
    # restore the files from any recorded step.
    turn_snapshots_enabled: bool = True
    # Parallel read-only tool dispatch. When enabled and the turn's
    # tool_calls are all read-only (>1 call, no mutation/bash),
    # dispatch() runs concurrently via a ThreadPoolExecutor. Guardrail
    # state still updates sequentially per-tc after concurrent I/O.
    parallel_readonly_enabled: bool = False
    parallel_max_workers: int = 4
    # Injection subsystem (keyword/path-triggered markdown fragments).
    # Off by default; data-directory convention .harness/injections/.
    injections_enabled: bool = False
    injections_dir: str = ".harness/injections"
    injections_path_rules_enabled: bool = False
    injections_path_rule_repeat: bool = False
    loop_detect_recovery: str = (
        "<system-reminder>Loop detected: the last {streak} tool calls all "
        "have identical name and arguments. Stop repeating. Re-read the "
        "task, read a file you have not inspected yet, or change approach. "
        "One more repeat ends the session.</system-reminder>"
    )
    done_guard_enabled: bool = True
    rumination_enabled: bool = True
    error_ladder_enabled: bool = True
    # Pre-flight overflow backstop: when the projected prompt exceeds
    # context_fill_ratio at the top of a turn, re-clip the single
    # largest oversized message in token space (head+tail, ctx/2-token
    # budget, visible notice) and re-project once before ending the
    # session context_full. See _loop/compaction.py::preflight_reclip_oversized.
    preflight_reclip_enabled: bool = True
    bash_transforms_universal_enabled: bool = True
    bash_transforms_task_format_enabled: bool = True
    bash_transforms_structured_output_enabled: bool = False  # parse test output into digest; replace raw with digest in context
    bash_transforms_sink_threshold_chars: int = 0  # write raw bash output to .tool_output/ when result exceeds N chars (0 = disabled)
    # ── Guardrail internals surfaced from hardcoded values.
    rumination_gate_grace_calls: int = 1       # warned non-write calls allowed before full blocking
    rumination_min_threshold: int = 6          # absolute floor on the derived rumination nudge threshold
    done_require_mutation: bool = True         # done_guard: accept only after at least one successful mutation
    done_require_verify: bool = True           # done_guard: accept only after verified_since_mutation flipped
    done_verified_bash_min_chars: int = 200    # content-blind threshold for "substantial" bash run that counts as verification
    done_require_pretest_parity: bool = False  # done_guard: accept only when latest test run matches the pretest-failing set now PASSED and no pretest-passing regressed (requires [output_parser])
    done_parity_runs_required: int = 1         # number of consecutive parity-green runs required before done accepts (guards against flakiness)
    # ── Adaptive policy controller (config-driven phase switch).
    adaptive_policy_enabled: bool = False
    adaptive_switch_min_turn: int = 0
    adaptive_requires_mutation: bool = True
    adaptive_requires_test_signal: bool = True
    adaptive_low_pressure_window: int = 0
    adaptive_low_pressure_max_events: int = 0
    adaptive_phase2_done_guard_enabled: bool = True
    adaptive_phase2_bash_task_format_enabled: bool = True
    adaptive_phase2_bash_structured_output_enabled: bool = True
    adaptive_phase2_bash_sink_threshold_chars: int = 0
    # ── Adaptive hurdle control (in-process pause hook).
    # This control is separate from the adaptive_policy phase switch above.
    # It stays off unless a runtime overlay enables it.
    adaptive_control_enabled: bool = False
    # Delivery mode for control changes:
    #   "in_place" applies an overlay during the current session.
    #   "stop_resume" writes a stop note and ends the session. The caller
    #   resumes the work and may apply the overlay named in that note.
    # Both modes use the same cadence, watch window, cooldown, and limits.
    adaptive_control_delivery: str = "user_turn"
    adaptive_control_ledger_path: str = ""
    adaptive_control_evidence_regime: str = "causal_live"
    adaptive_control_model: str = "in_process_pause"
    adaptive_control_target_hurdle_mode: str = ""
    adaptive_control_source_hindsight_hurdle_mode: str = ""
    adaptive_control_online_signal_id: str = ""
    adaptive_control_online_signal_ids: tuple[str, ...] = ()
    adaptive_control_intervention_target: str = ""
    adaptive_control_candidate_medicine_knob: str = ""
    adaptive_control_candidate_config_path: str = ""
    adaptive_control_baseline_config_paths: tuple[str, ...] = ()
    adaptive_control_source_static_cell_id: str = ""
    adaptive_control_source_instance_id: str = ""
    adaptive_control_source_wave_id: str = ""
    adaptive_control_source_cell_id: str = ""
    adaptive_control_source_run_dir: str = ""
    adaptive_control_debug: str = "none"  # none | summary | verbose
    adaptive_control_debug_ledger_path: str = ""
    adaptive_control_debug_include_prefix: bool = False
    # Runtime lookup selection, versions, and disabled-by-default budget controls.
    adaptive_control_lookup_table_path: str = ""
    adaptive_control_policy_version: str = "adaptive_policy_v0_replay"
    adaptive_control_detector_mode: str = "manual"  # manual | adaptive
    adaptive_control_detector_version: str = "zero_detector_v0"
    adaptive_control_detector_input_contract_path: str = ""
    adaptive_control_detector_rule_catalog_path: str = ""
    adaptive_control_medicine_table_version: str = "medicine_runtime_classes_v0"
    adaptive_control_intervention_space_version: str = "toml_overlay_control_v1"
    adaptive_control_runtime_executor_id: str = ""
    adaptive_control_executor_status: str = ""
    adaptive_control_max_interventions: int = 1
    adaptive_control_max_same_signal_interventions: int = 1
    adaptive_control_disallow_repeat_intervention: bool = True
    adaptive_control_watch_window_turns: int = 5
    adaptive_control_multi_intervention_enabled: bool = False
    adaptive_control_max_interventions_per_attempt: int = 1
    adaptive_control_max_interventions_per_hurdle_episode: int = 1
    adaptive_control_max_distinct_hurdle_episodes_per_attempt: int = 1
    adaptive_control_cooldown_after_apply_slots: int = 5
    guardrails_arm_after_turn: int = 0
    llm_hurdle_detector_backend: str = "llm"
    adaptive_control_branch_bundle_enabled: bool = False
    adaptive_control_branch_bundle_root: str = ""
    adaptive_control_branch_bundle_source_run_id: str = ""
    adaptive_control_branch_bundle_max_per_attempt: int = 1
    adaptive_control_branch_watch_policy_id: str = "prefix_rewind_watch_v1"
    # HarnessObservation: live, prefix-only mechanical concern packets.
    # First implementation is halflife-only and off by default.
    harness_observation_enabled: bool = False
    harness_observation_grace_activity_turns: int = 2
    harness_observation_cadence_turns: int = 10
    harness_observation_packet_char_budget: int = 1200
    harness_observation_evidence_lines: int = 3
    # LLM hurdle detector: separate no-tool detector call over a
    # harness-built live evidence packet. Off by default.
    llm_hurdle_detector_enabled: bool = False
    llm_hurdle_detector_cadence_turns: int = 1
    llm_hurdle_detector_atlas_dictionary_path: str = ""
    llm_hurdle_detector_input_contract_path: str = ""
    llm_hurdle_detector_log_path: str = ""
    llm_hurdle_detector_max_trace_events: int = 80
    llm_hurdle_detector_max_field_chars: int = 800
    llm_hurdle_detector_max_state_snapshots: int = 24
    llm_hurdle_detector_prompt_version: str = "llm_hurdle_detector_prompt_v4"
    # trace_nets backend thresholds. A runtime overlay may change them through
    # [llm_hurdle_detector.trace_nets]. A non-positive value uses the default.
    trace_nets_fail_min_streak: int = 4
    trace_nets_pass_lookback: int = 20
    trace_nets_pass_min_prior: int = 2
    trace_nets_pass_min_gap: int = 2
    trace_nets_reread_min_args_len: int = 20
    trace_nets_reread_min_gap: int = 3
    trace_nets_reread_max_gap: int = 30
    trace_nets_window: int = 30
    # Durable trace rows are telemetry; full bytes live in raw transcript
    # and/or .tool_output pointers.
    trace_result_summary_chars: int = 1200
    context_slot_max_candidates: int = 1
    context_slot_inline_files: int = 1
    focused_compound_trace_lines: int = 0  # Trace budget override for focused_compound (0 = use solver_trace_lines).
    focused_compound_evidence_lines: int = 0  # Evidence budget override for focused_compound (0 = use solver_evidence_lines).
    focused_compound_recent_tool_results_chars: int = 0  # Rolling tool-result budget override for focused_compound (0 = use recent_tool_results_chars).
    focused_compound_include_resolved_evidence: bool = False  # Whether focused_compound renders resolved/passing evidence.
    compound_selective_trace_lines: int = 0  # Trace budget override for compound_selective (0 = use solver_trace_lines).
    compound_selective_unresolved_evidence_lines: int = 0  # Unresolved evidence budget override for compound_selective (0 = use solver_evidence_lines).
    compound_selective_resolved_evidence_lines: int = 0  # Resolved evidence budget override for compound_selective (0 = hide resolved evidence).
    compound_selective_resolved_evidence_stub_chars: int = 0  # Result stub chars for resolved evidence in compound_selective (0 = use trace_stub_chars).
    compound_selective_recent_tool_results_chars: int = 0  # Rolling tool-result budget override for compound_selective (0 = use recent_tool_results_chars).
    compound_selective_trace_action_repeat_cap: int = 0  # Max identical trace actions kept in compound_selective (0 = no cap).
    compound_selective_resolved_action_repeat_cap: int = 0  # Max identical resolved-evidence actions kept in compound_selective (0 = no cap).
    compound_selective_trace_anchor_lines: int = 0  # Older trace actions reserved as anchors in compound_selective (0 = no anchors).
    compound_selective_resolved_anchor_lines: int = 0  # Older resolved-evidence actions reserved as anchors in compound_selective (0 = no anchors).
    compound_selective_trace_source_anchor_lines: int = 0  # Older non-test source anchors reserved in compound_selective trace selection (0 = disabled).
    compound_selective_trace_test_anchor_lines: int = 0  # Older test/verification anchors reserved in compound_selective trace selection (0 = disabled).
    compound_selective_resolved_source_anchor_lines: int = 0  # Older non-test source anchors reserved in compound_selective resolved evidence (0 = disabled).
    compound_selective_resolved_test_anchor_lines: int = 0  # Older test/verification anchors reserved in compound_selective resolved evidence (0 = disabled).
    halflife_context_limit_tokens: int = 0  # Context limit for halflife activation (0 = use context_size).
    halflife_no_decay_ratio: float = 0.50  # Keep full verbatim transcript below this fill ratio.
    halflife_verbatim_tool_results: int = 4  # Newest tool results kept verbatim after decay activates.
    halflife_cap_7_chars: int = 4096  # Tool-result cap for halflife tool age <= 7.
    halflife_cap_15_chars: int = 2048  # Tool-result cap for halflife tool age <= 15.
    halflife_cap_31_chars: int = 1024  # Tool-result cap for halflife tool age <= 31.
    halflife_cap_63_chars: int = 512  # Tool-result cap for halflife tool age <= 63.
    halflife_cap_older_chars: int = 256  # Tool-result cap for older halflife tool results.


# Every key must exist in config.toml — no silent defaults at read time.
# Values here are the hardcoded safety net only for keys intentionally optional.
_REQUIRED_SECTIONS = ("server", "model", "loop", "output", "tools", "experiment", "prompts")




from ._config_loader import _extract_config_fields, _require, _validate_coupling

def load_config(
    user_config: Path | list[Path] | None = None,
    overrides: dict | None = None,
    strict_dial_gates: bool = False,
) -> Config:
    """Load layered config, merge optional extra user TOML(s), apply CLI overrides.

    user_config: a single path OR a list of paths. When a list, overlays
                 layer in the given order — later entries win on conflict.
                 This lets a caller compose atomic toggles (e.g.
                 configs/substantive.toml + configs/toggles/intent.on.toml)
                 without pre-baking every combination.
    overrides:   flat dict of CLI flag overrides (highest priority)
    """
    base = copy.deepcopy(_LAYERED)  # start from already-merged base + local
    user_set_keys: set[str] = set()

    if user_config is not None:
        paths: list[Path] = (
            [user_config] if isinstance(user_config, (str, Path))
            else list(user_config)
        )
        for p in paths:
            layer = _load_toml(Path(p))
            _collect_leaf_keys(layer, user_set_keys)
            _deep_merge(base, layer)

    flat = _extract_config_fields(base)

    if overrides:
        for k, v in overrides.items():
            if v is not None and k in flat:
                flat[k] = type(flat[k])(v)

    cfg = Config(**flat)
    _validate_coupling(cfg, strict_dial_gates=strict_dial_gates,
                       user_set_keys=frozenset(user_set_keys))
    _validate_post_edit_checks(cfg)
    return cfg


def _collect_leaf_keys(d: dict, out: set) -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            _collect_leaf_keys(v, out)
        else:
            out.add(k)


def _validate_post_edit_checks(cfg: Config) -> None:
    """Reject post_edit_checks entries with unknown trigger / on_fail."""
    for spec in (cfg.post_edit_checks or []):
        validate_post_edit_check_dict(spec)


def require_runtime_mode(cfg: Config, *, expected: str, caller: str) -> None:
    """Reject an entry point invoked under the wrong runtime mode."""
    if expected not in VALID_RUNTIME_MODES:
        raise ValueError(f"unknown runtime mode expectation: {expected!r}")
    if cfg.runtime_mode != expected:
        raise ValueError(
            f"{caller} requires runtime.mode={expected!r}, "
            f"but resolved runtime.mode={cfg.runtime_mode!r}"
        )


def dump_config(cfg: Config) -> dict:
    """Return a serializable snapshot of a resolved Config for run metadata."""
    from dataclasses import asdict
    d = asdict(cfg)

    def _redact_target_keys(value):
        if isinstance(value, dict):
            return {
                key: ("<redacted>" if key == "api_key" else _redact_target_keys(child))
                for key, child in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [_redact_target_keys(child) for child in value]
        return value

    # Role/fallback targets may carry endpoint-local credentials. They are
    # needed at runtime but never belong in metrics provenance or config hashes.
    d["api_key"] = "<redacted>"
    d["model_roles"] = _redact_target_keys(d["model_roles"])
    d["model_fallback_chain"] = _redact_target_keys(d["model_fallback_chain"])
    # Fixed environment values can themselves be credentials. Preserve only
    # the names in public run metadata/config hashes.
    d["sandbox_env_set"] = {
        name: "<redacted>" for name in sorted(cfg.sandbox_env_set)
    }
    # retry_backoff is a tuple; convert for JSON
    d["retry_backoff"] = list(cfg.retry_backoff)
    return d
