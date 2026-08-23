"""Tool registry + dispatcher + back-compat re-exports.

The actual tool implementations live under ``_tools/`` (one module per
tool). This file owns the dispatch table, the ``ToolRegistry`` shape,
and the post-dispatch pipeline (output filters, redaction, optional
unified envelope). Public names (``bash``, ``read``, …, ``_resolve``,
``truncate_output``, …) are re-exported here so existing callers and
tests continue to import from ``llm_solver.harness.tools``.
"""
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import Config
from .._shared.telemetry_paths import telemetry_dir

# ── Public re-exports ────────────────────────────────────────────────
# Tools (one per file under _tools/)
from ._tools.apply_patch import apply_patch_tool
from ._tools.bash import bash
from ._tools.edit import edit, _whitespace_normalized_match
from ._tools.glob import glob_files
from ._tools.grep import grep_files
from ._tools.list_definitions import list_definitions
from ._tools.read import read
from ._tools.run_tests import run_tests
from ._tools.write import write
# Cross-tool helpers (imported by tests as `from harness.tools import _resolve`)
from ._tools._common import _resolve, _xml_attr
# Sandbox runner — re-exported here so tests' `mock.patch.object(tools_mod,
# "_run_in_sandbox", …)` keeps working. bash() and run_tests() do a
# function-local `from ..tools import _run_in_sandbox` so the patch
# intercepts at call time.
from ._tools._run_in_sandbox import _run_in_sandbox
# pytest hint constants + helpers (test_leakage_closures.py imports them)
from ._tools._pytest_hints import (
    _PYTEST_PATH_MISSING_HINT, _pytest_path_missing,
)
# Filter helpers (test_harness_pipeline_tools.py imports them via this module)
from ._tool_filters import (
    _collapse_duplicate_lines, _collapse_similar_lines,
    _filter_bash_output, _line_skeleton, truncate_output,
)
from .tool_specs import ACTIVE_TOOL_NAMES, is_native_envelope
from .process_manager import AdmittedProcessOutput, ProcessManagerError
from .sandbox.ignore_policy import (
    IgnorePolicy,
    activate_ignore_policy,
    active_ignore_policy,
)
from .sandbox.env_policy import (
    DEFAULT_FIXED_ENVIRONMENT,
    EnvironmentPolicy,
    activate_environment,
    active_environment,
)

log = logging.getLogger(__name__)


# Unified <tool_result> envelope schema version. Bump when the attribute
# set or wrap shape changes (e.g. adding error_subkind, latency_ms).
# Version 1 has tool_name, status, optional error_kind, and v="1".
_UNIFIED_ENVELOPE_VERSION = "1"


def _bash_unreadable_paths(
    cwd, cfg, ignore_policy: IgnorePolicy | None = None,
) -> tuple[str, ...]:
    """Configured masks plus run-owned telemetry and ignore-file masks.

    Under bwrap the whole host is bound read-only, so the telemetry sibling of
    the task workspace is readable even though it sits outside cwd. Mask it:
    harness-owned files are never part of the model's world. Container mode
    mounts only cwd, so the entry is inert there.

    Marked ``optional:`` — the dir may not exist for callers that never opened
    a trace (tests, ad-hoc tool use), and a zero-match must not fail closed.
    """
    configured = tuple(getattr(cfg, "unreadable_paths", ()) or ())
    additions: tuple[str, ...] = ()
    if cwd:
        additions += (f"optional:{telemetry_dir(Path(cwd))}",)
    policy = ignore_policy or active_ignore_policy(cwd)
    if policy is not None:
        additions += policy.sandbox_unreadable_paths()
    # Preserve declaration order while avoiding repeated configured,
    # telemetry, or dynamically rediscovered model-view mounts.
    return tuple(dict.fromkeys((*configured, *additions)))


def _dispatch_bash(args, cwd, cfg):
    if bool(args.get("background", False)):
        return "ERROR: background process manager is unavailable"
    effective_env, allow_login_shell = active_environment()
    if effective_env is None:
        effective_env, allow_login_shell = _effective_command_environment(cfg)
    return bash(
        args["cmd"], cwd=cwd, timeout=cfg.bash_timeout, sandbox=cfg.sandbox_bash,
        bwrap_bin=cfg.bwrap_bin,
        sandbox_required=getattr(cfg, "sandbox_required", False),
        unreadable_paths=_bash_unreadable_paths(cwd, cfg),
        sandbox_backend=getattr(cfg, "sandbox_backend", "bwrap"),
        container_runtime=getattr(
            cfg, "sandbox_container_runtime", "docker"
        ),
        container_image=getattr(cfg, "sandbox_container_image", ""),
        container_flags=tuple(
            getattr(cfg, "sandbox_container_flags", ()) or ()
        ),
        effective_env=effective_env,
        allow_login_shell=allow_login_shell,
    )


def _effective_command_environment(cfg: Config) -> tuple[dict[str, str], bool]:
    """Resolve one Config's command-only environment.

    ``Session`` and the outer driver retain this result for their lifetime.
    This fallback keeps direct tool calls and small unit fixtures faithful to
    the same public configuration contract.
    """
    policy = EnvironmentPolicy(
        inherit=getattr(cfg, "sandbox_env_inherit", "core"),
        set=getattr(cfg, "sandbox_env_set", DEFAULT_FIXED_ENVIRONMENT),
        filters=getattr(cfg, "sandbox_env_filters", {}),
        ignore_default_excludes=getattr(
            cfg, "sandbox_env_ignore_default_excludes", False
        ),
        allow_login_shell=getattr(
            cfg, "sandbox_env_allow_login_shell", False
        ),
    )
    return policy.resolve(), policy.allow_login_shell


_DISPATCH = {
    "bash": _dispatch_bash,
    "bash_poll": lambda args, cwd, cfg: (
        "ERROR: background process manager is unavailable"
    ),
    "bash_kill": lambda args, cwd, cfg: (
        "ERROR: background process manager is unavailable"
    ),
    "read": lambda args, cwd, cfg: read(
        args["path"], cwd=cwd, offset=args.get("offset", 0),
        limit=args.get("limit", 0), cfg=cfg,
    ),
    "write": lambda args, cwd, cfg: write(
        args["path"], args["content"], cwd=cwd, cfg=cfg,
    ),
    "edit": lambda args, cwd, cfg: edit(
        args["path"], args["old_str"], args["new_str"], cwd=cwd, cfg=cfg,
    ),
    "glob": lambda args, cwd, cfg: glob_files(
        args["pattern"], args.get("path", "."), cwd=cwd,
        page=int(args.get("page", 1)), cfg=cfg,
    ),
    "grep": lambda args, cwd, cfg: grep_files(
        args["pattern"], args.get("path", "."), args.get("glob", ""),
        cwd=cwd, timeout=cfg.grep_timeout,
        page=int(args.get("page", 1)), cfg=cfg,
    ),
    "lsp": lambda args, cwd, cfg: (
        "ERROR: lsp manager is unavailable for this dispatch context"
    ),
    "task": lambda args, cwd, cfg: (
        "ERROR: task tool is unavailable outside a configured Session"
    ),
    "done": lambda args, cwd, cfg: "done",
    "run_tests": lambda args, cwd, cfg: run_tests(
        path=args.get("path", ""),
        k=args.get("k", ""),
        last_failed=bool(args.get("last_failed", False)),
        cwd=cwd, cfg=cfg,
    ),
    "list_definitions": lambda args, cwd, cfg: list_definitions(
        args["path"], cwd=cwd, cfg=cfg,
        symbol=args.get("symbol"), kind=args.get("kind"),
        repo_wide=bool(args.get("repo_wide", False)),
        page=int(args.get("page", 1)),
    ),
    "apply_patch": lambda args, cwd, cfg: apply_patch_tool(
        args["patch"], cwd=cwd, cfg=cfg,
    ),
}

if tuple(_DISPATCH) != ACTIVE_TOOL_NAMES:
    raise RuntimeError(
        "Tool dispatch order/names drifted from harness.tool_specs: "
        f"dispatch={tuple(_DISPATCH)}, specs={ACTIVE_TOOL_NAMES}"
    )


@dataclass(frozen=True)
class ToolRegistry:
    """Composable tool-dispatch registry."""

    handlers: dict[str, Callable[[dict, str, Config], str]]


def build_tool_registry(
    *,
    overrides: dict[str, Callable[[dict, str, Config], str]] | None = None,
) -> ToolRegistry:
    """Build the effective tool registry with optional handler overrides."""
    handlers = dict(_DISPATCH)
    if overrides:
        handlers.update(overrides)
    return ToolRegistry(handlers=handlers)


def validate_tool_handlers(schema_names: list[str], *,
                           allow_extra_handlers: bool = True,
                           registry: ToolRegistry | None = None) -> None:
    """Fail fast when tool schemas and dispatch handlers drift."""
    declared = set(schema_names)
    reg = registry or build_tool_registry()
    handlers = set(reg.handlers.keys())
    missing_handlers = declared - handlers
    undeclared_handlers = handlers - declared
    if missing_handlers or (undeclared_handlers and not allow_extra_handlers):
        raise ValueError(
            "Tool surface mismatch: "
            f"missing handlers={sorted(missing_handlers)}, "
            f"undeclared handlers={sorted(undeclared_handlers)}"
        )


def admit_tool_output(
    name: str,
    result: object,
    *,
    arguments: dict,
    cfg: Config,
    output_control=None,
    redactions=None,
    filter_shell_output: bool = True,
) -> str:
    """Apply the single model-facing output-admission pipeline.

    Background polls call this before emitting ``proc_poll``. Their marked
    result then passes through :func:`dispatch` without a second transform.
    """
    result = str(result)
    if name in {"bash", "bash_poll"} and filter_shell_output:
        cmd = str(arguments.get("cmd", ""))
        result = _filter_bash_output(result, cmd, cfg)
        if output_control is not None and name == "bash":
            from ..bash_quirks import condense_output
            result = condense_output(result, cmd, output_control)
    elif name == "run_tests":
        argv: list[str] = ["pytest"]
        if arguments.get("last_failed"):
            argv.append("--lf")
        if arguments.get("k"):
            argv.extend(["-k", str(arguments["k"])])
        if arguments.get("path"):
            argv.append(str(arguments["path"]))
        synthesized_cmd = " ".join(argv)
        result = _filter_bash_output(result, synthesized_cmd, cfg)
        if output_control is not None:
            from ..bash_quirks import condense_output
            result = condense_output(result, synthesized_cmd, output_control)

    if redactions and not is_native_envelope(result):
        from ..bash_quirks import apply_redactions
        result = apply_redactions(result, redactions)

    if (
        getattr(cfg, "tools_unified_envelope_enabled", False)
        and not is_native_envelope(result)
    ):
        from .._shared.classification import derive_envelope_status
        status, error_kind = derive_envelope_status(result)
        attrs = f' tool_name="{_xml_attr(name)}" status="{status}"'
        if error_kind:
            attrs += f' error_kind="{_xml_attr(error_kind)}"'
        attrs += f' v="{_UNIFIED_ENVELOPE_VERSION}"'
        result = f'<tool_result{attrs}>\n{result}\n</tool_result>'

    return truncate_output(result, cfg)


def dispatch(name: str, arguments: dict, *, cwd: str, cfg: Config,
             output_control=None, universal_rewrites=None,
             forbidden_rules=None, redirect_rules=None, redactions=None,
             tool_registry: ToolRegistry | None = None,
             stale_guard=None,
             active_tools=(), redirect_event_sink=None,
             rewrite_log: list | None = None,
             execution_metadata: dict | None = None,
             ignore_policy: IgnorePolicy | None = None,
             effective_env=None,
             allow_login_shell: bool | None = None) -> str:
    """Route a tool call to its implementation, truncate output.

    output_control: optional OutputControl from bash_quirks, loaded
    from the active language_quirks/<runner>.toml [output_control].
    When present, test commands get rewritten (failure-only flags) and
    output gets condensed (passing lines stripped). When None, no
    runner-specific transforms apply.
    universal_rewrites: optional list of RewriteRule from bash_quirks,
    loaded from bash_quirks/rewrites.toml. When present, noisy commands
    (pip, npm, make) get quieted. When None, no universal rewrites apply.
    forbidden_rules: optional list of ForbiddenRule from bash_quirks,
    loaded from bash_quirks/forbidden.toml. When a pattern matches, the
    bash command is replaced with `false  # [HARNESS: <reason>]` before
    execution. None disables the layer.
    redirect_rules: optional compound-aware rules checked against the original
    bash command before rewrites. A rule applies only when its target appears
    in ``active_tools``.
    """
    reg = tool_registry or build_tool_registry()
    handler = reg.handlers.get(name)
    if handler is None:
        return f"ERROR: unknown tool '{name}'"

    # Dedicated-tool redirects inspect the exact model command before any
    # rewrite. A match is a typed tool error: no shell handler runs, no stale
    # read credit is earned, and the ordinary result admission path still
    # applies its configured byte cap.
    redirected = False
    if name == "bash" and redirect_rules:
        from .command_redirect import find_redirect, render_redirect_error

        decision = find_redirect(
            str(arguments.get("cmd", "")),
            redirect_rules,
            active_tools=active_tools,
            read_side_enabled=bool(
                getattr(cfg, "tools_bash_redirect_read_side", False)
            ),
        )
        if decision is not None:
            redirected = True
            result = render_redirect_error(
                decision, max_chars=int(cfg.max_output_chars)
            )
            if redirect_event_sink is not None:
                try:
                    redirect_event_sink({
                        "event": "redirect_rule",
                        **decision.trace_fields(),
                    })
                except Exception as exc:
                    log.warning("redirect event emit failed: %s", exc)

    # Pre-execution: rewrite bash commands (quiet flags, test flags, or
    # refuse forbidden patterns). When rewrite_log is provided and a
    # rewrite actually happens, append a record of the ORIGINAL cmd and
    # the rule kind. The trace event's args_summary records a short form
    # of the original arguments. cmd_pre_rewrite keeps the full command
    # and the rule kind so later code can classify the rewrite.
    original_bash_cmd = str(arguments.get("cmd", "")) if name == "bash" else ""
    bash_was_rewritten = False
    if (
        name == "bash"
        and not redirected
        and (output_control is not None or universal_rewrites or forbidden_rules)
    ):
        from ..bash_quirks import rewrite_command
        original_cmd = arguments.get("cmd", "")
        _rules_fired: list = []
        rewritten_cmd = rewrite_command(
            original_cmd, output_control, universal_rewrites,
            forbidden_rules=forbidden_rules, rule_log=_rules_fired,
        )
        if rewritten_cmd != original_cmd:
            bash_was_rewritten = True
            arguments = {**arguments, "cmd": rewritten_cmd}
            if rewrite_log is not None:
                rewrite_log.append({
                    "tool": name,
                    "original": original_cmd,
                    "rewritten": rewritten_cmd,
                    "rules": _rules_fired,
                })

    stale_decision = None
    stale_precheck_error = ""
    if name == "edit" and stale_guard is not None and not redirected:
        try:
            stale_decision = stale_guard.check_edit(str(arguments.get("path", "")))
        except Exception as exc:
            log.warning("stale guard pre-edit check failed: %s", exc)
            stale_precheck_error = (
                f"ERROR: stale_file: read {arguments.get('path', '')} first"
            )

    executed = False
    if redirected:
        pass
    elif stale_precheck_error:
        result = stale_precheck_error
    elif stale_decision is not None and stale_decision.blocked:
        result = stale_decision.message
    else:
        if effective_env is None:
            effective_env, resolved_login_shell = (
                _effective_command_environment(cfg)
            )
        else:
            resolved_login_shell = bool(
                getattr(cfg, "sandbox_env_allow_login_shell", False)
                if allow_login_shell is None
                else allow_login_shell
            )
        with (
            activate_ignore_policy(ignore_policy),
            activate_environment(
                effective_env,
                allow_login_shell=resolved_login_shell,
            ),
        ):
            try:
                executed = True
                result = handler(arguments, cwd, cfg)
            except ProcessManagerError as e:
                result = f"ERROR: {e}"
            except (KeyError, TypeError) as e:
                return f"ERROR: bad arguments for {name}: {e}"

    applied_operations = tuple(getattr(result, "applied_operations", ()))
    raw_exit_status = getattr(result, "exit_status", None)
    raw_timed_out = bool(getattr(result, "timed_out", False))

    # Update the mechanical read ledger only after an operation actually
    # succeeded. Observation failures never turn a completed tool call into a
    # harness exception; they simply leave the next edit conservatively stale.
    if stale_guard is not None and not redirected:
        from .._shared.classification import is_error_result
        raw_result = str(result)
        succeeded = not is_error_result(raw_result)
        try:
            if succeeded and name == "read":
                stale_guard.observe_read(str(arguments.get("path", "")))
            elif succeeded and name in {"write", "edit"}:
                stale_guard.observe_mutation(
                    str(arguments.get("path", "")), source=name
                )
            elif succeeded and name == "apply_patch":
                for operation_kind, operation_path in applied_operations:
                    if operation_kind == "delete":
                        stale_guard.forget(operation_path, source="apply_patch")
                    else:
                        stale_guard.observe_mutation(
                            operation_path, source="apply_patch"
                        )
            elif name == "bash" and not bash_was_rewritten and not raw_timed_out:
                from .stale_guard import classify_single_file_read
                shell_read = classify_single_file_read(original_bash_cmd)
                read_exit = raw_exit_status == 0 or (
                    raw_exit_status == 1
                    and shell_read is not None
                    and shell_read.verb in {"grep", "egrep", "fgrep", "rg"}
                )
                if shell_read is not None and read_exit:
                    stale_guard.observe_shell_read(original_bash_cmd)
        except Exception as exc:
            log.warning("stale guard observation failed for %s: %s", name, exc)

    if stale_decision is not None and stale_decision.message and not stale_decision.blocked:
        result = str(result) + "\n\n" + stale_decision.message

    already_admitted = isinstance(result, AdmittedProcessOutput)
    if execution_metadata is not None:
        execution_metadata["executed"] = executed
        if hasattr(result, "exit_status"):
            execution_metadata["exit_status_known"] = True
            execution_metadata["exit_status"] = getattr(result, "exit_status")
            execution_metadata["timed_out"] = bool(getattr(result, "timed_out", False))
    if already_admitted:
        result = str(result)
    else:
        result = admit_tool_output(
            name,
            result,
            arguments=arguments,
            cfg=cfg,
            output_control=output_control,
            redactions=redactions,
            filter_shell_output=not redirected,
        )
    if execution_metadata is not None:
        execution_metadata["output_sha256"] = hashlib.sha256(
            result.encode("utf-8", errors="replace")
        ).hexdigest()
    return result
