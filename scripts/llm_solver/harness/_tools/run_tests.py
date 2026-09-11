"""run_tests tool: invoke the project's test runner with a structured envelope."""
import re
import hashlib
import shlex
from dataclasses import replace
from pathlib import Path

from ...config import Config
from ...language_quirks import load_language_advice, load_run_tests_quirk_object
from ..injections import UserTurnInjection
from ..time_budget import budgeted_test_execution, command_timeout
from ._common import ToolExecutionText, _resolve, _tool_advice, _xml_attr
from ._pytest_hints import _pytest_binary_missing, _pytest_path_missing


@budgeted_test_execution
def run_tests(
    path: str = "",
    k: str = "",
    last_failed: bool = False,
    base_cmd_override: str = "",
    component_source: str | list[str] = "",
    *,
    cwd: str,
    cfg: Config,
) -> str:
    """Invoke the detected test runner inside the sandbox with deterministic flags.

    Bypasses :func:`bash`'s string-only contract and goes straight to
    :func:`_run_in_sandbox` so the real exit code and timeout flag
    survive into the envelope. The invocation (base command, flags, env
    selection) comes from the ``[run_tests]`` table of whichever
    language_quirks TOML matches ``cwd`` (pytest / cargo / go / jest /
    ctest — see ``load_run_tests_quirk_object``); the model controls
    *what* to run via ``path``, ``k``, ``last_failed``, not *how* the
    runner formats output.

    Output protocol: when ``cfg.tools_run_tests_structured_output`` is
    true (default) the result is wrapped in
    ``<test_results status="..." exit_code="N" runner="...">…</test_results>``
    where ``status`` is one of {passed, failed, collection_error,
    internal_error, usage_error, no_tests_collected, timed_out, error}
    for pytest, plus runner_unavailable when the registered runner cannot
    start, or the runner's own ``status_map`` vocabulary otherwise
    (e.g. cargo/go/jest/ctest all reduce to {passed, failed, timed_out,
    error} — see each TOML's ``[run_tests.status_map]``).
    The status field discriminates exit codes that are often
    indistinguishable in raw output once the tracebacks are stripped,
    and separates a hard timeout from any other failure path. The
    ``runner`` attribute makes the trace replayable without
    re-detecting from cwd contents.
    When the knob is false the function returns the raw output for
    callers that want the legacy bash-string contract.

    Disabled by ``cfg.tools_run_tests_enabled`` — a disabled call
    returns ERROR rather than silently succeeding so accidental wiring
    is loud. Profiles that want to expose the tool must also flip the
    knob; the loop drops the schema when the knob is false (see
    ``_filter_disabled_tools`` in ``loop.py``) so a disabled handler
    is reached only via direct dispatch.
    """
    if not getattr(cfg, "tools_run_tests_enabled", False):
        return "ERROR: run_tests tool is disabled (tools.run_tests.enabled=false)"
    # Use the file tools' selected task view for containment and aliases.
    # The runner receives a path relative to its native task root.
    if path:
        try:
            safe = _resolve(cwd, path)
        except Exception as e:
            return f"ERROR: run_tests path resolution failed: {e}"
        try:
            from ..task_path import resolve_task_path
            path = str(safe.resolve().relative_to(resolve_task_path(cwd, '.')))
        except ValueError:
            return (
                f"ERROR: run_tests path {path!r} resolves outside cwd; "
                "use a path inside the working directory."
            )
    # Startup records the permitted project evidence and observed runtime.
    # Descriptors own command arguments; unavailable or ambiguous discovery
    # does not authorize a guessed interpreter or runner.
    runner = getattr(cfg, "analysis_task_format", "auto")
    quirk = load_run_tests_quirk_object(cwd, runner="generic" if runner in ("", "auto") else runner)
    selection = getattr(cfg, "runtime_test_selection", None)
    refresh = None
    from ..task_path import active_task_host_root
    task_root = active_task_host_root(cwd) or str(Path(cwd).resolve())
    if (selection is not None and selection.get("declaration_inputs") is not None
            and selection.get("task_root", task_root) == task_root):
        from ..runtime_discovery import selection_inputs
        from ..tools import _bash_unreadable_paths
        previous_inputs = selection["declaration_inputs"]
        current_inputs = selection_inputs(Path(cwd), previous_inputs, _bash_unreadable_paths(cwd, cfg))
        if current_inputs != previous_inputs:
            refresh = {"previous": dict(selection)}
    if (selection is None and runner in ("", "auto")) or refresh is not None:
        from ..runtime_discovery import discover_runtime
        from ..sandbox.env_policy import active_environment
        from ..tools import _effective_command_environment, _bash_unreadable_paths
        env, _login = active_environment()
        if env is None:
            env, _login = _effective_command_environment(cfg, cwd=cwd)
        requested = "auto" if selection is None or selection.get("request_source") == "project_declarations" else runner
        report = discover_runtime(Path(cwd), replace(cfg, analysis_task_format=requested), effective_env=env,
                                  unreadable_paths=_bash_unreadable_paths(cwd, cfg), selection_only=True)
        fresh = report["runner_selection"]
        if refresh is not None:
            refresh.update(current=dict(fresh), probes=report["probes"])
            # Config owns a copy; startup provenance retains the old observation.
            task_root = selection.get("task_root")
            selection.clear()
            selection.update(fresh)
            if task_root is not None:
                selection["task_root"] = task_root
        else:
            selection = fresh
    if selection is not None:
        from ..task_path import active_task_host_root
        chosen = selection.get("selected", {})
        task_root = active_task_host_root(cwd) or str(Path(cwd).resolve())
        root_matches = selection.get("task_root", task_root) == task_root
        if selection.get("status") == "selected" and root_matches:
            from ...language_quirks import load_run_tests_quirk_for_runner
            quirk = replace(load_run_tests_quirk_for_runner(chosen["runner"]),
                            base_cmd=chosen["base_cmd"], env_activate_prefix="")
        else:
            quirk = replace(quirk, base_cmd="")
    command_timeout()  # Discovery may return unresolved after consuming the allowance.
    if not quirk.base_cmd:
        status = "selection_unresolved" if selection is not None else "runner_unavailable"
        body = ("No unambiguous usable test runner was established. Review the "
                "observed project checks and environments, then use bash with the "
                "project's documented verification command.")
        if getattr(cfg, "tools_run_tests_structured_output", True):
            body = (f'<test_results status="{status}" runner="{_xml_attr(quirk.runner)}">'
                    f'\n{body}\n</test_results>')
        else:
            body = "ERROR: " + body
        return ToolExecutionText(body, exit_status=None, executed=False, verification_status=status,
                                 runner_request={"selection_refresh": refresh} if refresh else None)
    if quirk.extra_fields.get("reject_unsupported_filters") and (path or k or last_failed):
        return "ERROR: the declared test script has no supported filter contract. Run its full suite or use bash with the project's documented arguments."
    parts: list[str] = [base_cmd_override or quirk.base_cmd]
    if last_failed and quirk.arg_last_failed:
        parts.append(quirk.arg_last_failed)
    if k and quirk.arg_k_template:
        parts.append(quirk.arg_k_template.format(expr=shlex.quote(k)))
    if path and quirk.arg_path_style == "positional":
        parts.append(shlex.quote(quirk.extra_fields.get("arg_path_prefix", "") + path))
    cmd = quirk.env_activate_prefix + " ".join(parts)
    request = {
        "family": "" if base_cmd_override else quirk.runner,
        "basis": ("custom_command" if base_cmd_override else
                  "runtime_selection" if selection else "explicit_configuration"),
        "targets": [path] if path and quirk.arg_path_style == "positional" and not base_cmd_override else [],
        "target_status": ("unknown" if base_cmd_override else
                          "recorded" if path and quirk.arg_path_style == "positional" else "unspecified"),
        "command_sha256": hashlib.sha256(cmd.encode()).hexdigest(),
    }
    if refresh is not None:
        request["selection_refresh"] = refresh
    if base_cmd_override:
        from ..runner_invocations import runner_request_record
        parsed = runner_request_record(cmd)
        if parsed.get('family') or parsed.get('requests'):
            request = {**parsed, "command_sha256": request["command_sha256"]}

    from ..runner_invocations import bind_runner_workspace
    request = bind_runner_workspace(
        request, cmd, cwd, sandbox=cfg.sandbox_bash,
        backend=getattr(cfg, "sandbox_backend", "bwrap"),
    )

    def execution_result(*args, **kwargs):
        return ToolExecutionText(*args, runner_request=request, **kwargs)

    timeout = command_timeout()
    timeout_message = (f"ERROR: command timed out after {timeout:g}s" if timeout is not None
                       else "ERROR: command timed out")
    # Function-local import: tests patch `harness.tools._run_in_sandbox`
    # via mock.patch.object — looking the symbol up via the public
    # `tools` module here makes that patch intercept this call.
    from ..sandbox.env_policy import active_environment
    from ..tools import (
        _bash_unreadable_paths,
        _bash_readable_paths,
        _effective_command_environment,
        _run_in_sandbox,
    )
    from .._tool_filters import output_cleanup_enabled
    effective_env, allow_login_shell = active_environment()
    if effective_env is None:
        effective_env, allow_login_shell = _effective_command_environment(cfg, cwd=cwd)
    from ..sandbox.policy import sandbox_execution_kwargs
    if selection and selection.get("status") == "selected":
        effective_env = {**effective_env, **selection["selected"].get("environment", {})}
    from ..component_selection import component_selection
    with component_selection(cwd, component_source, quirk, effective_env) as (selected_env, component):
        out, exit_code, timed_out = _run_in_sandbox(
            cmd, cwd=cwd, timeout=timeout,
            bwrap_bin=cfg.bwrap_bin,
            **sandbox_execution_kwargs(cfg),
            unreadable_paths=_bash_unreadable_paths(cwd, cfg),
            readable_paths=_bash_readable_paths(cfg),
            effective_env=selected_env,
            allow_login_shell=allow_login_shell,
            normalize_output=output_cleanup_enabled(cfg),
        )
    if component_source:
        request["component_selection"] = component
    user_turn_injections: list[UserTurnInjection] = []
    status = _verification_status(quirk, out, exit_code, timed_out)
    if base_cmd_override:
        from ..shell_verification import shell_verification_status
        attributed = shell_verification_status(cmd, exit_code, timed_out)
        if attributed not in {'passed', 'failed', 'timed_out'}:
            status = attributed
    if (component_source and component.get("status") != "selected"
            and status in {"passed", "no_tests_collected"}):
        status = "selection_" + component.get("status", "unavailable")

    if not getattr(cfg, "tools_run_tests_structured_output", True):
        # Legacy bash-string contract. Mirror bash() exactly so existing
        # callers see no change.
        if timed_out:
            return execution_result(
                timeout_message,
                exit_status=None,
                timed_out=True,
                verification_status=status,
            )
        if exit_code is None:
            return execution_result(out, exit_status=None, verification_status=status)
        if exit_code != 0:
            out += f"\n[exit code: {exit_code}]"
        if quirk.runner == "pytest":
            python_advice = load_language_advice("python")
            pytest_advice = load_language_advice("pytest")
            if _pytest_binary_missing(out, exit_code):
                user_turn_injections.append(_tool_advice(
                    python_advice["python_runner_missing"],
                    mechanism="pytest_binary_missing_hint",
                    tool_name="run_tests", runner=quirk.runner,
                    exit_code=exit_code,
                ))
            elif _pytest_path_missing(out, exit_code):
                user_turn_injections.append(_tool_advice(
                    pytest_advice["pytest_path_missing"],
                    mechanism="pytest_path_missing_hint",
                    tool_name="run_tests", runner=quirk.runner,
                    exit_code=exit_code,
                ))
        return execution_result(
            out,
            exit_status=exit_code,
            verification_status=status,
            user_turn_injections=user_turn_injections,
        )

    if timed_out:
        status = "timed_out"
        body = timeout_message
        ec_attr = ""
    elif exit_code is None:
        # Non-timeout exception inside _run_in_sandbox; `out` already
        # holds an ERROR string.
        status = "error"
        body = out
        ec_attr = ""
    else:
        body = out if out else "(no output)"
        ec_attr = f' exit_code="{exit_code}"'
        # pytest-specific recovery hints: only meaningful when pytest is
        # the runner that actually ran. Firing these against cargo/go/
        # jest output would misdirect the model toward a conda-activate
        # dance that doesn't apply.
        if quirk.runner == "pytest":
            python_advice = load_language_advice("python")
            pytest_advice = load_language_advice("pytest")
            if _pytest_binary_missing(body, exit_code):
                user_turn_injections.append(_tool_advice(
                    python_advice["python_runner_missing"],
                    mechanism="pytest_binary_missing_hint",
                    tool_name="run_tests", runner=quirk.runner,
                    exit_code=exit_code,
                ))
            elif _pytest_path_missing(body, exit_code):
                user_turn_injections.append(_tool_advice(
                    pytest_advice["pytest_path_missing"],
                    mechanism="pytest_path_missing_hint",
                    tool_name="run_tests", runner=quirk.runner,
                    exit_code=exit_code,
                ))
            # `--lf` with an empty lastfailed cache → exit 5
            # (no_tests_collected), indistinguishable from "no tests at
            # all" without the harness hint. Tied to the input arg so we
            # don't false-fire on legitimately-empty test directories
            # called without --lf.
            if last_failed and status == "no_tests_collected":
                user_turn_injections.append(_tool_advice(
                    pytest_advice["pytest_lf_cache_empty"],
                    mechanism="pytest_lf_cache_empty_hint",
                    tool_name="run_tests", runner=quirk.runner,
                    exit_code=exit_code,
                ))
        # Add source context around each failing assertion so the model
        # can see the surrounding code with the verdict. This runs for
        # `failed` and
        # `collection_error` — both produce `tests/foo.py:N:` frames in
        # `--tb=short` output. Skip `timed_out` and `error`, which have no
        # source frames.
        if status in ("failed", "collection_error"):
            ctx_blocks = _extract_failing_assertion_context(
                body, cwd,
                radius=int(getattr(cfg, "tools_run_tests_assertion_context_lines", 5)),
                max_failures=int(getattr(cfg, "tools_run_tests_assertion_context_max", 3)),
            )
            if ctx_blocks:
                body += "\n" + "\n".join(ctx_blocks)
    # `runner` identifies which language_quirks template produced this
    # invocation (pytest / cargo / go / jest / ctest). Without it
    # the trace can't tell which runner ran without re-detecting from
    # cwd contents, and a re-run on a repo that gained a Cargo.toml
    # silently flips the output shape.
    return execution_result(
        (
            f'<test_results status="{status}"{ec_attr} '
            f'runner="{quirk.runner}">\n{body}\n</test_results>'
        ),
        exit_status=exit_code,
        timed_out=timed_out,
        verification_status=status,
        user_turn_injections=user_turn_injections,
    )


def _verification_status(quirk, output: str, exit_code: int | None, timed_out: bool) -> str:
    """Derive the tool-owned status once, independently of its rendering."""
    if timed_out:
        return "timed_out"
    if exit_code is None:
        return "error"
    if exit_code in {126, 127} or (
        quirk.runner == "pytest" and _pytest_binary_missing(output, exit_code)
    ):
        return "runner_unavailable"
    return quirk.status_map.get(exit_code, quirk.status_default or f"error_{exit_code}")


_PYTEST_FAIL_FRAME_RE = re.compile(
    # Matches both
    #   tests/test_foo.py:42: in test_bar
    #   tests/test_foo.py:42:                in test_bar
    # and the bare frame line emitted by --tb=short. It also matches Go
    # (`.go:`), Rust (`.rs:`), and JavaScript/TypeScript
    # (`.js:`/`.ts:`) source frames, which share the same
    # `<path>:<line>:` shape in their own tracebacks/panics. `.py:` keeps
    # matching exactly as before — this only widens the extension set.
    r"^(?P<path>[\w\-/.]+\.(?:py|go|rs|jsx?|tsx?)):(?P<line>\d+):(?:\s+in\s+\S+)?\s*$",
    re.MULTILINE,
)


def _extract_failing_assertion_context(
    body: str, cwd: str, *, radius: int = 5, max_failures: int = 3,
) -> list[str]:
    """Return XML snippet blocks with ±radius lines around each pytest assert.

    Walks the pytest --tb=short body for `<path>.py:<line>:` frames and
    reads the cited line from cwd/<path> (resolved through tools._resolve
    so the path-traversal protections apply). De-dupes on (path, line)
    pairs so a chained traceback doesn't return the same snippet twice.
    Caps at `max_failures` blocks to keep the appended context bounded.
    Best-effort: any unreadable path / out-of-range line is silently
    skipped — the model still has the original --tb=short body.
    """
    seen: set[tuple[str, int]] = set()
    blocks: list[str] = []
    for m in _PYTEST_FAIL_FRAME_RE.finditer(body):
        if len(blocks) >= max_failures:
            break
        rel_path = m.group("path")
        try:
            line_no = int(m.group("line"))
        except ValueError:
            continue
        key = (rel_path, line_no)
        if key in seen:
            continue
        seen.add(key)
        try:
            abs_path = _resolve(cwd, rel_path)
        except Exception:
            continue
        if not abs_path.is_file():
            continue
        try:
            file_lines = abs_path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        if not (1 <= line_no <= len(file_lines)):
            continue
        lo = max(1, line_no - radius)
        hi = min(len(file_lines), line_no + radius)
        snippet_lines: list[str] = []
        for n in range(lo, hi + 1):
            marker = ">" if n == line_no else " "
            snippet_lines.append(f"{marker} {n:5d}  {file_lines[n - 1]}")
        snippet = "\n".join(snippet_lines)
        blocks.append(
            f'<failing-assertion file="{_xml_attr(rel_path)}" '
            f'line="{line_no}" radius="{radius}">\n{snippet}\n</failing-assertion>'
        )
    return blocks
