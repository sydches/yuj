"""run_tests tool: invoke the project's test runner with a structured envelope."""
import re
import shlex
from pathlib import Path

from ...config import Config
from ...language_quirks import load_run_tests_quirk_object
from ._common import _resolve, _xml_attr
from ._pytest_hints import (
    _PYTEST_BINARY_MISSING_HINT, _PYTEST_LF_CACHE_EMPTY_HINT,
    _PYTEST_PATH_MISSING_HINT, _PYTEST_STATUS,
    _pytest_binary_missing, _pytest_path_missing,
)


def run_tests(
    path: str = "",
    k: str = "",
    last_failed: bool = False,
    *,
    cwd: str,
    cfg: Config,
) -> str:
    """Invoke the detected test runner inside the sandbox with deterministic flags.

    Bypasses :func:`bash`'s string-only contract and goes straight to
    :func:`_run_in_sandbox` so the real exit code and timeout flag
    survive into the envelope. The invocation (base command, flags, env
    activation) comes from the ``[run_tests]`` table of whichever
    language_quirks TOML matches ``cwd`` (pytest / cargo / go / jest /
    ctest — see ``load_run_tests_quirk_object``); the model controls
    *what* to run via ``path``, ``k``, ``last_failed``, not *how* the
    runner formats output.

    Output protocol: when ``cfg.tools_run_tests_structured_output`` is
    true (default) the result is wrapped in
    ``<test_results status="..." exit_code="N" runner="...">…</test_results>``
    where ``status`` is one of {passed, failed, collection_error,
    internal_error, usage_error, no_tests_collected, timed_out, error}
    for pytest, or the runner's own ``status_map`` vocabulary otherwise
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
    # Path-traversal guard: same `_resolve` containment as read/edit/
    # write/list_definitions. An absolute or `..`-bearing path is
    # silently re-rooted at cwd so it can't escape. We pass the
    # already-relative form down to pytest, which then runs `python -m
    # pytest <safe_path>` with the testbed env active.
    if path:
        try:
            safe = _resolve(cwd, path)
        except Exception as e:
            return f"ERROR: run_tests path resolution failed: {e}"
        # Convert back to a cwd-relative string for the argv. _resolve
        # returns an absolute Path; we want the form pytest expects.
        try:
            path = str(safe.resolve().relative_to(Path(cwd).resolve()))
        except ValueError:
            # safe lands outside cwd (shouldn't happen — _resolve strips
            # leading / and ./ — defensive).
            return (
                f"ERROR: run_tests path {path!r} resolves outside cwd; "
                "use a path inside the working directory."
            )
    # Resolve the right invocation by inspecting the cwd. Each runner's
    # template — base command, env-activate prefix, arg styles — lives
    # in scripts/llm_solver/language_quirks/<runner>.toml under
    # [run_tests]. detect_runner() picks the first matching runner by
    # descriptor detection_priority; add a TOML descriptor to support a
    # new language without touching this tool.
    quirk = load_run_tests_quirk_object(cwd)
    parts: list[str] = [quirk.base_cmd]
    if last_failed and quirk.arg_last_failed:
        parts.append(quirk.arg_last_failed)
    if k and quirk.arg_k_template:
        parts.append(quirk.arg_k_template.format(expr=shlex.quote(k)))
    if path and quirk.arg_path_style == "positional":
        parts.append(shlex.quote(path))
    cmd = quirk.env_activate_prefix + " ".join(parts)
    timeout = int(getattr(cfg, "tools_run_tests_timeout", 60))
    # Function-local import: tests patch `harness.tools._run_in_sandbox`
    # via mock.patch.object — looking the symbol up via the public
    # `tools` module here makes that patch intercept this call.
    from ..sandbox.env_policy import active_environment
    from ..tools import (
        _bash_unreadable_paths,
        _effective_command_environment,
        _run_in_sandbox,
    )
    effective_env, allow_login_shell = active_environment()
    if effective_env is None:
        effective_env, allow_login_shell = _effective_command_environment(cfg)
    out, exit_code, timed_out = _run_in_sandbox(
        cmd, cwd=cwd, timeout=timeout,
        sandbox=cfg.sandbox_bash, bwrap_bin=cfg.bwrap_bin,
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

    if not getattr(cfg, "tools_run_tests_structured_output", True):
        # Legacy bash-string contract. Mirror bash() exactly so existing
        # callers see no change.
        if timed_out:
            return f"ERROR: command timed out after {timeout}s"
        if exit_code is None:
            return out
        if exit_code != 0:
            out += f"\n[exit code: {exit_code}]"
        if quirk.runner == "pytest":
            if _pytest_binary_missing(out, exit_code):
                out += _PYTEST_BINARY_MISSING_HINT
            elif _pytest_path_missing(out, exit_code):
                out += _PYTEST_PATH_MISSING_HINT
        return out

    if timed_out:
        status = "timed_out"
        body = f"ERROR: command timed out after {timeout}s"
        ec_attr = ""
    elif exit_code is None:
        # Non-timeout exception inside _run_in_sandbox; `out` already
        # holds an ERROR string.
        status = "error"
        body = out
        ec_attr = ""
    else:
        # Each language_quirks TOML can declare its own multilingual
        # exit-code-to-status mapping in [run_tests.status_map]
        # (+ optional status_default for unmapped nonzero codes) so a
        # runner's real exit-code semantics (e.g. cargo panic=101, go/
        # jest/ctest nonzero=failed) aren't forced through pytest's
        # vocabulary. Only when a runner declares NO status_map at all do
        # we fall back to the hardcoded pytest table — this preserves
        # pytest behavior byte-identically regardless of whether
        # pytest.toml happens to carry an explicit table.
        if quirk.status_map:
            status = quirk.status_map.get(
                exit_code, quirk.status_default or f"error_{exit_code}"
            )
        else:
            status = _PYTEST_STATUS.get(exit_code, f"error_{exit_code}")
        body = out if out else "(no output)"
        ec_attr = f' exit_code="{exit_code}"'
        # pytest-specific recovery hints: only meaningful when pytest is
        # the runner that actually ran. Firing these against cargo/go/
        # jest output would misdirect the model toward a conda-activate
        # dance that doesn't apply.
        if quirk.runner == "pytest":
            if _pytest_binary_missing(body, exit_code):
                body += _PYTEST_BINARY_MISSING_HINT
            elif _pytest_path_missing(body, exit_code):
                body += _PYTEST_PATH_MISSING_HINT
            # `--lf` with an empty lastfailed cache → exit 5
            # (no_tests_collected), indistinguishable from "no tests at
            # all" without the harness hint. Tied to the input arg so we
            # don't false-fire on legitimately-empty test directories
            # called without --lf.
            if last_failed and status == "no_tests_collected":
                body += _PYTEST_LF_CACHE_EMPTY_HINT
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
    return (
        f'<test_results status="{status}"{ec_attr} '
        f'runner="{quirk.runner}">\n{body}\n</test_results>'
    )


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
