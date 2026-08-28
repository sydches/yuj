"""bash tool: run a shell command in the sandbox, return stdout+stderr."""
from pathlib import Path
import shlex

from ..sandbox import _DEFAULT_BWRAP_BIN, container_mode
from ..sandbox.ignore_policy import IgnorePolicy, active_ignore_policy
from ._common import (
    ToolExecutionText,
    _require_external_readable,
    _resolve_read,
)
from ._env_hints import (
    _PYTHON_ENV_MISSING_HINT, _SEALED_INSTALL_FAILURE_HINT,
    _python_env_missing, _sealed_install_failure,
)
from ._pytest_hints import (
    _PYTEST_BINARY_MISSING_HINT, _PYTEST_PATH_MISSING_HINT,
    _pytest_binary_missing, _pytest_path_missing,
)


# Shell metachars that change cmd semantics if interpreted by bash.
# Any token containing one of these is ineligible for the in-process
# fast path — fall through to the sandbox so bash handles the
# expansion / pipe / redir / subshell correctly.
_SHELL_METACHARS = frozenset("*?[]$`|&;()<>\\")


def _try_inproc_trivial_read(
    cmd: str, cwd: str, ignore_policy: IgnorePolicy | None = None,
    readonly_roots: tuple[str, ...] = (),
    unreadable_paths: tuple[str, ...] = (),
) -> tuple[str, int, bool] | None:
    """In-process fast path for whitelisted single-target read commands.

    Bypasses the sandbox entirely (no bwrap, no subprocess, no
    persistent-bash round-trip). Returns ``(output, exit_code,
    timed_out=False)`` on match, or ``None`` to fall through to the
    sandbox path so bash() handles unsupported shape.

    Bytewise output parity required: the model must see identical
    results regardless of which path serviced the call.

    Whitelist:
      - ``cat <path>``                     (single argument)
      - ``head <path>``                    (default 10 lines)
      - ``head -n <N> <path>``             (positive N)

    Security:
      - Path is resolved with ``_resolve`` which clamps under cwd
        (raises ValueError on ``..`` / absolute escape). Same
        primitive used by read/write/edit/glob/grep, so the cwd
        perimeter is consistent across the in-process tool surface.
      - Tokens containing shell metachars cause a fall-through so
        bash handles expansion correctly.
      - Path-prefix-stripped verb match (``/usr/bin/cat`` → ``cat``).
    """
    try:
        argv = shlex.split(cmd, posix=True)
    except ValueError:
        return None
    if not argv:
        return None
    for tok in argv:
        if any(ch in _SHELL_METACHARS for ch in tok):
            return None
    verb = argv[0].rsplit("/", 1)[-1]
    if verb == "cat" and len(argv) == 2:
        return _read_cat(
            argv[1], cwd, ignore_policy=ignore_policy,
            readonly_roots=readonly_roots,
            unreadable_paths=unreadable_paths,
        )
    if verb == "head" and len(argv) == 2:
        return _read_head(
            argv[1], cwd, n=10, ignore_policy=ignore_policy,
            readonly_roots=readonly_roots,
            unreadable_paths=unreadable_paths,
        )
    if verb == "head" and len(argv) == 4 and argv[1] == "-n":
        try:
            n = int(argv[2])
        except ValueError:
            return None
        if n <= 0:
            return None  # head -n 0 / negative — out of scope
        return _read_head(
            argv[3], cwd, n=n, ignore_policy=ignore_policy,
            readonly_roots=readonly_roots,
            unreadable_paths=unreadable_paths,
        )
    if verb == "ls" and ignore_policy is not None:
        return _read_ls(
            argv[1:],
            cwd,
            ignore_policy=ignore_policy,
            readonly_roots=readonly_roots,
            unreadable_paths=unreadable_paths,
        )
    return None


def _read_cat(
    path: str, cwd: str, *, ignore_policy: IgnorePolicy | None = None,
    readonly_roots: tuple[str, ...] = (),
    unreadable_paths: tuple[str, ...] = (),
) -> tuple[str, int, bool]:
    """In-process equivalent of ``cat <path>``. Bytewise output match."""
    try:
        target = _resolve_read(cwd, path, readonly_roots=readonly_roots)
        _require_external_readable(
            cwd, target, unreadable_paths=unreadable_paths,
        )
        if ignore_policy is not None and (
            target == ignore_policy.root or ignore_policy.root in target.parents
        ):
            ignore_policy.require_visible(target, is_dir=target.is_dir())
    except ValueError:
        return f"cat: {path}: No such file or directory\n", 1, False
    except FileNotFoundError:
        return f"cat: {path}: No such file or directory\n", 1, False
    try:
        data = target.read_bytes()
    except FileNotFoundError:
        return f"cat: {path}: No such file or directory\n", 1, False
    except IsADirectoryError:
        return f"cat: {path}: Is a directory\n", 1, False
    except PermissionError:
        return f"cat: {path}: Permission denied\n", 1, False
    return data.decode("utf-8", errors="replace"), 0, False


def _read_head(
    path: str,
    cwd: str,
    *,
    n: int,
    ignore_policy: IgnorePolicy | None = None,
    readonly_roots: tuple[str, ...] = (),
    unreadable_paths: tuple[str, ...] = (),
) -> tuple[str, int, bool]:
    """In-process equivalent of ``head [-n N] <path>`` (default N=10)."""
    try:
        target = _resolve_read(cwd, path, readonly_roots=readonly_roots)
        _require_external_readable(
            cwd, target, unreadable_paths=unreadable_paths,
        )
        if ignore_policy is not None and (
            target == ignore_policy.root or ignore_policy.root in target.parents
        ):
            ignore_policy.require_visible(target, is_dir=target.is_dir())
    except ValueError:
        return (
            f"head: cannot open '{path}' for reading: "
            "No such file or directory\n", 1, False,
        )
    except FileNotFoundError:
        return (
            f"head: cannot open '{path}' for reading: "
            "No such file or directory\n", 1, False,
        )
    try:
        data = target.read_bytes()
    except FileNotFoundError:
        return (
            f"head: cannot open '{path}' for reading: "
            "No such file or directory\n", 1, False,
        )
    except IsADirectoryError:
        return (
            f"head: error reading '{path}': Is a directory\n", 1, False,
        )
    except PermissionError:
        return (
            f"head: cannot open '{path}' for reading: Permission denied\n",
            1, False,
        )
    text = data.decode("utf-8", errors="replace")
    # GNU head emits the first N newline-terminated lines. Split on
    # '\n' and rejoin first N parts; re-add a trailing newline iff
    # the file had at least N full lines (i.e. an N-th '\n' existed).
    parts = text.split("\n")
    head_text = "\n".join(parts[:n])
    if len(parts) > n:
        head_text += "\n"
    return head_text, 0, False


def _read_ls(
    args: list[str],
    cwd: str,
    *,
    ignore_policy: IgnorePolicy,
    readonly_roots: tuple[str, ...] = (),
    unreadable_paths: tuple[str, ...] = (),
) -> tuple[str, int, bool]:
    """Serve simple captured-output ``ls`` calls from the filtered view.

    Captured GNU ``ls`` writes one entry per line. The filtered view supports
    common ``-1``, ``-a``, ``-A``, ``-l``, and ``-h`` combinations; long
    metadata is intentionally omitted rather than risking a masked entry in a
    host-formatted line. Other options fail closed while a repository ignore
    policy is active.
    """
    show_all = False
    almost_all = False
    paths: list[str] = []
    options_done = False
    for arg in args:
        if not options_done and arg == "--":
            options_done = True
            continue
        if not options_done and arg.startswith("--"):
            if arg == "--all":
                show_all = True
                continue
            if arg == "--almost-all":
                almost_all = True
                continue
            return "ls: option is unavailable in the filtered view\n", 2, False
        if not options_done and arg.startswith("-") and arg != "-":
            flags = arg[1:]
            if any(flag not in "1aAlh" for flag in flags):
                return "ls: option is unavailable in the filtered view\n", 2, False
            show_all = show_all or "a" in flags
            almost_all = almost_all or "A" in flags
            continue
        paths.append(arg)
    if len(paths) > 1:
        return "ls: multiple paths are unavailable in the filtered view\n", 2, False
    display = paths[0] if paths else "."
    try:
        target = _resolve_read(cwd, display, readonly_roots=readonly_roots)
        inside_project = (
            target == ignore_policy.root or ignore_policy.root in target.parents
        )
        _require_external_readable(
            cwd, target, unreadable_paths=unreadable_paths,
        )
        if inside_project and ignore_policy.is_model_hidden(
            target, is_dir=target.is_dir(),
        ):
            raise FileNotFoundError(display)
    except (ValueError, FileNotFoundError):
        return (
            f"ls: cannot access '{display}': No such file or directory\n",
            2,
            False,
        )
    if target.is_file():
        return f"{target.name}\n", 0, False
    if not target.is_dir():
        return (
            f"ls: cannot access '{display}': No such file or directory\n",
            2,
            False,
        )
    try:
        names = []
        inside_project = (
            target == ignore_policy.root or ignore_policy.root in target.parents
        )
        mask_roots = tuple(
            Path(path) for path in ignore_policy.existing_ignored_paths()
        ) if inside_project else ()
        for entry in target.iterdir():
            if entry.name.startswith(".") and not (show_all or almost_all):
                continue
            is_dir = entry.is_dir()
            if inside_project:
                ignored = ignore_policy.is_ignored(entry, is_dir=is_dir)
                hidden_directory = is_dir and any(
                    entry == root or entry.is_relative_to(root)
                    for root in mask_roots
                )
                if ignored and (not is_dir or hidden_directory):
                    continue
            else:
                try:
                    _require_external_readable(
                        cwd, entry, unreadable_paths=unreadable_paths,
                    )
                except FileNotFoundError:
                    continue
            names.append(entry.name)
    except OSError as exc:
        return f"ls: cannot open directory '{display}': {exc.strerror}\n", 2, False
    names.sort()
    if show_all:
        names[:0] = [".", ".."]
    return ("\n".join(names) + ("\n" if names else ""), 0, False)


# Verb -> {exit_code: explanation} table for _semantic_exit_annotation.
# Conservative: only verbs whose non-zero exit is documented as a
# non-error condition. Exit codes from POSIX / GNU manpages.
_SEMANTIC_EXIT_TABLE: dict[str, dict[int, str]] = {
    "grep":   {1: "no matches"},
    "egrep":  {1: "no matches"},
    "fgrep":  {1: "no matches"},
    "rg":     {1: "no matches"},
    "ripgrep": {1: "no matches"},
    "ag":     {1: "no matches"},
    "diff":   {1: "files differ"},
    "cmp":    {1: "files differ"},
    "find":   {1: "errors during traversal (consult stderr)"},
    "test":   {1: "condition is false"},
    "[":      {1: "condition is false"},
}


def _semantic_exit_annotation(cmd: str, exit_code: int) -> str | None:
    """Return a short non-error explanation for known verb+exit pairs.

    Examines the FIRST shell verb of the command (post `set -o pipefail`
    stripping not needed — pipefail is added by _build_bwrap_argv after
    the model's text is captured). Returns None for verbs not in the
    table, exit codes not in the verb's row, or commands that are too
    complex to safely classify (multiple verbs in a pipe → ambiguous
    which exit code we have).
    """
    try:
        parts = shlex.split(cmd, posix=True)
    except ValueError:
        return None
    if not parts:
        return None
    # Only annotate single-verb commands. A pipe means the exit comes from
    # the LAST verb under pipefail — but we can't tell which without
    # running it, so be conservative and skip multi-verb cases.
    if any(p in ("|", "&&", "||", ";") for p in parts):
        return None
    verb = parts[0]
    # Strip path prefix: /usr/bin/grep → grep
    verb = verb.rsplit("/", 1)[-1]
    row = _SEMANTIC_EXIT_TABLE.get(verb)
    if row is None:
        return None
    return row.get(exit_code)


def bash(cmd: str, *, cwd: str, timeout: int, sandbox: bool = True,
         bwrap_bin: str = _DEFAULT_BWRAP_BIN,
         sandbox_required: bool = False,
         unreadable_paths: tuple[str, ...] = (),
         readable_paths: tuple[str, ...] = (),
         sandbox_backend: str = "bwrap",
         container_runtime: str = "docker",
         container_runtime_bin: str = "",
         container_image: str = "",
         container_flags: tuple[str, ...] = (),
         effective_env=None,
         allow_login_shell: bool = False,
         transform_output: bool = True) -> str:
    """Run a shell command, return stdout+stderr.

    With the bwrap backend, the command runs in a mount namespace that treats
    the host as read-only and only ``cwd`` as writable. With the container
    backend, it runs in one no-network Docker/Podman container with ``cwd``
    mounted at the identical absolute path.

    A selected sandbox backend must be available. Missing sandbox machinery
    fails closed; host execution occurs only when the resolved policy is the
    explicit ``none`` choice and the caller supplies ``sandbox=False``.

    ``transform_output=False`` returns the captured stdout and stderr without
    deterministic normalization, exit annotations, hints, or empty-output
    replacement.
    """
    # In-process fast path for whitelisted trivial reads (cat / head)
    # that would otherwise round-trip through bwrap+bash. Only when
    # sandbox=True; sandbox=False means the caller explicitly asked
    # for the unsandboxed path (rare; typically tests) and going
    # through Python file I/O still respects cwd containment so it's
    # safe — but staying on the configured path is less surprising.
    ignore_policy = active_ignore_policy(cwd)
    if (
        ignore_policy is not None
        and ignore_policy.enabled
        and ignore_policy.sources
    ):
        # Preserve ignore-file invisibility for the common direct read/list
        # commands even where a single-file mount would leave its directory
        # entry enumerable (and for container backends with the same limit).
        inproc = _try_inproc_trivial_read(
            cmd, cwd, ignore_policy=ignore_policy,
            readonly_roots=readable_paths,
            unreadable_paths=unreadable_paths,
        )
    elif (
        sandbox
        and sandbox_backend == "bwrap"
        and container_mode() is None
    ):
        inproc = _try_inproc_trivial_read(
            cmd, cwd, readonly_roots=readable_paths,
            unreadable_paths=unreadable_paths,
        )
    else:
        inproc = None
    if inproc is not None:
        out, exit_code, timed_out = inproc
    else:
        # Function-local import: tests patch `harness.tools._run_in_sandbox`
        # via mock.patch.object — looking the symbol up via the public
        # `tools` module here makes that patch intercept this call.
        from ..tools import _run_in_sandbox
        out, exit_code, timed_out = _run_in_sandbox(
            cmd, cwd=cwd, timeout=timeout, sandbox=sandbox, bwrap_bin=bwrap_bin,
            sandbox_required=sandbox_required,
            unreadable_paths=unreadable_paths,
            readable_paths=readable_paths,
            sandbox_backend=sandbox_backend,
            container_runtime=container_runtime,
            container_runtime_bin=container_runtime_bin,
            container_image=container_image,
            container_flags=container_flags,
            effective_env=effective_env,
            allow_login_shell=allow_login_shell,
            normalize_output=transform_output,
        )
    if timed_out:
        return ToolExecutionText(
            f"ERROR: command timed out after {timeout}s",
            exit_status=None,
            timed_out=True,
        )
    if exit_code is None:
        # Non-timeout exception path; out already carries "ERROR: …".
        return ToolExecutionText(out, exit_status=None)
    if transform_output and exit_code != 0:
        # Semantic exit-code annotation for known non-error verbs (grep,
        # rg, find, diff). Without this, exit=1 from `grep pattern foo`
        # looks like a generic failure and the model spends 1-2 turns
        # probing whether the command itself was wrong, when in fact it
        # simply found no matches. This follows Hermes `_interpret_exit_code`.
        annotation = _semantic_exit_annotation(cmd, exit_code)
        if annotation is not None:
            out += f"\n[exit code: {exit_code} — {annotation}]"
        else:
            out += f"\n[exit code: {exit_code}]"
    # Empty-output substitution. When the command produced no characters
    # at all (e.g. silent `mv`, `chmod`, `git add`, `touch`, `rm` on a
    # pre-existing file, sed -i with no match), the model otherwise re-
    # probes ("did my command run?") burning a turn. SWE-agent uses
    # exactly this pattern. See offload P0-1 / compaction tier 1 #3.
    # Only applied on success — failures need to surface their actual
    # (possibly empty) error stream rather than a misleading "no output"
    # message.
    if transform_output and exit_code == 0 and out.strip() == "":
        out = "(command produced no output)"
    if transform_output:
        if _pytest_binary_missing(out, exit_code):
            out += _PYTEST_BINARY_MISSING_HINT
        elif _pytest_path_missing(out, exit_code):
            out += _PYTEST_PATH_MISSING_HINT
        elif _sealed_install_failure(cmd, out, exit_code):
            out += _SEALED_INSTALL_FAILURE_HINT
        elif _python_env_missing(out, exit_code):
            out += _PYTHON_ENV_MISSING_HINT
    return ToolExecutionText(out, exit_status=exit_code)
