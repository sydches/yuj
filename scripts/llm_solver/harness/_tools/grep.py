"""grep tool: search file contents with regex via ripgrep or grep fallback."""
import os
import re
import shutil
import subprocess

from ...config import Config
from .._tool_filters import _strip_cwd_absolute
from ..sandbox.ignore_policy import IgnorePolicy, active_ignore_policy
from ._common import _paginated_envelope, _resolve
from ..task_path import TaskPath


# "path:lineno:content" — the shape both rg -n and grep -rn emit. Non-greedy
# path so the first ":<digits>:" wins; a path containing a colon still parses
# because a literal colon is not followed by digits-then-colon at that point.
_MATCH_LINE_RE = re.compile(r'^(.*?):(\d+):')


_NATIVE_SEARCH = r'''
native_search() {
    local readlink_bin=$1 pattern=$2 input_fd code target index
    local -a batch targets inputs opened
    local -A labels
    while :; do
        mapfile -d '' -n 64 -t batch
        (( ${#batch[@]} )) || break
        mapfile -d '' -t targets < <("$realpath_bin" -m -z -- "${batch[@]}")
        wait "$!" || return 74
        (( ${#targets[@]} == ${#batch[@]} )) || return 74
        inputs=(); labels=()
        for index in "${!targets[@]}"; do
            target=${targets[$index]}
            case "$target" in "$root"|"${root%/}/"*) ;; *) return 77 ;; esac
            [[ -f "$target" ]] || continue
            exec {input_fd}< "$target" || return 74
            inputs+=("/proc/self/fd/$input_fd")
            labels[$input_fd]=${batch[$index]}
        done
        (( ${#inputs[@]} )) || continue
        mapfile -d '' -t opened < <("$readlink_bin" -z -- "${inputs[@]}")
        wait "$!" || return 74
        (( ${#opened[@]} == ${#inputs[@]} )) || return 74
        for target in "${opened[@]}"; do
            case "$target" in "$root"|"${root%/}/"*) ;; *) return 77 ;; esac
        done
        "$utility" -n --with-filename --color=never -- "$pattern" "${inputs[@]}" |
            while IFS= read -r line || [[ -n "$line" ]]; do
                if [[ "$line" =~ ^/proc/self/fd/([0-9]+):(.*)$ ]]; then
                    printf '%s:%s\n' "${labels[${BASH_REMATCH[1]}]}" "${BASH_REMATCH[2]}"
                else
                    printf '%s\n' "$line"
                fi
            done
        code=${PIPESTATUS[0]}
        for input_fd in "${!labels[@]}"; do exec {input_fd}<&-; done
        (( code <= 1 )) || return "$code"
    done
}
'''


def _sort_key(line: str) -> tuple:
    """Order matches by (path, line number) — file order, then position."""
    m = _MATCH_LINE_RE.match(line)
    if not m:
        # Not a match line (rg context/summary output). Keep such lines
        # together and after real matches rather than interleaving them by
        # accident; the tuple's first element does the separating.
        return (1, line, 0)
    return (0, m.group(1), int(m.group(2)))


def _sorted_matches(raw: str) -> str:
    """Deterministic match order, independent of the backend's walk order.

    ripgrep does not guarantee walk order, and grep follows file-system order.
    Sort the result so pagination shows the same matches for the same tree.

    Sorting here rather than via `rg --sort path` keeps rg's parallel walk (the
    flag forces single-threaded) and makes the tool behave identically whether
    or not rg is installed.
    """
    if not raw:
        return raw
    lines = raw.splitlines()
    trailing_newline = raw.endswith("\n")
    out = "\n".join(sorted(lines, key=_sort_key))
    result = out + "\n" if trailing_newline and out else out
    from ..savings import get_ledger
    get_ledger().record_transform(
        bucket="search_normalize",
        layer="harness",
        mechanism="grep_match_sort",
        before=raw,
        after=result,
        surface="tool_output",
        change_count=1,
    )
    return result


def _filter_ignored_matches(raw: str, policy: IgnorePolicy) -> str:
    """Remove match rows whose path is outside the model-visible view."""
    if not raw:
        return raw
    kept: list[str] = []
    for line in raw.splitlines():
        match = _MATCH_LINE_RE.match(line)
        if match is None or not policy.is_ignored(
            match.group(1), is_dir=False
        ):
            kept.append(line)
    result = "\n".join(kept) + (
        "\n" if kept and raw.endswith("\n") else ""
    )
    from ..savings import get_ledger
    get_ledger().record_transform(
        bucket="search_filter",
        layer="harness",
        mechanism="ignored_match_filter",
        before=raw,
        after=result,
        surface="tool_output",
        change_count=max(1, len(raw.splitlines()) - len(kept)),
    )
    return result


def grep_files(
    pattern: str, path: str = ".", glob_filter: str = "",
    *, cwd: str, timeout: int = 30,
    page: int = 1, cfg: Config | None = None,
) -> str:
    """Search file contents with regex using ripgrep or grep fallback.

    When ``cfg.search_pagination_enabled`` is true, wraps the result
    in a ``<search_result/>`` envelope with total/shown/page/next_page
    attributes. When false or ``cfg`` is None, returns the raw
    line-per-match text (backwards compatible).
    """
    try:
        resolved = _resolve(cwd, path)
        resolved_path = str(resolved)
    except ValueError as e:
        return f"ERROR: {e}"
    policy = active_ignore_policy(cwd)
    if policy is not None and policy.is_model_hidden(
        resolved, is_dir=resolved.is_dir()
    ):
        return "No matches found."
    rg = shutil.which("rg") if not isinstance(resolved, TaskPath) else None
    try:
        if isinstance(resolved, TaskPath):
            from ..time_budget import command_time_budget
            from ..task_files import TaskUtilityUnavailable
            with command_time_budget(timeout):
                try:
                    candidates = [TaskPath(resolved.files, path) for path in
                                  resolved.files.search_files(str(resolved), glob_filter)]
                    native_selection = True
                except TaskUtilityUnavailable:
                    candidates = ([resolved] if resolved.is_file()
                                  else resolved.glob('**/*', files_only=True))
                    native_selection = False
                selected = []
                for candidate in candidates:
                    if not native_selection and glob_filter and not candidate.path.match(glob_filter):
                        continue
                    if policy is not None and policy.is_ignored(candidate, is_dir=False):
                        continue
                    selected.append(os.fsencode(str(candidate)))
                files = resolved.files
                utility = 'rg' if native_selection else 'grep'
                output = files._call('search_batch', files.root, utility=utility,
                    args=(files._utility('readlink'), pattern),
                    data=b''.join(path + b'\0' for path in selected),
                    script_prefix=_NATIVE_SEARCH)
            result = subprocess.CompletedProcess([], 0, output.decode('utf-8', 'replace'), '')
        else:
            from ._local_grep import local_grep
            result = local_grep(cwd, resolved, pattern, glob_filter, rg=rg,
                                policy=policy, timeout=timeout)
        # rg/grep exit-code semantics: 0 = matches, 1 = no matches
        # (legitimate empty result), 2+ = error (bad regex, missing
        # path, unreadable file, ...). Surface stderr instead of
        # silently returning total=0.
        if result.returncode >= 2:
            stderr = (result.stderr or "").strip().splitlines()
            first = stderr[0] if stderr else f"exit code {result.returncode}"
            first = _strip_cwd_absolute(first, cwd)
            return f"ERROR: grep failed: {first}"
        raw = result.stdout
        # Stable ordering and arm-neutral paths are harness invariants. They
        # must not disappear when the cleanup factor is ablated.
        if raw:
            raw = _strip_cwd_absolute(raw, str(resolved.files.root) if isinstance(resolved, TaskPath) else cwd)
        if policy is not None:
            raw = _filter_ignored_matches(raw, policy)
        raw = _sorted_matches(raw)
        if cfg is None or not cfg.search_pagination_enabled:
            return raw or "No matches found."
        lines = raw.splitlines() if raw else []
        scope = f"{path}" + (f" glob={glob_filter}" if glob_filter else "")
        return _paginated_envelope(
            tool="grep", pattern=pattern, scope=scope,
            lines=lines, page=page,
            per_page=cfg.grep_max_matches_per_page,
            before_text=raw,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: grep timed out after {timeout}s"
    except Exception as e:
        return f"ERROR: {e}"
