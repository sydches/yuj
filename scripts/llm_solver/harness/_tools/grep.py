"""grep tool: search file contents with regex via ripgrep or grep fallback."""
import re
import shutil
import subprocess

from ...config import Config
from .._tool_filters import _strip_cwd_absolute
from ._common import _paginated_envelope, _resolve


# "path:lineno:content" — the shape both rg -n and grep -rn emit. Non-greedy
# path so the first ":<digits>:" wins; a path containing a colon still parses
# because a literal colon is not followed by digits-then-colon at that point.
_MATCH_LINE_RE = re.compile(r'^(.*?):(\d+):')


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
    return out + "\n" if trailing_newline and out else out


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
        resolved_path = str(_resolve(cwd, path))
    except ValueError as e:
        return f"ERROR: {e}"
    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "-n", "--no-heading"]
        if glob_filter:
            cmd.extend(["--glob", glob_filter])
        cmd.extend([pattern, resolved_path])
    else:
        cmd = ["grep", "-rn"]
        if glob_filter:
            cmd.extend(["--include", glob_filter])
        cmd.extend([pattern, resolved_path])
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        # rg/grep exit-code semantics: 0 = matches, 1 = no matches
        # (legitimate empty result), 2+ = error (bad regex, missing
        # path, unreadable file, ...). Surface stderr instead of
        # silently returning total=0.
        if result.returncode >= 2:
            stderr = (result.stderr or "").strip().splitlines()
            first = stderr[0] if stderr else f"exit code {result.returncode}"
            return f"ERROR: grep failed: {first}"
        raw = _strip_cwd_absolute(result.stdout, cwd) if result.stdout else result.stdout
        raw = _sorted_matches(raw)
        if cfg is None or not cfg.search_pagination_enabled:
            return raw or "No matches found."
        lines = raw.splitlines() if raw else []
        scope = f"{path}" + (f" glob={glob_filter}" if glob_filter else "")
        return _paginated_envelope(
            tool="grep", pattern=pattern, scope=scope,
            lines=lines, page=page,
            per_page=cfg.grep_max_matches_per_page,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: grep timed out after {timeout}s"
    except Exception as e:
        return f"ERROR: {e}"
