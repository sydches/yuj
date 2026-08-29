"""Output filtering helpers for tools.py — extracted to reduce file size.

All pure functions; the bash() tool uses these to normalize subprocess
output before returning to the model.
"""
from dataclasses import dataclass
import logging
import re
import shlex
import shutil
import subprocess
from typing import Callable
from pathlib import Path

from ..config import Config
from .sandbox import _DEFAULT_BWRAP_BIN, _build_bwrap_argv
from .tool_policy import PermissionPolicy, PermissionResolution

log = logging.getLogger(__name__)


def resolve_tool_permission(
    *,
    policy: PermissionPolicy,
    tool_name: str,
    arguments: dict,
    cfg: Config,
    approval_available: bool,
) -> PermissionResolution:
    """Evaluate the general policy before any tool-specific filter layer."""
    return policy.evaluate(
        tool_name=tool_name,
        arguments=arguments,
        runtime_mode=getattr(cfg, "runtime_mode", "measurement"),
        ask_fallback=getattr(cfg, "permissions_ask_fallback", "deny"),
        approval_available=approval_available,
    )

# ── ANSI pattern ────────────────────────────────────────────────────────
# Terminal control protocol — universal across subprocess output. Stripping
# it does not inspect task content. The harness does not rewrite paths that an
# outside task runner adds; that runner must clean its own output.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# ls -l style listing: permissions + metadata + date + filename.
# Under temp=0, varying wall-clock mtimes on harness-owned files (e.g.
# .trace.jsonl which appends every turn) flip the model's sampled path
# turn-over-turn.  Replace the date column with a fixed placeholder
# while preserving column alignment; filename and content are untouched.
_LS_LONG_RE = re.compile(
    r'^([-dlcbps][rwxstST-]{9}[+.@]?\s+\d+\s+\S+\s+\S+\s+\d+\s+)'
    r'[A-Z][a-z]{2}\s+\d{1,2}\s+(?:\d{1,2}:\d{2}|\d{4})'
    r'(\s+\S)',
    re.MULTILINE,
)



def _strip_ls_timestamps(output: str) -> str:
    """Replace ls -l style date columns with a fixed placeholder.

    Only touches lines matching the exact ls long-format shape; other
    dates elsewhere in output are left alone.
    """
    result, count = _LS_LONG_RE.subn(r'\1Jan  1  2020\2', output)
    return _record_text_change(
        output, result,
        bucket="tool_output_normalize",
        mechanism="ls_timestamp_normalization",
        change_count=count,
    )


def _resolve(cwd: str, path: str) -> Path:
    """Resolve a tool path relative to cwd, even if absolute.

    Absolute paths are resolved relative to cwd (sandbox-style)
    so the model can't escape the working directory.
    """
    if path.startswith("/"):
        path = path.lstrip("/")
    return Path(cwd) / path


def _path_hint(cwd: str, path: str) -> str:
    """Suggest a corrected path when a file-not-found error occurs.

    Catches the `.seaborn/X` → `seaborn/X` pattern where the model
    confuses `.solver/` (hidden dir) with the package directory.
    """
    stripped = path.lstrip("./")
    if stripped != path:
        candidate = Path(cwd) / stripped
        if candidate.exists():
            return f" (did you mean '{stripped}'?)"
    return ""


def _record_text_change(*args, **kwargs) -> str:
    """Record one tool-output text change and return its output."""
    from .savings import record_text_transform
    return record_text_transform(*args, **kwargs)


def output_cleanup_enabled(cfg: Config) -> bool:
    """Return whether the high-level output transformation is active."""
    return not bool(getattr(cfg, "transformations_explicit", False)) or bool(
        getattr(cfg, "output_cleanup_and_normalization", True)
    )


def truncate_output(text: str, cfg: Config) -> str:
    """Head+tail truncation when output exceeds max_output_chars.

    Budget-targeted slice: any tool result at or under max_output_chars
    passes through untouched. Over-budget results are head+tail sliced
    such that the resulting output is roughly max_output_chars, split
    by ``cfg.truncate_head_ratio`` (default 0.4 → 40 % head / 60 % tail
    per Hermes pattern; failing-test traceback / [exit code: N] land at
    the tail and need to survive intact) and rounded to full line
    boundaries when possible.

    The earlier design sliced by fixed truncate_head_lines /
    truncate_tail_lines counts, which was a problem once max_output_chars
    was raised: an 87 KB Python source file with ~100-char lines would
    be cut to ~30 KB (300 lines) even though the budget allowed 80 KB,
    silently throwing away ~70% of the code on first read. The budget
    SHOULD be what governs; line counts are a derived thing.

    Line-count fields (truncate_head_lines / truncate_tail_lines) are
    still used as a floor — for command output with a few very long
    lines, they guarantee at least N logical lines of each end survive
    even if the char budget math would leave nothing. This preserves
    the readability property for bash log tails.
    """
    if not output_cleanup_enabled(cfg):
        return text

    budget = cfg.max_output_chars
    if len(text) <= budget:
        return text

    # Reserve a small overhead for the "[... omitted ...]" marker so the
    # final result fits under budget.
    marker_reserve = 80
    slice_budget = max(1, budget - marker_reserve)
    head_budget = int(slice_budget * cfg.truncate_head_ratio)
    tail_budget = slice_budget - head_budget

    # Char-based head/tail respecting full lines where possible.
    head = text[:head_budget]
    # Back up to the last newline so we don't cut a line mid-way.
    last_nl = head.rfind("\n")
    if last_nl > head_budget // 2:
        head = head[: last_nl + 1]

    tail = text[-tail_budget:]
    first_nl = tail.find("\n")
    if 0 <= first_nl < tail_budget // 2:
        tail = tail[first_nl + 1 :]

    omitted = len(text) - len(head) - len(tail)
    truncated = f"{head}\n[... {omitted} chars omitted ...]\n{tail}"
    return _record_text_change(
        text, truncated,
        bucket="truncate_output",
        mechanism="head_tail_truncation",
        ctx={"head_ratio": cfg.truncate_head_ratio},
    )


def _collapse_duplicate_lines(output: str) -> str:
    """Collapse runs of byte-identical consecutive lines into '<line> [×N]'.

    Content-blind redundancy compression: compresses on byte equality,
    knows nothing about what the lines represent. Works on retry-loop
    spam, progress-bar repeats, log rotation, test runners that print
    identical status lines — anything that repeats verbatim. No task-
    format vocabulary is named.

    A run of length 1 passes through unchanged (no overhead annotation
    for unique lines). Runs of 2+ identical lines collapse to the line
    followed by a compact count suffix.
    """
    lines = output.split("\n")
    out: list[str] = []
    prev: str | None = None
    count = 0
    collapsed_runs = 0
    for line in lines:
        if prev is not None and line == prev:
            count += 1
            continue
        if prev is not None:
            if count > 1:
                out.append(f"{prev} [×{count}]")
                collapsed_runs += 1
            else:
                out.append(prev)
        prev = line
        count = 1
    if prev is not None:
        if count > 1:
            out.append(f"{prev} [×{count}]")
            collapsed_runs += 1
        else:
            out.append(prev)
    result = "\n".join(out)
    return _record_text_change(
        output, result,
        bucket="tool_output_filter",
        mechanism="collapse_duplicate_lines",
        change_count=collapsed_runs,
    )


# ── Structural skeleton patterns ────────────────────────────────────────
# Used by _collapse_similar_lines to detect structurally identical lines
# that differ only in variable alphanumeric content (names, numbers, etc.).
_ALNUM_RE = re.compile(r"[A-Za-z0-9]+")
_WS_RE = re.compile(r"\s+")


def _line_skeleton(line: str) -> str:
    """Return the structural skeleton of a line.

    Replaces every run of alphanumeric characters with a NUL placeholder
    and collapses whitespace runs.  Lines with identical skeletons share
    the same punctuation/delimiter template and differ only in their
    variable alphanumeric content (names, numbers, percentages).

    Content-blind: operates only on character-class properties, not on
    any knowledge of what the alphanumeric values represent.
    """
    s = _ALNUM_RE.sub("\x00", line)
    s = _WS_RE.sub(" ", s)
    return s.strip()


def _collapse_similar_lines(output: str) -> str:
    """Collapse high-frequency structural templates, keep rare lines verbatim.

    Two-pass, frequency-based:
      1. Skeleton every line, count skeleton frequencies.
      2. Skeletons that account for > 50% of non-blank lines are "bulk" —
         consecutive runs of bulk lines collapse to first + count + last.
         All other lines pass through unchanged.

    The effect: in a 14K-line pytest run, the 12K PASSED lines share one
    dominant skeleton and collapse.  The 119 FAILED lines, the header,
    the summary, and every structurally unique line survive intact.

    For small outputs or outputs with no dominant template, nothing
    collapses — every skeleton is rare.

    Content-blind: decides by skeleton frequency, not by what the lines
    say.  Works on any tool output where one structural pattern dominates.
    """
    lines = output.split("\n")
    n_nonblank = sum(1 for l in lines if l.strip())
    if n_nonblank < 10:
        return output

    # Pass 1: skeleton frequencies.
    skeletons = []
    freq: dict[str, int] = {}
    for line in lines:
        if not line.strip():
            skeletons.append("")
            continue
        skel = _line_skeleton(line)
        skeletons.append(skel)
        freq[skel] = freq.get(skel, 0) + 1

    # Bulk threshold: skeletons covering > 50% of non-blank lines.
    threshold = n_nonblank * 0.5
    bulk = {s for s, c in freq.items() if s and c > threshold}

    if not bulk:
        return output

    # Pass 2: emit. Consecutive bulk lines collapse; rare lines pass through.
    out: list[str] = []
    i = 0
    collapsed_runs = 0
    while i < len(lines):
        skel = skeletons[i]
        if skel not in bulk:
            out.append(lines[i])
            i += 1
            continue

        # Start of a bulk run — scan forward.
        j = i + 1
        while j < len(lines) and skeletons[j] == skel:
            j += 1
        run_len = j - i
        if run_len < 3:
            out.extend(lines[i:j])
        else:
            out.append(lines[i])
            out.append(f"  ... [×{run_len} similar lines]")
            out.append(lines[j - 1])
            collapsed_runs += 1
        i = j

    result = "\n".join(out)
    return _record_text_change(
        output, result,
        bucket="tool_output_filter",
        mechanism="collapse_similar_lines",
        change_count=collapsed_runs,
    )



def _filter_bash_output(output: str, cmd: str, cfg: Config) -> str:
    """Content-agnostic filtering of bash output before truncation.

    Transforms (each gated by a config toggle):
      1. Strip ANSI escape sequences  (cfg.strip_ansi)
      2. Collapse runs of blank lines  (cfg.collapse_blank_lines)
      3. Collapse runs of byte-identical lines  (cfg.collapse_duplicate_lines)

    Every transformation is content-blind: no task-format vocabulary,
    no test-runner detection, no error-message pattern matching. Each
    operates on universal properties of text — terminal control
    protocol, whitespace, byte equality. Task-format parsing belongs
    in the analysis layer, never in the solve loop.
    """
    if not output_cleanup_enabled(cfg):
        return output

    # 1. ANSI escapes — terminal control protocol, universal noise.
    if cfg.strip_ansi:
        before = output
        output, count = _ANSI_RE.subn("", output)
        output = _record_text_change(
            before, output,
            bucket="tool_output_filter",
            mechanism="strip_ansi",
            change_count=count,
        )

    # 2. Collapse blank-line runs (3+ consecutive newlines → 2).
    if cfg.collapse_blank_lines:
        before = output
        output, count = re.subn(r"\n{3,}", "\n\n", output)
        output = _record_text_change(
            before, output,
            bucket="tool_output_filter",
            mechanism="collapse_blank_lines",
            change_count=count,
        )

    # 3. Collapse runs of byte-identical consecutive lines.
    if cfg.collapse_duplicate_lines:
        output = _collapse_duplicate_lines(output)

    # 4. Collapse runs of structurally similar lines (same skeleton).
    #    Only activates when the output is large enough that truncation
    #    is a real threat (>50% of the char budget).  Small outputs pass
    #    through untouched — collapsing an `ls` or short grep loses
    #    unique information (filenames, paths) for negligible savings.
    if cfg.collapse_similar_lines and len(output) > cfg.max_output_chars * 0.5:
        output = _collapse_similar_lines(output)

    # 5. Fold byte-identical consecutive Python-traceback FRAMES into
    #    one line. Recursion errors and N-times-retried subprocess
    #    failures emit hundreds of identical
    #      File "/some/path.py", line 42, in foo
    #          do_thing()
    #    pairs that compress to one block + a count. Content-blind
    #    (operates on the literal `File "..."` syntax). Independent of
    #    collapse_duplicate_lines because that one operates on single
    #    lines while a Python frame is a 2-line tuple.
    output = _fold_traceback_frames(output)

    return output


# Python traceback frame: matches the canonical
#   File "<path>", line <num>, in <name>
# emitted by every CPython traceback (including chained, with-cause,
# and recursion variants). The next line is the source snippet (single
# line; multi-line snippets are rare and would still match per-frame).
_PY_FRAME_RE = re.compile(
    r"^(?P<frame>\s*File \"[^\"]+\", line \d+, in \S+)\n"
    r"(?P<src>(?:[ \t]+.*)?)\n",
    re.MULTILINE,
)


def _fold_traceback_frames(output: str, *, min_run: int = 3) -> str:
    """Collapse byte-identical consecutive Python frames into '<frame> [×N]'.

    A frame here = the two-line tuple
        File "<path>", line <line>, in <name>
            <source line>
    that CPython emits for every stack-walk entry. Recursion failures
    produce hundreds of identical frames; subprocess wrappers that
    re-raise with the same traceback in a retry loop multiply this
    further.

    Only collapses runs of `min_run` (default 3) or more identical
    frames so a normal 2-line traceback (test_foo → asserter) is left
    intact. Operates on output AFTER ANSI strip / blank-line collapse
    so byte equality is the right comparison.

    Idempotent — re-running on already-collapsed output is a no-op
    because the elision marker is not itself a frame.
    """
    matches = list(_PY_FRAME_RE.finditer(output))
    if not matches:
        return output

    # Find runs of consecutive matches with identical full text whose
    # raw spans also abut (no other content between the frames in the
    # original output).
    out_parts: list[str] = []
    cursor = 0
    i = 0
    collapsed_runs = 0
    while i < len(matches):
        m = matches[i]
        # Walk forward as long as the next match abuts and has identical text.
        run_end = i
        while (run_end + 1 < len(matches)
               and matches[run_end + 1].start() == matches[run_end].end()
               and matches[run_end + 1].group(0) == m.group(0)):
            run_end += 1
        run_len = run_end - i + 1
        if run_len >= min_run:
            # Emit prefix verbatim, then one frame + elision marker.
            out_parts.append(output[cursor : m.start()])
            out_parts.append(m.group(0))
            out_parts.append(f"  [... above frame repeated ×{run_len} (folded by harness) ...]\n")
            cursor = matches[run_end].end()
            collapsed_runs += 1
        i = run_end + 1
    out_parts.append(output[cursor:])
    result = "".join(out_parts)
    return _record_text_change(
        output, result,
        bucket="tool_output_filter",
        mechanism="fold_traceback_frames",
        change_count=collapsed_runs,
    )


_STATUS_WORD_RE = re.compile(
    r'\b(?:passed|failed|error|warnings?|deselected|no tests ran|no tests collected)\b'
)
_TIMING_RE = re.compile(r'\s*in\s+\d+\.\d+s')


def _strip_runner_timing(output: str) -> str:
    """Strip wall-clock timing from pytest/unittest lines in bash output.

    Applies to any line containing a pytest status word (passed/failed/
    error/warning/deselected) or the "no tests ran"/"no tests collected"
    phrases. Removes ` in X.YZs` — varies sub-second per invocation and
    flips temp=0 paths.
    """
    out_lines = []
    change_count = 0
    for line in output.split('\n'):
        if _STATUS_WORD_RE.search(line):
            line, count = _TIMING_RE.subn('', line)
            change_count += count
        out_lines.append(line)
    result = '\n'.join(out_lines)
    return _record_text_change(
        output, result,
        bucket="tool_output_normalize",
        mechanism="runner_timing_normalization",
        change_count=change_count,
    )


def _strip_cwd_absolute(output: str, cwd: str) -> str:
    """Rewrite lexical and resolved cwd paths to ``.`` in tool output.

    ``pwd`` and Python ``__file__`` resolve to the task's absolute path
    which embeds the run_dir timestamp. Under temp=0 the timestamp
    bytes flip the model's sampled next token on subsequent turns.
    Collapsing to ``.`` makes output byte-identical across runs.
    """
    roots = sorted(
        {
            root
            for root in (
                cwd,
                str(Path(cwd).resolve(strict=False)) if cwd else "",
            )
            if root
        },
        key=len,
        reverse=True,
    )
    result = output
    change_count = 0
    for root in roots:
        change_count += result.count(root)
        result = result.replace(root, ".")
    return _record_text_change(
        output, result,
        bucket="tool_output_normalize",
        mechanism="cwd_path_normalization",
        change_count=change_count,
    )


# Memory allocation is non-deterministic across otherwise identical runs.
# Match only word-bounded address-shaped values on subprocess output. File
# reads do not pass through this function, so source hex constants stay exact.
_MEMORY_ADDRESS_RE = re.compile(r'\b0x[0-9a-fA-F]{6,16}\b')


def _normalize_memory_addresses(output: str) -> str:
    """Replace memory addresses with stable per-output identity tokens."""
    identities: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        address = match.group(0).lower()
        token = identities.get(address)
        if token is None:
            token = f"0xADDR{len(identities) + 1}"
            identities[address] = token
        return token

    result, count = _MEMORY_ADDRESS_RE.subn(replace, output)
    return _record_text_change(
        output, result,
        bucket="tool_output_normalize",
        mechanism="memory_address_normalization",
        change_count=count,
    )
