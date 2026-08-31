"""Task-format output control + structured parser + condensation."""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class OutputControl:
    """Runner-specific output control loaded from language_quirks/*.toml."""
    failure_only_flag: str
    passed_marker: str
    failed_marker: str
    verification_patterns: tuple[re.Pattern, ...]
    output_parser: "OutputParser | None" = field(
        default=None,
        compare=False,
        hash=False,
        repr=False,
    )


@dataclass(frozen=True)
class OutputParser:
    """Runner-specific structured-output parser loaded from language_quirks/*.toml.

    Each runner's TOML may declare:
      [output_parser.summary]   — field: passed / failed / errors (each a regex)
      [output_parser.per_test]
        regex with named captures: test_id, verdict

    Using one regex per summary field (rather than a single alternation)
    makes parsing order-agnostic: pytest sometimes prints
    "2 failed, 8 passed" and sometimes "8 passed, 2 failed", and either
    needs to work. Per-test uses a single regex because the format is
    uniform (verdict + test_id at line start).

    All fields are optional — absent keys simply produce partial
    parses.
    """
    summary_fields: dict[str, re.Pattern]
    per_test_regex: re.Pattern | None


def load_output_control(task_format_path) -> OutputControl | None:
    """Load [output_control] from a task format TOML file.

    Returns None if the file has no [output_control] section.
    """
    from .._shared.toml_compat import tomllib

    path = Path(task_format_path)
    if not path.is_file():
        return None

    with open(path, "rb") as f:
        data = tomllib.load(f)

    oc = data.get("output_control")
    if not oc:
        return None

    patterns = tuple(
        re.compile(p, re.IGNORECASE | re.VERBOSE)
        for p in data.get("verification_patterns", [])
    )

    return OutputControl(
        failure_only_flag=oc.get("failure_only_flag", ""),
        passed_marker=oc.get("passed_marker", ""),
        failed_marker=oc.get("failed_marker", ""),
        verification_patterns=patterns,
        output_parser=load_output_parser(path),
    )


def load_output_parser(task_format_path) -> OutputParser | None:
    """Load [output_parser] from a task format TOML file.

    Returns None when the file is absent or has no [output_parser] section.
    Individual fields may be missing; the parser tolerates partial config.
    """
    from .._shared.toml_compat import tomllib

    path = Path(task_format_path)
    if not path.is_file():
        return None

    with open(path, "rb") as f:
        data = tomllib.load(f)

    op = data.get("output_parser")
    if not op:
        return None

    # Summary fields: case-insensitive (runners emit "passed" / "PASSED"
    # variably; human-readable summary words). MULTILINE + VERBOSE for
    # formatted pattern strings in the TOML.
    summary_flags = re.MULTILINE | re.VERBOSE | re.IGNORECASE
    summary_fields: dict[str, re.Pattern] = {}
    for field_name, pattern in (op.get("summary") or {}).items():
        if pattern:
            summary_fields[field_name] = re.compile(pattern, summary_flags)

    # Per-test verdicts: the TOML literal is uppercase (PASSED|FAILED|ERROR).
    # IGNORECASE on a regex matching uppercase literals against a large log
    # adds Unicode case-folding overhead to every line-start position for
    # no match gain — drop it. MULTILINE must stay for the ^ anchor.
    per_test_flags = re.MULTILINE | re.VERBOSE
    per_test_cfg = op.get("per_test") or {}
    per_test_regex = None
    if per_test_cfg.get("regex"):
        per_test_regex = re.compile(per_test_cfg["regex"], per_test_flags)

    if not summary_fields and per_test_regex is None:
        return None

    return OutputParser(
        summary_fields=summary_fields,
        per_test_regex=per_test_regex,
    )


# Summary scan window: pytest and most runners emit their terminal
# summary line in the last few hundred chars. Searching only the tail
# avoids iterating hundreds of intermediate "N passed" matches on a
# 100K+ pytest log. Fall back to full-scan if the tail window misses
# (rare — happens when the whole output is short enough that tail ==
# output, or when the runner emits the summary mid-stream).
_SUMMARY_TAIL_CHARS = 4000
_FAILURE_DETAIL_CHARS = 500


# Canonical verdict map — normalizes runner-specific PASS/FAIL tokens
# so downstream consumers (done-parity, render_digest, run_summary)
# don't need to know each runner's vocabulary. Unknown verdicts are
# passed through uppercased (safe default: done-parity checks treat
# them as non-passing).
_VERDICT_NORMALIZE = {
    # canonical
    "PASSED": "PASSED", "FAILED": "FAILED", "ERROR": "ERROR",
    "SKIPPED": "SKIPPED",
    # short forms (pytest -rA, jest CI)
    "PASS": "PASSED", "FAIL": "FAILED", "SKIP": "SKIPPED",
    # cargo
    "OK": "PASSED", "IGNORED": "SKIPPED",
    # jest default reporter (unicode marks)
    "✓": "PASSED", "✕": "FAILED",
    # go
    # (PASS/FAIL already covered)
}


def _normalize_verdict(raw: str) -> str:
    """Map a runner-emitted verdict literal to a canonical PASSED/FAILED/etc.

    Unknown tokens are returned uppercased unchanged. Downstream code
    treats non-canonical tokens as non-passing, so the failure mode of
    an unrecognized verdict is a false negative on done-parity
    checks, not a false positive.
    """
    key = (raw or "").strip().upper()
    # Handle unicode marks before uppercase (✓ is already canonical).
    if raw in _VERDICT_NORMALIZE:
        return _VERDICT_NORMALIZE[raw]
    return _VERDICT_NORMALIZE.get(key, key)


_SHELL_OPERATORS = ("&&", "||", "|&", ";", "|", "&", "\n")


def _shell_command_spans(command: str) -> tuple[tuple[int, int], ...]:
    """Return non-empty top-level command-fragment spans.

    Separators inside quotes, subshells, command substitutions, and braced
    groups stay inside their surrounding fragment. Redirection forms such as
    ``2>&1`` and ``&>file`` are not mistaken for background operators.
    """
    spans: list[tuple[int, int]] = []
    quote = ""
    escaped = False
    paren_depth = 0
    brace_depth = 0
    start = 0
    index = 0

    def add_span(raw_start: int, raw_end: int) -> None:
        while raw_start < raw_end and command[raw_start].isspace():
            raw_start += 1
        while raw_end > raw_start and command[raw_end - 1].isspace():
            raw_end -= 1
        if raw_start < raw_end:
            spans.append((raw_start, raw_end))

    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if quote:
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in "'\"`":
            quote = char
            index += 1
            continue
        if char == "(":
            paren_depth += 1
            index += 1
            continue
        if char == ")" and paren_depth:
            paren_depth -= 1
            index += 1
            continue
        if char == "{":
            brace_depth += 1
            index += 1
            continue
        if char == "}" and brace_depth:
            brace_depth -= 1
            index += 1
            continue
        if paren_depth or brace_depth:
            index += 1
            continue

        operator = ""
        for candidate in _SHELL_OPERATORS:
            if command.startswith(candidate, index):
                operator = candidate
                break
        if operator == "&" and (
            (index > 0 and command[index - 1] in "<>")
            or command.startswith("&>", index)
        ):
            index += 1
            continue
        if not operator:
            index += 1
            continue
        add_span(start, index)
        index += len(operator)
        start = index

    add_span(start, len(command))
    return tuple(spans)


def _strip_leading_assignments(segment: str) -> str:
    """Strip simple leading ``NAME=value`` words from a command fragment."""
    value = segment.strip()
    while True:
        match = re.match(r"^[A-Za-z_][A-Za-z0-9_]*=\S+\s+", value)
        if not match:
            return value
        value = value[match.end():]


def _pattern_matches_command_head(segment: str, pattern: re.Pattern) -> bool:
    """Return whether ``pattern`` identifies the fragment's executable."""
    value = _strip_leading_assignments(segment)
    match = pattern.search(value)
    if match is None:
        return False
    if match.start() == 0:
        return True

    # A rule may match the final slash plus executable in an absolute or
    # relative path. Whitespace before the match means it is in an argument.
    prefix = value[:match.start()]
    return not any(char.isspace() for char in prefix) and match.group(0).startswith("/")


def _find_command_span(
    command: str,
    patterns: tuple[re.Pattern, ...],
) -> tuple[int, int] | None:
    """Find the first top-level fragment invoked by one of ``patterns``."""
    for start, end in _shell_command_spans(command):
        segment = command[start:end]
        if any(_pattern_matches_command_head(segment, pattern) for pattern in patterns):
            return start, end
    return None


def parse_structured(output: str, parser: OutputParser) -> dict:
    """Extract {summary: {passed, failed, errors, ...}, tests: {id: verdict}}.

    Per-field regexes run independently — order-agnostic, so pytest's
    "2 failed, 8 passed, 1 error" parses the same as "8 passed, 2
    failed, 1 error". Last numeric match per field wins (runners
    sometimes emit the tally twice; the second instance is the
    terminal summary). Scan is bounded to the tail of the output when
    the output is large, to avoid sweeping intermediate lines.

    Per-test verdicts are normalized to canonical PASSED/FAILED/
    ERROR/SKIPPED via _normalize_verdict. Callers depending on
    canonical verdict strings (done-parity, regression detection,
    render_digest) work uniformly across runners.
    """
    tail = output if len(output) <= _SUMMARY_TAIL_CHARS else output[-_SUMMARY_TAIL_CHARS:]
    summary: dict[str, int] = {}
    for field_name, rx in parser.summary_fields.items():
        match = None
        for match in rx.finditer(tail):
            pass  # keep the last match
        if match is None and tail is not output:
            # Fall back to full-scan only when the tail missed — covers
            # runners that emit their summary mid-stream.
            for match in rx.finditer(output):
                pass
        if match is None:
            continue
        try:
            summary[field_name] = int(match.group(1))
        except (IndexError, ValueError, TypeError):
            continue

    tests: dict[str, str] = {}
    failure_details: list[str] = []
    if parser.per_test_regex:
        for m in parser.per_test_regex.finditer(output):
            tid = m.groupdict().get("test_id")
            verdict = m.groupdict().get("verdict")
            if tid and verdict:
                normalized = _normalize_verdict(verdict)
                tests[tid] = normalized
                if normalized in {"FAILED", "ERROR"} and len(failure_details) < 3:
                    line_end = output.find("\n", m.end())
                    if line_end < 0:
                        line_end = len(output)
                    detail = output[m.start() : line_end].strip()
                    if detail:
                        failure_details.append(
                            detail
                            if len(detail) <= _FAILURE_DETAIL_CHARS
                            else detail[: _FAILURE_DETAIL_CHARS - 3] + "..."
                        )

    return {
        "summary": summary or None,
        "tests": tests,
        "failure_details": failure_details,
    }


def render_digest(parsed: dict, *, max_failures_shown: int = 10) -> str:
    """Render a parsed test-run record as compact text for the model."""
    lines: list[str] = []
    summary = parsed.get("summary") or {}
    if summary:
        parts: list[str] = []
        # Preserve stable ordering for the model: passed, failed, errors, skipped,
        # then any runner-specific fields alphabetically.
        for k in ("passed", "failed", "errors", "skipped"):
            if k in summary:
                parts.append(f"{summary[k]} {k}")
        for k in sorted(summary):
            if k in ("passed", "failed", "errors", "skipped"):
                continue
            parts.append(f"{summary[k]} {k}")
        if parts:
            lines.append("[digest] " + ", ".join(parts))
    tests = parsed.get("tests") or {}
    failing = [tid for tid, v in tests.items() if v in ("FAILED", "FAIL", "ERROR")]
    if failing:
        shown = failing[:max_failures_shown]
        lines.append(f"[digest] failing ({len(failing)}): " + ", ".join(shown))
        if len(failing) > max_failures_shown:
            lines.append(f"[digest] ... {len(failing) - max_failures_shown} more failing tests")
    failure_details = parsed.get("failure_details") or []
    if failure_details:
        lines.append(f"[digest] first failure: {failure_details[0]}")
    if not lines:
        return ""
    return "\n".join(lines)


def _segment_starts_with_test_pattern(segment: str, verification_patterns) -> bool:
    """Return True iff ``segment`` is a test invocation, not a search.

    A "segment" here is one pipe stage; the segment is a test command
    only if a verification pattern matches at its head (after leading
    whitespace and optional ``env=val`` / ``VAR=val`` assignments). If
    the pattern matches deeper into the segment it's almost certainly
    an arg to grep/ls/find/etc.
    """
    return any(
        _pattern_matches_command_head(segment, pattern)
        for pattern in verification_patterns
    )


@functools.lru_cache(maxsize=256)
def _is_test_command(cmd: str, oc: OutputControl) -> bool:
    """Cached: OutputControl is frozen/hashable, cmds repeat across turns.

    ``rewrite_command``, ``condense_output``, and the harness's
    ``_project_and_sink`` each check whether a bash cmd is a
    verification gate; without caching that loops 6 regexes per
    check × multiple checks per tool call. Cache key is (cmd, oc);
    size 256 covers a long task without unbounded growth.

    A command is a test command only when a pipe segment starts with a
    verification pattern. This avoids adding test flags to commands such
    as `grep pytest`.
    """
    # Split on unquoted top-level shell separators and check whether any
    # *segment* starts with a verification pattern. Catches:
    #   pytest tests/                                 → YES
    #   python -m pytest tests/                       → YES
    #   cd /repo && pytest tests/                     → YES (after &&)
    #   ls /opt/miniconda3/bin/ | grep pytest         → NO
    #   ls -la /opt/miniconda3/envs/ | grep pytest    → NO
    #   which pytest                                  → NO
    return _find_command_span(cmd, oc.verification_patterns) is not None


def condense_output(output: str, cmd: str, oc: OutputControl | None) -> str:
    """Strip passing-result lines from test command output.

    Replaces lines containing passed_marker (but not failed_marker) with a
    neutral omission count. It does not claim that the command or suite
    passed; the exit status and retained failure lines own that verdict.
    No-op when oc is None, markers are empty, or the command isn't a test
    invocation.
    """
    if not oc or not oc.passed_marker:
        return output
    if not _is_test_command(cmd, oc):
        return output

    lines = output.split("\n")
    kept: list[str] = []
    passed_count = 0
    for line in lines:
        if oc.passed_marker in line and oc.failed_marker not in line:
            passed_count += 1
            continue
        kept.append(line)
    if passed_count:
        kept.insert(
            max(len(kept) - 1, 0),
            f"[{passed_count} passing-result lines omitted]",
        )
    result = "\n".join(kept)
    if passed_count:
        # Record the exact before/after text, not only an inferred saving.
        from ..harness.savings import get_ledger
        get_ledger().record_transform(
            bucket="bash_output_condense",
            layer="L2_bash_quirks",
            mechanism="passed_line_stripping",
            before=output,
            after=result,
            surface="tool_output",
            change_count=passed_count,
            ctx={"passed_stripped": passed_count,
                 "passed_marker": oc.passed_marker},
        )
    return result
