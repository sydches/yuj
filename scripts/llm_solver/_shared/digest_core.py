"""Content-blind rendering core for `.trace.jsonl` digests.

This module is content-blind: it operates on trace event dicts (the
shape `harness/loop.py::_write_trace` emits) and harness-generated
markers only. It has no model, language, or task-format knowledge.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


_SR_RE = re.compile(r"<(?:search_result|file_content|test_results|tool_result)[^>]*>")
_SR_CLOSE_RE = re.compile(r"</(?:search_result|file_content|test_results|tool_result)>")
_HARNESS_RE = re.compile(r"\[HARNESS:[^\]]*\]")
_OUTSIDE_CWD_RE = re.compile(r"(?<![\w])/(?:home/(?!task)[^/\s'\"`]+|usr/|etc/|proc/|sys/|var/|root/|opt/)")
_WRITE_TOOLS = {
    "write", "edit", "notebook_edit", "str_replace", "create",
    "apply_patch", "udiff",
}

# Sandbox-denial keyword set ported from Codex
# (codex-rs/core/src/exec.rs::is_likely_sandbox_denied). Surfacing escape
# attempts in the digest's flag column lets readers locate denied operations.
# Keywords are
# matched case-insensitively against the rendered result_summary.
_SANDBOX_DENY_KEYWORDS = (
    "operation not permitted",
    "permission denied",
    "read-only file system",
    "seccomp",
    "landlock",
)
# SIGSYS produces shell exit 128+31 = 159 (seccomp kill). Exit codes 2,
# 126, 127 are NOT sandbox-related (parse error / not-executable /
# command-not-found), so we exclude them when sniffing the result text
# for "[exit code: N" markers.
_SANDBOX_DENY_EXIT = {159}
_SANDBOX_DENY_EXIT_BLOCKLIST = {2, 126, 127}


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _strip_xml(s: str) -> str:
    s = _SR_RE.sub("", s)
    s = _SR_CLOSE_RE.sub("", s)
    return s.strip()


def _flat(s: str) -> str:
    """Single-line, control-char-free."""
    return _strip_xml(s).replace("\n", " ⏎ ").replace("\t", " ").strip()


def _key_arg(tool: str, args: str) -> str:
    """One short representation of the action."""
    m = re.search(r"cmd=['\"]([^'\"]*)['\"]", args)
    if m:
        return f"`{_truncate(m.group(1), 90)}`"
    m = re.search(r"(?:path|file_path|target)=['\"]([^'\"]*)['\"]", args)
    if m:
        path = m.group(1)
        extras = re.findall(r"(\w+)=['\"]([^'\"]*)['\"]", args)
        extras = [(k, v) for k, v in extras if k not in {"path", "file_path", "target"}]
        if extras:
            e = " ".join(f"{k}={_truncate(v, 18)}" for k, v in extras[:2])
            return f"{path} {e}".rstrip()
        return path
    m = re.search(r"pattern=['\"]([^'\"]*)['\"]", args)
    if m:
        path = ""
        m2 = re.search(r"(?:path|scope)=['\"]([^'\"]*)['\"]", args)
        if m2:
            path = f" in {m2.group(1)}"
        return f"pattern='{_truncate(m.group(1), 50)}'{path}"
    return _truncate(args, 80)


def _is_sandbox_denial(rs_lower: str) -> bool:
    """Heuristic port of Codex's is_likely_sandbox_denied (exec.rs).

    True when the result text mentions any sandbox-deny keyword AND
    there's no in-band marker for an exit code that is provably NOT
    sandbox-related (parse error, not-executable, command-not-found).
    SIGSYS exit (159) is also a positive signal even without keywords.
    """
    if any(k in rs_lower for k in _SANDBOX_DENY_KEYWORDS):
        # Avoid false positive when the failure was a clear non-sandbox class.
        for code in _SANDBOX_DENY_EXIT_BLOCKLIST:
            if f"[exit code: {code}" in rs_lower:
                return False
        return True
    for code in _SANDBOX_DENY_EXIT:
        if f"[exit code: {code}" in rs_lower:
            return True
    return False


def _flags(turn: dict, args: str, rs: str) -> str:
    """Single-string flag column.

    M = mutation, H = harness fire, E = error/non-zero exit,
    G = gate_blocked, ! = outside-cwd or absolute-system path,
    S = sandbox-denial keyword in result (port of Codex
        is_likely_sandbox_denied; see _is_sandbox_denial).
    """
    out = []
    tool = turn.get("tool_name") or ""
    if tool in _WRITE_TOOLS:
        out.append("M")
    if _HARNESS_RE.search(rs):
        out.append("H")
    rl = rs.lower()
    if (
        "[exit code: 1" in rl
        or "[exit code: 2" in rl
        or "[exit code: 127" in rl
        or "traceback" in rl
        or "error:" in rl
        or "exception" in rl
    ):
        out.append("E")
    if turn.get("gate_blocked"):
        out.append("G")
    if _OUTSIDE_CWD_RE.search(args + "\n" + rs):
        out.append("!")
    if _is_sandbox_denial(rl):
        out.append("S")
    return "".join(out) or " "


@dataclass
class Turn:
    n: int
    tool: str
    args: str
    rs: str
    reasoning: str
    flags: str
    pt: int = 0
    ct: int = 0


def _load(trace_path: Path) -> list[Turn]:
    rows: list[Turn] = []
    with trace_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("event") != "tool_call":
                continue
            args = str(d.get("args_summary", ""))
            rs = str(d.get("result_summary", ""))
            rows.append(Turn(
                n=int(d.get("turn_number", 0)),
                tool=d.get("tool_name") or "?",
                args=args,
                rs=rs,
                reasoning=str(d.get("reasoning") or ""),
                flags=_flags(d, args, rs),
                pt=int(d.get("prompt_tokens", 0) or 0),
                ct=int(d.get("completion_tokens", 0) or 0),
            ))
    return rows


def _harness_text(rs: str) -> str:
    m = _HARNESS_RE.search(rs)
    return m.group(0) if m else ""


def _collapse_repeated_harness(rows: list[Turn]) -> list[tuple[str, list[int]]]:
    """Group consecutive turns with identical harness text.

    Returns a list of (mode, turn_indices). mode is 'single' (one turn)
    or 'collapse' (range of >=2 turns sharing the same harness text).
    """
    out: list[tuple[str, list[int]]] = []
    i = 0
    while i < len(rows):
        r = rows[i]
        ht = _harness_text(r.rs)
        if ht and "H" in r.flags:
            j = i + 1
            while j < len(rows) and "H" in rows[j].flags and _harness_text(rows[j].rs) == ht:
                j += 1
            if j - i >= 2:
                out.append(("collapse", list(range(i, j))))
                i = j
                continue
        out.append(("single", [i]))
        i += 1
    return out


def _render_one_line(t: Turn, head_chars: int = 60) -> str:
    action = f"{t.tool}({_key_arg(t.tool, t.args)})"
    head = _truncate(_flat(t.rs), head_chars)
    return f"T{t.n:>3} {t.flags:<3} {_truncate(action, 60):<60}  | {head}"


def _render_expanded(t: Turn, with_reasoning: bool, tail_threshold: int) -> list[str]:
    """Multi-line render for a flagged turn."""
    out: list[str] = []
    if with_reasoning and t.reasoning:
        out.append(f"T{t.n:>3} {t.flags:<3} reasoning: {_truncate(t.reasoning.strip(), 160)}")
        out.append(f"      action:    {_truncate(t.tool + '(' + _key_arg(t.tool, t.args) + ')', 140)}")
    else:
        out.append(f"T{t.n:>3} {t.flags:<3} {_truncate(t.tool + '(' + _key_arg(t.tool, t.args) + ')', 140)}")
    body = _flat(t.rs)
    if len(body) > tail_threshold:
        head = _truncate(body[: tail_threshold // 2], tail_threshold // 2)
        tail = _truncate(body[-(tail_threshold // 2):], tail_threshold // 2)
        out.append(f"      head:      {head}")
        out.append(f"      tail:      {tail}")
    else:
        out.append(f"      result:    {_truncate(body, 200)}")
    return out


def _render_collapsed_harness(rows: list[Turn], idxs: list[int]) -> str:
    first = rows[idxs[0]]
    last = rows[idxs[-1]]
    ht = _harness_text(first.rs)
    span = f"T{first.n}-T{last.n}"
    n = len(idxs)
    return f"{span:>9} {first.flags:<3} {n}× {first.tool} → {_truncate(_flat(ht), 100)}"


@dataclass
class DigestOptions:
    reasoning: bool = False
    expand_flags: tuple[str, ...] = ()
    tail_threshold: int = 1000
    collapse_harness: bool = False
    head_chars: int = 60
    turn_lo: int | None = None
    turn_hi: int | None = None


def render_digest(rows: list[Turn], opts: DigestOptions) -> str:
    if opts.turn_lo is not None or opts.turn_hi is not None:
        lo = opts.turn_lo if opts.turn_lo is not None else 0
        hi = opts.turn_hi if opts.turn_hi is not None else 10**9
        rows = [r for r in rows if lo <= r.n <= hi]

    out: list[str] = []
    groups = _collapse_repeated_harness(rows) if opts.collapse_harness else [("single", [i]) for i in range(len(rows))]

    for mode, idxs in groups:
        if mode == "collapse":
            out.append(_render_collapsed_harness(rows, idxs))
            continue
        t = rows[idxs[0]]
        # Decide whether to expand this turn.
        should_expand = bool(opts.expand_flags) and any(c in t.flags for c in opts.expand_flags)
        if should_expand:
            out.extend(_render_expanded(t, opts.reasoning, opts.tail_threshold))
        else:
            line = _render_one_line(t, head_chars=opts.head_chars)
            if opts.reasoning and t.reasoning and any(c in t.flags for c in ("M", "H", "E", "G", "!", "S")):
                # prepend reasoning line for flagged turns when --reasoning is on
                out.append(f"T{t.n:>3} {t.flags:<3} reasoning: {_truncate(t.reasoning.strip(), 140)}")
                out.append(line)
            else:
                out.append(line)
    return "\n".join(out)


# ─── Compaction helpers (content-blind; consumed by harness/loop.py) ────

def render_digest_for_compaction(rows: list[Turn], keep_recent: int) -> str:
    """Digest of all-but-last-N turns for context compaction at fill threshold.

    Wraps render_digest() with the canonical compaction options
    (collapse harness fires, no per-turn reasoning, no tail expansion).
    The caller is responsible for emitting the last `keep_recent` turns
    verbatim outside this output. Returns "" when there are <= keep_recent
    turns (nothing to compact yet).
    """
    older = rows[:-keep_recent] if keep_recent > 0 else rows
    if not older:
        return ""
    opts = DigestOptions(
        reasoning=False,
        expand_flags=(),
        tail_threshold=10**9,  # disable tail expansion
        collapse_harness=True,
        head_chars=60,
    )
    return render_digest(older, opts)


def gate_allows_compaction(
    mutation_count: int,
    recent_g_cluster: int = 0,
    recent_e_streak: int = 0,
    *,
    min_mutations: int = 1,
    max_g_cluster: int = 3,
    max_e_streak: int = 5,
) -> tuple[bool, str]:
    """Decide whether to compact at fill threshold.

    Returns (allow, reason). When False, the caller ends the session
    cleanly with finish_reason=context_full_uncompactable rather than
    extending a pathological session. M=0 sessions, gate cliffs, and
    error streaks all block compaction.
    """
    if mutation_count < min_mutations:
        return False, f"M={mutation_count} below gate min={min_mutations}"
    if recent_g_cluster >= max_g_cluster:
        return False, f"G-cluster {recent_g_cluster} >= {max_g_cluster}"
    if recent_e_streak >= max_e_streak:
        return False, f"E-streak {recent_e_streak} >= {max_e_streak}"
    return True, "ok"
