"""Focus / dedup helpers — extracted from loop.py."""
from __future__ import annotations

import json
import logging
import re
import shlex
from pathlib import Path

from .._shell_patterns import TEST_COMMAND_RE as _TEST_COMMAND_RE

log = logging.getLogger(__name__)


_TRAILING_PIPE_RE = re.compile(
    r"""
    \s*                          # optional leading whitespace before pipe
    (?:                          # group: one pipe segment
        \|                       # the pipe character
        \s*                      # optional whitespace after pipe
        (?:head|tail|grep|cat|sort|uniq|wc|tee|less|more)  # common filter commands
        (?:\s+[^\|]*)?)          # their arguments (up to next pipe or end)
    +                            # one or more trailing pipe segments
    $                            # anchored at end
    """,
    re.VERBOSE,
)
_STDERR_REDIRECT_RE = re.compile(r"\s*2>&1\s*")
_BASH_READ_TARGET_RE = re.compile(
    r"^\s*(cat|head|tail|less|more|file)\s+([^\s|;&<>`$()]+)\s*$"
)
_PATH_SUFFIXES = (
    ".py", ".pyi", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh",
    ".rs", ".go", ".java", ".js", ".jsx", ".ts", ".tsx",
)
_SHELL_SEPARATORS = frozenset({"&&", "||", "|", ";"})


def _dedup_signature(tc) -> tuple[str, str]:
    """Build a normalized (name, args) signature for duplicate detection.

    For bash calls, normalizes the command to strip trivial variants.
    For all other tools, uses the raw arguments as-is.
    """
    args = tc.arguments
    if tc.name == "bash" and "cmd" in args:
        normalized = dict(args)
        normalized["cmd"] = _normalize_bash_for_dedup(args["cmd"])
        return (tc.name, json.dumps(normalized, sort_keys=True))
    return (tc.name, json.dumps(args, sort_keys=True))


def _normalize_bash_for_dedup(cmd: str) -> str:
    """Normalize a bash command for duplicate detection.

    Strips trailing pipe chains (| head, | tail, | grep, etc.) and
    stderr redirects (2>&1) so the model can't evade duplicate_abort by
    appending different tail/head limits to the same command.

    The normalized form is used ONLY for dedup comparison.  The actual
    command executes unmodified.
    """
    # Strip 2>&1 first (can appear before or after pipes)
    cmd = _STDERR_REDIRECT_RE.sub(" ", cmd).strip()
    # Strip trailing pipe chains
    cmd = _TRAILING_PIPE_RE.sub("", cmd).strip()
    return cmd


def _focus_signature(tc, args_summary: str, cwd: str) -> tuple[str, str]:
    """Content-blind focus target used by rumination guardrails.

    Goal: detect repeated inspection of the same file or same normalized bash
    command without waiting for the coarse duplicate-abort threshold.
    """
    if tc.name in {"read", "write", "edit"}:
        path = tc.arguments.get("path") or tc.arguments.get("file_path")
        if isinstance(path, str) and path:
            return _encode_focus_path(path, cwd)
    # Give run_tests and patch dialects a stable focus key so
    # mutation_repeat_guard and contract_gate
    # rumination_ladder see test-tool repeats with the right
    # granularity. These once fell into the generic JSON fallback
    # at line 90, which keyed off whatever args dict shape happened.
    if tc.name == "run_tests":
        target = (
            tc.arguments.get("target")
            or tc.arguments.get("path")
            or tc.arguments.get("file_path")
            or ""
        )
        if isinstance(target, str) and target:
            return f"run_tests:{target}", f"run_tests({target})"
        return "run_tests:<all>", "run_tests(<all>)"
    if tc.name in {"apply_patch", "udiff"}:
        # Use a digest of the patch text as the focus key so the same
        # patch issued twice is recognised as identical without leaking
        # the full text through the focus_display.
        patch_text = tc.arguments.get("patch", "")
        if isinstance(patch_text, str) and patch_text:
            import hashlib as _hashlib
            digest = _hashlib.sha1(patch_text.encode("utf-8", errors="ignore")).hexdigest()[:12]
            return f"{tc.name}:{digest}", f"{tc.name}(<patch>)"
        return f"{tc.name}:<empty>", f"{tc.name}(<patch>)"
    if tc.name == "bash":
        cmd = tc.arguments.get("cmd", "")
        if isinstance(cmd, str) and cmd.strip():
            normalized = _normalize_bash_for_dedup(cmd)
            if not normalized:
                return "", ""
            focus = _extract_bash_focus_target(normalized, cwd)
            if focus is not None:
                return focus
            return f"bash:{normalized}", _truncate_focus_display(normalized)
    if tc.arguments:
        raw = json.dumps(tc.arguments, sort_keys=True)
        return f"{tc.name}:{raw}", f"{tc.name}({args_summary})"
    return "", ""


def _canon_focus_path(path: str) -> str:
    if not path:
        return ""
    if path.startswith("/"):
        return path.rstrip("/") or "/"
    stripped = path.lstrip("./").rstrip("/")
    return stripped or "."


def _truncate_focus_display(text: str, max_chars: int = 96) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _encode_focus_path(path: str, cwd: str) -> tuple[str, str]:
    canon = _canon_focus_path(path)
    if path.startswith("/") and not _path_within_cwd(path, cwd):
        return f"outside:{canon}", path
    return f"file:{canon}", path


def _encode_focus_target(key_base: str, display: str, *, root_path: str, cwd: str) -> tuple[str, str]:
    if root_path.startswith("/") and not _path_within_cwd(root_path, cwd):
        return f"outside:{key_base}", display
    return f"bash:{key_base}", display


def _path_within_cwd(path: str, cwd: str) -> bool:
    try:
        path_res = Path(path).resolve()
        cwd_res = Path(cwd).resolve()
        path_res.relative_to(cwd_res)
        return True
    except (ValueError, OSError):
        return False


def _split_bash_segments(cmd: str) -> list[list[str]]:
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        return []
    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in _SHELL_SEPARATORS:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    return segments


def _looks_like_path_token(token: str) -> bool:
    if not token or token.startswith("-"):
        return False
    base = token.split("::", 1)[0].rstrip(",")
    if base in {".", "..", "tests", "test"}:
        return True
    if "/" in base or base.startswith("/"):
        return True
    if base.endswith("/"):
        return True
    if base.lower().startswith("test"):
        return True
    return base.endswith(_PATH_SUFFIXES)


def _extract_test_target_from_command(cmd: str) -> str:
    if not _TEST_COMMAND_RE.search(cmd or ""):
        return ""
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        return ""
    for token in tokens:
        candidate = token.split("::", 1)[0].rstrip(",")
        if _looks_like_path_token(candidate) and "test" in candidate.lower():
            return candidate
    return ""


def _extract_bash_focus_target(cmd: str, cwd: str) -> tuple[str, str] | None:
    m = _BASH_READ_TARGET_RE.match(cmd)
    if m:
        return _encode_focus_path(m.group(2), cwd)

    test_target = _extract_test_target_from_command(cmd)
    if test_target:
        return _encode_focus_path(test_target, cwd)

    segments = _split_bash_segments(cmd)
    for segment in reversed(segments):
        if not segment:
            continue
        name = segment[0]
        rest = segment[1:]
        if name in {"ls", "tree", "du"}:
            for token in reversed(rest):
                if _looks_like_path_token(token):
                    return _encode_focus_path(token, cwd)
            return _encode_focus_path(".", cwd)
        if name == "find":
            # Scan EVERY token before deciding. Prior version broke out
            # of the loop on the first non-flag positional (the root),
            # so a `find <root> -name <pattern>` invocation lost the
            # pattern. Caused
            # test_focus_signature_extracts_outside_cwd_find_target to
            # fail. Fix: walk the whole arg list, recording the first
            # positional as `root` and each `-name`/`-iname`/`-path`/
            # `-wholename` value as the pattern. Stripped surrounding
            # quotes (find tokens often arrive as `"x.py"` from the
            # model's quoting).
            root = "."
            name_pattern = ""   # -name / -iname value (more specific, preferred)
            path_pattern = ""   # -path / -wholename value (used only if no -name)
            root_set = False
            i = 0
            while i < len(rest):
                token = rest[i]
                if token in {"-name", "-iname"} and i + 1 < len(rest):
                    name_pattern = rest[i + 1].strip("'\"")
                    i += 2
                    continue
                if token in {"-path", "-wholename"} and i + 1 < len(rest):
                    path_pattern = rest[i + 1].strip("'\"")
                    i += 2
                    continue
                if token in {"2>/dev/null", "1>/dev/null"}:
                    i += 1
                    continue
                if not token.startswith("-") and not root_set:
                    root = token
                    root_set = True
                i += 1
            pattern = name_pattern or path_pattern
            if pattern:
                display = f"{pattern} under {root}"
                key_base = f"{_canon_focus_path(root)}::{pattern}"
                return _encode_focus_target(key_base, display, root_path=root, cwd=cwd)
            return _encode_focus_path(root, cwd)
        if name in {"grep", "rg", "fd"}:
            for token in reversed(rest):
                if _looks_like_path_token(token):
                    return _encode_focus_path(token, cwd)
    return None

# Error-taxonomy constants (NORMAL_LIFECYCLE, MODEL_STUCK, _TRANSIENT_ERRORS)
# stay in loop.py since they depend on openai and are part of the harness
# public surface.
