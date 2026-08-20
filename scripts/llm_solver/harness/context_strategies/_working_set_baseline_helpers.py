"""Pure-helper functions extracted from _working_set_baseline.py."""
from __future__ import annotations

import json
import re
import shlex

from .._shell_patterns import TEST_COMMAND_RE as _TEST_COMMAND_RE
from ..._shared.classification import classify_outcome as _classify_outcome


_PATH_KEYS = ("path", "file_path")
_BASH_READ_RE = re.compile(
    r"^\s*(cat|head|tail|less|more|file)\s+([^\s|;&<>`$()]+)\s*$"
)
_ACTION_PATH_RE = re.compile(
    r"(?:path|file_path)='([^']+)'|"
    r"(?:path|file_path)=\"([^\"]+)\"|"
    r"\"(?:path|file_path)\"\s*:\s*\"([^\"]+)\""
)
_ACTION_CMD_RE = re.compile(
    r"cmd='([^']+)'|cmd=\"([^\"]+)\"|\"cmd\"\s*:\s*\"([^\"]+)\""
)
_INSPECT_CMD_PREFIXES = ("ls", "find", "grep", "rg", "fd", "tree", "cat", "head", "tail")
_PATH_SUFFIXES = (
    ".py", ".pyi", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh",
    ".rs", ".go", ".java", ".js", ".jsx", ".ts", ".tsx",
)

def _pick_path(args: dict) -> str:
    for key in _PATH_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _path_from_read_cmd(cmd: str) -> str:
    if not isinstance(cmd, str):
        return ""
    match = _BASH_READ_RE.match(cmd)
    if not match:
        return ""
    return match.group(2)


def _cmd_display(cmd_sig: str, fallback: str) -> str:
    if not cmd_sig:
        return fallback
    try:
        blob = json.loads(cmd_sig)
        cmd = blob.get("cmd") if isinstance(blob, dict) else None
        if isinstance(cmd, str) and cmd:
            return cmd
    except (ValueError, TypeError):
        pass
    return fallback or cmd_sig


def _cmd_text(cmd_sig: str) -> str:
    try:
        blob = json.loads(cmd_sig)
        if isinstance(blob, dict):
            cmd = blob.get("cmd")
            if isinstance(cmd, str):
                return cmd
    except (ValueError, TypeError):
        pass
    return ""


def _looks_like_test_path(path: str) -> bool:
    return bool(path) and "test" in path.lower()


def _extract_action_target(action: str) -> str:
    match = _ACTION_PATH_RE.search(action)
    if match:
        return next((group for group in match.groups() if group), "")
    cmd_match = _ACTION_CMD_RE.search(action)
    if not cmd_match:
        return ""
    cmd = next((group for group in cmd_match.groups() if group), "")
    return _extract_focus_target_from_command(cmd)


def _extract_test_target_from_action(action: str) -> str:
    match = _ACTION_PATH_RE.search(action)
    if match:
        path = next((group for group in match.groups() if group), "")
        return path if _looks_like_test_path(path) else ""
    cmd_match = _ACTION_CMD_RE.search(action)
    if not cmd_match:
        return ""
    cmd = next((group for group in cmd_match.groups() if group), "")
    return _extract_test_target_from_command(cmd)


def _extract_test_target_from_command(cmd: str) -> str:
    if not _TEST_COMMAND_RE.search(cmd or ""):
        return ""
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        return ""
    for token in tokens:
        candidate = token.split("::", 1)[0].rstrip(",")
        if _looks_like_test_path(candidate) and _looks_like_path_token(candidate):
            return candidate
    return ""


def _extract_focus_target_from_command(cmd: str) -> str:
    test_target = _extract_test_target_from_command(cmd)
    if test_target:
        return test_target
    read_target = _path_from_read_cmd(cmd)
    if read_target:
        return read_target
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        return ""
    if not tokens:
        return ""
    name = tokens[0]
    rest = tokens[1:]
    if name in {"ls", "tree", "du"}:
        for token in reversed(rest):
            if _looks_like_path_token(token):
                return token
        return "."
    if name == "find":
        root = "."
        pattern = ""
        for i, token in enumerate(rest):
            if token in {"-name", "-iname", "-path", "-wholename"} and i + 1 < len(rest):
                pattern = rest[i + 1]
            if token.startswith("-"):
                continue
            root = token
            break
        if pattern:
            return f"{pattern} under {root}"
        return root
    if name in {"grep", "rg", "fd"}:
        for token in reversed(rest):
            if _looks_like_path_token(token):
                return token
    return ""


def _looks_like_path_token(token: str) -> bool:
    if not token or token.startswith("-"):
        return False
    candidate = token.split("::", 1)[0].rstrip(",")
    if candidate in {".", "..", "tests", "test"}:
        return True
    if "/" in candidate or candidate.startswith("/") or candidate.endswith("/"):
        return True
    if candidate.lower().startswith("test"):
        return True
    return candidate.endswith(_PATH_SUFFIXES)


def _is_inspection_action(action: str) -> bool:
    if action.startswith(("read(", "glob(", "grep(")):
        return True
    cmd_match = _ACTION_CMD_RE.search(action)
    if not cmd_match:
        return False
    cmd = next((group for group in cmd_match.groups() if group), "")
    return any(cmd.startswith(prefix) for prefix in _INSPECT_CMD_PREFIXES)


def _clean_reasoning(reasoning: str, limit: int) -> str:
    short = (reasoning or "").replace("\n", " ").strip()
    if len(short) > limit:
        return short[: limit - 3] + "..."
    return short


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 24:
        return text[:max_chars]
    return text[: max_chars - 20] + f"\n[... +{len(text) - max_chars + 20} chars]"


def _fit_lines(lines: list[str], max_chars: int, max_lines: int | None = None) -> str:
    if not lines or max_chars <= 0:
        return ""
    chosen = lines[-max_lines:] if max_lines else lines
    kept_rev: list[str] = []
    used = 0
    for line in reversed(chosen):
        add = len(line) + (1 if kept_rev else 0)
        if used + add > max_chars and kept_rev:
            break
        kept_rev.append(line)
        used += add
    return "\n".join(reversed(kept_rev))


def _fit_blocks(blocks: list[str], max_chars: int, max_entries: int | None = None) -> str:
    if not blocks or max_chars <= 0:
        return ""
    chosen = blocks[-max_entries:] if max_entries else blocks
    kept_rev: list[str] = []
    used = 0
    for block in reversed(chosen):
        add = len(block) + (2 if kept_rev else 0)
        if used + add > max_chars and kept_rev:
            break
        kept_rev.append(block)
        used += add
    return "\n".join(reversed(kept_rev))
