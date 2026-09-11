"""Recognize shell commands that read one explicit file."""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass


@dataclass(frozen=True)
class ShellRead:
    """A shell command proven to read one explicit file."""

    verb: str
    path: str


_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", re.DOTALL)
_GLOB_CHARS = frozenset("*?[]{}")
_READ_VERBS = {"cat", "head", "tail", "sed", "grep", "egrep", "fgrep", "rg"}


def _has_unquoted_shell_control(command: str) -> bool:
    """Reject compounds, substitutions, redirections, and backgrounding."""
    quote = ""
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
        elif char == "\\" and quote != "'":
            escaped = True
        elif quote:
            if char == quote:
                quote = ""
        elif char in "'\"":
            quote = char
        elif char in ";&|<>\n`":
            return True
        elif char == "$" and index + 1 < len(command) and command[index + 1] == "(":
            return True
        index += 1
    return bool(quote or escaped)


def _plain_file_token(token: str) -> bool:
    return bool(
        token
        and token not in {"-", ".", ".."}
        and not token.endswith("/")
        and not any(char in token for char in _GLOB_CHARS)
    )


def _single_cat_path(args: list[str]) -> str | None:
    paths = []
    options_done = False
    for token in args:
        if token == "--" and not options_done:
            options_done = True
        elif not options_done and token.startswith("-"):
            if not re.fullmatch(r"-[AbEnsTuv]+", token):
                return None
        else:
            paths.append(token)
    return paths[0] if len(paths) == 1 and _plain_file_token(paths[0]) else None


def _single_head_tail_path(verb: str, args: list[str]) -> str | None:
    paths: list[str] = []
    index = 0
    options_done = False
    value_options = {"-n", "--lines", "-c", "--bytes"}
    if verb == "tail":
        value_options |= {
            "--pid", "-s", "--sleep-interval", "--max-unchanged-stats",
        }
    while index < len(args):
        token = args[index]
        if token == "--" and not options_done:
            options_done = True
        elif not options_done and token in value_options:
            index += 1
            if index >= len(args):
                return None
        elif not options_done and (
            token.startswith("--lines=")
            or token.startswith("--bytes=")
            or token.startswith("--pid=")
            or token.startswith("--sleep-interval=")
            or re.fullmatch(r"-[nc][+-]?\d+", token)
            or re.fullmatch(r"[+-]\d+", token)
        ):
            pass
        elif not options_done and token.startswith("-"):
            # Formatting/follow flags do not add another file operand.
            if not re.fullmatch(r"-(?:q|v|f|F|r|z)+", token):
                return None
        else:
            paths.append(token)
        index += 1
    return paths[0] if len(paths) == 1 and _plain_file_token(paths[0]) else None


def _single_sed_path(args: list[str]) -> str | None:
    quiet = False
    scripts = 0
    operands: list[str] = []
    index = 0
    options_done = False
    while index < len(args):
        token = args[index]
        if token == "--" and not options_done:
            options_done = True
        elif not options_done and token in {"-n", "--quiet", "--silent"}:
            quiet = True
        elif not options_done and token in {"-e", "--expression"}:
            index += 1
            if index >= len(args):
                return None
            scripts += 1
        elif not options_done and (
            token.startswith("-e") and token != "-e"
            or token.startswith("--expression=")
        ):
            scripts += 1
        elif not options_done and (
            token == "-i" or token.startswith("-i")
            or token == "--in-place" or token.startswith("--in-place=")
            or token in {"-f", "--file"} or token.startswith("--file=")
        ):
            return None
        elif not options_done and token.startswith("-"):
            return None
        else:
            operands.append(token)
        index += 1
    if not quiet:
        return None
    if scripts == 0:
        if not operands:
            return None
        operands.pop(0)  # the positional sed program
    return operands[0] if len(operands) == 1 and _plain_file_token(operands[0]) else None


_GREP_AGGREGATE = {
    "-c", "--count", "--count-matches", "-l", "-L",
    "--files-with-matches", "--files-without-match", "-q", "--quiet",
    "--stats", "--json",
}
_GREP_RECURSIVE = {"-r", "-R", "--recursive", "--files"}
_GREP_VALUE_OPTIONS = {
    "-e", "--regexp", "-m", "--max-count", "-A", "--after-context",
    "-B", "--before-context", "-C", "--context", "-g", "--glob",
    "-t", "--type", "-T", "--type-not", "--color", "--encoding",
}
_GREP_FLAG_OPTIONS = {
    "--line-number", "--with-filename", "--no-filename", "--ignore-case",
    "--invert-match", "--word-regexp", "--line-regexp", "--fixed-strings",
    "--extended-regexp", "--perl-regexp", "--text", "--binary-files=text",
    "--no-messages", "--only-matching", "--hidden", "--no-heading",
    "--smart-case", "--multiline", "--case-sensitive",
}


def _single_grep_path(args: list[str]) -> str | None:
    operands: list[str] = []
    pattern_supplied = False
    index = 0
    options_done = False
    while index < len(args):
        token = args[index]
        if token == "--" and not options_done:
            options_done = True
        elif not options_done and token in _GREP_AGGREGATE | _GREP_RECURSIVE:
            return None
        elif not options_done and token in {"-f", "--file"}:
            return None  # a pattern file would be a second file read
        elif not options_done and token in _GREP_VALUE_OPTIONS:
            index += 1
            if index >= len(args):
                return None
            if token in {"-e", "--regexp"}:
                pattern_supplied = True
        elif not options_done and token in _GREP_FLAG_OPTIONS:
            pass
        elif not options_done and any(
            token.startswith(prefix)
            for prefix in (
                "--regexp=", "--max-count=", "--after-context=",
                "--before-context=", "--context=", "--glob=", "--type=",
                "--type-not=", "--color=", "--encoding=",
            )
        ):
            if token.startswith("--regexp="):
                pattern_supplied = True
        elif not options_done and re.fullmatch(r"-(?:m|A|B|C)\d+", token):
            pass
        elif not options_done and token.startswith("-"):
            # Common presentation/matching flags. Unknown option shapes fail
            # closed rather than accidentally treating an option value as path.
            if not re.fullmatch(r"-[nHhivwxyFoUaIsSP]+", token):
                return None
        else:
            operands.append(token)
        index += 1
    if not pattern_supplied:
        if not operands:
            return None
        operands.pop(0)
    return operands[0] if len(operands) == 1 and _plain_file_token(operands[0]) else None


def classify_single_file_read(command: str) -> ShellRead | None:
    """Classify a non-compound shell command that reveals one file's text."""
    if not command.strip() or _has_unquoted_shell_control(command):
        return None
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return None
    while tokens and _ASSIGNMENT_RE.fullmatch(tokens[0]):
        tokens.pop(0)
    if not tokens:
        return None
    verb = tokens.pop(0).rsplit("/", 1)[-1]
    if verb not in _READ_VERBS:
        return None
    if verb == "cat":
        path = _single_cat_path(tokens)
    elif verb in {"head", "tail"}:
        path = _single_head_tail_path(verb, tokens)
    elif verb == "sed":
        path = _single_sed_path(tokens)
    else:
        path = _single_grep_path(tokens)
    return ShellRead(verb=verb, path=path) if path is not None else None
