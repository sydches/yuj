"""Shared shell-surface regexes used by multiple harness modules.

``TEST_COMMAND_RE`` — "is this bash call a test/verification command?" —
is derived from the union of every registered runner's
``verification_patterns`` (see ``language_quirks.all_verification_patterns``),
so it covers pytest / go / cargo / jest / ctest without inheriting broad
analysis-only patterns from ``generic.toml``. ``CHECK_COMMAND_RE`` also
recognizes descriptor-owned custom probes for context tagging. It must not
be used to satisfy formal verification. Missing data never invents runners.
"""
import ast
import re
import shlex

from ..language_quirks import all_custom_check_patterns, all_verification_patterns
from .command_redirect import split_shell_fragments, strip_leading_assignments


def _build_test_command_re() -> "re.Pattern[str]":
    pats = all_verification_patterns()
    joined = "|".join(f"(?:{p})" for p in pats) or r"(?!)"
    return re.compile(joined, re.IGNORECASE | re.VERBOSE)


TEST_COMMAND_RE = _build_test_command_re()
CHECK_COMMAND_RE = re.compile(
    "|".join(f"(?:{p})" for p in (TEST_COMMAND_RE.pattern, *all_custom_check_patterns())),
    re.IGNORECASE | re.VERBOSE,
)


def command_from_summary(summary: str) -> str:
    """Read a recorded cmd keyword without executing or guessing its contents."""
    if not summary.lstrip().startswith("cmd="):
        return summary
    try:
        call = ast.parse(f"record({summary})", mode="eval").body
        value = next(keyword.value for keyword in call.keywords if keyword.arg == "cmd")
        command = ast.literal_eval(value)
        return command if isinstance(command, str) else ""
    except (SyntaxError, ValueError, StopIteration, AttributeError):
        return ""


def matches_command(command: str, pattern=TEST_COMMAND_RE, *, allow_partial: bool = False) -> bool:
    """Recognize top-level invocations, excluding quoted mentions and reads.

    Shell groups, substitutions and commands after heredocs may remain unknown.
    This recognizer never executes shell text to resolve those cases.
    """
    for fragment in split_shell_fragments(command):
        text = strip_leading_assignments(_without_comment(fragment.text))
        # Context can retain a known command prefix from a clipped trace.
        # Formal verification requires the complete shell words below.
        if allow_partial and _has_heredoc(text) and _prefix_match(pattern, text):
            return True
        try:
            argv = shlex.split(text)
        except ValueError:
            if allow_partial and _prefix_match(pattern, text):
                return True
            continue
        if argv and argv[0].rsplit("/", 1)[-1] == "env":
            argv = argv[1:]
            while argv and (argv[0] in {"-i", "--ignore-environment"}
                            or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[0])):
                argv = argv[1:]
            if argv and argv[0] == "--":
                argv = argv[1:]
        if argv:
            argv[0] = argv[0].rsplit("/", 1)[-1]
            normalized = shlex.join(argv)
            if _prefix_match(pattern, normalized):
                return True
        # Do not interpret heredoc data as subsequent shell commands.
        if _has_heredoc(text):
            return False
    return False


def _without_comment(text: str) -> str:
    """A shell comment starts at a word boundary, never inside a word."""
    quote, escaped, word_start = '', False, True
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == '\\' and quote != "'":
            escaped = True
        elif quote:
            if char == quote:
                quote = ''
            continue
        elif char in "'\"`":
            quote = char
        elif char == '#' and word_start:
            return text[:index]
        word_start = char.isspace() or char in ';&|()'
    return text


def _prefix_match(pattern, text: str) -> bool:
    match = pattern.match(text)
    return bool(match and (match.end() == len(text)
                           or text[match.end()].isspace()
                           or text[match.end() - 1].isspace()))


def _has_heredoc(text: str) -> bool:
    quote = ""
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
        elif char == "\\" and quote != "'":
            escaped = True
        elif quote:
            if char == quote:
                quote = ""
        elif char in "'\"`":
            quote = char
        elif text.startswith("<<", index):
            return True
    return False
