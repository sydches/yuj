"""Attribute a shell result to supported check invocations, without output hints.

Recognition labels invocations, not coverage or task correctness. Only a direct
check or a successful conjunction of checks and directory changes has a result
we can attribute here. Other shell structures retain an unresolved verdict.
"""
import functools
import re
import shlex

from ._shell_patterns import CHECK_COMMAND_RE, matches_command
from .command_redirect import split_shell_fragments, strip_leading_assignments


@functools.lru_cache(maxsize=1)
def _non_execution_modes():
    from ..language_quirks import FORMATS_DIR, _load_runner_quirk_dict
    rules = []
    for path in sorted(FORMATS_DIR.glob("*.toml")):
        spec = _load_runner_quirk_dict(path.stem)
        flags = spec.get("non_execution_flags", ())
        check_flags = spec.get("non_test_check_flags", ())
        patterns = spec.get("verification_patterns", ())
        if (flags or check_flags) and patterns:
            rules.append((re.compile("|".join(f"(?:{pattern})" for pattern in patterns),
                                     re.IGNORECASE | re.VERBOSE), frozenset(flags), frozenset(check_flags)))
    return tuple(rules)


def _literal_simple_command(text):
    quote = ""
    escaped = False
    for char in text:
        if escaped:
            escaped = False
        elif char == "\\" and quote != "'":
            escaped = True
        elif quote:
            if char == quote:
                quote = ""
            elif quote == '"' and char in "$`":
                return False
        elif char == "#":
            # Comments and their newlines need a shell parser to distinguish
            # comments from word suffixes; do not infer execution from them.
            return False
        elif char in "'\"":
            quote = char
        elif char in "$`(){}<>;|&\n*?[~":
            return False
    return not quote and not escaped


def shell_verification_status(command, exit_code, timed_out=False):
    """Return tool-owned status for the exact command submitted to the shell."""
    fragments = split_shell_fragments(command)
    if not fragments:
        return "not_a_check"
    kinds = []
    for fragment in fragments:
        if fragment.operator_after not in ("", "&&") or fragment.operator_before not in ("", "&&"):
            return "shell_unresolved"
        if not _literal_simple_command(fragment.text):
            return "shell_unresolved"
        text = strip_leading_assignments(fragment.text)
        try:
            argv = shlex.split(text)
        except ValueError:
            return "shell_unresolved"
        if matches_command(text):
            mode_flags = {arg.split("=", 1)[0] for arg in argv[1:]}
            modes = [(flags, check_flags) for pattern, flags, check_flags in _non_execution_modes()
                     if matches_command(text, pattern)]
            if any(mode_flags & flags for flags, _ in modes):
                return "not_a_check"
            kinds.append("custom" if any(mode_flags & flags for _, flags in modes) else "formal")
        elif matches_command(text, CHECK_COMMAND_RE):
            kinds.append("custom")
        elif len(fragments) > 1 and argv and argv[0] == "cd":
            kinds.append("directory_change")
        else:
            return "not_a_check" if len(fragments) == 1 else "shell_unresolved"
    if not any(kind in {"formal", "custom"} for kind in kinds):
        return "not_a_check"
    if timed_out:
        return "timed_out"
    if exit_code is None:
        return "error"
    if exit_code != 0:
        # For a conjunction, failure might belong to cd or an earlier check.
        if len(fragments) > 1:
            return "shell_unresolved"
        if len(fragments) == 1 and exit_code in {126, 127}:
            return "runner_unavailable"
        return "failed" if "formal" in kinds else "custom_failed"
    return "passed" if "formal" in kinds else "custom_passed"
