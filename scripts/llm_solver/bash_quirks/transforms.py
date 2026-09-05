"""Bash command rewriting and output filtering — public surface.

The four concerns live in sibling modules:

  - ``_rewrites``   — universal quiet flags (rewrites.toml)
  - ``_output``     — task-format output control + parser + condensation
  - ``_redactions`` — secret masking (redactions.toml)
  - ``_forbidden``  — hard-refusal patterns (forbidden.toml)

This file ties them together with the ``rewrite_command`` orchestrator
and re-exports every public symbol so existing callers
(``from bash_quirks.transforms import …``) keep working.
"""
from __future__ import annotations

import re
import shlex

# ── Public re-exports ────────────────────────────────────────────────
from ._forbidden import ForbiddenRule, apply_forbidden, load_forbidden_rules
from ._output import (
    OutputControl,
    OutputParser,
    _find_command_span,
    _is_test_command,
    _normalize_verdict,
    _SUMMARY_TAIL_CHARS,
    _VERDICT_NORMALIZE,
    condense_output,
    load_output_control,
    load_output_parser,
    parse_structured,
    render_digest,
)
from ._redactions import RedactionRule, apply_redactions, load_redactions
from ._rewrites import (
    RewriteRule,
    load_universal_rewrites,
    strip_display_only_test_pipeline,
)


_SHELL_WORD = r'''(?:[^\s'"\\]|\\.|"(?:\\.|[^"\\])*"|'[^']*')+'''


def _quiet_fragment(fragment: str, oc: OutputControl) -> str:
    """Remove declared display flags without re-quoting shell arguments."""
    words = list(re.finditer(_SHELL_WORD, fragment))
    short = {flag[1:] for flag in oc.quiet_flags if len(flag) == 2}
    changes = []
    skip_value = False
    for index, word in enumerate(words):
        raw = word.group()
        if skip_value:
            skip_value = False
            continue
        # Shell quoting/expansion is not a display flag. Leave its bytes alone.
        if any(char in raw for char in "\"'\\$`"):
            continue
        if raw == "--":
            break
        if raw in oc.argument_flags:
            skip_value = True
            continue
        replacement = raw
        if raw in oc.quiet_flags or (oc.quiet_flags and raw in oc.failure_only_flag.split()):
            replacement = ""
        elif raw.startswith("--"):
            for flag in oc.quiet_flags:
                if "=" not in flag:
                    continue
                name, value = flag.split("=", 1)
                if raw == name and index + 1 < len(words):
                    next_word = words[index + 1]
                    if shlex.split(next_word.group()) == [value]:
                        changes.append((next_word.start(), next_word.end(), ""))
                        replacement = ""
                        skip_value = True
        elif raw.startswith("-"):
            wildcard = next((flag[:-1] for flag in oc.quiet_flags
                             if flag.endswith("*") and raw.startswith(flag[:-1])), None)
            if wildcard:
                replacement = ""
                if raw == wildcard and index + 1 < len(words):
                    next_word = words[index + 1]
                    if not next_word.group().startswith("-"):
                        changes.append((next_word.start(), next_word.end(), ""))
                        skip_value = True
            # Keep bundled control flags such as -x; never split value options.
            elif raw[1:] and set(raw[1:]) <= short | set(oc.preserve_short_flags):
                remaining = "".join(c for c in raw[1:] if c not in short)
                replacement = "-" + remaining if remaining else ""
        if replacement != raw:
            changes.append((word.start(), word.end(), replacement))
    for start, end, replacement in sorted(changes, reverse=True):
        if not replacement and start > 0 and fragment[start - 1].isspace():
            start -= 1
        fragment = fragment[:start] + replacement + fragment[end:]
    return fragment.rstrip()


def rewrite_command(cmd: str, oc: OutputControl | None,
                    universal_rewrites: list[RewriteRule] | None = None,
                    forbidden_rules: list[ForbiddenRule] | None = None,
                    rule_log: list | None = None,
                    transform_log: list | None = None) -> str:
    """Rewrite a bash command to reduce output volume — or refuse it.

    Order of operations:
      1. Forbidden rules (hard refusal — replaces cmd with `false  #
         [HARNESS: ...]` if any pattern matches).
      2. Universal rewrites (pip -q, npm --loglevel=error, etc.).
      3. Task-format-specific flags (--tb=short for pytest).

    Forbidden runs first because once we refuse, the rewrites and
    test-flag transforms are moot.
    """
    cmd = apply_forbidden(
        cmd,
        forbidden_rules,
        rule_log=rule_log,
        transform_log=transform_log,
    )
    if "false  # [HARNESS:" in cmd:
        return cmd

    # Append flags only to single-line commands. A flag after a heredoc
    # terminator or later line would change the command. A ``make`` rule
    # may also match plain text inside a heredoc.
    if "\n" in cmd or "<<" in cmd:
        return cmd

    # Universal rewrites — always apply.
    if universal_rewrites:
        for rule in universal_rewrites:
            span = _find_command_span(cmd, (rule.pattern,))
            if span is None:
                continue
            start, end = span
            fragment = cmd[start:end]
            if any(skip in fragment for skip in rule.skip_if):
                continue
            before = cmd
            cmd = cmd[:end] + " " + rule.flag + cmd[end:]
            if rule_log is not None:
                rule_log.append({"kind": "universal", "flag": rule.flag})
            if transform_log is not None:
                transform_log.append({
                    "kind": "universal",
                    "name": rule.name,
                    "flag": rule.flag,
                    "before": before,
                    "after": cmd,
                })
            break  # one rewrite per command

    # Task-format rewrite — test runner flags.
    if oc and oc.failure_only_flag:
        span = _find_command_span(
            cmd, oc.command_patterns if oc.command_patterns is not None
            else oc.verification_patterns,
        )
        if span is not None:
            start, end = span
            fragment = cmd[start:end]
        else:
            fragment = ""
        if span is not None:
            before = cmd
            fragment = _quiet_fragment(fragment, oc)
            separator = next((word.start() for word in re.finditer(_SHELL_WORD, fragment)
                              if word.group() == "--"), len(fragment))
            # Strip old quiet flags too, so a second rewrite stays quiet.
            flags = " ".join(flag for flag in oc.failure_only_flag.split()
                             if flag not in fragment[:separator].split())
            prefix = fragment[:separator].rstrip()
            suffix = fragment[separator:]
            fragment = prefix + (" " + flags if flags else "") + (" " + suffix if suffix else "")
            cmd = cmd[:start] + fragment + cmd[end:]
            if rule_log is not None:
                rule_log.append({"kind": "test_flag",
                                 "flag": oc.failure_only_flag})
            if transform_log is not None:
                transform_log.append({
                    "kind": "test_flag",
                    "flag": oc.failure_only_flag,
                    "before": before,
                    "after": cmd,
                })

    if oc:
        before = cmd
        cmd = strip_display_only_test_pipeline(
            cmd,
            oc.verification_patterns,
        )
        if cmd != before:
            if rule_log is not None:
                rule_log.append({"kind": "test_output_filter_removed"})
            if transform_log is not None:
                transform_log.append({
                    "kind": "test_output_filter_removed",
                    "before": before,
                    "after": cmd,
                })

    return cmd
