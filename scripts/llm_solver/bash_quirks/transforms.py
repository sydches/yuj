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
from ._rewrites import RewriteRule, load_universal_rewrites


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
        span = _find_command_span(cmd, oc.verification_patterns)
        if span is not None:
            start, end = span
            fragment = cmd[start:end]
        else:
            fragment = ""
        if span is not None and oc.failure_only_flag not in fragment:
            before = cmd
            cmd = cmd[:end] + " " + oc.failure_only_flag + cmd[end:]
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

    return cmd
