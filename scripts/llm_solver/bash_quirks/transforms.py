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
                    rule_log: list | None = None) -> str:
    """Rewrite a bash command to reduce output volume — or refuse it.

    Order of operations:
      1. Forbidden rules (hard refusal — replaces cmd with `false  #
         [HARNESS: ...]` if any pattern matches).
      2. Universal rewrites (pip -q, npm --loglevel=error, etc.).
      3. Task-format-specific flags (--tb=short for pytest).

    Forbidden runs first because once we refuse, the rewrites and
    test-flag transforms are moot.
    """
    cmd = apply_forbidden(cmd, forbidden_rules)
    if "false  # [HARNESS:" in cmd:
        if rule_log is not None:
            rule_log.append({"kind": "forbidden"})
        return cmd

    # Append flags only to single-line commands. A flag after a heredoc
    # terminator or later line would change the command. A ``make`` rule
    # may also match plain text inside a heredoc.
    if "\n" in cmd or "<<" in cmd:
        return cmd

    # Universal rewrites — always apply.
    if universal_rewrites:
        for rule in universal_rewrites:
            if not rule.pattern.search(cmd):
                continue
            if any(skip in cmd for skip in rule.skip_if):
                continue
            # Append flag before trailing pipe chain.
            pipe_idx = cmd.find("|")
            if pipe_idx > 0:
                cmd = cmd[:pipe_idx].rstrip() + " " + rule.flag + " " + cmd[pipe_idx:]
            else:
                cmd = cmd.rstrip() + " " + rule.flag
            if rule_log is not None:
                rule_log.append({"kind": "universal", "flag": rule.flag})
            break  # one rewrite per command

    # Task-format rewrite — test runner flags.
    if oc and oc.failure_only_flag:
        if _is_test_command(cmd, oc) and oc.failure_only_flag not in cmd:
            pipe_idx = cmd.find("|")
            if pipe_idx > 0:
                cmd = cmd[:pipe_idx].rstrip() + " " + oc.failure_only_flag + " " + cmd[pipe_idx:]
            else:
                cmd = cmd.rstrip() + " " + oc.failure_only_flag
            if rule_log is not None:
                rule_log.append({"kind": "test_flag",
                                 "flag": oc.failure_only_flag})

    return cmd
