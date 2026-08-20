"""Bash quirks — absorbs shell-command noise and runner-specific output shape.

Two mechanisms:
- Universal rewrites (``rewrites.toml``) — quiet flags for noisy commands
  (pip, npm, make) applied regardless of task format.
- Task-format output control — runner-specific flags (pytest --tb=short) and
  output condensation (strip PASSED lines), loaded from the language_quirks
  package.

Adding a new quiet flag = adding a TOML entry in ``rewrites.toml``.
Adding runner-specific output control = adding a ``[output_control]``
section in a ``language_quirks/<runner>.toml`` file.

The harness imports ``rewrite_command`` / ``condense_output`` and applies
them around every bash tool call. No runner vocabulary in harness code.
"""
from .transforms import (
    OutputControl,
    OutputParser,
    RedactionRule,
    RewriteRule,
    apply_redactions,
    condense_output,
    load_output_control,
    load_output_parser,
    load_redactions,
    load_universal_rewrites,
    parse_structured,
    render_digest,
    rewrite_command,
)

__all__ = [
    "OutputControl",
    "OutputParser",
    "RedactionRule",
    "RewriteRule",
    "apply_redactions",
    "condense_output",
    "load_output_control",
    "load_output_parser",
    "load_redactions",
    "load_universal_rewrites",
    "parse_structured",
    "render_digest",
    "rewrite_command",
]
