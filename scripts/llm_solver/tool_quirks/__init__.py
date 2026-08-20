"""tool_quirks — non-bash tool result transforms.

Parallel absorber to bash_quirks/. Bash quirks owns transforms over bash
output (rewrites, sink-and-surface, structured output). Tool quirks owns
transforms over OTHER tool results — glob, grep, future tools — where the
SAME architectural pattern applies (model produces wasteful action → harness
rewrites the result before it reaches the model) but the dispatcher is
not bash.

``bash_quirks`` remains bash-only by name. This directory is its sibling
for the rest of the tool surface.

Each tool gets its own TOML data file (`glob.toml`, future `grep.toml`)
and a function in `transforms.py` that consumes the data + cfg and
returns the modified result.
"""
