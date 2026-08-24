# Named agents

Each `agents/<name>.toml` file defines one agent that the optional `task` tool
can run. Agent names may contain letters, digits, `.`, `_`, and `-` and cannot
contain a path separator.

```toml
[agent]
model_profile = "_base"
tools = ["read", "glob", "grep", "bash", "done"]
system_prompt_file = "prompts/research.md"
max_turns = 12
read_only = true
```

`model_profile` selects a profile by name or family under `profiles/`. `tools`
is an allowlist of public model tools. `system_prompt_file` is an existing
Markdown file under `agents/`, resolved relative to the descriptor.
`max_turns` is capped by `tools.subagent_max_turns`.

The allowlist can restrict globally enabled tools; it cannot turn on a public
tool whose own configuration gate is off.

Agents are read-only unless `read_only = false` is explicit. A read-only agent
cannot allow `write`, `edit`, `apply_patch`, `udiff`, `exec_cell`, `run_tests`,
background-process tools, or `task`. Its `bash` tool accepts only a fail-closed
allowlist of simple inspection commands: `cat`, `grep`, `head`, `ls`, `pwd`,
`stat`, `tail`, and `wc`. Shell control, redirection, substitution, command
paths, and unknown commands are rejected.
