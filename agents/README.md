# Named agents

Each `agents/<name>.toml` file defines one agent for the optional `task` tool.
An agent name may contain letters, digits, `.`, `_`, and `-`. It may not
contain a path separator.

```toml
[agent]
model_profile = "_base"
tools = ["read", "glob", "grep", "bash", "done"]
system_prompt_file = "prompts/research.md"
max_turns = 12
read_only = true
```

| Field | Meaning |
| --- | --- |
| `model_profile` | Profile name or family under `profiles/`. |
| `tools` | Complete allowlist of public model tools. |
| `system_prompt_file` | Existing Markdown file under `agents/`, resolved from the descriptor. |
| `max_turns` | Agent turn limit, capped by `tools.subagent_max_turns`. |
| `read_only` | Reject mutation tools and restrict `bash` to simple inspection commands. Defaults to `true`. |

The allowlist can restrict globally enabled tools; it cannot turn on a public
tool whose own configuration gate is off.

Agents are read-only unless the descriptor sets `read_only = false`. A
read-only agent cannot use `write`, `edit`, `notebook_edit`, `apply_patch`,
`udiff`, `exec_cell`, `run_tests`, background-process tools, or `task`. Its
`bash` tool accepts only `cat`, `grep`, `head`, `ls`, `pwd`, `stat`, `tail`, and
`wc`. It rejects shell control, redirection, substitution, command paths, and
unknown commands.
