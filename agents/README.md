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
workspace = "shared"
```

| Field | Meaning |
| --- | --- |
| `model_profile` | Profile name or family under `profiles/`. |
| `tools` | Complete allowlist of public model tools. |
| `system_prompt_file` | Existing Markdown file under `agents/`, resolved from the descriptor. |
| `max_turns` | Agent turn limit, capped by `tools.subagent_max_turns`. |
| `read_only` | Reject mutation tools and restrict `bash` to simple inspection commands. Defaults to `true`. |
| `workspace` | Choose `shared` or `isolated`. Defaults to `shared`. Read-only agents must use `shared`. |

The allowlist can restrict globally enabled tools; it cannot turn on a public
tool whose own configuration gate is off.

Agents are read-only unless the descriptor sets `read_only = false`. A
read-only agent cannot use `write`, `edit`, `notebook_edit`, `structural_edit`,
`apply_patch`, `udiff`, `exec_cell`, `run_tests`, background-process tools, or
`task`. Its
`bash` tool accepts only `cat`, `grep`, `head`, `ls`, `pwd`, `stat`, `tail`, and
`wc`. It rejects shell control, redirection, substitution, command paths, and
unknown commands.

A write-capable agent can use either workspace mode. Set `read_only = false`
and `workspace = "isolated"` when the child must not change parent files
directly. Yuj copies the parent's exact Git workspace into a temporary
worktree. The copy includes tracked, staged, unstaged, and untracked files.

An isolated child returns a bounded change set after it finishes. The parent
can inspect that patch with `subagent_changes`. The parent can apply it with
`apply_subagent`. The apply step uses normal tool permissions and rejects the
patch when the parent workspace no longer matches the child's starting state.

An isolated workspace requires a Git repository with a valid `HEAD`. Without
an active sandbox, an isolated descriptor may use contained file tools but
cannot use process-backed tools such as `bash`, `run_tests`, or `exec_cell`.
Shared read-only agents keep their existing lightweight behavior.
