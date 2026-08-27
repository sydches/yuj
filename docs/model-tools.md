---
layout: default
title: Model tools
nav_order: 6
---

# Model tools

Yuj gives the model a small set of named tools. These tools let the model read,
change, and test files in the task repository.

This page describes the public tool interface. A tool is not a command that a
person types into a terminal.

## Shipped tools

| Tool | Required inputs | Optional inputs | What it does |
| --- | --- | --- | --- |
| `bash` | `cmd` | `background` | Run one shell command and return its output. When background work is enabled, return a process ID instead of waiting. |
| `bash_poll` | `proc_id` | `timeout_s` | Return new output from one background command and, when known, its exit status. |
| `bash_kill` | `proc_id` | None | Stop one background process group. |
| `terminal_start` | `cmd` | None | Start the session's one bounded interactive terminal process and return its terminal ID. |
| `terminal_io` | `terminal_id` | `input`, `append_newline`, `timeout_s`, `terminate` | Send input, return new output and status, or stop the interactive process. |
| `read` | `path` | `offset`, `limit` | Read a file with line numbers. `offset` starts at 0. `limit=0` means no line limit. Paths normally stay in the task cwd; an enabled Agent Skill's listed absolute directory is also readable. |
| `write` | `path`, `content` | None | Create or replace a file. Create missing parent directories. |
| `edit` | `path`, `old_str`, `new_str` | None | Replace the first exact copy of `old_str`. |
| `notebook_edit` | `path`, `old_source`, `new_source` | One of `cell_index` or `cell_id` | Replace the exact source of one existing code or Markdown cell without rewriting unrelated notebook data. |
| `glob` | `pattern` | `path`, `page` | Find paths that match a glob pattern. `path` defaults to `.`. `page` defaults to 1. |
| `grep` | `pattern` | `path`, `glob`, `page` | Search file text with a regular expression. `path` defaults to `.`. `glob` limits file names. `page` defaults to 1. |
| `write_todos` | `todos` | None | Replace the whole session todo list. Each item has a `description` and `status`; at most one item may be `in_progress`. |
| `checkpoint` | `goal` | None | Mark a complete conversation turn before exploration. Becomes active after every call/result pair in that turn completes. |
| `rewind` | `report` | None | Return conversation context to the active checkpoint and retain the short findings report. Never restores files. |
| `lsp` | `kind`, `path` | `line`, `character` | Ask a configured language server for `definition`, `references`, or document `symbols`. Line and character offsets are zero-based. |
| `think` | `thought` | None | Record free-form scratchpad reasoning without running a process or touching the filesystem. Return an empty success envelope. |
| `run_tests` | None | `path`, `k`, `last_failed` | Run the detected test runner. Limit the run by path or test name. `last_failed=true` repeats failed tests with pytest, Jest, or CTest. Cargo and Go ignore it. |
| `list_definitions` | `path` | `symbol`, `kind`, `repo_wide`, `page` | With `path` alone, list one Python file's outline. With `repo_wide=true`, find exact symbol definitions or references across the repository. Do not run source files. |
| `apply_patch` | `patch` | None | Apply one checked patch that may add, change, or delete several files. |
| `task` | `agent`, `prompt` | None | Run one named agent in a separate context and return its final text. |
| `udiff` | `patch` | None | Apply a checked standard unified diff with safe unique-context recovery. |
| `list_functions` | None | None | In code mode, list the function names injected into `exec_cell`. |
| `get_function_details` | `names` | None | In code mode, return selected injected-function schemas on demand. |
| `exec_cell` | `source` | None | In code mode, run Python inside the shell sandbox and return printed text. |
| `load_tools` | `names` | None | Add hidden registered tools to the active set for later model requests. Present only when deferred loading is enabled. |
| `exit_plan_mode` | None | None | Validate `.solver/plan.md` and unlock implementation tools during a required planning phase. |
| `ask_user` | `question` | None | In the top-level assistant session, save one exact clarification question and pause for one operator answer. Never requests or grants permission. |
| `done` | None | `message` | Ask Yuj to end the task. |

The exact parameter shapes live in
[`profiles/_base/tool_schemas.toml`](https://github.com/sydches/yuj/blob/main/profiles/_base/tool_schemas.toml).
Read [Extend Yuj with TOML files](extending-yuj.html) before you change a tool
schema, description, or result rule.

## Tools that are on by default

| Tool | Shipped setting |
| --- | --- |
| `read`, `glob`, `grep`, `bash`, `done` | On |
| `edit` | Selected by the shipped profile's `exact` edit format. |
| `write` | On as the file-creation companion to the selected `exact` replacement tool. |
| `apply_patch`, `udiff` | Available replacement dialects, but not selected by the shipped profile. |
| `notebook_edit` | Off |
| `load_tools` | On only while `[tools].lazy_loading_enabled` is true. |
| `exit_plan_mode` | On only when `[loop].plan_mode = "required"`. |
| `ask_user` | On for the top-level assistant session. Never present in child-agent or measurement requests. |
| `think`, `write_todos`, `checkpoint`, `rewind`, `list_definitions`, `run_tests`, `lsp`, `bash_poll`, `bash_kill`, `terminal_start`, `terminal_io`, `task` | Off |
| `list_functions`, `get_function_details`, `exec_cell` | Off; enabled together by code mode. |

Turn on the optional tools in a small settings file:

```toml
[tools.list_definitions]
enabled = true

[tools.run_tests]
enabled = true

[tools]
think_enabled = true
think_keep_turns = 4
edit_format = "apply_patch"
background_enabled = true
terminal_enabled = true
task_enabled = true
todos_enabled = true
checkpoint_enabled = true
notebook_edit_enabled = true

[lsp]
tool_enabled = true
```

Apply the file to a coding session:

```bash
yuj --config more-tools.toml "Fix the issue and run the tests."
```

When `bash` is on, the model can run a test command through `bash` even when
`run_tests` is off. The `run_tests` tool gives Yuj a fixed test command and a
structured result.

`think` is a visible, short-lived model scratchpad. It does not run a process
or change a file. Yuj removes it from model context after `think_keep_turns`,
but keeps its raw trace row for audit and replay.

`write_todos` replaces the whole list on every call. Its statuses are
`pending`, `in_progress`, `completed`, `cancelled`, and `blocked`. Yuj allows
only one `in_progress` item and no more than `todos_max_items` items. An empty
list clears it. The tool changes harness state, not source files.

A model profile may limit the number of visible tools. `done` always remains.
The top-level assistant session also keeps `ask_user`. Deferred loading keeps
`load_tools`. Required plan mode keeps `exit_plan_mode` and the exact plan-file
write so the model can leave the phase. When interactive terminals are enabled,
Yuj keeps both terminal controls inside the limit. The shipped eight-tool
profiles still keep `bash`, `read`, `write`, and the selected edit tool.

## Choose a tool set

### Required planning phase

Required plan mode starts with inspection tools, read-only `bash`, `write`, and
`exit_plan_mode`. The top-level assistant session also keeps `ask_user`.
`write` may target only `.solver/plan.md`. Yuj rejects edits, tests, mutating
or unknown shell commands, subagents, code cells, deferred-tool activation,
and `done` before they run.

`exit_plan_mode` takes no input. It succeeds only after the model writes a
nonempty `.solver/plan.md`, then restores the normal tool set. The plan
file does not count as an implementation change. See
[Configuration](configuration.html#require-a-plan-before-implementation) for
the turn limit, CLI flag, trace events, resume rule, and state projection.

### Deferred tool loading

With deferred loading, Yuj first sends `active_default`, `load_tools`, and
`done`. The top-level assistant session also receives `ask_user`. Other
enabled tools stay hidden. The model calls
`load_tools(names=[...])` to add exact names for the next request. Loading never
removes a tool. Yuj rejects the whole call if the result would exceed the
profile's `max_tools` limit.

A direct call to a hidden tool returns `tool_not_active` and names
`load_tools`. A disabled tool cannot be loaded.
See [Configuration](configuration.html#defer-tools-until-the-model-needs-them)
for knobs, trace/state behavior, replay fidelity, and token metrics.

### Conversation checkpoint and rewind

`checkpoint` and `rewind` use one setting and remain visible as a pair. The
model calls `checkpoint(goal)` before an exploration. It later calls
`rewind(report)` to return its conversation to that point and keep a short
findings report. Yuj waits for every tool result in the current turn before it
changes the conversation.

The raw trace keeps the abandoned exploration, but later model context and
state follow the checkpoint. File changes remain. This is not the operator
rewind that restores conversation and files together. See
[Configuration](configuration.html#collapse-a-model-exploration-branch).

### Ask the operator for one clarification

In a top-level assistant session, the model can call:

```json
{"question":"Which database should the migration target?"}
```

`question` must be a nonempty string. No other field is accepted. Yuj checks
this schema even when general tool schema validation is off.

A valid call saves the exact question and ends the run segment with
`input_required`. Yuj does not dispatch another call from the same model
response and does not make another model request. Use `yuj status` or `yuj
show` to read the question and exact answer command. One coding session can
create only one clarification request.

The operator answer is information, not authorization. It cannot approve a
tool, satisfy an approval request, change a permission rule, or bypass the
sandbox. The next resume sends the recorded answer once. Measurement mode
never sends the `ask_user` schema and never enters `input_required`.
Read [Answer a clarification question](using-yuj.html#answer-a-clarification-question)
for the operator commands.

### Code mode

Code mode replaces the native catalog with three meta-tools and `done`.
Assistant code mode also keeps `ask_user`. Call `list_functions`, request the
needed schemas with `get_function_details`, then send Python to `exec_cell`.
The cell provides `read`, `grep`, `glob`,
`list_definitions`, and `bash`. Each function returns text through the normal
dispatcher. The Python program must print the text that should become the
cell result.

Code mode and deferred tool loading are alternative compact surfaces. A
configuration that enables both is rejected.

Cells run model-written Python under the resolved command policy. An explicit
`none` choice runs the cell as the Yuj account; every requested sandbox fails
closed if it cannot start. Cells cannot start background commands and stop at
the whole-cell timeout. Inner calls keep
the normal filters, redaction, result envelope, ignore rules, permissions, and
trace behavior. See
[Configuration](configuration.html#run-a-sandboxed-python-cell) for the two
settings and [Sandbox](sandbox.html#python-code-mode) for the boundary.

### Edit format

Yuj gives a tool-calling model one replacement dialect at a time. It also
supplies `write` with the `exact`, `apply_patch`, and `udiff` dialects because
those tools cannot create a missing file. The `whole` dialect uses `write` for
both creation and replacement.

| Format | Replacement tool | File-creation tool |
| --- | --- | --- |
| `exact` | `edit` | `write` |
| `apply_patch` | `apply_patch` | `write` |
| `udiff` | `udiff` | `write` |
| `whole` | `write` | `write` |

The model profile selects the normal format. Override it with
`[tools].edit_format` or `--edit-format`. See
[Configuration](configuration.html#select-the-models-edit-format) for the
precedence rules.

### Notebook cell edits

Enable `notebook_edit` when the task includes Jupyter notebooks:

```toml
[tools]
notebook_edit_enabled = true
```

Read the notebook before each edit. Then select one existing cell with either
its zero-based `cell_index` or its `cell_id`. Do not pass both selectors. Pass
the complete current cell source as `old_source` and the replacement as
`new_source`.

`notebook_edit` supports code and Markdown cells. It keeps the source as the
same JSON type, either a string or an array of strings. It also preserves cell
order, IDs, metadata, outputs, attachments, and every byte outside the selected
`source` value. A missing cell, duplicate ID, stale source, invalid notebook,
or no clear selector changes no file.

The tool uses the normal workspace, permission, approval, stale-file, workspace
checkpoint, and post-edit controls. A configured post-edit check uses the
`edit` trigger. Read
[Configuration](configuration.html#edit-one-jupyter-notebook-cell) for the
setting.

### Language-server feedback

With `[lsp].enabled = true`, Yuj adds matching diagnostics to a successful edit
result. This does not require the `lsp` navigation tool. The severity setting
controls which messages the model sees. Enable `[lsp].tool_enabled` separately
to expose definition, reference, and symbol queries. See
[Configuration](configuration.html#return-language-server-diagnostics-after-edits).

## File and shell limits

The file tools keep their paths inside the task repository. They do not give
the model a general path into the host system.

The `bash`, `terminal_start`, and `terminal_io` tools follow the active sandbox
and approval settings. Read
[Sandbox](sandbox.html) before you let the model work on private files or use a
Docker socket.

### Permission rules

`[permissions].rules` can allow, ask, or deny a tool by its stable match value.
The last matching `*` or `?` rule wins. Empty rules keep the normal behavior,
and `allow` does not bypass shell-specific refusals. Assistant sessions pause
for `yuj approve` or `yuj reject`; measurement runs deny `ask`. See
[Configuration](configuration.html#apply-per-tool-permission-rules) for the
rule table and each tool's match value.

### Background commands

With background commands enabled, the model passes `background = true` to
`bash` and receives a `proc_id`. `bash_poll` returns only new output.
`bash_kill` stops the process group. Polls keep the normal output filters,
redaction, envelope, and size limit. Yuj stops every remaining child when the
session ends.

### Interactive terminal process

Interactive terminal support is for a debugger, REPL, or another program that
will not work without a terminal. It is assistant-only and off by default.
`terminal_start` returns a `terminal_id`. `terminal_io` can send one bounded
input string and then return only new output. Omit `input` to read or inspect
status. Set `terminate=true`, without `input`, to stop the process and return
its final output and status.

Yuj owns the pseudo-terminal. It never gives the process the operator's
terminal, and it never inserts process output between model turns. One process
may be live at a time. The selected command boundary, permission rules, output
filters, redaction, security scan, and session cleanup still apply. A risky
initial command or risky input uses the normal approval gate.

Yuj records accepted input by hash and size, not by copying it into the
`terminal_input` trace row. The raw terminal log may still contain input echoed
by the program. Read
[Configuration](configuration.html#run-an-interactive-terminal-process) for
the limits and [Saved files](harness_artifacts.html) for the evidence boundary.

### Shell redirects

Yuj can stop a shell fragment when an active dedicated tool owns the same
operation. The error names the tool to use. Write-side rules cover in-place
editors and file redirection. Set `bash_redirect_read_side = true` to route
simple file display, text search, and path discovery to `read`, `grep`, and
`glob`. Count-only commands and pipe stages that read standard input remain
shell commands. A rule does nothing when its target tool is not active.

The `glob` and `grep` tools return one page at a time. Read `next_page` in the
result when another page exists. Yuj can refuse a search that starts too broad.
Narrow its pattern or path when that happens.

### Read before edit

The stale-file guard can require current contents before an edit, including a
`notebook_edit`. A typed
`read` or one simple single-file shell read records the content hash. If the
file changes later, that read becomes stale. `warn` applies the edit and adds a
warning. `block` returns `ERROR: stale_file: read PATH first` without changing
the file. A successful edit refreshes its affected paths.

### Patch behavior

`apply_patch` uses the Codex V4A `*** Begin Patch` and `*** End Patch`
grammar. `udiff` uses standard `---`, `+++`, and `@@` lines. Yuj checks every
file and hunk before it writes anything. A hunk line number is a hint, so Yuj
may use one unique exact or whitespace-normalized whole-line match. A missing
or ambiguous hunk changes no files and returns repair candidates.

### Repository structural search

Repository-wide `list_definitions` returns
`path:line kind name signature`. `symbol` matches an exact name, and `kind` is
`def` or `ref`. Use `page` when the result names a nonzero `next_page`. Enable
both `list_definitions` and `ast_search_enabled`. Yuj ships local parsers for
Python, JavaScript and TypeScript, Go, Rust, and Java. A missing parser returns
a setup error instead of downloading one.

## Named subagents

With `task_enabled = true`, `task(agent, prompt)` selects
`agents/<name>.toml`. That file chooses the child's model profile, tool
allowlist, system prompt, turn limit, and read-only status. Agents are
read-only by default.

A child uses the parent's task directory and sandbox policy, but starts a new
conversation and model client. Children run one at a time. Depth and turn
settings limit their work. The parent sees only the final text. Yuj saves a
separate child trace for audit and replay and includes child tokens in the run
metrics. See [Configuration](configuration.html#run-named-subagents) and
[`agents/README.md`](https://github.com/sydches/yuj/blob/main/agents/README.md).

## Argument schema rejection

With `schema_validation = "reject"`, Yuj checks a call against the schema sent
to the model before any handler or guard runs. Missing fields, undeclared
fields, and wrong JSON types are errors. A rejected call does not run and
returns a model-visible error that begins:

```text
ERROR: {"error":{"errors":[...],"message":"Tool arguments do not match the declared schema.","tool":"read","type":"tool_schema_reject","version":1}}
```

Each error names the JSON path, failed schema rule, expected shape, and actual
type. It omits argument values from both the returned error and trace metadata.

## Test runners

`run_tests` detects one of these runners from files in the task repository:

| Project files | Runner |
| --- | --- |
| Python project files | `pytest` |
| `Cargo.toml` | Cargo tests |
| `go.mod` | Go tests |
| `package.json` | Jest |
| CMake build files | CTest |

Yuj returns the runner name and result status in a `<test_results>` block. It
also returns `exit_code` when the test runner exits. A timeout or tool error has
no `exit_code`. The default timeout is 240 seconds.

## Finish rule

The shipped treatment and plain bases accept `done` without requiring a file
change or test. Both bases set `done_guard_enabled = false` under `[loop]`.

Set `done_guard_enabled = true` under `[loop]` to turn on the guard. Set
`done_require_mutation` under `[loop]` to require a file change. Set
`done_require_verify` under `[loop]` to require a later check. When the guard
rejects `done`, Yuj tells the model what it still needs to do.

## Tool descriptions

Yuj sends a short description with each tool. The public release ships one
description mode: `minimal`.

The measurement command accepts `--tool-desc minimal`. The installed `yuj`
command reads the mode from `[experiment].tool_desc` in its settings.

Read the [CLI reference](using-yuj.html) for the installed command. Read
[Measurements](measurement.html) for the measurement command.
