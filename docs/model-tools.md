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
| `bash` | `cmd` | None | Run one shell command and return its output. |
| `read` | `path` | `offset`, `limit` | Read a file with line numbers. `offset` starts at 0. `limit=0` means no line limit. |
| `write` | `path`, `content` | None | Create or replace a file. Create missing parent directories. |
| `edit` | `path`, `old_str`, `new_str` | None | Replace the first exact copy of `old_str`. |
| `glob` | `pattern` | `path`, `page` | Find paths that match a glob pattern. `path` defaults to `.`. `page` defaults to 1. |
| `grep` | `pattern` | `path`, `glob`, `page` | Search file text with a regular expression. `path` defaults to `.`. `glob` limits file names. `page` defaults to 1. |
| `lsp` | `kind`, `path` | `line`, `character` | Ask a configured language server for `definition`, `references`, or document `symbols`. Line and character offsets are zero-based. |
| `run_tests` | None | `path`, `k`, `last_failed` | Run the detected test runner. Limit the run by path or test name. `last_failed=true` repeats failed tests with pytest, Jest, or CTest. Cargo and Go ignore it. |
| `list_definitions` | `path` | `symbol`, `kind`, `repo_wide`, `page` | With `path` alone, list one Python file's outline. With `repo_wide=true`, find exact symbol definitions or references across the repository. Do not run source files. |
| `apply_patch` | `patch` | None | Apply one checked patch that may add, change, or delete several files. |
| `udiff` | `patch` | None | Apply a checked standard unified diff with safe unique-context recovery. |
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
| `write`, `apply_patch`, `udiff` | Available edit dialects, but not selected by the shipped profile. |
| `list_definitions`, `run_tests`, `lsp`, `bash_poll`, `bash_kill` | Off |

Turn on the optional tools in a small settings file:

```toml
[tools.list_definitions]
enabled = true

[tools.run_tests]
enabled = true

[tools]
edit_format = "apply_patch"
background_enabled = true

[lsp]
tool_enabled = true
```

Apply the file to a coding session:

```bash
yuj code --config more-tools.toml "Fix the issue and run the tests."
```

When `bash` is on, the model can run a test command through `bash` even when
`run_tests` is off. The `run_tests` tool gives Yuj a fixed test command and a
structured result.

A model profile can also limit how many enabled tools Yuj sends to the model.
The `done` tool is not removed by that limit.

For a tool-calling profile, Yuj retains exactly one of `edit`, `apply_patch`,
`udiff`, and `write` according to the effective edit format. `exact` selects
`edit`, `apply_patch` selects the Codex V4A patch tool, `udiff` selects standard
unified diffs, and `whole` selects `write`. Set inherited
`[profile].edit_format` for a model or override it with `[tools].edit_format`
or `--edit-format`. See
[Configuration](configuration.html#select-the-models-edit-format) for the
precedence rules.

When `[lsp].enabled` is true, Yuj automatically appends diagnostics after a
successful edit-dialect mutation; this does not require the navigation tool to
be enabled. The configured severity threshold controls which messages enter
the model-facing result. See [Configuration](configuration.html) for the
server table and timeout settings.

## File and shell limits

The file tools keep their paths inside the task repository. They do not give
the model a general path into the host system.

The `bash` tool follows the active sandbox and approval settings. Read
[Sandbox](sandbox.html) before you let the model work on private files or use a
Docker socket.

`[permissions].rules` can allow, ask for approval, or deny a tool by its
canonical command/path argument. The last matching `*`/`?` rule wins. Empty
rules allow current behavior, and an allow still passes through bash-specific
forbidden rules. Assistant `ask` decisions use `yuj approve|reject`; measurement
runs deny them. See
[Configuration](configuration.html#apply-per-tool-permission-rules) for the
table and exact match fields.

With `[tools].background_enabled = true`, pass `background = true` to `bash`
to receive a `proc_id` without waiting for the command. `bash_poll` returns
only output added since the preceding poll and waits no longer than
`[tools].background_poll_timeout`; `bash_kill` terminates the process group.
The live-process limit is `[tools].background_max_procs`. Poll output uses the
same filters, redaction, result envelope, and output limit as other tools. Yuj
kills every remaining child at session end.

Yuj can reject a shell fragment when an active dedicated tool owns the same
operation. The result says `Blocked:` and names the tool to use. Write-side
redirects cover in-place editors and file redirections. Set
`[tools].bash_redirect_read_side = true` to also redirect `cat`/`head`/`tail`
to `read`, `grep`/`rg` to `grep`, and `find`/`fd` to `glob`. The matcher checks
compound commands and leading environment assignments, but leaves
stdin-consuming pipe stages and aggregate commands such as `grep -c`,
`rg --count`, `wc -l`, and `cat FILE | wc -l` alone. A rule does nothing when
its target tool is not in the model's active tool set.

The `glob` and `grep` tools return one page at a time. Read `next_page` in the
result when another page exists. Yuj can refuse a search that starts too broad.
Narrow its pattern or path when that happens.

The read-before-edit guard can require current evidence for an `edit`. A typed
`read`, or one successful single-file `cat`, `head`, `tail`, `sed -n`, `grep`,
or `rg` shell command, records the current content hash. An external content
change makes that observation stale. In `warn` mode Yuj applies the edit and
adds a warning inside its result envelope; in `block` mode it returns
`ERROR: stale_file: read PATH first` without changing the file. Successful
`write`, `edit`, `apply_patch`, and `udiff` calls refresh their affected paths.

The `apply_patch` dialect keeps the existing Codex V4A
`*** Begin Patch`/`*** End Patch` grammar. The `udiff` dialect accepts ordinary
`---`/`+++` file headers and `@@` hunks. It checks every file and hunk before
the first write. Hunk line numbers are hints: Yuj may use one unique exact
offset or one unique whitespace-normalized whole-line match. Missing or
ambiguous hunks leave every file unchanged and return the same ranked
`<candidates>` repair block used by a strict exact-string `edit` miss.

Repository-wide `list_definitions` rows use
`path:line kind name signature`. `symbol` is an exact name, and `kind` is
`def` or `ref`; omit either to keep both. Use `page` when the result envelope
names a nonzero `next_page`. This mode requires the separately disabled
`[tools].ast_search_enabled` setting as well as the normal
`[tools.list_definitions].enabled` gate. The installed structural-search
dependencies provide Python, JavaScript/TypeScript, Go, Rust, and Java
grammars locally; a missing backend returns a setup error instead of
downloading during the tool call.

## Argument schema rejection

When `[tools].schema_validation = "reject"`, Yuj checks a call against the
same effective schema sent to the model before any handler or guard runs.
Parameter objects are closed: fields not declared by the tool are invalid as
well as missing required fields and values of the wrong JSON type. A rejected
call does not execute and returns a normal model-visible error beginning:

```text
ERROR: {"error":{"errors":[...],"message":"Tool arguments do not match the declared schema.","tool":"read","type":"tool_schema_reject","version":1}}
```

Each error names a JSON field path, failed schema keyword, expected shape, and
actual JSON type. Argument values are excluded from the validation error and
its `schema_reject` trace metadata so the model can repair the shape without
duplicating potentially sensitive values.

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
