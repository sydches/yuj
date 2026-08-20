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
| `run_tests` | None | `path`, `k`, `last_failed` | Run the detected test runner. Limit the run by path or test name. `last_failed=true` repeats failed tests with pytest, Jest, or CTest. Cargo and Go ignore it. |
| `list_definitions` | `path` | None | List top-level imports, `__all__`, all-capital names, and annotated names. List all classes and functions. Do not run the file. |
| `apply_patch` | `patch` | None | Apply one checked patch that may add, change, or delete several files. |
| `done` | None | `message` | Ask Yuj to end the task. |

The exact parameter shapes live in
[`profiles/_base/tool_schemas.toml`](https://github.com/sydches/yuj/blob/main/profiles/_base/tool_schemas.toml).
Read [Extend Yuj with TOML files](extending-yuj.html) before you change a tool
schema, description, or result rule.

## Tools that are on by default

| Tool | Shipped setting |
| --- | --- |
| `read`, `glob`, `grep`, `write`, `edit`, `bash`, `done` | On |
| `list_definitions`, `apply_patch`, `run_tests` | Off |

Turn on the optional tools in a small settings file:

```toml
[tools.list_definitions]
enabled = true

[tools.apply_patch]
enabled = true

[tools.run_tests]
enabled = true
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

## File and shell limits

The file tools keep their paths inside the task repository. They do not give
the model a general path into the host system.

The `bash` tool follows the active sandbox and approval settings. Read
[Sandbox](sandbox.html) before you let the model work on private files or use a
Docker socket.

The `glob` and `grep` tools return one page at a time. Read `next_page` in the
result when another page exists. Yuj can refuse a search that starts too broad.
Narrow its pattern or path when that happens.

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
