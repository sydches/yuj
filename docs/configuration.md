---
layout: default
title: Configuration
nav_order: 4
---

# Configuration

Use `yuj setup` for normal use. Change a TOML file only when you need a
setting that the command does not offer.

This page explains settings files and their order. Read
[Extend Yuj with TOML files](extending-yuj.html) when you want to add a model
runtime, model profile, test runner, or tool rule.

Activate the environment where Yuj is installed before you run a command on
this page. Otherwise, replace `yuj` with that environment's `bin/yuj` path.

## Save model settings

For a local server, run:

```bash
yuj setup --provider local --model YOUR_SERVED_MODEL_ID
```

For an online model service, put the key in an environment variable:

```bash
export OPENROUTER_API_KEY='...'
yuj setup --provider openrouter --model PROVIDER_MODEL_ID \
  --api-key-env OPENROUTER_API_KEY
```

Claude and Codex can instead use an eligible subscription:

```bash
yuj setup --provider claude --auth subscription --model YOUR_MODEL_ID
yuj setup --provider codex --auth subscription --model YOUR_MODEL_ID
```

Run `yuj doctor` after you change the service or model.

### `--provider` settings

The `--provider` setting tells Yuj how to reach the service that runs the
model.

| Setup name | API format | Address saved by setup | Usual key variable |
| --- | --- | --- | --- |
| `local` | OpenAI-compatible | `http://localhost:8080/v1` | No variable. Yuj uses the key value `local`. |
| `claude` | Anthropic Messages | Provider endpoint fixed by the selected authentication method | Provider credential file |
| `codex` | OpenAI-compatible for API keys; subscription transport for browser sign-in | Provider endpoint fixed by the selected authentication method | Provider credential file |
| `openai` | OpenAI-compatible | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `anthropic` | Anthropic Messages | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` |
| `openrouter` | OpenAI-compatible | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `zai` | OpenAI-compatible | `https://api.z.ai/api/paas/v4` | `ZAI_API_KEY` |
| `custom` | OpenAI-compatible | value from `--base-url` | value from `--api-key-env` |

Run `yuj setup` with no options when you want an interactive setup.

For `claude` or `codex`, choose `--auth api-key` or `--auth subscription`.
The API-key form accepts `--api-key-env NAME` or `--api-key VALUE`. The
subscription form opens a browser sign-in. Both forms save a provider-scoped
credential outside `config.local.toml` and put only the non-secret
`yuj-host-credential` marker in the server setting.

For the other online choices, prefer `--api-key-env NAME`. Yuj saves
`$ENV:NAME` and reads the key when a command starts.

`--api-key VALUE` saves the key itself in `config.local.toml` for the other
service choices. Git ignores this file. Claude and Codex never save the key
there.

Provider credentials live under `$XDG_CONFIG_HOME/yuj/auth`, or
`~/.config/yuj/auth` when `XDG_CONFIG_HOME` is unset. Yuj creates this
directory with mode `0700` and atomically replaces provider and selection
files with mode `0600`. The credential record is not a configuration layer.
It is never rendered by `yuj config`.

Use `--force` to replace an existing file without an interactive question.

On `code`, `run`, and `smoke`, `--provider local` changes only the API format.
It keeps the loaded base address and key. Run `yuj setup --provider local` to
save the localhost address and `local` key.

## Understand setting order

Yuj calls the starting group of settings a base.

Yuj applies settings in this order:

1. `config.toml` supplies the shipped or checked-in defaults.
2. The resolved machine-local file supplies this machine's service and model.
3. `--treatment` or `--no-treatment` selects a base.
4. Each `--config` file applies from left to right.
5. Model and model-service options on the command apply last.

A later value replaces an earlier value for the same field.

This order keeps local keys out of Git. It also lets you change one setting
without copying the full main file.

Yuj chooses its main runtime root in this order:

1. If `YUJ_CONFIG` is set, use that exact main file and its parent directory.
   Keep every relative profile, agent, base, and runtime file that it names
   under that directory.
2. Otherwise, use an editable source checkout when one is active.
3. Otherwise, use the installed package's immutable runtime bundle.

Yuj does not copy package defaults into a task or user directory.

`YUJ_CONFIG_LOCAL` names an exact machine-local file. Otherwise, a
`YUJ_CONFIG` tree and a source checkout use `config.local.toml` beside the
main file. A package install uses `$XDG_CONFIG_HOME/yuj/config.local.toml`, or
`~/.config/yuj/config.local.toml` when `XDG_CONFIG_HOME` is unset. The file is
optional; `yuj setup` creates its parent only when you explicitly save setup.

Yuj never searches the target repository for a settings overlay. Pass project
or per-run settings explicitly with `--config`; project instruction and
`.yujignore` discovery are separate features.

A paper comparison does not rely on the CLI defaults alone. Follow the exact
order in the
[paper configuration guide](https://github.com/sydches/yuj/blob/main/configs/paper/README.md).

## Inspect the resolved settings

Run a configuration preflight before model work:

```bash
yuj config
```

The command applies and validates the same layers, in the same order, as a new
`yuj` session. It prints every resolved setting with the layer that won.
This includes nested tables and values that still come from `config.toml`.
The preflight loads and validates the full shipped runtime-resource manifest,
model profiles, configured model roles and fallbacks, and enabled named-agent
descriptors. Use `--agent NAME` to validate another named agent even when the
`task` tool is off. Human output reports the resource origin and counts; JSON
places the same values under `references.runtime_resources`.

The command does not contact a model service, start a process, create a coding
session, or write run artifacts. It returns a nonzero status for a missing or
malformed file, a missing environment variable, profile, or named agent, an
invalid value, or an incompatible group of settings.

Use the same base, overlay, and command-line choices that you plan to run:

```bash
yuj config --no-treatment \
  --config first.toml --config second.toml \
  --provider openai --model YOUR_MODEL_ID
```

`--config` remains repeatable and applies left-to-right. The command also
accepts the `code` settings `--treatment`/`--no-treatment`, `--context`,
`--model`, `--thinking`, `--plan-mode`, `--permission-preset`,
`--edit-format`, `--provider`, `--base-url`, and `--api-key-env`.

Use JSON for automation:

```bash
yuj config --json
```

The stable JSON document has schema name `yuj.config-inspection` and
`schema_version` `1`. Its top-level fields are `status`, `success`, `layers`,
`selection`, `settings`, `references`, and `diagnostics`, together with the
schema fields. `layers` is the ordered low-to-high layer list. Each member of
`settings` contains a TOML-style `path`, `path_components`, the effective
`value`, its `source_layer`, a `redacted` boolean, and
`redaction_reasons`. Environment-backed entries also name the environment
variable. Arrays and object keys have deterministic order, so the same inputs
produce the same JSON bytes.

Yuj never prints resolved credential values. The policy redacts API keys,
tokens, passwords, authorization and cookie values, request headers and extra
request bodies, fixed sandbox environment values, future key-like paths, and
every environment-derived value. An exact string in the form `$ENV:NAME` is
resolved by the shared runtime loader. Inspection may show `NAME`, but never
the variable's value. The same redaction policy also applies to validation
diagnostics.

## Find a setting

Start from what you want to change. Each linked section gives the small TOML
overlay and the limits that matter for that choice.

| You want to | Read |
| --- | --- |
| Check which settings and source layers will apply | [Inspect the resolved settings](#inspect-the-resolved-settings) |
| Choose a service or model | [Save model settings](#save-model-settings) |
| Require a plan, correct a known response pattern, or choose an edit format | [Shape the model's work](#shape-the-models-work) |
| Select a sandbox, control command variables, or hide paths | [Control the command boundary](#control-the-command-boundary) |
| Use a worktree, save file checkpoints, or rewind | [Isolate and restore work](#isolate-and-restore-work) |
| Add diagnostics, search, a scratchpad, schema checks, or todos | [Configure model tools](#configure-model-tools) |
| Run background commands, named agents, or Python cells | [Run background work, agents, or code cells](#run-background-work-agents-or-code-cells) |
| Select a fixed permission preset, set permission rules, scan untrusted text, or run lifecycle hooks | [Add policy and trusted automation](#add-policy-and-trusted-automation) |
| Route side requests, ask an advisor, or fall back to another model | [Route model requests](#route-model-requests) |
| Control prompt caching or reasoning effort | [Tune model requests](#tune-model-requests) |
| Choose, compact, or continue context | [Choose a context mode](#choose-a-context-mode) |
| Load project instructions or Agent Skills | [Load project instruction files](#load-project-instruction-files) |

The checked-in [`config.toml`](https://github.com/sydches/yuj/blob/main/config.toml)
lists every public field and default. Use a small overlay for your changes.
Do not copy the full file into your project.

Read [Model tools](model-tools.html) for the model-facing tool interface. Read
[Sandbox](sandbox.html) for isolation boundaries. Read
[Saved files](harness_artifacts.html) for trace and provenance fields.

## Shape the model's work

### Require a plan before implementation

Plan mode stops the model from changing the project until it has written a
plan. Enable it for one coding session:

```bash
yuj --plan-mode required "Plan the change, then implement it."
```

To make it part of a settings layer, use:

```toml
[loop]
plan_mode = "required"       # off | required
plan_mode_max_turns = 15
```

During the planning phase, the model may inspect the project with its enabled
read tools. It may also run shell commands that Yuj classifies as read-only.
The only file it may write is `.solver/plan.md`.

The model finishes the phase by writing a nonempty plan and calling
`exit_plan_mode`. Yuj then restores the normal tool set. It rejects edits,
tests, subagents, background commands, code cells, and `done` before that
point, so a normal model stop does not complete the task.

`plan_mode_max_turns` applies across resumes. After that many planning turns,
Yuj leaves only the plan write and `exit_plan_mode` available. Yuj rebuilds
the phase from `plan_mode_enter` and `plan_mode_exit` trace rows, not from the
presence of the plan file. The plan is a control file and does not count as an
implementation change.

### Correct known response patterns during generation

Use a stream rule when a repository knows how to correct a specific response
pattern. Store the rules as Markdown files under
`.harness/stream_rules/*.md`, then enable them:

```toml
[loop]
stream_rules_enabled = true
stream_rules_dir = ".harness/stream_rules"
stream_rules_context_mode = "discard"
stream_rules_repeat_gap = 10
```

Yuj handles a match according to how the client receives the response:

| Response mode | What Yuj does |
| --- | --- |
| Streaming with `YUJ_STREAMING=1` | Check each chunk. An interrupting match stops the response, adds the rule text, and requests the same logical turn again. |
| Non-streaming | Check the complete response. Add matching prose guidance at the next turn boundary, or add a tool reminder to the matching result. Do not retry the response. |

For an interrupted stream, `discard` removes all partial output from the next
request. `keep` retains completed prose but still removes incomplete tool
calls.

Yuj loads and validates the rule directory before the first model request. A
bad file stops startup and names the file. A missing directory supplies no
rules. The directory must stay inside the task repository.

`repeatMode = "once"` lets a rule fire once per session.
`repeatMode = "after-gap"` uses the rule's `repeatGap`, or
`stream_rules_repeat_gap` when the rule omits one. Yuj counts completed model
turns, not stream chunks or retries. Read
[Extend Yuj with TOML files](extending-yuj.html#add-a-mid-stream-rule) for the
file format and supported scopes.

### Select the model's edit format

For a profile that supports tool calls, Yuj selects one replacement dialect.
The selected model profile supplies the normal choice through inherited
`[profile].edit_format`. The shipped `_base` profile selects `exact`.

The `exact`, `apply_patch`, and `udiff` dialects cannot create a missing file,
so Yuj also supplies `write` as their file-creation tool. The `whole` dialect
uses `write` for both creation and replacement.

| Value | Replacement tool | File-creation tool | Input shape |
| --- | --- | --- | --- |
| `exact` | `edit` | `write` | `path`, exact `old_str`, and `new_str` |
| `apply_patch` | `apply_patch` | `write` | Codex V4A `*** Begin Patch` text in `patch` |
| `udiff` | `udiff` | `write` | Standard `---`/`+++` unified-diff text in `patch` |
| `whole` | `write` | `write` | `path` and complete replacement `content` |

Override the profile for one settings layer with the public setting:

```toml
[tools]
edit_format = "udiff"
```

The checked-in `config.toml` uses an empty string, which means inherit the
profile. Resolution order is `--edit-format` when supplied, the merged
`[tools].edit_format` value when non-empty, the legacy
`[tools.apply_patch].enabled = true` compatibility selector, then the effective
profile. A nonempty public setting wins over the legacy selector. New
settings files should use only `[tools].edit_format`.

Yuj validates profile and settings values before a run. It filters the tool
schemas before schema simplification and the profile tool-count cap. With the
fixed tool surface, each request contains only the selected replacement tool
and its `write` companion. The `whole` dialect contains only `write`. With
deferred tool loading, the registry excludes the unselected replacement tools
but retains `write`. The request includes `write` when `[tools].active_default`
names it or the model loads it. An edit-dialect entry in
`[tools].active_default` resolves to the selected replacement tool.

## Control the command boundary

### Select a shell sandbox backend

`sandbox.backend` is the one sandbox-use and backend choice. It accepts these
values:

| Value | Result |
| --- | --- |
| `bwrap` | Require Linux `bubblewrap`. This is the default and preserves existing Linux behavior. |
| `docker` | Require Docker and a local container image. |
| `podman` | Require Podman and a local container image. |
| `auto` | Use the first installed, operational sandbox in platform order. It never selects `none`. |
| `none` | Explicitly run model commands as the Yuj account without a sandbox. |

Sandboxing is optional. The shipped configuration selects `bwrap`. If `bwrap`
is unsupported, missing, or fails its startup check, Yuj stops before model or
tool work. Save another choice with `yuj setup --sandbox NAME` or set
`sandbox.backend` in a later configuration layer. Yuj never silently falls
back to host execution.

On Linux, `auto` tries `bwrap`, Docker, then Podman. On macOS, it tries
Docker, then Podman. Native Windows has no backend that preserves Yuj's
absolute-path identity contract. Run Yuj in WSL2, which follows the Linux
order and also permits an explicit `none` choice inside WSL2.

To use Docker, name an image that already exists on the host:

```toml
[sandbox]
backend = "docker"  # or "podman"
container_image = "local/yuj-task@sha256:YOUR_DIGEST"
container_flags = ["--memory", "4g", "--pids-limit", "512"]
```

`auto` also needs `container_image` when it may resolve to Docker or Podman.
Yuj inspects the image at task startup and pins commands to that image ID. It
never pulls an image. The container mounts the task directory read-write at
the same absolute path, keeps the image root read-only, disables networking,
and drops capabilities.

Yuj resolves the setting before a model request or tool execution. A named
backend must start exactly as named. `auto` may choose another installed
sandbox backend, but no selected path silently falls back to `none`. Use
`yuj config` to see the supported, available, unavailable, selected, and
capability-resolved backends. Available means that Yuj found a supported
executable; unavailable also includes backends that this platform does not
support. Use `yuj doctor` or `yuj --dry-run` for the operational startup
check.

The shipped fixed measurement configurations continue to require `bwrap`.
Change that choice only through an explicit measurement configuration.

Legacy `tools.sandbox_bash = true` plus `tools.sandbox_required = true`
migrates to `bwrap`. The matching `false`/`false` pair migrates to `none`.
Mixed legacy booleans are invalid. A copied legacy full configuration may keep
a consistent backend and boolean pair in one layer. Contradictory pairs fail.
The legacy shared-container spelling migrates to its exact Docker or Podman
runtime. New settings files should use only `sandbox.backend`. A legacy
`YUJ_CONTAINER` mode cannot be combined with another selected backend.

`container_flags` accepts resource limits and metadata that does not change
execution. It rejects
flags that change mounts, networking, the environment, the entry point,
devices, privileges, or the security boundary. Read [Sandbox](sandbox.html)
before you enable this backend.

The same resolved policy runs foreground and background shell commands,
`run_tests`, post-edit checks, language servers, Python cells, and named-agent
sessions. `none` changes only the process-isolation boundary. Permission
rules, approvals, command and path validation, hooks, output handling, and
artifact ownership still apply.

### Control the command environment

The command environment is explicit and is resolved once before the first
session:

```toml
[sandbox.env]
inherit = "core"  # all | core | none
set = { PAGER = "cat", TERM = "dumb" }
ignore_default_excludes = false
allow_login_shell = false

[sandbox.env.filters]
"AWS_*" = "exclude"
# "PATH" = "include"
```

Yuj builds the command environment once, in this order:

1. Inherit names from `all`, `core`, or `none`.
2. Remove inherited names that contain `KEY`, `SECRET`, or `TOKEN`, unless
   `ignore_default_excludes` is true.
3. Apply the custom exclude patterns.
4. Apply the fixed values from `set`.
5. If any include pattern exists, keep only matching names.

Pattern matching ignores case. A fixed value can restore a name removed in
step 2, but it must still pass the final include list.

Keep `allow_login_shell = false` unless startup files are part of the run you
intend to test. Those files can change the environment after Yuj builds it.

Yuj gives the same mapping to shell calls, test runners, post-edit checks, and
language servers under every command backend. The mapping does not change the
harness process, so the model client can still read its provider credentials.
Saved provenance records variable names, not values, and redacts fixed values.
For a provider-scoped Claude or Codex API key, Yuj always removes that key's
environment name from this mapping, even when `ignore_default_excludes` or a
fixed value would otherwise restore it. Every provider-scoped Claude or Codex
session also removes `YUJ_AUTH_HOME`, so a model-issued command cannot discover
the managed credential directory through its environment.

### Hide repository paths from the model view

Yuj loads `.yujignore` from the task root by default:

```toml
[state]
ignore_file_enabled = true
ignore_file_names = [".yujignore"]
```

The file uses Gitignore syntax. It supports comments, `*`, `?`, character
classes, `**`, root anchors, directory rules, and later `!` exceptions. The
last matching rule in one file wins. If you list several ignore files, the
first file that decides a path wins.

Yuj loads the policy once, before the first model request. It applies the same
view to file tools, search tools, shell commands, tests, post-edit checks, and
language servers. Editing `.yujignore` during a run does not change that run.

This setting hides paths from the model. It does not delete them, and an
external test or scorer can still read them. The trace records the loaded file
names and a hash of their bytes, but not the patterns or hidden paths. Set
`ignore_file_enabled = false` when the model must see the whole repository.

## Isolate and restore work

### Isolate a session in a Git worktree

The default value, `off`, lets the model work in the current checkout. Use
`auto` to give the session its own retained Git worktree:

```toml
[runtime]
worktree = "auto"
```

`auto` creates `<repository>/.yuj_worktrees/<run-id>` on
`worktree-<run-id>`. You may give an explicit branch name instead. Yuj refuses
to create the worktree when the source checkout is dirty, the branch exists,
or the target path belongs to something else.

Yuj keeps the worktree after every exit so `resume` can return to the same
files. It records the path, branch, and base commit, and refuses a resume when
they no longer match.

An explicit `yuj fork SESSION` of a session that owns a managed worktree
creates a distinct retained worktree and branch for the child. It starts from
the source worktree's recorded `HEAD` and copies its current tracked,
uncommitted, and untracked files into child-owned files. The source worktree
keeps its path, branch, and bytes. Yuj refuses the fork before publishing a
child session when it cannot validate or create the independent worktree. A
source without a managed worktree does not gain one during a fork.

Remove an assistant worktree after you merge or save its work:

```bash
yuj worktree rm SESSION
```

The command refuses dirty files and unmerged commits. `--force` discards both.
Measurement runs do not use this cleanup command. Leave worktree mode off when
an external benchmark launcher already creates an isolated task copy.

### Save restorable file checkpoints

Enable file checkpoints when an operator may need to restore an earlier turn:

```toml
[tools]
file_checkpoints_enabled = true
```

After each shell or edit call that may have changed files, Yuj saves a commit
in a separate shadow Git repository. It also saves a checkpoint after a shell
call that fails or times out, because that call may still have changed files.
It does not checkpoint a call rejected before execution.

The shadow repository lives outside the task directory and does not change the
project's index, branches, or history. Model tools cannot read it. Restore is
an operator function, not a model tool.

Checkpoints include tracked files and unignored untracked files. They exclude
project `.gitignore` matches and `[tools].file_checkpoints_exclude` matches.
They preserve file bytes, executable bits, and symlink targets, but not owners,
ACLs, or extended attributes. Read [Saved files](harness_artifacts.html) for
the trace and cost records.

### Rewind conversation and files together

Conversation rewind restores the messages and files from the same completed
turn. It requires file checkpoints:

```toml
[loop]
rewind_enabled = true
rewind_max_per_session = 1

[tools]
file_checkpoints_enabled = true
```

At each completed turn, Yuj binds the exact model-facing messages to a file
checkpoint. An operator can then run `yuj rewind SESSION TURN`, or an enabled
guardrail can request a rewind. The target must be an earlier completed turn
in the current session. `rewind_max_per_session` limits all successful rewind
actions together.

Yuj restores the files at once. The next `yuj resume` restores the matching
messages before it contacts the model. The raw trace stays append-only: Yuj
adds a `rewind` event and rebuilds `.solver/state.json` to show the active
branch.

### Collapse a model exploration branch

This separate feature lets the model discard an unhelpful exploration from
its conversation while keeping a short report. Enable both model-facing tools
with one setting:

```toml
[tools]
checkpoint_enabled = true
```

The model calls `checkpoint(goal)` before it explores. The checkpoint becomes
active after every tool call in that turn has a result. A later checkpoint
replaces it.

The model can then call `rewind(report)`. After the current turn finishes, Yuj
returns the conversation to the checkpoint and adds:

```text
<rewind-report goal="GOAL">
SHORT FINDINGS REPORT
</rewind-report>
```

`rewind` returns `no_active_checkpoint` when the model has not set a
checkpoint. A successful rewind consumes that checkpoint. The raw trace keeps
the full exploration and report, while the
model-facing conversation and state view follow the selected branch. Replay
uses the same turn-boundary behavior.

This feature does not restore files. Any file changes made during exploration
remain in place. Use
[`file_checkpoints_enabled`](#save-restorable-file-checkpoints) when an
operator needs to restore the workspace too. The two rewind features share a
name, but their trace fields identify which one ran.

## Configure model tools

### Return language-server diagnostics after edits

Language-server support needs `[lsp].enabled = true` and at least one
`[lsp.servers]` entry. Each entry names a `command` and its `extensions`.
It may also name `root_markers` and `initialization` data.

```toml
[lsp]
enabled = true
diagnostics_timeout_s = 2.0
min_severity = "error"
tool_enabled = false
```

Yuj starts a server only after an edit touches a matching file. It runs the
server inside the selected shell sandbox, with networking disabled, and stops
it at the end of the run segment.

After a successful edit, Yuj adds matching diagnostics to the same tool
result. `min_severity` controls which diagnostics the model sees. The trace
still counts messages below that level. A timeout, missing binary, or failed
server produces a warning but does not fail the task.

Set `[lsp].tool_enabled = true` to let the model ask for definitions,
references, and document symbols. Install every language-server binary before
the run. Yuj never downloads one.

### Require current file evidence before editing

Set `[tools].stale_guard_mode` to control whether the model must have current
file contents before it edits:

| Mode | When the file was not read, or changed after the read |
| --- | --- |
| `off` | Run the edit without this check. |
| `warn` | Run the edit and add `WARNING: stale_file: read PATH first`. This is the default. |
| `block` | Refuse the edit with `ERROR: stale_file: read PATH first`. |

A successful `read` records the file's content hash and metadata. A successful
file edit refreshes that record. A simple, single-file `cat`, `head`, `tail`,
`sed -n`, `grep`, or `rg` command also counts as a read. Compound commands,
pipelines, redirects, recursive searches, and count-only searches do not.

Yuj rebuilds this session ledger from `stale_guard_observe` trace rows on
resume. It does not copy the ledger into `.solver/state.json`.

### Redirect shell commands to dedicated tools

Redirect rules stop shell commands when an active, dedicated tool owns the same
job. Yuj does not run or rewrite a matching command. It returns a
`redirect_rule` error that names the tool to use.

Write-side rules redirect in-place editors and shell output redirection to an
edit tool. Set `[tools].bash_redirect_read_side = true` to cover these read
commands too:

| Shell job | Dedicated tool |
| --- | --- |
| Display a file | `read` |
| Search text | `grep` |
| Find paths | `glob` |

A rule applies only when its target tool is active. It understands compound
commands and leading environment assignments, but it leaves input-consuming
pipe stages and count-only commands such as `grep -c` or `wc -l` alone.

### Defer tools until the model needs them

Deferred loading starts the model with a smaller tool set. The model can add
other enabled tools when it needs them. Enable it with:

```toml
[tools]
lazy_loading_enabled = true
active_default = ["bash", "read", "edit", "glob", "grep", "done"]
```

Yuj registers every enabled tool, but sends only `active_default`,
`load_tools`, and `done` on the first request. The top-level assistant session
also keeps the fixed `ask_user` control. An optional tool still needs its own
enable setting. The selected profile's `max_tools` limits the active set, and
Yuj rejects a load that would exceed it.

Deferred loading and code mode are two different ways to reduce tool schemas.
Yuj rejects a configuration that enables both.

The model activates hidden registered tools additively:

```json
{"names":["write","run_tests"]}
```

`load_tools` adds tools for the next model request. It never removes one. A
second call in the same response still uses the old tool set. A direct call to
a hidden tool returns `tool_not_active` and does not run.

Each new harness session starts from `active_default`. The trace records every
successful activation, and `.solver/state.json` shows the current set. Replay
checks the recorded tool-name order when the source transcript contains it.
`metrics.json` reports the size of the initial tool block.

### Search definitions and references across a repository

Turn on both settings to let `list_definitions` search a whole repository:

```toml
[tools.list_definitions]
enabled = true

[tools]
ast_search_enabled = true
```

The model sets `repo_wide = true`, uses `path` as the search root, and may
filter by an exact `symbol` and `kind = "def"` or `"ref"`. Results use the
stable form `path:line kind name signature` and support pages.
`ast_search_max_rows` caps the result set before paging.

Yuj ships local parsers for Python, JavaScript and TypeScript, Go, Rust, and
Java. It never downloads a grammar during a tool call. A missing parser
returns `backend_unavailable`. Unreadable-path rules still apply.

A call with only one file path keeps the existing Python outline behavior and
does not require structural-search support.

### Give the model a temporary scratchpad

The optional `think` tool gives the model a short-lived scratchpad. It does not
run a process or change a file:

```toml
[tools]
think_enabled = true
think_keep_turns = 4

[loop]
think_streak_nudge_after = 3
```

`think(thought)` returns an empty successful result. The raw trace keeps the
`thought` text for audit and replay. Model context removes the call and
result after `think_keep_turns`; state-backed views then keep only a `think()`
breadcrumb. The trace remains unchanged.

Yuj counts `think` as a non-write action. After
`think_streak_nudge_after` consecutive calls, it gives the existing rumination
nudge. Another tool resets the streak. Set either number to `0` to hide a
thought immediately or disable the streak nudge.

### Validate and constrain tool-call arguments

Both settings are off by default:

```toml
[tools]
schema_validation = "off"       # off | reject
constrained_decoding = "off"    # off | json_schema | grammar
```

Set `schema_validation = "reject"` to check each native tool call against the
schema sent to the model. Yuj rejects missing fields, wrong JSON types, and
undeclared fields before approval checks, guardrails, or the tool handler. The
model receives a repairable error, and the trace stores only the tool name and
field errors, not the rejected values.

The assistant-only `ask_user` control always validates its single required
`question` string and rejects every extra field, even when
`schema_validation = "off"`. It has no enable setting. Top-level assistant
requests keep it available in native mode, code mode, deferred loading, and
required plan mode. Child-agent and measurement requests remove it before
profile shaping and cannot finish with `input_required`.

`constrained_decoding` asks llama-server to constrain the tool-call span with
JSON Schema or a generated GBNF grammar. The selected model profile must also
set `[model].supports_constrained_tools = true`. Shipped profiles leave this
off until their exact server, template, reasoning mode, and tool wrapper have
been tested. Unsupported profiles send no constraint, but runtime schema
validation can still reject bad calls.

### Maintain a model todo list

The model-callable todo list is disabled by default. Enable it with a small
overlay:

```toml
[tools]
todos_enabled = true
todos_max_items = 20

[state]
todos_char_budget = 2000
```

`write_todos` accepts a complete list of `{description, status}` items. A
status must be `pending`, `in_progress`, `completed`, `cancelled`, or
`blocked`. The list may contain only one `in_progress` item and no more than
`todos_max_items` items. Each call replaces the full list; an empty list clears
it.

Yuj writes the list to the trace and projects the newest list into
`.solver/state.json`. The model never writes that file. State-backed context
modes show the list on later turns. `todos_char_budget` limits only the block
sent to the model; it does not cut the trace or saved state.

## Add policy and trusted automation

### Select a fixed assistant permission preset

Use a preset when you want a fixed starting policy without writing every rule.
Select one preset for a coding session:

```bash
yuj --permission-preset ask-before-changes "Inspect, plan, and make the change."
```

Save one as the machine setting with `yuj setup --permission-preset NAME`, or
put it in a settings file:

```toml
[assistant]
permission_preset = "ask-before-changes"
```

An empty value selects no preset and preserves the ordinary `loop.plan_mode`,
`permissions.ask_fallback`, and `permissions.rules` behavior. Yuj provides
these three presets. All three set `permissions.ask_fallback = "deny"`.

| Preset | Plan mode | Catch-all rule | Explicitly allowed groups | Common result for `read` / `edit` / `bash` after planning |
| --- | --- | --- | --- | --- |
| `read-only` | `off` | `deny` | Inspection and session control | `allow` / `deny` / `deny` |
| `ask-before-changes` | `required` | `ask` | Inspection and session control | `allow` / `ask` / `ask` |
| `allow-edits` | `off` | `ask` | Inspection, session control, and file edit | `allow` / `allow` / `ask` |

The groups expand to these exact tool names:

- Inspection: `read`, `glob`, `grep`, `list_definitions`, `list_functions`,
  `get_function_details`, and `lsp`.
- Session control: `ask_user`, `checkpoint`, `done`, `exit_plan_mode`,
  `load_tools`, `think`, and `write_todos`.
- File edit: `apply_patch`, `edit`, `udiff`, and `write`.

Yuj adds an `allow` rule for every tool in the listed groups after the preset's
catch-all rule. Every tool outside those groups keeps the catch-all result.

A preset expands only when `runtime.mode = "assistant"`. Measurement mode
validates the name but does not apply its plan or permission values.

Yuj expands the preset in the configuration layer before it compiles the
ordinary permission policy. An explicit `loop.plan_mode` or
`permissions.ask_fallback` value replaces the preset value. Yuj inserts the
preset rules first and then applies configured `permissions.rules` in their
existing order. A matching configured rule therefore wins, including an
explicit `deny` over a preset `allow`.

Plan mode still checks an action before permission dispatch. An `ask` still
uses the normal approval request. An `allow` still passes through command
checks and the sandbox. No preset changes or skips those checks. With
`ask-before-changes`, the `.solver/plan.md` write also matches `ask`, so Yuj
requests approval before it writes the plan.

Run `yuj config --permission-preset NAME` to inspect the result. The human and
JSON views show `assistant.permission_preset`, each expanded effective rule,
and the source of each value. They place expanded rules under
`permissions.preset_rules` and configured rules under `permissions.rules` so
their order remains visible. Expanded values use the
`assistant-permission-preset` source layer. Configured values keep the source
layer that supplied them.

### Apply per-tool permission rules

The default permission table is empty and therefore allows every call to
continue through Yuj's existing approval, guardrail, and bash-quirk layers:

```toml
[permissions]
ask_fallback = "deny"

[permissions.rules]

# Example overlay:
[permissions.rules.bash]
"*" = "ask"
"git *" = "allow"
"rm *" = "deny"

[permissions.rules.read]
"*" = "deny"
"docs/*" = "allow"
```

Yuj reads rules in TOML order, and the last match wins. Matching is
case-sensitive. `*` and `?` are the only wildcards, and both can match `/` or a
newline. Brackets are literal. A string such as `read = "deny"` means the same
as a `"*" = "deny"` rule for that tool. Use the global `"*"` tool entry for a
baseline, then override it with a later tool entry.

Each tool matches one stable value:

| Tools | Match value |
| --- | --- |
| `bash` | `cmd` |
| `read`, `write`, `edit`, `glob`, `grep`, `run_tests`, `list_definitions`, `lsp` | `path`; `glob` and `grep` use `.` when omitted |
| `apply_patch`, `udiff` | `patch` |
| `done` | `message` |
| `task` | `agent` |
| `think` | `thought` |
| `bash_poll`, `bash_kill` | `proc_id` |
| `write_todos` | The canonical todo array |
| Any other executable tool | The sorted JSON argument object |

`ask_user` is a session-control boundary, not an action permission. It does
not enter this match table or create `approval_request.json`. Its recorded
answer also does not change a permission decision. If the model later calls a
tool that matches `ask`, Yuj still creates the normal approval request and
waits for a separate `yuj approve` or `yuj reject` command.

In an assistant session, `ask` writes `approval_request.json` and pauses. Use
`yuj approve` or `yuj reject` to continue. `--always` stores the exact action
identity for any tool. If the session cannot write an approval request,
`ask_fallback` decides whether to allow or deny. Measurement runs always deny
`ask` and never create an approval request.

An `allow` rule does not bypass other controls. Yuj applies schema validation,
permissions, operator approval, shell redirects and refusals, then the tool
handler. An enabled `pre_tool` hook runs before this chain, so Yuj checks any
replacement arguments from the hook. Permission trace rows store the tool,
matched rule, and decision, but not the matched argument.

### Scan untrusted text for prompt injection

The checked-in default is a visible warning policy:

```toml
[security]
scan_mode = "flag"  # off | flag | block
patterns_file = "security/patterns.toml"
block_classes = [
  "destructive_command",
  "exfiltration",
  "prompt_injection",
  "invisible_unicode",
  "embedded_tool_call",
]
```

The modes differ only in what Yuj does after a match:

| Mode | Result |
| --- | --- |
| `off` | Do not load or apply the registry. |
| `flag` | Run the tool and add a value-free `<security-finding .../>` marker to the result. |
| `block` | Reject matches whose class appears in `block_classes`; flag other matches. |

An empty `block_classes` list makes `block` behave like `flag`. An argument
block happens before the tool runs. A result block happens after the tool runs,
so it can hide the result from the model but cannot undo side effects. Both
forms return `error_kind="security_block"`.

Yuj scans string values in tool arguments and raw tool results. At startup it
also scans these model inputs:

- resolved `--system-prompt` files;
- project instruction files;
- Agent Skills catalog metadata;
- enabled injection and stream-rule bodies; and
- harness pretest output.

A startup block stops before the first model request. Yuj does not scan the
task prompt, built-in system header, or trusted model-profile preamble. A skill
body read later passes through the normal tool-result scan.

The default local registry, `security/patterns.toml`, makes no model or network
request. Each `[[pattern]]` entry needs a unique lowercase `rule`, a lowercase
`class`, a Python `regex`, and one or both of the `args` and `result` stages.
Yuj rejects an unreadable registry, duplicate rules, bad expressions, and
expressions that match empty text. In `block` mode, every named block class
must exist in the active registry.

Each match writes `security_finding{id, rule, stage, action}` to the raw trace.
The event never stores matched text, argument values, result values, or source
paths. The related tool-call row carries the model-visible marker or error.

### Run trusted lifecycle hooks

Lifecycle hooks let an operator run an existing external program at a harness
event. They are disabled by default. Configure them in a small file passed
with `--config`:

```toml
[hooks]
enabled = true

[[hooks.pre_tool]]
matcher = "re:write|edit"
command = ["/opt/yuj-hooks/check-change", "--policy", "strict"]
timeout_s = 5
```

Each event accepts one handler or an ordered list. `matcher` defaults to `"*"`.
For tool events, it matches the tool name. For other events, it matches the
event name. Use a literal for an exact match, `"*"` for every match, or a
`re:` prefix for a full regular-expression match.

Give `command` as one executable path or an array of executable arguments.
Yuj does not start a shell or split a command string.

Yuj sends one JSON object to the handler's standard input. Every payload has
`event`, `run_id`, `session`, `turn`, `model`, and `profile_name`. Tool events
add the tool-call ID, name, and arguments. `post_tool` also adds the result.
`session_end` adds the finish reason, completion flag, and turn count. Treat
all model and tool data as untrusted input.

Handlers may return these effects:

| Handler result | Yuj behavior |
| --- | --- |
| Exit `0` with no supported JSON field | Allow the event. |
| Exit `2` | Block the event. JSON `reason`, `stopReason`, or `error`, or otherwise stdout text, becomes the explanation. |
| JSON `continue = false`, `decision = "deny"`/`"block"`, or nested `hookSpecificOutput.permissionDecision = "deny"`/`"ask"` | Block the event. |
| JSON `updated_input` or `updatedInput` from `pre_tool` | Replace the complete tool argument object. Yuj then applies schema validation, permissions, approval, and guardrails to the replacement. |
| JSON `additional_context` or `additionalContext` | Append the text in an `<injected-fragment source="hook">` envelope. `systemMessage` is accepted as the same annotation. |
| Timeout | Record `timeout`, warn, and fail open. Yuj kills the handler's process group. |
| Any other nonzero exit | Record `error`, warn, and fail open. |

Yuj runs matching handlers in their declared order until one blocks. A
`pre_tool` block stops the tool. A `post_tool` block replaces the returned
result but cannot undo the tool's side effects. A `done` block lets the model
continue when turns remain. An annotation from `session_end` stays in the
trace because no later model request can receive it.

Hooks run as host processes, outside every model-command sandbox. They receive
the Yuj process environment plus `YUJ_RUN_DIR`, `YUJ_RUN_ID`, `YUJ_TASK_CWD`,
and `YUJ_HOOK_EVENT`. Install and review the executable separately. Do not run
hook code from the task repository. Yuj rejects executable paths and path
arguments that resolve inside the task directory under every sandbox choice,
including `none`. Read
[Sandbox](sandbox.html#keep-lifecycle-hooks-outside-the-task) for this
boundary.

Each invocation writes a `hook` trace row with the event, command, exit status,
duration, outcome, and any accepted effect. Replay applies those recorded
effects and never starts the program. A different configured command at the
same replay position is an error. See
[Replay a saved run](replay_mode_spec.html#what-happens-on-each-turn) and
[Saved files](harness_artifacts.html).

## Run background work, agents, or code cells

### Run background commands

Enable background shell work and set its limits:

```toml
[tools]
background_enabled = true
background_max_procs = 4
background_poll_timeout = 300
```

The model can then call `bash` with `background = true`. Yuj returns a
session-local `proc_id` at once and exposes `bash_poll` and `bash_kill`.
`background_max_procs` limits live children. `background_poll_timeout` limits
one poll, even when the model asks to wait longer.

The command uses the selected shell sandbox and has no network access. Yuj
writes combined output to `.procs/<proc_id>.log`. Only new bytes returned by
`bash_poll` enter model context, after the normal filters, redaction, and output
limit. Yuj stops every remaining process group when the session ends. Read
[Saved files](harness_artifacts.html) for the trace boundary.

### Run named subagents

The `task` tool is absent unless an overlay enables it:

```toml
[tools]
task_enabled = true
subagent_depth = 1
subagent_max_turns = 20
```

A call uses `task(agent, prompt)`. The agent name selects
`agents/<name>.toml`. The descriptor names a model profile, complete tool
allowlist, Markdown system prompt, and turn limit. Yuj uses the smaller of the
descriptor limit and `subagent_max_turns`. See
[`agents/README.md`](https://github.com/sydches/yuj/blob/main/agents/README.md)
for the descriptor format.

The root session has depth `0`, and each nested `task` call adds one. At
`subagent_depth`, Yuj removes `task` from the child and rejects a direct nested
call. Agents are read-only unless their descriptor explicitly allows writes.
A read-only agent may run only a small, fail-closed set of inspection commands.

Children run one at a time in the parent's task directory and use the same
sandbox policy. Each child gets a fresh conversation and model client. The
parent receives only its final text. Yuj saves a separate child trace and adds
child use to the normal token totals. Replay returns the saved result instead
of calling the child model again.

### Run a sandboxed Python cell

Code mode is off by default:

```toml
[tools]
exec_cell_enabled = true
exec_cell_timeout = 30
```

Code mode replaces the native tool set with `list_functions`,
`get_function_details`, `exec_cell`, and `done`. The model first lists the
available functions, then asks for only the schemas it needs.

`exec_cell` runs Python with `read`, `grep`, `glob`, `list_definitions`, and
`bash` injected as functions. Each one returns text through the normal tool
dispatcher. The model must print the text that it wants the cell to return.
Permissions, ignore rules, shell rules, output filters, and redaction still
apply. Repository-wide definition search still needs `ast_search_enabled`.

The Python process uses the resolved command policy. With `bwrap`, Docker, or
Podman, it stays inside that sandbox. With explicit `none`, it runs as the Yuj
account. A requested sandbox that cannot start fails before the cell runs.
The timeout covers the whole cell and every inner call. A cell cannot start a
background command.

The raw trace stores the accepted source and output sizes. It records every
injected function call as a child tool call. The state writer projects those
calls as ordinary tool steps but never runs the cell again. Yuj keeps the
complete four-tool code-mode set even when the model profile sets a
smaller native-tool cap.

## Route model requests

### Declare image input support

Yuj rejects image input unless it knows that the selected model accepts it.
Each model profile inherits this default:

```text
[model]
supports_image_inputs = false
```

Yuj recognizes these directly hosted model families:

| Service | Recognized image-capable model IDs |
| --- | --- |
| OpenAI | General-purpose `gpt-4o`, `gpt-4.1`, and GPT-5 models; full `o1` and `o3` models; `o4-mini`; and their dated snapshots |
| Anthropic | Claude 3 and Claude 4 model families |

Yuj does not infer support from a shared prefix for text-only or specialized
models. For example, it rejects `o1-mini`, `o3-mini`, and GPT-4o audio,
realtime, search, and transcription models. Yuj treats every other provider
and model ID as unsupported unless its selected profile sets
`supports_image_inputs = true`.

Use the explicit profile declaration only after testing that exact model and
OpenAI-compatible endpoint. The declaration applies to request content, not
authentication: API-key and subscription access through the same transport
receive the same image blocks. A false or missing declaration stops an image
task before model work. It does not change text-only requests.

An image-bearing session stays on the selected primary model target. It does
not enter a configured model fallback chain, because a fallback must not drop
the saved visual evidence or send it to an unchecked transport. Text-only
sessions keep the configured fallback behavior.

### Configure auxiliary model roles

Named roles let Yuj send side requests to another model without changing the
model that works on the task. Yuj defines two roles: `weak` and `editor`.

Use a profile name when the role shares the main endpoint:

```toml
[models.roles]
weak = "qwen3-small"
```

Use an inline target when a role has its own served model or endpoint:

```toml
[models.roles.weak]
profile = "qwen3-small"
endpoint = "http://127.0.0.1:8181/v1"
model = "served-small"
context_size = 32768
```

`endpoint` must be an absolute HTTP or HTTPS URL with no embedded credentials.
An inline target may also set `api_key`, but prefer an environment-backed key
in ignored local configuration. Yuj validates every configured role before the
first model request.

Checkpoint summaries, fresh-session handoffs, and the model-backed hurdle
classifier use `weak`. A blank role uses the actual main client. Yuj creates a
role client only when something uses it, then reuses that client. Yuj does not
start another server, so run a second server yourself when the role uses a
different endpoint.

`metrics.json` charges each response once to its effective role under
`metrics.tokens_by_role`.

### Run a passive second-opinion advisor

The advisor is off by default. Enable it in a small overlay:

```toml
[advisor]
enabled = true
model = "served-review-model"
endpoint = "http://127.0.0.1:8181/v1"
every_n_turns = 5
immune_turns = 3
max_note_chars = 1200
```

An empty `model` uses `[model].name`, and an empty `endpoint` uses
`[server].base_url`. The advisor shares the main model profile and API key, so
a separate endpoint must accept the same message and tool-call format. Yuj
does not start or schedule that server. Frequent reviews or two servers on one
GPU can add substantial delay.

Each review starts an isolated conversation. The advisor receives only a
limited view of the completed primary turn: assistant reasoning, new tool
calls, and limited result rows. It does not receive the task prompt, primary
transcript, trace, state file, or other harness records. It may inspect visible
task files with `read`, `grep`, and `glob`. A visible root `WATCHDOG.md` may add
up to 16,000 characters of repository-specific review guidance.

The advisor must call `advise({severity, note})` or return exactly
`NO_ADVISORY`. It has no shell or mutation tool. Yuj quarantines fabricated
tools, mixed calls, malformed input, free-form notes, bad severity values, and
notes above `max_note_chars`. It also removes duplicate notes. After Yuj accepts
a note, `immune_turns` delays the next eligible review. Cadence and cooldown
continue across fresh context sessions in the same run.

Yuj inserts an accepted note into the next model request as:

```text
<injected-fragment source="advisor" severity="concern">
Concise, actionable note.
</injected-fragment>
```

Yuj places an accepted note in the next model request. Projection modes keep it
until one request succeeds; append-log modes keep the normal conversation
history. The raw trace stores the note's severity, size, source turn, order,
and hash, but not its text. `advisor.jsonl` stores the isolated review and note
text. `.solver/state.json` stores neither. Replay never runs the advisor.

### Configure model fallback

Fallback is off by default. Each role's chain is an empty list until you opt
in. A string entry uses exact `<profile>@<endpoint>` syntax:

Provider-scoped Claude and Codex sessions keep fallback off even when this
table configures one. An authentication or provider failure must not change
their provider, account, credential, model, endpoint, or billing method.

```toml
[models]
fallback_revert = "never"

[models.fallback_chain]
main = ["qwen3-small@http://127.0.0.1:8181/v1"]
weak = []
editor = []
```

An inline target may also set `model`, `context_size`, or an endpoint-specific
`api_key`. Yuj uses credentials for requests but excludes them from trace and
provenance artifacts:

```toml
[[models.fallback_chain.main]]
profile = "qwen3-small"
endpoint = "http://127.0.0.1:8181/v1"
model = "served-small"
context_size = 32768
```

Yuj validates every fallback profile at startup. It first spends the normal
retry budget on the active target. It advances the chain only after an eligible
connection, timeout, server, out-of-memory, or context-overflow failure.
Authentication errors, other bad requests, bad profiles, and tool-protocol
errors remain fatal.

Before Yuj uses a candidate, it asks for the live context window, applies the
candidate profile, and checks whether the prompt fits `context_fill_ratio`.
It skips and traces a candidate that cannot fit. When a candidate fits, Yuj
switches the client, profile, context limits, tool schemas, and token estimator
together, then gives it a fresh retry budget.

`fallback_revert = "never"` keeps the selected target for later sessions.
`next_session` returns to the primary target when the next session starts.
Every transition changes the run treatment, so Yuj records it in the trace and
in post-run fallback metrics. Studies can use those fields to exclude runs that
changed models.

## Tune model requests

### Configure llama-server prompt caching

The `[server]` cache settings apply to the OpenAI-compatible llama-server
client. They do not describe provider TTLs.

Use the cache settings only with the OpenAI-compatible llama-server client:

| Setting | What it controls |
| --- | --- |
| `request_extra` | Extra JSON body fields sent through the OpenAI SDK's `extra_body`. |
| `cache_affinity = false` | Do not select a slot. |
| `cache_affinity = true` | Select slot 0. |
| `cache_affinity = N` | Hash the stable product session ID across `N` slots. |
| `cache_retention = "off"` | Send `cache_prompt=false`. |
| `cache_retention = "session"` | Send `cache_prompt=true` on normal model turns. |
| `cache_miss_warn_ratio` | Warn after the first request when the observed hit ratio falls below this value. `0` disables the warning. |

Do not configure more affinity slots than the server exposes. Cache policy
owns `cache_prompt` and `id_slot`, so it overrides those fields in
`request_extra`.

Compaction, handoff, and other Yuj-owned side requests always disable prompt
retention and omit the slot. Missing server telemetry remains unknown and does
not produce a false warning. The trace stores per-turn prompt and cached token
counts. `metrics.json` reports the token-weighted run ratio.

### Count tokens

The checked-in `config.toml` leaves `[model].tokenizer_id` empty. Yuj then
estimates one token for every four characters, which avoids a model-specific
download during normal use.

Released paper runtime files set the tokenizer for each reported model. Apply
the files in the
[paper configuration guide](https://github.com/sydches/yuj/blob/main/configs/paper/README.md)
when you reproduce an experiment.

### Select reasoning effort

Set `[model].thinking_level` or pass `--thinking` with one of `off`,
`minimal`, `low`, `medium`, `high`, `xhigh`, or `max`. The checked-in default
is `off`.

Each model profile maps its supported levels to request fields under
`[reasoning_levels.<level>]`. The base profile maps `off` and `on` through
`chat_template_kwargs.enable_thinking`. A model-specific profile may use fields
such as `reasoning_effort` or `thinking_budget` instead.

When a profile does not support the exact level, Yuj warns and chooses the
closest supported effort that does not exceed the request when possible. A
boolean profile maps every positive level to `on`. Yuj applies the effective
level to normal model requests and forces thinking off for harness side
requests.

Profile server launch fields do not replace this per-request setting. The
trace and run provenance record the requested and effective levels and whether
Yuj clamped the request.

## Choose a context mode

Context is the text that Yuj gives the model before its next action.

The selected base chooses the normal mode:

| Base | Context mode | What the mode does |
| --- | --- | --- |
| Treatment | `halflife` | Keep messages in time order. Shorten the bodies of older tool results when the input nears its limit. |
| Plain | `full` | Keep the full in-memory message log. |

Override the base choice when you need another mode:

```bash
yuj --context full "Fix the issue."
```

Yuj registers these modes:

| Source | Modes | What Yuj reads |
| --- | --- | --- |
| In-memory messages | `full`, `compact`, `yuj`, `halflife` | Messages kept by the active process. |
| In-memory messages and current files | `concise`, `slot` | An in-memory working set and the current contents of files that the model touched. |
| Messages, saved state, and current files | `yconcise`, `yslot` | An in-memory working set, `.solver/state.json` when present, and the current contents of touched files. |
| Saved state | `stateful`, `compound`, `focused_compound`, `compound_selective`, `salience` | `.solver/state.json` and a small in-memory window of recent tool results. On resume, Yuj also loads the current contents of files named by the saved run. |

Normal context modes do not read `.trace.jsonl` directly.

All modes apply `[tools].think_keep_turns` to the optional scratchpad tool.
Expired thought arguments leave model-facing messages, compact progress rows,
working-set artifacts, recent-result windows, and state-derived sections even
though their raw `tool_call` rows remain in the append-only trace.

Transcript files record model request and response data. Normal context modes
do not read them.

Before `.solver/state.json` is ready, `stateful`, `compound`,
`focused_compound`, `compound_selective`, and `salience` can use in-memory
messages instead. They also use messages when the file is absent or the
settings say to ignore it. During this fallback, Yuj uses either messages or
saved state. It does not mix them.

`yconcise` and `yslot` are different. They combine messages and saved state on
purpose.

A context mode changes what the model can see. Record the mode when you compare
sessions.

### Add a ranked repository map to the task

Set a token budget to add one compact `<repo-map>` block after the task text:

```toml
[context]
repo_map_tokens = 1024
repo_map_refresh = "auto"
```

The default budget is `0`, which adds no map. Yuj uses local parsers for
Python, JavaScript and TypeScript, Go, Rust, and Java. It ranks definitions and
references from paths named in the task, with stable tie-breaking. No model
writes or ranks the map.

The budget counts only the added block. Yuj uses `tokenizer_id` when set,
otherwise the active profile estimator and then the one-token-per-four-
characters fallback. If even the wrapper and top definition do not fit, Yuj
omits the map.

The map stays fixed for one harness session and remains in the task prefix
through compaction. A later session may build a new map. `repo_map_refresh`
controls the symbol cache at that boundary:

| Value | Cache behavior at session start |
| --- | --- |
| `auto` | Reuse while supported source paths, byte sizes, and nanosecond mtimes match. |
| `always` | Parse the readable source tree again and replace the cache. |
| `files` | Content-hash every supported source file before deciding whether to reuse. |
| `manual` | Reuse any valid cache for the same repository; parse only when no valid cache exists. |

Yuj stores the cache with private run artifacts, outside the model's file root,
and masks it from shell commands. It applies unreadable-path and `.yujignore`
rules before it fingerprints or parses files. The trace records the token
count, content hash, refresh mode, file and symbol counts, and cache hit. It
does not record the map body or cache path.

### Compact a nearly full context

Compaction runs only after the existing context threshold and mutation gate
allow it. These settings live under `[context]`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `compaction_method` | `"digest"` | Use the deterministic trace digest, or opt into a model-written `"checkpoint"`. |
| `compaction_hook` | `""` | Trusted synchronous `module:function` called after the normal threshold and mutation gate. Empty disables the hook. |
| `checkpoint_keep_recent_tokens` | `0` | Verbatim recent-tail target. Zero means 20% of the live context window, with a 4,096-token minimum. |
| `checkpoint_max_summary_tokens` | `4000` | Maximum checkpoint response; the runtime also applies a 4,000-token hard cap and the available-reserve limit. |
| `digest_compaction_safety_margin` | `0.05` | Margin used by the derived compaction threshold. |
| `digest_keep_recent_turns` | `8` | Digest tail size and the close-compaction guard window. |
| `digest_compaction_gate_min_mutations` | `0` | Minimum successful mutations before compaction may run. |

`digest` builds a deterministic summary from trace facts. `checkpoint` makes
one no-tool request through the `weak` model role with thinking off. Yuj keeps
the system prompt and task unchanged, inserts the validated checkpoint after
the task, and keeps a recent tail that starts at an assistant-turn boundary.
The checkpoint must use the required sections, name every observed modified
path, fit the budget, and make the prompt smaller. Any failure falls back to
`digest`.

`compaction_hook` names trusted in-process Python. Yuj imports it before model
work starts. The hook may use the built-in method, cancel one attempt, or
return a replacement that passes the same checks. A hook error or bad result
falls back to `digest`. Read [Compaction hooks](compaction.html) for the Python
contract. The model-command sandbox does not isolate this code.

The trace and state store compaction metadata, not model-written summary text.
If two compactions occur within `digest_keep_recent_turns`, Yuj uses `digest`
for the later attempt to avoid a loop.

### Summarize work for a fresh session

Set `[loop].handoff_summary_enabled = true` to ask the `weak` role for a
summary when a session ends at `context_full`, `length`, or `max_turns` and
another session can run. `[prompts].handoff_max_tokens` limits the response and
defaults to `2000`. Yuj turns thinking off for this request.

Yuj checks the required sections, size, and every modified path in the trace.
It places a valid `<handoff>` after the task and before the existing resume
tail. If the request or any check fails, Yuj leaves that tail unchanged. The
trace records status and token metadata, not the summary text, and
`.solver/state.json` does not store the summary.

### Recover an interrupted tool turn

The default, `[loop].interrupted_turn_mode = "mechanical"`, repairs a trace
that ended during a tool turn. Before resume reads the trace, Yuj removes only
a malformed final JSON fragment and adds `turn_aborted`. The next user message
names each tool call that started without a durable result and marks its
outcome as unknown. Yuj closes any dangling transcript edge with a synthetic
result. It never reruns the call or guesses whether its file effects happened.

Before each dispatch, Yuj saves and syncs a small `tool_start` event. Normal
exit paths also record pending calls in `session_exit`. A hard kill may skip
the exit handler, but the durable start remains.

Set the mode to `off` to disable repair. Resume then leaves a malformed suffix
in place, so normal trace loading may reject the trace or stop at that suffix.

### Continue a length-limited response

The default, `[loop].length_continue_max = 0`, makes one request per model
turn. Set a positive limit to continue the same turn when the server returns
`finish_reason = "length"`. The active profile must also set
`[model].supports_prefill = true`.

Each follow-up starts from the original prepared request and adds the full
partial assistant response. Yuj removes only an exact overlap between pieces.
It does not change whitespace or guess at near matches. For llama-server, the
request uses `continue_final_message = true` and
`add_generation_prompt = false`. Yuj never sends those fields through the
default path, an unsupported profile, a legacy client, or replay.

Some providers return an incomplete function call as structured `tool_calls`.
For that shape, Yuj may repeat the exact prefix with a larger cumulative output
cap until the argument becomes complete. It never exceeds the context room
reported by the previous call.

If the allowed follow-ups still do not complete the response, the turn remains
length-limited and uses the normal fresh-session rollover. The trace records
attempt numbers and completion-token counts, not request or response text.
Metrics count the extra requests, while normal token totals include them.

### Load conditional injection rules

Injection rules are off by default. Enable the rule loader and path matching
with a small overlay:

```toml
[injections]
enabled = true
dir = ".harness/injections"
path_rules_enabled = true
path_rule_repeat = false
```

Yuj loads each `*.md` file in alphabetical filename order before the model
starts. A malformed enabled file stops startup. Each file uses strict TOML
frontmatter between `+++` fences:

```markdown
+++
name = "python-tests"
paths = ["src/**/*.py", "tests/**/*.py"]
keywords = ["pytest"]
repeat = false
+++

Follow this repository's Python and test conventions.
```

The frontmatter fields have these jobs:

| Field | Rule |
| --- | --- |
| `name` | Required nonempty rule name. |
| `paths` | Optional list of project-relative POSIX globs. |
| `keywords` | Optional list of nonempty strings. |
| `repeat` | Optional boolean. `false` is the default. |

A rule with no `paths` or `keywords` is unscoped, so Yuj adds it once at
session start. The older `trigger` and `fire_once` fields remain readable. Do
not combine `repeat` and `fire_once`.

Path globs support `*`, `?`, character classes, and slash-aware `**`. Yuj
rejects absolute paths, parent traversal, directory-only patterns, empty
entries, and non-string entries. It checks both the model's path and the
symlink target, then reports only the normalized task-relative path.

A path rule may fire after a file tool runs or after Yuj proves that a simple
`cat`, `head`, `tail`, `sed -n`, `grep`, or `rg` command read one file.
Compound commands, pipelines, redirection, recursive searches, multi-file
reads, and rewritten commands do not count.

By default, the first path or keyword match consumes the rule. Set the rule's
`repeat = true` to let either trigger repeat. `path_rule_repeat = true` changes
only the default for path rules that omit `repeat`.

A path fire appends this shape to the same model-visible tool result:

```xml
<injected-fragment rule="python-tests" trigger="path" path="src/app.py" source="python-tests">
Follow this repository's Python and test conventions.
</injected-fragment>
```

Each fire writes `rule`, `trigger`, and `path` to an `injection` trace row.
Keyword rows use an empty path. The metadata does not enter
`.solver/state.json`; the visible fragment remains part of the ordinary tool
result. Startup logs name armed rules and trigger types without copying their
bodies.

### Load project instruction files

Repository instruction discovery is an opt-in prompt treatment. The public
defaults preserve bench parity:

```toml
[prompts]
project_docs_enabled = false
project_doc_names = ["AGENTS.md", "CLAUDE.md"]
project_doc_max_bytes = 32768
project_root_markers = [".git", ".hg", ".sl"]
project_doc_global_dir = "~/.config/yuj"
imports_enabled = true
imports_max_depth = 5
```

When enabled, Yuj finds instruction files in this order:

1. Check the global directory, unless `project_doc_global_dir` is blank.
2. Find the nearest directory that contains a configured project-root marker.
3. Walk from that root down to the task working directory.

At each location, Yuj loads at most one nonempty file.
`AGENTS.override.md` wins first. The names in `project_doc_names` follow in
their listed order. An empty file lets Yuj try the next name.

With `imports_enabled = true`, a standalone `@path/to/file.md` line imports
another Markdown file. This works in `--system-prompt` files, selected project
instructions, and enabled injection fragments. A relative path starts beside
the file that contains it. Direct imports have depth 1, and
`imports_max_depth` limits nesting. Directives inside code blocks or inline
code remain literal. Set `imports_enabled = false` to leave every import line
literal.

An import must stay inside the boundary owned by its source:

| Source | Allowed import area |
| --- | --- |
| System-prompt file | Its own parent or the project root |
| Project instruction | The project root |
| Global instruction | The global instruction directory |
| Injection fragment | The project root |

Symlinks cannot escape these areas. Unreadable-path rules apply before Yuj
reads a file. A missing, cyclic, too-deep, unreadable, non-Markdown, or outside
import becomes a short `yuj-import-error` HTML comment. Yuj does not put host
exception text in the prompt.

After imports expand, `project_doc_max_bytes` caps the complete selected
instruction chain in UTF-8 bytes. Yuj cuts only at a valid character boundary.
It wraps each selected file in a `<project-instructions>` block with a safe
project-relative or `global/<name>` label.

Yuj builds the prompt in this order: resolved `--system-prompt`, project
instructions, `prompts.system_header`, then the optional Agent Skills catalog.
The model-profile preamble remains outside that sequence. Run-start provenance
stores safe source labels, import status, depth, and byte counts, but no prompt
body or absolute path. `metrics.json` stores only the final prompt hash and
size. Because these files change model input, record this setting as part of
any comparison.

### Load Agent Skills on demand

Yuj supports [Agent Skills](https://agentskills.io/specification) directories.
This is progressive disclosure: startup reads and validates only YAML
frontmatter, then adds each model-invocable skill's name, description, and
absolute `SKILL.md` path to a `<skills>` system-prompt block. The Markdown body
and bundled scripts, references, and assets stay out of the prompt until the
model reads them.

The feature is off by default. Enable it with a small overlay:

```toml
[prompts]
skills_enabled = true
skills_dirs = [
  "~/.pi/agent/skills",
  "~/.agents/skills",
  ".pi/skills",
  ".agents/skills",
]
skill_paths = ["/opt/team-skills/release/SKILL.md"]
```

Yuj discovers skills in two layers:

1. Read `skill_paths` in order. An entry may name `SKILL.md` or its directory.
   A missing exact path stops startup.
2. Search `skills_dirs` in order. A missing collection is ignored. A relative
   collection is searched from the task directory up to the nearest project
   root. An absolute or expanded path names one collection directly.

Collection search is deterministic and stops after six directory levels or
2,000 directories. A skill must live in a subdirectory that contains a file
named exactly `SKILL.md`. Name a collection-root skill explicitly in
`skill_paths`.

If two valid skills have the same frontmatter `name`, Yuj warns and keeps the
first. Exact paths therefore take priority over collection discovery.

Yuj checks the frontmatter before the first model request:

| Field | Rule |
| --- | --- |
| `name` | Required; at most 64 characters; lowercase letters, digits, and single hyphens; must match the parent directory. |
| `description` | Required, nonempty, and at most 1,024 characters. |
| `license`, `allowed-tools` | Optional nonempty strings. |
| `compatibility` | Optional nonempty string of at most 500 characters. |
| `metadata` | Optional map from string keys to string values. |
| `disable-model-invocation` | Optional boolean. |

`disable-model-invocation = true` leaves the skill out of the model catalog.
The trace still records it, and an explicit path can still be read.
`allowed-tools` is metadata only. A skill cannot weaken permissions,
approvals, the sandbox, or the profile tool set.

The catalog tells the model to read the listed absolute `SKILL.md` path and to
resolve resources from that skill's directory. Yuj gives external skill
directories read-only access through `read`, `bwrap`, and the first-class
container backend. File mutation tools reject those external paths. A skill
inside the task directory follows normal task rules and may remain writable.
Unreadable-path rules apply to external skills, and `.yujignore` applies to
project skills. Yuj skips a masked discovered skill, but treats a masked exact
path as a startup conflict.

Run-start provenance stores each loaded skill's canonical path and
`disable_model_invocation` value, but not its description or body. Resolved
configuration lists the read-only skill directories. The savings ledger counts
only the visible catalog text.

Apply the overlay with the normal CLI command:

```bash
yuj --config skills.toml "Prepare the release."
```

All profiles use the same discovery, catalog, `read` tool, and sandbox rules.
When a profile caps tool count, `read` gets first priority after `done` if any
skill loads. A cap too small to keep `read` stops startup. Yuj adds no separate
skill-loader tool. These settings apply at run start and cannot change after
Yuj fixes the catalog and readable roots.

## Apply a small TOML file

Change only the values that you need:

```toml
# longer-task.toml
[loop]
max_turns = 320

[tools]
bash_timeout = 240
```

Apply the file after the selected base:

```bash
yuj --config longer-task.toml \
  "Complete the migration and run its tests."
```

Do not copy all of `config.toml` to change two values. A small file shows the
change. It also lets later Yuj defaults still apply.

Repeat `--config` to apply more files. Yuj applies them from left to right.

## Use another model service for one session

Set the key before you start the session:

```bash
export ANTHROPIC_API_KEY='...'
yuj --provider anthropic --model YOUR_MODEL_ID \
  "Fix the failing test."
```

Use `--base-url` with `--provider custom`.

Use `--api-key-env NAME` when the key uses another variable name.

These options affect only the new session. Yuj saves them in the session's
`provider.toml`.

When you use `--api-key-env`, the file stores the variable reference. It does
not store the key.

If you give `--base-url` or `--api-key-env` without `--provider`, Yuj uses a
`custom` OpenAI-compatible connection for that coding session.

## Environment variables

These variables form the ordinary-user interface. The serving and replay
guides name their own special variables where needed.

| Variable | What it changes |
| --- | --- |
| `YUJ_CONFIG` | Use this exact main settings file and its parent as the logical runtime-resource root. |
| `YUJ_CONFIG_LOCAL` | Use this exact optional machine-local settings file. |
| `XDG_CONFIG_HOME` | Own installed-package machine settings at `$XDG_CONFIG_HOME/yuj/config.local.toml` when `YUJ_CONFIG_LOCAL` is unset. |
| `YUJ_AUTH_HOME` | Use this exact provider-credential directory. Keep it outside every target repository. |
| `HARNESS_ASSIST_HOME` | Use this exact assistant session-state root. |
| `XDG_STATE_HOME` | Own installed-package session state at `$XDG_STATE_HOME/yuj` when `HARNESS_ASSIST_HOME` is unset. |
| `YUJ_CONTAINER=ambient` | Use the current outer container as the shell boundary. |
| `YUJ_CONTAINER=<container-id>` | Run model shell commands in this existing task container. |
| Model-service key variables | Supply a key named by `$ENV:NAME`, such as `OPENAI_API_KEY`. |

These process controls are optional:

| Variable | What it changes |
| --- | --- |
| `YUJ_STREAMING=1` | Read model replies as a stream. Streaming is off by default. Enabled stream rules can close and retry a matching response only in this mode. |
| `YUJ_PERSISTENT_BASH=0` | Start a new shell process for each `bash` tool call. Yuj normally reuses one eligible `bwrap` shell during a run segment. |

Read [Treatment](treatment.html) for treatment settings. Read
[Sandbox](sandbox.html) for shell access. Read the
[CLI reference](using-yuj.html) for every command-line option.
