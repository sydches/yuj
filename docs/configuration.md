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

Activate the Yuj virtual environment before you run a command on this page.
Otherwise, replace `yuj` with `/path/to/yuj/.venv/bin/yuj`.

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

Run `yuj doctor` after you change the service or model.

### `--provider` settings

The `--provider` setting tells Yuj how to reach the service that runs the
model.

| Setup name | API format | Address saved by setup | Usual key variable |
| --- | --- | --- | --- |
| `local` | OpenAI-compatible | `http://localhost:8080/v1` | No variable. Yuj uses the key value `local`. |
| `openai` | OpenAI-compatible | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `anthropic` | Anthropic Messages | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` |
| `openrouter` | OpenAI-compatible | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `zai` | OpenAI-compatible | `https://api.z.ai/api/paas/v4` | `ZAI_API_KEY` |
| `custom` | OpenAI-compatible | value from `--base-url` | value from `--api-key-env` |

Run `yuj setup` with no options when you want an interactive setup.

Prefer `--api-key-env NAME`. Yuj saves `$ENV:NAME` and reads the key when a
command starts.

`--api-key VALUE` saves the key itself in `config.local.toml`. Git ignores
this file.

Use `--force` to replace an existing file without an interactive question.

On `code`, `run`, and `smoke`, `--provider local` changes only the API format.
It keeps the loaded base address and key. Run `yuj setup --provider local` to
save the localhost address and `local` key.

## Understand setting order

Yuj calls the starting group of settings a base.

Yuj applies settings in this order:

1. `config.toml` supplies the checked-in defaults.
2. `config.local.toml` supplies this machine's service and model.
3. `--treatment` or `--no-treatment` selects a base.
4. Each `--config` file applies from left to right.
5. Model and model-service options on the command apply last.

A later value replaces an earlier value for the same field.

This order keeps local keys out of Git. It also lets you change one setting
without copying the full main file.

A paper comparison does not rely on the CLI defaults alone. Follow the exact
order in the
[paper configuration guide](https://github.com/sydches/yuj/blob/main/configs/paper/README.md).

## Change common settings

| Setting | Change it when you need to |
| --- | --- |
| `[server].provider` | Select the OpenAI-compatible or Anthropic API format. |
| `[server].base_url` | Use another online model service or local model server. |
| `[server].api_key` | Use a key value or an `$ENV:VARIABLE_NAME` reference. |
| `[server].request_extra` | Add llama-server JSON request fields through the OpenAI SDK `extra_body`. |
| `[server].cache_affinity` | Pin one product session to a deterministic llama-server slot. |
| `[server].cache_retention` | Turn explicit per-request prompt-cache writes off or on for the session. |
| `[server].cache_miss_warn_ratio` | Warn after the first turn when observed prefix reuse falls below this ratio. |
| `[model].name` | Set the exact model ID that the service accepts. |
| `[model].profile_name` | Use settings for a model's message and tool-call format. |
| `[model].context_size` | Give Yuj an input limit when the service does not report one. |
| `[model].tokenizer_id` | Count tokens with this Hugging Face tokenizer. Leave it empty to estimate the count from text length. |
| `[model].thinking_level` | Select the requested per-run reasoning effort. |
| `[models.roles].weak` | Choose an optional profile or endpoint for summaries and classifiers. Empty uses the main model. |
| `[models.roles].editor` | Choose an optional profile or endpoint for edit-focused side work. Empty uses the main model. |
| `[models.fallback_chain].main` | Opt into ordered replacement profiles/endpoints after eligible main-model failures. Empty by default. |
| `[models].fallback_revert` | Keep a selected fallback, or return to the primary target at the next session. |
| `[loop].max_turns` | Limit the number of model tool-call turns in one session. |
| `[loop].interrupted_turn_mode` | Repair an interrupted trace and resume without replaying a dangling tool call. Defaults to `mechanical`. |
| `[loop].handoff_summary_enabled` | Ask for a validated summary before an eligible fresh-session rollover. Off by default. |
| `[prompts].handoff_max_tokens` | Bound the optional model-written rollover summary. |
| `[tools].bash_timeout` | Limit the time for one shell command. |
| `[tools].sandbox_bash` | Turn the shell sandbox on or off. |
| `[tools].sandbox_required` | Stop if Yuj cannot start the selected shell sandbox. |
| `[sandbox].backend` | Select the first-class `bwrap` or `container` command backend. |
| `[sandbox].container_runtime` | Select `docker` or `podman` for the container backend. |
| `[sandbox].container_image` | Name an already-local trusted image. Required for the container backend. |
| `[sandbox].container_flags` | Add allowlisted resource or metadata flags without weakening isolation. |
| `[tools].file_checkpoints_enabled` | Capture an independent workspace snapshot after every potentially mutating model tool call. Off by default. |
| `[tools].file_checkpoints_exclude` | Keep harness output and other declared relative paths outside checkpoint and restore scope. |
| `[tools].stale_guard_mode` | Apply the session read-before-edit ledger as `off`, `warn`, or `block`. The default is `warn`. |
| `[tools].bash_redirect_read_side` | Redirect eligible `cat`/search/path-discovery shell reads to bounded dedicated tools. Off by default. |
| `[tools].ast_search_enabled` | Add repository-wide definition/reference lookup to `list_definitions`. Off by default. |
| `[tools].ast_search_max_rows` | Cap the rows available to one repository structural query before pagination. |
| `[lsp].enabled` | Start a configured language server lazily after the first matching `edit` or `write`, and return diagnostics. Off by default. |
| `[lsp].servers` | Declare language-server commands, file extensions, project-root markers, and optional initialization data. |
| `[lsp].diagnostics_timeout_s` | Limit how long an edit waits for a diagnostics publication. |
| `[lsp].min_severity` | Choose the least-severe diagnostic shown to the model: `error`, `warning`, `information`, or `hint`. |
| `[lsp].tool_enabled` | Add the optional `lsp` navigation tool. Off by default. |

The checked-in [`config.toml`](https://github.com/sydches/yuj/blob/main/config.toml)
defines and comments the full public settings shape. Use a small settings file
for changes. Do not copy the full file into your project.

Read [Model tools](model-tools.html) for optional tool settings. Read
[Run a local model](serving_overlay.html) for runtime files and profiles.
Read [Extend Yuj with TOML files](extending-yuj.html) before you add or change
a model, language, or tool descriptor.

### Select a shell sandbox backend

The shipped `[sandbox].backend = "bwrap"` keeps the existing Linux behavior.
Set it to `"container"`, choose `docker` or `podman`, and name a trusted local
image to start one short-lived container for each shell-like command. Yuj
mounts the task directory read-write at the identical absolute path, disables
networking, keeps the image root read-only, drops capabilities, and never
pulls a missing image. `bash`, `run_tests`, post-edit checks, and configured
language servers use the same selected backend. Persistent shell reuse and
host-side trivial-read shortcuts are disabled for container calls.

At task startup Yuj resolves the runtime and inspects the local image ID. With
`[tools].sandbox_required = true`, a missing runtime or image stops before a
model command. With it false, a failed container preflight produces a warning
and the run explicitly degrades to unsandboxed command execution. Successful
runs pin calls to the inspected image ID. A first-class container selection
cannot be combined with the legacy `YUJ_CONTAINER` modes.

`[sandbox].container_flags` accepts only reviewed resource and metadata
options such as memory, CPU, PID-count, labels, and `--init`. Mount, network,
environment, entrypoint, device, privilege, and security-boundary flags are
rejected. See [Sandbox](sandbox.html) for the exact boundary and local-image
preflight.

### Save restorable file checkpoints

Set `[tools].file_checkpoints_enabled = true` to create one independent
shadow-Git commit after every executed `bash`, `write`, `edit`, or
`apply_patch` call. A nonzero or timed-out shell call is still checkpointed
because it may have changed files before it stopped. Calls rejected before
execution are not checkpointed.

The shadow repository is harness-owned and lives outside the task directory.
It uses the task directory as its Git work tree without changing the
project's own Git index, refs, or history. Model file tools cannot traverse to
it, and sandboxed shell commands mask its absolute path. Restore is an
operator/harness function, not a model tool:
`workspace_checkpoints.restore_checkpoint(workspace, turn)`.

Tracked files and non-ignored untracked files are included. Project
`.gitignore` entries and `[tools].file_checkpoints_exclude` patterns are left
outside both capture and restore. Git trees preserve file bytes, executable
mode, and symlink targets; they do not preserve owners, ACLs, or extended
attributes. Each raw `checkpoint` trace row identifies the commit and cost,
and `metrics.json.file_checkpoints.per_call` reports duration, file count, and
byte count for every capture.

### Return language-server diagnostics after edits

Language-server support is off until `[lsp].enabled` is true and at least one
entry exists under `[lsp.servers]`. Each server entry needs `command` and
`extensions`; `root_markers` and `initialization` are optional. Yuj starts a
matching server only after an `edit` or `write` touches one of its extensions.
The server runs as a session child inside the shell sandbox with networking
disabled, and Yuj stops it when the run segment ends.

The default `min_severity = "error"` still counts warnings in the raw
`lsp_diagnostics` trace row but does not put warning text in the model-facing
result. Diagnostics are inserted inside the same `<tool_result>` envelope as
the successful edit and are clipped with the rest of that result to
`[output].max_output_chars`. A timeout is a traced no-op. A missing or failing
server is also a no-op and produces one harness warning per configured server;
it never fails the task run.

Set `[lsp].tool_enabled = true` to add the `lsp` tool with `definition`,
`references`, and `symbols` queries. Navigation uses the same lazy server pool.
Language-server binaries must already exist in the sealed runtime; Yuj never
downloads them.

### Require current file evidence before editing

`[tools].stale_guard_mode` controls a session-local read ledger. `off` skips
the check. `warn` (the shipped default) runs an edit but adds
`WARNING: stale_file: read PATH first` to its result when the file was not
read or its content changed since the last read. `block` refuses that edit
with the same message under an `ERROR` prefix. A blocked result has the stable
error class `stale_file`, so repeated hits participate in the normal error
ladder.

A successful `read` records the file's size, modification time, and content
hash. A successful `write`, `edit`, or `apply_patch` refreshes the affected
paths; a patch deletion removes its path from the ledger. A successful
single-file `cat`, `head`, `tail`, `sed -n`, `grep`, or `rg` shell command also
earns read credit. No-match exit status 1 from `grep` or `rg` still earns
credit because the file was read. Compounds, pipelines, redirects,
substitutions, recursive or multi-file searches, and aggregate/count forms do
not earn credit.

The ledger is harness state, not model-side state. Raw
`stale_guard_observe` trace rows are its only resume source;
`.solver/state.json` does not copy the ledger.

### Redirect shell commands to dedicated tools

The `[[redirect]]` rules in `scripts/llm_solver/bash_quirks/forbidden.toml`
intercept shell fragments that have a safer bounded tool equivalent. A match
does not execute or rewrite the command. It returns
`Blocked: MESSAGE` in a `<tool_result status="error"
error_kind="redirect_rule">` envelope, emits a raw `redirect_rule` trace row,
and participates in the ordinary error ladder.

Write-side rules route in-place editors and shell redirections to `edit` or
`write`. Set `[tools].bash_redirect_read_side = true` to additionally route
file display commands to `read`, text searches to `grep`, and path discovery
to `glob`. Each rule is inert when its target tool is absent from the current
profile-filtered tool set. Matching is compound-aware: quoted operators and
leading `NAME=value` assignments are preserved, while a pipe stage consuming
stdin is not redirected. Aggregate uses such as `grep -c`, `rg --count`, and
`cat FILE | wc -l` remain shell commands.

### Search definitions and references across a repository

Repository structural search is an optional mode of `list_definitions`. Turn
on both `[tools.list_definitions].enabled` and
`[tools].ast_search_enabled`. A repository call uses `repo_wide = true`, treats
`path` as the search root, and accepts an exact `symbol` plus
`kind = "def" | "ref"`. Results are deterministic
`path:line kind name signature` rows. `page` follows the same page size as
`grep` when search pagination is enabled. `[tools].ast_search_max_rows` caps
the available result set before pages are selected, and
`[output].max_output_chars` can reduce a page to complete rows without cutting
one in the middle.

The installed package includes tree-sitter itself, the tag-query package, and
prebuilt Python, JavaScript/TypeScript, Go, Rust, and Java grammar wheels.
Those supported parsers are loaded locally; a model tool call never downloads
a grammar. If the dependencies are absent, the tool returns a typed
`backend_unavailable` error with reinstall guidance. Files matched by
`[sandbox].unreadable_paths` are rejected before they are read.

Repeated calls reuse a bounded in-process content-hash cache. Repository
queries remain ordinary `tool_call` trace rows; neither search results nor the
cache are projected into `.solver/state.json`. Calling `list_definitions` with
only a file `path` preserves the existing standard-library Python outline and
does not require tree-sitter.

The main `config.toml` leaves `tokenizer_id` empty. Yuj then estimates one
token for every four characters. This avoids a model-specific download during
normal use.

The paper runtime files set the tokenizer for each reported model. Apply the
files in the [paper configuration guide](https://github.com/sydches/yuj/blob/main/configs/paper/README.md)
when you reproduce an experiment.

## Configure auxiliary model roles

Named roles let harness-owned side requests use a smaller model without
changing the model that works on the task. The public roles are `weak` and
`editor`. A blank role uses the main model and endpoint. Set a profile name for
a role that shares the main endpoint:

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

`endpoint` must be an absolute HTTP or HTTPS URL without embedded
credentials. An inline target may also set `api_key`, but keep literal secrets
in ignored local configuration and prefer environment-backed credentials.
Yuj loads and validates the main profile and every configured role profile at
startup. An invalid role stops the run before model work begins.

Checkpoint summaries, fresh-session handoffs, and the model-backed hurdle
classifier request the `weak` role. If it is unset, the resolver returns the
actual main client and records that fallback in side-request telemetry. Role
clients are created only when first used and are reused for the same resolved
target. Yuj does not launch or supervise another server: a distinct endpoint
normally means you must run a second llama-server process yourself.

Every model response is charged once to its effective role. Post-run
`metrics.json` reports request, prompt, completion, cached, and total token
counts under `metrics.tokens_by_role`.

## Configure model fallback

Fallback is off by default. Each role's chain is an empty list until you opt
in. A string entry uses exact `<profile>@<endpoint>` syntax:

```toml
[models]
fallback_revert = "never"

[models.fallback_chain]
main = ["qwen3-small@http://127.0.0.1:8181/v1"]
weak = []
editor = []
```

An inline target may also set `model`, `context_size`, or an endpoint-specific
`api_key`. Credentials are used for requests but excluded from trace and
provenance artifacts:

```toml
[[models.fallback_chain.main]]
profile = "qwen3-small"
endpoint = "http://127.0.0.1:8181/v1"
model = "served-small"
context_size = 32768
```

Yuj validates every fallback profile at startup. During a solver turn, it
uses the configured transient retry budget on the active target first. It may
then advance the `main` chain for an exhausted connection/timeout/server
failure, a server out-of-memory failure, or a recognized context-overflow
response. Authentication failures, arbitrary bad requests, malformed
profiles, and tool/protocol errors stay fatal.

Before sending any task message to a replacement, Yuj queries that target's
live context window, applies its profile to the canonical messages and tool
schemas, and checks the resulting prompt against `context_fill_ratio`. A
candidate that cannot fit is traced and skipped. A candidate that fits gets a
fresh retry budget. Client, profile, context-derived limits, tool schemas, and
the context token estimator switch together; old-profile wire messages are
never reused.

`fallback_revert = "never"` keeps the selected target for later sessions.
`"next_session"` returns to the primary target when the next session begins.
Every transition changes the treatment and is recorded as `model_fallback`.
Post-run metrics include `model_fallback_used`, `model_fallback_count`,
`model_fallback_roles`, and `model_fallback_active_targets`, so studies can
exclude runs that changed models.

## Configure llama-server prompt caching

The `[server]` cache settings apply to the OpenAI-compatible llama-server
client. They do not describe provider TTLs.

`request_extra` is a TOML table of additional JSON body fields. Yuj passes
these through the OpenAI SDK's `extra_body`. `cache_affinity = false` disables
slot selection; `true` selects slot 0; a positive integer hashes the stable
product session ID across that many slot numbers. Configure no more slots than
the active llama-server actually exposes.

`cache_retention = "off"` sends `cache_prompt=false`. Set it to `"session"`
to send `cache_prompt=true` on normal solver turns. When affinity is enabled,
the same requests also carry the derived `id_slot`. Cache policy owns both
fields and overrides copies placed in `request_extra`.

`cache_miss_warn_ratio` accepts a value from 0 through 1. Zero disables the
warning. A positive value logs a warning when an observed cache hit ratio is
below the threshold after the session's first request. Missing server cache
telemetry stays unknown and does not produce a false miss warning.

Compaction, handoff, and other harness side requests always send
`cache_prompt=false` and omit `id_slot`, so they cannot replace the solver
conversation's retained prefix. Each solver response records prompt tokens,
cached tokens, and its hit ratio in a `turn` trace row. `metrics.json` contains
the token-weighted run ratio under `metrics.prompt_cache`, and the installed
session summary reports that latest ratio.

## Select reasoning effort

Set `[model].thinking_level` or pass `--thinking` with one of `off`,
`minimal`, `low`, `medium`, `high`, `xhigh`, or `max`. The checked-in default
is `off`.

Each model profile declares the request body for its supported levels under
`[reasoning_levels.<level>]`. The base profile provides boolean `off` and `on`
mappings through `chat_template_kwargs.enable_thinking`. A model-specific
profile may instead map levels to fields such as `reasoning_effort` or
`thinking_budget`.

If the profile does not declare the exact requested level, Yuj clamps
deterministically and logs a warning. It chooses the closest supported effort
that does not exceed the request when possible; a boolean-only profile maps
any positive effort to its internal `on` capability. The effective level is
applied on every normal model request. Harness side requests still force
thinking off.

Profile `[server].reasoning_mode` and `reasoning_disable_flag` remain model
server launch defaults. They do not replace the per-request choice. Each
`session_start` trace row records the effective level (and the requested level
when clamped), while run provenance records requested, effective, and clamp
status.

## Choose a context mode

Context is the text that Yuj gives the model before its next action.

The selected base chooses the normal mode:

| Base | Context mode | What the mode does |
| --- | --- | --- |
| Treatment | `halflife` | Keep messages in time order. Shorten the bodies of older tool results when the input nears its limit. |
| Plain | `full` | Keep the full in-memory message log. |

Override the base choice when you need another mode:

```bash
yuj code --context full "Fix the issue."
```

Yuj registers these modes:

| Source | Modes | What Yuj reads |
| --- | --- | --- |
| In-memory messages | `full`, `compact`, `yuj`, `halflife` | Messages kept by the active process. |
| In-memory messages and current files | `concise`, `slot` | An in-memory working set and the current contents of files that the model touched. |
| Messages, saved state, and current files | `yconcise`, `yslot` | An in-memory working set, `.solver/state.json` when present, and the current contents of touched files. |
| Saved state | `stateful`, `compound`, `focused_compound`, `compound_selective`, `salience` | `.solver/state.json` and a small in-memory window of recent tool results. On resume, Yuj also loads the current contents of files named by the saved run. |

Normal context modes do not read `.trace.jsonl` directly.

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

### Compact a nearly full context

Compaction runs only after the existing context threshold and mutation gate
allow it. These settings live under `[context]`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `compaction_method` | `"digest"` | Use the deterministic trace digest, or opt into a model-written `"checkpoint"`. |
| `checkpoint_keep_recent_tokens` | `0` | Verbatim recent-tail target. Zero means 20% of the live context window, with a 4,096-token minimum. |
| `checkpoint_max_summary_tokens` | `4000` | Maximum checkpoint response; the runtime also applies a 4,000-token hard cap and the available-reserve limit. |
| `digest_compaction_safety_margin` | `0.05` | Margin used by the derived compaction threshold. |
| `digest_keep_recent_turns` | `8` | Digest tail size and the close-compaction guard window. |
| `digest_compaction_gate_min_mutations` | `0` | Minimum successful mutations before compaction may run. |

Checkpoint mode makes one no-tool call through the `weak` model role. Thinking
is off for that call. Yuj keeps the system prompt and task message unchanged, places
the validated checkpoint after the task, and keeps a verbatim recent tail
beginning at an assistant-turn boundary. The checkpoint must contain every
required section and every mechanically observed modified path, fit the
budget, and reduce the prompt token count. Any request, response, validation,
or size failure uses the deterministic digest instead.

Yuj records only compaction metadata in the trace and state projection; it
does not copy model-written checkpoint text into `.solver/state.json`. If two
compactions occur within `digest_keep_recent_turns`, later compactions in that
run segment use digest to avoid a compaction loop.

### Summarize work for a fresh session

Set `[loop].handoff_summary_enabled = true` to make one no-tool `weak`-role side request
when a session ends because of `context_full`, `length`, or `max_turns` and
another session is available. `[prompts].handoff_max_tokens` defaults to
`2000` and limits the returned summary. Thinking is off for this request.

Yuj validates the seven required sections, the response size, and coverage of
every modified path found in the raw trace. A valid `<handoff>` is placed after
the task statement and before the existing mechanical resume tail. A missing
section, omitted path, oversized response, request failure, or model failure
leaves that mechanical prompt byte-for-byte unchanged. The trace records only
the attempt's token count, validity, fallback, and model role; model-written
handoff text is not copied into `.solver/state.json`.

### Recover an interrupted tool turn

`[loop].interrupted_turn_mode = "mechanical"` is the default. Before a
trace-backed resume reads prior events, Yuj removes only a malformed trailing
JSON fragment, fsyncs the repaired prefix, and appends one `turn_aborted`
event. The next user message names any call that had started without a durable
result and says its outcome is unknown. Transcript resume closes the dangling
assistant/tool protocol edge with a synthetic tool result; it never reruns the
call or claims that its filesystem effects succeeded or failed.

Before every real dispatch Yuj fsyncs a bounded `tool_start` event. SIGTERM,
SIGINT, fatal exceptions, normal scope completion, and process exit record a
`session_exit` event with the pending-call set. A hard kill cannot run an exit
handler, so the already-durable start row remains the recovery evidence.

Set the mode to `"off"` to disable mechanical repair. In that mode resume does
not truncate a malformed suffix and does not append `turn_aborted`; ordinary
trace loading may therefore reject or stop at the damaged tail.

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
yuj code --config longer-task.toml \
  "Complete the migration and run its tests."
```

Do not copy all of `config.toml` to change two values. A small file shows the
change. It also lets later Yuj defaults still apply.

Repeat `--config` to apply more files. Yuj applies them from left to right.

## Use another model service for one session

Set the key before you start the session:

```bash
export ANTHROPIC_API_KEY='...'
yuj code --provider anthropic --model YOUR_MODEL_ID \
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
| `YUJ_CONFIG` | Use this `config.toml` file as the main settings file. Yuj reads `config.local.toml` from the same directory when that file exists. |
| `HARNESS_ASSIST_HOME` | Use this path instead of `<yuj-installation>/.llm_assist` for sessions. |
| `YUJ_CONTAINER=ambient` | Use the current outer container as the shell boundary. |
| `YUJ_CONTAINER=<container-id>` | Run model shell commands in this existing task container. |
| Model-service key variables | Supply a key named by `$ENV:NAME`, such as `OPENAI_API_KEY`. |

These process controls are optional:

| Variable | What it changes |
| --- | --- |
| `YUJ_STREAMING=1` | Read model replies as a stream. Streaming is off by default. |
| `YUJ_PERSISTENT_BASH=0` | Start a new shell process for each `bash` tool call. Yuj normally reuses one eligible `bwrap` shell during a run segment. |

Do not use `YUJ_CONFIG_LOCAL` to move local settings. `yuj setup` writes to
that path, and `doctor` may print that path. `code`, `run`, `smoke`, `models`,
and `doctor` still load `config.local.toml` beside the main `config.toml`.

Read [Treatment](treatment.html) for treatment settings. Read
[Sandbox](sandbox.html) for shell access. Read the
[CLI reference](using-yuj.html) for every command-line option.
