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
| `[model].name` | Set the exact model ID that the service accepts. |
| `[model].profile_name` | Use settings for a model's message and tool-call format. |
| `[model].context_size` | Give Yuj an input limit when the service does not report one. |
| `[model].tokenizer_id` | Count tokens with this Hugging Face tokenizer. Leave it empty to estimate the count from text length. |
| `[loop].max_turns` | Limit the number of model tool-call turns in one session. |
| `[loop].handoff_summary_enabled` | Ask for a validated summary before an eligible fresh-session rollover. Off by default. |
| `[prompts].handoff_max_tokens` | Bound the optional model-written rollover summary. |
| `[tools].bash_timeout` | Limit the time for one shell command. |
| `[tools].sandbox_bash` | Turn the shell sandbox on or off. |
| `[tools].sandbox_required` | Stop if Yuj cannot start the requested `bwrap` sandbox. |

The checked-in [`config.toml`](https://github.com/sydches/yuj/blob/main/config.toml)
defines and comments the full public settings shape. Use a small settings file
for changes. Do not copy the full file into your project.

Read [Model tools](model-tools.html) for optional tool settings. Read
[Run a local model](serving_overlay.html) for runtime files and profiles.
Read [Extend Yuj with TOML files](extending-yuj.html) before you add or change
a model, language, or tool descriptor.

The main `config.toml` leaves `tokenizer_id` empty. Yuj then estimates one
token for every four characters. This avoids a model-specific download during
normal use.

The paper runtime files set the tokenizer for each reported model. Apply the
files in the [paper configuration guide](https://github.com/sydches/yuj/blob/main/configs/paper/README.md)
when you reproduce an experiment.

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

Checkpoint mode makes one no-tool call to the active model. Thinking is off
for that call. Yuj keeps the system prompt and task message unchanged, places
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

Set `[loop].handoff_summary_enabled = true` to make one no-tool side request
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
