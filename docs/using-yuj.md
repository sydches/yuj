---
layout: default
title: Yuj CLI reference
nav_order: 3
---

# Yuj CLI reference

The `yuj` command starts and controls coding sessions.

Use these terms in this guide:

| Term | Meaning |
| --- | --- |
| Task | The request that you give Yuj. |
| Coding session | One saved task record with one session ID. A coding session can continue after a resume. |
| Run segment | Model work between a start or resume and the next end, pause, or interrupt. |
| Target repository | The Git repository that the model may change. |
| Session directory | The directory where Yuj saves records for one coding session. |

Run `yuj --help` for the command list. Run `yuj COMMAND --help` for the
current options for one command. Every command accepts `-h` and `--help`.

Activate the environment where Yuj is installed before you run a command on
this page. Otherwise, replace `yuj` with that environment's `bin/yuj` path.

## Command list

| Command | What it does |
| --- | --- |
| `yuj setup` | Save the model connection settings for this machine. |
| `yuj login` | Save and select a Claude or Codex credential. |
| `yuj auth-status` | Show the selected provider and authentication method without credentials. |
| `yuj logout` | Remove one provider-scoped credential. |
| `yuj config` | Validate and explain the resolved settings without model work. |
| `yuj models` | List models from the selected service. |
| `yuj doctor` | Check the settings, model connection, Git, and `bwrap`. |
| `yuj smoke` | Ask the model to fix and test a small throwaway directory. |
| `yuj code` | Start a coding session. |
| `yuj run` | Run the same command as `yuj code`. |
| `yuj current` | Run `yuj status latest`. |
| `yuj status` | Show one session's status and the next user action. |
| `yuj show` | Show one session's settings and recent activity. |
| `yuj sessions` | List saved sessions. |
| `yuj resume` | Continue a paused session. |
| `yuj answer` | Record one exact answer for a pending clarification. |
| `yuj rewind` | Restore a stopped session to an earlier conversation and tree turn. |
| `yuj worktree rm` | Remove a retained session worktree and branch. |
| `yuj approve` | Allow a tool action that needs approval. |
| `yuj reject` | Refuse a tool action that needs approval. |

When you run `yuj` with no command, Yuj normally prints its help. On the first
interactive run with no local settings file, Yuj starts interactive setup.

## Inspect settings

### `yuj config`

Validate the complete configuration before starting a coding session:

```bash
yuj config
yuj config --json
```

Human output shows each effective value and its winning source layer. JSON
output uses the stable `yuj.config-inspection` schema version 1. Secret and
environment-derived values are redacted in both modes. The command performs
no model request and writes no session artifacts.

| Option | What it does |
| --- | --- |
| `--json` | Write deterministic machine-readable JSON. |
| `--config PATH`, `-c PATH` | Apply this TOML file. Repeat from left to right. |
| `--treatment`, `--no-treatment` | Inspect the treatment or plain base. |
| `--context NAME` | Validate this context mode. |
| `--model NAME`, `-m NAME` | Apply this model override last. |
| `--thinking LEVEL` | Apply this reasoning-effort override last. |
| `--plan-mode MODE` | Apply this planning-mode override last. |
| `--edit-format FORMAT` | Apply this profile edit-format override last. |
| `--provider NAME` | Apply this model-service choice last. |
| `--base-url URL` | Apply this service address last. |
| `--api-key-env NAME` | Read the key from this variable and redact its value. |
| `--agent NAME` | Validate this named-agent descriptor; repeat for more agents. |

The command returns `0` only when resolution and validation succeed. It
returns `1` and an actionable diagnostic otherwise. Read
[Configuration](configuration.html#inspect-the-resolved-settings) for layer
order, validation coverage, the JSON fields, and the redaction boundary.

## Start a task

Move to the Git repository that the model may edit. Then run:

```bash
yuj code "Fix the failing tests and check the change."
```

The model can read files, change code, run commands, and run tests. Yuj
continues until the model finishes or a stopping rule ends the session.

Keep the terminal process running while the run segment is active.

`yuj run` and `yuj code` use the same code and accept the same options.

Use the local startup seam when you want to validate an installation or task
repository without contacting the model service:

```bash
yuj code --dry-run "check local startup"
```

This follows the same local startup path as an ordinary session. It loads the
settings and shipped resources. It validates the selected profile, tool
surface, sandbox, and each enabled agent, project-file, skill, injection,
stream-rule, language-rule, and security feature. It writes no session or run
artifact. It exits before model discovery and prints
`Model network: not contacted`.

### Model tools

Yuj offers these tools to the model:

| Job | Tools | Shipped setting |
| --- | --- | --- |
| Read files and find code | `read`, `glob`, `grep` | On |
| List names in a Python file | `list_definitions` | Off |
| Change files | One of `edit`, `apply_patch`, `udiff`, or `write` | Profile-selected; `_base` uses `edit`. |
| Run shell commands | `bash` | On |
| Run tests | `run_tests` | Off |
| Ask one clarification question | `ask_user` | On in the top-level assistant session; absent from child agents and measurements. |
| Finish the task | `done` | On |
| Leave a required planning phase | `exit_plan_mode` | On only when plan mode is required. |

`run_tests` is a separate test tool. When `bash` is on, the model can still run
test commands through `bash`.

Turn on an optional tool in a TOML file:

```toml
[tools.list_definitions]
enabled = true

[tools.run_tests]
enabled = true

[tools]
edit_format = "apply_patch"
```

A model profile can limit the number of enabled tools that Yuj sends to the
model.

Yuj applies the sandbox and approval rules when it runs a tool. Read
[Model tools](model-tools.html) for every tool input. Read
[Sandbox](sandbox.html) for the access rules.

### Give Yuj the task

Give exactly one task source.

| Form | Example |
| --- | --- |
| Text after the command | `yuj code "Fix the failing tests."` |
| `--prompt-text` | `yuj code --prompt-text "Fix the failing tests."` |
| `--prompt-file` | `yuj code --prompt-file /path/to/task.txt` |

Use a file for a long task:

```bash
yuj code --prompt-file /path/to/task.txt
```

Use `--cwd` when the repository is not the current directory:

```bash
yuj code --cwd /path/to/project "Update the parser and its tests."
```

`--cwd` also sets the repository that session-selection commands prefer.

### Options for `code` and `run`

Yuj calls the starting group of settings a base.

| Option | What it does |
| --- | --- |
| `TASK ...` | Join the remaining words and use them as the task text. |
| `--cwd PATH` | Edit this repository. The current directory is the default. |
| `--prompt-text TEXT` | Use this text as the task. |
| `--prompt-file PATH` | Read the task from this file. |
| `--model NAME`, `-m NAME` | Use this model ID or known short name. |
| `--thinking LEVEL` | Use `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max` reasoning effort for every normal request. |
| `--plan-mode MODE` | Use `off` or require a nonempty `.solver/plan.md` and explicit `exit_plan_mode` before implementation. |
| `--edit-format FORMAT` | Override the model profile with `exact`, `apply_patch`, `udiff`, or `whole`. |
| `--provider NAME` | Use `local`, `claude`, `codex`, `openai`, `anthropic`, `openrouter`, `zai`, or `custom`. |
| `--base-url URL` | Use this API base address. `custom` requires it. |
| `--api-key-env NAME` | Read the API key from this environment variable. |
| `--config PATH`, `-c PATH` | Apply this TOML file. Repeat the option to apply more files from left to right. |
| `--system-prompt PATH` | Add this file before Yuj's normal system prompt. The file may import another file with an `@path` line. |
| `--treatment` | Use the treatment base. This is the default. |
| `--no-treatment` | Use the plain base. |
| `--context NAME` | Use this context mode instead of the base default. |
| `--dry-run` | Complete local startup through the model-network boundary, write no session artifacts, and exit. |

A context mode controls which earlier messages, saved facts, and current files
the model receives before its next action.

`--provider`, `--base-url`, `--api-key-env`, `--thinking`, `--plan-mode`, and
`--edit-format` change only the new session. Yuj saves these settings in the
session's `provider.toml`.

When you use `--api-key-env`, `provider.toml` stores the variable name. It
does not store the key.

Before Yuj creates the coding session, it asks the service for its model list.
Yuj uses the selected model when the exact ID appears in that list. If it does
not appear, Yuj normally uses the first returned model. When the same command
also gives a remote service address and an explicit model, Yuj keeps that
explicit model even if the list omits it. Read the `model:` line that Yuj
prints at the start of each coding session.

If you give `--base-url` or `--api-key-env` without `--provider`, Yuj treats
the new connection as `custom` and OpenAI-compatible.

On `code`, `run`, and `smoke`, `--provider local` changes only the API format.
It does not replace a saved base address or key. Use `yuj setup --provider
local` to save the normal localhost address and `local` key.

An `@path` line in a system-prompt file imports that file. A relative import
starts beside the file that contains it. Imports may nest five levels. Yuj
adds a visible marker for a missing file, a cycle, or a deeper import.

`--provider claude` and `--provider codex` use the provider-scoped credential
selected by `setup` or `login`. Yuj rejects a different endpoint or a
different selected provider before it reads that credential. A provider
failure does not switch the provider, account, credential, model, endpoint,
or billing method.

Read [Configuration](configuration.html) for setting order, model services,
context modes, and saved settings.

## Set up a model

### `yuj setup`

Run `yuj setup` with no options for an interactive setup.

Use options when you want a setup command that you can run again:

```bash
yuj setup --provider openai --model YOUR_MODEL_ID \
  --api-key-env OPENAI_API_KEY
```

For an eligible Claude or ChatGPT subscription, use browser sign-in:

```bash
yuj setup --provider claude --auth subscription --model YOUR_MODEL_ID
yuj setup --provider codex --auth subscription --model YOUR_MODEL_ID
```

For a provider-scoped API key, use `--auth api-key` with exactly one key
source:

```bash
yuj setup --provider claude --auth api-key --model YOUR_MODEL_ID \
  --api-key-env ANTHROPIC_API_KEY
yuj setup --provider codex --auth api-key --model YOUR_MODEL_ID \
  --api-key-env OPENAI_API_KEY
```

For an installed package, Yuj writes
`$XDG_CONFIG_HOME/yuj/config.local.toml`, or
`~/.config/yuj/config.local.toml` when `XDG_CONFIG_HOME` is unset. An editable
source checkout keeps `config.local.toml` at the checkout root, where Git
ignores it. `YUJ_CONFIG_LOCAL` selects an exact alternative path.

| Option | What it does |
| --- | --- |
| `--provider NAME` | Save `local`, `claude`, `codex`, `openai`, `anthropic`, `openrouter`, `zai`, or `custom`. |
| `--auth METHOD` | With `claude` or `codex`, use `api-key` or `subscription`. |
| `--model NAME`, `-m NAME` | Save the default model ID. |
| `--base-url URL` | Save the API base address. `custom` requires it. |
| `--api-key-env NAME` | Save `$ENV:NAME`. Yuj reads the key from that variable when a command starts. |
| `--api-key VALUE` | Save a key. Claude and Codex keep it in their provider credential file; other service choices keep it in `config.local.toml`. |
| `--force` | Replace an existing local settings file without asking. |

Without `--force`, a non-interactive command does not replace an existing
file. An interactive command asks first.

Always give `--provider` when you use setup options. Without it, Yuj starts
the interactive provider question.

With Claude or Codex, give only one of `--api-key` and `--api-key-env`.

### `yuj login`, `yuj auth-status`, and `yuj logout`

Use `login` to replace and select one provider credential without rewriting
the model settings:

```bash
yuj login --provider claude --auth subscription
yuj login --provider codex --auth api-key --api-key-env OPENAI_API_KEY
```

`login` uses `subscription` when you omit `--auth`. Run `yuj auth-status` to
print only the active provider, authentication method, and whether its
credential record exists. The command returns status `1` when no credential is
selected or the selected record is invalid.

Name the provider when you remove a credential:

```bash
yuj logout --provider claude
yuj logout --provider codex
```

Omit `--provider` to remove the active provider's credential. `logout` never
removes the other provider's file.

Subscription credentials refresh when they approach expiration. Missing,
malformed, expired without a usable refresh token, revoked, ineligible, and
unsupported credentials stop with distinct errors. Yuj does not try another
credential or service after any of these errors.

Yuj stores provider credentials under `$XDG_CONFIG_HOME/yuj/auth`, or
`~/.config/yuj/auth` when `XDG_CONFIG_HOME` is unset. The directory has mode
`0700`. Credential and active-selection files have mode `0600` and are
replaced atomically. Credential values do not enter the target repository,
model messages, session artifacts, trace, logs, configuration output, or
model-command environment.

### `yuj models`

List the models from the selected service:

```bash
yuj models
```

Yuj marks the selected model with `*`. It returns a nonzero status if the
service returns no models.

Apply one or more temporary settings files when needed:

```bash
yuj models --config another-provider.toml
```

| Option | What it does |
| --- | --- |
| `--config PATH`, `-c PATH` | Apply this TOML file. Repeat the option to add more files. |

### `yuj doctor`

Check the current setup:

```bash
yuj doctor
```

`doctor` reports:

- whether Yuj can load the settings
- the model service, API base address, and selected model
- whether the service returns a model list
- whether a local server offers the selected model
- whether `config.local.toml` exists
- whether the current directory is a Git repository root
- whether `bwrap` is installed

`doctor` reports Git and `bwrap` problems as warnings. It checks only whether
the current directory itself contains `.git`. It does not check whether a
parent directory is a Git repository.

For an OpenAI-compatible address that starts with `http://localhost`, a
missing selected model is an error. For other addresses, it is a warning
because some hosted services do not list every valid model ID.

A coding session can still stop later if its settings require `bwrap`.

| Option | What it does |
| --- | --- |
| `--config PATH`, `-c PATH` | Apply this TOML file. Repeat the option to add more files. |

### `yuj smoke`

Run a small check through the model and tool loop:

Install Yuj with the `test` extra first. The final smoke check runs `pytest`.

```bash
yuj smoke
```

`smoke` creates a throwaway directory. The directory contains one broken
addition function and one test. It is not a Git repository.

The model must fix the function. Yuj then checks all four results:

1. The expected code change exists.
2. The target test passes.
3. No approval request remains open.
4. No clarification exchange remains unresolved.

A successful smoke task tests this one path. It does not measure the model's
general coding skill.

The command prints the throwaway path and keeps it after the check.

| Option | What it does |
| --- | --- |
| `--root PATH` | Use this throwaway directory instead of a new temporary directory. Yuj writes `calc.py` and `tests/test_calc.py` there. |
| `--assist-home PATH` | Save the smoke session under this session root. |
| `--model NAME`, `-m NAME` | Use this model ID or known short name. |
| `--edit-format FORMAT` | Override the model profile with `exact`, `apply_patch`, `udiff`, or `whole`. |
| `--provider NAME` | Use a model service setting. |
| `--base-url URL` | Use this API base address. |
| `--api-key-env NAME` | Read the API key from this environment variable. |
| `--config PATH`, `-c PATH` | Apply this TOML file. Repeat the option to add more files. |
| `--system-prompt PATH` | Add this file before Yuj's normal system prompt. |
| `--treatment` | Use the treatment base. This is the default. |
| `--no-treatment` | Use the plain base. |
| `--context NAME` | Use this context mode instead of the base default. |

Do not give `--root` a directory that contains work you need.

Later commands have no `--assist-home` option. Set `HARNESS_ASSIST_HOME` to
the same path before you run `status`, `show`, `answer`, `approve`, `reject`,
or `resume` for that smoke session.

## Understand Git changes

Yuj asks the model to change the target repository. When the repository is
dirty, Yuj also tries to make a Git checkpoint after a normal, failed, or
error ending.

For that checkpoint, Yuj runs `git add -A`. It then creates this commit:

```text
yuj: session N checkpoint (FINISH_REASON)
```

The commit uses `yuj-harness <yuj@localhost>` as its Git identity. It includes
all dirty files in the target repository, even changes that existed before
Yuj started. Commit or store your own work before you start a task.

Yuj does not try this checkpoint when it pauses for approval. An interrupt can
also stop before the attempt. If Git fails, Yuj logs a warning and leaves the
working tree as it is.

## Inspect sessions

Use these commands instead of reading raw files during normal work.

The trace is Yuj's time-ordered record of all run segments in a coding session.

| Command | What it shows |
| --- | --- |
| `yuj current` | The active or newest session for the current repository. If that repository has no session, the newest saved session. |
| `yuj status [SESSION]` | Status, finish reason, latest ended segment's turn count, repository, model, pinned provider and authentication method, clarification, approval, process lock, interrupt mark, and next action |
| `yuj show [SESSION]` | Status, times, saved-file path, pinned provider and authentication method, context mode, task source, clarification, approval, next action, recent turns, and recent trace events |
| `yuj sessions --limit N` | Up to `N` recent sessions from all repositories; the default is 20 |

Use `yuj show --turns N --trace-lines N [SESSION]` to choose how many recent
turns and trace events to print.

| Command | Option | What it does |
| --- | --- | --- |
| `yuj status` | `SESSION` | Select a session. The default is `latest`. |
| `yuj show` | `SESSION` | Select a session. The default is `latest`. |
| `yuj show` | `--turns N` | Show this many recent turns. The default is 5. |
| `yuj show` | `--trace-lines N` | Show this many recent trace events. The default is 10. |
| `yuj sessions` | `--limit N` | List at most this many sessions. The default is 20. |

The current parser accepts any integer for these three options. A value of
`0` prints no matching rows. A negative `--turns` or `--trace-lines` value
also prints no tail. A negative `--limit` removes the database row limit.
Use a positive value for normal work.

### Select a session

A session has a full ID and a short ID. You can also give a unique start of
either ID.

Use `latest` or `last` to let Yuj choose. If you omit the ID, Yuj uses
`latest`.

For `status` and `show`, Yuj chooses in this order:

1. the active session for the current repository
2. the newest session for the current repository
3. the newest session from any repository

`current` runs `status latest`.

The active-session entry is a saved pointer. It can still point to a completed
session. Yuj does not use the word `active` to prove that a process is running.

For `resume`, Yuj first looks for an active session that has not completed.
It then looks for the newest incomplete session in the current repository.
Finally, it looks for the newest incomplete session from any repository.

For `approve` and `reject`, Yuj first looks for a waiting request in the
active session. It then looks in the current repository. Finally, it looks in
the other sessions in its 200-session search.

`answer` requires both an explicit session reference and the exact request ID.
It does not choose a pending question automatically.

Yuj stops with an error if an ID matches no session or more than one session.

Automatic selection checks the active-session pointer first when the command
uses that pointer. It then searches the 200 newest saved sessions. A full ID
uses a direct lookup. A short ID or unique ID start searches the 1,000 newest
saved sessions.

### Session statuses

| Status | Meaning |
| --- | --- |
| `created` | Yuj saved the first session files but has not started the task. |
| `running` | The trace has a start event and no later end event. Check the separate `lock` line to see whether a process currently owns the session. |
| `approval_pending` | A tool action waits for your choice. |
| `input_required` | The model asked one clarification question. Record the answer before resume. |
| `input_ready` | One clarification answer is recorded and waiting for one resume delivery. |
| `paused` | The last run segment stopped without success. The coding session can continue. |
| `completed` | The model finished successfully or declared the task done. |
| `error` | Yuj recorded an error as the final status. |

## Answer a clarification question

When the model needs one missing fact, it can pause the run segment with one
exact question. Inspect the session:

```bash
yuj status SESSION
yuj show SESSION
```

Both commands print the question, request ID, and exact next command. Record
one answer with the displayed IDs:

```bash
yuj answer SESSION REQUEST_ID 'ANSWER'
```

Yuj records the `ANSWER` argument exactly. Quote it when it contains spaces or
shell characters. There is no default or interactive prompt. Yuj refuses an
empty answer, a wrong session or request ID, a second answer, and an answer
when no question is pending. A refusal leaves the clarification records and
trace unchanged.

The answer supplies information only. It does not approve any tool action.
If an approval request is also pending, use `yuj approve` or `yuj reject`
separately.

## Resume a session

Resume the session that Yuj selects:

```bash
yuj resume
```

Resume a named session:

```bash
yuj resume SESSION
```

The command accepts a full ID, short ID, unique ID start, `latest`, or
`last`.

Yuj keeps the files already changed in the target repository. It starts a new
model context with the original task and a short summary built from the most
recent ended run segment in the trace.

Ordinary resume does not restore the full prior conversation. If an interrupt
leaves no end event, the summary may be absent. The new context then starts
from the original task and the current files.

The trace, savings record, and system log continue in the same session
directory. `transcript.log` describes the newest run segment. Before resume,
Yuj moves the preceding model transcript to the next
`transcript.pre_seg_N.log` file. `checkpoint.json` and `metrics.json` describe
only the newest run segment because a resume replaces them.

`resume` does not accept new model, context, or settings options.

`resume` refuses a session whose clarification still lacks an answer. When an
answer is ready, resume adds the exact answer to one model request and records
its consumption before transport. A later resume does not add it again.

If you name a completed coding session, `resume` prints its result and exits
without starting another run segment.

Press Ctrl-C during `code`, `run`, `smoke`, or `resume` to pause the
session. Yuj saves an interrupt mark, prints the resume command, and returns
status 130.

### Rewind before the next resume

Enable conversation rewind and file checkpoints before you start the session:

```toml
[loop]
rewind_enabled = true
rewind_max_per_session = 1

[tools]
file_checkpoints_enabled = true
```

Stop the session before you rewind it. Choose a completed turn earlier than
the latest turn in the newest run segment, then run:

```bash
yuj rewind SESSION TURN
```

Add `--reason TEXT` when you want to record why you rewound. Yuj checks that
the saved messages match their file checkpoint, restores the files, adds a
`rewind` event without deleting trace rows, rebuilds the state view, and leaves
the session paused. Run the printed `yuj resume SESSION` command to continue
from the messages saved at that turn. `rewind_max_per_session` still limits the
number of successful rewinds.

A rewind marks any pending or answered-but-unconsumed clarification as
rewound. Its durable records remain available for audit, but the answer cannot
enter the conversation after the rewind.

## Remove an isolated session worktree

When `[runtime].worktree` is enabled, Yuj keeps the session's worktree and
branch after every exit so `resume` sees exactly the same files. Remove it by
session ID or unique reference only when you no longer need that workspace:

```bash
yuj worktree rm SESSION
```

The command refuses uncommitted files and commits that are not merged into the
source checkout. Use `--force` only when you intend to discard both:

```bash
yuj worktree rm SESSION --force
```

Before removal, Yuj verifies the saved path, branch, base commit, Git
registration, and ownership metadata. It never removes a worktree
automatically.

## Approve or reject a tool action

A permission rule can require approval for any tool. Yuj also applies a fixed
approval check to risky shell commands.

Yuj checks each shell segment in a `bash` command. It pauses when a segment
starts with one of these command forms:

- `rm`
- `git reset --hard`
- `git clean`
- `git checkout --`
- `chmod`
- `chown`
- `cp` or `mv` with a path outside the target repository

This is a fixed command check. It is not a general test of whether a command
is safe.

Inspect the request before you choose:

```bash
yuj show
```

Approve the action:

```bash
yuj approve [SESSION]
```

Resume the session:

```bash
yuj resume [SESSION]
```

Reject the action. Give the model a reason with `--reason`:

```bash
yuj reject [SESSION] --reason "Do not remove that directory"
```

Resume the session:

```bash
yuj resume [SESSION]
```

| Command | Option | What it does |
| --- | --- | --- |
| `yuj approve` | `SESSION` | Select a session with a waiting request. |
| `yuj approve` | `--always` | Approve the same tool action for the rest of this session. |
| `yuj reject` | `SESSION` | Select a session with a waiting request. |
| `yuj reject` | `--reason TEXT` | Send this reason to the model. |
| `yuj reject` | `--always` | Reject the same tool action for the rest of this session. |

If you omit `--reason`, Yuj sends `operator rejected the action`.

`--always` matches the exact tool action. For `bash`, this includes the command
text. For other tools, Yuj uses the stable argument identity described in
[Configuration](configuration.html#apply-per-tool-permission-rules). The
choice applies only to this coding session.

Yuj does not resume while a request is waiting. Approve or reject the request
first.

An approval or rejection records your choice. Run `resume` to continue the
session.

The approval check does not replace the sandbox.

## Start another task

Start a new session after one task finishes:

```bash
yuj code "Add a test for the fixed case."
```

The new session sees the files left by the old session. It does not receive the
old conversation.

Yuj locks a coding session, not a target repository. Do not run two coding
sessions against the same repository at the same time.

Use `yuj resume` when you want to continue a paused session.

## Find the saved files

Yuj stores its session index here:

```text
<assist_home>/sessions.sqlite3
```

Yuj stores each session here:

```text
<assist_home>/sessions/<session_id>/
```

For an installed package, `<assist_home>` is `$XDG_STATE_HOME/yuj`, or
`~/.local/state/yuj` when `XDG_STATE_HOME` is unset. For an editable/source
checkout it is `<checkout>/.llm_assist`. Set `HARNESS_ASSIST_HOME` to use an
exact alternative.

The session directory can contain the task text, settings, trace, segmented
model messages, `.solver/state.json`, clarification evidence, approvals,
final status, metrics, and context-saving records.

Most Yuj records stay in the session directory. Large kept tool output can
also appear under `.tool_output/` in the target repository.

Read [Saved files](harness_artifacts.html) for every file name and its allowed
uses.

## Exit statuses

| Status | Meaning |
| ---: | --- |
| 0 | The command succeeded. A coding session reported success. A smoke task also passed its four checks. |
| 1 | The command failed, the session did not finish successfully, a required check failed, or Yuj rejected a semantic input such as an unknown session. |
| 2 | The option parser rejected the command form or option combination. |
| 130 | You stopped an active session with Ctrl-C, and Yuj paused it. |

An unknown or unclear session ID also returns a nonzero status and prints the
reason.

## Measurements and replay

The installed `yuj` command is for normal coding work. A separate command runs
fixed measurements and replay.

Read [Measurements](measurement.html) for every measurement option and its
task-directory contract. Read [Replay](replay_mode_spec.html) for replay rules
and current limits.
