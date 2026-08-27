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
| Coding session | One saved task record with one session ID and, optionally, one manual label. A coding session can continue after a resume. |
| Run segment | Model work between a start or resume and the next end, pause, or interrupt. |
| Target repository | The Git repository that the model may change. |
| Session directory | The directory where Yuj saves records for one coding session. |

Run `yuj --help` for the coding-session options and secondary command list.
Run `yuj --version` to print the installed version.
Run `yuj COMMAND --help` for the options of one secondary command. Every
command accepts `-h` and `--help`.

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
| `yuj doctor` | Check the settings, sandbox resolution, model connection, and Git. |
| `yuj smoke` | Ask the model to fix and test a small throwaway directory. |
| `yuj init` | Analyze one repository and propose one project instruction file for review. |
| `yuj` | Start a coding session. Enter the task when prompted or pass it as an argument. |
| `yuj current` | Run `yuj status latest` over unarchived sessions. |
| `yuj status` | Show one session's status and the next user action. |
| `yuj show` | Show one session's settings and recent activity. |
| `yuj export` | Print one redacted Markdown report for a saved session. |
| `yuj diff` | Print changes in one retained session worktree. |
| `yuj usage` | Show exact persisted usage for one session without a provider request. |
| `yuj sessions` | List, filter, or select saved sessions. |
| `yuj trust` | Inspect or revoke one workspace's startup trust. |
| `yuj label` | Set, replace, or clear one manual session label. |
| `yuj fork` | Create an independent child from one stopped saved session. |
| `yuj archive` | Hide one stopped session from ordinary selection without changing its evidence. |
| `yuj unarchive` | Restore one archived session to ordinary selection. |
| `yuj purge` | Preview or permanently remove one archived session and its owned artifacts. |
| `yuj resume` | Continue a paused session. |
| `yuj correct` | Record one exact correction for a stopped session. |
| `yuj answer` | Record one exact answer for a pending clarification. |
| `yuj rewind` | Restore a stopped session to an earlier conversation and tree turn. |
| `yuj worktree rm` | Remove a retained session worktree and branch. |
| `yuj approve` | Allow a tool action that needs approval. |
| `yuj reject` | Refuse a tool action that needs approval. |

When you run `yuj` with no task, Yuj prompts for one and starts a coding
session. On the first interactive run with no local settings file, Yuj runs
setup before it asks for the task.

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

The `selection.sandbox` object reports the platform's supported, installed,
available, and unavailable backends, the configured choice, and the
capability-resolved backend. It does not start a sandbox. Use `doctor` or
`yuj --dry-run` for the operational preflight.

| Option | What it does |
| --- | --- |
| `--json` | Write deterministic machine-readable JSON. |
| `--config PATH`, `-c PATH` | Apply this TOML file. Repeat from left to right. |
| `--treatment`, `--no-treatment` | Inspect the treatment or plain base. |
| `--context NAME` | Validate this context mode. |
| `--model NAME`, `-m NAME` | Apply this model override last. |
| `--thinking LEVEL` | Apply this reasoning-effort override last. |
| `--plan-mode MODE` | Apply this planning-mode override last. |
| `--permission-preset NAME` | Expand this fixed assistant permission preset. |
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
yuj
```

Enter the task at the `Task:` prompt. You can also pass it directly:

```bash
yuj "Fix the failing tests and check the change."
```

The model can read files, change code, run commands, and run tests. Yuj
continues until the model finishes or a stopping rule ends the session.

Keep the terminal process running while the run segment is active.

### Create a project instruction file

Use `yuj init` when a repository needs its first useful instruction file:

```bash
yuj init -C /path/to/project --output AGENTS.md
```

You must name the output file. The name must be `AGENTS.override.md` or one of
the names in `prompts.project_doc_names`. Use `-C` to select the directory
where that file belongs.

The command gives the model only `read`, `glob`, `grep`, `ask_user`, `done`,
and one constrained `write`. The write accepts only the selected filename.
The proposed file can contain at most 80 lines and 8,000 characters. Yuj also
hides Git-ignored paths, configured ignore paths, `.git`, `.internal`,
`.solver`, `.tool_output`, and `.procs` during this analysis.

Yuj prints the absolute destination and the complete proposed content. It
does not write the file yet. Inspect the proposal in that output, or show it
again later:

```bash
yuj show SESSION
```

Approve and resume only when the content is correct:

```bash
yuj approve SESSION
yuj resume SESSION
```

The approval belongs to the exact filename and content. A changed proposal
needs another approval. If the file already exists, Yuj still pauses before
replacing it. Reject the proposal when you want to keep the repository
unchanged:

```bash
yuj reject SESSION --reason "Keep the current instructions."
```

The command keeps workspace trust, the configured permission policy, file
permissions, sandboxing, and security scanning active. It writes to the
selected checkout instead of creating an isolated worktree. A rejection,
interruption, or failure before approval leaves the repository unchanged.

### Trust repository behavior

Before Yuj loads behavior supplied by the selected repository, it shows the
behavior categories and exact paths. This includes enabled project
instructions, project skills, injections, stream rules, `.yujignore` files,
repository settings, and configured extension points. An interactive command
asks once whether to trust that workspace.

For a non-interactive start, make the choice on the command line:

```bash
yuj --trust-workspace "Fix the failing tests."
```

`--no-trust-workspace` refuses repository behavior for that invocation. If no
repository behavior is enabled, Yuj starts without a trust decision.

The decision belongs to the selected workspace and remains in effect until
you revoke it. Editing an instruction or adding another enabled behavior does
not produce repeated prompts for a workspace that you already trust. Inspect
or revoke the decision without loading the repository behavior:

```bash
yuj trust status -C /path/to/project
yuj trust revoke -C /path/to/project
```

Trust allows Yuj to load the listed repository behavior. It does not grant a
tool action, weaken the sandbox or permission policy, expose a secret, or
skip prompt-injection scanning. Read
[Trust repository-provided startup behavior](configuration.html#trust-repository-provided-startup-behavior)
for the complete boundary.

Use the local startup seam when you want to validate an installation or task
repository without contacting the model service:

```bash
yuj --dry-run "check local startup"
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
| Find or preview changes by source structure | `structural_search` | Off |
| Change files | One of `edit`, `apply_patch`, `udiff`, or `write` | Profile-selected; `_base` uses `edit`. |
| Change one Jupyter code or Markdown cell | `notebook_edit` | Off |
| Apply a previewed structural change | `structural_edit` | Off |
| Run shell commands | `bash` | On |
| Use a debugger, REPL, or other terminal-dependent program | `terminal_start`, `terminal_io` | Off; assistant sessions only. |
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
terminal_enabled = true
notebook_edit_enabled = true
structural_enabled = true
```

A model profile can limit the number of enabled tools that Yuj sends to the
model.

Yuj applies the sandbox and approval rules when it runs a tool. Read
[Model tools](model-tools.html) for every tool input. Read
[Sandbox](sandbox.html) for the access rules.

### Give Yuj the task

Run `yuj` with no task text to enter a multiline task. Type or paste the task,
then press Ctrl-D on an empty line. Yuj reads every line before it starts the
session, so pasted lines cannot remain in the shell input queue.

To start with another task source, give exactly one source.

| Form | Example |
| --- | --- |
| Multiline prompt | `yuj` |
| Text after the command | `yuj "Fix the failing tests."` |
| `--prompt-text` | `yuj --prompt-text "Fix the failing tests."` |
| `--prompt-file` | `yuj --prompt-file /path/to/task.txt` |
| Standard input | `printf '%s\n' 'Fix the tests.' | yuj --prompt-file -` |

Use a file for a long task:

```bash
yuj --prompt-file /path/to/task.txt
```

Use `-` to read the task from standard input. Yuj keeps the supplied line
breaks and whitespace:

```bash
printf '%s\n' 'Fix the parser.' 'Run its tests.' | yuj --prompt-file -
```

Use `-C`, `--cd`, or `--cwd` when the repository is not the current directory:

```bash
yuj -C /path/to/project "Update the parser and its tests."
```

These options also set the repository that session-selection commands prefer.

### Attach repository paths

Use a repeated `--path` option when you know which repository files the model
needs:

```bash
yuj --path src/parser.py --path tests/parser \
  "Fix the parser error shown by these tests."
```

Each value must name a file or directory inside the repository selected by
`-C`. A relative path starts at that repository. An absolute path must remain
inside it. Yuj sorts files inside a selected directory by repository path.

Yuj applies workspace trust before it reads the selected content. It also
applies the configured unreadable-path and ignore rules. Git-ignored content
and version-control metadata stay hidden. A selected directory skips hidden
descendants. A path that directly names hidden content fails.

Yuj accepts UTF-8 text in regular files. It rejects a missing path, binary
content, a symbolic-link path, a visible symbolic link below a selected
directory, or a path outside the repository. It also rejects a file larger
than 128 KiB. One task can select at most 20 paths, include at most 100 files,
and include at most 512 KiB before redaction.

Before model work, Yuj scans the text for prompt injection and applies its
secret-redaction rules. A blocking scan stops the session. A flagging scan
adds a value-free warning to the admitted text. The session saves only the
admitted text, repository-relative paths, content hashes, sizes, redaction
status, and value-free scan findings. It does not save the original bytes when
redaction changed them.

Yuj binds this evidence to the original task. A resume verifies the saved
copy and builds the same path attachment block. It never reopens the original
repository path. An offline replay uses the saved transcript that contains
the original block. `status` and `show` print the paths, sizes, hashes,
redaction status, and scan rule names. They do not print attached content.

### Attach local images

Attach one or more images to the task text with a repeated `-i` or `--image`
option:

```bash
yuj -i ./failure.png -i ./expected.webp \
  "Compare these screens and fix the rendering error."
```

Yuj accepts PNG, JPEG, GIF, and WebP files. It detects the media type from the
file bytes, not the file name. Each path must name a readable regular file;
the selected file itself cannot be a symbolic link. The limits are 20 images,
5 MiB for one image, 20 MiB across the coding session, and 8,000 pixels on
either dimension.

The selected provider and model must declare image support. Yuj stops before
the coding session or model request when the image, limit, provider, or model
check fails. Read [Image-capable models](configuration.html#declare-image-input-support)
for the capability rule.

Yuj copies the validated bytes into the session directory before model work.
Every request uses that session-owned copy. Changing or removing the source
path later does not change a resume or replay. `status` and `show` print the
saved name, detected media type, byte count, dimensions, and SHA-256 digest.
They do not print the source path or image bytes.

### Options for `yuj`

Yuj calls the starting group of settings a base.

| Option | What it does |
| --- | --- |
| `TASK ...` | Join the remaining words and use them as the task text. |
| `-C PATH`, `--cd PATH`, `--cwd PATH` | Edit this repository. The current directory is the default. |
| `--prompt-text TEXT` | Use this text as the task. |
| `--prompt-file PATH` | Read the task from this file. Use `-` for standard input. |
| `-i PATH`, `--image PATH` | Attach this local image. Repeat for more images. |
| `--path PATH` | Attach this repository file or directory. Repeat for more paths. |
| `-m NAME`, `--model NAME` | Use this model ID or known short name. |
| `--thinking LEVEL` | Use `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max` reasoning effort for every normal request. |
| `--plan-mode MODE` | Use `off` or require a nonempty `.solver/plan.md` and explicit `exit_plan_mode` before implementation. |
| `--permission-preset NAME` | Use `read-only`, `ask-before-changes`, or `allow-edits` for this session. |
| `--edit-format FORMAT` | Override the model profile with `exact`, `apply_patch`, `udiff`, or `whole`. |
| `--provider NAME` | Use `local`, `claude`, `codex`, `openai`, `anthropic`, `openrouter`, `zai`, or `custom`. |
| `--base-url URL` | Use this API base address. `custom` requires it. |
| `--api-key-env NAME` | Read the API key from this environment variable. |
| `-c PATH`, `--config PATH` | Apply this TOML file. Repeat the option to apply more files from left to right. |
| `--system-prompt PATH` | Add this file before Yuj's normal system prompt. The file may import another file with an `@path` line. |
| `--treatment` | Use the treatment base. This is the default. |
| `--no-treatment` | Use the plain base. |
| `--context NAME` | Use this context mode instead of the base default. |
| `--trust-workspace`, `--no-trust-workspace` | Trust or refuse repository-provided startup behavior for the selected workspace. Non-interactive starts need an explicit `--trust-workspace` until that workspace is trusted. |
| `--dry-run` | Complete local startup through the model-network boundary, write no session artifacts, and exit. |
| `-V`, `--version` | Print the installed Yuj version and exit. |

A context mode controls which earlier messages, saved facts, and current files
the model receives before its next action.

`--provider`, `--base-url`, `--api-key-env`, `--thinking`, `--plan-mode`,
`--permission-preset`, and `--edit-format` change only the new session. Yuj
saves these settings in the session's `provider.toml`.

Read [Fixed assistant permission presets](configuration.html#select-a-fixed-assistant-permission-preset)
for the one exact mapping and its precedence rules. A preset does not skip
plan mode, approval, command, runtime-mode, or sandbox checks.

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

### Receive a terminal notification

Set `assistant.notifications = "bell"` in a small TOML overlay when you want
an interactive run to ring the terminal bell after it stops. Yuj reports only
the short session reference and whether the run completed, failed, needs an
approval, or needs an answer.

The setting is off by default. It stays silent for piped and non-interactive
runs, and notification failure does not change the session result. Run
`yuj config` to check the resolved value before starting the task. See
[Receive a terminal notification](configuration.html#receive-a-terminal-notification)
for the exact setting and privacy limits.

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
| `--permission-preset NAME` | Save `read-only`, `ask-before-changes`, or `allow-edits` as the assistant default. |
| `--sandbox NAME` | Save `none`, `auto`, `bwrap`, `docker`, or `podman`. The default is `bwrap`. |
| `--sandbox-image IMAGE` | Save the already-local image for Docker, Podman, or automatic selection. Required when the setup-time choice resolves to Docker or Podman. |
| `--base-url URL` | Save the API base address. `custom` requires it. |
| `--api-key-env NAME` | Save `$ENV:NAME`. Yuj reads the key from that variable when a command starts. |
| `--api-key VALUE` | Save a key. Claude and Codex keep it in their provider credential file; other service choices keep it in `config.local.toml`. |
| `--force` | Replace an existing local settings file without asking. |

Sandboxing is optional. The shipped settings select `bwrap`. If the host
cannot run it, save another choice before the first task. A named backend must
work exactly as selected. `auto` stops if no supported backend passes its
startup check. Only `--sandbox none` permits unsandboxed model commands. Read
[Sandbox](sandbox.html#choose-a-mode) before you select a different boundary.

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
- the sandbox backends supported, available, and unavailable on this platform
- the configured sandbox choice and exact operational resolution

An unavailable selected sandbox is an error. `auto` tries installed supported
backends in platform order and never selects unsandboxed execution. An
explicit `none` is reported as not engaged. `doctor` checks only whether the
current directory itself contains `.git`; a missing root is a warning. It
does not check whether a parent directory is a Git repository.

For an OpenAI-compatible address that starts with `http://localhost`, a
missing selected model is an error. For other addresses, it is a warning
because some hosted services do not list every valid model ID.

A coding session repeats the same fail-closed startup resolution before model
or tool work.

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
| `--permission-preset NAME` | Use a fixed assistant permission preset for the smoke session. |
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
the same path before you run `status`, `show`, `usage`, `label`, `answer`,
`approve`, `reject`, or `resume` for that smoke session.

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

Inspect the current changes in an isolated session before you continue or
accept its result:

```bash
yuj diff SESSION
```

Give a full ID, exact label, short ID, unique ID start, `latest`, or `last`.
Yuj compares the retained worktree with the commit from which it created that
worktree. The patch on standard output includes tracked changes and untracked
files that Git does not ignore. This includes changes already committed on the
session branch. Redirect or pipe it like any other unified diff:

```bash
yuj diff SESSION > session.patch
yuj diff SESSION | less
```

Ownership and state go to standard error, so they do not enter the patch.
`ownership: session-worktree` means that Yuj verified the saved path, branch,
base commit, Git registration, and worktree metadata. `diff_state: clean`
means that the verified worktree still matches its base.

A direct session has no start-of-session tree checkpoint that can establish
ownership. Yuj prints `ownership: unknown`, `baseline: missing`, and
`diff_state: unavailable` instead of presenting other repository changes as
the session's work. A removed worktree or missing base commit is also explicit.
Unavailable results leave standard output empty and return status 2. The
command does not stage files, refresh the Git index, write session data, or
make a model request.

## Inspect sessions

Use these commands instead of reading raw files during normal work.

The trace is Yuj's time-ordered record of all run segments in a coding session.

| Command | What it shows |
| --- | --- |
| `yuj current` | Status details for the active or newest unarchived session in the current repository. If that repository has none, show the newest unarchived session. |
| `yuj status [SESSION]` | Session identity, run and archive state, model and sandbox, saved input summaries, pending operator action, and the next command. |
| `yuj show [SESSION]` | The status details plus saved paths, context and task sources, recent turns, and recent trace events. Use `--full` to show the complete saved turn view. |
| `yuj export [SESSION]` | A redacted Markdown report with the task, operator follow-ups, final assistant responses, tool summaries, usage, status, and evidence hashes. |
| `yuj diff [SESSION]` | A pipeable unified diff for a verified retained worktree, or an explicit reason that ownership cannot be established. |
| `yuj usage [SESSION]` | Run-segment count, input, output, and cached token totals, cache ratio, cost, and quota from persisted evidence |
| `yuj sessions` | A compact list of short ID, status, label, flags, and working directory for up to 20 recent unarchived sessions. |
| `yuj sessions --full` | Complete identity, status, model, path, and time fields for each matching session. |

The default `show` view shortens reasoning and tool results. It prints the five
most recent turns and ten most recent trace events. Show every saved turn and
trace event without shortening their text:

```bash
yuj show --full SESSION
```

Use `--turns N` or `--trace-lines N` to set a limit in either view. Use the
section controls when you need only part of the record:

```bash
yuj show --full --no-reasoning --no-trace SESSION
yuj show --no-tools --trace SESSION
```

| Command | Option | What it does |
| --- | --- | --- |
| `yuj status` | `SESSION` | Select a session. The default is `latest`. |
| `yuj show` | `SESSION` | Select a session. The default is `latest`. |
| `yuj show` | `--full` | Show the saved task, every turn, and every trace event without display shortening. Explicit turn and trace limits still apply. |
| `yuj show` | `--turns N` | Show this many recent turns. The summary default is 5. The full-view default is all turns. |
| `yuj show` | `--trace-lines N` | Show this many recent trace events. The summary default is 10. The full-view default is all events. |
| `yuj show` | `--reasoning`, `--no-reasoning` | Include or omit reasoning summaries. |
| `yuj show` | `--tools`, `--no-tools` | Include or omit tool calls. Omitting calls also omits their results. |
| `yuj show` | `--results`, `--no-results` | Include or omit tool results while keeping tool calls. |
| `yuj show` | `--trace`, `--no-trace` | Include or omit trace events. |
| `yuj show` | `--pager`, `--no-pager` | Force or disable the pager. Long terminal output uses a pager by default. |
| `yuj export` | `SESSION` | Select a session. The default is `latest`. Archived sessions are allowed. |
| `yuj export` | `--pager`, `--no-pager` | Force or disable the pager. Long terminal output uses a pager by default. |
| `yuj usage` | `SESSION` | Select a session. The default is `latest`. |
| `yuj sessions` | `--limit N` | List at most this many sessions. The default is 20. |
| `yuj sessions` | `--archived` | List archived sessions instead of unarchived sessions. |
| `yuj sessions` | `--all` | Include both unarchived and archived sessions. |
| `yuj sessions` | `--status STATUS` | Keep an exact saved status. Repeat the option to keep more than one status. |
| `yuj sessions` | `--cwd DIR` | Keep the exact saved working directory. Relative paths resolve from the current directory. |
| `yuj sessions` | `--label LABEL` | Keep the exact, case-sensitive manual label. |
| `yuj sessions` | `--full` | Show complete fields instead of the compact listing. |
| `yuj sessions` | `--select` | Number the matching sessions and ask for one choice. This requires an interactive terminal. |
| `yuj label` | `SESSION` | Select the saved session to change. |
| `yuj label` | `LABEL` | Set or replace the exact manual label. |
| `yuj label` | `--clear` | Clear the label instead of setting one. |
| `yuj fork` | `SESSION` | Select the stopped source session. This argument is required. |
| `yuj archive` | `SESSION` | Select the stopped session to archive. |
| `yuj unarchive` | `SESSION` | Select the archived session to restore. |
| `yuj purge` | `FULL_SESSION_ID --preview` | List the owned artifact entries and their logical byte total without changing them. |
| `yuj purge` | `FULL_SESSION_ID --confirm FULL_SESSION_ID` | Permanently remove the archived session after the two IDs match exactly. |

The current parser accepts any integer for `--turns`, `--trace-lines`, and
`--limit`. A value of `0` prints no matching rows. A negative `--turns` or
`--trace-lines` value also prints no tail. A negative `--limit` removes the
database row limit. Use a positive value for normal work.

Yuj reads `$PAGER` when it pages a long view. It uses `less -FRX` when `less`
is available and `$PAGER` is empty. Set `PAGER=cat` or pass `--no-pager` to
print directly. Yuj never starts a pager when you pipe or redirect the output.

These inspection commands read saved session evidence. They do not contact a
model or change the session, trace, transcript, or working tree. The full
`show` view expands the content saved in the trace. The raw provider request
log remains in the session artifact directory for direct inspection.

### Export a redacted report

Print one Markdown report to standard output:

```bash
yuj export SESSION --no-pager > session-report.md
```

The report includes the original task, digest-matched operator follow-ups,
final assistant responses, tool-call summaries, session status, usage, and
evidence hashes. It reads every saved transcript segment, but it does not copy
raw provider requests or responses into the report.

The report omits system prompts, configuration values and paths, credential
identity, workspace and artifact paths, and model reasoning. Yuj applies its
secret patterns to every remaining text field. It also removes common
authorization headers, private keys, secret assignments, and URL credentials.
Automatic redaction cannot identify every private fact, so read the report
before you share it.

`yuj export` accepts a full session ID, exact label, unique ID prefix, or
`latest`. It also accepts archived sessions. The command refuses linked raw
trace or transcript evidence instead of reading a file outside the session.
It makes no model or network request and writes no session data. An unchanged
session produces the same report each time.

### Inspect usage

Show the usage evidence for one coding session:

```bash
yuj usage [SESSION]
```

The report uses this shape:

```text
session_id: SESSION_ID
session_ref: SHORT_ID
segments: 2
input_tokens: 300
output_tokens: 50
cached_tokens: 120
cache_ratio: 40.00%
cost: unknown
quota: unknown
```

The command reads the session index and trace without changing either one. It
does not create a model client, read a credential, refresh authentication, or
contact a service. Repeating the command against unchanged files prints the
same report.

`input_tokens`, `output_tokens`, and `cached_tokens` cover every complete
model response in each recorded run segment, including responses from a named
model role or child agent. Yuj adds each segment fact once. It does not add a
turn row or child summary again after it reads that segment fact.

The cache ratio is the total cached-token count divided by the total input-token
count. Yuj prints it only when both counts are known and input tokens are
greater than zero.

Cost needs an exact decimal amount and currency in every run-segment fact.
Quota needs a remaining amount, limit, unit, and scope in every fact. Yuj does
not derive either value from the model name, service address, credential, or a
price table. Missing evidence makes only the affected report field
`unknown`. Incompatible currencies or quota meanings also produce `unknown`.
The current provider writers do not supply owned cost or quota evidence, so
both fields currently report `unknown`.

Older run segments do not have the all-response usage fact. If one coding
session mixes an older segment with a new segment, Yuj keeps aggregate fields
unknown instead of joining narrower turn data with all-response data. Read
[Saved files](harness_artifacts.html#trace-event-fields) for the exact trace
field contract.

### Label a session

Set or replace one manual label on a saved session:

```bash
yuj label SESSION LABEL
```

Clear it with:

```bash
yuj label SESSION --clear
```

A label has 1 to 64 ASCII characters. Its first character must be a letter.
The remaining characters may be ASCII letters, digits, `.`, `_`, or `-`.
Yuj stores the label exactly as entered. Labels are case-sensitive, so
`Release.One` and `release.one` are different labels.

The values `latest` and `last`, in any letter case, are reserved selectors.
A label also cannot look like a full session ID or an eight-character
hexadecimal short ID. Yuj rejects a duplicate label or a label that can
already select a session by ID prefix. It leaves the old label unchanged after
any rejection.

`sessions`, `status`, `current`, and `show` print `-` when a session has no
label. A label changes only the local SQLite session index. It does not change
the session ID, saved files, trace, replay, repository, worktree, approval,
clarification, correction, or archive status. The command makes no model
request, and labels do not enter model input or measurement mode.

### Fork a saved session

Create an independent child from one stopped saved session:

```bash
yuj fork SESSION
```

Give a full ID, exact label, short ID, or unique ID start. The argument is
required. `latest` and `last` are not accepted because forking is an explicit
one-time action.

The source must be unarchived, unlocked, not selected as the active session,
and free of unresolved approval, clarification, or correction input. If the
source is archived, run the exact `yuj unarchive SESSION` command printed by
the refusal before trying again. A fork makes no model request.

The child receives a new immutable session ID and records the source immutable
ID as its parent. Its status is `paused`, its finish reason is `forked`, and it
starts without a label or archive time. `status` and `show` print the parent
ID. Run the exact `yuj resume CHILD` command printed by `fork` to continue.

Yuj verifies and privately copies the source endpoint, including its prompt,
settings, trace, state, checkpoints, provider settings, attachments, and other
session evidence. Every mutable file belongs to the child. New model turns,
tool events, approvals, answers, corrections, labels, archive changes, and
artifacts can change only the child. Later changes or removal on either side
do not change the other side. The parent ID remains historical metadata.

Forking rejects symbolic links, malformed saved paths, and evidence that does
not belong to the source session directory. It verifies the source files and
source index row again before publishing the child index row. A handled
failure removes staged child files and any child worktree. Because the index
row is published last, an interruption cannot leave a half-created child that
session commands can resolve.

If the source owns a managed worktree, the child receives a distinct retained
worktree and branch at the source endpoint. Yuj copies the source worktree's
current files, including uncommitted and untracked files, without changing the
source path, branch, or files. It refuses the fork before publishing the child
when it cannot validate or create that independent worktree. Forking a source
that does not own a managed worktree does not create one.

### Archive a session

Archive one stopped session by full ID, exact label, short ID, or unique ID
start:

```bash
yuj archive SESSION
```

Archive is reversible operator metadata. It is not deletion, compression,
relocation, or export. It frees no storage. Yuj leaves the session ID, label,
repository, retained worktree identity, session directory, trace, replay
input, and every other saved byte unchanged.

Yuj refuses to archive the active-session pointer, a running or locked
session, or a session with a pending approval, clarification, or correction.
A refusal changes nothing. Stop the running work, resolve the pending input,
or start another session as the refusal directs.

Ordinary `sessions` output and automatic `current`, `latest`, `last`,
`resume`, `approve`, and `reject` selection skip archived sessions. List only
archived sessions with:

```bash
yuj sessions --archived
```

Give an archived session's explicit reference to `status`, `show`, or `usage`
to inspect it without restoring it. `status` and `show` print `archived: yes`,
the archive time, and the exact `unarchive` command. Other commands that
change a session refuse an archived session and print that same instruction.

Restore the session with:

```bash
yuj unarchive SESSION
```

Unarchive returns the session to the ordinary selection order that its
unchanged metadata defines. Repeating `archive` or `unarchive` reports
`changed: no` and leaves the current state unchanged. Both commands are local
SQLite changes. They make no model request, run no model tool, and do not
change measurement mode.

### Permanently purge an archived session

Purge is irreversible. Archive the stopped session and inspect the preview
before you confirm deletion.

Use the full immutable session ID for the preview:

```bash
yuj purge 20260825_120000_deadbeef --preview
```

The preview is read-only. It lists each directory and regular file in sorted
session-relative order. It also reports the exact sum of the regular files'
logical byte sizes. It does not print file contents, task text, credentials,
or credential IDs.

Repeat the same full ID to confirm permanent deletion:

```bash
yuj purge 20260825_120000_deadbeef \
  --confirm 20260825_120000_deadbeef
```

Purge does not accept a label, a short ID, an ID start, `latest`, or `last`.
It also rejects a missing or different confirmation ID. These checks happen
before Yuj starts deletion.

The session must be archived and unlocked. Resolve any pending approval,
clarification, or correction before you archive and purge it. Purge also
refuses an active-session pointer, a running status, malformed saved metadata,
or an artifact boundary that it cannot prove.

Remove a live retained worktree through its separate lifecycle before purge.
First unarchive the session. Then run `yuj worktree rm FULL_SESSION_ID`, and
archive the session again. Worktree removal keeps the saved path, branch, and
base commit as historical identity. Purge derives the expected worktree path
from the repository and session ID, verifies that Git no longer registers it
and that the path is absent, and never uses the saved path as deletion
authority. Purge never removes a repository, worktree, branch, credential,
another session, or measurement file.

Yuj derives the only deletion boundary from `<assist_home>/sessions/` and the
confirmed immutable ID. It does not use a saved absolute path as deletion
authority. It rejects symbolic links, hard-linked files, mount points,
unsupported entry types, changed entries, and paths that exceed the bounded
preview limits.

Before deletion, Yuj records a purge journal in `sessions.sqlite3`. It then
moves the exact session directory to `<assist_home>/purge-staging/` on the same
file system. Yuj removes only the journaled entries and removes the session row
last. The completed journal keeps the entry count, estimated bytes, manifest
digest, and completion time, but clears the path manifest.

If deletion stops, run the same preview command to see the journal phase and
the remaining entry and byte counts. Ordinary session commands stop resolving
an incomplete purge. Retry the same confirmed purge command after you fix the
reported local cause. Yuj checks the remaining staged entries against the
journal before it continues, so it never reports success while a row or owned
artifact remains.

Purging a parent does not change its child. Purging a child does not change its
parent. A retained child's `parent_session_id` remains the deleted immutable
ID as historical evidence, but that ID no longer resolves to a session.

Purge makes no model request, runs no model tool, and does not change
measurement mode. It performs no age-based cleanup, bulk cleanup, background
deletion, or automatic retention action.

### Select a session

A session has a full ID, a short ID, and at most one manual label. Give an
exact label or a unique start of either ID. Label prefixes do not select a
session.

Use `latest` or `last` to let Yuj choose. If you omit the ID, Yuj uses
`latest`.

Automatic selection and ordinary `sessions` output exclude archived
sessions. Use `sessions --archived` to list them. An explicit reference still
uses the complete session index, so `status`, `show`, and `usage` can inspect
an archived session.

The ordinary list fits an 80-column redirected view. Long labels and paths are
shortened there; `sessions --full` prints every value. Narrow the list before
you choose:

```bash
yuj sessions --status paused --cwd .
yuj sessions --all --label release-check
```

Use `--status` more than once to include several exact statuses. `--archived`
shows only archived sessions, while `--all` includes both archive states.
Filters run before `--limit` is applied.

Ask for a numbered choice when both input and output are attached to a
terminal:

```bash
yuj sessions --status paused --select
```

The result prints the immutable full ID, short ID, label, status, path, and a
`yuj show` command. It does not change the active-session pointer. Redirected
or piped use never opens the selector; an explicit `--select` then stops with
an error.

For `status`, `show`, `export`, and `usage`, Yuj chooses in this order:

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

`correct` also requires an explicit session reference. It does not select a
session automatically.

Yuj stops with an error if a reference matches no session or can select more
than one session. This also covers a label that conflicts with the prefix of a
session created later. Use the immutable full ID to resolve that conflict.

Automatic selection checks the active-session pointer first when the command
uses that pointer. It then searches the 200 newest saved sessions. An explicit
full ID, exact label, short ID, or unique ID start checks the complete session
index through one resolver.

### Session statuses

Archive state is separate from session status. `status` and `show` keep the
last run status and print archive state on another line.

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

## Correct a stopped session

Record one exact correction before you resume a stopped session:

```bash
yuj correct SESSION 'CORRECTION'
```

Yuj records the `CORRECTION` argument exactly. Quote it when it contains
spaces or shell characters. The command requires an explicit full ID, exact
label, short ID, or unique ID start. It refuses an empty correction, an
unknown or unclear session, an active or locked session, and a session that
cannot resume. It also refuses a second correction. A refusal leaves the
correction files and trace unchanged.

`status` and `show` print `correction: pending` until resume consumes the
correction. They also print its ID, SHA-256 hash, character count, and a
bounded JSON-quoted preview. They do not label it as an approval or a
clarification answer.

A correction supplies ordinary task input only. It does not approve or reject
a tool action, answer a clarification, change a permission rule, or change the
sandbox. You may record it while an approval or clarification is pending, but
you must resolve that separate request before resume can continue.

On resume, Yuj adds the exact correction as the last user message before the
first model request. It records consumption immediately before transport. A
later resume, interrupt recovery, or rewind does not add the consumed text
again. An unknown transport result also does not reopen it. If that resume
has image input, the images stay on their own text prompt; the correction
remains a separate text-only user message.

## Resume a session

Resume the session that Yuj selects:

```bash
yuj resume
```

Resume a named session:

```bash
yuj resume SESSION
```

Add follow-up text to the next resume:

```bash
yuj resume SESSION --prompt-text "Check the parser edge cases too."
```

Use `--prompt-file PATH` for a longer follow-up. Use `--prompt-file -` to read
the follow-up from standard input:

```bash
printf '%s\n' 'Also check empty input.' | yuj resume SESSION --prompt-file -
```

Attach new visual evidence to a resume with new follow-up text:

```bash
yuj resume SESSION --prompt-text "This screen shows the remaining error." \
  --image ./remaining-error.png
```

Each image resume requires exactly one follow-up text source, but follow-up
text does not require an image. Repeat `--image` for more images. Yuj validates
and saves the new image segment before the first resumed model request.

The command accepts a full ID, exact label, short ID, unique ID start,
`latest`, or `last`.

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

`resume` does not accept new model, context, or settings options. It accepts
one optional follow-up text source for the next model request. A prior
`correct` command may have recorded one separate correction for the next
resume. Ordinary resume without either form keeps its existing behavior.
It also accepts `--trust-workspace` or `--no-trust-workspace` when the saved
workspace needs an explicit trust choice.

For each follow-up, Yuj records the input source, character count, and content
hash in the trace. It does not copy the follow-up text into that event.

`resume` refuses a session whose clarification still lacks an answer. When an
answer is ready, resume adds the exact answer to one model request and records
its consumption before transport. A later resume does not add it again.

When a correction is pending, resume adds it only after any resume-time
conversation restore and pre-model hook. This keeps the exact correction as
the last user message. If the context needs compaction at this boundary, Yuj
uses the content-blind digest path. It does not run a checkpoint model or a
compaction hook before the correction reaches the main model. Approval and
clarification checks remain authoritative.

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

Yuj refuses `rewind` while a correction is pending. Resume and consume that
correction first. You may record a correction after a successful rewind; the
next resume adds it after restoring the saved conversation.

## Remove an isolated session worktree

When `[runtime].worktree` is enabled, Yuj keeps the session's worktree and
branch after every exit so `resume` sees exactly the same files. Remove it by
full ID, exact label, short ID, or unique ID start only when you no longer
need that workspace:

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

A permission rule, including one supplied by a fixed preset, can require
approval for any tool. Yuj also applies a fixed approval check to risky shell
commands.

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

For `write`, exact `edit`, `notebook_edit`, and `structural_edit` requests,
`show` compares the current workspace file with the proposed content. For
`apply_patch` and `udiff`, it shows the exact proposed patch. The preview names
every displayed path and labels the content as proposed and not applied.
Generating or viewing it does not change the workspace.

Yuj stores at most the first 120 lines and 16,000 escaped characters. A
truncation marker gives the displayed and original character counts. It also
escapes terminal control characters. If the source, proposal, or path cannot
be represented safely, `show` says that the preview is unavailable instead
of displaying an invented diff. Shell commands receive this result because
their file effects can depend on runtime behavior.

Use `yuj show --pager` when you want to force a pager. Automatic paging remains
the default at a terminal.

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
session. Rejection does not run the proposed tool. After an approved mutation
runs in an isolated session worktree, use `yuj diff SESSION` to inspect the
ordinary accumulated workspace diff.

The approval check does not replace the sandbox.

## Start another task

Start a new session after one task finishes:

```bash
yuj "Add a test for the fixed case."
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

After Yuj stages an incomplete purge, it keeps the remaining files here:

```text
<assist_home>/purge-staging/<session_id>/
```

The optional manual label exists only in `sessions.sqlite3`. The label does
not appear in the session directory or any model-facing record.

The optional archive time also exists only in `sessions.sqlite3`. Archiving
does not change or move the session directory, and it frees no storage.

A forked session's immutable parent ID exists in `sessions.sqlite3` and
`session.json`. The parent link is identification only. It does not share
writable state between session directories.

The purge journal exists only in `sessions.sqlite3`. It records the bounded
entry manifest while deletion is incomplete. A completed journal clears that
manifest and keeps only the summary and completion evidence. Session
selection does not read completed journals as session records.

For an installed package, `<assist_home>` is `$XDG_STATE_HOME/yuj`, or
`~/.local/state/yuj` when `XDG_STATE_HOME` is unset. For an editable/source
checkout it is `<checkout>/.llm_assist`. Set `HARNESS_ASSIST_HOME` to use an
exact alternative.

The session directory can contain the task text, settings, trace, segmented
model messages, session-owned image attachments and their digest manifest,
`.solver/state.json`, correction evidence, clarification evidence, approvals,
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

An unknown or unclear session reference also returns a nonzero status and
prints the reason.

## Measurements and replay

The installed `yuj` command is for normal coding work. A separate command runs
fixed measurements and replay.

Read [Measurements](measurement.html) for every measurement option and its
task-directory contract. Read [Replay](replay_mode_spec.html) for replay rules
and current limits.
