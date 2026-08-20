---
layout: default
title: Getting started
nav_order: 2
---

# Getting started

Use this guide to install Yuj. Connect a model next. Then start one coding
task. Read the [CLI reference](using-yuj.html) when you need every command and
option.

## Check the requirements

You need a writable Git repository and a model service or local model server.

| Your system | What you need |
| --- | --- |
| Linux | Python 3.11 or newer, Git, and the `bubblewrap` package |
| Windows | WSL2 with Linux, Python 3.11 or newer, Git, and `bubblewrap` |
| macOS | A Linux virtual machine with Python 3.11 or newer, Git, and `bubblewrap` |

The model can run on another computer if the Linux system can reach it.

## Install Yuj

Run:

```bash
git clone https://github.com/sydches/yuj.git
cd yuj
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
command -v bwrap
```

The `test` extra installs `pytest`. The `yuj smoke` command uses it for its
final check.

The last command must print the path to `bwrap` for the normal sandbox.

The install puts the `yuj` command in `.venv/bin/`. The examples use that
path until you move to the repository that the model will edit.

Read the [sandbox guide](sandbox.html) if another container already limits
shell access. The guide also explains how to run without `bwrap`.

## Connect an online model service

Yuj has settings for OpenAI, Anthropic, OpenRouter, and Z.AI. Keep the API key
in an environment variable.

For OpenAI, run:

```bash
export OPENAI_API_KEY='...'
.venv/bin/yuj setup --provider openai --model YOUR_MODEL_ID \
  --api-key-env OPENAI_API_KEY
```

Use `anthropic`, `openrouter`, or `zai` in place of `openai` for those
services.

Use `custom` for another OpenAI-compatible service:

```bash
export MY_MODEL_API_KEY='...'
.venv/bin/yuj setup --provider custom \
  --base-url https://provider.example/v1 \
  --model YOUR_MODEL_ID \
  --api-key-env MY_MODEL_API_KEY
```

Run `.venv/bin/yuj setup` with no options if you want Yuj to ask for each
value.

Use `--api-key-env` when you can. Yuj then saves only
`$ENV:VARIABLE_NAME` in `config.local.toml`.

The `--api-key` option saves the key itself in that local file. Git ignores
the file, but an environment variable gives the key less exposure.

## Connect a local model

Start an OpenAI-compatible server such as `llama-server` or vLLM.

Yuj uses `http://localhost:8080/v1` for the `local` setting unless you give
another address.

Ask the server for its model ID:

```bash
curl -fsS http://localhost:8080/v1/models
```

Use that ID when you set up Yuj:

```bash
.venv/bin/yuj setup --provider local --model YOUR_SERVED_MODEL_ID
```

Add `--base-url` if the server uses another address.

The `_base` profile uses the standard OpenAI tool-call format. Use a reviewed
profile under `profiles/` when a model needs a different message or tool-call
format.

Read [Run a local model](serving_overlay.html) if you want Yuj to start a
released local runtime for you.

## Check the setup

Run:

```bash
.venv/bin/yuj doctor
.venv/bin/yuj models
```

`doctor` checks the saved settings, model connection, selected model, Git,
and `bwrap`. It reports Git and `bwrap` problems as warnings. A coding
session can still stop later if its settings require `bwrap`.

`models` lists the exact model IDs from the selected service. It marks the
selected model with `*`.

## Run a small check

Run:

```bash
.venv/bin/yuj smoke
```

`smoke` creates a throwaway directory with one broken function. It asks the
model to fix the function. Yuj then checks the code change, runs one test, and
checks that no approval is waiting.

Yuj prints the directory path and keeps the directory after the check. Do not
give `--root` a directory that contains work you need.

A successful smoke task checks one small path through Yuj. It does not measure
the model's general coding skill.

## Start a coding task

Move to the Git repository that the model may edit:

```bash
cd /path/to/your-project
/path/to/yuj/.venv/bin/yuj code \
  "Fix the failing tests and check the change."
```

Use `--prompt-file /path/to/task.txt` for a long task description.

Use `--cwd /path/to/project` when you do not want to change the current
directory.

Each start or resume begins a run segment.

After most run segments, Yuj tries to stage all uncommitted changes in the
target repository and make a checkpoint commit. It does not try this when the
run pauses for approval. An interrupt can also stop before the attempt. Read
the [CLI reference](using-yuj.html) for the exact rule.

Yuj uses the treatment base by default. Add `--no-treatment` only when you
want the plain comparison base. Read [Treatment](treatment.html) before you
compare these settings.

## Continue

Read the [CLI reference](using-yuj.html) to inspect, pause, approve, reject, or
resume a session.

Read [Saved files](harness_artifacts.html) to learn what Yuj records and when
another tool may use each file.

Read [Extend Yuj with TOML files](extending-yuj.html) when you want to add
support for another model, test runner, or tool rule.
