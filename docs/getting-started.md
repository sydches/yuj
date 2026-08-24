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

### Install a built package

A wheel or source distribution is the normal choice when you do not want to
keep a Yuj source checkout. Yuj is not currently published to PyPI. Obtain a
trusted artifact, then run:

```bash
python3 -m venv ~/.venvs/yuj
~/.venvs/yuj/bin/pip install /path/to/yuj-0.1.0-py3-none-any.whl
command -v bwrap
~/.venvs/yuj/bin/yuj --help
```

You can use `/path/to/yuj-0.1.0.tar.gz` instead of the wheel. `pip` builds the
source distribution and installs the same package. Neither workflow needs the
original checkout after installation.

The examples below use `yuj`. Activate the environment with
`source ~/.venvs/yuj/bin/activate`, or replace `yuj` with
`~/.venvs/yuj/bin/yuj`.

Install `pytest` in the environment before using `yuj smoke`; the ordinary
`yuj code` command does not require the Yuj source tree or its test suite.

### Install an editable source checkout

Use this workflow when developing Yuj itself:

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

### What the installed package carries

The wheel carries one immutable runtime bundle containing `config.toml`, the
treatment and plain bases, treatment dictionaries and overlays, profiles and
their inheritance/rules/templates, named-agent definitions and prompts, tool
schemas and descriptions, and the security pattern registry. Language, shell,
and tool rule TOML stays beside the package code that loads it.

Paper and study files, benchmarks, CI/tests, sessions, traces, caches, local
settings, and private campaign or release material are not in the wheel. The
source distribution adds public documentation and source-only server examples,
but applies the same exclusions.

Yuj resolves the immutable runtime root in this order: an exact `YUJ_CONFIG`
file and its parent resource tree, an editable/source checkout, then the
installed bundle. It never copies that bundle into a task or user directory.
For an installed package, `yuj setup` writes mutable machine settings to
`$XDG_CONFIG_HOME/yuj/config.local.toml`, or
`~/.config/yuj/config.local.toml` when `XDG_CONFIG_HOME` is unset. Source
checkouts keep the existing checkout-local `config.local.toml`. Set
`YUJ_CONFIG_LOCAL` to choose an exact local-settings path.

Session state defaults to `$XDG_STATE_HOME/yuj`, or `~/.local/state/yuj` for
an installed package. A source checkout keeps `.llm_assist/` at its root.
`HARNESS_ASSIST_HOME` always selects an exact alternative. Target-repository
settings are not found implicitly: pass each settings overlay with `--config`.

### Verify the install without a model request

From a directory outside the Yuj source tree, run:

```bash
yuj --help
yuj code --help
yuj config
yuj config --json --agent research
yuj code --dry-run --cwd /path/to/target \
  "verify local startup"
```

`config` reports the runtime-resource origin and validates the shipped
resources. `code --dry-run` performs ordinary local startup through profile,
agent, tool, prompt, project-file, language-rule, security, and sandbox
validation, then stops at the model-network boundary. It creates no coding
session or run artifact and prints `Model network: not contacted`.

Read the [sandbox guide](sandbox.html) if another container already limits
shell access. The guide also explains how to run without `bwrap`.

## Connect an online model service

Yuj has settings for OpenAI, Anthropic, OpenRouter, and Z.AI. Keep the API key
in an environment variable.

For OpenAI, run:

```bash
export OPENAI_API_KEY='...'
yuj setup --provider openai --model YOUR_MODEL_ID \
  --api-key-env OPENAI_API_KEY
```

Use `anthropic`, `openrouter`, or `zai` in place of `openai` for those
services.

Use `custom` for another OpenAI-compatible service:

```bash
export MY_MODEL_API_KEY='...'
yuj setup --provider custom \
  --base-url https://provider.example/v1 \
  --model YOUR_MODEL_ID \
  --api-key-env MY_MODEL_API_KEY
```

Run `yuj setup` with no options if you want Yuj to ask for each
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
yuj setup --provider local --model YOUR_SERVED_MODEL_ID
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
yuj doctor
yuj models
```

`doctor` checks the saved settings, model connection, selected model, Git,
and `bwrap`. It reports Git and `bwrap` problems as warnings. A coding
session can still stop later if its settings require `bwrap`.

`models` lists the exact model IDs from the selected service. It marks the
selected model with `*`.

## Run a small check

Run:

```bash
yuj smoke
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
yuj code \
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
