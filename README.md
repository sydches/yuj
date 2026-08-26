# Yuj

Yuj lets a coding model read, change, and test a Git repository while it
manages the model's tools, shell access, context, and saved session.

Yuj connects to an online model service or a model server that you run
yourself. The name Yuj means “to yoke” or “to harness” in Sanskrit.

[Getting started](docs/getting-started.md) ·
[CLI reference](docs/using-yuj.md) ·
[Extend Yuj](docs/extending-yuj.md) ·
[Paper and results](paper/README.md)

## Install Yuj

You need Git and Python 3.11 or newer. Sandboxing is optional. The shipped
settings select Linux [bubblewrap](https://github.com/containers/bubblewrap)
(`bwrap`). If the host
cannot run `bwrap`, select Docker, Podman, automatic sandbox selection, or
explicit unsandboxed execution during `yuj setup`. Yuj stops before model work
when the selected sandbox is unavailable. macOS supports Docker and Podman.
Windows users run Yuj in WSL2.

Install a built wheel or source distribution when you want to use Yuj without
keeping its source checkout:

```bash
python3 -m venv ~/.venvs/yuj
~/.venvs/yuj/bin/pip install /path/to/yuj-0.1.0-py3-none-any.whl
~/.venvs/yuj/bin/yuj --help
```

You can give `pip` the `.tar.gz` source distribution instead. Both artifacts
install the same `yuj` command and runtime defaults. Yuj is not currently
published to PyPI, so obtain an artifact from a trusted build or build it from
the public source.

Use an editable install to develop Yuj:

```bash
git clone https://github.com/sydches/yuj.git
cd yuj
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/yuj --help
```

The `test` extra installs `pytest`, which `yuj smoke` needs for its final
check. A built-package user can install `pytest` separately before using that
command. `yuj --help` shows the Yuj help text. Run `yuj setup` to save the
model and sandbox choices. Run `yuj doctor` to check both choices.

The package contains the public runtime files that Yuj needs. It does not
contain paper, study, benchmark, session, trace, or private campaign material.
See [Getting started](docs/getting-started.md) for the complete package
boundary and the writable settings locations.

## Connect a model

Choose an online model service or a local model server.

For OpenAI, keep the API key in an environment variable:

```bash
export OPENAI_API_KEY='...'
yuj setup --provider openai --model YOUR_MODEL_ID \
  --api-key-env OPENAI_API_KEY
```

Claude and Codex also support provider-scoped API keys and eligible
subscription sign-in. For browser sign-in, run one of:

```bash
yuj setup --provider claude --auth subscription --model YOUR_MODEL_ID
yuj setup --provider codex --auth subscription --model YOUR_MODEL_ID
```

Yuj opens the provider's sign-in page and saves the resulting credential in
the user configuration directory. Credential secret values stay out of the
target repository and session records; the private session index retains only
the non-secret credential identifier needed to keep a session pinned. Run
`yuj auth-status` to see the selected provider and authentication method
without showing the credential.

For a local OpenAI-compatible server at `localhost:8080`, run:

```bash
yuj setup --provider local --model YOUR_SERVED_MODEL_ID
```

See [Getting started](docs/getting-started.md) for Anthropic, OpenRouter,
Z.AI, custom servers, and other local server addresses.

## Run a coding task

Open the Git repository that you want the model to change:

```bash
cd /path/to/your-project
yuj doctor
yuj
```

Type or paste the task when Yuj prompts you. Press Ctrl-D on an empty line to
start the session. You can also pass the task directly:

```bash
yuj "Fix the failing tests and check the change."
```

Run a small test task before you use a real project:

```bash
yuj smoke
```

## Choose what you want to do

| You want to | Start here |
| --- | --- |
| Use Yuj for a coding task | [Getting started](docs/getting-started.md) |
| Reproduce a released paper comparison | [Measurements](docs/measurement.md), then the [paper configuration guide](configs/paper/README.md) |
| Run your own comparison or extend a model, language, or tool | [Extend Yuj with TOML files](docs/extending-yuj.md) |

## What Yuj does

Yuj calls one saved task record a coding session. A coding session can
continue through more than one run segment.

- The model can read files, change code, run commands, and run tests.
- You can attach PNG, JPEG, GIF, or WebP images when you start or resume a
  session with an image-capable model.
- Yuj runs model command surfaces through one explicit sandbox policy. Linux
  defaults to `bubblewrap`; Docker, Podman, automatic selection, and explicit
  unsandboxed execution are available where supported.
- You can inspect, label, pause, correct, approve, reject, resume, fork,
  archive, unarchive, or permanently purge a session.
- You can inspect persisted token and cache use without contacting the model
  service. Yuj reports cost and quota only when saved evidence supports them.
- The model can pause once for a missing fact. You can record one answer and
  resume the same session.
- Yuj can shorten old command output when the model nears its input limit.
- Yuj can suggest a next step when the model repeats a known failed step.
- Yuj can ask an isolated, read-only model to review a completed turn.
- Yuj can list validated Agent Skills and let the model read one when needed.
- A Yuj quirk is a TOML rule that users can read, share, and measure.
- Yuj saves the settings, model-message records, tool calls, and final status.
- After most run segments, Yuj tries to stage all uncommitted changes in the
  target repository and make a checkpoint commit.

The [sandbox guide](docs/sandbox.md) explains which paths and services a model
shell command can reach. The [saved-files guide](docs/harness_artifacts.md)
explains what each file contains and when another tool may read it.

Yuj does not include benchmark task lists, launchers, programs that check or
score completed work, or saved scores. Each benchmark keeps those files in its
own repository.

## Common commands

| Command | What it does |
| --- | --- |
| `yuj` | Start a coding session. Enter the task when prompted or pass it as an argument. |
| `yuj setup` | Save settings for an online model service or a local model server. |
| `yuj config` | Validate and explain every resolved setting without contacting a model. |
| `yuj doctor` | Check the settings, sandbox resolution, model connection, and Git. |
| `yuj smoke` | Test Yuj in a small throwaway directory. |
| `yuj status` | Show a session's status and the next user action. |
| `yuj resume` | Continue a paused session. |

The [CLI reference](docs/using-yuj.md) lists every option for the installed
`yuj` command. It also explains how Yuj selects a session, what each status
means, and which exit status each command returns.

## Default and plain settings

Yuj calls its default group of long-task and recovery rules the treatment
base. `yuj` uses this base and the `halflife` context mode by default.

`--no-treatment` selects the plain base and the `full` context mode. Use
this setting when you want to compare the two bases.

These CLI settings do not reproduce a complete paper comparison by
themselves. Each paper comparison also fixes the model, input limit, runtime
settings, and detector limits. The
[paper configuration guide](configs/paper/README.md) lists the exact files and
their order.

The study compares a complete treatment setting with a complete control
setting. It changes all treatment rules together. Its results do not show
which single rule caused a result.
Read the [treatment guide](docs/treatment.md) for the runtime rules. Read the
[experiment contract](paper/experiment.md) for the study design and tests.

## Documentation

| Guide | What it covers |
| --- | --- |
| [Getting started](docs/getting-started.md) | Install Yuj, connect a model, and run a first task |
| [CLI reference](docs/using-yuj.md) | Every installed `yuj` command, option, session status, and exit status |
| [Model tools](docs/model-tools.md) | The tools that Yuj can give the model and each tool's inputs |
| [Run a local model](docs/serving_overlay.md) | Start `llama-server` or vLLM from a released runtime file |
| [Treatment](docs/treatment.md) | The default base, the plain base, and the paper boundary |
| [Configuration](docs/configuration.md) | Inspect resolved settings; configure model services, context modes, and environment variables |
| [Extend Yuj with TOML files](docs/extending-yuj.md) | Model runtime files, profiles, test runners, tool rules, and their code limits |
| [Compaction hooks](docs/compaction.md) | Trusted Python hook input, return, validation, and trace contract |
| [Sandbox](docs/sandbox.md) | Explicit sandbox selection, path access, and unsandboxed security effects |
| [Saved files](docs/harness_artifacts.md) | Each saved file, its source, and its allowed uses |
| [Measurements](docs/measurement.md) | Run one task or an externally prepared task set with fixed settings |
| [Replay](docs/replay_mode_spec.md) | Run recorded model actions again and continue them live |
| [Paper and results](paper/README.md) | The experiment, task results, and source records |

You can also read these guides on the
[Yuj documentation site](https://sydches.github.io/yuj/).

## Reproduce a paper comparison

The installed `yuj` command is for normal coding work. The measurement command
is for fixed comparisons and replay.

Read [Measurements](docs/measurement.md) before you run
`python -m scripts.llm_solver`. Apply paper files in the order shown in the
[paper configuration guide](configs/paper/README.md).

## Develop Yuj

Run these commands from the Yuj repository:

```bash
.venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

Build and inspect both distribution formats with:

```bash
.venv/bin/pip install build
.venv/bin/python -m build
.venv/bin/python tests/distribution_contract.py dist/*.whl dist/*.tar.gz
```

## License

The [MIT License](LICENSE) covers Yuj's original work unless a file says
otherwise. Included third-party code keeps its original license. See
[Third-party notices](THIRD_PARTY_NOTICES.md) and the included
[Apache License 2.0](LICENSES/Apache-2.0.txt).
