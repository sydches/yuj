# Yuj

Yuj means “to yoke” or “to harness” in Sanskrit.

Yuj is a coding-agent harness that watches the LLM and keeps it on course. It
maintains relevant context without extra LLM calls.

The model can read, change, and test code in a Git repository. Yuj can use an
online model service or a model server that you run yourself.

[Getting started](docs/getting-started.md) ·
[CLI reference](docs/using-yuj.md) ·
[Extend Yuj](docs/extending-yuj.md) ·
[Paper and results](paper/README.md)

## Install Yuj

You need Linux, Git, Python 3.11 or newer, and
[bubblewrap](https://github.com/containers/bubblewrap). Windows users can use
WSL2. macOS users need a Linux virtual machine.

```bash
git clone https://github.com/sydches/yuj.git
cd yuj
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
command -v bwrap
.venv/bin/yuj --help
```

The `test` extra installs `pytest`, which `yuj smoke` needs for its final
check. `command -v bwrap` shows the path to `bwrap`. `yuj --help` shows the
Yuj help text.

## Connect a model

Choose an online model service or a local model server.

For OpenAI, keep the API key in an environment variable:

```bash
export OPENAI_API_KEY='...'
.venv/bin/yuj setup --provider openai --model YOUR_MODEL_ID \
  --api-key-env OPENAI_API_KEY
```

For a local OpenAI-compatible server at `localhost:8080`, run:

```bash
.venv/bin/yuj setup --provider local --model YOUR_SERVED_MODEL_ID
```

See [Getting started](docs/getting-started.md) for Anthropic, OpenRouter,
Z.AI, custom servers, and other local server addresses.

## Run a coding task

Open the Git repository that you want the model to change:

```bash
cd /path/to/your-project
/path/to/yuj/.venv/bin/yuj doctor
/path/to/yuj/.venv/bin/yuj code \
  "Fix the failing tests and check the change."
```

Run a small test task before you use a real project:

```bash
/path/to/yuj/.venv/bin/yuj smoke
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
- Yuj normally runs model shell commands through Linux `bubblewrap`.
- You can inspect, pause, approve, reject, and resume a session.
- Yuj can shorten old command output when the model nears its input limit.
- Yuj can suggest a next step when the model repeats a known failed step.
- Yuj can optionally ask an isolated, read-only second model for a bounded next-turn advisory.
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
| `yuj setup` | Save settings for an online model service or a local model server. |
| `yuj doctor` | Check the settings, model connection, Git, and `bwrap`. |
| `yuj models` | List the models that the selected service offers. |
| `yuj code "task"` | Start a coding session. |
| `yuj run "task"` | Run the same command as `yuj code`. |
| `yuj smoke` | Test Yuj in a small throwaway directory. |
| `yuj current` | Show the active or newest session for this repository. If it has none, show the newest saved session. |
| `yuj status` | Show a session's status and the next user action. |
| `yuj show` | Show settings and recent session activity. |
| `yuj resume` | Continue a paused session. |
| `yuj approve` | Allow a shell action that needs approval. |
| `yuj reject` | Refuse a shell action that needs approval. |
| `yuj sessions` | List saved sessions. |

The [CLI reference](docs/using-yuj.md) lists every option for the installed
`yuj` command. It also explains how Yuj selects a session, what each status
means, and which exit status each command returns.

## Default and plain settings

Yuj calls its default group of long-task and recovery rules the treatment
base. `yuj code` uses this base and the `halflife` context mode by default.

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
| [Configuration](docs/configuration.md) | Setting order, model services, context modes, and environment variables |
| [Extend Yuj with TOML files](docs/extending-yuj.md) | Model runtime files, profiles, test runners, tool rules, and their code limits |
| [Sandbox](docs/sandbox.md) | `bwrap`, container modes, path access, and how to turn the sandbox off |
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

## License

The [MIT License](LICENSE) covers Yuj's original work unless a file says
otherwise. Included third-party code keeps its original license. See
[Third-party notices](THIRD_PARTY_NOTICES.md) and the included
[Apache License 2.0](LICENSES/Apache-2.0.txt).
