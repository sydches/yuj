# Yuj

Yuj means “to yoke” or “to harness” in Sanskrit.

Yuj is an open-source AI coding agent for your terminal. It reads and edits
files, runs commands, and tests changes. Use it with a supported model
provider or a model you host yourself.

[Get started](docs/getting-started.md) ·
[Documentation](https://sydches.github.io/yuj/) ·
[Paper](https://arxiv.org/abs/2608.26218) ·
[Results](paper/README.md)

## What you can do

- Explain code, fix bugs, add features, and run tests.
- Give the model files, images, or a GitHub issue or pull request as context.
- Save and resume sessions, review changes, and export the work.
- Choose a sandbox and control which actions need approval.
- Configure tools and context handling, then inspect the recorded model
  messages and tool calls.

Yuj supports OpenAI, Anthropic, OpenRouter, Z.AI, and OpenAI-compatible
servers. Claude and Codex also support eligible subscription sign-in. Image
support depends on the model you choose.

## Get started

Follow the [getting-started guide](docs/getting-started.md) to install Yuj,
connect a model, and run your first task. It covers Linux, macOS, and Windows
through WSL2, along with model and sandbox setup.

The [CLI reference](docs/using-yuj.md) covers daily use, attachments,
permissions, Git checkpoints, and saved sessions.

## Research and customization

Yuj is also a harness for studying coding-agent performance. You can change
its tools, context handling, and settings, then inspect the recorded runs.
The [extension guide](docs/extending-yuj.md) explains what you can change in
TOML files and what requires Python code.

The [paper](https://arxiv.org/abs/2608.26218) studies how harness changes affect
coding performance while holding the model and other experimental conditions
fixed. The [results](paper/README.md) include the experiment design, task
outcomes, and provenance. Follow the [measurement guide](docs/measurement.md)
and [paper configuration guide](configs/paper/README.md) to run fixed
comparisons. Benchmark setup and scoring live in separate repositories.

## Guides

| Guide | What it covers |
| --- | --- |
| [Configuration](docs/configuration.md) | Model providers, context modes, and settings |
| [Model tools](docs/model-tools.md) | Available tools and their inputs |
| [Sandbox](docs/sandbox.md) | File access and command isolation |
| [Run a local model](docs/serving_overlay.md) | Start `llama-server` or vLLM |
| [Treatment](docs/treatment.md) | Default rules, plain settings, and experiment limits |
| [Saved files](docs/harness_artifacts.md) | Recorded messages, tool calls, and results |
| [Replay](docs/replay_mode_spec.md) | Replay recorded actions and continue with a model |

## License

The [MIT License](LICENSE) covers Yuj's original work unless a file says
otherwise. Included third-party code keeps its original license. See
[Third-party notices](THIRD_PARTY_NOTICES.md) and the included
[Apache License 2.0](LICENSES/Apache-2.0.txt).
