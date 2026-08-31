---
layout: default
title: Measurements
nav_order: 11
---

# Run a measurement

Use the measurement command for fixed comparisons and replay. Use the
installed `yuj` command for normal coding work.

The two commands use the same harness. They have different inputs and saved
files.

A run segment is model work between a start or resume and the next end, pause,
or interrupt. A task may have more than one run segment.

Follow the [paper configuration guide](https://github.com/sydches/yuj/tree/main/configs/paper)
to reproduce a released paper comparison. Read
[Extend Yuj with TOML files](extending-yuj.html) to compare another model or
change model, language, or tool support. That change makes a new experiment.

## Before you run it

Run measurement commands from the Yuj repository. Use its Python environment.

Before a live measurement, make sure the selected model service is available.
The command reads its normal service address from `[server].base_url`. A dry
run and a replay without live handover do not need that service.

Prepare a fresh copy of each task repository. Yuj changes the task files and
may stage and commit dirty work in that repository.

## Run one task

Give the command a new `RUN_DIR`. Give `--task` the task repository.

Give at most one prompt override:

```bash
.venv/bin/python -m scripts.llm_solver RUN_DIR \
  --task /path/to/task \
  --prompt-text "Fix the failing tests."
```

If you omit `--prompt-text` and `--prompt-file`, Yuj reads
`/path/to/task/prompt.txt`.

Yuj does not copy a prompt override to `prompt.txt`. Read
[Saved files](harness_artifacts.html) for the records that keep this input.

The command returns status 0 when the task completes. It returns status 1 when
the task does not complete.

## Apply settings

The measurement command applies settings in this order:

1. `config.toml`
2. `config.local.toml`
3. each `--config` file from left to right
4. command-line values

A later value replaces an earlier value for the same field.

`RUN_DIR` holds run-level records, such as `session.json` and the harness log.
The task repository holds its trace, state, checkpoint, and metrics. In
one-task mode, the default transcript and savings paths come from the task
path, not from `RUN_DIR`. Use `--transcript-dir` and `--savings-dir` to choose
other paths.

The measurement command has no `--treatment` or `--no-treatment` option.
Select a released base with `--config`. Select its context mode separately.

For example, apply the released treatment base like this:

```bash
.venv/bin/python -m scripts.llm_solver RUN_DIR \
  --task /path/to/task \
  --config configs/regimes/treatment.toml \
  --context halflife \
  --prompt-text "Fix the failing tests."
```

This example is not a complete paper comparison. A paper comparison also fixes
the model runtime and detector limits. Follow the exact order in the
[paper configuration guide](https://github.com/sydches/yuj/tree/main/configs/paper).

## Run a prepared task set

An outside benchmark repository must prepare this layout:

```text
RUN_DIR/
└── repos/
    ├── task-a/
    │   └── prompt.txt
    └── task-b/
        └── prompt.txt
```

Run without `--task`:

```bash
.venv/bin/python -m scripts.llm_solver RUN_DIR \
  --config SETTINGS.toml \
  --context full
```

Yuj sorts the task directory names. It skips a task only when that task has a
`checkpoint.json` whose `status` is `completed`. It treats a broken checkpoint
as pending. It ignores a task directory that has no `prompt.txt`.

The outside benchmark owns task lists, task setup, launch control, checks,
scores, and their saved output. Those files do not belong in this repository.

## Options

### Task and model

| Input | What it does |
| --- | --- |
| `RUN_DIR` | Write run-level records here. Read `RUN_DIR/repos/` in multi-task mode. |
| `--task PATH` | Run one task repository. |
| `--prompt-file PATH` | Read the one-task prompt from this file. Add `--task`. |
| `--prompt-text TEXT` | Use this one-task prompt. Add `--task`. |
| `--model NAME`, `-m NAME` | Use this model ID or known short name. |
| `--port N`, `-p N` | Replace the port in `[server].base_url`. |
| `--system-prompt PATH` | Add this file before the normal system prompt. |

Use only one of `--prompt-file` and `--prompt-text`.

### Settings and run labels

| Option | What it does |
| --- | --- |
| `--config PATH`, `-c PATH` | Apply this settings file. Repeat the option to apply more files from left to right. |
| `--context NAME` | Use a registered context mode. The default is `full`. |
| `--edit-format FORMAT` | Override the model profile with `exact`, `apply_patch`, `udiff`, or `whole`. |
| `--max-sessions N` | Set the largest number of run segments for each task. |
| `--prompt-addendum TEXT` | Add this text to the task prompt. |
| `--variant-name NAME` | Save this name with the result. |
| `--tool-desc minimal` | Use the shipped `minimal` model-tool descriptions. |
| `--rumination-threshold N` | Set the no-change warning point as a percentage of `max_turns`. This option works only when `rumination_enabled` is true. `rumination_nudge_threshold_abs` overrides it. The guard never starts below `rumination_min_threshold`. |
| `--duplicate-abort N` | End a run segment after this many identical calls in a row when the duplicate-call guard is on. |
| `--require-intent` | Reject a tool call that has no assistant text. |

Read [Configuration](configuration.html) for context modes. Read
[Model tools](model-tools.html) for the tools and their inputs.

### Saved-file paths and output

| Option | What it does |
| --- | --- |
| `--transcript-dir PATH` | In one-task mode, save model messages here. |
| `--savings-dir PATH` | In one-task mode, save context-saving records here. |
| `--dry-run` | Print resolved settings and the task or pending-task list. Do not run a task. |
| `--verbose`, `-v` | Print debug-level process information. |

`--transcript-dir` and `--savings-dir` have no effect in multi-task mode.

`--dry-run` still creates `RUN_DIR`. It also writes a timestamped harness log
and `RUN_DIR/session.json`.

### Continue from a transcript

| Option | What it does |
| --- | --- |
| `--resume PATH` | Replay the last balanced request from this transcript without adding a message. Add `--task`. |
| `--resume-message-file PATH` | Add this file as a deliberate next user message. Add `--resume`. |

`--resume` requires `--task`. By itself, it restores the exact saved message
list and lets the model generate the next assistant turn. If the last
assistant turn ended partway through generation, Yuj drops that incomplete
turn and generates it again from the last saved request.

Use `--resume-message-file` when you want an explicit handoff or recovery
message. In that mode, the loader adds the last saved assistant reply when one
exists. It adds placeholder tool results for any unanswered tool calls. It
then adds the message from the file. An empty message file is invalid.

Resume does not restore task files, settings, trace rows, guard counters, or
`.solver/state.json`. Prepare the task repository yourself before you resume.

Do not combine `--resume` with `--prompt-file` or `--prompt-text`. Transparent
resume ignores those prompt values. An explicit resume message replaces them.
`--resume-message-file` requires `--resume`.

## Replay options

| Option | What it does |
| --- | --- |
| `--replay-from PATH` | Read a source run directory. Add `--task`. |
| `--replay-stop-turn N` | For a source with one run segment, stop after trace turn `N`. `0` means the full replay. |
| `--replay-allow-divergence` | Record a divergence and continue. The default stops at the first divergence. |
| `--replay-continue-live` | Request a live handover after a positive stop turn. |
| `--replay-overlay PATH` | Apply this settings file at live handover. |
| `--replay-watch-turns N` | Pass the intended live-turn limit. `0` makes no change. The current loop does not enforce the limit. |
| `--replay-extra-config PATH` | Add a measurement-only settings file after the source settings. Repeatable. |

Do not add `--model`, `--config`, or `--edit-format` with `--replay-from`.
Replay loads those values from the source run.

Current replay also requires the source `session.json` to list at least one
config path. A run made without `--config` lists none and cannot be a replay
source.

The other replay options have no effect without `--replay-from`.
`--replay-overlay` and `--replay-watch-turns` have no effect unless a live
handover occurs.

Read [Replay](replay_mode_spec.html) before you use any replay option. The
replay page lists current limits that the short option descriptions cannot
show.

## Number values

The current parser does not reject every zero or negative number.

Use a positive value for `--port`, `--max-sessions`,
`--rumination-threshold`, and `--duplicate-abort`. Use a non-negative value
for replay turn and watch values. A value of 0 has the special meaning stated
in the replay table.

## Measurement environment

Use `YUJ_HOLD_UNTIL=/path/to/file` when an outside launcher must delay the
first model request. Yuj waits until that file exists. It stops with an error
after 1,800 seconds by default. Set `YUJ_HOLD_TIMEOUT_S` to change that time.

The command can save a released setting name in `session.json`. Set
`YUJ_REGIME_NAME` to that name. Set `YUJ_REGIME_CATALOG_VERSION` when the name
comes from a versioned catalog. These values label the run. They do not change
the active settings.

`YUJ_MODEL_PARAMS_JSON` can hold a JSON object with model-server values that
the command should save as declared metadata. The command also saves these
individual values when you set them:

```text
YUJ_MODEL_FILE
YUJ_MODEL_QUANT
YUJ_MODEL_NAME
YUJ_CHAT_TEMPLATE_FILE
YUJ_CHAT_TEMPLATE_SHA256
YUJ_SAMPLING
YUJ_TEMPERATURE
YUJ_TOP_K
YUJ_TOP_P
YUJ_MIN_P
YUJ_REPEAT_PENALTY
YUJ_PRESENCE_PENALTY
YUJ_SEED
SEED
YUJ_LLAMA_SERVER_PID
YUJ_LLAMA_SERVER_BIN
```

An individual `YUJ_` value replaces the same field from
`YUJ_MODEL_PARAMS_JSON`. `YUJ_SEED` replaces `SEED`.

These values record a claim from the launch environment. Yuj does not use
them to configure the model server. Check the server record before you treat
them as proof of the server setup.

Read [Configuration](configuration.html) for `YUJ_STREAMING` and
`YUJ_PERSISTENT_BASH`.

## Help

Show the current parser help:

```bash
.venv/bin/python -m scripts.llm_solver --help
```

The command accepts `-h` and `--help`.
