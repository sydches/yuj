---
layout: default
title: Home
nav_order: 1
---

# Yuj documentation

Yuj is a coding-agent harness that watches the LLM and keeps it on course. It
maintains relevant context without extra LLM calls.

The `yuj` command starts and controls coding sessions. A coding session is one
saved task record. It can continue after a resume.

A Yuj quirk is a TOML rule that users can read, share, and measure.

## Start here

| You want to | Start here |
| --- | --- |
| Use Yuj for a coding task | [Getting started](getting-started.html) |
| Reproduce a released paper comparison | [Measurements](measurement.html), then the [paper configuration guide](https://github.com/sydches/yuj/tree/main/configs/paper) |
| Run your own comparison or extend a model, language, or tool | [Extend Yuj with TOML files](extending-yuj.html) |

## Find a topic

| You want to | Read |
| --- | --- |
| Find an installed `yuj` command or option | [CLI reference](using-yuj.html) |
| See what tools the model can use | [Model tools](model-tools.html) |
| Start a local model server | [Run a local model](serving_overlay.html) |
| Understand the default and plain settings | [Treatment](treatment.html) |
| Change a model, time limit, or context mode | [Configuration](configuration.html) |
| Control shell access | [Sandbox](sandbox.html) |
| Understand the files that Yuj saves | [Saved files](harness_artifacts.html) |
| Run a fixed comparison | [Measurements](measurement.html) |
| Run recorded actions again | [Replay](replay_mode_spec.html) |
| Read the experiment and results | [Paper and results](https://github.com/sydches/yuj/tree/main/paper) |

## First commands

Install Yuj and connect a model first.

Move to the Git repository that the model may edit:

```bash
cd /path/to/your-project
```

Replace `/path/to/yuj` with the path to your Yuj clone:

```bash
/path/to/yuj/.venv/bin/yuj doctor
/path/to/yuj/.venv/bin/yuj smoke
/path/to/yuj/.venv/bin/yuj code "Fix the failing tests"
/path/to/yuj/.venv/bin/yuj status
/path/to/yuj/.venv/bin/yuj show
```

Use `/path/to/yuj/.venv/bin/yuj approve` or
`/path/to/yuj/.venv/bin/yuj reject` when a shell action needs your choice.
Use `/path/to/yuj/.venv/bin/yuj resume` after you approve, reject, or stop a
session.

The [CLI reference](using-yuj.html) also covers `setup`, `models`, `run`,
`current`, `sessions`, every option, session selection, and exit statuses.

## What the study compares

Normal `yuj code` sessions use the treatment base. Add `--no-treatment`
to select the plain base.

A complete paper comparison also fixes the model, input limit, runtime
settings, and detector limits. The paper compares each complete treatment
setting with its complete control setting. It does not test one treatment rule
at a time.

Read the [experiment contract](https://github.com/sydches/yuj/blob/main/paper/experiment.md)
for the study design. Read the
[paper configuration guide](https://github.com/sydches/yuj/blob/main/configs/paper/README.md)
for the exact setting order.

The [repository README](https://github.com/sydches/yuj) gives the shortest path
from a new clone to a first task.
