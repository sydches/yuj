---
layout: default
title: Home
nav_order: 1
---

# Yuj documentation

Yuj lets a coding model read, change, and test a Git repository while it
manages the model's tools, shell access, context, and saved session.

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
| Sign in to Claude or Codex | [Getting started](getting-started.html#use-a-claude-or-codex-credential) |
| Answer a model clarification question | [CLI reference](using-yuj.html#answer-a-clarification-question) |
| Inspect token, cache, cost, or quota evidence | [CLI reference](using-yuj.html#inspect-usage) |
| Choose a fixed assistant permission preset | [Configuration](configuration.html#select-a-fixed-assistant-permission-preset) |
| See what tools the model can use | [Model tools](model-tools.html) |
| Start a local model server | [Run a local model](serving_overlay.html) |
| Understand the default and plain settings | [Treatment](treatment.html) |
| Inspect or change a model, time limit, or context mode | [Configuration](configuration.html) |
| Add a model, profile, rule, or named agent | [Extend Yuj with TOML files](extending-yuj.html) |
| Add trusted Python at the compaction boundary | [Compaction hooks](compaction.html) |
| Control shell access | [Sandbox](sandbox.html) |
| Understand the files that Yuj saves | [Saved files](harness_artifacts.html) |
| Run a fixed comparison | [Measurements](measurement.html) |
| Run recorded actions again | [Replay](replay_mode_spec.html) |
| Read the experiment and results | [Paper and results](https://github.com/sydches/yuj/tree/main/paper) |

## First commands

Install Yuj and connect a model first. Activate the environment where you
installed Yuj, or replace `yuj` below with that environment's `bin/yuj` path.

Move to the Git repository that the model may edit:

```bash
cd /path/to/your-project
```

```bash
yuj config
yuj doctor
yuj smoke
yuj code "Fix the failing tests"
yuj status
yuj show
yuj usage
```

Use `yuj approve` or `yuj reject` when a tool action needs your choice. If the
model asks a clarification question, run the exact `yuj answer` command shown
by `yuj status`. Run `yuj resume` after you approve, reject, answer, or stop a
session.

The [CLI reference](using-yuj.html) covers every command and option, including
provider sign-in, session selection, and exit statuses.

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
