---
layout: default
title: Treatment
nav_order: 10
---

# Treatment

Yuj calls its default group of long-task and recovery settings the treatment
base. The `--treatment` option selects this base.

Activate the environment that contains Yuj before you run a command on this
page. Otherwise, replace `yuj` with that environment's `bin/yuj` path.

## Use the default base

Run:

```bash
yuj "Fix the issue and run the relevant tests."
```

Yuj applies four groups of rules:

| Group | What Yuj does |
| --- | --- |
| `halflife` context | Keep recent tool results in the form first sent to the model. Shorten older tool results when the model nears its input limit. |
| Command handling | Correct known shell and task-format problems before Yuj sends a command. |
| Output handling | Limit large or repeated command output before it fills the model input. |
| Recovery | Record completed observations and apply the guards enabled by the selected settings. |

The assistant CLI applies the
[treatment base file](https://github.com/sydches/yuj/blob/main/configs/regimes/treatment.toml)
for command handling, output handling, and recovery. It selects the
`halflife` context mode separately.

Yuj may filter or shorten command output before the model sees it. The trace
saves the result that Yuj kept, not necessarily every byte from the command.
Yuj may put a large kept result in `.tool_output/`. Read
[Saved files](harness_artifacts.html) for the exact rules.

### CLI and paper settings differ

The CLI treatment base is not a complete paper setting.

Each paper comparison also fixes the model, input limit, runtime settings, and
detector limits. The
[paper configuration guide](https://github.com/sydches/yuj/blob/main/configs/paper/README.md)
lists the exact files and their order.

## Use the plain base

Run:

```bash
yuj --no-treatment "Fix the issue and run the relevant tests."
```

`--no-treatment` selects the
[plain base file](https://github.com/sydches/yuj/blob/main/configs/regimes/baselines/plain_long_solve.toml)
and the `full` context mode.

The paper starts its control settings from the same plain base. A complete
paper control also fixes its model, input limit, runtime settings, and detector
limits.

Use the plain base when you want to compare the two bases. Do not treat this
single CLI option as a complete paper reproduction.

## Change one value

Put your change in a small TOML file:

```toml
# my-yuj.toml
[loop]
max_turns = 160

[llm_hurdle_detector]
cadence_turns = 4

[llm_hurdle_detector.trace_nets]
reread_min_gap = 9
```

Apply the file after the base:

```bash
yuj --config my-yuj.toml "Fix the issue."
```

Repeat `--config` to apply more files. Yuj applies them from left to right.
A later value replaces an earlier value.

Your changed values make a new setting. Do not call it the shipped default or
an exact paper setting.

Read [Configuration](configuration.html) for the full setting order.

## Repeated source inspections

The current trace_nets backend records observations. It does not choose a
response from the medicine ladder. To enable the repetition policy, add these
values to your settings file:

```toml
[loop]
duplicate_guard_enabled = true
duplicate_warn_count = 2
duplicate_abort = 0
```

The second identical completed call gets one warning. From the third call,
eligible unchanged reads and source searches return a reference to the earlier
answer. A changed file, changed query or range, pending background work, or an
unknown search input requires fresh execution. Compaction and rewind clear the
reference. Arbitrary shell commands continue to execute. Repetition never ends
a session; the overall turn, time and token limits still apply.

The policy uses file digests for reads and native entry metadata for searches
over explicit task subtrees. These checks do not form an atomic filesystem
snapshot. The trace records when an answer was reused without executing its
tool again.

## Change only the context mode

Keep the other treatment settings. Choose another context mode:

```bash
yuj --context full "Fix the issue."
```

Read the [CLI reference](using-yuj.html) for every command-line option. Read
[Saved files](harness_artifacts.html) to see which records remain complete
when a context mode shortens the model's view.
