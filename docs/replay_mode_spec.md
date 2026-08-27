---
layout: default
title: Replay
nav_order: 12
---

# Replay a saved run

This page states the replay contract. It also lists current limits where the
code does not yet meet that contract.

Replay gives saved model replies to Yuj in their original order. Yuj runs the
saved tool calls again on a fresh copy of the task. It writes a new run. For a
source with one run segment, it can stop after a chosen trace turn.

A replay that finishes does not yet prove an exact match. The current code
compares the exact ordered tool names when the saved input contains a `tools`
array, but it does not compare full model requests. It can miss some tool
calls. It cannot safely align turns from a source with more than one run
segment. Read
[Current fidelity limits](#current-fidelity-limits) before you report a replay
result.

Use these terms on this page:

| Term | Meaning |
| --- | --- |
| Run segment | Model work between a start or resume and the next end, pause, or interrupt. |
| Source run | The saved run that replay reads. |
| Replay run | The new run that replay writes. |
| Fidelity check | A check that the replay run still matches the source run. |
| Divergence | A difference between the replay run and the source run. |
| Handover | The point where Yuj stops using saved replies and starts using a live model. |
| Config parity | A replay requirement: use the same config bytes and settings as the source run. |
| Normalization rule | A named rule that removes a value that may change when the same action runs again. |

## What replay is for

Use replay to repeat a reported bug one turn at a time. Use it to check whether
a Yuj code change alters an old run.

A scientific study can use replay. The study does not own replay.

## What replay reads

Each saved file has one role:

| File | What replay reads from it |
| --- | --- |
| `.trace.jsonl` | The main time-ordered event record, session numbers, per-session 0-based turn numbers, saved tool calls and results, and recorded lifecycle-hook effects. |
| `subagents/<id>/.trace.jsonl` | The exact terminal result and accounting for the matching parent `subagent` event. |
| Transcript | The saved pre-profile input and model reply for each turn. Replay checks the input fields named below. An image-bearing input keeps its transport-encoded image blocks here. |
| `attachments.json` and `attachments/` | Nothing directly. Live assistant start and resume use these files to verify image bytes before writing the model request to the transcript. |
| `path_attachments.json` and `path_attachments/` | Nothing directly during offline replay. Live assistant start and resume verify the admitted text and rebuild the path-labelled input block before writing the model request to the transcript. |
| `clarification_request.json` | The exact assistant question and request identity. |
| `clarification_answer.json` | The exact operator answer and its hash. |
| `clarification_consumption.json` | The answer hash and the one permitted assistant-resume delivery attempt. |
| `correction.json` | The exact operator correction, correction and session identities, source run segment, and text hash. |
| `correction_consumption.json` | The correction hash, model-facing transcript segment, and the one permitted assistant-resume delivery attempt. |
| `.solver/state.json` | Nothing directly. This file is a view built from `.trace.jsonl`. |
| `session.json` | The current replay loader reads the model, config paths, and context mode. It ignores the other settings in this file. |

The trace is the source of truth for events. The transcript supplies the full
model replies that replay returns.

For an image-bearing turn, replay uses the saved transcript record and never
reopens the original local image path. The saved attachment files prove the
bytes used by the live assistant, but the general replay check still does not
compare every input message block. Treat the recorded image block as evidence,
not as a full request-parity proof.

For a task with repository-path attachments, replay uses the saved transcript
record and never opens the original repository paths. The path manifest and
admitted text prove the path identity and content hashes used by the live
assistant. A live assistant resume verifies those saved files before it builds
the next request.

A long run or an assistant resume may split its transcript into numbered
files. If the main file is `repo.log`, replay first reads
`repo.pre_seg_1.log`, `repo.pre_seg_2.log`, and each later numbered part. It
then reads `repo.log`. Replay joins the parts and gives their turns one
continuous order.

## What happens on each turn

1. When the saved input has a `tools` array, the replay client compares its
   ordered tool names with the current request's ordered tool names.
2. The replay client returns the saved model reply.
3. Yuj sends each saved tool call through the normal tool code and sandbox,
   except for the recorded execution exceptions described below.
4. Yuj updates the normal guard counters.
5. Yuj writes a fresh trace.
6. When `state.writer_enabled` is true, Yuj writes `.solver/state.json`.
7. Yuj runs the tool-call fidelity check that the current run loop calls.

Replay does not restore a saved working tree. It runs the saved actions again
on the fresh task copy.

Some recorded features need special replay behavior:

| Feature | What replay does |
| --- | --- |
| Conversation and file rewind | At the recorded boundary, save the replay's own messages and file checkpoint, then run the normal rewind with the saved target and reason. Compare source and target turns, reason, and delivery mode. Do not compare the Git object ID because the replay creates new objects. A next-session rewind still has the multi-segment limit below. |
| Model `checkpoint` and `rewind(report)` | Run both through their normal turn-boundary handlers. Restore the conversation and report, but leave file changes in place. The new trace remains append-only. |
| Background command | Check the saved start-command hash. Return recorded poll bytes and consume recorded kills without starting a process. |
| Lifecycle hook | Match the saved hook position and apply its recorded block, rewrite, note, timeout, or error. Never start the external command. Stop if the configured command differs from the source. |
| Named subagent | Verify the saved child identity, result hash and size, turns, and tokens. Return the recorded final text instead of running a child model. Copy the child trace when the replay writes to another run directory. |
| Stream rule | Reproduce the saved interrupt and hidden injection, then consume the next response for the same logical turn. Do not read the current rule file to rebuild the retry. |
| Assistant image input | Use the recorded turn and never reopen or substitute the original local file. The general request-check limit below still applies. |
| Assistant clarification | Require one matching request, answer, and consumption record and one of each matching trace event. Return the saved `ask_user` reply, then replace context with the exact recorded `messages` array from the next request. Do not contact an operator or model, enter `input_required`, or create clarification files in the replay run. |
| Paused-session correction | Require matching correction and consumption records plus matching creation and consumption trace events. Add the exact recorded correction only at its source resume request. Write replay trace events, but do not contact an operator or create correction files in the replay run. |

The ordinary tool-call fidelity check still compares every model-visible result
from these paths. Focused rewind tests compare each model-facing message, but
the general replay command still lacks a full request check.

The replay run writes a new transcript. For each replayed turn, the input block
holds `messages` and `tools` from before model-profile conversion. The output
block copies the recorded model reply. Yuj flushes the file after each turn.

One logical turn may therefore consume more than one transcript pair. For a
stream-rule interrupt, the saved partial output remains valid JSON and includes
the internal marker and exact hidden-injection record. The replay trace writes
the trigger and injection events again.

After a handover, the live client adds HTTP request and response data to the
same file. A replay-to-live transcript can therefore contain two input formats.

## How Yuj checks a replay

The contract requires a full request check and a tool-call/result check. The
current code implements the second with the limits below. For requests, it
implements only one narrow subset:

| Check | Contract | Current code |
| --- | --- | --- |
| Model request | Compare the complete post-profile request for the same turn. | Compare only the ordered tool names from the pre-profile request, and only when the source transcript contains `tools`. |
| Tool call and result | Compare the tool, action, and result after normal execution. | Run this check for the source row that the current loader retains. |

The tool-name check proves that a saved `load_tools` call changes the same next
request. When the transcript has no `tools` array, Yuj skips this check. It does
not infer the array from `tools_activated` trace rows.

Measurement requests never contain `ask_user`. For a source with one validated
assistant clarification or correction exchange, the tool-name check removes
only that name from the recorded assistant surface before comparing it with
the measurement surface. Every other tool name and its order remain strict.

The current run loop separately calls the tool-call and result check. For each
tool call that reaches this check, Yuj applies these rules in order:

| Field | Current rule |
| --- | --- |
| `tool_name` | Compare the names after the normalization rules run. |
| `args_summary` | Compare the action text after the normalization rules run. |
| `output_sha256` | If both rows contain this field, compare the raw hashes. |
| `result_summary` | If either row lacks `output_sha256`, compare these short result parts after the normalization rules run. |

Read an exit status, pass or fail result, and error class only from the named
fields that Yuj writes for them. Do not guess these values from output text.

Replay stops at the first divergence by default. The replay log records the
reason and exact turn. An empty output does not prove that two runs match.

Use `--replay-allow-divergence` only when the replay run must continue after a
divergence. Yuj still records the divergence.

### Current fidelity limits

The current code has five important limits:

- The current code has no full request comparison. It compares ordered tool
  names when the source input contains `tools`, but not their schema bodies or
  descriptions. An unused helper compares only the trailing tool-result
  messages. The code does not compare all messages, model and token fields, or
  model-profile conversion.
- The tool-call check does not check trace turn `0`. Current turn-number
  handling treats `0` as a missing value.
- The source trace loader keeps only one tool-call row for each turn number.
  When one turn has several tool calls, each later row replaces the earlier
  row. A later session also replaces the same turn number from an earlier
  session. The fidelity check can therefore check at most one such tool call.
- When both trace rows contain `output_sha256`, Yuj compares the raw hashes.
  It does not apply the normalization rules before that comparison.
- Replay still runs when the source trace is missing. It then checks no tool
  calls. A strict replay with no source trace does not prove tool fidelity.

Treat these as code limits, not as weaker replay rules.

## Stop replay or continue with a live model

For a source with one run segment, use `--replay-stop-turn N` to stop after
Yuj runs 0-based trace turn `N`. At that point, the task copy, conversation,
and counters include work through turn `N`.

Use `0` to replay the full source run.

The current CLI cannot stop or hand over after trace turn `0`. The value `0`
turns off the stop test and means full replay.

A trace starts its turn numbers again for each run segment. A transcript
keeps one count across all run segments. The current replay client uses the
transcript count for `--replay-stop-turn`, but it uses only the trace turn
number for fidelity checks. It does not reject a source with several run
segments.
Use a source with one run segment when exact stop and general tool fidelity
checks matter. Recorded clarification and correction transitions are narrow
exceptions. Replay joins their assistant transcript segments and validates
the exact operator evidence at the next request, but the general multi-segment
tool-call limits still apply.

Add `--replay-continue-live` to request a live handover. The current handover
works only when the stop turn is greater than `0`, the source has another
recorded turn, and the run reaches another model call. It does not occur
after a full replay, at the final recorded turn, or when the stop turn ends the
run segment.

Yuj refuses `--replay-continue-live` when the source contains a recorded
clarification exchange. That replay must remain offline, so no later model or
operator can replace the recorded answer.

For a source with one run segment, the run loop calls handover on the next
loop turn.
For a stop after turn `N`, the process log therefore names turn `N+1` as the
handover turn.

At handover, `--replay-overlay PATH` can apply one TOML file.
`--replay-watch-turns N` is meant to limit the live part. The current loop fixes
its number of turns before handover. This flag changes `max_turns`, but it does
not limit the live model calls. Do not rely on it as a live-turn limit.

The replay contract requires Yuj to record the handover turn, overlay path and
SHA, and watch limit in the trace and controller ledger. The current code
writes the turn, overlay path, apply status, and watch window only to the
process log. It does not calculate an overlay SHA for this record.

If the handover overlay fails to apply, the client has already switched to the
live client. The current loop can then treat the saved stop reply as a normal
reply with no tool call. When implicit completion is on, it can mark the task
complete instead of reporting the overlay failure as the task result. Check
the process log before you accept a handover result.

## Use the source settings

The replay contract covers every registered context mode. This includes
`full`, `compound`, `halflife`, and the other modes in
[Configuration](configuration.html).

The replay contract requires replay to adopt every source setting that can
change the run.

The current loader reads only the model, context mode, and config file paths
from the source `session.json`. It then loads those config files. It refuses
`--model` and normal `--config` options with `--replay-from`.

`session.json` lists only files passed with `--config`. The loader also reads
the repository's `config.toml` and `config.local.toml`, when the local file
exists. Replay does not check these files against the source run unless the
user also passed them with `--config`. Changing `config.toml` can therefore
change replay. Adding, changing, or removing `config.local.toml` can also
change replay.

The current loader does not restore `system_prompt_path`, `cli_overrides`, the
resolved-config hash, or model-runtime settings from `session.json`. As a
result, it does not restore source values set with `--system-prompt`, `--port`,
`--max-sessions`, `--rumination-threshold`, `--duplicate-abort`,
`--require-intent`, `--prompt-addendum`, `--variant-name`, or `--tool-desc`.
Do not claim config parity when the source run used one of these values.

The loader normally does not restore the source task prompt or resume inputs.
The new replay command can supply different `--prompt-text`, `--prompt-file`,
`--resume`, or `--resume-message-file` values. These values can change the
model input. Supply the same input manually, and record what you supplied.

A recorded clarification is one narrow exception. Replay takes the exact
request messages after the recorded answer from the next transcript segment.

A recorded correction is another narrow exception. Replay takes the exact
text from `correction.json`, uses the consumption record's transcript segment
to locate the saved model-input boundary, and requires that text as the final
user message there. An earlier segment may contain the same text, and
compaction may remove that earlier copy. Later saved requests may retain the
correction as ordinary conversation history.

Use `--replay-extra-config PATH` only for measurement code that does not change
what the model sees. Repeat the option to add more than one file. A
tool-name check detects a changed active tool-name set or order when the
source transcript recorded `tools`, and a later tool-call check may find other
model-visible changes. The current code still cannot prove full request parity.

The parser uses `full` as both the normal context default and the sign that the
user omitted `--context`. As a result, replay can treat an explicit
`--context full` as omitted. Replay then adopts the source mode. Check the replay
`session.json` instead of using this flag to test context refusal.

### Check the recorded config files

Replay reads `session.json` from the source run root or its `harness_run/`
directory. It refuses a source that lacks a recorded model or config path.

A run made without `--config` records an empty config path list. Current replay
refuses that run as a source. Pass at least one `--config` when you make a run
that you plan to replay.

Each recorded config path must still exist. Replay refuses a missing original
file before it checks `replay_layers/`.

When `session.json` has a SHA for a config file, replay checks the current file
against that SHA.

If the original file exists but its bytes changed, put an exact copy here:

```text
<source_run_dir>/replay_layers/<recorded_sha><file_suffix>
```

Replay accepts the copy only when its SHA equals `<recorded_sha>`.

If the original path still exists but the recorded bytes are unavailable, put
a replacement at the same `replay_layers/` path. Set this variable:

```bash
export YUJ_REPLAY_LAYER_SUBSTITUTE_OK=1
```

This escape works only when the original config path still exists but has
different bytes. Yuj prints a config-parity warning. The replacement breaks
byte-for-byte config parity with the source run. Record and report this choice.

When `session.json` has no SHA for a config file, Yuj uses the file at the
recorded path without a byte check.

## Use trace turn numbers

Within one run segment, use the 0-based `turn_number` from `.trace.jsonl`
everywhere replay reads or writes a turn number. This includes
`--replay-stop-turn`, capture points, and fidelity events.

Transcript blocks use 1-based numbers. The replay client maps those numbers to
trace turns for a source with one run segment. For a source with several run
segments, use the current limitation described above.

## Normalize values that can change

Some commands return a different value each time even when the same action
runs. A process ID, temporary file name, and clock time are common examples.

Yuj names its current set of normalization rules `replay_volatile_norm_v14`.
The code applies the rules to `tool_name` and `args_summary`. It also applies them to
`result_summary` on legacy rows that do not both have an output hash.

The normalization rules cover these earlier values:

- Docker overlayfs device IDs
- time lines from `stat`
- `sed -i` temporary names in error messages
- times in `diff` headers
- Python temporary names
- total-block counts from `ls`
- the specific `SIGPIPE` exit pair `141` and `0`
- network-failure noise lines
- pytest durations
- item order in a printed set
- containerd snapshot IDs
- known long `chown` error blocks
- stash-pop SHAs
- SymPy runner seed, hash, and duration lines
- directory sizes from `ls -la`
- setuptools-scm hash suffixes in Sphinx versions
- `dd` time and speed
- omitted-character and similar-line counts in Yuj notes
- Sphinx LaTeX dates
- ISO dates from test runners
- directory sizes from `stat`
- the specific Python shutdown exit pair `120` and `1`
- logged outputs that contain the same lines in another order

Version `v12` adds:

- `CDLL` handle output
- `PID: N`
- setuptools-scm `.dYYYYMMDD` local dates
- the Matplotlib font-cache message
- `os.stat_result(st_dev=N)`
- `df` overlay usage rows
- temporary suffixes in `patch` permission errors

Version `v13` adds:

- `/proc/*/fd` `pipe:[N]` inode values
- GNU `stat` `Inode: N` values

Version `v14` adds the process ID in `/proc/<PID>/fd` paths.

The replay client writes `replay_volatile_norm_v14` to the log when it starts.
Source comments state which changing value each rule handles.

When replay reaches its stop marker or the end of the source, the client logs
the number of tool calls that it checked. It also logs how many checked calls
used a normalization rule. The log lists those turn numbers and any accepted
changes in line order. A strict divergence or another failure can end without
this report.

Add a normalization rule only after a real replay stops and names the turn.
Do not add a rule because a value might change. Any divergence outside the
named rules stops a strict replay.

The public source keeps the rules. Private study records stay outside this
repository.

## Rules that must stay true

- Write a new trace for the replay run.
- Never change the source run.
- When the state writer is on, build `.solver/state.json` from the new trace.
- Require the same trace data to make the same `.solver/state.json` when the
  state writer is on.
- Give the model only the current context. Do not give it hidden state from an
  earlier model call.
- Use the same sandbox rules as a live run.
- Never launch configured lifecycle-hook commands while replaying; consume
  their source `hook` rows.
- For a recorded clarification, never contact an operator or live model and
  never create pending-input state in the replay run.
- For a recorded correction, never contact an operator, reopen the correction,
  or create correction files in the replay run.
- Do not add a saved file format only for replay.
- Do not require a saved snapshot or branch bundle to restore a replay.
- Treat any cached task copy at turn `N` as disposable.
- Never treat a cached task copy as the source of truth.
- Do not use a later source turn to build or check the current replay turn.

Read [Sandbox](sandbox.html) for the shell boundary.

## Run a replay

Use the project virtual environment or another Python environment that has Yuj
installed.

Use a source run with one run segment. The current CLI does not reject a
source with several run segments, but its stop and general tool fidelity turn
numbers are not safe for that source. A validated assistant clarification can
cross its one resume boundary as described above.

Make a fresh copy of the task. Start it with the same files as the source run.

Give the fresh task the same task prompt as the source run.

Give `--task` the fresh task path.

Give the positional `<run_dir>` argument a new directory for the replay run.

Give `--replay-from` the source run directory. That directory must contain
`session.json` or `harness_run/session.json`. Put the transcript under
`<source_run_dir>/harness_run/transcripts/*.log` or
`<source_run_dir>/transcripts/*.log`. For tool-call fidelity checks, the source
run must also contain `<source_run_dir>/host_task/.trace.jsonl`.

An installed-command session directory is also a replay source when it
contains `session.json`, `.trace.jsonl`, and `transcript.log`. Replay reads its
numbered `transcript.pre_seg_N.log` files first. A clarification source must
also contain all three clarification JSON files and the matching trace events.
A correction source must also contain both correction JSON files and the
matching creation and consumption trace events.

Keep one transcript log in the selected transcript directory. When several
logs match, the current code silently reads the first path in sorted order.

Replay can start without the source trace. It then performs no tool-call
fidelity checks.

Run a full replay:

```bash
.venv/bin/python -m scripts.llm_solver <run_dir> \
    --task <fresh_task_copy> \
    --replay-from <source_run_dir>
```

Add `--replay-stop-turn N` to stop at a selected turn.

Replay to turn `N`, request a live handover, and pass a watch value of five:

```bash
.venv/bin/python -m scripts.llm_solver <run_dir> \
    --task <fresh_task_copy> \
    --replay-from <source_run_dir> \
    --replay-stop-turn N \
    --replay-continue-live \
    --replay-overlay <candidate.toml> \
    --replay-watch-turns 5
```

The current loop does not enforce that watch value as a live-turn limit.

### Replay options

| Option | What it does |
| --- | --- |
| `--replay-from PATH` | Start replay from a source run directory. |
| `--replay-stop-turn N` | For a source with one run segment, stop after 0-based trace turn `N`. Use `0` to replay to the end. |
| `--replay-allow-divergence` | Record a divergence and continue. The default stops at the first divergence. |
| `--replay-continue-live` | Request a live handover after a positive stop turn. The source must have another recorded turn, and the solver must reach another model call. |
| `--replay-overlay PATH` | Apply one TOML file at live handover. |
| `--replay-watch-turns N` | Set the intended live-turn limit. The current loop does not enforce it. A value of `0` makes no change. |
| `--replay-extra-config PATH` | Add a measurement-only file after the source settings. Repeat the option to add more files. |

The current CLI cannot start from a direct transcript path. It loads
`session.json` before it resolves the transcript. Use the source run directory.

## Why replay does not use saved snapshots

Replay does not need a saved snapshot or branch bundle to restore the task.
When branch-bundle capture is enabled, replay can still write a bundle at the
stop turn. Replay does not read that bundle to restore its work.

| Saved snapshots | Replay |
| --- | --- |
| Save a prefix trace, rendered context, task copy, guard state, and check records for each branch point. | Save the normal source run once. Replay it to turn `N` when needed. |
| Capture only a point chosen during the live run. | Choose a saved trace turn later. The current CLI cannot stop at turn `0`. |
| Name a branch with a bundle path. | Name a branch with the source run, trace turn, and optional overlay. |
| Keep separate restore code. | Reuse normal tool code, sandbox, guards, trace writing, and state writing. |

## Acceptance checks

Use these checks before you claim that replay meets the full contract:

1. Replay a run whose commands have fixed results from start to end. Require
   the same command sequence and zero divergence. Require the report to name
   the exact turn for any divergence.
2. Replay to turn `N`. Hand over without `--replay-overlay`. Use greedy
   decoding, which always chooses the highest-scoring next token. Require the
   next command to match the source run.
3. Replay a source turn that contains two tool calls. Require a separate
   fidelity check for each call and each result.
4. Test source-turn alignment, strict stopping, the stop-turn boundary,
   handover overlay use, and prefix-only reads.
5. Replay one saved run for every registered context mode. Require zero
   divergence from start to end.
6. Replay one assistant clarification through its resumed answer. Require no
   operator or model contact, no destination clarification files, and exactly
   one replayed request, answer, and consumption event.
7. Replay one paused-session correction through its resume boundary. Require
   no operator contact, no destination correction files, exactly one exact
   user message, and one creation, consumption, and replay trace event.

The current limits above mean that a passing replay does not yet prove full
request, tool-call, handover, or config fidelity.
