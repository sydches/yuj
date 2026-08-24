---
layout: default
title: Saved files
nav_order: 9
---

# Saved files

Yuj saves files as it works. This page says who writes each file, when the
file is ready, and what may read it.

This page sets the main rules for those files and their use.

Read this page before a context mode, detector, audit, or scoring program
uses a file. If this page does not name the file or the planned use, add that
rule here first.

Related guides:

- [CLI reference](using-yuj.html) - Start, view, and continue sessions.
- [Measurements](measurement.html) - Run a fixed task or task set.
- [Treatment](treatment.html) - Control what the model sees and how Yuj acts.
- [Replay](replay_mode_spec.html) - Run recorded model replies again.
- [Sandbox](sandbox.html) - Limit model commands while Yuj writes its own files.

Yuj writes the core run files. A program from an outside benchmark may check or
score completed work. It may add other files around a measurement run. This
page includes both groups so that the source of each file stays clear. This
repository does not include the outside task lists, launchers, programs, or
their output.

## Files from the installed `yuj` command

The installed command keeps most records outside the target repository.

```text
<assist_home>/
├── sessions.sqlite3
└── sessions/
    └── <session_id>/
```

For an installed package, the normal `assist_home` is
`$XDG_STATE_HOME/yuj`, or `~/.local/state/yuj` when `XDG_STATE_HOME` is
unset. An editable/source checkout uses `<checkout>/.llm_assist`. Set
`HARNESS_ASSIST_HOME` to use an exact alternative.

| File | What it means |
| --- | --- |
| `sessions.sqlite3` | Index of coding sessions, active-session pointers, process locks, and the non-secret provider/authentication identity pinned to each managed-provider session. Credential IDs are internal and are not printed by session commands. |
| `<session_id>/prompt.txt` | Original task text. |
| `<session_id>/session.json` | Model, original target repository, provider and authentication method when pinned, context mode, starting config paths, and retained worktree path/branch/base commit when enabled. It never contains a credential value or credential ID. A later `provider.toml` for that coding session is not added to this file in the current code. |
| `<session_id>/provider.toml` | Model-service, thinking-level, plan-mode, or edit-format overrides given on the `code`, `run`, or `smoke` command. Present only when that command adds one of those overrides. |
| `<session_id>/.trace.jsonl` | Append-only event record across run segments. |
| `<session_id>/subagents/<id>/.trace.jsonl` | Separate append-only event record for one named child, including its exact terminal result and token counts. |
| `<session_id>/subagents/<id>/transcript.log` | Model messages for one live named child. |
| `<session_id>/.solver/state.json` | Current state view when the state writer is on. |
| `<session_id>/transcript.log` | Model messages for the newest run segment. Resume replaces this file. |
| `<session_id>/advisor.jsonl` | Isolated advisor requests, responses, read-only tool results, quarantine decisions, and accepted note bodies when the advisor is enabled. |
| `<session_id>/savings.jsonl` | Append-only record of context and output changes. |
| `<session_id>/system_log.jsonl` | Append-only record of harness warnings and internal events. |
| `<session_id>/checkpoint.json` | End status for the newest run segment. Resume replaces this file. |
| `<session_id>/metrics.json` | Measures for the newest run segment. Resume replaces this file. |
| `<session_id>/.shadow_git/` or the task telemetry sibling's `.shadow_git/` | Independent Git object store for enabled file checkpoints. It is outside the model's task view. |
| `<session_id>/rewind_snapshots/*.json.gz` or the task telemetry sibling's `rewind_snapshots/*.json.gz` | Permission-restricted exact conversation snapshots bound to session, turn, and shadow-Git commit. Normal context and live detectors do not read them; only explicit rewind/resume and replay setup do. |
| `<session_id>/rewind_snapshots/rewind_pending.json` | One assistant-shell rewind waiting to restore its exact conversation on the next resume. It records identities and turns, not a second trace. |
| `<session_id>/approval_request.json` | Current tool approval request, stable action identity, matched permission rule when applicable, and status. Bash requests retain `cmd` for compatibility. |
| `<session_id>/approval_decisions.json` | Exact tool actions accepted or refused with `--always`; legacy bash command keys remain readable. |
| `<session_id>/shell_interrupt.json` | Time and reason for the latest user interrupt. Resume clears it when the new run segment starts. |
| `<session_id>/llm_hurdle_detector.jsonl` | Detector results when the selected treatment enables that file. |
| `<session_id>/adaptive_control_ledger.jsonl` | Controller actions when the selected treatment enables that file. |

Yuj may write `<target_repository>/.tool_output/*.log` when a kept tool result
is too large for the current model input. This is the main Yuj record that can
appear in the target repository.

Claude and Codex credential files are host configuration, not harness
artifacts. They remain outside the target repository and assistant session
directory. Model messages, transcripts, traces, metrics, logs, configuration
inspection, and model-command environments never receive their values.

## How to read the tables

Each use column says whether its named reader may read the file. Read the
limits in the same row before you use it.

| Term | Meaning |
| --- | --- |
| Run segment | Model work between a start or resume and the next end, pause, or interrupt. |
| Context | A context mode that builds the next model input. |
| Detector | Code that checks work through the current turn for a named hurdle. |
| Controller | Code that uses a detector result to decide whether to apply an intervention. |
| Hurdle | A named kind of problem in the model's work. |
| Intervention | A change that the controller applies while the run is active. |
| Audit | A person or program that checks what happened. |
| Scoring | A program that groups or measures completed work. |
| Branch attempt | A new run that starts from a saved point in a source run. |
| Measurement cell | One fixed combination of model, input limit, and Yuj settings. |

A `prefix` contains the rows from the start of a file and ends at a stated
turn.
A `projection` is a view that Yuj builds from another file. `Provenance` is
the saved record of where a run came from and which settings it used.
`Deterministic` means that the same trace always makes the same projection.

## When a file is ready

| Term | Meaning |
| --- | --- |
| `run-start` | The writer fixes the value before the model starts, or writes it once when the task or run segment starts. It may describe the settings or task setup. It cannot describe later behavior. |
| `live-prefix` | The file grows or changes while the run is active. A live reader may use only rows or state through the current turn. |
| `post-run` | The file is final only after the run, branch attempt, verifier, patch collector, or scorer finishes. A live reader must not use it. |

## Rules for using a file

- Use `.trace.jsonl` as the main structured record of a live run.
- Use only the trace rows through the current turn for a live decision.
- Use the raw transcript only for an audit or an explicit resume or replay.
- Treat `.solver/state.json` as a projection of `.trace.jsonl`.
- Treat `.trace.jsonl` as correct if those two files disagree.
- Treat `adaptive_debug.jsonl` and `adaptive_control_ledger.jsonl` as
  controller output.
- Do not treat controller output as raw proof that a hurdle existed.
- Use patch, verifier, verdict, and `score/*` files only after the run ends.
- Do not give those post-run files to a live context mode or detector.
- Keep benchmark task lists, launchers, eval wrappers, ledgers, run output,
  and score output in the outside repository for that benchmark.
- Use this page only to decide how a file from an outside benchmark may be
  used when it exists.
- Let a study rule limit the fields that its detector may read.
- Do not let a study rule allow a use that this page forbids.
- Update this page first if a study needs a new use.

The tables below give the use rules for both coding sessions and measurements.
Unless a row says otherwise, the file name and writer set the rules. The
folder does not change them.

## Core run files

| File | Writer | What it holds | Ready | Context | Detector | Audit | Scoring | Limits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `<task_cwd>/prompt.txt`, `<session_dir>/prompt.txt`, or `TaskSpec.prompt_text` | launcher / caller | task prompt supplied to solver | `run-start` | Yes, as the task message. | Only if a detector contract names it as fixed setup. | Yes. | No, except to group results for the same task. | Do not use it as evidence of the model's behavior or result. |
| `<task_cwd>/.trace.jsonl` or `<session_dir>/.trace.jsonl` | harness trace writer (`_loop/trace_schema.py`) | Append-only events for model turns, tools, controls, recovery, and run provenance. | `live-prefix` | Yes, but only through a projection of the current prefix. | Yes, but only through the current turn and for fields named by the detector contract. | Yes. | Yes, for post-run behavior and cost checks. | Use only rows through the current turn for a live decision. See [Trace event fields](#trace-event-fields). |
| `<run_dir>/subagents/<id>/.trace.jsonl` or `<session_dir>/subagents/<id>/.trace.jsonl` | named-subagent runtime (`harness/subagents.py`) | Child events and exact terminal result, hash, outcome, and token counts. | `live-prefix`; terminal result is ready when the child ends. | No. Replay may read the matching terminal result. | No. | Yes. | Yes, for post-run child costs. | The parent trace stores only summary metadata and the `task` result returned to the model. Replay verifies the child record and does not call the model. |
| `<run_dir>/transcripts/<task>.log`, `<run_dir>/harness_run/transcripts/<task>.log`, `<session_dir>/transcript.log`, child `transcript.log`, or legacy `<task_cwd>/transcript.log` | model client transcript writer | Saved model requests and responses. | `live-prefix`; only audit, explicit resume, or replay may read it. | No for normal context. | No. | Yes. | No. | Resume replaces the parent transcript. Streaming records are reconstructed client evidence, not exact wire bytes. A stream-rule interrupt keeps the partial response and injection needed for replay. |
| `<run_dir>/advisor.jsonl` or `<session_dir>/advisor.jsonl` | passive advisor (`advisor.py`) | Isolated reviews, read-only tool results, quarantine decisions, and accepted note text. | `live-prefix` when enabled. | No. Only an accepted note enters the next model request. | No. | Yes. | Yes, for advisor audit; use role metrics for cost. | Private advisor output, not primary-model evidence, detector input, replay input, or state. |
| `<task_cwd>/.solver/state.json` or `<session_dir>/.solver/state.json` | harness state writer (`state_writer.py`) | Deterministic current-state view built from `.trace.jsonl`. | `live-prefix` when `state.writer_enabled` | Yes, for state-backed modes. | No by default. Use the trace unless the detector contract names this file. | Yes, as a projection. | No, except to explain a completed run. | The model must not write it. If it disagrees with the trace, trust the trace. See [State projection fields](#state-projection-fields). |
| `<task_cwd>/.solver/plan.md` | model through `write`; plan mode limits its initial creation | Nonempty implementation plan required before explicit phase exit. | `live-prefix` | Only through an ordinary model file read. Yuj does not add it to saved state. | No by default. | Yes. | No. | Yuj checks only that it exists, contains text, and has a known size. It does not claim the plan is correct. The plan does not count as an implementation change. |
| `<run_dir>/savings/<task>.jsonl`, `<assist_home>/sessions/<session_id>/savings.jsonl`, legacy `<task_cwd>/.savings.jsonl` | savings ledger (`savings.py`) | records of changes to context and tool output, with their size or cost | `live-prefix` append-only | No. | No. | Yes. | Yes, but only to count tokens or costs. | Do not use it as evidence of behavior or task success. |
| `<task_cwd>/.tool_output/*.log` | harness output sink | the full result that remains after Yuj filters tool output and moves it out of the model input | `live-prefix` | Yes, but only through sink pointers, tails, or direct reads that the context mode allows. | Only if a detector contract names output that the sink wrote through the current turn. | Yes. | No. | If the sink is off, do not take a missing sink file to mean that the tool made no output. Do not read future files for a live decision. |
| `<session_dir>/.procs/<proc_id>.log` | background process manager | raw combined stdout and stderr from one background command | `live-prefix` | No directly. Only bytes returned by a traced `bash_poll` enter context. | No. | Yes. | No. | Harness-owned audit evidence, not a process-control channel, state input, or scoring input. Lifecycle control is only through `bash_poll`, `bash_kill`, and mandatory session teardown. |
| `<task_cwd>/checkpoint.json` or `<session_dir>/checkpoint.json` | harness solver (`write_checkpoint`) | final status, model, solver, and time | `post-run` | No. | No. | Yes. | Yes, for run completion status. | It is not ready during the run. Assistant resume replaces it. It does not explain why the run behaved as it did. |
| `<task_cwd>/metrics.json` or `<session_dir>/metrics.json` | harness solver (`write_run_metrics`) | Post-run token, cache, tool-loading, checkpoint, fallback, time, guardrail, and configuration measures. | `post-run` | No. | No. | Yes. | Yes, to group runs, check cost, and check provenance. | Assistant resume replaces it. Do not use final totals as live detector evidence. See [Post-run metric groups](#post-run-metric-groups). |
| `<run_dir>/session.json` or `<run_dir>/harness_run/session.json` | measurement command / outside launcher | run, model, config, and Git provenance | `run-start` to `post-run` | No. | Only for fixed provenance that a detector contract names. | Yes. | Yes, to group runs and check provenance. | Do not use it as evidence of behavior during a turn. |
| `<run_dir>/server_meta.json` or `<run_dir>/harness_run/server_meta.json` | measurement command / server metadata probe | model server metadata snapshot | `run-start` | No. | Only for fixed provenance that a detector contract names. | Yes. | Yes, to group runs and check provenance. | Do not use it as evidence of behavior on the task. |
| `<run_dir>/run_manifest.env`, `<run_dir>/container.id` | launcher / runtime wrapper | launch environment and container identity | `run-start` | No. | Only for fixed provenance that a detector contract names. | Yes. | Yes, to group runs and check provenance. | Do not use it as evidence of behavior. |
| `<run_dir>/harness_<model>_<time>.log`, `<run_dir>/harness.stdout.log`, `<run_dir>/harness_run/*.log` | measurement command / launcher | process logs and details used to find errors | `live-prefix` as logs; most readers use them `post-run` | No. | No. | Yes. | No. | Use these files only to debug or audit a run. Do not treat them as scoring results or detector input. |
| `<run_dir>/system_log.jsonl` or `<session_dir>/system_log.jsonl` | harness system log | warnings and internal harness events | `live-prefix` append-only | No. | No. | Yes. | No. | Use it to debug or audit the harness. Do not use it as model behavior or scoring evidence. |

### Trace event fields

The raw trace owns run events. These rows keep their meaning even when model
context or `.solver/state.json` shows a shorter active view.

| Event | What it records | Boundary |
| --- | --- | --- |
| `plan_mode_enter`, `plan_mode_exit` | Planning start; exit `turn` and `plan_chars`. | Resume rebuilds the phase from these rows, not from `.solver/plan.md`. |
| `advisor_note` | Severity, size, source turn, order, and note hash. | The note text stays in `advisor.jsonl`; state does not copy this row. |
| `tool_call` for `think` | Saved thought argument. | The trace keeps the text after context and state reduce it to `think()`. |
| `stream_rule_triggered`, `stream_rule_injection` | Rule, scope, offset, path, tool, interrupt status, delivery point, and context mode. | Never stores the rule body. State copies the gate fields. |
| `todos` | Complete replacement todo list. | State copies the latest list in the active branch. |
| `exec_cell` and child `tool_call` rows | Accepted cell source, output sizes, and each injected call linked by parent ID and inner index. | State projects the child calls as ordinary tool steps. |
| `rewind` | Source and target turns plus fields for the model or operator rewind form. | A rewind adds a branch instruction; it never deletes raw rows. |
| `compaction` | Method, role, hook, hook outcome, and size metadata. | A `cancel` records an attempt, not a replacement. Summary text stays out of trace and state. |
| `handoff` | Attempt tokens, validity, fallback, and effective role. | Summary text stays out of trace and state. An unset side role records its fallback to `main`. |
| `tools_activated` | Requested, new, already-active, and final tool names. | State copies the current set under `tools`. |
| `injection` | Rule, `path` or `keyword` trigger, and normalized task path. | State does not copy the metadata; the related tool result carries visible text. |
| `schema_reject` | Tool name and value-free field errors. | The rejected `tool_call` row records the attempted action. |
| `permission` | Tool, matched rule, and `allow`, `ask`, or `deny`. | Never stores the matched argument. |
| `security_finding` | Finding ID, rule, stage, and action. | Never stores matched text, argument values, result values, or source paths. |
| `hook` | Event, command, exit status, duration, outcome, and accepted effects. | State does not copy it. A replay row also sets `replayed = true`. |
| `subagent` | Child ID, descriptor, turns, tokens, and final-text size. | The child trace owns the exact result text and hash. |
| `model_fallback` | Source and target profiles, model IDs, context windows, and reason. | Raw treatment-change provenance; state does not copy it. |
| `proc_start`, `proc_poll`, `proc_kill` | Background-process lifecycle and poll bytes returned to the model. | State does not copy these rows. |
| `turn` | Prompt tokens, cached tokens, hit ratio, and effective role. | Missing cache data stays null. |
| `length_continue` | Attempt number and completion-token count. | Stores no request or response text. |
| `checkpoint` | Shadow-Git commit and capture time, files, and bytes. | State does not copy it. |
| `tool_start`, `session_exit`, `turn_aborted` | Pending calls and recovery facts. | These rows do not claim a tool outcome. |
| `lsp_diagnostics` | File, counts, time, server, and status. | Diagnostic text stays in the edit result returned to the model. |
| `stale_guard_observe`, `stale_guard` | Read-ledger observations and policy hits. | The trace is the only resume source for the ledger. |

### Run-start provenance

`session_start` records run conditions, not proof of model behavior or task
success.

| Fields | Meaning and limits |
| --- | --- |
| `thinking_level`, `thinking_level_requested` | Effective and, when clamped, requested reasoning level. |
| `stream_rule_files` | Safe file label, rule name, and loaded-byte hash. No rule body. |
| `edit_format` | Effective edit dialect after all profile, config, legacy, and command choices. |
| `sandbox_backend`, `container_runtime`, `container_image_digest` | Selected command boundary. Runtime and digest are null for `bwrap`. |
| Lazy-loading flag, active limit, registered tools, active tools | Initial deferred-tool set. Later changes use `tools_activated`. |
| `sandbox_env_names` | Sorted command variable names. No values. |
| `repo_map_tokens`, `repo_map_sha256`, `repo_map_refresh`, `repo_map_files`, `repo_map_symbols`, `repo_map_cache_hit` | Repository-map size and cache provenance. No map body, ranking input, or cache path. |
| `worktree_path`, `worktree_branch`, `worktree_base_commit` | Retained worktree identity for resume and cleanup. The worktree itself is task state. |
| `prompt_import_tree` | Safe prompt-source labels plus import status, depth, and byte counts. No imported body or absolute host path. |
| `project_instruction_*`, `project_instructions_truncated` | Selected safe labels and source, imported, resolved, and included byte counts. No instruction body. |
| `loaded_skills` | First-wins skill name, resolved `SKILL.md` path, and `disable_model_invocation`. No description or body. |
| `ignore_file_hash`, `ignore_file_names` | Hash of loaded ignore-file bytes and configured names. No patterns or hidden paths. |
| `model_target`, `model`, `profile_name`, `base_url`, `context_size` | Effective secret-free main target. API keys are excluded. |

These fields do not enter `.solver/state.json` unless the state table below
names an explicit copy.

### State projection fields

`.solver/state.json` builds a current view from the raw trace:

| State field | Trace source |
| --- | --- |
| `state.phase` | Latest `plan_mode_enter` or `plan_mode_exit`. |
| `gates` | Stream-rule trigger and injection metadata, without rule bodies. |
| `todos` | Latest complete `todos` event in the active branch. |
| `meta.edit_format` | Latest `session_start.edit_format`. |
| `tools` | Initial tool registration plus `tools_activated` rows. |
| `last_rewind`, `rewind_report` | Latest active rewind and, for a model rewind, its retained report. |
| `last_compaction` | Trace-derived compaction fields, including hook outcome. Never summary text. |
| `meta.event_count`, `meta.projected_event_count`, `meta.active_event_count` | Raw and active-view event counts. |

Handoff text, advisor records, hook rows, background-process rows, cache-only
turn data, and most run-start provenance do not enter the state file. A rewind
selects an earlier active branch without deleting raw events. If state and
trace disagree, trust the trace.

### Post-run metric groups

| Group | What it reports |
| --- | --- |
| `metrics.tool_loading` | Registered and initial tools, initial block size and count method, activation totals, and active-tool limit. |
| `metrics.length_continuations` | Number of same-turn follow-up requests. Normal token totals already include their use. |
| `metrics.file_checkpoints` | Enabled state and per-call time, file count, and byte count. No file contents. |
| `metrics.prompt_cache` | Observed and unobserved responses and token-weighted hit ratio. The ratio stays null if any required cache count is missing. |
| `metrics.tokens_by_role` | Each complete main or side response charged once to its effective role. |
| `metrics.subagents` | Child-call count and child-owned use. Normal run totals also include child use. |
| Model fallback fields | Whether a model changed, transition count, affected roles, and active targets. Use these fields to identify treatment-changing runs. |

Resolved configuration keeps secret-free fallback chains, fixed environment
names, and run settings. It redacts API keys and every `[sandbox.env].set`
value. `system_prompt_sha256` and `system_prompt_chars` describe the resolved
prompt without storing its body.

A one-task measurement can receive its prompt through `--prompt-text` or
`--prompt-file`. The command does not copy that prompt to `prompt.txt` or save
its source in `session.json`. Save the input yourself when you need a separate
prompt artifact. The transcript records the model request that contains the
prompt when the task runs.

## Adaptive-control files

| File | Writer | What it holds | Ready | Context | Detector | Audit | Scoring | Limits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `<cell_dir>/adaptive_debug.jsonl` | adaptive-control debug writer | a summary of how the controller and detector used the live prefix to reach a decision | `live-prefix` append-only when debug is enabled | No. | No. It is controller output, not input. | Yes. | Yes, but only to check how well the controller worked. | It is not raw turn evidence. Do not use this file as the correct answer when you train or check a live detector. |
| `<cell_dir>/llm_hurdle_detector.jsonl` | LLM hurdle detector | detector input summary, result, and effective model role for each model-backed call | `live-prefix` append-only when enabled | No. | No as input to the same detector call. | Yes. | Yes, but only to check detector behavior after the run. | It is detector output. Do not treat it as raw proof that a hurdle existed. |
| `<cell_dir>/adaptive_control_ledger.jsonl` | adaptive-control ledger writer | when the controller did or did not act, and how it applied each intervention | `live-prefix` append-only when enabled | No. | No. It is controller output, not input. | Yes. | Yes, but only to record when and how the controller applied interventions. | Do not use it as live detector input. Do not use it as proof that a hurdle existed. |
| `<branch_bundle_root>/<branch_point_id>/prefix.trace.jsonl` | branch bundle capture | source-run `.trace.jsonl` prefix through the branch slot | `live-prefix` snapshot for the source run; `run-start` for branch attempts | Yes, but only to rebuild a replay or branch. | No for the source run. A branch replay may read prefix facts from it. | Yes. | Yes, to check where the branch setup came from. | It must not contain source-run events after the branch slot. |
| `<branch_bundle_root>/<branch_point_id>/context_messages.json` | branch bundle capture | context messages visible when Yuj captures the branch | `run-start` for branch attempts | Yes, to rebuild the branch. | No. | Yes. | No. | Use it only to set up replay. After capture, do not use it as live evidence about the source run. |
| `<branch_bundle_root>/<branch_point_id>/solver_state.json` | branch bundle capture | copy of `.solver/state.json` when Yuj captures the branch | `run-start` for branch attempts | Yes, to rebuild the branch. | No. | Yes. | No. | It is only a projection snapshot. The source trace wins if they disagree. |
| `<branch_bundle_root>/<branch_point_id>/guard_state.json`, `runtime_state.json`, `adaptive_state.json` | branch bundle capture | harness and adaptive runtime state when Yuj captures the branch | `run-start` for branch attempts | Yes, to rebuild the branch. | No. | Yes. | No. | Do not treat it as separate evidence of task progress. |
| `<branch_bundle_root>/<branch_point_id>/repo_snapshot/**` | branch bundle capture | copy of the task worktree when Yuj captures the branch | `run-start` for branch attempts | Yes, to rebuild the branch. | No. | Yes. | Yes, to reproduce the branch. | Do not mix it with the source-run worktree after the branch point. |
| `<branch_attempt_dir>/branch.trace.jsonl` | branch runner | copied trace prefix plus tool events from the branch attempt | `live-prefix` during branch attempt; `post-run` for source-run studies | Yes, to build the branch state projection. | Yes only inside the branch attempt. Use only the prefix through the current turn. | Yes. | Yes, to classify what happened in the watch window. | It is not the original live-run trace. Keep the branch and source run separate. |
| `<branch_attempt_dir>/branch_attempt_manifest.json` | branch runner | branch bundle, candidate config, watch window, and digests | `post-run` | No. | No. | Yes. | Yes, to group branch attempts and check provenance. | It is not ready during the run. Do not use it as detector evidence. |
| `<branch_attempt_dir>/branch_watch_result.json` | branch runner | watch classifier result over the branch trace and window | `post-run` | No. | No. | Yes. | Yes, to check the branch result. | Do not use it as live controller or detector input. |
| `<branch_attempt_dir>/metrics.json` | branch runner | summary measures for the branch attempt | `post-run` | No. | No. | Yes. | Yes, to group branch results. | It is not ready during the run. |
| `<branch_attempt_dir>/transcript.log` | branch runner client transcript writer | data sent between the model client and the model during the branch attempt | `live-prefix` as a file; only audit or explicit replay may read it | No for normal context. Yes only for an explicit replay. | No. | Yes. | No. | Follow the same limits as the main task transcripts. |

## Patch, check, and scoring files

A program that checks or scores completed work normally writes these files
after a Yuj run.
These files can describe a completed run. A live part of Yuj must not read
them.

| File | Writer | What it holds | Ready | Context | Detector | Audit | Scoring | Limits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `<cell_dir>/<task>.patch`, `<cell_dir>/<task>.unfiltered.patch` | patch collector / launcher | final worktree diff made by the model | `post-run` | No. | No. | Yes. | Yes, as scorer input. | It is not ready during the run. Do not use it as adaptive-control evidence. |
| `<cell_dir>/git.status.txt` | launcher | snapshot of final worktree status | `post-run` | No. | No. | Yes. | Yes, to check the final worktree state. | It is not ready during the run. |
| `<cell_dir>/patch_gate.txt`, `<cell_dir>/patch_gate.stderr` | patch gate / launcher | final patch-gate result and error output | `post-run` | No. | No. | Yes. | Yes, for gate status. | It is not ready during the run. |
| `<cell_dir>/sealed_verify.log` | verifier / launcher | sealed verification command output | `post-run` | No. | No. | Yes. | Yes, for verification status. | It is not ready during the run. |
| `<cell_dir>/verdict.txt` | scorer / launcher | final verdict | `post-run` | No. | No. | Yes. | Yes. | Do not use it as live detector, context, or intervention input. |
| `<cell_dir>/score/preds.json` | scorer wrapper | prediction data made from the final patch | `post-run` | No. | No. | Yes. | Yes. | Use it only for scoring. |
| `<cell_dir>/score/score_result.json` | scorer wrapper | scorer result | `post-run` | No. | No. | Yes. | Yes. | Use it only for scoring. |
| `<cell_dir>/score/scorer.log`, `<cell_dir>/score/hook.log` | scorer wrapper | scorer and hook process logs | `post-run` | No. | No. | Yes. | Yes, to debug the scorer. | It is not ready during the run. |
| `<cell_dir>/score/*.json` per-task reports | scorer wrapper | one scoring report for each task, in a standard form | `post-run` | No. | No. | Yes. | Yes. | Use it only for scoring. |
