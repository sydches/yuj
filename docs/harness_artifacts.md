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

The normal `assist_home` is `<yuj-installation>/.llm_assist`. Set
`HARNESS_ASSIST_HOME` to use another directory.

| File | What it means |
| --- | --- |
| `sessions.sqlite3` | Index of coding sessions, active-session pointers, and process locks. |
| `<session_id>/prompt.txt` | Original task text. |
| `<session_id>/session.json` | Model, original target repository, context mode, starting config paths, and retained worktree path/branch/base commit when enabled. A later `provider.toml` for that coding session is not added to this file in the current code. |
| `<session_id>/provider.toml` | Model-service settings given on the `code`, `run`, or `smoke` command. Present only when that command changes the service. |
| `<session_id>/.trace.jsonl` | Append-only event record across run segments. |
| `<session_id>/.solver/state.json` | Current state view when the state writer is on. |
| `<session_id>/transcript.log` | Model messages for the newest run segment. Resume replaces this file. |
| `<session_id>/savings.jsonl` | Append-only record of context and output changes. |
| `<session_id>/system_log.jsonl` | Append-only record of harness warnings and internal events. |
| `<session_id>/checkpoint.json` | End status for the newest run segment. Resume replaces this file. |
| `<session_id>/metrics.json` | Measures for the newest run segment. Resume replaces this file. |
| `<session_id>/.shadow_git/` or the task telemetry sibling's `.shadow_git/` | Independent Git object store for enabled file checkpoints. It is outside the model's task view. |
| `<session_id>/approval_request.json` | Current tool approval request, stable action identity, matched permission rule when applicable, and status. Bash requests retain `cmd` for compatibility. |
| `<session_id>/approval_decisions.json` | Exact tool actions accepted or refused with `--always`; legacy bash command keys remain readable. |
| `<session_id>/shell_interrupt.json` | Time and reason for the latest user interrupt. Resume clears it when the new run segment starts. |
| `<session_id>/llm_hurdle_detector.jsonl` | Detector results when the selected treatment enables that file. |
| `<session_id>/adaptive_control_ledger.jsonl` | Controller actions when the selected treatment enables that file. |

Yuj may write `<target_repository>/.tool_output/*.log` when a kept tool result
is too large for the current model input. This is the main Yuj record that can
appear in the target repository.

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
| `<task_cwd>/.trace.jsonl` or `<session_dir>/.trace.jsonl` | harness trace writer (`_loop/trace_schema.py`) | model tool intent and reasoning, tool calls, tool-result summaries, background-process lifecycle, per-response cache telemetry, compaction, handoff, file-checkpoint, LSP-diagnostic, stale-guard, interrupted-turn, and length-continuation metadata, model-fallback metadata, harness events, flags, and timings | `live-prefix` | Yes, through a projection that uses only the current prefix or a context mode that uses current events. | Yes, but only through the current turn and only for fields named by the detector contract. | Yes. | Yes, for post-run behavior and cost checks. | `proc_start`, `proc_poll`, and `proc_kill` are raw lifecycle evidence; a poll stores the exact admitted model-visible bytes and hash, and none of these rows enters model state. One `turn` row per logical model response records aggregate `prompt_tokens`, `cached_tokens`, `cache_hit_ratio`, and effective role; unavailable cache counts stay null. A `length_continue` row records one bounded same-turn follow-up's attempt and completion tokens, carries no response/request text, and is not projected into model state. A `checkpoint` row records the external shadow-Git commit plus capture duration, file count, and byte count; it is not projected into model state. `tool_start`, `session_exit`, and `turn_aborted` are durable harness diagnostics: they preserve pending-call and recovery facts, not a claimed tool outcome, and none is projected into model state. An `lsp_diagnostics` row records file, counts, elapsed milliseconds, server, and status; diagnostic text appears only in the admitted edit result and is not projected into model state. `stale_guard_observe` rows are the only resume source for the read ledger; `stale_guard` rows record policy hits. Neither enters `.solver/state.json`. Model-backed `compaction` and `handoff` rows also record the effective role. An unset consumer role adds `requested_role` and `role_fallback=main`. Each `session_start` names its effective secret-free model target. A `model_fallback` row records the from/to target, reason code, profiles, model IDs, and context windows. Summary text and endpoint keys are never stored in these rows. Do not read future rows for a live decision. |
| `<run_dir>/transcripts/<task>.log`, `<run_dir>/harness_run/transcripts/<task>.log`, `<session_dir>/transcript.log`, or legacy `<task_cwd>/transcript.log` | model client transcript writer | saved model request and response records | `live-prefix` as a file; only audit, explicit resume, or replay may read it | No for normal context. Yes only for an explicit resume or replay. | No for a live detector. | Yes. | No. | An assistant resume replaces its transcript. With `YUJ_STREAMING=1`, the client saves the request before it adds stream fields and rebuilds the saved response from stream events. Do not call those records exact wire bytes. Do not use the file as live detector input. |
| `<task_cwd>/.solver/state.json` or `<session_dir>/.solver/state.json` | harness state writer (`state_writer.py`) | deterministic projection of `.trace.jsonl`, including the latest mechanical compaction metadata | `live-prefix` when `state.writer_enabled` | Yes for state-backed modes. | No by default. A detector should use facts from the trace prefix unless its contract names this file. | Yes, as a projection. | No, except to explain a completed run. | `last_compaction` contains no model summary text. Handoff rows and transport-only `turn` cache fields do not enter the projection. If state disagrees with `.trace.jsonl`, the trace wins. The model must not write this file. |
| `<run_dir>/savings/<task>.jsonl`, `<assist_home>/sessions/<session_id>/savings.jsonl`, legacy `<task_cwd>/.savings.jsonl` | savings ledger (`savings.py`) | records of changes to context and tool output, with their size or cost | `live-prefix` append-only | No. | No. | Yes. | Yes, but only to count tokens or costs. | Do not use it as evidence of behavior or task success. |
| `<task_cwd>/.tool_output/*.log` | harness output sink | the full result that remains after Yuj filters tool output and moves it out of the model input | `live-prefix` | Yes, but only through sink pointers, tails, or direct reads that the context mode allows. | Only if a detector contract names output that the sink wrote through the current turn. | Yes. | No. | If the sink is off, do not take a missing sink file to mean that the tool made no output. Do not read future files for a live decision. |
| `<session_dir>/.procs/<proc_id>.log` | background process manager | raw combined stdout and stderr from one background command | `live-prefix` | No directly. Only bytes returned by a traced `bash_poll` enter context. | No. | Yes. | No. | Harness-owned audit evidence, not a process-control channel, state input, or scoring input. Lifecycle control is only through `bash_poll`, `bash_kill`, and mandatory session teardown. |
| `<task_cwd>/checkpoint.json` or `<session_dir>/checkpoint.json` | harness solver (`write_checkpoint`) | final status, model, solver, and time | `post-run` | No. | No. | Yes. | Yes, for run completion status. | It is not ready during the run. Assistant resume replaces it. It does not explain why the run behaved as it did. |
| `<task_cwd>/metrics.json` or `<session_dir>/metrics.json` | harness solver (`write_run_metrics`) | token totals, token-weighted `prompt_cache` metrics, `tokens_by_role`, length-continuation and file-checkpoint costs, model-fallback study filters, wall time, guardrail counters, provenance, and resolved config | `post-run` | No. | No. | Yes. | Yes, to group runs, check cost, and check provenance. | `metrics.length_continuations` counts same-turn follow-up requests; the ordinary prompt/completion totals include their usage. `metrics.file_checkpoints` reports enabled state and per-call duration/file/byte counts; it does not contain file contents. `metrics.prompt_cache` separates observed and unobserved logical responses; its hit ratio stays null when any underlying response lacks cache counts. `metrics.tokens_by_role` charges each complete logical main or side response once to the effective role. `model_fallback_used`, count, roles, and active targets identify treatment-changing runs. Provenance keeps the secret-free configured chains, revert policy, and initial target; raw trace transitions recover later effective targets. It is not ready during the run. Assistant resume replaces it. Do not use final totals or counters as detector evidence. |
| `<run_dir>/session.json` or `<run_dir>/harness_run/session.json` | measurement command / outside launcher | run, model, config, and Git provenance | `run-start` to `post-run` | No. | Only for fixed provenance that a detector contract names. | Yes. | Yes, to group runs and check provenance. | Do not use it as evidence of behavior during a turn. |
| `<run_dir>/server_meta.json` or `<run_dir>/harness_run/server_meta.json` | measurement command / server metadata probe | model server metadata snapshot | `run-start` | No. | Only for fixed provenance that a detector contract names. | Yes. | Yes, to group runs and check provenance. | Do not use it as evidence of behavior on the task. |
| `<run_dir>/run_manifest.env`, `<run_dir>/container.id` | launcher / runtime wrapper | launch environment and container identity | `run-start` | No. | Only for fixed provenance that a detector contract names. | Yes. | Yes, to group runs and check provenance. | Do not use it as evidence of behavior. |
| `<run_dir>/harness_<model>_<time>.log`, `<run_dir>/harness.stdout.log`, `<run_dir>/harness_run/*.log` | measurement command / launcher | process logs and details used to find errors | `live-prefix` as logs; most readers use them `post-run` | No. | No. | Yes. | No. | Use these files only to debug or audit a run. Do not treat them as scoring results or detector input. |
| `<run_dir>/system_log.jsonl` or `<session_dir>/system_log.jsonl` | harness system log | warnings and internal harness events | `live-prefix` append-only | No. | No. | Yes. | No. | Use it to debug or audit the harness. Do not use it as model behavior or scoring evidence. |

Every `session_start` trace row records `thinking_level`, plus
`thinking_level_requested` when profile capabilities forced a clamp. The
matching `metrics.json` provenance records `thinking_level_requested`,
`thinking_level_effective`, and `thinking_level_clamped`. These are run
conditions, not model-side state or evidence of task success.

The same `session_start` row records `sandbox_backend`, `container_runtime`,
and `container_image_digest`. Runtime and digest are null for the bwrap
backend. These fields are run-start provenance and are not projected into
`.solver/state.json`.

Every `session_start` also records `sandbox_env_names`, the sorted names in the
immutable environment passed to command children for that run. Values are
not emitted in this provenance field or projected into `.solver/state.json`.
As with any model-visible value, a command can still explicitly print an
allowed variable into its ordinary traced tool-result evidence.
`metrics.json` resolved configuration preserves fixed variable names but
redacts every `[sandbox.env].set` value.

When runtime worktree isolation is enabled, each `session_start` also records
`worktree_path`, `worktree_branch`, and `worktree_base_commit`. The assistant
session store and `session.json` retain the same identity for strict resume
and operator cleanup. The working tree lives at
`<repository>/.yuj_worktrees/<run-id>` and is intentionally preserved on
exit; it is task state, not a harness artifact or model-side projection.

Every `session_start` row records `prompt_import_tree`, an ordered set of safe
source envelopes for the arm file, selected project instructions, and enabled
injection files. Its nested records contain only import request/status/depth and
byte metadata; they contain no imported body or absolute host path. An
injection envelope records resolution, not a claim that the fragment fired.
This raw provenance is not projected into `.solver/state.json`.

When project instruction discovery is enabled, the same row records
`project_instruction_files` (ordered safe labels plus source bytes, scope, and
truncation), `project_instruction_bytes`,
`project_instruction_imported_bytes`,
`project_instruction_resolved_bytes`, and
`project_instructions_truncated`. The resolved-byte cap is applied after import
expansion. Instruction bodies and absolute source paths do not enter the trace
or `.solver/state.json`. The matching `metrics.json` provenance
`system_prompt_sha256` and `system_prompt_chars` describe the exact resolved
prompt after the arm file and project blocks are assembled; the prompt body is
not stored in provenance.

A `schema_reject` row is raw validation metadata: it records the tool and
value-free field errors before any handler runs. It is not projected into
`.solver/state.json`; the associated gate-blocked `tool_call` row remains the
mechanical attempted-action record and is counted by the normal error ladder.

A `permission` row is raw control metadata emitted after schema validation and
before approval, bash quirks, or a handler. It records only the tool, matched
rule, and effective `allow`, `ask`, or `deny` decision—never the matched
argument. It is not projected into `.solver/state.json`. A denied call retains
a gate-blocked `tool_call` row and its model-visible error, so the ordinary
error ladder and replay history still see the attempted action.

Every `session_start` records `ignore_file_hash` and `ignore_file_names` for
the immutable repository model-view policy loaded at run start. The hash is
SHA-256 over the exact bytes for one loaded file, or a framed aggregate for
multiple files; it is `null` when no file was loaded or the feature is off.
Patterns and ignored path names are never copied into the trace or the
mechanical state projection.

The row also records `model_target`, `model`, `profile_name`,
`base_url`, and `context_size` for the effective main target. API keys are
excluded. `model_fallback` events and the post-run fallback metrics are raw
telemetry/provenance only; `.solver/state.json` does not project them.

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
