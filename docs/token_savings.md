# Transformation and savings accounting

The savings ledger has two kinds of records:

1. `transformation` records measure text that the harness actually changed.
2. `savings` records retain the older cost and counterfactual accounting.

Do not add the two kinds together. Transformation records use exact UTF-8
bytes. Older savings records use characters and may include estimates.

## Exact transformation rule

Whenever the harness changes a command, a tool result, a stored tool result,
a rendered context, or an inserted model-facing message, it writes one
`transformation` record. If the input and output are identical, it writes
nothing.

Set `loop.transform_log_mode` to one of these values:

| Mode | Saved data |
| --- | --- |
| `counts` | The transform name and location; run, task, session, turn, and tool-call ID; chain order; exact input, output, and delta bytes; character counts; change count; and SHA-256 hashes. This is the default. |
| `debug` | Everything in `counts`, plus located before/after excerpts in the JSONL record and complete before/after text files. |

`counts` does not retain the changed text. `debug` does. Debug data can
contain text that a later redaction removes, so protect it like a raw
transcript.

Yuj writes the ledger to one of these locations:

- `<run_dir>/savings/<task>.jsonl`
- `<project_root>/.llm_assist/sessions/<session_id>/savings.jsonl`
- legacy: `<task_cwd>/.savings.jsonl`

For a normal run, debug files are beside the task ledger:

```text
<run_dir>/savings/<task>.transform_debug/<event-id>.before.txt
<run_dir>/savings/<task>.transform_debug/<event-id>.after.txt
```

Assistant-mode debug files use `savings.transform_debug/` beside
`savings.jsonl`.

Logging is best effort. If Yuj cannot open or write the ledger, it warns,
disables accounting for that task, and continues the run. Yuj records a
debug-sidecar failure as `debug_write_error` when the JSONL write still
succeeds.

## Record fields

A transformation record contains:

| Field | Meaning |
| --- | --- |
| `event_id` | Unique event ID for this ledger-open cycle. |
| `run`, `task`, `session`, `turn`, `tool_call_id` | Where the change happened. Empty IDs mean that no tool call owned the change. |
| `surface` | `execution_command`, `tool_output`, `tool_output_fragment`, `stored_tool_output`, `context_render`, or `injected_message`. |
| `chain_id`, `chain_step` | Order of successive changes to the same text. The previous output hash must equal the next input hash within a multi-step chain. |
| `bucket`, `layer`, `mechanism` | Attribution labels. |
| `input_bytes`, `output_bytes`, `delta_bytes` | Exact `len(text.encode("utf-8"))` values. |
| `input_chars`, `output_chars`, `delta_chars` | Python string lengths, retained because several harness limits use characters. |
| `change_count` | Number of matches, rows, messages, or rules changed by this event. |
| `input_sha256`, `output_sha256` | Hashes of the complete UTF-8 values. |
| `ctx` | Non-content details such as rule name, path, age band, page, or delivery mode. |

Transformation records do not contain `delta_tokens_est`. Token counts depend
on the model tokenizer and are not inferred from bytes.

### Debug evidence

Debug JSON contains byte ranges and surrounding text, not an isolated control
character pair. For example:

```json
{
  "mechanism": "collapse_blank_lines",
  "input_bytes": 31,
  "output_bytes": 30,
  "delta_bytes": -1,
  "change_count": 1,
  "changes": [
    {
      "input_byte_range": [13, 16],
      "output_byte_range": [13, 15],
      "before": "previous line\n\n\nnext line",
      "after": "previous line\n\nnext line"
    }
  ]
}
```

The complete values live in the event's `input_full_path` and
`output_full_path`. Their byte counts and SHA-256 hashes must match the JSONL
record.

## Instrumented transformations

The following table is the implementation inventory. Dynamic rule names appear
after the mechanism prefix in the ledger.

| Area | Bucket and mechanisms | Exact before and after values | Main source |
| --- | --- | --- | --- |
| Command redirects | `bash_command_transform / redirect:<rule>`; `command_intervention / redirect_refusal:<rule>` | Original command to no execution; empty result to the refusal result | `harness/tools.py` |
| Forbidden commands | `bash_command_transform / forbidden:<rule>` | Original command to the safe refusal command that actually ran | `bash_quirks/_forbidden.py`, `harness/tools.py` |
| Quiet and test flags | `bash_command_transform / universal:<rule>`, `test_flag:<flag>` | Each command before and after that one rule | `bash_quirks/transforms.py`, `harness/tools.py` |
| Shell cleanup | `tool_output_filter`: ANSI stripping, blank-line collapse, duplicate-line collapse, similar-line collapse, traceback folding | Tool output before and after each cleanup | `harness/_tool_filters.py` |
| Deterministic output normalization | `tool_output_normalize`: `ls` timestamps, runner timing, current-working-directory paths, object addresses | Tool output before and after each normalization | `harness/_tool_filters.py` |
| Test-output condensation | `bash_output_condense / passed_line_stripping` | Complete test output before and after passed-line condensation | `bash_quirks/_output.py` |
| Redaction | `tool_output_redaction / <rule>` | Output before and after each redaction rule | `bash_quirks/_redactions.py` |
| Output limits | `truncate_output / head_tail_truncation`; `sink_surface / head_tail_with_pointer` | Complete admitted output before and after clipping or sinking | `harness/_tool_filters.py`, `harness/_loop/state_projection.py` |
| Structured test projection | `structured_projection / <runner>_digest` | Raw test output to digest plus raw-output pointer | `harness/_loop/state_projection.py` |
| Read projection | `read_projection`: range selection and line numbering; `tool_result_reminder`: empty, past-EOF, and truncated reminders | File text, selected text, numbered text, and reminder result in order | `harness/_tools/read.py` |
| Search projection | `search_normalize`, `search_filter`, `search_pagination`, `tool_quirks_glob_refusal` | Raw matches or paths through sorting, ignore filtering, pagination, or refusal | `harness/_tools/grep.py`, `harness/_tools/_common.py`, `tool_quirks/transforms.py` |
| Definition outline | `outline_vs_read / list_definitions` | Complete source text to the returned definition outline | `harness/_tools/list_definitions.py` |
| Tool-result envelopes | `tool_result_envelope`: security-marker insertion and unified envelope | Output before and after envelope work | `harness/tools.py` |
| Repeated output | `output_dedup / <tool>`; `dedup / <tier>` | Repeated output to its back-reference or stored-context stub | `harness/_loop/_dispatch_tool_call.py`, `context_strategies/solver_state_context.py` |
| Half-life | `context_projection / halflife_decay` | Each older tool result before and after its age-band cap | `context_strategies/halflife_context.py` |
| Thought retention | `context_projection / think_retention_window` | Complete message-list JSON before and after retention removes expired `think` pairs | `harness/context.py` |
| Other context modes | `context_projection / <mode>` | Complete message-list JSON before and after the mode's projection | `context_strategies/` |
| Digest or checkpoint compaction | `context_compaction / <method>` | Complete message-list JSON before and after compaction, including its overflow guard | `harness/_loop/compaction.py` |
| Pre-flight re-clip | `preflight_reclip / oversized_message_head_tail` | The oversized message before and after the re-clip | `harness/_loop/compaction.py` |
| Adaptive intervention | `adaptive_intervention / adaptive_user_turn`, `adaptive_tool_result_note` | Empty text to the inserted user message, or the tool result before and after the detector note | `harness/_loop/run_step.py`, `harness/_loop/_dispatch_tool_call.py` |
| Gate and policy rejection | `guardrail_intervention`: done guard, schema, inactive-tool, permission, plan-mode, mutation, contract, rumination, and pre-tool-hook blocks | Empty text to the rejection shown as the tool result | `harness/_loop/_dispatch_tool_call.py` |
| Guardrail warning | `guardrail_intervention`: error, security, contract, test-read, rumination, duplicate-call, and gate-grace warnings | Empty text to the warning delivered in the next synthetic user turn | `harness/_loop/_dispatch_tool_call.py` |
| Security intervention | `security_intervention / argument_block`, `result_block` | Empty text to an argument refusal, or raw result to a result refusal | `harness/tools.py` |
| Stale-file intervention | `stale_file_intervention / stale_precheck_error`, `stale_edit_block`, `stale_edit_warning` | Empty text to a refusal, or edit result before and after a warning | `harness/tools.py` |
| Hooks and injections | `hook_intervention`, `injection / <name>` | Empty text to the next synthetic user turn; terminal hook context remains attached to its terminal result | `harness/_loop/_dispatch_tool_call.py`, `harness/loop.py`, `harness/injections.py` |
| Stream rules | `stream_rule_intervention / next_turn_interrupt_fragment`, `retry_interrupt_fragment`, `tool_result_reminder` | Empty text to an inserted message, or tool result before and after a reminder | `harness/loop.py`, `harness/_loop/chat_io.py` |
| Diagnostics and post-edit checks | `diagnostic_intervention / lsp_diagnostics`, `post_edit_validation / <check>` | Tool result before and after diagnostics, or empty text to failed-check output | `harness/_loop/_dispatch_tool_call.py`, `harness/post_edit.py` |
| Advisor and observations | `advisor_intervention / advisor_note`, `harness_observation / open_red_observation_packet` | Empty text to the inserted message | `harness/advisor.py`, `harness/harness_observation.py` |

Whole message lists use compact UTF-8 JSON labeled
`message_list_json_utf8_v1` in `ctx.encoding`. This keeps roles, tool-call
IDs, and content together in debug evidence.

## Counting rules

- Sum `delta_bytes` for observed size changes. A negative delta removed
  bytes; a positive delta added bytes.
- Group by `surface` before interpreting totals. Command bytes,
  model-visible tool-output bytes, and repeated context-render bytes are
  different quantities.
- Sequential transforms are additive only when their chain is continuous.
  The summary command reports chain breaks.
- The ledger records half-life every time a request render shortens an older
  tool result. This measures repeated prompt reduction, not unique stored
  bytes deleted.
- A command rewrite records only the original and executed commands. It does
  not invent the output that the unmodified command might have produced.
- A redirect or forbidden-command record measures suppression/replacement and
  the refusal text. It does not claim counterfactual output savings.
- An intervention measures the text inserted or replaced. It does not assign
  later behavior to that intervention.
- A verbatim control should have no transformation records except mechanisms
  that its resolved configuration explicitly leaves enabled. The ledger makes
  those exceptions visible instead of assuming the control was verbatim.
- For an old run, exact reconstruction is possible only when retained
  artifacts contain both values or exact counts. A transcript cannot recover
  text that Yuj removed before writing the request.

## Summary command

```bash
# One task ledger.
python3 -m scripts.llm_solver.analysis.savings_summary \
    results/<run>/savings/<task>.jsonl

# Every task under one run.
python3 -m scripts.llm_solver.analysis.savings_summary results/<run>/

# Machine-readable output.
python3 -m scripts.llm_solver.analysis.savings_summary \
    --json results/<run>/
```

The report keeps exact transformation bytes separate from legacy
character-based records. It groups transformations by surface, layer, bucket,
and mechanism and reports per-run, per-task totals, so the same task in two
arms remains two rows.

## Legacy savings records

The older `savings` event remains for values that are not an observed
before/after text mutation:

- configured prompt, tool-schema, project-instruction, skill-catalog, profile,
  and pretest costs;
- counterfactual comparisons such as `apply_patch` versus an edit loop;
- estimates retained for compatibility.

These records use `input_chars`, `output_chars`, `delta_chars`,
`delta_tokens_est`, and `measure_type`. Exact and estimated legacy rows stay
separate. Do not fold them into transformation-byte totals.

## Adding a transformation

Record at the point that owns both values:

```python
from ..savings import get_ledger

get_ledger().record_transform(
    bucket="tool_output_filter",
    layer="harness",
    mechanism="new_filter",
    before=raw,
    after=transformed,
    surface="tool_output",
    change_count=matches,
    ctx={"tool_name": tool_name},
)
```

Do not write a record for a no-op. Do not put before/after content in `ctx`;
normal `counts` mode must remain content-free. Add a new inventory row when
you introduce a bucket or model-visible surface.
