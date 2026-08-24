---
layout: default
title: Compaction hooks
nav_order: 5.5
---

# Compaction hooks

Yuj can call one trusted Python function when the normal context threshold and
mutation gate request compaction. The function can keep the built-in behavior,
cancel that attempt, or supply a replacement checkpoint summary.

This is a narrow Python extension point, not a Yuj quirk or a shell hook. It
runs in the harness process with the harness user's permissions. Review the
module before you enable it.

## Configure a hook

Put an importable synchronous function in a Python module:

```python
# my_compaction.py
from scripts.llm_solver.harness.compaction_hooks import (
    Cancel,
    Compaction,
    CompactionPreparation,
)


def compact(preparation: CompactionPreparation):
    if preparation.file_ops.mutation_count == 0:
        return Cancel()
    return None  # use context.compaction_method for this attempt
```

Select it in a small settings overlay:

```toml
[context]
compaction_hook = "my_compaction:compact"
```

The module must be on the harness Python import path. Yuj imports the module,
finds the named attribute, and checks that it is a synchronous callable while
it loads configuration. A malformed reference, failed module import, missing
attribute, non-callable value, or async function stops startup before model
work begins. An empty string keeps the built-in path and imports nothing.

Apply the overlay through the normal settings surface:

```bash
yuj code --config my-hook.toml "Fix the issue and run its tests."
```

## Read the preparation

The function receives one frozen `CompactionPreparation` dataclass. Its
message dictionaries are detached copies: changing them does not change the
live conversation.

| Field | Type | Meaning |
| --- | --- | --- |
| `messages_to_summarize` | `tuple[dict, ...]` | Raw canonical messages after the fixed system/task prefix and before the suggested retained tail. Synthetic summaries from earlier compactions are not substituted for this raw process-local archive. |
| `kept_tail` | `tuple[dict, ...]` | Raw messages beginning at the suggested assistant-turn boundary. The boundary never separates an assistant tool call from its contiguous tool results. |
| `previous_summary` | `str` | The previous validated checkpoint or hook summary in this run segment, or an empty string. It is advisory input, not raw evidence. |
| `file_ops` | `CompactionFileOps` | Mechanical facts derived from the current raw trace prefix: `read_files`, `modified_files`, `last_test_runner_digest`, and `mutation_count`. |
| `tokens_before` | `int` | Pre-flight count for the visible prompt that crossed the compaction budget. |
| `first_kept_turn` | `int` | Suggested zero-based assistant-turn boundary in the raw conversation archive. |
| `knobs` | read-only mapping | Effective `compaction_method` plus the configured checkpoint and digest compaction knobs. |

The `knobs` mapping contains these keys:

- `compaction_method`
- `checkpoint_keep_recent_tokens`
- `checkpoint_max_summary_tokens`
- `digest_compaction_safety_margin`
- `digest_keep_recent_turns`
- `digest_compaction_gate_min_mutations`

The hook runs only after the existing derived token threshold and mutation
gate pass. It is not called on every turn.

## Return one of three results

Return `None` to use the configured built-in method for this attempt. Digest
remains deterministic. Checkpoint still makes its normal validated no-tool
request through the `weak` model role.

Return `Cancel()` to leave the current messages unchanged. Yuj records the
canceled attempt, then continues through the ordinary prompt and server
limits. Cancellation does not claim that the oversized prompt is safe.

Return `Compaction(summary, first_kept_turn)` to replace the built-in result.
The boundary may differ from the suggestion, but it must name an available
assistant turn and cannot move behind a previously accepted boundary.

The summary passes the same mechanical validator as a built-in checkpoint. It
must:

- contain the seven required Markdown sections in their required order;
- keep `Long-term goal` to one non-empty line;
- use different `Mid-term goal` and `Near-term goal` text;
- label `Done`, `In progress`, and `Blocked` under `Progress`;
- mention every mechanically observed modified path;
- preserve a well-formed assistant/tool tail; and
- fit the compaction budget and reduce the prompt token count.

The required section names are `Long-term goal`, `Mid-term goal`, `Near-term
goal`, `Constraints`, `Progress`, `Key decisions`, and `Critical context`.
Yuj appends the trace-derived file/test appendix after validation. It does not
trust a hook to author those mechanical facts.

A hook exception, unsupported return value, invalid boundary, or failed
summary validation uses the deterministic digest. Hook code cannot disable
the post-compaction overflow guard.

## Read the saved result

Each completed compaction, including a hook-canceled attempt, writes one raw
`compaction` trace row. New rows include:

| Field | Values |
| --- | --- |
| `hook` | Normalized `module:function`, or the empty string when no hook is configured. |
| `hook_outcome` | `not_configured`, `default`, `cancel`, `replace`, or `fallback_digest`. |

`method` is `hook` for cancel, replacement, and hook-failure fallback rows.
`fallback` is `digest` only when the hook result failed and digest was used.
For a direct hook result, `role` is null because Yuj did not invoke a named
model role on the hook's behalf.

`.solver/state.json` mechanically copies `hook` and `hook_outcome` into
`state.last_compaction` when the source trace row contains them. A cancel row
therefore describes a canceled attempt, not a context replacement. Summary
text is never written to the trace or state projection; it remains only in the
live model conversation. If the trace and state disagree, the trace wins.
