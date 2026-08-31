---
layout: default
title: Compaction hooks
nav_order: 5.5
---

# Compaction hooks

Yuj can call a trusted Python function when a run needs context compaction.
The function may use the built-in method, cancel that attempt, or return a
replacement checkpoint summary.

The function runs inside the Yuj process with your account's permissions. It
is not a quirk or a shell hook. Review the whole module before you enable it.

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

Put the module on Yuj's Python import path. For an assistant coding session,
Yuj first inspects repository behavior and obtains workspace trust. It then
imports the module and checks the named function during ordinary configuration
loading. A bad reference, failed import, missing name, non-callable value, or
async function stops startup before model work begins. An empty setting
imports nothing and keeps the built-in path.

Apply the overlay through the normal command:

```bash
yuj --config my-hook.toml "Fix the issue and run its tests."
```

## Read the preparation

The function receives one frozen `CompactionPreparation` value. Its message
dictionaries are copies, so changing them does not change the live
conversation.

| Field | Type | Meaning |
| --- | --- | --- |
| `messages_to_summarize` | `tuple[dict, ...]` | Raw messages between the fixed system/task prefix and the suggested tail. Earlier synthetic summaries do not replace this process-local archive. |
| `kept_tail` | `tuple[dict, ...]` | Raw messages from the suggested assistant-turn boundary. The boundary never separates a tool call from its results. |
| `previous_summary` | `str` | Previous validated checkpoint or hook summary in this run segment, or an empty string. Treat it as advice, not raw evidence. |
| `file_ops` | `CompactionFileOps` | Trace-derived `read_files`, `modified_files`, `last_test_runner_digest`, and `mutation_count`. |
| `tokens_before` | `int` | Token count for the prompt that crossed the budget. |
| `first_kept_turn` | `int` | Suggested zero-based assistant-turn boundary in the raw conversation archive. |
| `knobs` | read-only mapping | Effective `compaction_method` plus the configured checkpoint and digest compaction knobs. |

The `knobs` mapping contains these keys:

- `compaction_method`
- `checkpoint_keep_recent_tokens`
- `checkpoint_max_summary_tokens`
- `digest_compaction_safety_margin`
- `digest_keep_recent_turns`
- `digest_compaction_gate_min_mutations`

Yuj calls the hook only after the normal token threshold and mutation gate
pass. It does not call it on every turn.

## Return one of three results

| Return value | What Yuj does |
| --- | --- |
| `None` | Use the configured built-in method for this attempt. |
| `Cancel()` | Keep the current messages and continue under the normal prompt and server limits. This does not claim that an oversized prompt is safe. |
| `Compaction(summary, first_kept_turn)` | Use the replacement after it passes the normal boundary, content, and size checks. |

A replacement boundary must name an available assistant turn. It cannot move
behind a boundary that Yuj already accepted.

The summary passes the same fixed checks as a built-in checkpoint. It
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
Yuj appends the trace-derived file and test facts after validation. The hook
does not write those facts.

A hook exception, unsupported value, bad boundary, or invalid summary falls
back to the deterministic digest. The hook cannot disable the overflow check.

## Read the saved result

Each completed compaction, including a hook-canceled attempt, writes one raw
`compaction` trace row. New rows include:

| Field | Values |
| --- | --- |
| `hook` | Normalized `module:function`, or the empty string when you have not configured a hook. |
| `hook_outcome` | `not_configured`, `default`, `cancel`, `replace`, or `fallback_digest`. |

`method` is `hook` for cancel, replacement, and hook-failure fallback rows.
`fallback` is `digest` only when Yuj used digest after a hook failure.
For a direct hook result, `role` is null because Yuj did not invoke a named
model role on the hook's behalf.

When a trace row has these fields, `.solver/state.json` copies them into
`state.last_compaction`. A `cancel` row describes an attempt, not a context
replacement. Summary text stays only in the live conversation. Yuj never
writes it to the trace or state. If the two records disagree, trust the trace.
