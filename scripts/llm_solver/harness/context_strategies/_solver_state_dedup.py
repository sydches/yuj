"""Two-tier dedup logic for SolverStateContext.add_tool_result.

Pulled into a free function so the class method becomes a thin
record-and-mutate sequence.
"""
from __future__ import annotations

import json
from collections import deque

from ._solver_state_helpers import _dedup_message


_DEDUP_EXEMPT = frozenset({"read"})


def apply_dedup(
    content: str,
    *,
    tool_name: str,
    cmd_signature: str,
    recent_tool_results: deque,
    dedup_counts: dict[int, int],
    dedup_epoch: int,
    turn_count: int,
) -> tuple[str, bool, str]:
    """Run tier-1 (cmd-signature) and tier-2 (byte-identical) dedup.

    Mutates ``dedup_counts`` in place when a tier fires. Returns
    (possibly-rewritten content, dedup_fired, dedup_tier).

    Tier 1 — command-signature dedup (bash only): catches pipe
      variations like `cat file` vs `cat file | head -100` that
      produce different output but read the same data.
    Tier 2 — content dedup (all tools): catches byte-identical
      output from any tool.
    Both tiers share the same dedup_counts escalation state and
    reset on successful write/edit (epoch bump).

    Exempt: `read` tool (compound context shows stubs — the model
    genuinely needs to re-read files before editing).
    """
    # Tier 1: command-signature dedup for bash
    if cmd_signature and tool_name not in _DEDUP_EXEMPT and len(content) > 200:
        for i, existing in enumerate(recent_tool_results):
            if existing.get("_epoch") != dedup_epoch:
                continue
            if existing.get("_cmd_sig") == cmd_signature:
                turn_ref = turn_count - (len(recent_tool_results) - i)
                sig_key = hash(("cmd", cmd_signature))
                dedup_counts[sig_key] = dedup_counts.get(sig_key, 0) + 1
                count = dedup_counts[sig_key]
                try:
                    cmd = json.loads(cmd_signature).get("cmd", cmd_signature)
                except (ValueError, TypeError):
                    cmd = cmd_signature
                rewritten = _dedup_message(cmd, existing["content"], count, turn_ref)
                return rewritten, True, "tier1_cmd_signature"

    # Tier 2: byte-identical content dedup (all non-exempt tools)
    if tool_name not in _DEDUP_EXEMPT:
        for i, existing in enumerate(recent_tool_results):
            if existing.get("_epoch") != dedup_epoch:
                continue
            if existing["content"] == content and len(content) > 200:
                turn_ref = turn_count - (len(recent_tool_results) - i)
                content_key = hash(content)
                dedup_counts[content_key] = dedup_counts.get(content_key, 0) + 1
                count = dedup_counts[content_key]
                if count >= 2:
                    rewritten = (
                        f"ERROR: BLOCKED — identical output {count + 1} times (see turn {turn_ref}).\n"
                        f"REASON: Re-running will not produce new information.\n"
                        f"ACTION REQUIRED: You MUST change your approach — "
                        f"edit code, read a different file, or try a different command."
                    )
                else:
                    rewritten = (
                        f"WARNING: Same output as turn {turn_ref}.\n"
                        f"Re-running will not help — change your approach."
                    )
                return rewritten, True, "tier2_byte_identical"

    return content, False, ""
