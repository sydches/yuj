"""Render exact repeated text against full results retained in the same view."""
from __future__ import annotations

import json
from collections import Counter

from ._solver_state_helpers import _dedup_message


_DEDUP_EXEMPT = frozenset({"read"})


def apply_dedup(content, *, tool_name, cmd_signature, anchors, dedup_epoch):
    """Compare supplied text only; never infer execution, outcomes or progress."""
    if tool_name in _DEDUP_EXEMPT:
        return content, "", ""
    matches = [row for row in anchors
               if row.get("_epoch") == dedup_epoch and row.get("content") == content]
    if not matches:
        return content, "", ""
    command_matches = [row for row in matches
                       if cmd_signature and row.get("_cmd_sig") == cmd_signature]
    source = (command_matches or matches)[0]
    reference = source["tool_call_id"]
    rewritten = _dedup_message(reference)
    # Both values are measured in the rolling policy's character units.
    if len(rewritten) >= len(content):
        return content, "", ""
    return rewritten, ("tier1_cmd_signature" if command_matches
                       else "tier2_byte_identical"), reference


def render_recent_results(recent_tool_results, char_budget, dedup_epoch):
    """Select newest first, keeping every reference's full source in this view.

    Raw entries are never replaced. Only already selected, uncompressed results
    can become anchors; reference chains and references to evicted rows cannot
    arise. The existing newest-result exemption remains a separate policy.
    """
    ids = Counter(row.get("tool_call_id") for row in recent_tool_results
                  if isinstance(row.get("tool_call_id"), str))
    anchors, parts, references = [], [], []
    chars_used = 0
    for row in reversed(recent_tool_results):
        content = row.get("content") or ""
        call_id = row.get("tool_call_id")
        identifiable = isinstance(call_id, str) and bool(call_id) and ids[call_id] == 1
        epoch_matches = row.get("_epoch") == dedup_epoch
        rendered, tier, reference = content, "", ""
        if identifiable and epoch_matches:
            rendered, tier, reference = apply_dedup(
                content, tool_name=row.get("_tool_name", ""),
                cmd_signature=row.get("_cmd_sig", ""), anchors=anchors,
                dedup_epoch=dedup_epoch,
            )
        label = ("Tool call " + json.dumps(call_id, ensure_ascii=True)
                 if isinstance(call_id, str) and call_id else "Tool result (call ID unavailable)")
        part = label + "\n" + rendered
        cost = len(part) + (len("\n---\n") if parts else 0)
        if parts and chars_used + cost > char_budget:
            break
        parts.append(part)
        chars_used += cost
        if identifiable and epoch_matches and not tier:
            anchors.append(row)
        if tier:
            references.append({
                "tool_call_id": call_id, "reference_tool_call_id": reference,
                "mechanism": tier, "tool_name": row.get("_tool_name", ""),
                "reference_scope": "same_render_full_result",
                "comparison": "supplied_text_equality", "dedup_epoch": dedup_epoch,
                "input_chars": len(content), "output_chars": len(rendered),
            })
    while len(recent_tool_results) > len(parts):
        recent_tool_results.popleft()
    if not parts:
        return "", references
    label = (f"=== Tool results (last {len(parts)}, newest last) ==="
             if len(parts) > 1 else "=== Tool result from your last action ===")
    return label + "\n" + "\n---\n".join(reversed(parts)), references
