"""Pure contracts for the model-facing ``think`` scratchpad tool.

The raw trace remains the durable record of every thought. Context views may
forget old thought arguments, but they must do so without mutating the append
log or inventing a tool result.
"""
from __future__ import annotations


THINK_TOOL_NAME = "think"
EMPTY_THINK_RESULT = (
    '<tool_result tool_name="think" status="ok" v="1"></tool_result>'
)


def thought_is_expired(
    turn: object,
    *,
    current_turn: int,
    keep_turns: int | None,
    session_number: object | None = None,
    current_session: int | None = None,
) -> bool:
    """Return whether one thought is outside the model-facing turn window.

    Unknown turn metadata stays visible. That conservative fallback avoids
    silently deleting a current thought from a legacy message or projection.
    A thought from an earlier run segment is expired because ordinary context
    managers start a fresh turn counter for each segment.
    """
    if keep_turns is None:
        return False
    if (
        current_session is not None
        and session_number is not None
        and session_number != current_session
    ):
        return True
    if isinstance(turn, bool) or not isinstance(turn, int):
        return False
    return current_turn - turn >= keep_turns


def filter_expired_thought_messages(
    messages: list[dict], keep_turns: int | None,
) -> list[dict]:
    """Remove expired think calls and their paired results from one prompt.

    The input is an append log and is never modified. Mixed assistant turns
    retain their non-think calls in the original order. A think-only
    assistant message with no textual content is removed together with its
    empty tool result, preserving the assistant/tool causal protocol.
    """
    if keep_turns is None:
        return messages
    assistant_turns = sum(
        1 for message in messages if message.get("role") == "assistant"
    )
    assistant_turn = 0
    expired_ids: set[str] = set()
    output: list[dict] = []
    changed = False

    for message in messages:
        role = message.get("role")
        if role == "assistant":
            assistant_turn += 1
            expired = assistant_turns - assistant_turn >= keep_turns
            calls = message.get("tool_calls")
            if expired and isinstance(calls, list):
                kept_calls: list[object] = []
                for call in calls:
                    function = (
                        call.get("function", {})
                        if isinstance(call, dict)
                        else {}
                    )
                    if function.get("name") != THINK_TOOL_NAME:
                        kept_calls.append(call)
                        continue
                    changed = True
                    call_id = call.get("id") if isinstance(call, dict) else None
                    if isinstance(call_id, str) and call_id:
                        expired_ids.add(call_id)
                if len(kept_calls) != len(calls):
                    if kept_calls:
                        copied = dict(message)
                        copied["tool_calls"] = kept_calls
                        output.append(copied)
                    elif message.get("content") not in (None, ""):
                        copied = dict(message)
                        copied.pop("tool_calls", None)
                        output.append(copied)
                    continue
        if role == "tool" and message.get("tool_call_id") in expired_ids:
            changed = True
            continue
        output.append(message)

    return output if changed else messages


def redact_expired_thought_state(
    data: dict,
    *,
    current_turn: int,
    keep_turns: int | None,
    current_session: int | None,
) -> dict:
    """Redact expired thought text in an in-memory state projection copy."""
    if keep_turns is None:
        return data
    trace = data.get("trace")
    if not isinstance(trace, list):
        return data

    changed = False
    visible_trace: list[object] = []
    last_trace_entry_expired = False
    for entry in trace:
        if not isinstance(entry, dict):
            visible_trace.append(entry)
            continue
        action = str(entry.get("action") or "")
        is_think = entry.get("tool_name") == THINK_TOOL_NAME or action.startswith(
            f"{THINK_TOOL_NAME}("
        )
        if not is_think or not thought_is_expired(
            entry.get("turn"),
            current_turn=current_turn,
            keep_turns=keep_turns,
            session_number=entry.get("session"),
            current_session=current_session,
        ):
            visible_trace.append(entry)
            continue
        copied = dict(entry)
        copied["action"] = "think()"
        copied["reasoning"] = ""
        visible_trace.append(copied)
        last_trace_entry_expired = len(visible_trace) == len(trace)
        changed = True

    if not changed:
        return data
    copied_data = dict(data)
    copied_data["trace"] = visible_trace
    state = data.get("state")
    if isinstance(state, dict) and last_trace_entry_expired:
        copied_state = dict(state)
        copied_state["current_attempt"] = "think()"
        copied_data["state"] = copied_state
    return copied_data


__all__ = [
    "EMPTY_THINK_RESULT",
    "THINK_TOOL_NAME",
    "filter_expired_thought_messages",
    "redact_expired_thought_state",
    "thought_is_expired",
]
