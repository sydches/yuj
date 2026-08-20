"""Parse a verbatim transcript log and reconstruct the message list for resume.

Transcript format (written by LlamaClient._write_transcript):

    === turn 001 input ===
    {<full chat completions request payload, JSON>}
    === turn 001 output ===
    {<full chat completions response, JSON>}
    === turn 002 input ===
    ...

Each request payload contains a `messages` field with the entire chat history
at that point. To resume, we take the LAST input's messages list and append the
LAST output's assistant message — that's the conversation state at the moment
the prior session ended.

The caller then adds whatever new user message they want as the next turn.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Tuple

from ...config import Config

_HEADER_RE = re.compile(r"^=== turn (\d+) (input|output) ===\s*$", re.MULTILINE)


def _load_trace_events(trace_path: Path) -> list[dict]:
    """Load trace events from an append-only JSONL trace file."""
    trace_path = Path(trace_path)
    if not trace_path.is_file():
        return []
    events: list[dict] = []
    with open(trace_path) as trace_file:
        for line in trace_file:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _next_session_number(trace_path: Path) -> int:
    """Return the next session number for a trace-backed task."""
    numbers = [
        int(event.get("session_number", 0) or 0)
        for event in _load_trace_events(trace_path)
        if event.get("event") in {"session_start", "session_end", "tool_call"}
    ]
    return max(numbers, default=0) + 1


def build_resume_prompt_from_trace(
    trace_path: Path,
    cfg: Config,
    task_description: str = "",
) -> str | None:
    """Reconstruct a new-session prompt from persisted trace artifacts."""
    events = _load_trace_events(trace_path)
    last_end = next(
        (event for event in reversed(events) if event.get("event") == "session_end"),
        None,
    )
    if last_end is None:
        return None

    session_number = int(last_end.get("session_number", 0) or 0)
    calls = [
        (str(event.get("tool_name") or "?"), str(event.get("args_summary") or ""))
        for event in events
        if event.get("event") == "tool_call"
        and int(event.get("session_number", 0) or 0) == session_number
    ]

    parts: list[str] = []
    if task_description:
        parts.append(f"Task:\n{task_description}")
    finish_reason = str(last_end.get("finish_reason") or "?")
    turns = int(last_end.get("turns", 0) or 0)
    prompt_tokens = int(last_end.get("total_prompt_tokens", 0) or 0)
    parts.append(
        f"Previous session ended after {turns} turns: "
        f"{finish_reason}. Consumed {prompt_tokens} prompt tokens."
    )

    if finish_reason == "duplicate_abort" and calls:
        name, args = calls[-1]
        parts.append(cfg.resume_duplicate_abort.format(n=len(calls), call=f"{name}({args})"))
    elif finish_reason == "context_full":
        parts.append(cfg.resume_context_full.format(pct=int(cfg.context_fill_ratio * 100)))
    elif finish_reason == "max_turns" and calls:
        recent = calls[-cfg.resume_last_n_actions:]
        summaries = "; ".join(f"{name}({args})" for name, args in recent)
        parts.append(cfg.resume_max_turns.format(n=len(recent), actions=summaries))
    elif finish_reason == "gate_escalation":
        parts.append(cfg.resume_gate_escalation.format(n=5))
    elif finish_reason == "length":
        parts.append(cfg.resume_length)
    elif finish_reason == "done_loop":
        parts.append(cfg.resume_done_loop)
    elif finish_reason == "mutation_repeat_abort":
        parts.append(cfg.resume_mutation_repeat_abort)
    elif finish_reason == "contract_recovery_abort":
        parts.append(cfg.resume_contract_recovery_abort)
    elif finish_reason == "contract_commit_abort":
        parts.append(cfg.resume_contract_commit_abort)
    elif finish_reason == "intent_abort":
        parts.append(cfg.resume_intent_abort.format(n=1))
    elif finish_reason == "loop_detected":
        parts.append(cfg.resume_loop_detect.format(streak=5))
    elif finish_reason == "no_tool_call":
        parts.append(cfg.resume_no_tool_call)
    elif finish_reason == "error":
        parts.append(cfg.resume_error)
    elif finish_reason == "stop":
        parts.append(cfg.resume_stop)

    parts.append(cfg.resume_base)
    return "\n\n".join(parts)


def parse_resume_transcript(path: Path) -> Tuple[list[dict], dict | None]:
    """Read a verbatim transcript log; return (prior_messages, last_assistant).

    `prior_messages` is the full message list (system + user + assistant + tool
    + ...) at the moment of the last HTTP request. `last_assistant` is the
    assistant message returned in the last HTTP response (or None if the last
    response was an error / truncated).

    To resume the conversation, the caller should:
        ctx.replace_all_messages(prior_messages + [last_assistant])
        ctx.add_user(<new prompt>)
    Then proceed with Session.run() as normal.
    """
    text = Path(path).read_text()
    headers = list(_HEADER_RE.finditer(text))
    if not headers:
        raise ValueError(f"no turn markers in transcript at {path}")

    # Build per-turn body dict: {(turn_no, kind): body_text}
    bodies: dict[tuple[int, str], str] = {}
    for i, m in enumerate(headers):
        turn_no = int(m.group(1))
        kind = m.group(2)
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        bodies[(turn_no, kind)] = text[start:end].strip()

    turn_numbers = sorted({k[0] for k in bodies.keys()})
    last_turn = turn_numbers[-1]

    last_input_body = bodies.get((last_turn, "input"))
    if not last_input_body:
        raise ValueError(f"transcript ends mid-turn (no input for turn {last_turn})")

    try:
        last_payload = json.loads(last_input_body)
    except json.JSONDecodeError as e:
        raise ValueError(f"last input payload is not valid JSON: {e}")
    prior_messages = last_payload.get("messages") or []
    if not prior_messages:
        raise ValueError("last input payload has empty/missing messages list")

    last_output_body = bodies.get((last_turn, "output"))
    last_assistant: dict | None = None
    if last_output_body:
        try:
            last_response = json.loads(last_output_body)
            choices = last_response.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                if msg:
                    last_assistant = {k: v for k, v in msg.items() if v is not None}
        except (json.JSONDecodeError, KeyError, TypeError):
            # Last output was an error string ("KeyError: ...") or otherwise
            # malformed — fall through with last_assistant=None. The caller
            # can decide whether that's acceptable (probably yes for resume,
            # since we're appending a fresh user message anyway).
            pass

    return prior_messages, last_assistant


def build_resumed_messages(
    prior_messages: list[dict],
    last_assistant: dict | None,
    new_user_message: str,
) -> list[dict]:
    """Assemble a clean message list for resume = continue the conversation.

    Returns: prior_messages + [last_assistant] + [synthesized tool_results
    for any unanswered tool_calls in last_assistant] + [new user message].

    The synthesized tool_results are required by the OpenAI API: an
    assistant message with `tool_calls` MUST be followed by a tool message
    for each tool_call_id before the next user message. The harness
    normally adds these after dispatching the tool, but a transcript
    captured at the END of a session cuts off before the harness writes
    the result for the final assistant turn (the most common case being a
    `done` tool call that ends the session). We synthesize a placeholder
    so the conversation is well-formed for resumption.
    """
    out = list(prior_messages)

    if last_assistant is not None:
        out.append(last_assistant)
        for tc in (last_assistant.get("tool_calls") or []):
            tcid = tc.get("id") or ""
            tname = (tc.get("function") or {}).get("name", "?")
            if tname == "done":
                content = "OK"
            else:
                content = (
                    f"[session resumed; original {tname}() result not "
                    "preserved — the prior session ended after this call "
                    "before its result was recorded]"
                )
            out.append({
                "role": "tool",
                "tool_call_id": tcid,
                "content": content,
            })

    out.append({"role": "user", "content": new_user_message})
    return out
